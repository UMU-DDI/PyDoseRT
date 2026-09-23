"""
Pencil-beam dose engine with a radiological depth per pencil.

``DoseEngine`` picks each depth plane's kernel by the radiological depth of the
central axis, so every voxel of a plane gets the same kernel. That is exact for a
flat surface at normal incidence and wrong elsewhere: a voxel just under oblique or
curved skin gets the central axis's post-build-up kernel, and a pencil behind bone
is attenuated like the central axis. ``PencilDepthEngine`` gives every pencil (beam's-
eye-view column) its own depth, from ``PencilDepthLayer``, and convolves each with the
kernel interpolated to that depth.

Everything else -- fluence, calibration, rotation, chunking, the machine's electron
contamination -- is ``DoseEngine``'s; only the step from fluence to beam-frame dose
is replaced, so the contamination follows each pencil's depth here.
"""
import torch
from torch import nn

from pydosert.engine.dose_engine import DoseEngine
from pydosert.layers.PencilDepthLayer import PencilDepthLayer


class PencilDepthEngine(DoseEngine):
    """``DoseEngine`` with the kernel picked per pencil rather than per depth plane.

    The kernel differs from pencil to pencil, so one convolution per plane no longer
    does. The kernel is instead tabulated at fixed depth nodes; each pencil's fluence
    is split between the two nodes around its own depth with linear weights, and each
    node's share is convolved with that node's kernel (``BeamWiseConvolutionalLayer.
    depth_binned``). That equals interpolating every pencil's kernel to its own depth.
    It is FFT-based, so the convolution backend is always ``"fft"``.

    Args:
        *args, **kwargs: As ``DoseEngine``. ``conv_backend`` may only be ``"fft"``.
        depth_nodes_mm: Increasing radiological depths (density x mm) the kernel is
            tabulated at. They set the cost: each plane is transformed once per node
            its pencils fall between. The default keeps the interpolation within 0.2%
            of the exact kernel. On 5 + 5 patients, half of them ran 10-25% faster
            within 0.2 gamma points (1%/1mm); a quarter ran ~30% faster but shifted
            the dose level by ~0.7%; an eighth broke down.

    Example:
        >>> engine = PencilDepthEngine(machine_config=config, kernel_size=51,
        ...                            dose_grid_spacing=spacing, dose_grid_shape=shape,
        ...                            beam_template=sequence)
        >>> dose = engine.compute_dose(sequence, density_image=density)
    """

    #: Fluence, relative to each beam's maximum, above which a pencil gets its own
    #: depth. Below it lies the transmission and head-scatter floor, which keeps the
    #: central-axis kernel.
    PENCIL_FLUENCE_FRACTION = 0.05

    #: Placed greedily until linearly interpolating the kernel between neighbours is
    #: within 0.2% of the exact kernel at every depth -- in integral and pointwise --
    #: for both 10 MV presets. They crowd into the first 2 cm because the build-up
    #: kernel is convex there: 1 mm spacing alone errs by 2.6% at 0.5 mm.
    DEPTH_NODES_MM = (
        0.0, 0.195, 0.391, 0.586, 0.781, 0.977, 1.172, 1.562, 1.953, 2.344, 3.125,
        3.906, 4.688, 5.469, 6.25, 7.031, 7.812, 8.594, 9.375, 10.938, 12.5, 14.062,
        15.625, 17.188, 18.75, 21.875, 25.0, 31.25, 37.5, 50.0, 75.0, 100.0, 125.0,
        150.0, 175.0, 200.0, 225.0, 250.0, 275.0, 300.0, 325.0, 350.0, 375.0, 400.0,
    )

    def __init__(self, *args, depth_nodes_mm: tuple[float, ...] = DEPTH_NODES_MM,
                 **kwargs) -> None:
        if kwargs.get("conv_backend", "fft") != "fft":
            raise ValueError("PencilDepthEngine convolves by FFT; conv_backend must be 'fft'")
        kwargs["conv_backend"] = "fft"
        # set before DoseEngine.__init__, which may calibrate and so run a forward pass
        self.depth_nodes_mm = tuple(float(d) for d in depth_nodes_mm)
        self._node_kernel_cache = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def _in_field_box(in_field: torch.Tensor, kernel_hw) -> tuple[int, int, int, int] | None:
        """Lateral BEV box holding every in-field pencil, grown by the kernel half-width.

        Args:
            in_field: [B*G, D, H, W, 1] boolean mask of pencils given their own depth.
            kernel_hw: (kH, kW) of the kernel, whose half-widths are the halo.

        Returns:
            (h0, h1, w0, w1), or None when no pencil is in field.
        """
        _, _, H, W, _ = in_field.shape
        rows = in_field.any(dim=4).any(dim=3).any(dim=1).any(dim=0)      # [H]
        cols = in_field.any(dim=4).any(dim=2).any(dim=1).any(dim=0)      # [W]
        if not bool(rows.any()):
            return None
        hr, wr = (int(kernel_hw[0]) - 1) // 2, (int(kernel_hw[1]) - 1) // 2
        r = rows.nonzero().flatten()
        c = cols.nonzero().flatten()
        return (max(0, int(r[0]) - hr), min(H, int(r[-1]) + 1 + hr),
                max(0, int(c[0]) - wr), min(W, int(c[-1]) + 1 + wr))

    def _depth_node_kernels(self):
        """Kernels at the depth nodes, via the same call get_nested_kernels makes.

        Cached per engine: they depend only on the kernel model, which is fixed
        once the engine is built.
        """
        if self._node_kernel_cache is not None:
            return self._node_kernel_cache
        pbm = self.pencil_beam_kernel_layer.pbm
        nodes = torch.tensor(self.depth_nodes_mm, device=self.device, dtype=torch.float32)
        rs = pbm.rs.to(device=self.device, dtype=torch.float32)
        kernels = pbm.get_pencil_beam(
            d=nodes.view(1, -1, 1, 1), r=rs.view(1, 1, *rs.shape),
            depth_threshold_mm=pbm.depth_threshold_mm)[0].to(self.device)
        self._node_kernel_cache = (nodes, kernels)
        return self._node_kernel_cache

    def _bev_dose(self, fluence_maps: torch.Tensor, kernels: torch.Tensor,
                  central_depths: torch.Tensor, density_image: torch.Tensor,
                  rotation_layer: nn.Module, keep_fluence: bool = False):
        """``DoseEngine._bev_dose`` with the in-field fluence convolved per pencil depth."""
        with torch.no_grad():
            depths = PencilDepthLayer(
                self.dose_grid_shape, self.iso_center, self.dose_grid_spacing,
                rotation_layer.rot_angles_rad, device=self.device,
                dtype=self.dtype)(density_image)                     # [B*G, D, H, W]
        fluence = self.fluence_volume_layer(fluence_maps)
        kept = fluence if keep_fluence else None
        # In-field fluence gets its own per-pencil depth. The remainder -- MLC
        # transmission and head-scatter tails, a few percent of the fluence spread
        # over the whole plane -- keeps the central-axis kernel: giving it per-pencil
        # depth too made every plane reach nearly every node and cost 15x, for a
        # second-order correction.
        nodes, node_kernels = self._depth_node_kernels()
        fmax = fluence.detach().amax(dim=(1, 2, 3, 4), keepdim=True)
        in_field = fluence >= self.PENCIL_FLUENCE_FRACTION * fmax
        # Crop the per-pencil convolution to where the in-field fluence is, grown by
        # the kernel half-width. That is exact -- the kernel carries no dose past its
        # own support -- and a VMAT aperture plus halo is a fraction of the patient
        # cross-section every transform otherwise spans.
        box = self._in_field_box(in_field, node_kernels.shape[-2:])
        remainder = fluence.masked_fill(in_field, 0)
        del in_field
        if box is not None:
            h0, h1, w0, w1 = box
            primary = fluence[:, :, h0:h1, w0:w1] - remainder[:, :, h0:h1, w0:w1]
        del fluence                     # freed here unless keep_fluence holds it
        dose = self.beam_wise_conv_layer(remainder, kernels)
        del remainder
        if box is not None:
            part = self.beam_wise_conv_layer.depth_binned(
                primary, depths[:, :, h0:h1, w0:w1], nodes, node_kernels)
            del primary
            dose[:, :, h0:h1, w0:w1] += part.to(dose.dtype)
            del part
        if self.machine_config.electron_contamination is not None:
            self._add_electron_contamination(dose, fluence_maps, depths)
        return dose, kept
