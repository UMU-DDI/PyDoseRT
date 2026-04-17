"""
HeterogeneityDoseEngine — Finite-Size Pencil-Beam engine with 3D density correction.

The baseline :class:`pydosert.engine.dose_engine.DoseEngine` approximates the
radiological-depth dependence of the pencil-beam kernel using a *single*
central-axis profile per beam. That loses accuracy whenever the beam traverses
laterally inhomogeneous regions (air cavities, bone, lung next to tissue, ...).

This engine implements the Finite-Size Pencil-Beam (FSPB) with 3D density
correction approach of Gu et al. (2011, *Phys. Med. Biol.* 56(11):3337)
[arXiv:1103.1164] — sometimes called "FSPB-3D" — adapted to PyDoseRT's
differentiable pipeline:

1. Generate pencil-beam kernels at a small handful of *fixed reference
   radiological depths* (for example ``[0, 10, 20, 40, 50, 100]`` mm). The
   kernel bank is independent of the CT and of the BEV depth slice, so it is
   built only once per gantry-angle set.
2. Convolve the BEV fluence volume with each reference kernel. Because the set
   is small (~6 kernels) and each kernel is the same across BEV depth slices,
   this is dramatically cheaper than the baseline engine's per-depth kernel
   convolution.
3. Ray-trace the CT density in BEV coordinates to build a full 3D
   radiological-depth map via
   :class:`pydosert.layers.VolumetricRadiologicalDepthLayer`.
4. Interpolate between the pre-convolved fluence volumes at each voxel using
   that voxel's own radiological depth via
   :class:`pydosert.layers.DepthInterpolatedDoseLayer`.

The trade-off is flipped compared to the baseline engine: fewer but fixed
convolutions, full 3D ray tracing per voxel. All operations are differentiable
with respect to the fluence maps (rad-depth and the reference kernels are
detached — they only depend on the fixed CT and machine physics).
"""
from __future__ import annotations

import torch

from pydosert.engine.base_dose_engine import BaseDoseEngine
from pydosert.layers.BeamValidationLayer import BeamValidationLayer
from pydosert.layers.FluenceMapLayer import FluenceMapLayer
from pydosert.layers.FluenceVolumeLayer import FluenceVolumeLayer
from pydosert.layers.VolumetricRadiologicalDepthLayer import VolumetricRadiologicalDepthLayer
from pydosert.layers.DepthInterpolatedDoseLayer import DepthInterpolatedDoseLayer
from pydosert.layers.PencilBeamKernelLayer import PencilBeamKernelLayer
from pydosert.layers.BeamWiseConvolutionalLayer import BeamWiseConvolutionalLayer
from pydosert.layers.BeamRotationLayer import BeamRotationLayer
from pydosert.data import MachineConfig, BeamSequence, Beam
from pydosert.geometry.rotations import rotate_2d_images


# Spans the surface region (0-20 mm), common treatment depths (40-100 mm) and
# deep inhomogeneities (200-500 mm) so the per-voxel interpolation always has
# bracketing reference depths for typical patient geometries.
DEFAULT_REFERENCE_DEPTHS_MM = (0.0, 10.0, 20.0, 40.0, 50.0, 100.0, 200.0, 500.0)


class HeterogeneityDoseEngine(BaseDoseEngine):
    """
    FSPB dose engine with 3D density (heterogeneity) correction via per-voxel
    depth interpolation.

    The public API mirrors :class:`pydosert.engine.dose_engine.DoseEngine`, so
    existing training / planning loops can swap engines transparently.

    Attributes:
        reference_depths_mm (tuple[float, ...]): Radiological depths in mm at
            which the reference pencil-beam kernels are evaluated.
    """

    def __init__(
        self,
        machine_config: MachineConfig,
        kernel_size: int,
        dose_grid_spacing: tuple[float, float, float],
        dose_grid_shape: tuple[int, int, int],
        beam_template: BeamSequence | Beam | None = None,
        reference_depths_mm: tuple[float, ...] | list[float] = DEFAULT_REFERENCE_DEPTHS_MM,
        adjust_values: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype = None,
        verbose: bool = False,
    ) -> None:
        self.reference_depths_mm = tuple(float(d) for d in reference_depths_mm)
        if len(self.reference_depths_mm) < 2:
            raise ValueError("Need at least 2 reference depths for interpolation.")
        super().__init__(
            machine_config=machine_config,
            kernel_size=kernel_size,
            dose_grid_spacing=dose_grid_spacing,
            dose_grid_shape=dose_grid_shape,
            beam_template=beam_template,
            adjust_values=adjust_values,
            device=device,
            dtype=dtype,
            verbose=verbose,
        )

    def _initialize_layers(self, new_beam_data: BeamSequence | Beam, overwrite: bool = False) -> None:
        flags = self._update_beam_template(new_beam_data, overwrite=overwrite)
        if flags is None:
            return

        if self._adjust_values and flags['beam_validation']:
            self.beam_validation_layer = BeamValidationLayer(
                self.machine_config,
                device=self.device,
                dtype=self.dtype,
                field_size=self.field_size,
            )

        if flags['fluence_map']:
            self.fluence_map_layer = FluenceMapLayer(
                self.machine_config,
                device=self.device,
                dtype=self.dtype,
                field_size=self.field_size,
                verbose=self.verbose,
            )

        if flags['fluence_volume']:
            self.fluence_volume_layer = FluenceVolumeLayer(
                self.machine_config,
                device=self.device,
                dtype=self.dtype,
                resolution=self.dose_grid_spacing,
                ct_array_shape=self.dose_grid_shape,
                sid=self.SID,
                iso_center=self.iso_center,
                field_size=self.field_size,
                verbose=self.verbose,
            )

        if flags['rad_depth']:
            self.rad_depth_layer = VolumetricRadiologicalDepthLayer(
                self.machine_config,
                device=self.device,
                dtype=self.dtype,
                resolution=self.dose_grid_spacing,
                ct_array_shape=self.dose_grid_shape,
                gantry_angles=self.gantry_angles,
                iso_center=self.iso_center,
                verbose=self.verbose,
            )

        if flags['pencil_beam_kernel']:
            self.pencil_beam_kernel_layer = PencilBeamKernelLayer(
                self.machine_config,
                device=self.device,
                dtype=self.dtype,
                resolution=self.dose_grid_spacing,
                kernel_size=self.kernel_size,
                verbose=self.verbose,
            )

        if flags['beam_wise_conv']:
            self.beam_wise_conv_layer = BeamWiseConvolutionalLayer(
                self.device,
                self.dtype,
                verbose=self.verbose,
            )

        # The depth-interpolation layer only depends on the reference-depths
        # list, so it does not need rebuilding when beam geometry changes.
        if not hasattr(self, 'depth_interp_layer'):
            self.depth_interp_layer = DepthInterpolatedDoseLayer(
                reference_depths=list(self.reference_depths_mm),
                device=self.device,
                dtype=self.dtype,
                verbose=self.verbose,
            )

        if flags['rotation']:
            self.rotation_layer = BeamRotationLayer(
                self.machine_config,
                device=self.device,
                dtype=self.dtype,
                ct_array_shape=self.dose_grid_shape,
                gantry_angles=self.gantry_angles,
                iso_center=self.iso_center,
                resolution=self.dose_grid_spacing,
                verbose=self.verbose,
            )

        self.layers_initialized = True

    # ------------------------------------------------------------------
    def _build_reference_kernels(self, batch_groups: int) -> torch.Tensor:
        """Return kernels at every reference depth, shape ``[kH, kW, B*G, N]``.

        The reference kernels depend only on the machine's TPR, the voxel
        resolution and the reference-depth list — none of which change between
        batches — so they could in principle be cached. We rebuild them here
        for clarity; the cost is negligible (N ~ 6 evaluations).
        """
        ref = torch.tensor(
            self.reference_depths_mm,
            device=self.device,
            dtype=self.dtype,
        )  # [N]
        N = ref.shape[0]
        # PencilBeamKernelLayer expects radiological_depth [BG, P, 1].
        rad_depth_input = ref.view(1, N, 1).expand(batch_groups, N, 1).contiguous()
        kernels = self.pencil_beam_kernel_layer(rad_depth_input)  # [kH, kW, BG, N]
        return kernels

    def _convolve_reference_kernels(
        self,
        fluence_volume: torch.Tensor,
        reference_kernels: torch.Tensor,
    ) -> torch.Tensor:
        """Convolve the BEV fluence volume with every reference kernel.

        Args:
            fluence_volume: ``[BG, D, H, W, 1]`` BEV fluence volume.
            reference_kernels: ``[kH, kW, BG, N]`` kernels at every reference
                depth.

        Returns:
            ``[BG, N, D, H, W]`` — BEV fluence convolved with each reference
            kernel, ready for depth interpolation.
        """
        BG, D, H, W, _ = fluence_volume.shape
        kH, kW, BG_k, N = reference_kernels.shape
        assert BG_k == BG, f"Kernel BG {BG_k} does not match fluence BG {BG}"

        convolved_per_depth = []
        for n in range(N):
            # The same kernel is applied across every BEV depth slice — the
            # whole point of the fixed-reference scheme — so we broadcast the
            # [kH, kW, BG, 1] slice over the D dimension.
            ker_n = reference_kernels[:, :, :, n:n + 1].expand(kH, kW, BG, D).contiguous()
            conv_n = self.beam_wise_conv_layer(fluence_volume, ker_n)  # [BG, D, H, W, 1]
            convolved_per_depth.append(conv_n.squeeze(-1))
        # Stack along a new N dimension: [BG, N, D, H, W]
        return torch.stack(convolved_per_depth, dim=1)

    # ------------------------------------------------------------------
    def forward(
        self,
        leaf_positions: torch.Tensor | None,
        mus: torch.Tensor | None,
        jaw_positions: torch.Tensor | None,
        density_image: torch.Tensor,
        return_intermediates: bool = False,
        fluence_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the FSPB-3D pipeline end-to-end.

        Mirrors :meth:`pydosert.engine.dose_engine.DoseEngine.forward` exactly
        in its I/O shapes. See that docstring for argument meanings.
        """
        if fluence_maps is not None:
            self._set_device_dtype(fluence_maps.device, fluence_maps.dtype)
        else:
            self._set_device_dtype(leaf_positions.device, leaf_positions.dtype)

        if not self.layers_initialized:
            raise Exception(
                "Layers haven't been initialized yet. Dose engine cannot perform dose calculations."
            )

        self._assert_sizes(
            density_image, leaf_positions, jaw_positions, mus,
            fluence_maps=fluence_maps,
        )

        with torch.amp.autocast(self.device.type, dtype=self.dtype):
            if density_image.dim() == 3:
                density_image = density_image.unsqueeze(0)

            G = self.number_of_beams

            # ---- Fluence map (skip when the caller provides their own) ----
            if fluence_maps is not None:
                if fluence_maps.dim() == 4:
                    B = fluence_maps.shape[0]
                    batched_fluence_maps = fluence_maps.reshape(
                        B * G, fluence_maps.shape[2], fluence_maps.shape[3]
                    )
                else:
                    B = fluence_maps.shape[0] // G
                    batched_fluence_maps = fluence_maps
            else:
                if self._adjust_values:
                    leaf_positions, jaw_positions, mus = self.beam_validation_layer(
                        leaf_positions=leaf_positions, jaw_positions=jaw_positions, mus=mus
                    )
                batched_fluence_maps = self.fluence_map_layer(leaf_positions, jaw_positions)
                B = leaf_positions.shape[0]

            # ---- Collimator rotation (in-plane) ----
            if (self.collimator_angles != 0.0).any():
                batched_fluence_maps = rotate_2d_images(
                    batched_fluence_maps,
                    self.collimator_angles,
                    device=self.device,
                    dtype=self.dtype,
                )

            # ---- Project fluence map to BEV volume ----
            batched_fluence_volumes = self.fluence_volume_layer(batched_fluence_maps)
            # [BG, D, H, W, 1]
            BG = batched_fluence_volumes.shape[0]

            # ---- Per-voxel radiological depth via ray-tracing through the CT ----
            # rad_depth has no gradient wrt the fluence; it only depends on the
            # CT density, which is a fixed geometric input.
            with torch.no_grad():
                rad_depth_bev = self.rad_depth_layer(density_image).detach()
                # Expand to match BG when B > 1; VolumetricRadiologicalDepthLayer
                # already returns [B*G, D, H, W], so nothing to do if shapes line
                # up.
                assert rad_depth_bev.shape[0] == BG, (
                    f"rad_depth BG {rad_depth_bev.shape[0]} != fluence BG {BG}"
                )

                # ---- Reference kernel bank ----
                reference_kernels = self._build_reference_kernels(BG).detach()
                # [kH, kW, BG, N]

            # ---- Convolve fluence with each reference kernel ----
            convolved_fluences = self._convolve_reference_kernels(
                batched_fluence_volumes, reference_kernels,
            )  # [BG, N, D, H, W]

            # ---- Per-voxel depth interpolation ----
            batched_accumulated_dose = self.depth_interp_layer(
                convolved_fluences, rad_depth_bev,
            )  # [BG, D, H, W, 1]
            batched_accumulated_dose = batched_accumulated_dose * self.machine_config.mean_photon_energy_MeV

            if not return_intermediates:
                del batched_fluence_volumes, batched_fluence_maps
                del convolved_fluences, reference_kernels

            D_, H_, W_, _ = batched_accumulated_dose.shape[1:]
            batched_accumulated_dose = batched_accumulated_dose.view(B, G, D_, H_, W_)
            if mus is not None:
                batched_accumulated_dose = batched_accumulated_dose * mus[:, :, None, None, None]

            batched_accumulated_dose = self.rotation_layer(batched_accumulated_dose)
            batched_accumulated_dose = batched_accumulated_dose.sum(dim=1).to(self.dtype)

        if return_intermediates:
            return (
                rad_depth_bev,
                batched_fluence_maps,
                convolved_fluences,
                batched_accumulated_dose,
            )
        return batched_accumulated_dose
