"""
Base class for PyDoseRT dose engines.

Provides the shared scaffolding (layer bookkeeping, input validation, beam-template
handling, calibration) that concrete dose engines extend. Concrete engines only
need to (1) build the layers they use in ``_initialize_layers`` and (2) implement
the ``forward`` pipeline that actually produces the dose.

Two engines currently derive from this class:

- :class:`pydosert.engine.dose_engine.DoseEngine` — the baseline pencil-beam
  convolution engine that extracts a single radiological-depth profile along the
  central axis and builds one kernel per BEV depth slice.
- :class:`pydosert.engine.volumetric_dose_engine.VolumetricDoseEngine` — the
  finite-size pencil-beam engine with full 3D density correction. It convolves
  the fluence with a small set of kernels at fixed reference radiological depths
  and then interpolates per-voxel using a ray-traced 3D radiological-depth map.
"""
from __future__ import annotations

import torch
from torch import nn

from pydosert.data import MachineConfig, Beam, BeamSequence


class BaseDoseEngine(nn.Module):
    """Shared scaffolding for dose-calculation engines.

    Concrete subclasses are responsible for:

    * Constructing the layers they rely on inside
      :meth:`_initialize_layers`.
    * Implementing :meth:`forward` to run the pipeline.

    The base class provides common geometry state (iso-center, gantry/collimator
    angles, CT grid), input shape validation and the :meth:`compute_dose` /
    :meth:`compute_dose_sequential` helpers that are identical across engines.
    """

    machine_config: MachineConfig | None = None
    dose_grid_shape: tuple[int, int, int] | None = None
    dose_grid_spacing: tuple[float, float, float] | None = None
    number_of_beams: int | None = None
    layers_initialized: bool = False
    gantry_angles: torch.Tensor | None = None
    collimator_angles: torch.Tensor | None = None
    field_size: tuple[float, float] | None = None
    SID: float | None = None
    iso_center: tuple[float, float, float] | None = None

    def __init__(
        self,
        machine_config: MachineConfig,
        kernel_size: int,
        dose_grid_spacing: tuple[float, float, float],
        dose_grid_shape: tuple[int, int, int],
        beam_template: BeamSequence | Beam | None = None,
        adjust_values: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self._adjust_values = adjust_values
        self.machine_config = machine_config
        self.dose_grid_spacing = dose_grid_spacing
        self.dose_grid_shape = dose_grid_shape
        self._initialize_layers(beam_template)

    # ------------------------------------------------------------------
    # Helpers shared across engines
    # ------------------------------------------------------------------
    def _set_device_dtype(self, device, dtype) -> None:
        if self.dtype is None:
            self.dtype = dtype
        if self.device is None:
            self.device = device

    def _update_beam_template(self, new_beam_data: BeamSequence | Beam, overwrite: bool = False):
        """Update cached beam-template state and report which layers need (re)building.

        Returns a dict with boolean flags describing which layer categories need
        to be reinitialised. Subclasses use these flags to decide what to rebuild.

        Returns None if there is nothing to update (no beam data provided or the
        template has already been processed and ``overwrite`` is False).
        """
        if new_beam_data is None:
            return None

        if (self.number_of_beams is not None) and (not overwrite):
            return None

        flags = dict(
            beam_validation=not hasattr(self, 'beam_validation_layer'),
            fluence_map=not hasattr(self, 'fluence_map_layer'),
            fluence_volume=not hasattr(self, 'fluence_volume_layer'),
            beam_wise_conv=not hasattr(self, 'beam_wise_conv_layer'),
            pencil_beam_kernel=not hasattr(self, 'pencil_beam_kernel_layer'),
            rad_depth=not hasattr(self, 'rad_depth_layer'),
            rotation=not hasattr(self, 'rotation_layer'),
        )

        if isinstance(new_beam_data, Beam):
            number_of_beams = 1
            gantry_angles = torch.tensor([new_beam_data.gantry_angle]).to(self.dtype).to(self.device)
            collimator_angles = torch.tensor([new_beam_data.collimator_angle]).to(self.dtype).to(self.device)
        elif isinstance(new_beam_data, BeamSequence):
            number_of_beams = len(new_beam_data)
            gantry_angles = new_beam_data.gantry_angles
            collimator_angles = new_beam_data.collimator_angles.to(self.dtype).to(self.device)
        else:
            raise TypeError(f"Unsupported beam data type: {type(new_beam_data)}")

        if self.dtype is None:
            self.dtype = new_beam_data.dtype
        if self.device is None:
            self.device = new_beam_data.device

        if self.number_of_beams is None or (self.number_of_beams != number_of_beams):
            flags['rad_depth'] = True
            flags['rotation'] = True
        elif self.gantry_angles is None or (self.gantry_angles != gantry_angles).any():
            flags['rad_depth'] = True
            flags['rotation'] = True
        elif self.collimator_angles is None or (self.collimator_angles != collimator_angles).any():
            flags['rad_depth'] = True
            flags['rotation'] = True

        self.number_of_beams = number_of_beams
        self.gantry_angles = gantry_angles
        self.collimator_angles = collimator_angles

        if self.field_size is None or (self.field_size != new_beam_data.field_size):
            flags['beam_validation'] = True
            flags['fluence_map'] = True
            flags['fluence_volume'] = True
        self.field_size = new_beam_data.field_size

        self.SID = new_beam_data.sid
        if self.iso_center is None or (self.iso_center != new_beam_data.iso_center):
            flags['fluence_volume'] = True
            flags['rad_depth'] = True
            flags['rotation'] = True
        self.iso_center = new_beam_data.iso_center

        if (self.dtype is None or self.device is None
                or self.dose_grid_shape is None or self.dose_grid_spacing is None
                or self.number_of_beams is None):
            return None

        return flags

    def _initialize_layers(self, new_beam_data: BeamSequence | Beam, overwrite: bool = False) -> None:
        """Concrete engines must override this to build their layers."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @property
    def iso_center_voxel(self) -> tuple[int, int, int] | None:
        if self.iso_center is None:
            return None

        sx, sy, sz = self.dose_grid_shape
        rx, ry, rz = self.dose_grid_spacing
        X, Y, Z = self.iso_center

        ix = int(X / rx)
        iy = int(Y / ry)
        iz = int(Z / rz)

        ix = max(0, min(sx - 1, ix))
        iy = max(0, min(sy - 1, iy))
        iz = max(0, min(sz - 1, iz))

        return (ix, iy, iz)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _assert_sizes(self, density_image, leaf_positions, jaw_positions, mus, fluence_maps=None):
        """Validate input tensor sizes and uniform device/dtype."""
        G = self.number_of_beams

        if fluence_maps is not None:
            fm_h, fm_w = self.field_size
            if fluence_maps.dim() == 4:
                B = fluence_maps.shape[0]
                expected_fm = (B, G, fm_h, fm_w)
                assert fluence_maps.shape == expected_fm, \
                    f"Fluence maps shape mismatch: expected {expected_fm}, got {fluence_maps.shape}"
            elif fluence_maps.dim() == 3:
                assert fluence_maps.shape[0] % G == 0, \
                    f"Fluence maps leading dim {fluence_maps.shape[0]} is not divisible by G={G}"
                B = fluence_maps.shape[0] // G
                expected_fm = (B * G, fm_h, fm_w)
                assert fluence_maps.shape == expected_fm, \
                    f"Fluence maps shape mismatch: expected {expected_fm}, got {fluence_maps.shape}"
            else:
                raise ValueError(
                    f"fluence_maps must be 3D [B*G, H, W] or 4D [B, G, H, W], got {fluence_maps.dim()}D"
                )

            if mus is not None:
                assert mus.dim() == 2, \
                    f"MUs needs 2 dimensions [B, G], got {mus.dim()}D: {mus.shape}"
                expected_mus = (B, G)
                assert mus.shape == expected_mus, \
                    f"MUs shape mismatch: expected {expected_mus}, got {mus.shape}"

            devices = {fluence_maps.device}
            dtypes = {fluence_maps.dtype}
            if mus is not None:
                devices.add(mus.device)
                dtypes.add(mus.dtype)
        else:
            B = leaf_positions.shape[0]
            assert leaf_positions.dim() == 4, \
                f"Leaf positions needs 4 dimensions [B, 2, CP, N], got {leaf_positions.dim()}D: {leaf_positions.shape}"
            assert mus.dim() == 2, \
                f"MUs needs 2 dimensions [B, CP], got {mus.dim()}D: {mus.shape}"

            assert leaf_positions.shape[0] == B and mus.shape[0] == B, \
                f"Batch size mismatch: ct={B}, leaf_positions={leaf_positions.shape[0]}, mus={mus.shape[0]}"

            expected_leaf = (B, G, self.machine_config.number_of_leaf_pairs, 2)
            assert leaf_positions.shape == expected_leaf, \
                f"Leaf positions shape mismatch: expected {expected_leaf}, got {leaf_positions.shape}"

            expected_mus = (B, G)
            assert mus.shape == expected_mus, \
                f"MUs shape mismatch: expected {expected_mus}, got {mus.shape}"

            if jaw_positions is not None:
                assert jaw_positions.dim() == 3, \
                    f"Jaw positions needs 3 dimensions [B, 2, CP], got {jaw_positions.dim()}D: {jaw_positions.shape}"
                assert jaw_positions.shape[0] == B, \
                    f"Batch size mismatch: ct={B}, jaw_positions={jaw_positions.shape[0]}"
                expected_jaw = (B, G, 2)
                assert jaw_positions.shape == expected_jaw, \
                    f"Jaw positions shape mismatch: expected {expected_jaw}, got {jaw_positions.shape}"

            devices = {leaf_positions.device, mus.device}
            if jaw_positions is not None:
                devices.add(jaw_positions.device)
            dtypes = {leaf_positions.dtype, mus.dtype}
            if jaw_positions is not None:
                dtypes.add(jaw_positions.dtype)

        if density_image is None:
            raise ValueError("CT image must be provided.")
        assert density_image.dim() == 4, \
            f"CT image needs 4 dimensions [B, D, H, W], got {density_image.dim()}D: {density_image.shape}"

        expected_ct = (B, *self.dose_grid_shape)
        assert density_image.shape == expected_ct, \
            f"CT shape mismatch: expected {expected_ct}, got {density_image.shape}"

        devices.add(density_image.device)
        dtypes.add(density_image.dtype)

        if len(devices) != 1:
            raise ValueError(f"Device mismatch among tensors: {devices}")
        if len(dtypes) != 1:
            raise ValueError(f"Dtype mismatch among tensors: {dtypes}")

    # ------------------------------------------------------------------
    # High-level convenience entry points
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def compute_dose(
        self,
        beam_input: BeamSequence | Beam,
        density_image: torch.Tensor | None = None,
        return_intermediates: bool = False,
        overwrite: bool = False,
        fluence_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dose from a ``BeamSequence`` or ``Beam``."""
        self._initialize_layers(beam_input, overwrite)

        if density_image is not None:
            ct_tensor = density_image
            if ct_tensor.dim() == 3:
                ct_tensor = ct_tensor.unsqueeze(0)
        else:
            ct_tensor = None

        if isinstance(beam_input, Beam):
            leaf_positions = beam_input.leaf_positions.unsqueeze(0).unsqueeze(0)
            mus = beam_input.mu.unsqueeze(0).unsqueeze(0)
            jaw_positions = beam_input.jaw_positions.unsqueeze(0).unsqueeze(0)
        elif isinstance(beam_input, BeamSequence):
            leaf_positions = beam_input.leaf_positions.unsqueeze(0)
            mus = beam_input.mus.unsqueeze(0)
            jaw_positions = beam_input.jaw_positions.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported beam_input type: {type(beam_input)}")

        if fluence_maps is not None and fluence_maps.dim() == 3:
            fluence_maps = fluence_maps.unsqueeze(0)

        return self.forward(
            leaf_positions=leaf_positions,
            mus=mus,
            jaw_positions=jaw_positions,
            density_image=ct_tensor,
            return_intermediates=return_intermediates,
            fluence_maps=fluence_maps,
        )

    def compute_dose_sequential(
        self,
        beam_sequence: BeamSequence,
        density_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Accumulate dose beam-by-beam (memory efficient)."""
        self._initialize_layers(beam_sequence)
        total_dose = None

        for beam in beam_sequence:
            beam_dose = self.compute_dose(
                beam,
                density_image=density_image,
                overwrite=True,
            )
            if total_dose is None:
                total_dose = beam_dose
            else:
                total_dose = total_dose + beam_dose

        self._initialize_layers(beam_sequence, overwrite=True)
        return total_dose

    def calibrate(
        self,
        calibration_mu: float | None = None,
        original_beam_template: BeamSequence | None = None,
        verbose: bool = True,
    ) -> None:
        if not self.layers_initialized:
            raise Exception("Layers must be fully initialized for calibration.")

        center_x, _, center_z = torch.tensor(self.dose_grid_spacing) * (
            torch.tensor(self.dose_grid_shape)
        ) / 2
        iso_center = (center_x.item(), 100.0, center_z.item())
        beam = Beam.create(
            0.0,
            self.machine_config.number_of_leaf_pairs,
            0.0,
            (100.0, 100.0),
            iso_center=iso_center,
            device=self.device,
            dtype=self.dtype,
        )
        if calibration_mu is None:
            calibration_mu = self.machine_config.calibration_mu
        beam.mu = calibration_mu * beam.mu
        water_attenuation = torch.ones(self.dose_grid_shape).to(self.device).to(self.dtype)

        self.layers_initialized = False
        old_kernel_size = self.kernel_size
        self.kernel_size = max(self.dose_grid_shape)

        dose = self.compute_dose(
            beam,
            density_image=water_attenuation,
            overwrite=True,
        )

        center_dose = dose[0, *self.iso_center_voxel].detach().cpu().numpy().item()
        calibration_factor = self.machine_config.mean_photon_energy_MeV / center_dose

        if abs(center_dose - 1.0) > 0.001:
            if verbose:
                print(f"Calibration failed. Adjusting calibration factor to: {calibration_factor}")
            self.machine_config.mean_photon_energy_MeV = calibration_factor

        self.kernel_size = old_kernel_size
        if original_beam_template is not None:
            self._initialize_layers(original_beam_template, True)
