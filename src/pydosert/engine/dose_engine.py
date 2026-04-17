"""
Baseline PyDoseRT dose engine.

Implements radiotherapy dose calculation using 2D pencil-beam convolution in
beam's-eye-view (BEV) coordinates. A single radiological-depth profile is
extracted along the central axis of each beam and each BEV depth slice is
convolved with a per-depth pencil-beam kernel. The approach is fast but only
accounts for density variations along the central ray — it cannot capture
lateral inhomogeneities.

See :class:`pydosert.engine.heterogeneity_dose_engine.HeterogeneityDoseEngine`
for a full 3D density-correction counterpart that shares this engine's base
class.
"""
import torch

from pydosert.engine.base_dose_engine import BaseDoseEngine
from pydosert.layers.BeamValidationLayer import BeamValidationLayer
from pydosert.layers.FluenceMapLayer import FluenceMapLayer
from pydosert.layers.FluenceVolumeLayer import FluenceVolumeLayer
from pydosert.layers.RadiologicalDepthLayer import RadiologicalDepthLayer
from pydosert.layers.PencilBeamKernelLayer import PencilBeamKernelLayer
from pydosert.layers.BeamWiseConvolutionalLayer import BeamWiseConvolutionalLayer
from pydosert.layers.BeamRotationLayer import BeamRotationLayer
from pydosert.data import BeamSequence, Beam
from pydosert.geometry.rotations import rotate_2d_images


class DoseEngine(BaseDoseEngine):
    """
    Baseline pencil-beam convolution dose engine.

    Usage:
        engine = DoseEngine(machine_config, ...)
        dose = engine.forward(leaf_positions, mus, jaw_positions, density_image)

    Or with BeamSequence:
        dose = engine.compute_dose(beam_seq, density_image)
    """

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
            self.rad_depth_layer = RadiologicalDepthLayer(
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

    def forward(
        self,
        leaf_positions: torch.Tensor | None,
        mus: torch.Tensor | None,
        jaw_positions: torch.Tensor | None,
        density_image: torch.Tensor,
        return_intermediates: bool = False,
        fluence_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Runs the full dose calculation pipeline.

        Args:
            leaf_positions: Leaf positions [B, G, N, 2]. Not required when fluence_maps is provided.
            mus: Monitor units [B, G]. Optional when fluence_maps is provided; if supplied the dose
                is scaled by MUs, if omitted the fluence maps are used as-is.
            jaw_positions: Jaw positions [B, G, 2]. Not required when fluence_maps is provided.
            density_image: CT image tensor [B, D, H, W].
            return_intermediates: If True, also return intermediate tensors.
            fluence_maps: Optional pre-computed fluence maps [B, G, H, W] or [B*G, H, W].
                If provided, the FluenceMapLayer is skipped and leaf_positions/jaw_positions
                are ignored. The maps are used directly as input to the FluenceVolumeLayer.

        Returns:
            Dose tensor [B, D, H, W].
        """
        if fluence_maps is not None:
            self._set_device_dtype(fluence_maps.device, fluence_maps.dtype)
        else:
            self._set_device_dtype(leaf_positions.device, leaf_positions.dtype)

        if not self.layers_initialized:
            raise Exception("Layers haven't been initialized yet. Dose engine cannot perform dose calculations.")

        self._assert_sizes(density_image, leaf_positions, jaw_positions, mus, fluence_maps=fluence_maps)

        with torch.amp.autocast(self.device.type, dtype=self.dtype):
            if density_image.dim() == 3:
                density_image = density_image.unsqueeze(0)
            with torch.no_grad():
                batched_radiological_depths = self.rad_depth_layer(density_image).detach()
                batched_kernels = self.pencil_beam_kernel_layer(batched_radiological_depths).detach()

            if not return_intermediates:
                del batched_radiological_depths
            H, D, W = self.dose_grid_shape
            G = self.number_of_beams

            if fluence_maps is not None:
                if fluence_maps.dim() == 4:
                    B = fluence_maps.shape[0]
                    batched_fluence_maps = fluence_maps.reshape(B * G, fluence_maps.shape[2], fluence_maps.shape[3])
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

            if (self.collimator_angles != 0.0).any():
                batched_fluence_maps = rotate_2d_images(
                    batched_fluence_maps,
                    self.collimator_angles,
                    device=self.device,
                    dtype=self.dtype,
                )

            batched_fluence_volumes = self.fluence_volume_layer(batched_fluence_maps)
            batched_accumulated_dose = self.beam_wise_conv_layer(
                batched_fluence_volumes, batched_kernels
            )
            batched_accumulated_dose.mul_(self.machine_config.mean_photon_energy_MeV)

            if not return_intermediates:
                del batched_fluence_volumes, batched_fluence_maps, batched_kernels

            D_, H_, W_, _ = batched_accumulated_dose.shape[1:]
            batched_accumulated_dose = batched_accumulated_dose.view(B, G, D_, H_, W_)
            if mus is not None:
                batched_accumulated_dose.mul_(mus[:, :, None, None, None])

            batched_accumulated_dose = self.rotation_layer(batched_accumulated_dose)
            batched_accumulated_dose = batched_accumulated_dose.sum(dim=1).to(self.dtype)

        if return_intermediates:
            return batched_radiological_depths, batched_fluence_maps, batched_fluence_volumes, batched_accumulated_dose
        else:
            return batched_accumulated_dose
