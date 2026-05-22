"""
Pencil-beam photon dose engine.

DoseEngine implements the radiotherapy dose calculation pipeline (preprocessing,
fluence modeling, kernel generation, beam-wise convolution, and geometric
rotation of dose volumes) on top of the shared scaffolding in
:class:`pydosert.engine.photon_base_engine.PhotonBaseEngine`. It supports batched
inputs and multiple beams.

To build a new photon engine variant, subclass ``PhotonBaseEngine`` and implement
the same hooks this class provides (``_initialize_layers``, ``_full_geometry``,
``_build_chunk_geometry`` and ``_forward_core``).
"""
import torch
from torch import nn

from pydosert.engine.photon_base_engine import PhotonBaseEngine
from pydosert.layers.FluenceMapLayer import FluenceMapLayer
from pydosert.layers.FluenceVolumeLayer import FluenceVolumeLayer
from pydosert.layers.RadiologicalDepthLayer import RadiologicalDepthLayer
from pydosert.layers.PencilBeamKernelLayer import PencilBeamKernelLayer
from pydosert.layers.BeamWiseConvolutionalLayer import BeamWiseConvolutionalLayer
from pydosert.layers.BeamRotationLayer import BeamRotationLayer
from pydosert.data import Beam, BeamSequence
from pydosert.geometry.rotations import rotate_2d_images


class DoseEngine(PhotonBaseEngine):
    """
    Pencil-beam convolution dose engine with beam-wise rotation.

    Usage:
        engine = DoseEngine(machine_config, kernel_size, dose_grid_spacing, dose_grid_shape)
        dose = engine.compute_dose(beam_sequence, density_image)

    See :class:`PhotonBaseEngine` for the construction and ``compute_dose``
    interface; this subclass provides the pencil-beam pipeline implementation.
    """

    def _initialize_layers(self, new_beam_data: BeamSequence | Beam, overwrite: bool = False) -> None:
        """Build or refresh the pencil-beam pipeline layers from a beam template.

        Sets the geometry attributes the base class relies on (number_of_beams,
        gantry_angles, collimator_angles, field_size, SID, iso_center) and rebuilds
        only the layers whose defining inputs changed. No-op when new_beam_data is None.

        Args:
            new_beam_data (BeamSequence | Beam | None): Beam template defining the
                treatment geometry. A single Beam is treated as one beam (G=1).
            overwrite (bool): Reserved flag for forcing re-initialization; the rebuild
                decision is currently driven by changes to the beam geometry.
        """
        if new_beam_data is None:
            return

        initialize_fluence_map_layer = not hasattr(self, 'fluence_map_layer')
        initialize_fluence_volume_layer = not hasattr(self, 'fluence_volume_layer')
        initialize_beam_wise_conv_layer = not hasattr(self, 'beam_wise_conv_layer')
        initialize_pencil_beam_kernel_layer = not hasattr(self, 'pencil_beam_kernel_layer')
        initialize_rad_depth_layer = not hasattr(self, 'rad_depth_layer')
        initialize_rotation_layer = not hasattr(self, 'rotation_layer')

        if isinstance(new_beam_data, Beam):
            number_of_beams = 1
            gantry_angles = torch.tensor([new_beam_data.gantry_angle]).to(self.dtype).to(self.device)
            collimator_angles = torch.tensor([new_beam_data.collimator_angle]).to(self.dtype).to(self.device)
        elif isinstance(new_beam_data, BeamSequence):
            number_of_beams = len(new_beam_data)
            gantry_angles = new_beam_data.gantry_angles
            collimator_angles = new_beam_data.collimator_angles.to(self.dtype).to(self.device)

        if self.dtype is None:
            self.dtype = new_beam_data.dtype
        if self.device is None:
            self.device = new_beam_data.device

        if  self.number_of_beams is None or (self.number_of_beams != number_of_beams):
            initialize_rad_depth_layer = True
            initialize_rotation_layer = True
        elif self.gantry_angles is None or (self.gantry_angles != gantry_angles).any():
            initialize_rad_depth_layer = True
            initialize_rotation_layer = True
        elif self.collimator_angles is None or (self.collimator_angles != collimator_angles).any():
            initialize_rad_depth_layer = True
            initialize_rotation_layer = True
        self.number_of_beams = number_of_beams
        self.gantry_angles = gantry_angles
        self.collimator_angles = collimator_angles


        if self.field_size is None or (self.field_size != new_beam_data.field_size):
            initialize_fluence_map_layer = True
            initialize_fluence_volume_layer = True
        self.field_size = new_beam_data.field_size


        self.SID = new_beam_data.sid
        if self.iso_center is None or (self.iso_center != new_beam_data.iso_center):
            initialize_fluence_volume_layer = True
            initialize_rad_depth_layer = True
            initialize_rotation_layer = True
        self.iso_center = new_beam_data.iso_center

        if self.dtype is None:
            return
        if self.device is None:
            return
        if self.dose_grid_shape is None:
            return
        if self.dose_grid_spacing is None:
            return
        if self.number_of_beams is None:
            return

        if initialize_fluence_map_layer:
            self.fluence_map_layer = FluenceMapLayer(
                self.machine_config,
                device = self.device,
                dtype=self.dtype,
                field_size=self.field_size,
                verbose=self.verbose
            )

        if initialize_fluence_volume_layer:
            self.fluence_volume_layer = FluenceVolumeLayer(
                self.machine_config,
                device = self.device,
                dtype=self.dtype,
                resolution=self.dose_grid_spacing,
                ct_array_shape=self.dose_grid_shape,
                sid=self.SID,
                iso_center=self.iso_center,
                field_size=self.field_size,
                verbose=self.verbose
            )

        if initialize_rad_depth_layer:
            self.rad_depth_layer = RadiologicalDepthLayer(
                self.machine_config,
                device = self.device,
                dtype=self.dtype,
                resolution=self.dose_grid_spacing,
                ct_array_shape=self.dose_grid_shape,
                gantry_angles=self.gantry_angles,
                iso_center=self.iso_center,
                verbose=self.verbose
            )

        if initialize_pencil_beam_kernel_layer:
            self.pencil_beam_kernel_layer = PencilBeamKernelLayer(
                self.machine_config,
                device = self.device,
                dtype=self.dtype,
                resolution=self.dose_grid_spacing,
                kernel_size=self.kernel_size,
                verbose=self.verbose
            )

        if initialize_beam_wise_conv_layer:
            self.beam_wise_conv_layer = BeamWiseConvolutionalLayer(
                self.device,
                self.dtype,
                verbose=self.verbose
            )

        if initialize_rotation_layer:
            self.rotation_layer = BeamRotationLayer(
                self.machine_config,
                device=self.device,
                dtype=self.dtype,
                ct_array_shape=self.dose_grid_shape,
                gantry_angles=self.gantry_angles,
                iso_center=self.iso_center,
                resolution=self.dose_grid_spacing,
                verbose=self.verbose
            )

        self.layers_initialized = True

    def _full_geometry(self) -> tuple[nn.Module, nn.Module]:
        """Geometry context for the full beam set: the radiological-depth and rotation layers."""
        return (self.rad_depth_layer, self.rotation_layer)

    def _build_chunk_geometry(self, chunk_size: int) -> list[tuple[int, int, tuple[nn.Module, nn.Module]]]:
        """
        Build the geometry-dependent layers (radiological depth and rotation) for each
        beam chunk. These carry no learnable parameters and are the expensive part to
        recompute, so they are cached and reused across calls.

        Args:
            chunk_size (int): Number of beams per chunk.

        Returns:
            list[tuple[int, int, tuple[nn.Module, nn.Module]]]: One
                (start, end, (rad_depth_layer, rotation_layer)) entry per chunk,
                where [start, end) indexes into the full set of beams.
        """
        chunks = []
        for start in range(0, self.number_of_beams, chunk_size):
            end = min(start + chunk_size, self.number_of_beams)
            gantry_angles = self.gantry_angles[start:end]
            rad_depth_layer = RadiologicalDepthLayer(
                self.machine_config,
                device=self.device,
                dtype=self.dtype,
                resolution=self.dose_grid_spacing,
                ct_array_shape=self.dose_grid_shape,
                gantry_angles=gantry_angles,
                iso_center=self.iso_center,
                verbose=self.verbose,
            )
            rotation_layer = BeamRotationLayer(
                self.machine_config,
                device=self.device,
                dtype=self.dtype,
                ct_array_shape=self.dose_grid_shape,
                gantry_angles=gantry_angles,
                iso_center=self.iso_center,
                resolution=self.dose_grid_spacing,
                verbose=self.verbose,
            )
            chunks.append((start, end, (rad_depth_layer, rotation_layer)))
        return chunks

    def _forward_core(
        self,
        leaf_positions: torch.Tensor | None,
        mus: torch.Tensor | None,
        jaw_positions: torch.Tensor | None,
        density_image: torch.Tensor,
        geometry: tuple[nn.Module, nn.Module],
        collimator_angles: torch.Tensor,
        number_of_beams: int,
        return_intermediates: bool = False,
        fluence_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run the pencil-beam pipeline for a (possibly partial) set of beams.

        The geometry-dependent layers (radiological depth and rotation) and the
        beam count are passed in so the same pipeline can serve both the full beam
        set and individual chunks. All other layers are beam-count-agnostic and read
        from self.

        Args:
            leaf_positions (torch.Tensor | None): Leaf positions [B, G, N, 2].
                Ignored when fluence_maps is provided.
            mus (torch.Tensor | None): Monitor units [B, G]. If None, fluence is used unscaled.
            jaw_positions (torch.Tensor | None): Jaw positions [B, G, 2], or None.
            density_image (torch.Tensor): CT/density volume [B, D, H, W].
            geometry (tuple[nn.Module, nn.Module]): The (radiological-depth, rotation)
                layers for these beams.
            collimator_angles (torch.Tensor): Collimator angles [G].
            number_of_beams (int): Number of beams G in this call.
            return_intermediates (bool): If True, also return intermediate tensors.
            fluence_maps (torch.Tensor | None): Pre-computed fluence maps [B, G, H, W]
                or [B*G, H, W]; skips the FluenceMapLayer when given.

        Returns:
            torch.Tensor: Dose tensor [B, D, H, W] summed over the given beams. If
                return_intermediates is True, returns a tuple (radiological_depths,
                fluence_maps, fluence_volumes, dose).
        """
        rad_depth_layer, rotation_layer = geometry
        with torch.amp.autocast(self.device.type, dtype=self.dtype):
            if density_image.dim() == 3:
                density_image = density_image.unsqueeze(0)
            with torch.no_grad():
                batched_radiological_depths = rad_depth_layer(density_image).detach()
                batched_kernels = self.pencil_beam_kernel_layer(batched_radiological_depths).detach()

            if not(return_intermediates):
                del batched_radiological_depths
            H, D, W = self.dose_grid_shape

            G = number_of_beams

            if fluence_maps is not None:
                # Use provided fluence maps directly, skipping the FluenceMapLayer
                if fluence_maps.dim() == 4:
                    # [B, G, H, W] -> [B*G, H, W]
                    B = fluence_maps.shape[0]
                    batched_fluence_maps = fluence_maps.reshape(B * G, fluence_maps.shape[2], fluence_maps.shape[3])
                else:
                    # Already [B*G, H, W]
                    B = fluence_maps.shape[0] // G
                    batched_fluence_maps = fluence_maps
            else:
                batched_fluence_maps = self.fluence_map_layer(leaf_positions, jaw_positions)
                B = leaf_positions.shape[0]

            # Apply collimator rotation (beam limiting device angle)
            # This rotates the fluence map in-plane before projection to 3D
            if (collimator_angles != 0.0).any():
                batched_fluence_maps = rotate_2d_images(
                    batched_fluence_maps,
                    collimator_angles,
                    device=self.device,
                    dtype=self.dtype
                )  # [B*G, H, W]

            batched_fluence_volumes = self.fluence_volume_layer(
                batched_fluence_maps
            )
            batched_accumulated_dose = self.beam_wise_conv_layer(
                batched_fluence_volumes, batched_kernels
            )
            batched_accumulated_dose.mul_(self.machine_config.mean_photon_energy_MeV)

            if not(return_intermediates):
                del batched_fluence_volumes, batched_fluence_maps, batched_kernels

            D_, H_, W_, _ = batched_accumulated_dose.shape[1:]
            batched_accumulated_dose = batched_accumulated_dose.view(B, G, D_, H_, W_)
            if mus is not None:
                batched_accumulated_dose.mul_(mus[:, :, None, None, None])

            batched_accumulated_dose = rotation_layer(batched_accumulated_dose)

            batched_accumulated_dose = batched_accumulated_dose.sum(dim=1).to(self.dtype)

        if return_intermediates:
            return batched_radiological_depths, batched_fluence_maps, batched_fluence_volumes, batched_accumulated_dose
        else:
            return batched_accumulated_dose