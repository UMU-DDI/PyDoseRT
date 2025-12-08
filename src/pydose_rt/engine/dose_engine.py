"""
Main class for radiotherapy dose calculation using pencil beam convolution and beam-wise rotation.

This class orchestrates the pipeline for dose calculation, including preprocessing, fluence modeling,
kernel generation, convolution, and geometric rotation of dose volumes. It supports batched inputs and
multiple beams, and can optionally perform upsampling and debugging visualizations.
"""
import torch
import torch.nn.functional as F
from torch import nn

from pydose_rt.layers.BeamValidationLayer import BeamValidationLayer
from pydose_rt.layers.FluenceMapLayer import FluenceMapLayer
from pydose_rt.layers.FluenceVolumeLayer import FluenceVolumeLayer
from pydose_rt.layers.RadiologicalDepthLayer import RadiologicalDepthLayer
from pydose_rt.layers.PencilBeamKernelLayer import PencilBeamKernelLayer
from pydose_rt.layers.BeamWiseConvolutionalLayer import BeamWiseConvolutionalLayer
from pydose_rt.layers.BeamRotationLayer import BeamRotationLayer
from pydose_rt.data import MachineConfig, Beam, BeamSequence
from pydose_rt.geometry.rotations import rotate_2d_images


class DoseEngine(nn.Module):
    """
    Implements the full dose calculation pipeline for radiotherapy.

    Usage:
        engine = DoseEngine(machine_config)
        dose = engine.forward(leaf_positions, mus, jaw_positions, density_image)

    Or with BeamSequence:
        dose = engine.forward_beam_sequence(beam_seq, density_image)

    Attributes:
        machine_config (MachineConfig): Machine physics parameters.
        device (torch.device): PyTorch device for computation.
    """
    machine_config: MachineConfig | None = None
    dose_grid_shape: tuple[int, int, int] | None = None
    dose_grid_spacing: tuple[float, float, float] | None = None
    number_of_beams: int | None = None
    precomputed_radiological_depths: torch.Tensor | None = None
    precomputed_kernels: torch.Tensor | None = None

    def __init__(
        self,
        machine_config: MachineConfig,
        kernel_size: int,
        dose_grid_spacing: tuple[float, float, float],
        dose_grid_shape: tuple[int, int, int],
        beam_template: BeamSequence | Beam | None = None,
        adjust_values: bool = False, # Move to nn.Module
        device: torch.device | str | None = None,
        dtype: torch.dtype = None,
        verbose: bool = False,
    ) -> "DoseEngine":
        """
        Initializes the DoseEngine pipeline.

        Args:
            machine_config: Machine physics and MLC specifications.
            kernel_size: Size of the pencil beam dose kernel.
            dose_grid_spacing: Voxel spacing in mm (depth, height, width).
            dose_grid_shape: Shape of the output grid tensor (depth, height, width) in pixels.
            beam_input: Beam or BeamSequence defining the treatment geometry.
            adjust_values: Whether to validate and adjust parameter values (default: False).
            device: PyTorch device for computation.
            dtype: Data type for tensors.
            verbose: Enable verbose output (default: False).
        """
        super().__init__()
        self.kernel_size = kernel_size

        # Handle device default
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self._adjust_values = adjust_values
        self.layers_initialized = False

        self.machine_config = machine_config
        self.dose_grid_spacing = dose_grid_spacing
        self.dose_grid_shape = dose_grid_shape
        self._add_beam_information(beam_template)

    def _add_data_information(self, new_density_image: torch.Tensor) -> None:
        if new_density_image is None:
            return
        
        if (self.dose_grid_shape is not None):
            return
        
        if self.dtype is None:
            self.dtype = new_density_image.dtype
        if self.device is None:
            self.device = new_density_image.device
        self.dose_grid_shape = new_density_image.shape
        self.precomputed_kernels = None
        self.precomputed_radiological_depths = None
        self.layers_initialized = False
        self._initialize_layers()
        return
        
    def _add_beam_information(self, new_beam_data: BeamSequence | Beam, overwrite: bool = False) -> None:
        if new_beam_data is None:
            return
        
        if (self.number_of_beams is not None) and (not overwrite):
            # TODO: Check should be performed to ensure that things match.
            return

        if isinstance(new_beam_data, Beam):
            self.number_of_beams = 1
            self.gantry_angles = torch.tensor([new_beam_data.gantry_angle]).to(self.dtype).to(self.device)
            self.collimator_angles = torch.tensor([new_beam_data.collimator_angle]).to(self.dtype).to(self.device)
        elif isinstance(new_beam_data, BeamSequence):
            self.number_of_beams = len(new_beam_data)
            self.gantry_angles = new_beam_data.gantry_angles
            self.collimator_angles = new_beam_data.collimator_angles.to(self.dtype).to(self.device)

        if self.dtype is None:
            self.dtype = new_beam_data.dtype
        if self.device is None:
            self.device = new_beam_data.device
        self.field_size = new_beam_data.field_size
        self.SID = new_beam_data.sid
        self.iso_center = new_beam_data.iso_center
        self.precomputed_kernels = None
        self.precomputed_radiological_depths = None
        self.layers_initialized = False
        self._initialize_layers()
        return
    
    def _set_device_dtype(self, device, dtype) -> None:
        if self.dtype is None:
            self.dtype = dtype
        if self.device is None:
            self.device = device
        self._initialize_layers()

    def _initialize_layers(self) -> None:
        if self.layers_initialized:
            return
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
        

        if self._adjust_values:
            self.valid_parameters_layer = BeamValidationLayer(
                self.machine_config,
                device = self.device,
                dtype=self.dtype,
                field_size=self.field_size,
            )
        self.fluence_map_layer = FluenceMapLayer(
            self.machine_config,
            device = self.device,
            dtype=self.dtype,
            field_size=self.field_size,
            verbose=self.verbose
        )

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
        self.pencil_beam_kernel_layer = PencilBeamKernelLayer(
            self.machine_config, 
            device = self.device,
            dtype=self.dtype,
            resolution=self.dose_grid_spacing,
            kernel_size=self.kernel_size,
            verbose=self.verbose
        )
        self.beam_wise_conv_layer = BeamWiseConvolutionalLayer(
            self.device, 
            self.dtype,
            verbose=self.verbose
        )
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

    @property
    def iso_center_voxel(self) -> tuple[int, int, int]:
        if self.iso_center is None:
            return None

        sx, sy, sz = self.dose_grid_shape
        rx, ry, rz = self.dose_grid_spacing
        X, Y, Z = self.iso_center  # physical coords, origin at isocenter corner
        X_center, Y_center, Z_center = (X - rx / 2, Y - ry / 2, Z - rz / 2)

        # Convert physical coords to voxel indices and round to nearest voxel
        ix = int(X_center / rx)
        iy = int(Y_center / ry)
        iz = int(Z_center / rz)

        # Optionally clamp to valid voxel range
        ix = max(0, min(sx - 1, ix))
        iy = max(0, min(sy - 1, iy))
        iz = max(0, min(sz - 1, iz))

        return (ix, iy, iz)

    def compute_kernels(self, attenuation_map) -> tuple[torch.Tensor, torch.Tensor]:
        if attenuation_map.dim() == 3:
            attenuation_map = attenuation_map.unsqueeze(0)

        with torch.no_grad():
            batched_radiological_depths = self.rad_depth_layer(attenuation_map)
            batched_kernels = torch.tensor(
                self.pencil_beam_kernel_layer(batched_radiological_depths),
                device=self.device,
                dtype=self.dtype,
            )

        return (batched_radiological_depths.detach(), batched_kernels.detach())

    def _assert_sizes(self, density_image, leaf_positions, jaw_positions, mus):
        """Validate input tensor sizes."""

        B = leaf_positions.shape[0]
        assert leaf_positions.dim() == 4, \
            f"Leaf positions needs 4 dimensions [B, 2, CP, N], got {leaf_positions.dim()}D: {leaf_positions.shape}"
        assert jaw_positions.dim() == 3, \
            f"Jaw positions needs 3 dimensions [B, 2, CP], got {jaw_positions.dim()}D: {jaw_positions.shape}"
        assert mus.dim() == 2, \
            f"MUs needs 2 dimensions [B, CP], got {mus.dim()}D: {mus.shape}"

        assert leaf_positions.shape[0] == B and mus.shape[0] == B and jaw_positions.shape[0] == B, \
            f"Batch size mismatch: ct={B}, leaf_positions={leaf_positions.shape[0]}, jaw_positions={jaw_positions.shape[0]}, mus={mus.shape[0]}"


        expected_leaf = (B, self.number_of_beams, self.machine_config.number_of_leaf_pairs, 2)
        assert leaf_positions.shape == expected_leaf, \
            f"Leaf positions shape mismatch: expected {expected_leaf}, got {leaf_positions.shape}"

        expected_jaw = (B, self.number_of_beams, 2)
        assert jaw_positions.shape == expected_jaw, \
            f"Jaw positions shape mismatch: expected {expected_jaw}, got {jaw_positions.shape}"

        expected_mus = (B, self.number_of_beams)
        assert mus.shape == expected_mus, \
            f"MUs shape mismatch: expected {expected_mus}, got {mus.shape}"
        
        if self.precomputed_kernels is None:
            if density_image is None:
                raise ValueError("CT image must be provided.")
            assert density_image.dim() == 4, \
                f"CT image needs 4 dimensions [B, D, H, W], got {density_image.dim()}D: {density_image.shape}"
            
            expected_ct = (B, *self.dose_grid_shape)
            assert density_image.shape == expected_ct, \
                f"CT shape mismatch: expected {expected_ct}, got {density_image.shape}"
        
        
        devices = {leaf_positions.device, jaw_positions.device, mus.device}
        if density_image is not None:
            devices.add(density_image.device)

        if len(devices) != 1:
            raise ValueError(f"Device mismatch among tensors: {devices}")

        # Check that all tensors share the same dtype
        dtypes = {leaf_positions.dtype, jaw_positions.dtype, mus.dtype}
        if density_image is not None:
            dtypes.add(density_image.dtype)

        if len(dtypes) != 1:
            raise ValueError(f"Dtype mismatch among tensors: {dtypes}")
        
        
    def forward(
        self,
        leaf_positions: torch.Tensor,
        mus: torch.Tensor,
        jaw_positions: torch.Tensor,
        density_image: torch.Tensor,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        Runs the full dose calculation pipeline.

        Args:
            leaf_positions: Leaf positions [B, 2, CP, N].
            mus: Monitor units [B, CP].
            jaw_positions: Jaw positions [B, 2, CP].
            density_image: CT image tensor [B, D, H, W].
            single_cp: If set, return dose for single control point only.

        Returns:
            Dose tensor [B, H, D, W].
        """

        self._set_device_dtype(leaf_positions.device, leaf_positions.dtype)
        if not self.layers_initialized:
            raise Exception("Layers haven't been initialized yet. Dose engine cannot perform dose calculations.")

        self._assert_sizes(density_image, leaf_positions, jaw_positions, mus)

        with torch.amp.autocast(self.device.type, dtype=self.dtype):
            if self.precomputed_kernels is not None:
                batched_radiological_depths = self.precomputed_radiological_depths
                batched_kernels = self.precomputed_kernels
            else:
                batched_radiological_depths, batched_kernels = self.compute_kernels(density_image)

            if not(return_intermediates):
                del batched_radiological_depths
            H, D, W = self.dose_grid_shape

            if self._adjust_values:
                leaf_positions, jaw_positions, mus = self.valid_parameters_layer(
                    leaf_positions=leaf_positions, jaw_positions=jaw_positions, mus=mus
                )

            batched_fluence_maps = self.fluence_map_layer(leaf_positions, jaw_positions)

            # Apply collimator rotation (beam limiting device angle)
            # This rotates the fluence map in-plane before projection to 3D
            if (self.collimator_angles != 0.0).any():
                batched_fluence_maps = rotate_2d_images(
                    batched_fluence_maps,
                    self.collimator_angles,
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

            B = leaf_positions.shape[0]
            G = self.number_of_beams
            D_, H_, W_, _ = batched_accumulated_dose.shape[1:]
            batched_accumulated_dose = batched_accumulated_dose.view(B, G, D_, H_, W_)
            batched_accumulated_dose.mul_(mus[:, :, None, None, None])

            batched_accumulated_dose = self.rotation_layer(batched_accumulated_dose)

            batched_accumulated_dose = batched_accumulated_dose.sum(dim=1).to(self.dtype)

        if return_intermediates:
            return batched_radiological_depths, batched_fluence_maps, batched_fluence_volumes, batched_accumulated_dose
        else:
            return batched_accumulated_dose

    def compute_dose(
        self,
        beam_input: BeamSequence | Beam,
        density_image: torch.Tensor | None = None,
        return_intermediates: bool = False,
        overwrite: bool = False
    ) -> torch.Tensor:
        """
        Compute dose from a BeamSequence.

        Args:
            beam_sequence: BeamSequence (shapes: mus [CP], leaf_positions [CP, N, 2], jaw_positions [CP, 2])
            density_image: CT image tensor [1, D, H, W]

        Returns:
            Dose tensor [1, H, D, W]
        """
        self._add_data_information(density_image)
        self._add_beam_information(beam_input, overwrite)

        # Add batching dimension to parameters
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

        return self.forward(
            leaf_positions=leaf_positions,
            mus=mus,
            jaw_positions=jaw_positions,
            density_image=ct_tensor,
            return_intermediates=return_intermediates
        )

    def compute_dose_sequential(
        self,
        beam_sequence: BeamSequence,
        density_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute dose by processing beams sequentially (memory efficient).

        Args:
            beam_sequence: BeamSequence containing all control points
            density_image: CT image tensor [1, D, H, W]

        Returns:
            Accumulated dose tensor [1, H, D, W]
        """
        total_dose = None
        self._add_data_information(density_image)
        self._add_beam_information(beam_sequence)

        for i, beam in enumerate(beam_sequence):
            beam_dose = self.compute_dose(
                beam,
                density_image=density_image,
                overwrite=True
            )

            if total_dose is None:
                total_dose = beam_dose
            else:
                total_dose = total_dose + beam_dose

        self._add_beam_information(beam_sequence, overwrite=True)
        return total_dose

    def calibrate(self, 
                  calibration_mu: float = None,
                  original_beam_template: BeamSequence | None = None) -> None: # Keep in dose engine
        if not self.layers_initialized:
            raise Exception("Layers must be fully initialized for calibration.")

        center_x, _, center_z = torch.tensor(self.dose_grid_spacing) * (torch.tensor(self.dose_grid_shape) + 1) / 2
        iso_center = (center_x, 100.0, center_z)
        beam = Beam.create(0.0, self.machine_config.number_of_leaf_pairs, 0.0, (100.0, 100.0), iso_center=iso_center, device=self.device, dtype=self.dtype)
        if calibration_mu is None:
            calibration_mu = self.machine_config.calibration_mu
        beam.mu = calibration_mu * beam.mu
        water_attenuation = torch.ones(self.dose_grid_shape).to(self.device).to(self.dtype)

        self.precomputed_kernels = None
        self.precomputed_radiological_depths = None
        self.layers_initialized = False
        old_kernel_size = self.kernel_size

        self.kernel_size = max(self.dose_grid_shape)

        dose = self.compute_dose(
            beam,
            density_image=water_attenuation,
            overwrite=True
            )


        # Get center dose (at 10cm depth - index 50 for 100 voxels)
        center_dose = dose[0, *self.iso_center_voxel].detach().cpu().numpy().item()

        # Calculate calibration factor
        # This gives the factor to normalize to 1 Gy per MU at reference conditions
        calibration_factor = self.machine_config.mean_photon_energy_MeV / center_dose

        if (abs(center_dose - 1.0) > 0.001):
            print(f"Calibration failed. Adjusting calibration factor to: {calibration_factor}")
            self.machine_config.mean_photon_energy_MeV = calibration_factor

        if original_beam_template is not None:
            self._add_beam_information(original_beam_template, True)

        self.kernel_size = old_kernel_size
        self.layers_initialized = False
        self.precomputed_kernels = None
        self.precomputed_radiological_depths = None
        