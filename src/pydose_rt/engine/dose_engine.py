#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main class for radiotherapy dose calculation using pencil beam convolution and beam-wise rotation.

This class orchestrates the pipeline for dose calculation, including preprocessing, fluence modeling,
kernel generation, convolution, and geometric rotation of dose volumes. It supports batched inputs and
multiple beams, and can optionally perform upsampling and debugging visualizations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pydose_rt.layers.ValidParametersLayer import ValidParametersLayer
from pydose_rt.layers.FluenceMapLayer import FluenceMapLayer
from pydose_rt.layers.FluenceVolumeLayer import FluenceVolumeLayer
from pydose_rt.layers.RadiologicalDepthLayer import RadiologicalDepthLayer
from pydose_rt.layers.PencilBeamKernelLayer import PencilBeamKernelLayer
from pydose_rt.layers.BeamWiseConvolutionalLayer import BeamWiseConvolutionalLayer
from pydose_rt.layers.CPRotationLayer import CPRotationLayer
from pydose_rt.data import MachineConfig, TreatmentConfig, Beam, BeamSequence
from pydose_rt.geometry.rotations import rotate_2d_images


class DoseEngine(nn.Module):
    """
    Implements the full dose calculation pipeline for radiotherapy.

    Usage:
        engine = DoseEngine(machine_config, treatment_config)
        dose = engine.forward(leaf_positions, mus, jaw_positions, ct_image)

    Or with BeamSequence:
        dose = engine.forward_beam_sequence(beam_seq, ct_image)

    Attributes:
        machine_config (MachineConfig): Machine physics parameters.
        treatment_config (TreatmentConfig): Treatment settings.
        device (torch.device): PyTorch device for computation.
    """

    def __init__(
        self,
        machine_config: MachineConfig,
        treatment_config: TreatmentConfig,
        beam_sequence: BeamSequence = None,
        leafs_centered: bool = False,
        crop_volume: bool = False,
        permute_ct: bool = False,
        adjust_values: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ) -> "DoseEngine":
        """
        Initializes the DoseEngine pipeline.

        Args:
            machine_config: Machine physics and MLC specifications.
            treatment_config: Treatment settings.
            leafs_centered: Whether leaf positions are centered.
            crop_volume: Whether to crop the dose volume.
            permute_ct: Whether to permute CT dimensions.
            adjust_values: Whether to adjust parameter values.
            verbose: Enable verbose output.
            debug: Enable debug mode.
        """
        super().__init__()
        self.machine_config = machine_config
        self.treatment_config = treatment_config
        self.verbose = verbose
        self.debug = debug
        self.crop_volume = crop_volume
        self.permute_ct = permute_ct
        self._leafs_centered = leafs_centered
        self._adjust_values = adjust_values

        self.device = treatment_config.device
        self.dtype = treatment_config.dtype

        if len(treatment_config.gantry_angles) != 0:
            self.number_of_cps = treatment_config.number_of_cps
            self.gantry_angles = torch.from_numpy(treatment_config.gantry_angles)
            self.collimator_angles = torch.tensor(
                self.treatment_config.beam_limiting_device_angle,
                dtype=self.dtype,
                device=self.device
            )
        elif beam_sequence is not None:
            self.number_of_cps = len(beam_sequence)
            self.gantry_angles = beam_sequence.gantry_angles
            self.collimator_angles = beam_sequence.beam_limiting_device_angles.to(self.dtype).to(self.device)

        self._initialize_layers()

    def _initialize_layers(self) -> None:
        """Initialize all processing layers."""
        self.resolution = tuple([
            x * y for x, y in zip(
                self.machine_config.resolution,
                self.treatment_config.downsampling_factor
            )
        ])
        self.ct_array_shape = tuple([
            int(x / y) for x, y in zip(
                self.machine_config.ct_array_shape,
                self.treatment_config.downsampling_factor
            )
        ])

        self.valid_parameters_layer = ValidParametersLayer(
            self.machine_config,
            device = self.device,
            dtype=self.dtype,
            field_size=self.treatment_config.field_size,
            leafs_centered=self._leafs_centered,
            adjust_values=self._adjust_values
        )
        self.fluence_map_layer = FluenceMapLayer(
            self.machine_config,
            device = self.device,
            dtype=self.dtype,
            resolution=self.resolution,
            field_size=self.treatment_config.field_size,
            verbose=self.verbose
        )

        self.fluence_volume_layer = FluenceVolumeLayer(
            self.machine_config, 
            device = self.device,
            dtype=self.dtype,
            resolution=self.resolution,
            ct_array_shape=self.ct_array_shape,
            sid=self.treatment_config.SID,
            iso_center=self.treatment_config.iso_center,
            field_size=self.treatment_config.field_size,
            verbose=self.verbose
        )

        self.rad_depth_layer = RadiologicalDepthLayer(
            self.machine_config, 
            device = self.device,
            dtype=self.dtype,
            resolution=self.resolution,
            ct_array_shape=self.ct_array_shape,
            gantry_angles=self.gantry_angles,
            downsampling_factor=self.treatment_config.downsampling_factor,
            lookup_table=torch.from_numpy(self.treatment_config.lookup_table),
            verbose=self.verbose
        )
        self.pencil_beam_kernel_layer = PencilBeamKernelLayer(
            self.machine_config, 
            device = self.device,
            dtype=self.dtype,
            resolution=self.resolution,
            kernel_size=self.treatment_config.kernel_size,
            verbose=self.verbose
        )
        self.beam_wise_conv_layer = BeamWiseConvolutionalLayer(
            self.device, 
            self.dtype,
            verbose=self.verbose
        )
        self.rotation_layer = CPRotationLayer(
            self.machine_config, 
            device=self.device, 
            dtype=self.dtype,
            gantry_angles=self.gantry_angles,
            verbose=self.verbose
        )

    def get_open_parameters(self, field_size: float = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get open field parameters (fully retracted leaves, open jaws)."""
        if field_size is None:
            field_size = self.treatment_config.field_size
        else:
            field_size = [field_size, field_size]

        num_cps = self.number_of_cps
        mlcs = torch.zeros((1, 2, num_cps, self.machine_config.number_of_leaf_pairs), dtype=self.dtype, device=self.device)
        mlcs[:, 0, :, :] = -field_size[0] / 2
        mlcs[:, 1, :, :] = field_size[0] / 2
        mlcs = mlcs.clone().detach().requires_grad_(True)

        jaws = torch.zeros((1, 2, num_cps), dtype=self.dtype, device=self.device)
        jaws[:, 0, :] = -field_size[1] / 2
        jaws[:, 1, :] = field_size[1] / 2
        jaws = jaws.clone().detach().requires_grad_(True)

        mus = torch.ones((1, num_cps), dtype=self.dtype, device=self.device)
        mus = mus.clone().detach().requires_grad_(True)

        return mlcs, jaws, mus

    def _assert_sizes(self, ct_image, leaf_positions, jaw_positions, mus):
        """Validate input tensor sizes."""
        if ct_image is None:
            raise ValueError("CT image must be provided.")

        B = ct_image.shape[0]
        
        assert ct_image.dim() == 4, \
            f"CT image needs 4 dimensions [B, D, H, W], got {ct_image.dim()}D: {ct_image.shape}"
        assert leaf_positions.dim() == 4, \
            f"Leaf positions needs 4 dimensions [B, 2, CP, N], got {leaf_positions.dim()}D: {leaf_positions.shape}"
        assert jaw_positions.dim() == 3, \
            f"Jaw positions needs 3 dimensions [B, 2, CP], got {jaw_positions.dim()}D: {jaw_positions.shape}"
        assert mus.dim() == 2, \
            f"MUs needs 2 dimensions [B, CP], got {mus.dim()}D: {mus.shape}"

        assert leaf_positions.shape[0] == B and mus.shape[0] == B and jaw_positions.shape[0] == B, \
            f"Batch size mismatch: ct={B}, leaf_positions={leaf_positions.shape[0]}, jaw_positions={jaw_positions.shape[0]}, mus={mus.shape[0]}"

        expected_ct = (B, *self.machine_config.ct_array_shape)
        assert ct_image.shape == expected_ct, \
            f"CT shape mismatch: expected {expected_ct}, got {ct_image.shape}"

        expected_leaf = (B, self.number_of_cps, self.machine_config.number_of_leaf_pairs, 2)
        assert leaf_positions.shape == expected_leaf, \
            f"Leaf positions shape mismatch: expected {expected_leaf}, got {leaf_positions.shape}"

        expected_jaw = (B, self.number_of_cps, 2)
        assert jaw_positions.shape == expected_jaw, \
            f"Jaw positions shape mismatch: expected {expected_jaw}, got {jaw_positions.shape}"

        expected_mus = (B, self.number_of_cps)
        assert mus.shape == expected_mus, \
            f"MUs shape mismatch: expected {expected_mus}, got {mus.shape}"
        
    def _compute_crop_indices(self, leaf_positions_list, jaws, batched_kernels):
        """
        Compute crop indices in (h, w) based on min/max leaf and jaw positions.
        """
        leaf_left = leaf_positions_list[:, 0, :, :]
        leaf_right = leaf_positions_list[:, 1, :, :]
        min_leaf = torch.min(leaf_left)
        max_leaf = torch.max(leaf_right)
        jaw_low = jaws[:, 0, :]
        jaw_high = jaws[:, 1, :]
        min_jaw = torch.min(jaw_low)
        max_jaw = torch.max(jaw_high)

        H, D, W = self.ct_array_shape
        kH, kW = batched_kernels.shape[0], batched_kernels.shape[1]

        SID = float(self.treatment_config.SID)
        dz = self.resolution[0]
        d_idx = D - 1
        depth = self.treatment_config.iso_center[0] + SID - ((D - 1) / 2) * dz + d_idx * dz
        scale = SID / depth
        H_field, W_field = self.treatment_config.field_size_in_pixels

        min_leaf_pix = min_leaf * (W_field - 1)
        max_leaf_pix = max_leaf * (W_field - 1)
        min_jaw_pix = min_jaw * (H_field - 1)
        max_jaw_pix = max_jaw * (H_field - 1)

        w_min_phys = (min_leaf_pix - (W_field - 1) / 2) * scale + (W - 1) / 2
        w_max_phys = (max_leaf_pix - (W_field - 1) / 2) * scale + (W - 1) / 2
        h_min_phys = (min_jaw_pix - (H_field - 1) / 2) * scale + (H - 1) / 2
        h_max_phys = (max_jaw_pix - (H_field - 1) / 2) * scale + (H - 1) / 2

        w_min_idx = torch.clamp(w_min_phys.floor().long() - kW, 0, W - 1)
        w_max_idx = torch.clamp(w_max_phys.ceil().long() + kW, 0, W - 1)
        h_min_idx = torch.clamp(h_min_phys.floor().long() - kH, 0, H - 1)
        h_max_idx = torch.clamp(h_max_phys.ceil().long() + kH, 0, H - 1)
        return h_min_idx, h_max_idx, w_min_idx, w_max_idx

    def forward(
        self,
        leaf_positions: torch.Tensor,
        mus: torch.Tensor,
        jaw_positions: torch.Tensor,
        ct_image: torch.Tensor,
        single_cp: int = None
    ) -> torch.Tensor:
        """
        Runs the full dose calculation pipeline.

        Args:
            leaf_positions: Leaf positions [B, 2, CP, N].
            mus: Monitor units [B, CP].
            jaw_positions: Jaw positions [B, 2, CP].
            ct_image: CT image tensor [B, D, H, W].
            single_cp: If set, return dose for single control point only.

        Returns:
            Dose tensor [B, H, D, W].
        """
        H, D, W = self.machine_config.ct_array_shape

        self._assert_sizes(ct_image, leaf_positions, jaw_positions, mus)

        with torch.amp.autocast(self.device.type, dtype=self.dtype):
            # Compute kernels from CT (no caching)
            if self.treatment_config.downsampling_factor != (1, 1, 1):
                ct_for_kernel = F.avg_pool3d(
                    ct_image.unsqueeze(1), self.treatment_config.downsampling_factor
                ).squeeze(1)
            else:
                ct_for_kernel = ct_image

            with torch.no_grad():
                batched_radiological_depths = self.rad_depth_layer(ct_for_kernel)
                batched_kernels = torch.tensor(
                    self.pencil_beam_kernel_layer(batched_radiological_depths),
                    device=self.device,
                    dtype=self.dtype,
                ).detach()

            if single_cp is not None:
                single_radiological_depth = batched_radiological_depths[single_cp:single_cp+1, ...]
            del batched_radiological_depths

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
            
            if self.crop_volume:
                h_min_idx, h_max_idx, w_min_idx, w_max_idx = self._compute_crop_indices(
                    leaf_positions, jaw_positions, batched_kernels
                )
            else:
                h_min_idx = 0
                h_max_idx = H - 1
                w_min_idx = 0
                w_max_idx = W - 1

            batched_fluence_volumes = self.fluence_volume_layer(
                batched_fluence_maps, (h_min_idx, h_max_idx, w_min_idx, w_max_idx)
            )
            batched_accumulated_dose = self.beam_wise_conv_layer(
                batched_fluence_volumes, batched_kernels
            )
            batched_accumulated_dose.mul_(self.machine_config.mean_photon_energy_MeV)

            if single_cp is not None:
                single_fluence_map = batched_fluence_maps[single_cp:single_cp+1, ...]
            del batched_fluence_volumes, batched_fluence_maps, batched_kernels

            B = leaf_positions.shape[0]
            G = self.number_of_cps
            D_, H_, W_, _ = batched_accumulated_dose.shape[1:]
            batched_accumulated_dose = batched_accumulated_dose.view(B, G, D_, H_, W_)
            batched_accumulated_dose.mul_(mus[:, :, None, None, None])

            batched_accumulated_dose = self.rotation_layer(batched_accumulated_dose)

            if single_cp is None:
                batched_accumulated_dose = batched_accumulated_dose.sum(dim=1)
            else:
                batched_accumulated_dose = batched_accumulated_dose[:, single_cp, ...]

            if self.treatment_config.downsampling_factor != (1, 1, 1):
                batched_accumulated_dose = F.interpolate(
                    batched_accumulated_dose.unsqueeze(1),
                    scale_factor=self.treatment_config.downsampling_factor,
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(1)

            if self.permute_ct:
                batched_accumulated_dose = torch.permute(
                    batched_accumulated_dose, (0, 2, 3, 1)
                )

            if single_cp is None:
                return batched_accumulated_dose
            else:
                return batched_accumulated_dose, single_fluence_map, single_radiological_depth

    def forward_beam_sequence(
        self,
        beam_sequence: BeamSequence,
        ct_image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dose from a BeamSequence.

        Args:
            beam_sequence: BeamSequence (shapes: mus [CP], leaf_positions [CP, N, 2], jaw_positions [CP, 2])
            ct_image: CT image tensor [1, D, H, W]

        Returns:
            Dose tensor [1, H, D, W]
        """
        # Add batching dimension to parameters
        leaf_positions = beam_sequence.leaf_positions.unsqueeze(0)
        mus = beam_sequence.mus.unsqueeze(0)
        jaw_positions = beam_sequence.jaw_positions.unsqueeze(0)

        return self.forward(
            leaf_positions=leaf_positions,
            mus=mus,
            jaw_positions=jaw_positions,
            ct_image=ct_image,
        )

    def forward_single_beam(
        self,
        beam: Beam,
        ct_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dose from a single Beam.

        Args:
            beam: Single Beam (shapes: mu scalar, leaf_positions [N, 2], jaw_positions [2])
            ct_image: CT image tensor [1, D, H, W]
            gantry_angle: Override gantry angle in radians (uses beam.gantry_angle if None)

        Returns:
            Dose tensor [1, H, D, W]
        """

        # Convert from Beam format to forward() format:
        # [N, 2] -> [2, N] -> [2, 1, N] -> [1, 2, 1, N]
        leaf_positions = beam.leaf_positions.unsqueeze(0).unsqueeze(0)
        jaw_positions = beam.jaw_positions.unsqueeze(0).unsqueeze(0)
        mus = beam.mu.unsqueeze(0).unsqueeze(0)

        dose = self.forward(
            leaf_positions=leaf_positions,
            mus=mus,
            jaw_positions=jaw_positions,
            ct_image=ct_image,
        )

        return dose

    def forward_sequential(
        self,
        beam_sequence: BeamSequence,
        ct_image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dose by processing beams sequentially (memory efficient).

        Args:
            beam_sequence: BeamSequence containing all control points
            ct_image: CT image tensor [1, D, H, W]

        Returns:
            Accumulated dose tensor [1, H, D, W]
        """
        total_dose = None

        for i, beam in enumerate(beam_sequence):
            beam_dose = self.forward_single_beam(
                beam,
                ct_image=ct_image,
                gantry_angle=beam_sequence.gantry_angles[i].item(),
            )

            if total_dose is None:
                total_dose = beam_dose
            else:
                total_dose = total_dose + beam_dose

        return total_dose