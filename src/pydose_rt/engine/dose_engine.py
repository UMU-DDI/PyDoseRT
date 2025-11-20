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
from pydose_rt.data import MachineConfig, TreatmentConfig


class DoseEngine(nn.Module):
    """
    Implements the full dose calculation pipeline for radiotherapy, including preprocessing,
    fluence modeling, kernel generation, convolution, and geometric rotation of dose volumes.

    Attributes:
        config (MachineConfig): Configuration object containing model and beam parameters.
        verbose (bool): If True, enables verbose output for debugging.
        debug (bool): If True, enables debug mode with additional outputs.
        device (torch.device): PyTorch device for computation.
        valid_parameters_layer (ValidParametersLayer): Layer for validating input parameters.
        fluence_map_layer (FluenceMapLayer): Layer for generating fluence maps from leaf positions.
        fluence_volume_layer (FluenceVolumeLayer): Layer for expanding fluence maps to volumes.
        rad_depth_layer (RadiologicalDepthLayer): Layer for computing radiological depth.
        pencil_beam_kernel_layer (PencilBeamKernelLayer): Layer for generating pencil beam kernels.
        beam_wise_conv_layer (BeamWiseConvolutionalLayer): Layer for performing beam-wise convolution.
        rot_angles_rad (list): List of gantry angles (in radians) for beam rotation.
    """

    def __init__(
        self,
        machine_config: MachineConfig,
        treatment_config: TreatmentConfig,
        ct_image: torch.Tensor = None,
        leafs_centered: bool = False,
        crop_volume: bool = False,
        permute_ct: bool = False,
        adjust_values: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ) -> "DoseEngine":
        """
        Initializes the DoseEngine pipeline and its layers. Precomputes radiological depths and pencil beam kernels.

        Args:
            ct_image (torch.Tensor): CT image tensor of shape [B, D, H, W].
            config (MachineConfig): Configuration object with model and beam parameters.
            kernel_size (int): Size of the pencil beam kernel.
            verbose (bool, optional): Enables verbose output. Defaults to False.
            debug (bool, optional): Enables debug mode. Defaults to False.
        """
        super().__init__()
        self.machine_config = machine_config
        self.treatment_config = treatment_config
        self.verbose = verbose
        self.debug = debug
        self.device = treatment_config.device
        self.dtype = treatment_config.dtype

        self.crop_volume = crop_volume
        self.permute_ct = permute_ct

        self.resolution = tuple([x * y for x, y in zip(machine_config.resolution,  treatment_config.downsampling_factor)])
        self.ct_array_shape = tuple([int(x / y) for x, y in zip(machine_config.ct_array_shape,  treatment_config.downsampling_factor)])

        self.valid_parameters_layer = ValidParametersLayer(machine_config, treatment_config, leafs_centered, adjust_values=adjust_values)
        self.fluence_map_layer = FluenceMapLayer(machine_config, treatment_config, verbose)
        self.fluence_volume_layer = FluenceVolumeLayer(machine_config, treatment_config, verbose)
        self.rad_depth_layer = RadiologicalDepthLayer(machine_config, treatment_config, verbose)
        self.pencil_beam_kernel_layer = PencilBeamKernelLayer(
            machine_config, treatment_config, verbose
        )
        self.beam_wise_conv_layer = BeamWiseConvolutionalLayer(treatment_config.device, treatment_config.dtype)
        self.rotation_layer = CPRotationLayer(machine_config, treatment_config, verbose)

        self.ct_image = ct_image
        if self.ct_image is not None:
            if self.treatment_config.downsampling_factor != (1, 1, 1):
                self.ct_image = F.avg_pool3d(
                    self.ct_image.unsqueeze(1), treatment_config.downsampling_factor
                ).squeeze(1)

            batched_radiological_depths = self.rad_depth_layer(self.ct_image)

            self.batched_kernels = torch.tensor(
                self.pencil_beam_kernel_layer(batched_radiological_depths),
                device=self.device,
                dtype=treatment_config.dtype,
            ).detach()

    def get_open_parameters(self, field_size: float = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if field_size is None:
            field_size = self.treatment_config.field_size
        else:
            field_size = [field_size, field_size]
        mlcs = torch.zeros((1, 2, self.treatment_config.number_of_cps, self.machine_config.number_of_leaf_pairs), dtype=self.dtype, device=self.device)
        mlcs[:, 0, :, :] = - field_size[0] / 2
        mlcs[:, 1, :, :] = field_size[0] / 2
        mlcs = mlcs.clone().detach().requires_grad_(True)

        jaws = torch.zeros((1, 2, self.treatment_config.number_of_cps), dtype=self.treatment_config.dtype, device=self.treatment_config.device)
        jaws[:, 0, :] = - field_size[1] / 2
        jaws[:, 1, :] = field_size[1] / 2
        jaws = jaws.clone().detach().requires_grad_(True)

        mus = torch.ones((1, self.treatment_config.number_of_cps), dtype=self.treatment_config.dtype, device=self.treatment_config.device)
        mus = mus.clone().detach().requires_grad_(True)

        return mlcs, jaws, mus
    
    def _assert_sizes(self, ct_image, leaf_positions, jaw_positions, mus):

        # Validate inputs
        if self.ct_image is None:
            if ct_image is None:
                raise ValueError(
                    "CT image must be provided either at initialization or in forward()."
                )

        B = ct_image.shape[0]
        assert (
            ct_image.shape[0] == B
            and leaf_positions.shape[0] == B
            and mus.shape[0] == B
        ), "Batch size mismatch."
        assert ct_image.dim() == 4, "CT image needs 4 dimensions [B, D, H, W]"
        assert (
            leaf_positions.dim() == 4
        ), "Leaf positions requires a 4D tensor [B, 2, CP, LVS]"
        assert mus.dim() == 2, "MUs requires a 2D tensor [B, CP]"
        if self.ct_image is None:
            expected = (
                B,
                self.machine_config.ct_array_shape[0],
                self.machine_config.ct_array_shape[1],
                self.machine_config.ct_array_shape[2],
            )
            assert ct_image.shape == expected, f"Incorrect CT input shape. {ct_image.shape} vs {expected}"
        else:
            assert ct_image.shape == (
                B,
                self.machine_config.ct_array_shape[0],
                self.machine_config.ct_array_shape[1],
                self.machine_config.ct_array_shape[2],
            ), "Incorrect CT input shape."
        assert leaf_positions.shape == (
            B,
            2,
            self.treatment_config.number_of_cps,
            self.machine_config.number_of_leaf_pairs,
        ), "Incorrect leaf positions shape"
        assert mus.shape == (B, self.treatment_config.number_of_cps), "Incorrect mus shape"

    def _compute_crop_indices(self, leaf_positions_list, jaws):
        """
        Compute crop indices in (h, w) based on min/max leaf and jaw positions across all batches, expanded by kernel size.
        Uses correct field scaling at the deepest depth (as in FluenceVolumeLayer).
        Returns: (h_min, h_max, w_min, w_max) as integer indices for slicing.
        """
        # leaf_positions_list: [B, 2, G, num_leaf_pairs] (normalized [0,1])
        # jaws: [B, 2, G] (normalized [0,1])
        # Get min/max normalized positions
        leaf_left = leaf_positions_list[:, 0, :, :]  # [B, G, num_leaf_pairs]
        leaf_right = leaf_positions_list[:, 1, :, :]  # [B, G, num_leaf_pairs]
        min_leaf = torch.min(leaf_left)
        max_leaf = torch.max(leaf_right)
        jaw_low = jaws[:, 0, :]  # [B, G]
        jaw_high = jaws[:, 1, :]  # [B, G]
        min_jaw = torch.min(jaw_low)
        max_jaw = torch.max(jaw_high)

        H, D, W = self.ct_array_shape  # (H, D, W)
        kH, kW = self.batched_kernels.shape[0], self.batched_kernels.shape[1]

        SID = float(self.treatment_config.SID)
        dz = self.resolution[0]
        # Deepest depth index
        d_idx = D - 1
        # Compute physical depth for last slice
        depth = self.treatment_config.iso_center[0] + SID - ((D - 1) / 2) * dz + d_idx * dz
        scale = SID / depth
        # Field size in pixels (MLC plane)
        H_field, W_field = self.treatment_config.field_size_in_pixels

        # Map normalized [0,1] to field pixel indices
        min_leaf_pix = min_leaf * (W_field - 1)
        max_leaf_pix = max_leaf * (W_field - 1)
        min_jaw_pix = min_jaw * (H_field - 1)
        max_jaw_pix = max_jaw * (H_field - 1)

        # Project to CT pixel indices at deepest depth using scale
        # Centered at (W-1)/2, (H-1)/2
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
        jaw_positions: torch.Tensor = None,
        ct_image: torch.Tensor = None,
        single_cp: int = None,
        leaf_x: float = 0.0,
        leaf_y: float = 0.0,
        jaw_x: float = 0.0,
        jaw_y: float = 0.0,
    ) -> torch.Tensor:
        """
        Runs the full dose calculation pipeline for a batch of CT images and beam parameters.

        Args:
            leaf_positions (torch.Tensor): List of leaf positions for each beam/control point [1, 2, G, num_leaf_pairs].
            mus (torch.Tensor): Attenuation coefficients for each beam [B, G].
            jaw_positions (torch.Tensor): Tensor of jaw positions of shape [B, 2, G].
            ct_image (torch.Tensor): CT image tensor of shape [B, D, H, W].

        Returns:
            torch.Tensor: Final accumulated dose tensor of shape [B, H, D, W].
        """
        
        H, D, W = self.machine_config.ct_array_shape
        if (ct_image is not None) and self.permute_ct:
            # Convert ct image to be consistent with other dose engines with [B, D, W, H]
            ct_image = torch.permute(
                ct_image, (0, 3, 1, 2)
            )  # [B, D, W, H] -> [B, H, D, W]

        # Validate inputs
        self._assert_sizes(ct_image, leaf_positions, jaw_positions, mus)

        if self.ct_image is not None:
            ct_image = self.ct_image
            batched_kernels = self.batched_kernels
            
        with torch.amp.autocast(self.device.type, dtype=self.dtype):
            if self.ct_image is None:
                with torch.no_grad():
                    batched_radiological_depths = self.rad_depth_layer(ct_image)

                    batched_kernels = torch.tensor(
                        self.pencil_beam_kernel_layer(batched_radiological_depths),
                        device=self.device,
                        dtype=self.dtype,
                    ).detach()

            if single_cp is not None:
                single_radiological_depth = batched_radiological_depths[single_cp:single_cp+1, ...]
            del batched_radiological_depths

            leaf_positions, mus, jaw_positions = self.valid_parameters_layer(
                leaf_positions, mus, jaw_positions
            )

            batched_fluence_maps = self.fluence_map_layer(leaf_positions, jaw_positions, leaf_x, leaf_y, jaw_x, jaw_y)

            if self.crop_volume:
                h_min_idx, h_max_idx, w_min_idx, w_max_idx = self._compute_crop_indices(
                    leaf_positions, jaw_positions
                )
            else:
                # Default to full volume if indices not provided
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
                

            # Insert at correct x indices, keep z cropped
            # partial_shape = (
            #     batched_accumulated_dose.shape[0],
            #     D,
            #     batched_accumulated_dose.shape[2],
            #     W,
            #     1,
            # )
            # partial_dose = torch.zeros(
            #     partial_shape,
            #     dtype=batched_accumulated_dose.dtype,
            #     device=batched_accumulated_dose.device,
            # )
            # partial_dose[:, :, :, w_min_idx : w_max_idx + 1, :] = batched_accumulated_dose
            # batched_accumulated_dose = partial_dose

            # Reshape to [B, G, D, H, W]
            B = leaf_positions.shape[0]
            G = self.treatment_config.number_of_cps
            D_, H_, W_, _ = batched_accumulated_dose.shape[1:]
            batched_accumulated_dose = batched_accumulated_dose.view(B, G, D_, H_, W_)
            batched_accumulated_dose.mul_(mus[:, :, None, None, None])

            batched_accumulated_dose = self.rotation_layer(batched_accumulated_dose)

            if single_cp is None:
                batched_accumulated_dose = batched_accumulated_dose.sum(dim=1)  # [B, H, D, W]
            else:
                batched_accumulated_dose = batched_accumulated_dose[:, single_cp, ...]  # [B, H, D, W]

            # full_shape = (batched_accumulated_dose.shape[0], H, D, W, 1)
            # full_dose = torch.zeros(
            #     full_shape,
            #     dtype=batched_accumulated_dose.dtype,
            #     device=batched_accumulated_dose.device,
            # )
            # full_dose[:, h_min_idx : h_max_idx + 1, :, :, 0] = batched_accumulated_dose
            # batched_accumulated_dose = full_dose[..., 0]  # [B, H, D, W]
            # del full_dose

            if self.treatment_config.downsampling_factor != (1, 1, 1):
                batched_accumulated_dose = F.interpolate(
                    batched_accumulated_dose.unsqueeze(1),
                    scale_factor=self.treatment_config.downsampling_factor,
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(1)

            if self.permute_ct:
                # Convert batched_accumulated_dose back to be consistent with other dose engines
                batched_accumulated_dose = torch.permute(
                    batched_accumulated_dose, (0, 2, 3, 1)
                )  # [B, H, D, W] -> [B, D, W, H]

            if single_cp is None:
                return batched_accumulated_dose
            else:
                return batched_accumulated_dose, single_fluence_map, single_radiological_depth
