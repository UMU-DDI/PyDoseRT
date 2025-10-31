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

from .layers.ValidParametersLayer import ValidParametersLayer
from .layers.FluenceMapLayer import FluenceMapLayer
from .layers.FluenceVolumeLayer import FluenceVolumeLayer
from .layers.RadiologicalDepthLayer import RadiologicalDepthLayer
from .layers.PencilBeamKernelLayer import PencilBeamKernelLayer
from .layers.BeamWiseConvolutionalLayer import BeamWiseConvolutionalLayer
from .layers.CPRotationLayer import CPRotationLayer
from .ModelConfig import ModelConfig


class DoseEngine(nn.Module):
    """
    Implements the full dose calculation pipeline for radiotherapy, including preprocessing,
    fluence modeling, kernel generation, convolution, and geometric rotation of dose volumes.

    Attributes:
        config (ModelConfig): Configuration object containing model and beam parameters.
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
        config: ModelConfig,
        kernel_size: int,
        ct_image: torch.Tensor = None,
        leafs_centered: bool = False,
        crop_volume: bool = False,
        permute_ct: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ):
        """
        Initializes the DoseEngine pipeline and its layers. Precomputes radiological depths and pencil beam kernels.

        Args:
            ct_image (torch.Tensor): CT image tensor of shape [B, D, H, W].
            config (ModelConfig): Configuration object with model and beam parameters.
            kernel_size (int): Size of the pencil beam kernel.
            verbose (bool, optional): Enables verbose output. Defaults to False.
            debug (bool, optional): Enables debug mode. Defaults to False.
        """
        super().__init__()
        self.config = config
        self.verbose = verbose
        self.debug = debug
        self.device = self.config.device

        self.crop_volume = crop_volume
        self.permute_ct = permute_ct

        self.valid_parameters_layer = ValidParametersLayer(config, leafs_centered)
        self.fluence_map_layer = FluenceMapLayer(config, verbose)
        self.fluence_volume_layer = FluenceVolumeLayer(config, verbose)
        self.rad_depth_layer = RadiologicalDepthLayer(config, verbose)
        self.pencil_beam_kernel_layer = PencilBeamKernelLayer(
            config, kernel_size, verbose
        )
        self.beam_wise_conv_layer = BeamWiseConvolutionalLayer(config)
        self.rotation_layer = CPRotationLayer(config, verbose)

        self.ct_image = ct_image
        if self.ct_image is not None:
            if self.config.downsampling_factor != (1, 1, 1):
                self.ct_image = F.avg_pool3d(
                    self.ct_image.unsqueeze(1), self.config.downsampling_factor
                ).squeeze(1)

            batched_radiological_depths = self.rad_depth_layer(self.ct_image)

            self.batched_kernels = torch.tensor(
                self.pencil_beam_kernel_layer(batched_radiological_depths),
                device=self.device,
                dtype=self.config.dtype,
            ).detach()

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
                self.config.ct_array_shape[0] * self.config.downsampling_factor[0],
                self.config.ct_array_shape[1] * self.config.downsampling_factor[1],
                self.config.ct_array_shape[2] * self.config.downsampling_factor[2],
            )
            assert ct_image.shape == expected, f"Incorrect CT input shape. {ct_image.shape} vs {expected}"
        else:
            assert ct_image.shape == (
                B,
                self.config.ct_array_shape[0],
                self.config.ct_array_shape[1],
                self.config.ct_array_shape[2],
            ), "Incorrect CT input shape."
        assert leaf_positions.shape == (
            B,
            2,
            self.config.number_of_cps,
            self.config.number_of_leaf_pairs,
        ), "Incorrect leaf positions shape"
        assert mus.shape == (B, self.config.number_of_cps), "Incorrect mus shape"

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

        H, D, W = self.config.ct_array_shape  # (H, D, W)
        kH, kW = self.batched_kernels.shape[0], self.batched_kernels.shape[1]

        SID = float(self.config.SID)
        dz = self.config.resolution[0]
        # Deepest depth index
        d_idx = D - 1
        # Compute physical depth for last slice
        depth = self.config.iso_center[0] + SID - (D // 2) * dz + d_idx * dz
        scale = SID / depth
        # Field size in pixels (MLC plane)
        H_field, W_field = self.config.field_size_in_pixels

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

        if self.ct_image is None:
            with torch.no_grad():
                if self.config.downsampling_factor != (1, 1, 1):
                    ct_image = F.avg_pool3d(
                        ct_image.unsqueeze(1), self.config.downsampling_factor
                    ).squeeze(1)

                batched_radiological_depths = self.rad_depth_layer(ct_image)

                batched_kernels = torch.tensor(
                    self.pencil_beam_kernel_layer(batched_radiological_depths),
                    device=self.device,
                    dtype=self.config.dtype,
                ).detach()


        leaf_positions, mus, jaw_positions = self.valid_parameters_layer(
            leaf_positions, mus, jaw_positions
        )

        batched_fluence_maps = self.fluence_map_layer(leaf_positions, jaw_positions)

        if self.crop_volume:
            h_min_idx, h_max_idx, w_min_idx, w_max_idx = self._compute_crop_indices(
                leaf_positions, jaw_positions
            )
        else:
            # Default to full volume if indices not provided
            H, D, W = self.config.ct_array_shape
            h_min_idx = 0
            h_max_idx = H - 1
            w_min_idx = 0
            w_max_idx = W - 1

        batched_fluence_volumes = self.fluence_volume_layer(
            batched_fluence_maps, (h_min_idx, h_max_idx, w_min_idx, w_max_idx)
        )
        batched_fluence_volumes.mul_(self.config.mean_photon_energy_MeV)

        batched_accumulated_dose = self.beam_wise_conv_layer(
            batched_fluence_volumes, batched_kernels
        )
        if single_cp is not None:
            single_fluence_map = batched_fluence_maps[single_cp:single_cp+1, ...] 
        del batched_fluence_volumes, batched_fluence_maps, batched_kernels
            

        H, D, W = self.config.ct_array_shape
        # Insert at correct x indices, keep z cropped
        partial_shape = (
            batched_accumulated_dose.shape[0],
            D,
            batched_accumulated_dose.shape[2],
            W,
            1,
        )
        partial_dose = torch.zeros(
            partial_shape,
            dtype=batched_accumulated_dose.dtype,
            device=batched_accumulated_dose.device,
        )
        partial_dose[:, :, :, w_min_idx : w_max_idx + 1, :] = batched_accumulated_dose
        batched_accumulated_dose = partial_dose

        # Reshape to [B, G, D, H, W]
        B = leaf_positions.shape[0]
        G = len(self.config.gantry_angles)
        D_, H_, W_, _ = batched_accumulated_dose.shape[1:]
        batched_accumulated_dose = batched_accumulated_dose.view(B, G, D_, H_, W_)
        batched_accumulated_dose.mul_(mus[:, :, None, None, None])

        batched_accumulated_dose = self.rotation_layer(batched_accumulated_dose)

        if single_cp is None:
            batched_accumulated_dose = batched_accumulated_dose.sum(dim=1)  # [B, H, D, W]
        else:
            batched_accumulated_dose = batched_accumulated_dose[:, single_cp, ...]  # [B, H, D, W]

        full_shape = (batched_accumulated_dose.shape[0], H, D, W, 1)
        full_dose = torch.zeros(
            full_shape,
            dtype=batched_accumulated_dose.dtype,
            device=batched_accumulated_dose.device,
        )
        full_dose[:, h_min_idx : h_max_idx + 1, :, :, 0] = batched_accumulated_dose
        batched_accumulated_dose = full_dose[..., 0]  # [B, H, D, W]
        del full_dose

        if self.config.downsampling_factor != (1, 1, 1):
            batched_accumulated_dose = F.interpolate(
                batched_accumulated_dose.unsqueeze(1),
                scale_factor=self.config.downsampling_factor,
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
            return batched_accumulated_dose, single_fluence_map

    def plot_rotated_dose_slices_sequential(self, rotated_dose):
        """
        Plots the central H slice of the rotated dose for each gantry angle (first batch), one after the other.
        Args:
            rotated_dose (torch.Tensor): [B, G, D, H, W]
        """
        import matplotlib.pyplot as plt

        rotated_dose_np = rotated_dose[0].detach().cpu().numpy()  # [G, D, H, W]
        print("shape:", rotated_dose_np.shape)
        G = rotated_dose_np.shape[0]
        H = rotated_dose_np.shape[2]
        central_h = H // 2
        angles = range(0, G, G // 30)  # Plot 8 angles
        cumulative = None
        for g in range(G):
            slice_2d = rotated_dose_np[g, :, central_h, :]
            if cumulative is None:
                cumulative = slice_2d.copy()
            else:
                cumulative += slice_2d
            if g in angles:
                plt.figure(figsize=(6, 6))
                plt.imshow(cumulative, cmap="jet")
                plt.title(f"Cumulative dose up to angle {g}")
                plt.axis("off")
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.show()
