#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FluenceVolumeLayer module for projecting 2D fluence maps into 3D dose volumes in radiotherapy.

This module provides the FluenceVolumeLayer class, which takes a 2D fluence map and projects it through a CT volume,
applying geometric and profile corrections to generate a 3D volume suitable for dose calculation. It precomputes sampling grids
and profile corrections for efficient forward passes and accurate modeling of the dose distribution.

Typical usage example::

    from ..MachineConfig import MachineConfig
    import torch
    config = MachineConfig(...)
    layer = FluenceVolumeLayer(config)
    fluence_map = torch.tensor(...)
    fluence_volume = layer(fluence_map)

Classes:
    FluenceVolumeLayer: Torch layer for projecting 2D fluence maps into 3D dose volumes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pydose_rt.data import MachineConfig, TreatmentConfig


class FluenceVolumeLayer(nn.Module):
    """
    FluenceVolumeLayer for projecting 2D fluence maps into 3D dose volumes.

    This layer takes a 2D fluence map and projects it through the CT volume, applying geometric and profile corrections
    to generate a 3D volume suitable for dose calculation. It precomputes sampling grids and profile corrections for efficient forward passes.

    Attributes:
        config (MachineConfig): Configuration object containing CT array shape, resolution, SID, fluence profile, and iso center.
        verbose (bool): Flag to enable verbose logging.
        device (torch.device): Device on which computations are performed (CPU or CUDA).
        SID (float): Source-to-isocenter distance.
        profile_radius (torch.Tensor): Radii for fluence profile correction.
        profile_factors (torch.Tensor): Correction factors for fluence profile.
        D (int): Number of slices in CT volume.
        profile_corrections (torch.Tensor): Precomputed profile corrections for each depth.
        sampling_grids (torch.Tensor): Precomputed ray sampling grids for mapping MLC plane to CT volume.
    """

    def __init__(self, machine_config: MachineConfig, 
                 treatment_config: TreatmentConfig, 
                 verbose: bool = False):
        """
        Initializes the FluenceVolumeLayer and precomputes profile corrections and sampling grids.

        Args:
            config (MachineConfig): Configuration object with CT array shape, resolution, SID, fluence profile, and iso center.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.
        """
        super().__init__()

        self.device=treatment_config.device
        self.dtype=treatment_config.dtype
        self.machine_config = machine_config
        self.treatment_config = treatment_config
        self.verbose = verbose
        # Configuration & Constants
        self.SID = float(treatment_config.SID)
        self.profile_radius = torch.tensor(
            machine_config.fluence_profile[0], dtype=self.dtype
        )
        self.profile_factors = torch.tensor(
            machine_config.fluence_profile[1], dtype=self.dtype
        )
        self.resolution = tuple([x * y for x, y in zip(machine_config.resolution,  treatment_config.downsampling_factor)])
        self.ct_array_shape = tuple([int(x / y) for x, y in zip(machine_config.ct_array_shape,  treatment_config.downsampling_factor)])
        H, D, W = self.ct_array_shape
        self.D = D


        # Precompute the physical depth (distance from source for each depth slice)
        depths = (
            self.treatment_config.iso_center[1]
            + self.treatment_config.SID
            - ((self.D - 1) // 2) * self.resolution[1]
            + torch.arange(D, dtype=self.dtype) * self.resolution[1]
        )  # mm

        # Compute the physical coordinates for each pixel in the depth slice (center is (0,0))
        H_field, W_field = self.treatment_config.field_size
        hs = (
        torch.arange(H, dtype=self.dtype) - (H - 1) / 2
        ) * self.resolution[0]
        ws = (
            torch.arange(W, dtype=self.dtype) - (W - 1) / 2
        ) * self.resolution[2]
        WT, HT = torch.meshgrid(ws, hs, indexing="ij")  # Both [W, H]

        # Normalization factors use the field size (fluence map coordinates)
        WT_max = ((W_field - 1) / 2)
        HT_max = ((H_field - 1) / 2)

        # Calculate the inverse relative square distance for each depth
        corrections = []
        sample_grids = []
        for d in depths:
            scale = self.treatment_config.SID / d
            inv_square = scale**2
            corrections.append(inv_square)

            gy = (WT / WT_max) * scale
            gz = (HT / HT_max) * scale

            gs = torch.stack((gy, gz), dim=-1)
            sample_grids.append(gs)

        self.register_buffer(
            "profile_corrections", torch.stack(corrections).to(self.device)
        )  # [D,W,H]
        self.register_buffer(
            "sampling_grids", torch.stack(sample_grids).to(self.device)
        )  # [D,W,H,2]

    def forward(
        self, fluence_map: torch.Tensor, bbox: tuple[int, int, int, int] = (None, None, None, None)
    ) -> torch.Tensor:
        """
        Projects the 2D fluence map into the 3D CT volume, applying geometric and profile corrections.

        Args:
            fluence_map (torch.Tensor): Input fluence map of shape [B*G,1,H_field,W_field].
            bbox (h_min_idx, h_max_idx, w_min_idx, w_max_idx) (int): Crop indices for output volume.

        Returns:
            torch.Tensor: 3D volume grid of shape [B*G, D, cropped_W, cropped_H, 1] representing the projected fluence.
        """
        B = fluence_map.shape[0]
        fluence_map = fluence_map.unsqueeze(1)
        H, D, W = self.machine_config.ct_array_shape
        h_min_idx, h_max_idx, w_min_idx, w_max_idx = bbox
        h_min_idx = 0 if h_min_idx is None else h_min_idx
        h_max_idx = H - 1 if h_max_idx is None else h_max_idx
        w_min_idx = 0 if w_min_idx is None else w_min_idx
        w_max_idx = W - 1 if w_max_idx is None else w_max_idx
        
        vol_slices = []
        open_volumes = torch.sum(fluence_map, [1, 2, 3], keepdims=True)
        for d in range(self.D):
            # Get the precomputed sampling grid of the slice, crop to region
            grid = (
                self.sampling_grids[d][
                    w_min_idx : w_max_idx + 1, h_min_idx : h_max_idx + 1, :
                ]
                .unsqueeze(0)
                .repeat(B, 1, 1, 1)
            ).to(fluence_map.dtype)
            # Use the grid to sample the 2D fluence map into the slice
            sampled = F.grid_sample(
                fluence_map,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.permute(0, 2, 3, 1)  # [B*G,cropped_W,cropped_H,1]
            # Apply correction
            # corr = self.profile_corrections[d].unsqueeze(0).unsqueeze(-1)
            corr = (open_volumes / torch.sum(sampled, (1, 2, 3), keepdims=True)).to(self.dtype)
            vol_slices.append(sampled * corr)
        volume_grid = torch.stack(vol_slices, dim=1)  # [B*G,D,cropped_W,cropped_H,1]
        # Free Memory (TODO: Does this still work when using autograd?)
        del sampled, fluence_map, grid, corr, vol_slices, open_volumes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        volume_grid = volume_grid.permute(0, 1, 3, 2, 4)
        return volume_grid
