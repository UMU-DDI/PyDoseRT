#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FluenceVolumeLayer module for projecting 2D fluence maps into 3D dose volumes in radiotherapy.

This module provides the FluenceVolumeLayer class, which takes a 2D fluence map and projects it through a CT volume,
applying geometric and profile corrections to generate a 3D volume suitable for dose calculation. It precomputes sampling grids
and profile corrections for efficient forward passes and accurate modeling of the dose distribution.

Typical usage example::

    from ..ModelConfig import ModelConfig
    import torch
    config = ModelConfig(...)
    layer = FluenceVolumeLayer(config)
    fluence_map = torch.tensor(...)
    fluence_volume = layer(fluence_map)

Classes:
    FluenceVolumeLayer: Torch layer for projecting 2D fluence maps into 3D dose volumes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ModelConfig import ModelConfig


class FluenceVolumeLayer(nn.Module):
    """
    FluenceVolumeLayer for projecting 2D fluence maps into 3D dose volumes.

    This layer takes a 2D fluence map and projects it through the CT volume, applying geometric and profile corrections
    to generate a 3D volume suitable for dose calculation. It precomputes sampling grids and profile corrections for efficient forward passes.

    Attributes:
        config (ModelConfig): Configuration object containing CT array shape, resolution, SID, fluence profile, and iso center.
        verbose (bool): Flag to enable verbose logging.
        device (torch.device): Device on which computations are performed (CPU or CUDA).
        SID (float): Source-to-isocenter distance.
        profile_radius (torch.Tensor): Radii for fluence profile correction.
        profile_factors (torch.Tensor): Correction factors for fluence profile.
        D (int): Number of slices in CT volume.
        profile_corrections (torch.Tensor): Precomputed profile corrections for each depth.
        sampling_grids (torch.Tensor): Precomputed ray sampling grids for mapping MLC plane to CT volume.
    """

    def __init__(self, config: ModelConfig, verbose: bool = False):
        """
        Initializes the FluenceVolumeLayer and precomputes profile corrections and sampling grids.

        Args:
            config (ModelConfig): Configuration object with CT array shape, resolution, SID, fluence profile, and iso center.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.
        """
        super().__init__()
        self.config = config
        self.verbose = verbose
        self.device = self.config.device
        # Configuration & Constants
        self.SID = float(config.SID)
        self.profile_radius = torch.tensor(
            config.fluence_profile[0], dtype=self.config.dtype
        )
        self.profile_factors = torch.tensor(
            config.fluence_profile[1], dtype=self.config.dtype
        )

        H, D, W = config.ct_array_shape
        self.D = D

        # Precompute the physical depth (distance from source for each depth slice)
        depths = (
            self.config.iso_center[1]
            + self.config.SID
            - (self.D // 2) * self.config.resolution[1]
            + torch.arange(D, dtype=self.config.dtype) * config.resolution[1]
        )  # mm

        # Compute the physical coordinates for each pixel in the depth slice (center is (0,0))
        ws = (
            torch.arange(W, dtype=self.config.dtype) - (W - 1) / 2
        ) * config.resolution[2]
        hs = (
            torch.arange(H, dtype=self.config.dtype) - (H - 1) / 2
        ) * config.resolution[0]
        WT, HT = torch.meshgrid(ws, hs, indexing="ij")  # Both [W,H]
        H_field, W_field = config.field_size_in_pixels

        # Calculate the inverse relative square distance for each depth
        corrections = []
        sample_grids = []
        for d in depths:
            scale = self.config.SID / d
            # SAT: For now, no profile correction is added
            # r = torch.sqrt(WT**2 + HT**2)
            # scaled_r = r * scale  # project back to MLC plane
            # p = self._interpolate_profile(scaled_r)
            inv_square = scale**2
            corrections.append(inv_square)  # * p)

            gy = WT * scale / (((W_field - 1) / 2) * config.resolution[2])
            gz = HT * scale / (((H_field - 1) / 2) * config.resolution[0])

            gs = torch.stack((gz, gy), dim=-1)
            sample_grids.append(gs)

        self.register_buffer(
            "profile_corrections", torch.stack(corrections).to(self.device)
        )  # [D,W,H]
        self.register_buffer(
            "sampling_grids", torch.stack(sample_grids).to(self.device)
        )  # [D,W,H,2]

    def _interpolate_profile(self, scaled_r):
        """
        1D linear interpolation of the beam profile.
        """
        x = self.profile_radius
        y = self.profile_factors
        flat_r = scaled_r.flatten()
        indices = torch.searchsorted(x, flat_r, right=False)
        indices = torch.clamp(indices, 1, len(x) - 1)
        x0 = x[indices - 1]
        x1 = x[indices]
        y0 = y[indices - 1]
        y1 = y[indices]
        dydx = (y1 - y0) / (x1 - x0 + 1e-8)
        res = y0 + (flat_r - x0) * dydx
        return res.view_as(scaled_r)

    def forward(
        self, fluence_map: torch.Tensor, bbox: tuple[int, int, int, int] = (None, None, None, None)
    ) -> torch.Tensor:
        """
        Projects the 2D fluence map into the 3D CT volume, applying geometric and profile corrections.

        Args:
            fluence_map (torch.Tensor): Input fluence map of shape [B*G, W_field, H_field, 1].
            bbox (h_min_idx, h_max_idx, w_min_idx, w_max_idx) (int): Crop indices for output volume.

        Returns:
            torch.Tensor: 3D volume grid of shape [B*G, D, cropped_W, cropped_H, 1] representing the projected fluence.
        """
        B = fluence_map.shape[0]
        fluence_map = fluence_map.permute(0, 3, 1, 2)  # -> [B*G,1,H_field,W_field]
        H, D, W = self.config.ct_array_shape
        h_min_idx, h_max_idx, w_min_idx, w_max_idx = bbox
        h_min_idx = 0 if h_min_idx is None else h_min_idx
        h_max_idx = H - 1 if h_max_idx is None else h_max_idx
        w_min_idx = 0 if w_min_idx is None else w_min_idx
        w_max_idx = W - 1 if w_max_idx is None else w_max_idx
        
        vol_slices = []
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
            corr = self.profile_corrections[d].unsqueeze(0).unsqueeze(-1)
            vol_slices.append(sampled * corr)
        volume_grid = torch.stack(vol_slices, dim=1)  # [B*G,D,cropped_W,cropped_H,1]
        # Free Memory (TODO: Does this still work when using autograd?)
        del sampled, fluence_map, grid, corr, vol_slices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        """ print('Volume grid shape:', volume_grid.shape)
        import matplotlib.pyplot as plt
        vol = volume_grid[30, ..., 0].detach().cpu().numpy()  # [D, W, H]

        d, w, h = vol.shape
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(vol[d//2, :, :].T, cmap='viridis')
        axs[0].set_title('Axial (y=mid)')
        axs[1].imshow(vol[:, w//2, :].T, cmap='viridis')
        axs[1].set_title('Coronal (x=mid)')
        axs[2].imshow(vol[:, :, h//2], cmap='viridis')
        axs[2].set_title('Sagittal (z=mid)')
        plt.tight_layout()
        plt.show() """

        return volume_grid
