#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FluenceMapLayer module for generating and resampling fluence maps from leaf positions in radiotherapy.

This module provides the FluenceMapLayer class, which computes the fluence map based on the positions and widths
of multi-leaf collimator (MLC) leaves and jaws. The fluence map is resampled to match the output bin configuration of the
treatment machine, enabling accurate dose modeling and further processing.

Typical usage example::

    from ..MachineConfig import MachineConfig
    import torch
    config = MachineConfig(...)
    layer = FluenceMapLayer(config)
    leaf_positions = torch.tensor(...)
    fluence_map = layer(leaf_positions)

Classes:
    FluenceMapLayer: Torch layer for calculating and resampling fluence maps from leaf positions.
"""

import torch
import torch.nn as nn
from pydose_rt.data import MachineConfig

def fractional_box_overlap(d, left, right):
    """
    Compute fractional overlap using only standard PyTorch operations.
    """
    d32 = d.to(torch.float32)
    half_w = 0.5

    bin_start = d - half_w
    bin_end   = d + half_w
    
    overlap_start = torch.maximum(left, bin_start)
    overlap_end = torch.minimum(right, bin_end)
    overlap = torch.clamp(overlap_end - overlap_start, min=0.0, max=1.0).to(d.dtype)
    
    return overlap

class FluenceMapLayer(nn.Module):
    """
    FluenceMapLayer for generating and resampling fluence maps from leaf positions.

    This layer computes the fluence map based on leaf and jaw positions, resampling the map
    according to the configuration of the treatment machine. It handles the geometric mapping
    and overlap calculations required for accurate dose modeling.

    Attributes:
        config (MachineConfig): Configuration object containing field size, leaf sizes, and number of leafs.
        verbose (bool): Flag to enable verbose logging.
        device (torch.device): Device on which computations are performed (CPU or CUDA).
    """

    def __init__(
        self,
        config: MachineConfig,
        verbose: bool = False,
    ):
        """
        Initializes the FluenceMapLayer.

        Args:
            config (MachineConfig): Configuration object with field_size_in_pixels, leaf_widths, and number_of_leaf_pairs attributes.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.
        """
        super().__init__()
        self.config = config
        self.verbose = verbose
        self.device = self.config.device
        self.dtype = self.config.dtype

        # Precompute depth indices
        W = config.field_size[1]
        N = config.number_of_leaf_pairs
        centers = (torch.arange(W, dtype=self.dtype) + 0.5) - (W / 2)  # [H]
        depth_indices = centers.view(W, 1).repeat(1, N)  # [H, N]
        self.register_buffer("depth_indices", depth_indices.unsqueeze(0).to(self.dtype))  # [1, H, N]

        H = config.field_size[0]
        centers = (torch.arange(H, dtype=self.dtype) + 0.5) - (H / 2)  # [W]
        jaw_indices = centers.view(1, H).repeat(1, 1)
        self.register_buffer("jaw_indices", jaw_indices.unsqueeze(0).to(self.dtype))  # [1, W, N]

    def resample_fluence_map(self, values: torch.Tensor) -> torch.Tensor:
        """
        Resamples the fluence map based on leaf geometry and output bins. calculates
        the fluence values for each output bin by considering the overlapping leaf positions.
        Now one bin equals one pixel in the output fluence map.

        Args:
            values (torch.Tensor): Input fluence values of shape [B*G, W, N, 1].

        Returns:
            torch.Tensor: Resampled fluence map of shape [B*G, W, H, 1].
        """
        B, W, N, _ = values.shape
        H = self.config.field_size[0]
        total_length = sum(self.config.leaf_widths)

        # leaf_widths
        leaf_widths = torch.tensor(
            self.config.leaf_widths, device=values.device, dtype=self.config.dtype
        )

        # Compute start and end positions for each leaf along axis perpendicular to leaf movement
        start_positions = torch.cumsum(
            torch.cat(
                [
                    torch.tensor([0.0], device=values.device, dtype=self.config.dtype),
                    leaf_widths[:-1],
                ]
            ),
            dim=0,
        )
        end_positions = start_positions + torch.tensor(
            self.config.leaf_widths, device=values.device, dtype=self.config.dtype
        )

        # divide field in bin stripes parallel to leaf movement
        output_bin_edges = torch.linspace(
            0.0, total_length, H + 1, device=values.device, dtype=self.config.dtype
        )

        # Store start and end position of each bin
        output_bin_starts = output_bin_edges[:-1]
        output_bin_ends = output_bin_edges[1:]

        # Prepare for overlap calculation (Store leaf data in column vectors and bin data in row vectors)
        start_i = start_positions.view(N, 1)
        end_i = end_positions.view(N, 1)
        start_j = output_bin_starts.view(1, H)
        end_j = output_bin_ends.view(1, H)

        # Compute overlap between leaf and bin positions (Subtract later start with earlier end)
        overlap_start = torch.max(start_i, start_j)
        overlap_end = torch.min(end_i, end_j)
        overlap = (
            (overlap_end - overlap_start).clamp(min=0.0).to(dtype=self.config.dtype)
        )

        # For each bin and depth slice sum up the open area of overlapping leaf pairs in that depth and bin
        overlap_exp = overlap.view(1, 1, N, H)  # [1, 1, N, H]
        weighted = values * overlap_exp
        total_weighted = weighted.sum(dim=2)  # [B, W, H]

        # Isn't total_overlap just the bin width?
        # total_overlap = overlap.sum(dim=0)  # [H]
        total_overlap = overlap.sum(dim=0)  # [M]
        total_overlap = total_overlap.view(1, 1, H)
        # resolution = self.config.resolution[0]
        # total_overlap = torch.full((H,), resolution, device=self.device, dtype=self.config.dtype)
        # total_overlap = total_overlap.view(1, 1, H)

        result = total_weighted / (total_overlap + 1e-8)
        result = result.unsqueeze(-1)  # [B, W, H, 1]

        return result

    def forward(
        self, leaf_positions: torch.Tensor, jaw_positions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Computes the fluence map from leaf and jaw positions. The calculated
        leaf positions are stored in the mask.

        Args:
            leaf_positions (torch.Tensor): Tensor of leaf positions and widths of shape [B, 2, G, N].
            jaws (torch.Tensor): Tensor of jaw positions of shape [B, 2, G].

        Returns:
            torch.Tensor: Fluence map tensor of shape [B*G, W, H, 1].
        """
        B, _, G, N = leaf_positions.shape  # [B, G, 2, N]
        leaf_positions = leaf_positions.permute(0, 2, 1, 3).reshape(
            B * G, 2, N
        )  # [B*G, 2, N]

        left_positions = leaf_positions[:, 0, :]  # [B*G, N]
        right_positions = leaf_positions[:, 1, :]  # [B*G, N]

        W = self.config.field_size[1]

        left_positions = left_positions.unsqueeze(1).repeat(1, W, 1)  # [B*G, H, N]
        right_positions = right_positions.unsqueeze(1).repeat(1, W, 1)  # [B*G, H, N]

        d = self.depth_indices
        if d.device != leaf_positions.device:
            d = d.to(leaf_positions.device)  # [1, H, N]

        # ---------- new box (no sigmoids) ----------
        mask = fractional_box_overlap(d, left_positions, right_positions)
        # -------------------------------------------

        # Reshape
        mask = mask.view(B, G, W, N)
        mask = mask.view(B * G, W, N, 1)

        mask = self.resample_fluence_map(mask)  # [B*G, H, M, 1]

        if jaw_positions is not None:
            jaw_positions = jaw_positions.permute(0, 2, 1).reshape(B * G, 2)  # [B*G, 2]
            bottom_positions = jaw_positions[:, 0].unsqueeze(1)  # [B*G]
            top_positions = jaw_positions[:, 1].unsqueeze(1)  # [B*G]
            H = self.config.field_size[0]
            bottom_positions = bottom_positions.unsqueeze(2).repeat(1, 1, H)  # [B*G, H]
            top_positions = top_positions.unsqueeze(2).repeat(1, 1, H)

            j = self.jaw_indices
            if j.device != leaf_positions.device:
                j = j.to(leaf_positions.device)  # [1, H, N]
            jaw_mask = fractional_box_overlap(j, bottom_positions, top_positions)

            jaw_mask = jaw_mask.view(B, G, H, 1)
            jaw_mask = jaw_mask.view(B * G, H, 1, 1)
            jaw_mask = jaw_mask.repeat(1, 1, W, 1)

            mask *= jaw_mask

        """ print('Fluence map shape:', mask.shape)
        
        # Visualize the first mask in the batch (as example)
        import matplotlib.pyplot as plt
        mask_np = mask[30, ..., 0].detach().cpu().numpy()  # shape: [W, H]
        plt.figure(figsize=(8, 6))
        plt.imshow(mask_np.T, cmap="viridis", aspect="auto")
        plt.title("Fluence Mask at 60° without Jaws")
        plt.xlabel("Pixel (W, leaf movement)")
        plt.ylabel("Pixel (H, perpendicular)")
        # Use matplotlib's default pixel ticks
        plt.colorbar(label="Fluence")
        plt.tight_layout()
        plt.show() """

        return mask.permute(0, 3, 2, 1)
