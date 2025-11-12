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
from pydose_rt.physics.fluence.fluence_modeling import apply_source_penumbra
from pydose_rt.geometry.projections import fractional_box_overlap, resample_fluence_map



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

        left_positions = leaf_positions[:, 0, :]   # [B*G, N]
        right_positions = leaf_positions[:, 1, :]   # [B*G, N]

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

        mask = resample_fluence_map(mask, self.config.leaf_widths, self.config.field_size[0], self.config.dtype)  # [B*G, H, M, 1]

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
            jaw_mask = jaw_mask.view(B * G, 1, H, 1)
            jaw_mask = jaw_mask.repeat(1, W, 1, 1)

            mask *= jaw_mask

        fluence_map = mask.permute(0, 3, 2, 1)
        # fluence_map = apply_source_penumbra(fluence_map, source_size_mm=3.0, pixel_size_mm=self.config.resolution[2])

        return fluence_map
