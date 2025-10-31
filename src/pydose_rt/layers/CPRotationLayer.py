#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CPPRotationLayer module for performing beam-wise 2D rotation of dose volumes using grid sampling.

This module provides the CPPRotationLayer class, which rotates accumulated dose volumes for each gantry angle
using PyTorch's grid sampling. The layer is designed to handle 5D tensors representing dose distributions across batches,
gantry angles, depth, height, and width.

Typical usage example::
    layer = CPPRotationLayer(config)
    rotated_dose = layer(accumulated_dose)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ModelConfig import ModelConfig

class CPRotationLayer(nn.Module):
    """
    PyTorch module for performing beam-wise 2D rotation of dose volumes using grid sampling.

    Attributes:
        config (ModelConfig): Stores configuration parameters.
        verbose (bool): Verbosity flag.
        device (torch.device): Device on which computations are performed.
        rot_angles_rad (torch.Tensor): Tensor of gantry angles in radians.
    """
    def __init__(self, config: ModelConfig, verbose: bool = False):
        """
        Initializes the CPPRotationLayer.

        Args:
            config (ModelConfig): Configuration parameters for the layer.
            verbose (bool, optional): If True, enables verbose output for debugging. Defaults to False.
        """        
        super().__init__()
        self.config = config
        self.verbose = verbose
        self.device = self.config.device
        self.rot_angles_rad = torch.tensor(self.config.gantry_angles, dtype=self.config.dtype, device=self.device)

    def forward(self, accumulated_dose: torch.Tensor, center: tuple = None) -> torch.Tensor:
        """
        Rotates all [B, G, D, H, W] dose accumulated_dose for all gantry angles in parallel (fully vectorized).
        Args:
            accumulated_dose (torch.Tensor): [B, G, D, H, W]
            rot_angles_rad (torch.Tensor): tensor of G angles in radians
            center (tuple, optional): (cy, cx) voxel coordinates in [D, W] plane. If None, uses center of volume.
        Returns:
            torch.Tensor: Rotated [B, G, H, D, W]
        """

        # TODO: Implement iso center functionality
        B, G, D, H, W = accumulated_dose.shape
        device = accumulated_dose.device
        # Step 1: reshape into slices
        accumulated_dose = accumulated_dose.permute(0, 1, 3, 2, 4)   # [B, G, H, D, W]
        accumulated_dose = accumulated_dose.reshape(B*G*H, 1, D, W)   # [B*G*H, 1, D, W]

        # Step 2: build affine grids for each angle
        cos_a = torch.cos(self.rot_angles_rad)
        sin_a = torch.sin(self.rot_angles_rad)
        mats = torch.zeros((G, 2, 3), device=device, dtype=self.config.dtype)
        mats[:, 0, 0] = cos_a
        mats[:, 0, 1] = -sin_a
        mats[:, 1, 0] = sin_a
        mats[:, 1, 1] = cos_a

        # One grid per angle
        grid2d = F.affine_grid(mats, size=(G, 1, D, W), align_corners=False)  # [G, 1, D, W, 2]
        
        # Repeat each grid H times, and B times
        grid2d = grid2d.unsqueeze(1).unsqueeze(0)              # [1, G, 1, D, W, 2]
        grid2d = grid2d.repeat(B, 1, H, 1, 1, 1)               # [B, G, H, D, W, 2]
        grid2d = grid2d.reshape(B*G*H, D, W, 2)                # [B*G*H, D, W, 2]

        # Rotate
        accumulated_dose = F.grid_sample(accumulated_dose, grid2d,
                                    mode="bilinear",
                                    padding_mode="zeros",
                                    align_corners=False)    # [B*G*H, 1, D, W]

        # Free Memory (TODO: Does this still work when using autograd?)
        del grid2d, mats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Reshape back
        accumulated_dose = accumulated_dose.reshape(B, G, H, D, W)

        return accumulated_dose