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

from pydose_rt.data import MachineConfig
from pydose_rt.geometry.rotations import build_rotation_grids

class CPRotationLayer(nn.Module):
    """
    PyTorch module for performing beam-wise 2D rotation of dose volumes using grid sampling.

    Attributes:
        config (MachineConfig): Stores configuration parameters.
        verbose (bool): Verbosity flag.
        device (torch.device): Device on which computations are performed.
        rot_angles_rad (torch.Tensor): Tensor of gantry angles in radians.
    """
    def __init__(self,
                 machine_config: MachineConfig,
                 device: torch.device, 
                 dtype: type,
                 gantry_angles: list[float] | torch.Tensor = None,
                 verbose: bool = False,
                ):
        """
        Initializes the CPRotationLayer.
        Args:
            machine_config: Configuration parameters for the layer.
            verbose: If True, enables verbose output for debugging.
            gantry_angles: Gantry angles in radians. If None, uses treatment_config.gantry_angles.
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.machine_config = machine_config
        self.verbose = verbose

        self.rot_angles_rad = gantry_angles.to(dtype=self.dtype, device=self.device)

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
        grid2d = build_rotation_grids(accumulated_dose.shape, self.rot_angles_rad, self.device, self.dtype)
        accumulated_dose = accumulated_dose.permute(0, 1, 3, 2, 4)   # [B, G, H, D, W]
        accumulated_dose = accumulated_dose.reshape(B*G*H, 1, D, W)   # [B*G*H, 1, D, W]
        
        # Rotate
        accumulated_dose = F.grid_sample(accumulated_dose, grid2d,
                                    mode="bilinear",
                                    padding_mode="zeros",
                                    align_corners=False)    # [B*G*H, 1, D, W]

        # Free Memory (TODO: Does this still work when using autograd?)
        del grid2d
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Reshape back
        accumulated_dose = accumulated_dose.reshape(B, G, H, D, W)

        return accumulated_dose