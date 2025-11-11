#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RadiologicalDepthLayer module for computing radiological depth profiles through CT volumes for radiotherapy.

This module provides the RadiologicalDepthLayer class, which calculates the cumulative radiological depth
along lines through a CT volume at specified gantry angles. It rotates and samples the CT volume,
converts Hounsfield Units (HU) to density, and integrates the density along the beam path for each angle.

Typical usage example::

    from ..MachineConfig import MachineConfig
    import torch
    config = MachineConfig(...)
    layer = RadiologicalDepthLayer(config)
    ct_stack = torch.tensor(...)
    depth_profiles = layer(ct_stack)

Classes:
    RadiologicalDepthLayer: Torch layer for computing radiological depth profiles through CT volumes.
"""
import math
import torch
import torch.nn as nn

from pydose_rt.data.machine_config import MachineConfig
from pydose_rt.physics.attenuation.hu_density_conversion import convert_HU_to_density
from pydose_rt.geometry.rotations import get_radiological_depth_indices


class RadiologicalDepthLayer(nn.Module):
    """
    Torch layer for computing radiological depth profiles through CT volumes at specified gantry angles.

    This layer rotates and samples the CT volume along lines corresponding to different gantry angles,
    converts HU to density, and integrates the density along the beam path to produce radiological
    depth profiles for dose calculation.

    Attributes:
        config (MachineConfig): Configuration object containing CT array shape, gantry angles, resolution, and lookup table for HU-to-density conversion.
        verbose (bool): Flag to enable verbose logging.
        device (torch.device): Device on which computations are performed (CPU or CUDA).
        stacked_indices (torch.Tensor): Precomputed indices for sampling CT volume along rotated lines for each gantry angle.

    """

    def __init__(self, config: MachineConfig, verbose: bool = False):
        """
        Initializes the RadiologicalDepthLayer and precomputes sampling indices for each gantry angle.

        Args:
            config (MachineConfig): Configuration object with CT array shape, gantry angles, resolution, and lookup table.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.
        """
        super(RadiologicalDepthLayer, self).__init__()
        self.config = config
        self.verbose = verbose
        self.device = self.config.device
        stacked_indices = get_radiological_depth_indices(self.config.ct_array_shape, self.config.gantry_angles, self.config.dtype).to(self.device)

        # Final shape: [M, N, 3]
        self.register_buffer(
            "stacked_indices", stacked_indices
        )  # shape: [M, N, 3]

    def forward(self, ct_stack: torch.Tensor) -> torch.Tensor:
        """
        Computes radiological depth profiles through the CT volume for each gantry angle.

        Args:
            ct_stack (torch.Tensor): CT volume tensor of shape [B, H, W, D].

        Returns:
            torch.Tensor: Radiological depth profiles of shape [B*G, P, 1], where
            P is the number of sampled points along each line.
        """
        with torch.no_grad():
            B, H, D, W = ct_stack.shape
            _, G, P, _ = self.stacked_indices.shape

            # Prepare batched indices for sampling CT volume along rotated lines
            batch_ids = (
                torch.arange(B, device=self.device).view(B, 1, 1, 1).expand(B, G, P, 1)
            )
            index_expanded = (
                self.stacked_indices.expand(B, G, P, 3).to(self.device)
            )
            batched_indices = torch.cat(
                [batch_ids, index_expanded], dim=-1
            )  # [B, G, P, 4]

            # Flatten for advanced indexing
            flat_indices = batched_indices.view(-1, 4)
            b_idx, x_idx, y_idx, z_idx = (
                flat_indices[:, 0],
                flat_indices[:, 1],
                flat_indices[:, 2],
                flat_indices[:, 3],
            )

            # Gather voxel values along each line for each batch and angle
            gathered = ct_stack[b_idx, z_idx, y_idx, x_idx]

            # Reshape to [B, G, P]
            gathered = gathered.view(B, G, P)

            # Convert HU to density using lookup table
            density = convert_HU_to_density(
                gathered, self.config.lookup_table
            )  # shape: [B, M, N]

            # Calculate the actual step size for each angle
            step_sizes = []
            for i in range(self.config.number_of_cps):
                if P > 1:
                    diff = self.stacked_indices[0, i, 1:, :] - self.stacked_indices[0, i, :-1, :]
                    # Calculate physical step size accounting for resolution in all dimensions
                    # diff is in pixel units, so multiply by resolution to get physical distance
                    res_tensor = torch.tensor([self.config.resolution[0], self.config.resolution[1], self.config.resolution[2]],
                                             device=self.device, dtype=self.config.dtype)
                    physical_diff = diff * res_tensor
                    avg_step = torch.sqrt((physical_diff ** 2).sum(dim=-1)).mean()
                # Calculate the actual distance between consecutive sample points
                else:
                    # For 0 degrees, it should be close to resolution[1]
                    avg_step = self.config.resolution[1]
                
                step_sizes.append(avg_step)

            step_sizes = torch.tensor(step_sizes, device=self.device, dtype=self.config.dtype).view(1, G, 1)
            # Integrate density along each line (cumulative sum) and scale by resolution
            cumsum = (
                torch.cumsum(density, dim=-1) * step_sizes
            )  # shape: [B, G, P]
            cumsum = cumsum.view(B * G, P, 1)

            return cumsum
