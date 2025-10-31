#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RadiologicalDepthLayer module for computing radiological depth profiles through CT volumes for radiotherapy.

This module provides the RadiologicalDepthLayer class, which calculates the cumulative radiological depth
along lines through a CT volume at specified gantry angles. It rotates and samples the CT volume,
converts Hounsfield Units (HU) to density, and integrates the density along the beam path for each angle.

Typical usage example::

    from ..ModelConfig import ModelConfig
    import torch
    config = ModelConfig(...)
    layer = RadiologicalDepthLayer(config)
    ct_stack = torch.tensor(...)
    depth_profiles = layer(ct_stack)

Classes:
    RadiologicalDepthLayer: Torch layer for computing radiological depth profiles through CT volumes.
"""
import math
import torch
import torch.nn as nn

from pydose_rt.ModelConfig import ModelConfig
from pydose_rt.utils.plotting import convert_HU_to_density


class RadiologicalDepthLayer(nn.Module):
    """
    Torch layer for computing radiological depth profiles through CT volumes at specified gantry angles.

    This layer rotates and samples the CT volume along lines corresponding to different gantry angles,
    converts HU to density, and integrates the density along the beam path to produce radiological
    depth profiles for dose calculation.

    Attributes:
        config (ModelConfig): Configuration object containing CT array shape, gantry angles, resolution, and lookup table for HU-to-density conversion.
        verbose (bool): Flag to enable verbose logging.
        device (torch.device): Device on which computations are performed (CPU or CUDA).
        stacked_indices (torch.Tensor): Precomputed indices for sampling CT volume along rotated lines for each gantry angle.

    """

    def __init__(self, config: ModelConfig, verbose: bool = False):
        """
        Initializes the RadiologicalDepthLayer and precomputes sampling indices for each gantry angle.

        Args:
            config (ModelConfig): Configuration object with CT array shape, gantry angles, resolution, and lookup table.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.
        """
        super(RadiologicalDepthLayer, self).__init__()
        self.config = config
        self.verbose = verbose
        self.device = self.config.device
        H, D, W = self.config.ct_array_shape
        y = torch.linspace(0, D - 1, D)
        x = torch.linspace(0, W - 1, W)
        # Storing the two seperately enables more efficient torch operations
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")  # shape [W, D]

        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, W, D, 2]

        indices_list = []

        for angle in self.config.gantry_angles:
            theta = angle
            rot_matrix = torch.tensor(
                [
                    [math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)],
                ]
            )

            # Centered grid for rotation
            center_y = (D - 1) / 2.0
            center_x = (W - 1) / 2.0
            shifted = grid[0] - torch.tensor([center_x, center_y])
            rotated = torch.matmul(shifted, rot_matrix.T) + torch.tensor(
                [center_x, center_y]
            )

            mid_x = W // 2
            line_points = rotated[mid_x, :]  # shape [D, 2]

            z_index = H // 2
            z_col = torch.full(
                (line_points.shape[0], 1), z_index, dtype=self.config.dtype
            )
            indices = torch.cat([line_points, z_col], dim=-1)  # [D, 3]

            indices_list.append(indices.int())

        # Final shape: [M, N, 3]
        self.register_buffer(
            "stacked_indices", torch.stack(indices_list, dim=0)
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
            G, P, _ = self.stacked_indices.shape

            # Prepare batched indices for sampling CT volume along rotated lines
            batch_ids = (
                torch.arange(B, device=self.device).view(B, 1, 1, 1).expand(B, G, P, 1)
            )
            index_expanded = (
                self.stacked_indices.view(1, G, P, 3).expand(B, G, P, 3).to(self.device)
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

            # Integrate density along each line (cumulative sum) and scale by resolution (TODO doesnt the resolution change with gantry angle )
            cumsum = (
                torch.cumsum(density, dim=-1) * self.config.resolution[1]
            )  # shape: [B, G, P]
            cumsum = cumsum.view(B * G, P, 1)

            # Example: plot for batch 0 and angle 0
            # Reshape indices to [B, G, P] to select by batch and angle
            """ for angle_to_plot in (0,45,90,135):
                print(gathered[0, angle_to_plot, :10])  # [P]
                y_idx_reshaped = y_idx.view(B, G, P)
                x_idx_reshaped = x_idx.view(B, G, P)
                batch_to_plot = 0
                y_points = y_idx_reshaped[batch_to_plot, angle_to_plot, :]  # W axis (width)
                x_points = x_idx_reshaped[batch_to_plot, angle_to_plot, :]  # D axis (depth)
                print('X points:', x_points[:10])
                print('Y points:', y_points[:10])
                central_slice = ct_stack[batch_to_plot, H // 2, :, :]  # [W, D]
                import matplotlib.pyplot as plt
                plt.imshow(central_slice.cpu().numpy(), cmap='gray', origin='lower')
                plt.plot(x_points.cpu().numpy()[:64], y_points.cpu().numpy()[:64], 'r.-')
                # Draw an arrow indicating the direction (from first to last point)
                start_y, start_x = y_points[0].item(), x_points[0].item()
                end_y, end_x = y_points[-1].item(), x_points[-1].item()
                plt.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                        head_width=2, head_length=4, fc='yellow', ec='yellow', length_includes_head=True)
                plt.title('Sampled line overlay on H=H/2 slice (with direction)')
                plt.xlabel('Y axis')
                plt.ylabel('X axis')
                plt.show() """
            # Output shape: [B*G, P, 1]
            return cumsum
