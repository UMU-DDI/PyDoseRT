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
from pydose_rt.data import MachineConfig, TreatmentConfig
from pydose_rt.geometry.projections import fractional_box_overlap, resample_fluence_map
from pydose_rt.geometry.rotations import rotate_2d_images
from pydose_rt.physics.fluence.fluence_modeling import (
    apply_source_penumbra,
    apply_mlc_scatter,
    apply_head_scatter,
    apply_tongue_and_groove,
    LearnableFluenceKernel
)



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
        machine_config: MachineConfig,
        treatment_config: TreatmentConfig,
        verbose: bool = False,
        training_sharpness: float = 10.0,
    ):
        """
        Initializes the FluenceMapLayer.

        Args:
            config (MachineConfig): Configuration object with field_size_in_pixels, leaf_widths, and number_of_leaf_pairs attributes.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.
            training_sharpness (float, optional): Sharpness parameter for smooth gradients during training. Defaults to 10.0.
            eval_sharpness (float, optional): Sharpness parameter for sharp edges during evaluation. Defaults to 1000.0.
        """
        super().__init__()

        self.device=treatment_config.device
        self.dtype=treatment_config.dtype
        self.machine_config = machine_config
        self.treatment_config = treatment_config
        self.resolution = tuple([x * y for x, y in zip(machine_config.resolution,  treatment_config.downsampling_factor)])
        self.verbose = verbose
        self.training_sharpness = training_sharpness

        if self.machine_config.leaf_widths is None:
            self.leaf_widths = torch.ones((self.machine_config.number_of_leaf_pairs, ), dtype=self.dtype) * self.treatment_config.field_size[1] / self.machine_config.number_of_leaf_pairs
        else:
            self.leaf_widths = self.machine_config.leaf_widths

        # Precompute depth indices
        W = treatment_config.field_size[1]
        N = machine_config.number_of_leaf_pairs
        centers = (torch.arange(W, dtype=self.dtype) + 0.5) - (W / 2)  # [H]
        depth_indices = centers.view(W, 1).repeat(1, N)  # [H, N]
        self.register_buffer("depth_indices", depth_indices.unsqueeze(0).to(self.dtype))  # [1, H, N]

        H = treatment_config.field_size[0]
        centers = (torch.arange(H, dtype=self.dtype) + 0.5) - (H / 2)  # [W]
        jaw_indices = centers.view(1, H).repeat(1, 1)
        self.register_buffer("jaw_indices", jaw_indices.unsqueeze(0).to(self.dtype))  # [1, W, N]

        # Store collimator angles (beam limiting device angles) for rotating fluence maps
        self.collimator_angles = torch.tensor(
            treatment_config.beam_limiting_device_angle,
            dtype=self.dtype,
            device=self.device
        )  # [1, G] in radians

        if self.treatment_config.fluence_kernel_size > 0:
            self.learnable_kernel = LearnableFluenceKernel(
                kernel_size=self.treatment_config.fluence_kernel_size
            )


    def forward(
        self, leaf_positions: torch.Tensor, 
        jaw_positions: torch.Tensor = None,
        leaf_x: float=0.0,
        leaf_y: float=0.0,
        jaw_x: float=0.0,
        jaw_y: float=0.0,
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

        W = self.treatment_config.field_size[1]

        left_positions = left_positions.unsqueeze(1).repeat(1, W, 1)  # [B*G, H, N]
        right_positions = right_positions.unsqueeze(1).repeat(1, W, 1)  # [B*G, H, N]

        d = self.depth_indices
        if d.device != leaf_positions.device:
            d = d.to(leaf_positions.device)  # [1, H, N]

        # Use training-dependent sharpness: smooth gradients during training, sharp during eval
        sharpness = self.training_sharpness if self.training else None

        # ---------- new box (no sigmoids) ----------
        mask = fractional_box_overlap(d, left_positions + leaf_x, right_positions + leaf_y, sharpness)
        # -------------------------------------------

        # Reshape
        mask = mask.view(B, G, W, N)
        mask = mask.view(B * G, W, N, 1)

        mask = resample_fluence_map(mask, self.leaf_widths, self.treatment_config.field_size[0], self.dtype)  # [B*G, H, M, 1]

        if jaw_positions is not None:
            jaw_positions = jaw_positions.permute(0, 2, 1).reshape(B * G, 2)  # [B*G, 2]
            bottom_positions = jaw_positions[:, 0].unsqueeze(1)  # [B*G]
            top_positions = jaw_positions[:, 1].unsqueeze(1)  # [B*G]
            H = self.treatment_config.field_size[0]
            bottom_positions = bottom_positions.unsqueeze(2).repeat(1, 1, H)  # [B*G, H]
            top_positions = top_positions.unsqueeze(2).repeat(1, 1, H)

            j = self.jaw_indices
            if j.device != leaf_positions.device:
                j = j.to(leaf_positions.device)  # [1, H, N]
            jaw_mask = fractional_box_overlap(j, bottom_positions + jaw_x, top_positions + jaw_y, sharpness)

            jaw_mask = jaw_mask.view(B, G, H, 1)
            jaw_mask = jaw_mask.view(B * G, 1, H, 1)
            jaw_mask = jaw_mask.repeat(1, W, 1, 1)

            mask *= jaw_mask

        fluence_map = mask.permute(0, 3, 2, 1)

        if self.treatment_config.fluence_kernel_size > 0:
            fluence_map = fluence_map[:, 0, :, :]  # [B*G, H, W]
            fluence_map = self.learnable_kernel(fluence_map)
        else:
            # Apply MLC scatter tail (distance-dependent scatter from field edges)
            if self.machine_config.mlc_scatter_amplitude > 0:
                fluence_map = apply_mlc_scatter(
                    fluence_map,
                    scatter_amplitude=self.machine_config.mlc_scatter_amplitude,
                    scatter_range_mm=self.machine_config.mlc_scatter_range_mm,
                    pixel_size_mm=1.0
                )

            # Apply tongue-and-groove effect at leaf boundaries
            if self.machine_config.tongue_groove_reduction > 0:
                # Calculate leaf boundary positions in mm
                leaf_boundaries_mm = []
                if self.machine_config.leaf_widths is not None:
                    cumulative_pos = -self.treatment_config.field_size[0] / 2.0
                    for width in self.machine_config.leaf_widths[:-1]:  # Skip last boundary
                        cumulative_pos += width
                        leaf_boundaries_mm.append(cumulative_pos)
                else:
                    # Uniform leaf widths
                    n_leaves = self.machine_config.number_of_leaf_pairs
                    leaf_width = self.treatment_config.field_size[0] / n_leaves
                    for i in range(1, n_leaves):
                        boundary_pos = -self.treatment_config.field_size[0] / 2.0 + i * leaf_width
                        leaf_boundaries_mm.append(boundary_pos)

                fluence_map = apply_tongue_and_groove(
                    fluence_map,
                    leaf_boundaries_mm=leaf_boundaries_mm,
                    field_size_mm=self.treatment_config.field_size[0],
                    tg_reduction=self.machine_config.tongue_groove_reduction,
                    tg_width_mm=self.machine_config.tongue_groove_width_mm,
                    pixel_size_mm=1.0
                ).to(self.dtype)

            # Apply source penumbra (geometric blur from finite source size)
            fluence_map = apply_source_penumbra(
                fluence_map, 
                source_size_mm=self.machine_config.source_size_mm, 
                pixel_size_mm=1.0
            ).to(self.dtype)

            # Apply head scatter (long-range scatter from linac head)
            if self.machine_config.head_scatter_amplitude > 0:
                fluence_map = apply_head_scatter(
                    fluence_map,
                    scatter_amplitude=self.machine_config.head_scatter_amplitude,
                    scatter_range_mm=self.machine_config.head_scatter_range_mm,
                    pixel_size_mm=1.0
                ).to(self.dtype)
            fluence_map = fluence_map[:, 0, :, :]  # [B*G, H, W]

        # Apply collimator rotation (beam limiting device angle)
        # This rotates the fluence map in-plane before projection to 3D
        fluence_map = rotate_2d_images(
            fluence_map,
            self.collimator_angles,
            device=self.device,
            dtype=self.dtype
        )  # [B*G, H, W]
        
        return fluence_map
