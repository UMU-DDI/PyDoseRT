#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ValidParametersLayer module for validating and scaling leaf positions and monitor units (MUs).

This module provides the ValidParametersLayer class, which clamps and scales leaf positions and MUs
according to configuration parameters, ensuring that the values are within valid ranges for dose calculation
and beam delivery in radiotherapy planning models.

Typical usage example::

    from ..MachineConfig import MachineConfig
    import torch
    config = MachineConfig(...)
    layer = ValidParametersLayer(config)
    leaf_positions = torch.tensor(...)
    mus = torch.tensor(...)
    valid_leaf_positions, valid_mus = layer(leaf_positions, mus)

Classes:
    ValidParametersLayer: Torch layer for validating and scaling leaf positions and monitor units.
"""

import torch
import torch.nn as nn
from typing import Tuple

from pydose_rt.data.machine_config import MachineConfig


class ValidParametersLayer(nn.Module):
    """
    ValidParametersLayer for validating and scaling leaf positions, monitor units (MUs) and jaw positions.

    This layer clamps and scales leaf positions and MUs according to configuration parameters,
    ensuring that the values are within valid ranges for dose calculation and beam delivery.

    Attributes:
        config: Configuration object containing scaling and field size parameters.
        verbose (bool): Flag to enable verbose logging.
        device (torch.device): Device on which computations are performed (CPU or CUDA).

    Methods:
        __init__(config, slope=None, verbose=False): Initializes the ValidParametersLayer with configuration and verbosity.
        forward(leaf_positions, mus): Clamps and scales leaf positions and MUs, returning validated tensors.
    """
    def __init__(self, config: MachineConfig, leafs_centered: bool = False, verbose: bool = False):
        """
        Initializes the ValidParametersLayer.

        Args:
            config (MachineConfig): Configuration object with mu_scaling, minimum_leaf_overlap, and field_size attributes.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.
        """
        super().__init__()
        self.config = config
        self.verbose = verbose
        self.device = self.config.device
        self.leafs_centered = leafs_centered
        self.min_leaf_opening = (
            config.minimum_leaf_overlap / config.resolution[1]
        ) / config.field_size_in_pixels[1]
        self.min_jaw_opening = (
            config.minimum_jaw_overlap / config.resolution[0]
        ) / config.field_size_in_pixels[0]

    @staticmethod
    def _proj_ste(x, lo=None, hi=None):
        """
        Straight-through projection to [lo, hi]:
        forward uses clamped value; backward passes gradient as if identity.
        """
        x_proj = torch.clamp(x, min=lo, max=hi)
        return x + (x_proj - x).detach()
    
    def forward(self, leaf_positions: torch.Tensor, mus: torch.Tensor, jaw_positions: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Clamps and scales leaf positions and monitor units (MUs).

        Args:
            leaf_positions (torch.Tensor): Tensor of leaf positions to be validated and scaled.
            mus (torch.Tensor): Tensor of monitor units to be validated and scaled.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Validated and scaled leaf positions and MUs.
        """

        if self.leafs_centered:
            left_positions = leaf_positions[:, 0, :, :] - (
                leaf_positions[:, 1, :, :] / 2
            )
            right_positions = leaf_positions[:, 0, :, :] + (
                leaf_positions[:, 1, :, :] / 2
            )
            leaf_positions = torch.stack([left_positions, right_positions], dim=1)

            if jaw_positions is not None:
                bottom_positions = jaw_positions[:, 0, :] - (jaw_positions[:, 1, :] / 2)
                top_positions = jaw_positions[:, 0, :] + (jaw_positions[:, 1, :] / 2)
                jaw_positions = torch.stack([bottom_positions, top_positions], dim=1)

        # 1) MU: keep non-negative & scaled
        mus = self._proj_ste(mus, lo=0.1)

        # 2) Leafs: Keep widths open
        mlc_centers = (leaf_positions[:, 0, :, :] + leaf_positions[:, 1, :, :]) / 2
        mlc_widths  = (leaf_positions[:, 1, :, :] - leaf_positions[:, 0, :, :])
        min_w = self.min_leaf_opening
        mlc_centers = self._proj_ste(mlc_centers, 0.0, 1.0)
        mlc_widths = self._proj_ste(mlc_widths, lo=min_w)
        mlc_positions = torch.stack([mlc_centers - (mlc_widths / 2), mlc_centers + (mlc_widths / 2)], dim=1)
        mlc_positions = self._proj_ste(mlc_positions, 0.0, 1.0)

        # 3) Jaws: Keep widths open
        if jaw_positions is not None:
            jaw_centers = (jaw_positions[:, 0, :] + jaw_positions[:, 1, :]) / 2
            jaw_widths  = (jaw_positions[:, 1, :] - jaw_positions[:, 0, :])
            min_jaw_w = self.min_jaw_opening
            jaw_widths = self._proj_ste(jaw_widths, lo=min_jaw_w)
            jaw_centers = self._proj_ste(jaw_centers, 0.0, 1.0)
            jaw_positions = torch.stack([jaw_centers - (jaw_widths / 2), (jaw_centers + jaw_widths / 2)], dim=1)
            jaw_positions = self._proj_ste(jaw_positions, 0.0, 1.0)

        return mlc_positions, mus, jaw_positions