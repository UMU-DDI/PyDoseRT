"""
BeamValidationLayer module for validating and scaling leaf positions and monitor units (MUs).

This module provides the BeamValidationLayer class, which clamps and scales leaf positions and MUs
according to configuration parameters, ensuring that the values are within valid ranges for dose calculation
and beam delivery in radiotherapy planning models.

Typical usage example::

    from pydose_rt.data import MachineConfig
    import torch
    machine_config = MachineConfig(...)
    layer = BeamValidationLayer(machine_config, device, dtype, field_size)
    leaf_positions = torch.tensor(...)
    jaw_positions = torch.tensor(...)
    mus = torch.tensor(...)
    valid_leaf_positions, valid_jaw_positions, valid_mus = layer(leaf_positions, jaw_positions, mus)

Classes:
    BeamValidationLayer: Torch layer for validating and scaling leaf positions and monitor units.
"""

import torch
import torch.nn as nn
from typing import Tuple
from pydose_rt.data import MachineConfig

def adjust_mask(pos_a, pos_b, min_overlap, field_size):
    centers = (pos_a + pos_b) / 2
    widths  = (pos_b - pos_a)
    widths = torch.clamp(widths, min=min_overlap)
    centers = torch.clamp(centers, min=-field_size, max=field_size)
    adjusted_positions = torch.stack([centers - (widths / 2), centers + (widths / 2)], dim=-1)
    adjusted_positions = torch.clamp(adjusted_positions, min=-field_size, max=field_size)

    return adjusted_positions

class BeamValidationLayer(nn.Module):
    """
    BeamValidationLayer for validating and scaling leaf positions, monitor units (MUs) and jaw positions.

    This layer clamps and scales leaf positions and MUs according to configuration parameters,
    ensuring that the values are within valid ranges for dose calculation and beam delivery.

    Attributes:
        machine_config (MachineConfig): Configuration object containing machine parameters.
        device (torch.device): Device on which computations are performed (CPU or CUDA).
        dtype (type): Data type for tensors.
        field_size (tuple[float, float]): Field size (width, height).
        verbose (bool): Flag to enable verbose logging.

    Methods:
        __init__(config, slope=None, verbose=False): Initializes the BeamValidationLayer with configuration and verbosity.
        forward(leaf_positions, mus): Clamps and scales leaf positions and MUs, returning validated tensors.
    """
    def __init__(self, machine_config: MachineConfig, 
                 field_size: tuple[int, int] = (400, 400), 
                 device: torch.device | str | None = None,
                 dtype: torch.dtype = torch.float32,
                 verbose: bool = False) -> 'BeamValidationLayer':
        """
        Initializes the BeamValidationLayer.

        Args:
            machine_config (MachineConfig): Configuration object with machine parameters.
            field_size (tuple[int, int]): Field size (width, height).
            device (torch.device): Device for computation (CPU or CUDA).
            dtype (type): Data type for tensors.
        """
        super().__init__()

        # Handle device default
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device=device
        self.dtype=dtype
        self.machine_config = machine_config
        self.verbose = verbose
        self.min_leaf_opening = machine_config.minimum_leaf_opening
        self.min_jaw_opening = machine_config.minimum_jaw_opening
        self.half_field_width = field_size[1] / 2.0

    
    def forward(self, leaf_positions: torch.Tensor, mus: torch.Tensor, jaw_positions: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Clamps and scales leaf positions and monitor units (MUs).

        Args:
            leaf_positions (torch.Tensor): Tensor of leaf positions to be validated and scaled.
            mus (torch.Tensor): Tensor of monitor units to be validated and scaled.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Validated and scaled leaf positions and MUs.
        """

        mus = torch.clamp(mus, min=0.1)

        mlc_positions = adjust_mask(leaf_positions[..., 0], leaf_positions[..., 1], self.min_leaf_opening, self.half_field_width)

        if jaw_positions is not None:
            jaw_positions = adjust_mask(jaw_positions[..., 0], jaw_positions[..., 1], self.min_jaw_opening, self.half_field_width)

        return mlc_positions, jaw_positions, mus