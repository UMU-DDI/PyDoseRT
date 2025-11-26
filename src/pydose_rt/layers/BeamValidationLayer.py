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

class MaximumLeafTipProjector(nn.Module):
    def __init__(self, value=1.0, k=2.0, center=0.5):
        """
        Projects input ∈ [0,1] to output ∈ [center-value, center+value]
        
        Works with arbitrary batch dimensions.
        
        Args:
            value: Half-range of the output
            k: Steepness of tanh
            center: Center point of the projection
        """
        super().__init__()
        self.value = value
        self.k = k
        self.center = center
    
    def forward(self, x):
        """
        Args:
            x: Tensor of any shape [..., features]
        
        Returns:
            Projected tensor of same shape as x
        """
        return self.value * torch.tanh(self.k * (x - self.center)) + self.center
    
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
                 adjust_values: bool = True, 
                 verbose: bool = False) -> 'BeamValidationLayer':
        """
        Initializes the BeamValidationLayer.

        Args:
            machine_config (MachineConfig): Configuration object with machine parameters.
            field_size (tuple[int, int]): Field size (width, height).
            device (torch.device): Device for computation (CPU or CUDA).
            dtype (type): Data type for tensors.
            adjust_values (bool, optional): Whether to adjust parameter values. Defaults to True.
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
        self.adjust_values = adjust_values
        self.min_leaf_opening = machine_config.minimum_leaf_opening
        self.min_jaw_opening = machine_config.minimum_jaw_opening
        self.half_field_width = field_size[1] / 2.0

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

        left_positions = leaf_positions[..., 0]
        right_positions = leaf_positions[..., 1]
        if self.adjust_values:
            # 1) MU: keep non-negative & scaled
            mus = self._proj_ste(mus, lo=0.1)

            # 3) Leafs: Keep widths open
            mlc_centers = (left_positions + right_positions) / 2
            mlc_widths  = (right_positions - left_positions)
            min_w = self.min_leaf_opening
            mlc_widths = self._proj_ste(mlc_widths, lo=min_w)
            mlc_centers = self._proj_ste(mlc_centers, -self.half_field_width, self.half_field_width)
            mlc_positions = torch.stack([mlc_centers - (mlc_widths / 2), mlc_centers + (mlc_widths / 2)], dim=-1)
            mlc_positions = self._proj_ste(mlc_positions, -self.half_field_width, self.half_field_width)

            # 4) Jaws: Keep widths open
            if jaw_positions is not None:
                jaw_centers = (jaw_positions[..., 0] + jaw_positions[..., 1]) / 2
                jaw_widths  = (jaw_positions[..., 1] - jaw_positions[..., 0])
                min_jaw_w = self.min_jaw_opening
                jaw_widths = self._proj_ste(jaw_widths, lo=min_jaw_w)
                jaw_centers = self._proj_ste(jaw_centers, -self.half_field_width, self.half_field_width)
                jaw_positions = torch.stack([jaw_centers - (jaw_widths / 2), (jaw_centers + jaw_widths / 2)], dim=-1)
                jaw_positions = self._proj_ste(jaw_positions, -self.half_field_width, self.half_field_width)
        else:
            mlc_positions = torch.stack([left_positions, right_positions], dim=-1)
            if jaw_positions is not None:
                jaw_positions = jaw_positions

        return mlc_positions, jaw_positions, mus