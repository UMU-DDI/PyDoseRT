"""
BeamRotationLayer module for performing beam-wise 2D rotation of dose volumes using grid sampling.

This module provides the BeamRotationLayer class, which rotates accumulated dose volumes for each gantry angle
using PyTorch's grid sampling. The layer is designed to handle 5D tensors representing dose distributions across batches,
gantry angles, depth, height, and width.

Typical usage example::
    layer = BeamRotationLayer(machine_config, device, dtype, gantry_angles)
    rotated_dose = layer(accumulated_dose)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pydosert.data import MachineConfig
from pydosert.geometry.rotations import build_rotation_grids

class BeamRotationLayer(nn.Module):
    """
    PyTorch module for performing beam-wise 2D rotation of dose volumes using grid sampling.

    Attributes:
        machine_config (MachineConfig): Stores configuration parameters.
        device (torch.device): Device on which computations are performed.
        dtype (torch.dtype): Data type for tensors.
        ct_array_shape (tuple[float, float, float]): Shape of the CT array in voxels (H, D, W).
        iso_center (tuple[float, float, float]): Isocenter in physical coordinates (mm).
        resolution (tuple[float, float, float]): Voxel spacing in mm.
        verbose (bool): Verbosity flag.
        rot_angles_rad (torch.Tensor): Gantry angles in radians, shape [G].
        rot_grid (torch.Tensor): Precomputed rotation sampling grid of shape [1, G, 1, D, W, 2].
    """
    def __init__(self,
                 machine_config: MachineConfig,
                 ct_array_shape: tuple[float, float, float],
                 iso_center: tuple[float, float, float],                
                 resolution: tuple[float, float, float],
                 gantry_angles: list[float] | torch.Tensor = None,
                 device: torch.device | str | None = None,
                 dtype: torch.dtype = torch.float32,
                 verbose: bool = False,
                ) -> 'BeamRotationLayer':
        """
        Initializes the BeamRotationLayer and precomputes the per-angle rotation grids.

        Args:
            machine_config (MachineConfig): Configuration parameters for the layer.
            ct_array_shape (tuple[float, float, float]): Shape of CT array in voxels as (H, D, W).
            iso_center (tuple[float, float, float]): Isocenter in physical coordinates (mm), ordered (X, Y, Z).
            resolution (tuple[float, float, float]): Voxel spacing in mm, ordered (rx, ry, rz).
            gantry_angles (list[float] | torch.Tensor): G gantry angles in radians, shape [G].
            device (torch.device | str | None, optional): Device for computation. Defaults to CUDA if available, else CPU.
            dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
            verbose (bool, optional): If True, enables verbose output for debugging. Defaults to False.
        """
        super().__init__()
        # Handle device default
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.dtype = dtype
        self.machine_config = machine_config
        self.ct_array_shape = ct_array_shape        
        self.iso_center = iso_center
        self.resolution = resolution
        self.verbose = verbose

        self.rot_angles_rad = gantry_angles.to(dtype=self.dtype, device=self.device)        
        self.rot_grid = build_rotation_grids(
            (1, self.rot_angles_rad.shape[0], self.ct_array_shape[1], self.ct_array_shape[0], self.ct_array_shape[2]),
            self.rot_angles_rad,
            self.device,
            self.dtype,
            iso_center=iso_center,
            resolution=resolution
        )

    def forward(self, accumulated_dose: torch.Tensor) -> torch.Tensor:
        """
        Rotates the accumulated dose for all gantry angles in parallel (fully vectorized).

        Each (depth, width) slice is rotated by its beam's gantry angle using the
        precomputed rotation grids.

        Args:
            accumulated_dose (torch.Tensor): Dose volume of shape [B, G, D, H, W].

        Returns:
            torch.Tensor: Rotated dose volume of shape [B, G, H, D, W].
        """

        B, G, D, H, W = accumulated_dose.shape
        accumulated_dose = accumulated_dose.permute(0, 1, 3, 2, 4)   # [B, G, H, D, W]
        accumulated_dose = accumulated_dose.reshape(B*G*H, 1, D, W)   # [B*G*H, 1, D, W]
        rot_grid = self.rot_grid
        rot_grid = rot_grid.repeat(B, 1, H, 1, 1, 1)               # [B, G, H, D, W, 2]
        rot_grid = rot_grid.reshape(B*G*H, D, W, 2)                # [B*G*H, D, W, 2]
        
        
        # Rotate
        accumulated_dose = F.grid_sample(accumulated_dose, rot_grid,
                                    mode="bilinear",
                                    padding_mode="zeros",
                                    align_corners=False)    # [B*G*H, 1, D, W]

        # Reshape back
        accumulated_dose = accumulated_dose.reshape(B, G, H, D, W)

        return accumulated_dose