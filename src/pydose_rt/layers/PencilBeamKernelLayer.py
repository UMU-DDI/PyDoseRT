"""
PencilBeamKernelLayer module for generating pencil beam dose kernels based on radiological depth.

This module provides the PencilBeamKernelLayer class, which uses a pencil beam model to compute 
dose kernels for each voxel in the CT volume, based on the radiological depth.
Typical usage example::

    from pydose_rt.data import MachineConfig
    import torch
    machine_config = MachineConfig(...)
    layer = PencilBeamKernelLayer(machine_config, device, dtype, resolution, kernel_size)

Classes:
    PencilBeamKernelLayer: Torch layer for generating pencil beam dose kernels from radiological depth.
"""
import numpy as np
import torch
import torch.nn as nn

from pydose_rt.physics.kernels.pencil_beam_model import PencilBeamModel
from pydose_rt.data import MachineConfig

        

class PencilBeamKernelLayer(nn.Module):
    """
    Torch layer for generating pencil beam dose kernels from radiological depth.

    This layer uses a pencil beam model to compute dose kernels for each voxel 
    in the CT volume, based on the radiological depth. The kernels are used for 
    dose calculation in radiotherapy planning.

    Attributes:
        config (MachineConfig): Configuration object.
        kernel_size (int): Size of the dose kernel.
        verbose (bool): Verbosity flag.
        device (torch.device): Device for computation (CPU or CUDA).
        pbm: PencilBeamModel instance for kernel calculation.
    """
    def __init__(self, 
                 machine_config: MachineConfig, 
                 resolution: tuple[float, float, float],
                 kernel_size: tuple[int, int],
                 device: torch.device | str | None = None,
                 dtype: torch.dtype = torch.float32,
                 verbose: bool = False) -> 'PencilBeamKernelLayer':
        """
        Initializes the PencilBeamKernelLayer and creates the pencil beam model.

        Args:
            machine_config (MachineConfig): Configuration object with CT and beam parameters.
            resolution (tuple[float, float, float]): Voxel spacing in mm.
            kernel_size (tuple[int, int]): Size of the dose kernel (height, width).
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
        self.kernel_size = kernel_size
        self.verbose = verbose
        self.resolution = resolution

        self.pbm = PencilBeamModel(self.resolution, self.machine_config.tpr_20_10, kernel_size)

    def forward(self, radiological_depth: torch.Tensor) -> np.ndarray:
        """
        Generates pencil beam dose kernels for each voxel based on radiological depth.

        Args:
            radiological_depth (torch.Tensor): Tensor of shape [B*G, P, 1] representing radiological depth for each voxel.

        Returns:
            torch.Tensor: Dose kernels of shape [kH, kW, B*G, D].
        """

        kernels = self.pbm.get_nested_kernels(radiological_depth).to(radiological_depth.device).to(radiological_depth.dtype)
        kernels = torch.permute(kernels, (2, 3, 0, 1))

        return kernels