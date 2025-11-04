#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PencilBeamKernelLayer module for generating pencil beam dose kernels based on radiological depth.

This module provides the PencilBeamKernelLayer class, which uses a pencil beam model to compute 
dose kernels for each voxel in the CT volume, based on the radiological depth.
Typical usage example::

    from ..MachineConfig import MachineConfig
    import torch
    config = MachineConfig(...)
    layer = PencilBeamKernelLayer(config)
    radiological_depth = torch.tensor(...)
    kernels = layer(radiological_depth)

Classes:
    PencilBeamKernelLayer: Torch layer for generating pencil beam dose kernels from radiological depth.
"""
import numpy as np
import torch
import torch.nn as nn

from pydose_rt.physics.kernels.pencil_beam_model import PencilBeamModel
from pydose_rt.data.machine_config import MachineConfig

        

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
    def __init__(self, config: MachineConfig, kernel_size: int = 25, verbose: bool = False):
        """
        Initializes the PencilBeamKernelLayer and creates the pencil beam model.

        Args:
            config (MachineConfig): Configuration object with CT and beam parameters.
            kernel_size (int, optional): Size of the dose kernel. Defaults to 25.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.
        """
        super().__init__()
        self.config = config
        self.kernel_size = kernel_size
        self.verbose = verbose
        self.device = self.config.device

        self.pbm = PencilBeamModel(self.config, kernel_size)

    def forward(self, radiological_depth: torch.Tensor) -> np.ndarray:
        """
        Generates pencil beam dose kernels for each voxel based on radiological depth.

        Args:
            radiological_depth (torch.Tensor): Tensor of shape [B*G, P, 1] representing radiological depth for each voxel.

        Returns:
            np.ndarray: Dose kernels of shape [kH, kW, B*G, D].
        """
        with torch.no_grad():
            radiological_depth_numpy = radiological_depth.detach().cpu().numpy()
            kernels = self.pbm.get_nested_kernels(radiological_depth_numpy)
            kernels = np.transpose(kernels, (2, 3, 0, 1))

        return kernels