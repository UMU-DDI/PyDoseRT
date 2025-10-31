#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PencilBeamKernelLayer module for generating pencil beam dose kernels based on radiological depth.

This module provides the PencilBeamKernelLayer class, which uses a pencil beam model to compute 
dose kernels for each voxel in the CT volume, based on the radiological depth.
Typical usage example::

    from ..ModelConfig import ModelConfig
    import torch
    config = ModelConfig(...)
    layer = PencilBeamKernelLayer(config)
    radiological_depth = torch.tensor(...)
    kernels = layer(radiological_depth)

Classes:
    PencilBeamKernelLayer: Torch layer for generating pencil beam dose kernels from radiological depth.
"""
import numpy as np
import torch
import torch.nn as nn

from pydose_rt.utils.kernel import PencilBeamModel
from pydose_rt.ModelConfig import ModelConfig

        

class PencilBeamKernelLayer(nn.Module):
    """
    Torch layer for generating pencil beam dose kernels from radiological depth.

    This layer uses a pencil beam model to compute dose kernels for each voxel 
    in the CT volume, based on the radiological depth. The kernels are used for 
    dose calculation in radiotherapy planning.

    Attributes:
        config (ModelConfig): Configuration object.
        kernel_size (int): Size of the dose kernel.
        verbose (bool): Verbosity flag.
        device (torch.device): Device for computation (CPU or CUDA).
        pbm: PencilBeamModel instance for kernel calculation.
    """
    def __init__(self, config: ModelConfig, kernel_size: int = 25, verbose: bool = False):
        """
        Initializes the PencilBeamKernelLayer and creates the pencil beam model.

        Args:
            config (ModelConfig): Configuration object with CT and beam parameters.
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

            """ # Print and visualize kernel for D=64 and BG=30 if in bounds
            print("Kernels shape:", kernels.shape)
            kH, kW, BG, D = kernels.shape
            if BG > 30 and D > 64:
                kernel_to_show = kernels[:, :, 30, 64]
                print(f"Kernel shape for BG=30, D=64: {kernel_to_show.shape}")
                print(f"Sum of kernel (BG=30, D=64): {kernel_to_show.sum()}")
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(6, 5))
                    plt.imshow(kernel_to_show, cmap='jet', origin='lower')
                    plt.title('Pencil Beam Kernel (BG=30, D=64)')
                    plt.colorbar()
                    plt.tight_layout()
                    plt.show()
                except ImportError:
                    print("matplotlib not available for visualization.")
            else:
                print(f"Cannot show kernel: BG={BG}, D={D} (need BG>30, D>64)") """

        return kernels