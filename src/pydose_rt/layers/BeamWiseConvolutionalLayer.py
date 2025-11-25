"""
This module provides the BeamWiseConvolutionalLayer class, a PyTorch nn.Module for performing
beam-wise 2D convolution on fluence volumes using custom kernels.

It accepts batched fluence volumes and corresponding kernels for each beam/group, uses
grouped 2D convolution to apply the correct kernel to each fluence volume, handles reshaping and
permutation of tensors to match PyTorch's grouped convolution requirements and returns output
in the same shape as the input fluence volume.

Typical Usage:
    layer = BeamWiseConvolutionalLayer(device, dtype)
    output = layer(fluence_vol, kernels)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BeamWiseConvolutionalLayer(nn.Module):
    """
    PyTorch module for performing beam-wise 2D convolution on fluence maps using custom kernels,
    where each control point has its own fluence map and kernel.

    Attributes:
        device (torch.device): Device on which computations are performed.        
        dtype (type): Data type for tensors.
        verbose (bool): Verbosity flag.
    """

    def __init__(self, 
                 device: torch.device | str | None = None,
                 dtype: torch.dtype = torch.float32,
                 verbose: bool = False) -> 'BeamWiseConvolutionalLayer':
        """
        Initializes the BeamWiseConvolutionalLayer.

        Args:
            device (torch.device): Device for computation (CPU or CUDA).
            dtype (type): Data type for tensors.
            verbose (bool, optional): If True, enables verbose output for debugging. Defaults to False.
        """
        super().__init__()

        # Handle device default
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device=device
        self.dtype=dtype
        self.verbose = verbose

    def forward(self, fluence_vol: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        """
        Performs grouped 2D convolution on batched fluence volumes using provided kernels for each beam/group.

        Args:
            fluence_vol (torch.Tensor): Input tensor of shape [B*G, D, W, H, 1]
            kernels (torch.Tensor): Kernel tensor of shape [kH, kW, B*G, D]

        Returns:
            torch.Tensor: Output tensor of shape [B*G, D, H, W, 1], representing the convolved volumes.
        """

        BG, D, H, W, _ = fluence_vol.shape
        kH, kW = kernels.shape[0], kernels.shape[1]

        # [BG, D, 1, H, W] → [1, BG*D, H, W] (combine BG and D into batch)
        fluence_vol = fluence_vol.reshape(1, BG * D, H, W)

        # [kH, kW, BG, D] → [BG*D, 1, kH, kW]
        kernels = kernels.permute(2, 3, 0, 1).reshape(BG * D, 1, kH, kW)

        # Now group conv: BG*D inputs, BG*D kernels, 1 channel per group
        out = F.conv2d(
            fluence_vol, weight=kernels, groups=BG * D, padding="same"
        )  # [BG*D, 1, H, W]

        # Reshape back: [BG, D, H, W, 1]
        out = out.view(BG, D, H, W, 1)

        return out
