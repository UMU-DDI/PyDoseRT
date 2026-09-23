"""
This module provides the BeamWiseConvolutionalLayer class, a PyTorch nn.Module for performing
beam-wise 2D convolution on fluence volumes using custom kernels.

It accepts batched fluence volumes and corresponding kernels for each beam/group, uses
grouped 2D convolution to apply the correct kernel to each fluence volume, handles reshaping and
permutation of tensors to match PyTorch's grouped convolution requirements and returns output
in the same shape as the input fluence volume.

With ``backend="fft"`` the same convolution runs per plane through rfft2/irfft2,
at a cost that does not grow with the kernel size.

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
                 verbose: bool = False,
                 backend: str = "direct") -> 'BeamWiseConvolutionalLayer':
        """
        Initializes the BeamWiseConvolutionalLayer.

        Args:
            device (torch.device | str | None, optional): Device for computation. Defaults to CUDA if available, else CPU.
            dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
            verbose (bool, optional): If True, enables verbose output for debugging. Defaults to False.
            backend (str, optional): ``"direct"`` (grouped conv2d, cost grows with the
                kernel area) or ``"fft"`` (cost independent of the kernel support).
                Both produce the same result. Defaults to ``"direct"``.
        """
        super().__init__()
        if backend not in ("direct", "fft"):
            raise ValueError(f"backend must be 'direct' or 'fft', got {backend!r}")
        self.backend = backend

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
            fluence_vol (torch.Tensor): Input fluence volume of shape [B*G, D, H, W, 1].
            kernels (torch.Tensor): Per-(beam, depth) kernel tensor of shape [kH, kW, B*G, D].

        Returns:
            torch.Tensor: Convolved volume of shape [B*G, D, H, W, 1].
        """

        BG, D, H, W, _ = fluence_vol.shape
        kH, kW = kernels.shape[0], kernels.shape[1]

        if self.backend == "fft":
            return self._forward_fft(fluence_vol, kernels)

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

    @staticmethod
    def _fft_size(n: int) -> int:
        """Smallest size >= n with only factors 2, 3 and 5, which cuFFT handles fast."""
        while True:
            m = n
            for p in (2, 3, 5):
                while m % p == 0:
                    m //= p
            if m == 1:
                return n
            n += 1

    def _forward_fft(self, fluence_vol: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        """The same convolution by FFT, at a cost independent of the kernel support.

        Zero-padding both operands to the linear-convolution size and cropping
        reproduces the direct path exactly, once the kernel is flipped: it is a
        cross-correlation, the FFT a convolution. The transform runs in float32
        whatever the engine dtype, as cuFFT takes half precision only on
        power-of-two sizes.

        Args:
            fluence_vol (torch.Tensor): [B*G, D, H, W, 1].
            kernels (torch.Tensor): [kH, kW, B*G, D].

        Returns:
            torch.Tensor: [B*G, D, H, W, 1], in the dtype F.conv2d would return.
        """
        BG, D, H, W, _ = fluence_vol.shape
        kH, kW = kernels.shape[0], kernels.shape[1]
        # what the direct path would have returned: under autocast F.conv2d yields the
        # autocast dtype, and float32 here would double every tensor downstream
        dev_type = fluence_vol.device.type
        out_dtype = (torch.get_autocast_dtype(dev_type) if torch.is_autocast_enabled(dev_type)
                     else fluence_vol.dtype)
        x = fluence_vol.reshape(BG * D, H, W)
        k = kernels.permute(2, 3, 0, 1).reshape(BG * D, kH, kW)
        fh = self._fft_size(H + kH - 1)
        fw = self._fft_size(W + kW - 1)
        ch, cw = (kH - 1) // 2, (kW - 1) // 2
        out = torch.empty((BG * D, H, W), device=x.device, dtype=out_dtype)
        # groups sized to one direct-path volume: the spectra of every plane at once
        # cost ~3x its peak memory, for no measurable time
        group = max(1, (H * W * 8) // max(1, fh * (fw // 2 + 1)))
        group = max(1, min(BG * D, group * 4))
        with torch.autocast(device_type=x.device.type, enabled=False):
            for s in range(0, BG * D, group):
                e = min(s + group, BG * D)
                spec = torch.fft.rfft2(x[s:e].float(), s=(fh, fw))
                # flip: the direct path is a correlation, the FFT a convolution
                spec *= torch.fft.rfft2(torch.flip(k[s:e].float(), dims=(-2, -1)), s=(fh, fw))
                out[s:e] = torch.fft.irfft2(spec, s=(fh, fw))[:, ch:ch + H, cw:cw + W]
                del spec
        return out.reshape(BG, D, H, W, 1)
