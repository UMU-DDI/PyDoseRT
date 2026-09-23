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

    def depth_binned(self, fluence_vol: torch.Tensor, depth: torch.Tensor,
                     nodes_mm: torch.Tensor, node_kernels: torch.Tensor) -> torch.Tensor:
        """Convolve with a kernel chosen per PENCIL rather than per plane.

        Each pencil's fluence is split between the two depth nodes bracketing its
        own radiological depth, with linear weights that sum to one, and each node
        is convolved with that node's kernel::

            dose = sum_m  ( fluence * lambda_m(depth) )  (*)  K(d_m)

        which equals convolving every pencil with the kernel linearly interpolated
        to its own depth. The node kernels are shared by every plane and beam, so
        their spectra are computed once; per group of planes only the nodes the
        group's depths actually reach are transformed, and the spectra accumulate
        before a single inverse transform.

        Args:
            fluence_vol (torch.Tensor): [B*G, D, H, W, 1] BEV fluence.
            depth (torch.Tensor): [B*G, D, H, W] per-pencil radiological depth, in
                the same density x mm units the kernel model takes.
            nodes_mm (torch.Tensor): [M] increasing node depths, density x mm.
            node_kernels (torch.Tensor): [M, kH, kW] kernel at each node.

        Returns:
            torch.Tensor: [B*G, D, H, W, 1], in the dtype F.conv2d would return.
        """
        BG, D, H, W, _ = fluence_vol.shape
        M, kH, kW = node_kernels.shape
        dev_type = fluence_vol.device.type
        out_dtype = (torch.get_autocast_dtype(dev_type) if torch.is_autocast_enabled(dev_type)
                     else fluence_vol.dtype)
        fh = self._fft_size(H + kH - 1)
        fw = self._fft_size(W + kW - 1)
        ch, cw = (kH - 1) // 2, (kW - 1) // 2
        x = fluence_vol.reshape(BG * D, H, W)
        dep = depth.reshape(BG * D, H, W)
        out = torch.empty((BG * D, H, W), device=x.device, dtype=out_dtype)
        group = max(1, (H * W * 8) // max(1, fh * (fw // 2 + 1)))
        group = max(1, min(BG * D, group * 4))
        with torch.autocast(device_type=dev_type, enabled=False):
            nodes = nodes_mm.to(device=x.device, dtype=torch.float32)
            khat = torch.fft.rfft2(torch.flip(node_kernels.float(), dims=(-2, -1)), s=(fh, fw))
            for s in range(0, BG * D, group):
                e = min(s + group, BG * D)
                xs = x[s:e].float()
                ds = dep[s:e].float().clamp(float(nodes[0]), float(nodes[-1]))
                lo = (torch.searchsorted(nodes, ds.contiguous(), right=True) - 1).clamp(0, M - 2)
                t = (ds - nodes[lo]) / (nodes[lo + 1] - nodes[lo])
                # Which planes reach which nodes. A group spans many planes, shallow
                # to deep, so almost every node is reached by SOME plane -- but each
                # plane reaches only a few (2-4 at depth, ~10-20 at the skin).
                # Transforming only the planes that reach a node, instead of the
                # whole group for every node reached by any of them, is the
                # difference between ~4 and ~44 transforms per plane.
                # Node activity is decided by the pencils that carry fluence. A BEV
                # plane spans the whole patient cross-section, so it meets the skin
                # somewhere and would otherwise reach almost every node -- from
                # pencils with no fluence, which contribute nothing to any of them.
                live = xs != 0
                lo_min = torch.where(live, lo, torch.full_like(lo, M)).flatten(1).amin(1)
                hi_max = torch.where(live, lo, torch.full_like(lo, -1)).flatten(1).amax(1) + 1
                if not bool(live.any()):
                    out[s:e] = 0
                    continue
                acc = torch.zeros((e - s, fh, fw // 2 + 1), device=x.device,
                                  dtype=torch.complex64)
                for m in range(int(lo_min.min()), int(hi_max.max()) + 1):
                    planes = ((lo_min <= m) & (hi_max >= m)).nonzero().flatten()
                    if planes.numel() == 0:
                        continue
                    lo_p, t_p = lo[planes], t[planes]
                    w = torch.where(lo_p == m, 1.0 - t_p, torch.zeros_like(t_p))
                    w = w + torch.where(lo_p + 1 == m, t_p, torch.zeros_like(t_p))
                    acc[planes] += torch.fft.rfft2(xs[planes] * w, s=(fh, fw)) * khat[m]
                    del w, lo_p, t_p
                out[s:e] = torch.fft.irfft2(acc, s=(fh, fw))[:, ch:ch + H, cw:cw + W]
                del acc, xs, ds, lo, t, lo_min, hi_max
        return out.reshape(BG, D, H, W, 1)

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
