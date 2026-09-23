"""depth_binned convolves each pencil with the kernel interpolated to its own
radiological depth. The reference is the same sum built from direct convolutions.
"""

import torch
import torch.nn.functional as F

from pydosert.layers.BeamWiseConvolutionalLayer import BeamWiseConvolutionalLayer


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max())


def test_depth_binned_matches_per_pencil_interpolation_with_gradients(default_device):
    """Each pencil convolved with the kernel interpolated to its own depth, built from direct
    convolutions -- the forward result and both gradients."""
    torch.manual_seed(0)
    BG, D, H, W, k = 2, 3, 32, 36, 11
    nodes = torch.tensor([0.0, 5.0, 10.0, 20.0, 40.0], device=default_device, dtype=torch.float64)
    x = torch.rand(BG, D, H, W, 1, device=default_device, dtype=torch.float64, requires_grad=True)
    node_kernels = torch.rand(len(nodes), k, k, device=default_device, dtype=torch.float64,
                              requires_grad=True)
    depth = torch.rand(BG, D, H, W, device=default_device, dtype=torch.float64) * 40
    g = torch.randn(BG, D, H, W, 1, device=default_device, dtype=torch.float64)

    out = BeamWiseConvolutionalLayer(default_device, torch.float64, backend="fft").depth_binned(
        x, depth, nodes, node_kernels)

    lo = (torch.searchsorted(nodes, depth.contiguous(), right=True) - 1).clamp(0, len(nodes) - 2)
    t = (depth - nodes[lo]) / (nodes[lo + 1] - nodes[lo])
    planes = x[..., 0].reshape(1, BG * D, H, W)
    ref = 0
    for m in range(len(nodes)):
        w = torch.where(lo == m, 1 - t, 0.0) + torch.where(lo + 1 == m, t, 0.0)
        kern = node_kernels[m].expand(BG * D, 1, k, k)
        ref = ref + F.conv2d(planes * w.reshape(1, BG * D, H, W), kern, groups=BG * D,
                             padding="same")
    ref = ref.reshape(BG, D, H, W, 1)

    assert _rel(out, ref) < 1e-5
    got = torch.autograd.grad((out * g).sum(), (x, node_kernels))
    want = torch.autograd.grad((ref * g).sum(), (x, node_kernels))
    for a, b in zip(got, want):
        assert _rel(a, b) < 1e-5
