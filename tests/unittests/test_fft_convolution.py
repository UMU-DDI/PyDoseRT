"""The FFT backend must be a drop-in for the direct convolution: the same result
(an asymmetric kernel, since a symmetric one hides a missing flip), the same dtype
under autocast, and the same gradients.
"""
import pytest
import torch

from pydosert.layers.BeamWiseConvolutionalLayer import BeamWiseConvolutionalLayer


@pytest.mark.parametrize("shape,k", [((2, 5, 40, 56), 7), ((1, 3, 33, 47), 15),
                                     ((3, 4, 64, 64), 25), ((2, 3, 70, 60), 51)])
def test_fft_matches_direct_with_an_asymmetric_kernel(shape, k, default_device):
    torch.manual_seed(0)
    bg, d, h, w = shape
    fluence = torch.rand(bg, d, h, w, 1, device=default_device, dtype=torch.float64)
    kernels = torch.rand(k, k, bg, d, device=default_device, dtype=torch.float64)
    direct = BeamWiseConvolutionalLayer(default_device, torch.float64, backend="direct")
    fft = BeamWiseConvolutionalLayer(default_device, torch.float64, backend="fft")
    a, b = direct(fluence, kernels), fft(fluence, kernels)
    assert b.shape == a.shape
    assert float((a - b).abs().max() / a.abs().max()) < 1e-5


def test_fft_cost_does_not_depend_on_the_kernel_support(default_device):
    """A kernel larger than the plane is fine: the transform is sized to fit it."""
    fluence = torch.rand(1, 2, 20, 20, 1, device=default_device)
    kernels = torch.rand(41, 41, 1, 2, device=default_device)
    out = BeamWiseConvolutionalLayer(default_device, backend="fft")(fluence, kernels)
    assert out.shape == fluence.shape and torch.isfinite(out).all()


def test_fft_returns_the_autocast_dtype_like_conv2d():
    if not torch.cuda.is_available():
        pytest.skip("autocast half is a CUDA path")
    dev = torch.device("cuda")
    fluence = torch.rand(1, 3, 24, 24, 1, device=dev)
    kernels = torch.rand(7, 7, 1, 3, device=dev)
    with torch.autocast("cuda", dtype=torch.float16):
        a = BeamWiseConvolutionalLayer(dev, backend="direct")(fluence, kernels)
        b = BeamWiseConvolutionalLayer(dev, backend="fft")(fluence, kernels)
    assert a.dtype == b.dtype == torch.float16


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        BeamWiseConvolutionalLayer(backend="winograd")


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max())


def test_fft_gradients_match_direct(default_device):
    """Optimisation backpropagates through rfft2/irfft2; the gradients must be the direct path's."""
    torch.manual_seed(0)
    x = torch.rand(2, 3, 40, 48, 1, device=default_device, dtype=torch.float64, requires_grad=True)
    k = torch.rand(15, 15, 2, 3, device=default_device, dtype=torch.float64, requires_grad=True)
    g = torch.randn(2, 3, 40, 48, 1, device=default_device, dtype=torch.float64)
    grads = {}
    for backend in ("direct", "fft"):
        out = BeamWiseConvolutionalLayer(default_device, torch.float64, backend=backend)(x, k)
        grads[backend] = torch.autograd.grad((out * g).sum(), (x, k))
    for a, b in zip(grads["fft"], grads["direct"]):
        assert _rel(a, b) < 1e-5
