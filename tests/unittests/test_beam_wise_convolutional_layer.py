import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import torch
from pydose_rt.layers.BeamWiseConvolutionalLayer import BeamWiseConvolutionalLayer


@pytest.fixture
def beam_wise_conv_layer(default_device, default_dtype):
    """Fixture to create a BeamWiseConvolutionalLayer instance"""
    return BeamWiseConvolutionalLayer(
        device=default_device,
        dtype=default_dtype
    )


def test_beam_wise_conv_layer_output_shape(beam_wise_conv_layer, default_device, default_dtype):
    """Test that beam-wise convolutional layer produces correct output shape"""
    BG = 2  # Number of beams
    D = 64  # Depth dimension
    H = 32  # Height
    W = 32  # Width
    kernel_size = 7

    # Create input fluence volume [BG, D, H, W, 1]
    fluence_vol = torch.randn(BG, D, H, W, 1, device=default_device, dtype=default_dtype)

    # Create kernels [kH, kW, BG, D]
    kernels = torch.randn(kernel_size, kernel_size, BG, D, device=default_device, dtype=default_dtype)

    output = beam_wise_conv_layer(fluence_vol, kernels)

    expected_shape = (BG, D, H, W, 1)
    assert output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, but got {output.shape}"


def test_beam_wise_conv_layer_single_beam(beam_wise_conv_layer, default_device, default_dtype):
    """Test beam-wise convolutional layer with single beam"""
    BG = 1
    D = 32
    H = 64
    W = 64
    kernel_size = 5

    fluence_vol = torch.randn(BG, D, H, W, 1, device=default_device, dtype=default_dtype)
    kernels = torch.randn(kernel_size, kernel_size, BG, D, device=default_device, dtype=default_dtype)

    output = beam_wise_conv_layer(fluence_vol, kernels)

    assert output.shape == (BG, D, H, W, 1), \
        f"Expected shape {(BG, D, H, W, 1)}, got {output.shape}"


def test_beam_wise_conv_layer_multiple_beams(beam_wise_conv_layer, default_device, default_dtype):
    """Test beam-wise convolutional layer with multiple beams"""
    BG = 4
    D = 16
    H = 32
    W = 32
    kernel_size = 9

    fluence_vol = torch.randn(BG, D, H, W, 1, device=default_device, dtype=default_dtype)
    kernels = torch.randn(kernel_size, kernel_size, BG, D, device=default_device, dtype=default_dtype)

    output = beam_wise_conv_layer(fluence_vol, kernels)

    assert output.shape == (BG, D, H, W, 1), \
        f"Expected shape {(BG, D, H, W, 1)}, got {output.shape}"


def test_beam_wise_conv_layer_identity_kernel(beam_wise_conv_layer, default_device, default_dtype):
    """Test that identity kernel produces similar output"""

    BG = 1
    D = 8
    H = 16
    W = 16
    kernel_size = 3

    # Create a simple input with a single peak
    fluence_vol = torch.zeros(BG, D, H, W, 1, device=default_device, dtype=default_dtype)
    fluence_vol[0, D//2, H//2, W//2, 0] = 1.0

    # Create identity-like kernels (center is 1, rest is 0)
    kernels = torch.zeros(kernel_size, kernel_size, BG, D, device=default_device, dtype=default_dtype)
    kernels[kernel_size//2, kernel_size//2, :, :] = 1.0

    output = beam_wise_conv_layer(fluence_vol, kernels)

    # Output should have similar peak at the same location
    assert output[0, D//2, H//2, W//2, 0] > 0.9, \
        "Identity kernel should preserve central peak"


def test_beam_wise_conv_layer_gradients(beam_wise_conv_layer, default_device, default_dtype):
    """Test that gradients flow through beam-wise convolutional layer"""
    BG = 2
    D = 16
    H = 32
    W = 32
    kernel_size = 5

    fluence_vol = torch.randn(BG, D, H, W, 1, device=default_device, dtype=default_dtype, requires_grad=True)
    kernels = torch.randn(kernel_size, kernel_size, BG, D, device=default_device, dtype=default_dtype, requires_grad=True)

    output = beam_wise_conv_layer(fluence_vol, kernels)
    loss = output.sum()
    loss.backward()

    assert fluence_vol.grad is not None, "Gradients should flow to fluence volume"
    assert kernels.grad is not None, "Gradients should flow to kernels"
    assert torch.any(fluence_vol.grad != 0), "Fluence volume gradients should be non-zero"
    assert torch.any(kernels.grad != 0), "Kernel gradients should be non-zero"


def test_beam_wise_conv_layer_preserves_dtype(default_device):
    """Test that beam-wise convolutional layer preserves tensor dtype"""
    for dtype in [torch.float32, torch.float64]:
        layer = BeamWiseConvolutionalLayer(device=default_device, dtype=dtype)

        BG = 2
        D = 8
        H = 16
        W = 16
        kernel_size = 3

        fluence_vol = torch.randn(BG, D, H, W, 1, device=default_device, dtype=dtype)
        kernels = torch.randn(kernel_size, kernel_size, BG, D, device=default_device, dtype=dtype)

        output = layer(fluence_vol, kernels)

        assert output.dtype == dtype, f"Expected dtype {dtype}, but got {output.dtype}"


def test_beam_wise_conv_layer_different_kernel_sizes(beam_wise_conv_layer, default_device, default_dtype):
    """Test beam-wise convolutional layer with different kernel sizes"""
    BG = 2
    D = 16
    H = 32
    W = 32

    for kernel_size in [3, 5, 7, 9, 11]:
        fluence_vol = torch.randn(BG, D, H, W, 1, device=default_device, dtype=default_dtype)
        kernels = torch.randn(kernel_size, kernel_size, BG, D, device=default_device, dtype=default_dtype)

        output = beam_wise_conv_layer(fluence_vol, kernels)

        assert output.shape == (BG, D, H, W, 1), \
            f"Failed for kernel size {kernel_size}: expected shape {(BG, D, H, W, 1)}, got {output.shape}"


def test_beam_wise_conv_layer_non_square_input(beam_wise_conv_layer, default_device, default_dtype):
    """Test beam-wise convolutional layer with non-square input"""
    BG = 2
    D = 16
    H = 48
    W = 32
    kernel_size = 5

    fluence_vol = torch.randn(BG, D, H, W, 1, device=default_device, dtype=default_dtype)
    kernels = torch.randn(kernel_size, kernel_size, BG, D, device=default_device, dtype=default_dtype)

    output = beam_wise_conv_layer(fluence_vol, kernels)

    assert output.shape == (BG, D, H, W, 1), \
        f"Expected shape {(BG, D, H, W, 1)}, got {output.shape}"

