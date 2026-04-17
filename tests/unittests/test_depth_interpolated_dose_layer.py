import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import torch

from pydosert.layers import DepthInterpolatedDoseLayer


@pytest.fixture
def reference_depths():
    return [0.0, 10.0, 20.0, 40.0, 50.0, 100.0]


@pytest.fixture
def layer(reference_depths, default_device, default_dtype):
    return DepthInterpolatedDoseLayer(
        reference_depths=reference_depths,
        device=default_device,
        dtype=default_dtype,
    )


def test_output_shape(layer, default_device, default_dtype, reference_depths):
    BG, N, D, H, W = 2, len(reference_depths), 8, 6, 7
    convolved = torch.randn(BG, N, D, H, W, device=default_device, dtype=default_dtype)
    rad_depth = torch.rand(BG, D, H, W, device=default_device, dtype=default_dtype) * 90.0

    dose = layer(convolved, rad_depth)
    assert dose.shape == (BG, D, H, W, 1)


def test_identity_when_all_channels_equal(layer, default_device, default_dtype, reference_depths):
    """If every reference depth's convolved fluence is identical, the output
    should be that value regardless of per-voxel depth."""
    BG, N, D, H, W = 1, len(reference_depths), 4, 4, 4
    base = torch.randn(BG, 1, D, H, W, device=default_device, dtype=default_dtype)
    convolved = base.expand(BG, N, D, H, W).contiguous()
    rad_depth = torch.rand(BG, D, H, W, device=default_device, dtype=default_dtype) * 100.0

    dose = layer(convolved, rad_depth).squeeze(-1)
    assert torch.allclose(dose, base.squeeze(1), atol=1e-5)


def test_exact_reference_depth_picks_that_channel(layer, default_device, default_dtype, reference_depths):
    """When a voxel's rad-depth exactly hits a reference depth, the output
    equals that channel's value."""
    BG, D, H, W = 1, 1, 1, 1
    N = len(reference_depths)
    convolved = torch.arange(N, device=default_device, dtype=default_dtype).view(1, N, 1, 1, 1)
    for i, d in enumerate(reference_depths):
        rad_depth = torch.tensor([[[[d]]]], device=default_device, dtype=default_dtype)
        dose = layer(convolved, rad_depth)
        assert torch.isclose(dose.squeeze(), torch.tensor(float(i), device=default_device, dtype=default_dtype), atol=1e-5), \
            f"At exact reference depth {d}, expected channel {i}, got {dose.item()}"


def test_linear_interpolation_midpoint(layer, default_device, default_dtype, reference_depths):
    """At the midpoint of two adjacent reference depths, result is the average."""
    BG, D, H, W = 1, 1, 1, 1
    N = len(reference_depths)
    values = torch.arange(N, device=default_device, dtype=default_dtype).view(1, N, 1, 1, 1)
    # Midpoint between reference_depths[1]=10 and [2]=20 -> 15
    rad_depth = torch.tensor([[[[15.0]]]], device=default_device, dtype=default_dtype)
    dose = layer(values, rad_depth)
    assert torch.isclose(dose.squeeze(), torch.tensor(1.5, device=default_device, dtype=default_dtype), atol=1e-5)


def test_out_of_range_clamps(layer, default_device, default_dtype, reference_depths):
    BG, D, H, W = 1, 1, 1, 1
    N = len(reference_depths)
    values = torch.arange(N, device=default_device, dtype=default_dtype).view(1, N, 1, 1, 1)

    # below minimum
    rad_depth = torch.tensor([[[[-50.0]]]], device=default_device, dtype=default_dtype)
    dose = layer(values, rad_depth)
    assert torch.isclose(dose.squeeze(), torch.tensor(0.0, device=default_device, dtype=default_dtype), atol=1e-5)

    # above maximum
    rad_depth = torch.tensor([[[[500.0]]]], device=default_device, dtype=default_dtype)
    dose = layer(values, rad_depth)
    assert torch.isclose(dose.squeeze(),
                         torch.tensor(float(N - 1), device=default_device, dtype=default_dtype),
                         atol=1e-5)


def test_gradients_flow_through_convolved_fluences(layer, default_device, default_dtype, reference_depths):
    BG, N, D, H, W = 1, len(reference_depths), 3, 3, 3
    convolved = torch.randn(BG, N, D, H, W, device=default_device, dtype=default_dtype, requires_grad=True)
    rad_depth = torch.rand(BG, D, H, W, device=default_device, dtype=default_dtype) * 90.0

    dose = layer(convolved, rad_depth)
    dose.sum().backward()

    assert convolved.grad is not None
    assert torch.any(convolved.grad != 0)


def test_non_increasing_reference_depths_raises(default_device, default_dtype):
    with pytest.raises(ValueError):
        DepthInterpolatedDoseLayer(
            reference_depths=[0.0, 10.0, 5.0],
            device=default_device,
            dtype=default_dtype,
        )


def test_single_reference_depth_raises(default_device, default_dtype):
    with pytest.raises(ValueError):
        DepthInterpolatedDoseLayer(
            reference_depths=[10.0],
            device=default_device,
            dtype=default_dtype,
        )


def test_mismatched_channel_count_raises(layer, default_device, default_dtype, reference_depths):
    BG, D, H, W = 1, 2, 2, 2
    # Wrong number of channels (one less than reference depths)
    convolved = torch.randn(BG, len(reference_depths) - 1, D, H, W,
                            device=default_device, dtype=default_dtype)
    rad_depth = torch.rand(BG, D, H, W, device=default_device, dtype=default_dtype)
    with pytest.raises(ValueError):
        layer(convolved, rad_depth)
