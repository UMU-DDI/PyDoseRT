import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import numpy as np
import torch
from pydose_rt.layers.BeamValidationLayer import BeamValidationLayer, adjust_mask



def test_adjust_mask_minimum_overlap():
    """Test that adjust_mask enforces minimum overlap"""
    # Create positions that are too close
    pos_a = torch.tensor([[10.0]])
    pos_b = torch.tensor([[11.0]])  # Only 1mm apart
    min_overlap = 5.0
    field_size = 200.0

    result = adjust_mask(pos_a, pos_b, min_overlap, field_size)

    # Width should be at least min_overlap
    width = result[..., 1] - result[..., 0]
    assert torch.all(width >= min_overlap - 1e-5), \
        f"Width {width} should be at least {min_overlap}"


def test_adjust_mask_field_bounds():
    """Test that adjust_mask keeps positions within field bounds"""
    # Create positions outside field
    pos_a = torch.tensor([[-250.0]])
    pos_b = torch.tensor([[250.0]])
    min_overlap = 5.0
    field_size = 200.0

    result = adjust_mask(pos_a, pos_b, min_overlap, field_size)

    # Positions should be clamped to [-field_size, field_size]
    assert torch.all(result >= -field_size), "Positions should be >= -field_size"
    assert torch.all(result <= field_size), "Positions should be <= field_size"


@pytest.fixture
def beam_validation_layer(default_machine_config, default_field_size, default_device, default_dtype):
    """Fixture to create a BeamValidationLayer instance"""
    return BeamValidationLayer(
        machine_config=default_machine_config,
        field_size=default_field_size,
        device=default_device,
        dtype=default_dtype
    )


def test_beam_validation_layer_output_shape(beam_validation_layer, default_machine_config, default_number_of_beams, default_field_size, default_device, default_dtype):
    """Test that beam validation layer produces correct output shapes"""
    num_leaves = default_machine_config.number_of_leaf_pairs

    # Create input tensors
    leaf_positions = torch.randn(default_number_of_beams, num_leaves, 2, device=default_device, dtype=default_dtype)
    jaw_positions = torch.randn(default_number_of_beams, 2, device=default_device, dtype=default_dtype)
    mus = torch.randn(default_number_of_beams, device=default_device, dtype=default_dtype)

    mlc_out, jaw_out, mus_out = beam_validation_layer(leaf_positions, mus, jaw_positions)

    assert mlc_out.shape == leaf_positions.shape, \
        f"MLC output shape {mlc_out.shape} should match input {leaf_positions.shape}"
    assert jaw_out.shape == jaw_positions.shape, \
        f"Jaw output shape {jaw_out.shape} should match input {jaw_positions.shape}"
    assert mus_out.shape == mus.shape, \
        f"MUs output shape {mus_out.shape} should match input {mus.shape}"


def test_beam_validation_layer_enforces_minimum_mus(beam_validation_layer, default_number_of_beams, default_device, default_dtype):
    """Test that beam validation layer enforces minimum MU values"""
    # Create very small or negative MUs
    mus = torch.tensor([-1.0, 0.0, 0.05, 1.0], device=default_device, dtype=default_dtype)
    leaf_positions = torch.randn(4, 10, 2, device=default_device, dtype=default_dtype)

    _, _, mus_out = beam_validation_layer(leaf_positions, mus)

    # All MUs should be at least 0.1 (minimum from proj_ste)
    assert torch.all(mus_out >= 0.1 - 1e-5), \
        f"All MUs should be >= 0.1, but got min {mus_out.min()}"


def test_beam_validation_layer_enforces_minimum_leaf_opening(beam_validation_layer, default_machine_config, default_device, default_dtype):
    """Test that beam validation layer enforces minimum leaf opening"""
    # Create leaf positions that are too close together
    leaf_positions = torch.zeros(1, default_machine_config.number_of_leaf_pairs, 2, device=default_device, dtype=default_dtype)
    leaf_positions[..., 0] = -0.5  # Left leaf
    leaf_positions[..., 1] = 0.5   # Right leaf (only 1mm apart)

    mus = torch.ones(1, device=default_device, dtype=default_dtype)

    mlc_out, _, _ = beam_validation_layer(leaf_positions, mus)

    # Opening should be at least minimum_leaf_opening
    openings = mlc_out[..., 1] - mlc_out[..., 0]
    min_opening = default_machine_config.minimum_leaf_opening
    assert torch.all(openings >= min_opening - 1e-4), \
        f"All leaf openings should be >= {min_opening}, but got min {openings.min()}"


def test_beam_validation_layer_gradients(beam_validation_layer, default_machine_config, default_device, default_dtype):
    """Test that gradients flow through beam validation layer (STE behavior)"""
    leaf_positions = torch.randn(1, default_machine_config.number_of_leaf_pairs, 2, device=default_device, dtype=default_dtype, requires_grad=True)
    jaw_positions = torch.randn(1, 2, device=default_device, dtype=default_dtype, requires_grad=True)
    mus = torch.randn(100, device=default_device, dtype=default_dtype, requires_grad=True)

    mlc_out, jaw_out, mus_out = beam_validation_layer(leaf_positions, mus, jaw_positions)

    loss = mlc_out.sum() + jaw_out.sum() + mus_out.sum()
    loss.backward()

    # Check that gradients exist
    assert leaf_positions.grad is not None, "Gradients should flow to leaf positions"
    assert jaw_positions.grad is not None, "Gradients should flow to jaw positions"
    assert mus.grad is not None, "Gradients should flow to MUs"

    # Check that gradients are non-zero
    assert torch.any(leaf_positions.grad != 0), "Leaf position gradients should be non-zero"
    assert torch.any(jaw_positions.grad != 0), "Jaw position gradients should be non-zero"
    assert torch.any(mus.grad != 0), "MU gradients should be non-zero"


def test_beam_validation_layer_without_jaws(beam_validation_layer, default_machine_config, default_device, default_dtype):
    """Test that beam validation layer handles None jaw positions"""
    leaf_positions = torch.randn(1, default_machine_config.number_of_leaf_pairs, 2, device=default_device, dtype=default_dtype)
    mus = torch.ones(1, device=default_device, dtype=default_dtype)

    mlc_out, jaw_out, mus_out = beam_validation_layer(leaf_positions, mus, jaw_positions=None)

    assert mlc_out is not None, "MLC output should exist"
    assert jaw_out is None, "Jaw output should be None when input is None"
    assert mus_out is not None, "MUs output should exist"


def test_beam_validation_layer_preserves_dtype(beam_validation_layer, default_machine_config, default_device):
    """Test that beam validation layer preserves tensor dtype"""
    for dtype in [torch.float32, torch.float64]:
        leaf_positions = torch.randn(1, default_machine_config.number_of_leaf_pairs, 2, device=default_device, dtype=dtype)
        jaw_positions = torch.randn(1, 2, device=default_device, dtype=dtype)
        mus = torch.ones(1, device=default_device, dtype=dtype)

        mlc_out, jaw_out, mus_out = beam_validation_layer(leaf_positions, mus, jaw_positions)

        assert mlc_out.dtype == dtype, f"MLC dtype should be {dtype}"
        assert jaw_out.dtype == dtype, f"Jaw dtype should be {dtype}"
        assert mus_out.dtype == dtype, f"MU dtype should be {dtype}"