import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import numpy as np
import torch
from pydose_rt.layers.BeamRotationLayer import BeamRotationLayer


@pytest.fixture
def beam_rotation_layer(default_resolution, default_ct_array_shape, default_gantry_angles, default_iso_center, default_machine_config):
    """Fixture to create a BeamRotationLayer instance"""
    return BeamRotationLayer(
        machine_config=default_machine_config,
        resolution=default_resolution,
        ct_array_shape=default_ct_array_shape,
        gantry_angles=default_gantry_angles,
        iso_center=default_iso_center
    )


def test_beam_rotation_layer_output_shape(beam_rotation_layer, default_number_of_beams, default_ct_array_shape, default_device):
    """Test that beam rotation layer produces correct output shape"""
    # Create a dose volume for each beam
    dose_volumes = torch.randn(
        1,
        default_number_of_beams,
        default_ct_array_shape[0],
        default_ct_array_shape[1],
        default_ct_array_shape[2],
        device=default_device
    )

    rotated_dose = beam_rotation_layer(dose_volumes)

    expected_shape = (
        1,
        default_number_of_beams,
        default_ct_array_shape[0],
        default_ct_array_shape[1],
        default_ct_array_shape[2]
    )

    assert rotated_dose.shape == expected_shape, \
        f"Expected shape {expected_shape}, but got {rotated_dose.shape}"


def test_beam_rotation_layer_zero_angle_identity(default_resolution, default_ct_array_shape, default_iso_center, default_machine_config, default_device):
    """Test that zero gantry angle produces identity rotation (approximately)"""
    gantry_angles = torch.zeros(1, device=default_device)
    layer = BeamRotationLayer(
        machine_config=default_machine_config,
        resolution=default_resolution,
        ct_array_shape=default_ct_array_shape,
        gantry_angles=gantry_angles,
        iso_center=default_iso_center
    )

    # Create a simple dose volume with a distinctive pattern
    dose_volumes = torch.zeros(1, 1, *default_ct_array_shape, device=default_device)
    center = tuple(s // 2 for s in default_ct_array_shape)
    dose_volumes[0, 0, center[0], center[1], center[2]] = 1.0

    rotated_dose = layer(dose_volumes)

    # For zero rotation, the center should be preserved
    assert rotated_dose[0, 0, center[0], center[1], center[2]] > 0.5, \
        "Zero rotation should preserve central voxel"


def test_beam_rotation_layer_gradients(default_resolution, default_ct_array_shape, default_iso_center, default_machine_config, default_device):
    """Test that gradients flow through the beam rotation layer"""
    gantry_angles = torch.tensor([45.0], device=default_device, requires_grad=True)
    layer = BeamRotationLayer(
        machine_config=default_machine_config,
        resolution=default_resolution,
        ct_array_shape=default_ct_array_shape,
        gantry_angles=gantry_angles,
        iso_center=default_iso_center
    )

    dose_volumes = torch.randn(1, 1, *default_ct_array_shape, device=default_device, requires_grad=True)
    rotated_dose = layer(dose_volumes)
    loss = rotated_dose.sum()
    loss.backward()

    assert dose_volumes.grad is not None, "Gradients should flow to dose volumes"
    assert torch.any(dose_volumes.grad != 0), "Gradients should be non-zero"


def test_beam_rotation_layer_multiple_beams(default_resolution, default_ct_array_shape, default_iso_center, default_machine_config, default_device):
    """Test beam rotation layer with multiple gantry angles"""
    num_beams = 4
    gantry_angles = torch.tensor([0.0, 90.0, 180.0, 270.0], device=default_device)
    layer = BeamRotationLayer(
        machine_config=default_machine_config,
        resolution=default_resolution,
        ct_array_shape=default_ct_array_shape,
        gantry_angles=gantry_angles,
        iso_center=default_iso_center
    )

    dose_volumes = torch.randn(1, num_beams, *default_ct_array_shape, device=default_device)
    rotated_dose = layer(dose_volumes)

    expected_shape = (1, num_beams, *default_ct_array_shape)
    assert rotated_dose.shape == expected_shape, \
        f"Expected shape {expected_shape}, but got {rotated_dose.shape}"


def test_beam_rotation_layer_preserves_dtype(beam_rotation_layer, default_ct_array_shape, default_device, default_machine_config, default_dtype):
    """Test that beam rotation layer preserves tensor dtype"""
    dose_volumes = torch.randn(1, 1, *default_ct_array_shape, device=default_device, dtype=default_dtype)
    rotated_dose = beam_rotation_layer(dose_volumes)

    assert rotated_dose.dtype == default_dtype, \
        f"Expected dtype {default_dtype}, but got {rotated_dose.dtype}"