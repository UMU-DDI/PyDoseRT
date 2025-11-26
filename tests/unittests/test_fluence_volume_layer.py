from pathlib import Path
import sys

from pydose_rt.utils.utils import get_shapes
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import torch
from pydose_rt.data import MachineConfig
from pydose_rt.layers import FluenceVolumeLayer


# ---- Fixtures -----
@pytest.fixture
def fluence_volume_layer(default_machine_config, default_resolution, default_ct_array_shape):
    """Fixture to create a FluenceMapLayer instance"""
    return FluenceVolumeLayer(default_machine_config, default_resolution, default_ct_array_shape)


# ----- Tests -----
def test_fluence_volume_output_shape(fluence_volume_layer, default_machine_config, default_field_size, default_ct_array_shape, default_number_of_beams, default_dtype, default_device):
    """Test that fluence map behaves correctly based on input width."""
    # Arrange
    shapes = get_shapes(default_machine_config, 
                        number_of_beams=default_number_of_beams,
                        field_size=default_field_size,
                        ct_shape=default_ct_array_shape)
    fluence_map = torch.zeros(shapes["fluence_maps"], dtype=default_dtype, device=default_device)
    expected = shapes["fluence_volumes"]

    # Act
    fluence_volume = fluence_volume_layer(fluence_map)
    actual = fluence_volume.shape

    # Assert
    assert actual == expected, (
        f"Expected shape {expected}, but got {fluence_volume.shape}")
