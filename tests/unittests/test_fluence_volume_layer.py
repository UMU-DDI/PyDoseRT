from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import torch
from pydose_rt import ModelConfig
from pydose_rt.layers import FluenceVolumeLayer


# ---- Fixtures -----
@pytest.fixture
def fluence_volume_layer(default_config):
    """Fixture to create a FluenceMapLayer instance"""
    return FluenceVolumeLayer(default_config)


@pytest.fixture
def fluence_volume_layer_with_configurable_beams(request) -> tuple[FluenceVolumeLayer, ModelConfig]:
    """Fixture to create a FluenceMapLayer instance with configurable beams"""
    config = ModelConfig(
        preset="test",
        number_of_cps=request.param,
    )
    return FluenceVolumeLayer(config), config


# ----- Tests -----
def test_fluence_volume_output_shape(fluence_volume_layer, default_config):
    """Test that fluence map behaves correctly based on input width."""
    # Arrange
    fluence_map = torch.zeros(default_config.shape_fluence_map, dtype=torch.float32, device=default_config.device)
    expected = default_config.shape_fluence_volume

    # Act
    fluence_volume = fluence_volume_layer(fluence_map)
    actual = fluence_volume.shape

    # Assert
    assert actual == expected, (
        f"Expected shape {default_config.shape_fluence_volume}, but got {fluence_volume.shape}")
