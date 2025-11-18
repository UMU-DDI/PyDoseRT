from pathlib import Path
import sys

from pydose_rt.utils.utils import get_shapes
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import torch
from pydose_rt.data import MachineConfig, TreatmentConfig
from pydose_rt.layers import FluenceVolumeLayer


# ---- Fixtures -----
@pytest.fixture
def fluence_volume_layer(default_machine_config, default_treatment_config):
    """Fixture to create a FluenceMapLayer instance"""
    return FluenceVolumeLayer(default_machine_config, default_treatment_config)


@pytest.fixture
def fluence_volume_layer_with_configurable_beams(request) -> tuple[FluenceVolumeLayer, TreatmentConfig]:
    """Fixture to create a FluenceMapLayer instance with configurable beams"""
    config = TreatmentConfig(
        preset="src/pydose_rt/data/treatment_presets/test.json",
        number_of_cps=request.param,
    )
    return FluenceVolumeLayer(config), config


# ----- Tests -----
def test_fluence_volume_output_shape(fluence_volume_layer, default_machine_config, default_treatment_config):
    """Test that fluence map behaves correctly based on input width."""
    # Arrange
    shapes = get_shapes(default_machine_config, default_treatment_config)
    fluence_map = torch.zeros(shapes["fluence_maps"], dtype=torch.float32, device=default_treatment_config.device)
    expected = shapes["fluence_volumes"]

    # Act
    fluence_volume = fluence_volume_layer(fluence_map)
    actual = fluence_volume.shape

    # Assert
    assert actual == expected, (
        f"Expected shape {expected}, but got {fluence_volume.shape}")
