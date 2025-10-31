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


@pytest.mark.parametrize("fluence_volume_layer_with_configurable_beams", [1, 8, 60, 120], indirect=True)
def test_fluence_volume_benchmark(benchmark, fluence_volume_layer_with_configurable_beams):
    fluence_volume_layer_instance, config = fluence_volume_layer_with_configurable_beams

    y_mlc = torch.zeros(config.shape_fluence_map, dtype=torch.float32, device=config.device)
    benchmark(lambda: fluence_volume_layer_instance(y_mlc))
