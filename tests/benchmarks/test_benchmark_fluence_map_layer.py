from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import numpy as np
import torch
from DoseEngines import ModelConfig
from DoseEngines.layers import FluenceMapLayer
from engine.utils.grad_monitor import GradMonitor

@pytest.fixture
def fluence_map_layer(default_config):
    """Fixture to create a FluenceMapLayer instance"""
    return FluenceMapLayer(default_config)

@pytest.fixture
def fluence_map_layer_beams(request):
    """Fixture to create a FluenceMapLayer instance with configurable beams"""
    config = ModelConfig(
        preset="test",
        number_of_cps=request.param,
    )
    return FluenceMapLayer(config), config

@pytest.mark.parametrize("fluence_map_layer_beams", [1, 8, 60, 120], indirect=True)
def test_fluence_map_benchmark(benchmark, fluence_map_layer_beams):
    fluence_layer, config = fluence_map_layer_beams

    y_mlc = torch.zeros(config.shape_mlc[0], dtype=torch.float32, device=config.device)
    y_mlc[:, 0, :, :] = 0.0  # Set center
    y_mlc[:, 1, :, :] = 1.0  # Set width
    y_mlc = y_mlc.clone().detach().requires_grad_(True)

    benchmark(lambda: fluence_layer(y_mlc))
