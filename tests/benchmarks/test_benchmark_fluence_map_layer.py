from pathlib import Path
import sys

from pydose_rt.utils.utils import get_shapes
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import numpy as np
import torch
from pydose_rt.data import MachineConfig, TreatmentConfig
from pydose_rt.layers import FluenceMapLayer
from pydose_rt.utils.grad_monitor import GradMonitor

@pytest.fixture
def fluence_map_layer(default_machine_config, default_treatment_config):
    """Fixture to create a FluenceMapLayer instance"""
    return FluenceMapLayer(default_machine_config, default_treatment_config)

@pytest.fixture
def fluence_map_layer_beams(default_machine_config, request):
    """Fixture to create a FluenceMapLayer instance with configurable beams"""
    config = TreatmentConfig(
        preset="src/pydose_rt/data/optimization_presets/test.json",
        number_of_cps=request.param,
    )
    return FluenceMapLayer(default_machine_config, config), config

@pytest.mark.parametrize("fluence_map_layer_beams", [1, 8, 60, 120], indirect=True)
def test_fluence_map_benchmark(benchmark, fluence_map_layer_beams, default_machine_config):
    
    fluence_layer, config = fluence_map_layer_beams
    shapes = get_shapes(default_machine_config, config)

    y_mlc = torch.zeros(shapes["MLCs"], dtype=config.dtype, device=config.device)
    y_mlc[:, 0, :, :] = 0.0  # Set center
    y_mlc[:, 1, :, :] = 1.0  # Set width
    y_mlc = y_mlc.clone().detach().requires_grad_(True)

    benchmark(lambda: fluence_layer(y_mlc))
