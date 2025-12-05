from pathlib import Path
import sys

from pydose_rt.utils.utils import get_shapes
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import torch
from pydose_rt.layers import FluenceMapLayer

@pytest.fixture
def fluence_map_layer(default_machine_config, default_resolution):
    """Fixture to create a FluenceMapLayer instance"""
    return FluenceMapLayer(default_machine_config)

def test_fluence_map_benchmark(benchmark, fluence_map_layer, default_number_of_beams, default_dtype, default_device, default_machine_config):
    
    shapes = get_shapes(default_machine_config, 
                        number_of_beams=default_number_of_beams)

    y_mlc = torch.zeros(shapes["MLCs"], dtype=default_dtype, device=default_device)
    y_mlc[:, :, :, 0] = 0.0  # Set center
    y_mlc[:, :, :, 1] = 1.0  # Set width
    y_mlc = y_mlc.clone().detach().requires_grad_(True)

    benchmark(lambda: fluence_map_layer(y_mlc))
