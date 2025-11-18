from pathlib import Path
import sys

from pydose_rt.data.treatment_config import TreatmentConfig
from pydose_rt.utils.utils import get_shapes
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import torch
from pydose_rt.data import MachineConfig
from pydose_rt.layers import FluenceVolumeLayer


# ---- Fixtures -----
@pytest.fixture
def fluence_volume_layer(default_machine_config, default_treatment_config):
    """Fixture to create a FluenceMapLayer instance"""
    return FluenceVolumeLayer(default_machine_config, default_treatment_config)


@pytest.fixture
def fluence_volume_layer_with_configurable_beams(default_machine_config, request) -> tuple[FluenceVolumeLayer, TreatmentConfig]:
    """Fixture to create a FluenceMapLayer instance with configurable beams"""
    config = TreatmentConfig(
        preset="src/pydose_rt/data/treatment_presets/test.json",
        number_of_cps=request.param,
    )
    return FluenceVolumeLayer(default_machine_config, config), config


@pytest.mark.parametrize("fluence_volume_layer_with_configurable_beams", [1, 8, 60, 120], indirect=True)
def test_fluence_volume_benchmark(benchmark, default_machine_config, fluence_volume_layer_with_configurable_beams):
    fluence_volume_layer_instance, config = fluence_volume_layer_with_configurable_beams
    shapes = get_shapes(default_machine_config, config)

    y_mlc = torch.zeros(shapes["fluence_maps"], dtype=torch.float32, device=config.device)
    benchmark(lambda: fluence_volume_layer_instance(y_mlc))
