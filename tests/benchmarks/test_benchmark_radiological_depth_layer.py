import sys
sys.path.append("../../")
import pytest
import numpy as np
import torch
from pydose_rt.data import MachineConfig, TreatmentConfig
from pydose_rt.layers import RadiologicalDepthLayer


@pytest.fixture
def radiological_depth_layer(default_machine_config, default_treatment_config):
    """Fixture to create a FluenceMapLayer instance"""
    return RadiologicalDepthLayer(default_machine_config, default_treatment_config)


@pytest.fixture
def radiological_depth_layer_beams(default_machine_config, request):
    """Fixture to create a FluenceMapLayer instance with configurable beams"""
    config = TreatmentConfig(
        preset="src/pydose_rt/data/optimization_presets/test.json",
        number_of_cps=request.param,
    )
    return RadiologicalDepthLayer(default_machine_config, config), config


@pytest.mark.parametrize(
    "radiological_depth_layer_beams", [1, 8, 60, 120], indirect=True
)
def test_radiological_depth_benchmark(benchmark, default_machine_config, radiological_depth_layer_beams):
    radiological_depth_layer, config = radiological_depth_layer_beams

    ct_array = torch.zeros(
        (
            1,
            default_machine_config.ct_array_shape[0],
            default_machine_config.ct_array_shape[1],
            default_machine_config.ct_array_shape[2],
        ), dtype=config.dtype, device=config.device
    )
    benchmark(lambda: radiological_depth_layer(ct_array))
