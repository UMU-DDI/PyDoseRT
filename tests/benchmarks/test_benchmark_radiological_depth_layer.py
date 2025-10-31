import sys
sys.path.append("../../")
import pytest
import numpy as np
import torch
from pydose_rt import ModelConfig
from pydose_rt.layers import RadiologicalDepthLayer


@pytest.fixture
def radiological_depth_layer(default_config):
    """Fixture to create a FluenceMapLayer instance"""
    return RadiologicalDepthLayer(default_config)


@pytest.fixture
def radiological_depth_layer_beams(request):
    """Fixture to create a FluenceMapLayer instance with configurable beams"""
    config = ModelConfig(
        preset="test",
        number_of_cps=request.param,
    )
    return RadiologicalDepthLayer(config), config


@pytest.mark.parametrize(
    "radiological_depth_layer_beams", [1, 8, 60, 120], indirect=True
)
def test_radiological_depth_benchmark(benchmark, radiological_depth_layer_beams):
    radiological_depth_layer, config = radiological_depth_layer_beams

    ct_array = torch.zeros(
        (
            1,
            config.ct_array_shape[0],
            config.ct_array_shape[1],
            config.ct_array_shape[2],
        ), dtype=torch.float32, device=config.device
    )
    benchmark(lambda: radiological_depth_layer(ct_array))
