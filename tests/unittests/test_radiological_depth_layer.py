import sys
sys.path.append("../../")
import pytest
import numpy as np
import torch
from pydose_rt.data import MachineConfig
from pydose_rt.layers import RadiologicalDepthLayer


@pytest.fixture
def radiological_depth_layer(default_config):
    """Fixture to create a FluenceMapLayer instance"""
    return RadiologicalDepthLayer(default_config)


def test_radiological_depth_output_shape(radiological_depth_layer, default_config):
    """Test that fluence map behaves correctly based on input width."""
    ct_array = torch.zeros(
        (
            1,
            default_config.ct_array_shape[0],
            default_config.ct_array_shape[1],
            default_config.ct_array_shape[2],
        ), dtype=torch.float32, device=default_config.device
    )

    radiological_depths = radiological_depth_layer(ct_array)

    assert (
        radiological_depths.shape == default_config.shape_radiological_depth
    ), f"Expected shape {default_config.shape_radiological_depth}, but got {radiological_depths.shape}"


@pytest.fixture
def radiological_depth_layer_beams(request):
    """Fixture to create a FluenceMapLayer instance with configurable beams"""
    config = MachineConfig(
        preset="test",
        number_of_cps=request.param,
    )
    return RadiologicalDepthLayer(config), config

