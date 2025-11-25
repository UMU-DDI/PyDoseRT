import sys

from pydose_rt.utils.utils import get_shapes
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


def test_radiological_depth_output_shape(radiological_depth_layer, default_machine_config, default_treatment_config):
    """Test that fluence map behaves correctly based on input width."""
    expected = get_shapes(default_machine_config, default_treatment_config)["radiological_depths"]
    ct_array = torch.zeros(
        (
            1,
            default_machine_config.ct_array_shape[0],
            default_machine_config.ct_array_shape[1],
            default_machine_config.ct_array_shape[2],
        ), dtype=torch.float32, device=default_treatment_config.device
    )

    radiological_depths = radiological_depth_layer(ct_array)

    assert (
        radiological_depths.shape == expected
    ), f"Expected shape {expected}, but got {radiological_depths.shape}"


@pytest.fixture
def radiological_depth_layer_beams(request):
    """Fixture to create a FluenceMapLayer instance with configurable beams"""
    config = TreatmentConfig(
        preset="src/pydose_rt/data/optimization_presets/test.json",
        number_of_cps=request.param,
    )
    return RadiologicalDepthLayer(config), config

