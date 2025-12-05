import sys

from pydose_rt.utils.utils import get_shapes
sys.path.append("../../")
import pytest
import numpy as np
import torch
from pydose_rt.data import MachineConfig
from pydose_rt.layers import RadiologicalDepthLayer


@pytest.fixture
def radiological_depth_layer(default_machine_config, default_resolution, default_ct_array_shape, default_gantry_angles, default_iso_center):
    """Fixture to create a FluenceMapLayer instance"""
    return RadiologicalDepthLayer(default_machine_config, default_resolution, default_ct_array_shape, default_gantry_angles, default_iso_center)


def test_radiological_depth_output_shape(radiological_depth_layer, default_machine_config, default_number_of_beams, default_ct_array_shape, default_device):
    """Test that fluence map behaves correctly based on input width."""
    expected = get_shapes(default_machine_config, 
                          number_of_beams=default_number_of_beams,
                          ct_shape=default_ct_array_shape)["radiological_depths"]
    ct_array = torch.zeros(
        (
            1,
            default_ct_array_shape[0],
            default_ct_array_shape[1],
            default_ct_array_shape[2],
        ), dtype=torch.float32, device=default_device
    )

    radiological_depths = radiological_depth_layer(ct_array)

    assert (
        radiological_depths.shape == expected
    ), f"Expected shape {expected}, but got {radiological_depths.shape}"
