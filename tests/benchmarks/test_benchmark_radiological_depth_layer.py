import sys
import pytest
import numpy as np
import torch
from pydose_rt.layers import RadiologicalDepthLayer



@pytest.fixture
def radiological_depth_layer_beams(default_machine_config, default_resolution, default_ct_array_shape, default_iso_center, request):
    """Fixture to create a FluenceMapLayer instance with configurable beams"""
    gantry_angles = np.linspace(0, 360, int(request.param))
    return RadiologicalDepthLayer(default_machine_config, default_resolution, default_ct_array_shape, gantry_angles, default_iso_center), gantry_angles


@pytest.mark.parametrize(
    "radiological_depth_layer_beams", [1, 8, 60, 120], indirect=True
)
def test_radiological_depth_benchmark(benchmark, default_machine_config, default_ct_array_shape, default_device, default_dtype, radiological_depth_layer_beams):
    radiological_depth_layer, gantry_angles = radiological_depth_layer_beams

    ct_array = torch.zeros(
        (
            1,
            default_ct_array_shape[0],
            default_ct_array_shape[1],
            default_ct_array_shape[2],
        ), dtype=default_dtype, device=default_device
    )
    benchmark(lambda: radiological_depth_layer(ct_array))
