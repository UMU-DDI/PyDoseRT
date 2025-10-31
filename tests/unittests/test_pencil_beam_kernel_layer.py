import sys
sys.path.append("../../")
import pytest
import numpy as np
import torch
from DoseEngines import ModelConfig
from DoseEngines.layers import PencilBeamKernelLayer

@pytest.fixture
def pencil_beam_kernel_layer(request, default_config):
    """Fixture to create a PencilBeamKernelLayer with configurable kernel_size and number_of_cps."""
    kernel_size = request.param.get("kernel_size", 5)
    number_of_cps = request.param.get(
        "number_of_cps", default_config.number_of_cps
    )

    config = ModelConfig(
        ct_array_shape=default_config.ct_array_shape,
        resolution=default_config.resolution,
        field_size=default_config.field_size,
        number_of_leaf_pairs=default_config.number_of_leaf_pairs,
        tpr_20_10=default_config.tpr_20_10,
        number_of_cps=number_of_cps,
    )
    return PencilBeamKernelLayer(config, kernel_size), config


@pytest.mark.parametrize(
    "pencil_beam_kernel_layer",
    [
        {"kernel_size": 3, "number_of_cps": 1},
        {"kernel_size": 5, "number_of_cps": 8},
        {"kernel_size": 7, "number_of_cps": 25},
        {"kernel_size": 9, "number_of_cps": 64},
    ],
    indirect=True,
)
def test_pencil_beam_kernel_output_shape(pencil_beam_kernel_layer):
    """Test that output shape is as expected based on input width."""
    pencil_beam_kernel_layer, config = pencil_beam_kernel_layer
    kernel_size = pencil_beam_kernel_layer.kernel_size

    radiological_depth = torch.zeros(config.shape_radiological_depth, dtype=torch.float32, device=config.device)

    kernels = pencil_beam_kernel_layer(radiological_depth)

    expected_shape = (
        kernel_size,
        kernel_size,
        config.shape_radiological_depth[0],
        config.shape_radiological_depth[1],
    )
    assert (
        kernels.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {kernels.shape}"

