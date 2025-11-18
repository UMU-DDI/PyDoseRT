import sys

from pydose_rt.utils.utils import get_shapes
sys.path.append("../../")
import pytest
import numpy as np
import torch
from pydose_rt.data import MachineConfig
from pydose_rt.layers import PencilBeamKernelLayer

@pytest.fixture
def pencil_beam_kernel_layer(request, default_machine_config, default_treatment_config):
    """Fixture to create a PencilBeamKernelLayer with configurable kernel_size and number_of_cps."""
    kernel_size = request.param.get("kernel_size", 5)
    number_of_cps = request.param.get(
        "number_of_cps", default_treatment_config.number_of_cps
    )
    treatment_config = default_treatment_config
    treatment_config.number_of_cps = number_of_cps

    return PencilBeamKernelLayer(default_machine_config, treatment_config, kernel_size), treatment_config


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
def test_pencil_beam_kernel_output_shape(pencil_beam_kernel_layer, default_machine_config):
    """Test that output shape is as expected based on input width."""
    pencil_beam_kernel_layer, treatment_config = pencil_beam_kernel_layer
    shapes = get_shapes(default_machine_config, treatment_config)
    expected_shape = shapes["kernels"]

    radiological_depth = torch.zeros(shapes["radiological_depths"], dtype=torch.float32, device=treatment_config.device)
    kernels = pencil_beam_kernel_layer(radiological_depth)

    
    assert (
        kernels.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {kernels.shape}"

