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
        preset="test",
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
def test_pencil_beam_kernel(benchmark, pencil_beam_kernel_layer):
    """Benchmark radiological depth computation."""
    pencil_beam_kernel_layer, config = pencil_beam_kernel_layer

    radiological_depth = torch.zeros(config.shape_radiological_depth, dtype=torch.float32, device=config.device)
    benchmark(lambda: pencil_beam_kernel_layer(radiological_depth))
