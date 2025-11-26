import sys

from pydose_rt.utils.utils import get_shapes
sys.path.append("../../")
import pytest
import numpy as np
import torch
from pydose_rt.data import MachineConfig
from pydose_rt.layers import PencilBeamKernelLayer

@pytest.fixture
def pencil_beam_kernel_layer(request, default_machine_config, default_resolution, default_kernel_size):
    """Fixture to create a PencilBeamKernelLayer with configurable kernel_size and number_of_beams."""
    kernel_size = request.param
    return PencilBeamKernelLayer(default_machine_config, default_resolution, kernel_size), kernel_size

@pytest.mark.parametrize("pencil_beam_kernel_layer", [1, 5, 15, 51], indirect=True)
def test_pencil_beam_kernel(benchmark, default_machine_config, default_number_of_beams, default_dtype, default_ct_array_shape, default_device, pencil_beam_kernel_layer):
    """Benchmark radiological depth computation."""
    pencil_beam_kernel_layer, kernel_size = pencil_beam_kernel_layer
    shapes = get_shapes(default_machine_config, 
                        number_of_beams=default_number_of_beams,
                        kernel_size=kernel_size,
                        ct_shape=default_ct_array_shape)

    radiological_depth = torch.zeros(shapes["radiological_depths"], dtype=default_dtype, device=default_device)
    benchmark(lambda: pencil_beam_kernel_layer(radiological_depth))
