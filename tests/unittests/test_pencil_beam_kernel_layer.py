from pydose_rt.utils.utils import get_shapes
import pytest
import torch
from pydose_rt.data import MachineConfig
from pydose_rt.layers import PencilBeamKernelLayer

@pytest.fixture
def pencil_beam_kernel_layer(request, default_machine_config, default_resolution):
    """Fixture to create a PencilBeamKernelLayer with configurable kernel_size and number_of_beams."""
    kernel_size = request.param.get("kernel_size", None)
    return PencilBeamKernelLayer(default_machine_config, default_resolution, kernel_size), kernel_size


@pytest.mark.parametrize(
    "pencil_beam_kernel_layer",
    [
        {"kernel_size": 3, "number_of_beams": 1},
        {"kernel_size": 5, "number_of_beams": 8},
        {"kernel_size": 7, "number_of_beams": 25},
        {"kernel_size": 9, "number_of_beams": 64},
    ],
    indirect=True,
)
def test_pencil_beam_kernel_output_shape(pencil_beam_kernel_layer, default_ct_array_shape, default_machine_config, default_number_of_beams, default_dtype, default_device):
    """Test that output shape is as expected based on input width."""
    pencil_beam_kernel_layer, kernel_size = pencil_beam_kernel_layer
    shapes = get_shapes(default_machine_config, 
                        kernel_size=kernel_size,
                        number_of_beams=default_number_of_beams,
                        ct_shape=default_ct_array_shape)
    expected_shape = shapes["kernels"]

    radiological_depth = torch.zeros(shapes["radiological_depths"], dtype=default_dtype, device=default_device)
    kernels = pencil_beam_kernel_layer(radiological_depth)

    
    assert (
        kernels.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {kernels.shape}"

