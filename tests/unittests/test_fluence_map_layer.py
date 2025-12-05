from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import numpy as np
import torch
from pydose_rt.data import MachineConfig
from pydose_rt.layers import FluenceMapLayer
from pydose_rt.utils.grad_monitor import GradMonitor
from pydose_rt.utils.utils import get_shapes
 
@pytest.fixture
def fluence_map_layer(default_machine_config, default_resolution, default_field_size, default_device, default_dtype):
    """Fixture to create a FluenceMapLayer instance"""
    return FluenceMapLayer(
        default_machine_config,
        field_size=default_field_size,
        device=default_device,
        dtype=default_dtype
    )

@pytest.mark.parametrize(
    "center, width",
    [
        (-0.7, 0.05),
        (0.7, 0.05),
    ]
)
def test_fluence_map_leaves_center_per_width(fluence_map_layer, default_machine_config, default_number_of_beams, default_field_size, default_dtype, default_device, center, width):
    """Test that fluence map behaves correctly based on input width."""
    shapes = get_shapes(default_machine_config, 
                        number_of_beams=default_number_of_beams, 
                        field_size=default_field_size)
    y_mlc = torch.zeros(shapes["MLCs"], dtype=default_dtype, device=default_device)
    expected = (center * default_field_size[0] / 2) + default_field_size[0] / 2
    y_mlc[:, :, :, 0] = (center * default_field_size[0] / 2) - (width * default_field_size[0] / 2)  # Set left positions
    y_mlc[:, :, :, 1] = (center * default_field_size[0] / 2) + (width * default_field_size[0] / 2)  # Set right positions
    y_mlc = y_mlc.clone().detach().requires_grad_(True)
    y_jaws = torch.zeros(shapes["jaws"], dtype=default_dtype, device=default_device)
    y_jaws[:, :, 0] = - default_field_size[1] / 2
    y_jaws[:, :, 1] = default_field_size[1] / 2
    y_jaws = y_jaws.clone().detach().requires_grad_(True)

    fluence_map = fluence_map_layer(y_mlc, y_jaws)

    # Convert TensorFlow tensor to numpy if necessary
    if isinstance(fluence_map, torch.Tensor):
        fluence_map = fluence_map.cpu().detach().numpy()

    print(f"Test Case - Center: {center}, Width: {width}")
    print("Fluence Map Shape:", fluence_map.shape)
    actual = np.mean(np.argwhere(fluence_map > 0.5), (0))[2] # Look for the center of the one values

    assert actual == pytest.approx(expected, 0.01)

@pytest.mark.parametrize(
    "center, width",
    [
        (0.0, 0.0),
        (0.0, 0.001),
        (0.0, 0.5),
        (0.0, 1.0),
    ]
)
def test_fluence_map_leaves_open_per_width(fluence_map_layer, default_machine_config, default_number_of_beams, default_field_size, default_dtype, default_device, center, width):
    """Test that fluence map behaves correctly based on input width."""
    shapes = get_shapes(default_machine_config, 
                        number_of_beams=default_number_of_beams, 
                        field_size=default_field_size)
    y_mlc = torch.zeros(shapes["MLCs"], dtype=default_dtype, device=default_device)
    y_mlc[:, :, :, 0] = (center * default_field_size[0] / 2) - (width * default_field_size[0] / 2)  # Set left positions
    y_mlc[:, :, :, 1] = (center * default_field_size[0] / 2) + (width * default_field_size[0] / 2)  # Set right positions
    y_mlc = y_mlc.clone().detach().requires_grad_(True)
    y_jaws = torch.zeros(shapes["jaws"], dtype=default_dtype, device=default_device)
    y_jaws[:, :, 0] = - default_field_size[1] / 2
    y_jaws[:, :, 1] = default_field_size[1] / 2
    y_jaws = y_jaws.clone().detach().requires_grad_(True)

    fluence_map = fluence_map_layer(y_mlc, y_jaws)

    # Convert TensorFlow tensor to numpy if necessary
    if isinstance(fluence_map, torch.Tensor):
        fluence_map = fluence_map.cpu().detach().numpy()

    print(f"Test Case - Center: {center}, Width: {width}")
    print("Fluence Map Shape:", fluence_map.shape)

    ones = np.mean(fluence_map)  # Count pixels that are effectively one

    assert ones == pytest.approx(width, abs=0.01)

@pytest.mark.parametrize(
    "center, width",
    [
        (0.0, 0.0),
        (0.0, 0.5),
        (0.0, 1.0),
    ]
)
def test_fluence_map_jaws_open_per_width(fluence_map_layer, default_machine_config, default_number_of_beams, default_dtype, default_device, default_field_size, center, width):
    """Test that fluence map behaves correctly based on input width."""
    shapes = get_shapes(default_machine_config, 
                        number_of_beams=default_number_of_beams, 
                        field_size=default_field_size)
    y_mlc = torch.zeros(shapes["MLCs"], dtype=default_dtype, device=default_device)
    y_mlc[:, :, :, 0] = - default_field_size[0] / 2
    y_mlc[:, :, :, 1] = default_field_size[0] / 2
    y_mlc = y_mlc.clone().detach().requires_grad_(True)
    y_jaws = torch.zeros(shapes["jaws"], dtype=default_dtype, device=default_device)
    y_jaws[:, :, 0] = (center * default_field_size[1] / 2) - (width * default_field_size[1] / 2)  # Set bottom positions
    y_jaws[:, :, 1] = (center * default_field_size[1] / 2) + (width * default_field_size[1] / 2)  # Set top positions
    y_jaws = y_jaws.clone().detach().requires_grad_(True)

    fluence_map = fluence_map_layer(y_mlc, y_jaws)

    # Convert TensorFlow tensor to numpy if necessary
    if isinstance(fluence_map, torch.Tensor):
        fluence_map = fluence_map.cpu().detach().numpy()

    print(f"Test Case - Center: {center}, Width: {width}")
    print("Fluence Map Shape:", fluence_map.shape)

    ones = np.mean(fluence_map)  # Count pixels that are effectively one

    assert ones == pytest.approx(width, abs=0.01)

def test_fluence_map_output_shape(fluence_map_layer, default_machine_config, default_number_of_beams, default_dtype, default_device, default_field_size):
    """Test that fluence map behaves correctly based on input width."""
    shapes = get_shapes(default_machine_config, 
                        number_of_beams=default_number_of_beams, 
                        field_size=default_field_size)
    y_mlc = torch.zeros(shapes["MLCs"], dtype=default_dtype, device=default_device)
    expected = shapes["fluence_maps"]

    fluence_map = fluence_map_layer(y_mlc)

    assert fluence_map.shape == expected, f"Expected shape {expected}, but got {fluence_map.shape}"

def test_fluence_map_leaves_gradients_closing(fluence_map_layer, default_machine_config, default_number_of_beams, default_field_size, default_dtype, default_device):
    """Test that fluence map behaves correctly based on input width."""
    shapes = get_shapes(default_machine_config, 
                        number_of_beams=default_number_of_beams, 
                        field_size=default_field_size)
    y_mlc = torch.zeros(shapes["MLCs"], dtype=default_dtype, device=default_device)
    y_mlc[..., 0] = 0.0
    y_mlc[..., 1] = 1.0
    y_mlc = torch.nn.Parameter(y_mlc, requires_grad=True)

    monitor = GradMonitor(modules_to_watch=[""]).install(fluence_map_layer)
    fluence_map = fluence_map_layer(y_mlc)
    loss = torch.mean(torch.abs(torch.zeros_like(fluence_map).detach() - fluence_map))
    optimizer = torch.optim.Adam([y_mlc], lr=1e-2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, _, grad_max = np.array([float(x) for x in monitor.summary().split('(')[-1].split(')')[0].split(',')])

    assert grad_max > 0.0, "Gradients don't close fluence map"

def test_fluence_map_leaves_gradients_opening(fluence_map_layer, default_machine_config, default_number_of_beams, default_field_size, default_dtype, default_device):
    """Test that fluence map behaves correctly based on input width."""
    shapes = get_shapes(default_machine_config, 
                        number_of_beams=default_number_of_beams, 
                        field_size=default_field_size)
    y_mlc = torch.zeros(shapes["MLCs"], dtype=default_dtype, device=default_device)
    y_mlc[..., 0] = 0.5 # Set left positions
    y_mlc[..., 1] = 0.5 # Set right positions
    y_mlc = torch.nn.Parameter(y_mlc, requires_grad=True)

    monitor = GradMonitor(modules_to_watch=[""]).install(fluence_map_layer)
    fluence_map = fluence_map_layer(y_mlc)
    loss = torch.mean(torch.abs(torch.ones_like(fluence_map).detach() - fluence_map))
    optimizer = torch.optim.Adam([y_mlc], lr=1e-2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    grad_min, _, _ = np.array([float(x) for x in monitor.summary().split('(')[-1].split(')')[0].split(',')])

    assert grad_min < 0.0, "Gradients don't open fluence map"
