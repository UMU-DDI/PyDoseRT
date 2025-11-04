from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import numpy as np
import torch
from pydose_rt.data import MachineConfig
from pydose_rt.layers import FluenceMapLayer
from pydose_rt.utils.grad_monitor import GradMonitor

@pytest.fixture
def fluence_map_layer(default_config):
    """Fixture to create a FluenceMapLayer instance"""
    return FluenceMapLayer(default_config)

@pytest.fixture
def fluence_map_layer_beams(request):
    """Fixture to create a FluenceMapLayer instance with configurable beams"""
    config = MachineConfig(
        preset="test",
        number_of_cps=request.param,
    )
    return FluenceMapLayer(config), config

@pytest.mark.parametrize(
    "center, width",
    [
        (0.5, 0.0),
        (0.5, 0.5),
        (0.5, 1.0),
    ]
)
def test_fluence_map_leaves_open_per_width(fluence_map_layer, default_config, center, width):
    """Test that fluence map behaves correctly based on input width."""
    y_mlc = torch.zeros(default_config.shape_mlc[0], dtype=torch.float32, device=default_config.device)
    y_mlc[:, 0, :, :] = center - (width / 2)  # Set left positions
    y_mlc[:, 1, :, :] = center + (width / 2)  # Set right positions
    y_mlc = y_mlc.clone().detach().requires_grad_(True)

    fluence_map = fluence_map_layer(y_mlc)

    # Convert TensorFlow tensor to numpy if necessary
    if isinstance(fluence_map, torch.Tensor):
        fluence_map = fluence_map.cpu().detach().numpy()

    print(f"Test Case - Center: {center}, Width: {width}")
    print("Fluence Map Shape:", fluence_map.shape)

    ones = np.mean(fluence_map)  # Count pixels that are effectively one

    assert ones == pytest.approx(width, 0.1)

@pytest.mark.parametrize(
    "center, width",
    [
        (0.5, 0.0),
        (0.5, 0.5),
        (0.5, 1.0),
    ]
)
def test_fluence_map_jaws_open_per_width(fluence_map_layer, default_config, center, width):
    """Test that fluence map behaves correctly based on input width."""
    y_mlc = torch.zeros(default_config.shape_mlc[0], dtype=torch.float32, device=default_config.device)
    y_mlc[:, 0, :, :] = 0.0
    y_mlc[:, 1, :, :] = 1.0
    y_mlc = y_mlc.clone().detach().requires_grad_(True)
    y_jaws = torch.zeros(default_config.shape_jaws, dtype=torch.float32, device=default_config.device)
    y_jaws[:, 0, :] = center - (width / 2)  # Set bottom positions
    y_jaws[:, 1, :] = center + (width / 2)  # Set top positions
    y_jaws = y_jaws.clone().detach().requires_grad_(True)

    fluence_map = fluence_map_layer(y_mlc, y_jaws)

    # Convert TensorFlow tensor to numpy if necessary
    if isinstance(fluence_map, torch.Tensor):
        fluence_map = fluence_map.cpu().detach().numpy()

    print(f"Test Case - Center: {center}, Width: {width}")
    print("Fluence Map Shape:", fluence_map.shape)

    ones = np.mean(fluence_map)  # Count pixels that are effectively one

    assert ones == pytest.approx(width, 0.1)

def test_fluence_map_output_shape(fluence_map_layer, default_config):
    """Test that fluence map behaves correctly based on input width."""
    y_mlc = torch.zeros(default_config.shape_mlc[0], dtype=torch.float32, device=default_config.device)

    fluence_map = fluence_map_layer(y_mlc)

    assert fluence_map.shape == default_config.shape_fluence_map, f"Expected shape {default_config.shape_fluence_map}, but got {fluence_map.shape}"

def test_fluence_map_leaves_gradients_closing(fluence_map_layer, default_config):
    """Test that fluence map behaves correctly based on input width."""
    y_mlc = torch.zeros(default_config.shape_mlc[0], dtype=torch.float32, device=default_config.device)
    y_mlc[:, 0, ...] = 0.0
    y_mlc[:, 1, ...] = 1.0
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

def test_fluence_map_leaves_gradients_opening(fluence_map_layer, default_config):
    """Test that fluence map behaves correctly based on input width."""
    y_mlc = torch.zeros(default_config.shape_mlc[0], dtype=torch.float32, device=default_config.device)
    y_mlc[:, 0, ...] = 0.5 # Set left positions
    y_mlc[:, 1, ...] = 0.5 # Set right positions
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
