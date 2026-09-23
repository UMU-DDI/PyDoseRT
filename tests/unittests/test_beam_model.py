"""What PencilDepthEngine adds: a radiological depth per pencil, and the machine's
electron contamination attenuated with it.
"""

import math

import pytest
import torch

from pydosert import DoseEngine, PencilDepthEngine
from pydosert.data import BeamSequence, MachineConfig
from pydosert.layers.PencilDepthLayer import PencilDepthLayer


def test_pencil_depth_in_water_is_the_midpoint_depth(default_device):
    """In a water slab every in-slab pencil's depth is (k + 1/2) * step, at any gantry angle."""
    H, D, W, step = 24, 40, 40, 2.0
    density = torch.ones(1, H, D, W, device=default_device)
    iso = ((H - 1) * step / 2, (D - 1) * step / 2, (W - 1) * step / 2)   # voxel k at k * step
    layer = PencilDepthLayer((H, D, W), iso, (step,) * 3, torch.tensor([0.0, math.pi]),
                             device=default_device)
    depth = layer(density)                               # [G, D, H, W]
    expect = (torch.arange(D, device=default_device) + 0.5) * step
    for g in range(2):
        torch.testing.assert_close(depth[g, :, H // 2, W // 2], expect, atol=1e-3, rtol=0)


def _water(engine_cls, device, contamination=None, **kwargs):
    """Central-axis-plane dose of a 10 x 10 field in a water slab, 10 cm deep isocentre."""
    cfg = MachineConfig(preset="varian_10MV")
    cfg.electron_contamination = contamination
    shape, res = (80, 90, 80), (2.0, 2.0, 2.0)
    iso = (80.0, 101.0, 80.0)                            # water from voxel 1, 10 cm deep
    water = torch.zeros(1, *shape, device=device)
    water[:, :, 1:, :] = 1.0
    seq = BeamSequence.create([0.0], cfg.number_of_leaf_pairs, (400, 400), iso,
                              open_field_size=100.0, device=device, dtype=torch.float32,
                              requires_grad=False)
    engine = engine_cls(machine_config=cfg, kernel_size=51, dose_grid_spacing=res,
                        dose_grid_shape=shape, beam_template=seq, device=device,
                        dtype=torch.float32, conv_backend="fft", **kwargs)
    engine.calibrate(verbose=False)
    with torch.no_grad():
        return engine.compute_dose(seq, density_image=water)[0][:, :, 40]      # [H, D]


def test_contamination_follows_the_per_pencil_depth(default_device):
    """The machine's term is added by PencilDepthEngine too, attenuated per pencil."""
    off = _water(PencilDepthEngine, default_device)[40]
    on = _water(PencilDepthEngine, default_device, contamination=[1.2, 60.0, 9.5])[40]
    assert float(on[1] / off[1]) > 1.05                             # 1 mm deep
    assert float(on[51] / off[51]) == pytest.approx(1.0, abs=1e-4)  # 10 cm


def test_pencil_depth_matches_central_depth_on_a_flat_slab(default_device):
    """At normal incidence on flat water every pencil's depth is the central axis's, so
    the two engines agree up to the depth-node interpolation (0.2%)."""
    central = _water(DoseEngine, default_device)
    pencil = _water(PencilDepthEngine, default_device)
    field = central > 0.2 * float(central.max())
    field[:, :2] = False                                 # the first voxel sits on the surface
    assert float(((pencil - central).abs()[field] / central[field]).max()) < 0.005


def test_pencil_depth_engine_needs_the_fft_backend(default_device):
    with pytest.raises(ValueError, match="fft"):
        PencilDepthEngine(machine_config=MachineConfig(preset="varian_10MV"), kernel_size=25,
                          dose_grid_spacing=(2.0,) * 3, dose_grid_shape=(20, 20, 20),
                          device=default_device, conv_backend="direct")
