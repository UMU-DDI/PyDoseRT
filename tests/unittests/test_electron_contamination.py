"""The contamination term raises the build-up only, and only when the machine has it."""

import pytest
import torch

from pydosert import DoseEngine
from pydosert.data import BeamSequence, MachineConfig


def _water_cax(contamination, device):
    cfg = MachineConfig(preset="varian_10MV")
    cfg.electron_contamination = contamination
    shape, res = (80, 90, 80), (2.0, 2.0, 2.0)
    iso = (80.0, 101.0, 80.0)                       # water from voxel 1, 10 cm deep
    water = torch.zeros(1, *shape, device=device)
    water[:, :, 1:, :] = 1.0
    seq = BeamSequence.create([0.0], cfg.number_of_leaf_pairs, (400, 400), iso,
                              open_field_size=100.0, device=device, dtype=torch.float32,
                              requires_grad=False)
    engine = DoseEngine(machine_config=cfg, kernel_size=25, dose_grid_spacing=res,
                        dose_grid_shape=shape, beam_template=seq, device=device,
                        dtype=torch.float32)
    engine.calibrate(verbose=False)
    with torch.no_grad():
        return engine.compute_dose(seq, density_image=water)[0][40, :, 40]


def test_contamination_raises_the_build_up_and_not_the_reference_depth(default_device):
    off = _water_cax(None, default_device)
    on = _water_cax([1.2, 60.0, 9.5], default_device)
    assert float(on[1] / off[1]) > 1.05                             # 1 mm deep
    assert float(on[51] / off[51]) == pytest.approx(1.0, abs=1e-4)  # 10 cm, the calibration depth


def test_shipped_presets_have_no_contamination():
    """None on every preset, so the engine computes the dose it did before."""
    from pydosert.data.machine_config import list_machine_presets
    for name in list_machine_presets():
        assert MachineConfig(preset=name).electron_contamination is None
