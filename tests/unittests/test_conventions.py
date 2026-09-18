"""The axis conventions in :mod:`pydosert.geometry.conventions` are executable.

Each test here pins one sentence of that document against the code that
implements it, so the document cannot quietly drift away from the engine.

The half-voxel tests matter most: the fluence projection, the
radiological-depth ray caster and the beam rotation each convert the isocentre
from mm to voxels independently, and all three must land on iso / resolution.
The ray caster used to add +0.5, which offset the depth ray from the beam axis
whose radiological depth it measures.
"""

import math

import pytest
import torch

from pydosert import DoseEngine
from pydosert.data import BeamSequence, MachineConfig
from pydosert.geometry.conventions import CONVENTIONS
from pydosert.geometry.rotations import (
    build_rotation_grids,
    get_radiological_depth_indices,
)

SHAPE = (64, 64, 64)          # (H, D, W)
SPACING = (2.0, 2.0, 2.0)     # (res_H, res_D, res_W)
ISO_MM = (64.0, 64.0, 64.0)
#: Where every layer puts the isocentre: iso / spacing, with no half-voxel shift.
ISO_VOXEL = 32.0


def _machine_config() -> MachineConfig:
    return MachineConfig(preset="test", number_of_leaf_pairs=10, tpr_20_10=0.72)


def _centroid(volume: torch.Tensor, axis: int) -> float:
    """Intensity-weighted mean index of ``volume`` along ``axis``."""
    others = [a for a in range(volume.ndim) if a != axis]
    profile = volume.sum(dim=others)
    index = torch.arange(profile.shape[0], device=profile.device, dtype=profile.dtype)
    return float((profile * index).sum() / profile.sum())


def _dose(angles_deg, device, dtype=torch.float32) -> torch.Tensor:
    """Dose of an open field in uniform water, returned as [H, D, W]."""
    config = _machine_config()
    sequence = BeamSequence.create(
        gantry_angles_deg=angles_deg,
        number_of_leaf_pairs=config.number_of_leaf_pairs,
        field_size=(100, 100),
        iso_center=ISO_MM,
        sid=1000.0,
        device=device,
        dtype=dtype,
    )
    engine = DoseEngine(
        machine_config=config,
        kernel_size=15,
        dose_grid_spacing=SPACING,
        dose_grid_shape=SHAPE,
        beam_template=sequence,
        device=device,
        dtype=dtype,
    )
    water = torch.ones((1, *SHAPE), device=device, dtype=dtype)
    return engine.compute_dose(sequence, density_image=water).detach()[0]


# ------------------------------------------------- the isocentre, in three layers


def test_rotation_grid_centres_on_iso_over_spacing():
    """build_rotation_grids rotates about iso / spacing, with no half-voxel shift."""
    grid = build_rotation_grids(
        (1, 1, SHAPE[1], SHAPE[0], SHAPE[2]),
        torch.tensor([math.pi], dtype=torch.float64),
        "cpu",
        torch.float64,
        iso_center=ISO_MM,
        resolution=SPACING,
    )[0, 0, 0]

    depth, width = SHAPE[1], SHAPE[2]
    out_d = torch.arange(depth, dtype=torch.float64).view(depth, 1).expand(depth, width)
    out_w = torch.arange(width, dtype=torch.float64).view(1, width).expand(depth, width)
    # align_corners=False: normalised n maps back to index (n + 1) * size / 2 - 0.5.
    in_w = (grid[..., 0] + 1.0) * width / 2.0 - 0.5
    in_d = (grid[..., 1] + 1.0) * depth / 2.0 - 0.5

    # A 180 degree rotation about c sends index i to 2c - i, so (in + out) / 2 == c.
    assert float(((in_d + out_d) / 2).mean()) == pytest.approx(ISO_VOXEL, abs=1e-6)
    assert float(((in_w + out_w) / 2).mean()) == pytest.approx(ISO_VOXEL, abs=1e-6)


def test_radiological_depth_ray_runs_along_the_beam_axis():
    """The depth ray sits at iso / spacing, the same place as the beam axis.

    It used to sit half a voxel further along every axis, so it measured
    radiological depth for a line beside the beam rather than along it.
    """
    points = get_radiological_depth_indices(
        SHAPE,
        torch.tensor([0.0], dtype=torch.float64),
        torch.float64,
        iso_center=ISO_MM,
        resolution=SPACING,
    )[0, 0]                       # [D, 3], last axis ordered (x=W, y=D, z=H)

    assert float(points[:, 0].mean()) == pytest.approx(ISO_VOXEL, abs=1e-6)   # W
    assert float(points[:, 2].mean()) == pytest.approx(ISO_VOXEL, abs=1e-6)   # H


def test_beam_axis_lands_on_the_isocentre(default_device):
    """End to end, an open field at gantry 0 is centred on iso / spacing laterally.

    This covers the third conversion, the one inside FluenceVolumeLayer.
    """
    dose = _dose([0.0], default_device)
    assert _centroid(dose, 0) == pytest.approx(ISO_VOXEL, abs=0.05)   # H
    assert _centroid(dose, 2) == pytest.approx(ISO_VOXEL, abs=0.05)   # W


# ------------------------------------------------------------- gantry rotation


def test_gantry_rotates_in_the_depth_width_plane_only(default_device):
    """H is untouched by the gantry angle; only D and W see the rotation."""
    height_centroids = [_centroid(_dose([angle], default_device), 0) for angle in (0.0, 90.0, 180.0, 270.0)]
    for centroid in height_centroids:
        assert centroid == pytest.approx(height_centroids[0], abs=1e-3)


@pytest.mark.parametrize(
    "angle_deg, axis, entering_side",
    [
        (0.0, 1, "low"),     # gantry 0: travels +D, so it enters at low D
        (180.0, 1, "high"),
        (90.0, 2, "high"),   # gantry 90: travels -W, so it enters at high W
        (270.0, 2, "low"),
    ],
)
def test_beam_direction_matches_cos_minus_sin(angle_deg, axis, entering_side, default_device):
    """At gantry theta the beam runs along (cos theta, -sin theta) in (D, W).

    Attenuation puts more dose near the entrance, so the centroid sits on the
    side the beam enters from.
    """
    centroid = _centroid(_dose([angle_deg], default_device), axis)
    if entering_side == "low":
        assert centroid < ISO_VOXEL
    else:
        assert centroid > ISO_VOXEL


# ------------------------------------------------------------------- document


def test_conventions_document_states_the_axis_order():
    """The printable summary names the layout and directions the code actually uses."""
    assert "(H, D, W)" in CONVENTIONS
    assert "(cos theta, -sin theta)" in CONVENTIONS


def test_conventions_document_states_the_isocentre_rule():
    """The summary states the single mm-to-voxel rule every layer follows."""
    assert "iso_center / dose_grid_spacing" in CONVENTIONS
    assert "KNOWN DISCREPANCY" not in CONVENTIONS
