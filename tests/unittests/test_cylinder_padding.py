import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
import math

import numpy as np
import pytest
import torch

from pydosert.data.loaders import body_cylinder_radius_mm, crop_from_cylinder, pad_to_cylinder

RES = (3.0, 2.0, 2.5)          # (res_H, res_D, res_W), deliberately anisotropic


def _inscribed_radius_mm(shape, iso_center, res):
    """Distance from the isocentre to the nearest D/W edge of the grid."""
    _, D, W = shape
    return min(iso_center[1], D * res[1] - iso_center[1], iso_center[2], W * res[2] - iso_center[2])


@pytest.mark.parametrize("iso", [(30.0, 40.0, 25.0), (30.0, 150.0, 190.0), (30.0, 100.0, 125.0)])
def test_pad_centres_isocentre(iso):
    volume = torch.rand(20, 100, 100)
    padded, new_iso, _ = pad_to_cylinder(volume, RES, iso)
    _, D, W = padded.shape
    assert abs(new_iso[1] / RES[1] - D / 2) <= 0.5
    assert abs(new_iso[2] / RES[2] - W / 2) <= 0.5
    assert new_iso[0] == iso[0]
    assert padded.shape[0] == volume.shape[0]


def test_pad_covers_requested_radius():
    volume = torch.rand(20, 60, 80)
    iso = (30.0, 40.0, 25.0)                      # off centre: one side is short
    padded, new_iso, _ = pad_to_cylinder(volume, RES, iso, radius_mm=180.0)
    assert _inscribed_radius_mm(padded.shape, new_iso, RES) >= 180.0


def test_round_trip_is_exact_for_tensors_arrays_and_masks():
    ct = torch.randn(10, 50, 70)
    mask = torch.rand(10, 50, 70) > 0.5
    dose = np.random.rand(10, 50, 70)
    padded, _, info = pad_to_cylinder([ct, mask, dose], RES, (15.0, 30.0, 50.0),
                                      radius_mm=120.0, fill_value=[-1000.0, 0, 0.0])
    assert padded[1].dtype == torch.bool
    restored = crop_from_cylinder(padded, info)
    assert torch.equal(restored[0], ct)
    assert torch.equal(restored[1], mask)
    assert np.array_equal(restored[2], dose)


def test_fill_values_are_per_volume():
    ct = torch.zeros(4, 10, 10)
    dose = torch.ones(4, 10, 10)
    (ct_p, dose_p), _, _ = pad_to_cylinder((ct, dose), RES, (6.0, 0.0, 0.0),
                                              fill_value=(-1000.0, 0.0))
    assert ct_p[0, 0, 0] == -1000.0                # isocentre at the corner forces padding
    assert dose_p[0, 0, 0] == 0.0


def test_body_radius_of_a_disk():
    D, W = 120, 120
    iso = (0.0, 60.0 * RES[1], 60.0 * RES[2])
    d = (np.arange(D) + 0.5) * RES[1] - iso[1]
    w = (np.arange(W) + 0.5) * RES[2] - iso[2]
    r = np.sqrt(d[:, None] ** 2 + w[None, :] ** 2)
    ct = np.full((3, D, W), -1000.0)
    ct[:, r <= 90.0] = 0.0
    assert math.isclose(body_cylinder_radius_mm(ct, RES, iso), 90.0, abs_tol=max(RES))
    assert body_cylinder_radius_mm(np.full((3, D, W), -1000.0), RES, iso) == 0.0


def test_padded_body_survives_rotation():
    """The point of the padding: no body voxel may lie outside the circle that
    every rotation about the isocentre keeps inside the array."""
    ct = np.full((2, 60, 60), -1000.0)
    ct[:, 2:20, 40:58] = 0.0                       # body tucked into a corner
    iso = (3.0, 30.0 * RES[1], 30.0 * RES[2])
    radius = body_cylinder_radius_mm(ct, RES, iso)
    padded, new_iso, _ = pad_to_cylinder(ct, RES, iso, radius_mm=radius, fill_value=-1000.0)
    assert radius > _inscribed_radius_mm(ct.shape, iso, RES)          # it would have been clipped
    assert _inscribed_radius_mm(padded.shape, new_iso, RES) >= radius
