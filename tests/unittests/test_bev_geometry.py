"""Tests for the beam's-eye-view sampling grids and BEV crops.

Self-contained: everything is analytic, nothing is read from outside the
repository, and no dose engine is constructed.

The geometries mirror the ones the ion regression suite uses -- an isotropic
square grid, an anisotropic non-square grid (rh != rd != rw, H != D != W) and a
non-square isotropic grid -- because that is exactly where two different
rotation-grid conventions can agree on a cube and disagree everywhere else.
"""

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F

from pydosert.geometry.bev import (
    BevCrop,
    build_bev_crop,
    build_bev_sampling_grid,
    build_patient_sampling_grid,
    crop_slices,
    normalize_iso_centers,
)

# (H, D, W), (rh, rd, rw), iso mm, (field_h, field_w)
STD = ((48, 150, 48), (2.0, 2.0, 2.0), (47.0, 150.0, 47.0), (32, 32))
ANISO = ((44, 150, 52), (1.5, 2.0, 2.5), (32.25, 150.0, 63.75), (26, 30))
NONSQ = ((40, 150, 56), (2.0, 2.0, 2.0), (39.0, 150.0, 55.0), (28, 34))

GEOMETRIES = {"std": STD, "aniso": ANISO, "nonsq": NONSQ}

BUILDERS = (build_bev_sampling_grid, build_patient_sampling_grid)


def _rad(*degrees):
    return [math.radians(float(d)) for d in degrees]


def _denorm(grid, shape):
    """Undo the align_corners=False normalisation: (x, y) -> (w, d) voxel indices."""
    _H, D, W = shape
    x = ((grid[..., 0] + 1.0) * W - 1.0) * 0.5
    y = ((grid[..., 1] + 1.0) * D - 1.0) * 0.5
    return x, y


# --------------------------------------------------------------------------- shape / dtype


@pytest.mark.parametrize("name", list(GEOMETRIES))
@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_grid_shape_and_dtype(name, builder, dtype):
    shape, spacing, iso, _field = GEOMETRIES[name]
    angles = _rad(0, 45, 90, 270)
    grid = builder(shape, spacing, angles, iso, dtype=dtype)
    assert grid.shape == (len(angles), shape[1], shape[2], 2)
    assert grid.dtype is dtype
    assert torch.isfinite(grid).all()


# --------------------------------------------------------------------------- iso centres


@pytest.mark.parametrize("name", list(GEOMETRIES))
@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_shared_iso_equals_repeated_per_beam_iso(name, builder):
    """A shared (3,) centre must be bit-identical to the same centre repeated (G, 3)."""
    shape, spacing, iso, _field = GEOMETRIES[name]
    angles = _rad(0, 45, 90, 270)
    shared = builder(shape, spacing, angles, iso, dtype=torch.float64)
    per_beam = builder(
        shape, spacing, angles, [list(iso)] * len(angles), dtype=torch.float64
    )
    assert torch.equal(shared, per_beam)


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_per_beam_iso_is_used_per_beam(builder):
    """Beam g must see only row g of a per-beam isocentre."""
    shape, spacing, iso, _field = ANISO
    angles = _rad(0, 45, 90, 270)
    rows = [[iso[0] + 3.0 * k, iso[1] - 7.5 * k, iso[2] + 5.25 * k] for k in range(4)]
    per_beam = builder(shape, spacing, angles, rows, dtype=torch.float64)

    for g in range(4):
        alone = builder(shape, spacing, [angles[g]], rows[g], dtype=torch.float64)
        assert torch.equal(per_beam[g], alone[0])

    # ... and the rows really do differ, so the check above is not vacuous.
    assert not torch.equal(per_beam[0], per_beam[1])


def test_normalize_iso_centers_broadcasts():
    got = normalize_iso_centers((1.0, 2.0, 3.0), 3, torch.device("cpu"), torch.float64)
    assert got.shape == (3, 3)
    assert torch.equal(got, torch.tensor([[1.0, 2.0, 3.0]] * 3, dtype=torch.float64))


# --------------------------------------------------------------------------- gantry 0 identity


@pytest.mark.parametrize("name", list(GEOMETRIES))
@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_gantry_zero_is_the_identity_map(name, builder):
    """At gantry 0 the BEV frame is the patient frame, for both directions.

    Depth is measured from where the central ray enters the grid, which at
    gantry 0 is d = 0, so the depth axis maps index-for-index; the lateral axis
    is measured from the isocentre and then re-referenced to it, so it maps
    index-for-index too.
    """
    shape, spacing, iso, _field = GEOMETRIES[name]
    _H, D, W = shape
    grid = builder(shape, spacing, [0.0], iso, dtype=torch.float64)
    x, y = _denorm(grid[0], shape)

    expect_x = torch.arange(W, dtype=torch.float64).view(1, W).expand(D, W)
    expect_y = torch.arange(D, dtype=torch.float64).view(D, 1).expand(D, W)
    assert torch.allclose(x, expect_x, atol=1e-9)
    assert torch.allclose(y, expect_y, atol=1e-9)


def test_gantry_zero_depth_origin_is_grid_entry_not_isocenter():
    """Moving the isocentre along the beam axis must not shift the depth mapping."""
    shape, spacing, iso, _field = ANISO
    a = build_bev_sampling_grid(shape, spacing, [0.0], iso, dtype=torch.float64)
    moved = (iso[0], iso[1] - 60.0, iso[2])
    b = build_bev_sampling_grid(shape, spacing, [0.0], moved, dtype=torch.float64)
    assert torch.equal(a, b)


def test_gantry_zero_grid_sample_returns_the_volume():
    """The identity claim, exercised through grid_sample itself."""
    shape, spacing, iso, _field = NONSQ
    H, D, W = shape
    vol = torch.arange(H * D * W, dtype=torch.float64).reshape(H, D, W) / (H * D * W)
    grid = build_bev_sampling_grid(shape, spacing, [0.0], iso, dtype=torch.float64)[0]
    out = F.grid_sample(
        vol.unsqueeze(1),
        grid.unsqueeze(0).expand(H, -1, -1, -1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(1)
    assert torch.allclose(out, vol, atol=1e-12)


# --------------------------------------------------------------------------- rotation behaviour


@pytest.mark.parametrize("name", list(GEOMETRIES))
def test_ninety_degrees_swaps_the_axes(name):
    """At gantry 90 the BEV depth axis runs along patient -W."""
    shape, spacing, iso, _field = GEOMETRIES[name]
    _H, D, W = shape
    _rh, rd, rw = spacing
    grid = build_bev_sampling_grid(shape, spacing, _rad(90.0), iso, dtype=torch.float64)
    x, y = _denorm(grid[0], shape)

    # Entry is the +W face, at the isocentre's depth; each BEV depth step of
    # rd mm walks rd/rw voxels in -W, and the BEV lateral axis runs along +D.
    d_idx = torch.arange(D, dtype=torch.float64).view(D, 1)
    w_idx = torch.arange(W, dtype=torch.float64).view(1, W)
    expect_x = (W - 1) - d_idx * (rd / rw)
    expect_y = (iso[1] + w_idx * rw - iso[2]) / rd
    assert torch.allclose(x, expect_x.expand(D, W), atol=1e-9)
    assert torch.allclose(y, expect_y.expand(D, W), atol=1e-9)


@pytest.mark.parametrize("name", list(GEOMETRIES))
@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 45.0, 90.0, 180.0, 270.0])
def test_round_trip_patient_bev_patient(name, angle_deg):
    """patient -> BEV -> patient reproduces a smooth field where BEV covers it."""
    shape, spacing, iso, _field = GEOMETRIES[name]
    H, D, W = shape
    rh, rd, rw = spacing

    h = torch.arange(H, dtype=torch.float64).view(H, 1, 1) * rh
    d = torch.arange(D, dtype=torch.float64).view(1, D, 1) * rd
    w = torch.arange(W, dtype=torch.float64).view(1, 1, W) * rw
    vol = (
        1.0
        + 0.3 * torch.sin(d / 400.0)
        + 0.2 * torch.cos(w / 300.0)
        + 0.1 * torch.sin(h / 250.0)
    ).expand(H, D, W).contiguous()

    to_bev = build_bev_sampling_grid(shape, spacing, _rad(angle_deg), iso, dtype=torch.float64)[0]
    to_patient = build_patient_sampling_grid(shape, spacing, _rad(angle_deg), iso, dtype=torch.float64)[0]

    def _resample(v, grid):
        return F.grid_sample(
            v.unsqueeze(1),
            grid.unsqueeze(0).expand(H, -1, -1, -1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(1)

    back = _resample(_resample(vol, to_bev), to_patient)
    # Coverage: where the same double resampling of a constant field survives, the
    # round trip is a real round trip rather than an edge/padding artefact.
    covered = _resample(_resample(torch.ones_like(vol), to_bev), to_patient) > 1.0 - 1e-9
    assert covered.any(), "the BEV frame covers no patient voxel at all"
    assert torch.allclose(back[covered], vol[covered], atol=2e-3)


def test_the_two_grids_are_different_maps_off_axis():
    """Guard against wiring both engine consumers to the same grid."""
    shape, spacing, iso, _field = ANISO
    angles = _rad(45.0)
    a = build_bev_sampling_grid(shape, spacing, angles, iso, dtype=torch.float64)
    b = build_patient_sampling_grid(shape, spacing, angles, iso, dtype=torch.float64)
    assert not torch.allclose(a, b)


def test_anisotropic_spacing_actually_changes_the_grid():
    """rh/rd/rw must reach the grid; a spacing-blind implementation fails here."""
    shape, _spacing, iso, _field = ANISO
    angles = _rad(45.0)
    a = build_bev_sampling_grid(shape, (1.5, 2.0, 2.5), angles, iso, dtype=torch.float64)
    b = build_bev_sampling_grid(shape, (2.0, 2.0, 2.0), angles, iso, dtype=torch.float64)
    assert not torch.allclose(a, b)


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_bad_iso_center_raises(builder):
    shape, spacing, _iso, _field = STD
    angles = _rad(0, 90)
    with pytest.raises(ValueError, match="iso_center_mm must be shape"):
        builder(shape, spacing, angles, (1.0, 2.0))
    with pytest.raises(ValueError, match="iso_center_mm must be shape"):
        builder(shape, spacing, angles, [[1.0, 2.0, 3.0]] * 3)
    with pytest.raises(ValueError, match="iso_center_mm must be shape"):
        builder(shape, spacing, angles, [[1.0, 2.0, 3.0, 4.0]] * 2)


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_missing_iso_center_raises_instead_of_defaulting(builder):
    shape, spacing, _iso, _field = STD
    with pytest.raises(ValueError, match="iso_center_mm is required"):
        builder(shape, spacing, _rad(0.0), None)


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_bad_geometry_raises(builder):
    _shape, spacing, iso, _field = STD
    with pytest.raises(ValueError, match="grid_shape must be"):
        builder((48, 150), spacing, _rad(0.0), iso)
    with pytest.raises(ValueError, match="grid_shape must be positive"):
        builder((48, 0, 48), spacing, _rad(0.0), iso)
    with pytest.raises(ValueError, match="spacing_mm must be"):
        builder((48, 150, 48), (2.0, 2.0), _rad(0.0), iso)
    with pytest.raises(ValueError, match="spacing_mm must be strictly positive"):
        builder((48, 150, 48), (2.0, 0.0, 2.0), _rad(0.0), iso)


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_bad_angles_raise(builder):
    shape, spacing, iso, _field = STD
    with pytest.raises(ValueError, match="1-D sequence"):
        builder(shape, spacing, [[0.0], [1.0]], iso)
    with pytest.raises(ValueError, match="at least one gantry angle"):
        builder(shape, spacing, [], iso)


def test_bad_full_shape_raises():
    with pytest.raises(ValueError, match="full_shape_hw must be"):
        build_bev_crop(0.0, 0.0, 4, 4, (48, 150, 48))
    with pytest.raises(ValueError, match="full_shape_hw must be positive"):
        build_bev_crop(0.0, 0.0, 4, 4, (0, 48))


# --------------------------------------------------------------------------- crop slices


def test_crop_slices_centred():
    src, dst, target = crop_slices(24.0, 48, 32)
    assert (target.start, target.stop) == (8, 40)
    assert (src.start, src.stop) == (8, 40)
    assert (dst.start, dst.stop) == (0, 32)


@pytest.mark.parametrize(
    "center, expected_center_i",
    [
        (-11.5, -12),   # ties-to-even goes DOWN here; int(x + 0.5) would give -11
        (-10.5, -10),
        (-0.5, 0),
        (0.5, 0),
        (1.5, 2),
        (2.5, 2),       # int(x + 0.5) would give 3
        (3.5, 4),
        (11.5, 12),
        (12.5, 12),     # int(x + 0.5) would give 13
        (23.5, 24),
        (24.5, 24),
        (42.5, 42),     # int(x + 0.5) would give 43
    ],
)
def test_crop_slices_uses_bankers_rounding(center, expected_center_i):
    """Half-integer centres are the normal case (a spot on a voxel boundary).

    Python's round() breaks those ties to even, and the crop window inherits
    that. int(x + 0.5), numpy.round on a different dtype and torch.round are all
    observably different here.
    """
    crop_size = 32
    _src, _dst, target = crop_slices(center, 1000, crop_size)
    assert target.start == expected_center_i - crop_size // 2
    assert target.stop == target.start + crop_size


def test_bankers_rounding_reaches_the_crop():
    """The regression geometry that sits exactly on the tie: centre -11.5 on 48 voxels."""
    crop = build_bev_crop(center_h=23.5, center_w=-11.5, size_h=32, size_w=32, full_shape_hw=(48, 48))
    # round(-11.5) == -12 -> window [-28, 4); the ties-away-from-zero answer -11
    # would have produced [-27, 5) and one more column of dose.
    assert crop.target_w_start == -28
    assert (crop.w_src.start, crop.w_src.stop) == (0, 4)
    assert (crop.w_dst.start, crop.w_dst.stop) == (28, 32)


def test_crop_clipped_at_the_low_edge():
    crop = build_bev_crop(center_h=4.0, center_w=24.0, size_h=32, size_w=32, full_shape_hw=(48, 48))
    assert crop.target_h_start == -12
    assert (crop.h_src.start, crop.h_src.stop) == (0, 20)
    assert (crop.h_dst.start, crop.h_dst.stop) == (12, 32)
    assert crop.h_src.stop - crop.h_src.start == crop.h_dst.stop - crop.h_dst.start
    assert not crop.is_empty


def test_crop_clipped_at_the_high_edge():
    crop = build_bev_crop(center_h=46.0, center_w=24.0, size_h=32, size_w=32, full_shape_hw=(48, 48))
    assert crop.target_h_start == 30
    assert (crop.h_src.start, crop.h_src.stop) == (30, 48)
    assert (crop.h_dst.start, crop.h_dst.stop) == (0, 18)
    assert not crop.is_empty


@pytest.mark.parametrize("center_h", [-200.0, 248.0])
def test_crop_entirely_outside_is_empty_not_an_error(center_h):
    crop = build_bev_crop(center_h=center_h, center_w=24.0, size_h=32, size_w=32, full_shape_hw=(48, 48))
    assert crop.h_is_empty
    assert crop.h_src.stop <= crop.h_src.start
    assert crop.h_dst.stop <= crop.h_dst.start
    assert crop.is_empty
    assert not crop.w_is_empty


def test_fully_outside_crop_can_carry_a_negative_stop():
    """Sharp edge, preserved from the original: ``src.stop`` may be negative.

    A window entirely below the axis gives ``slice(0, negative)``, which Python
    would happily interpret as "up to N from the end" -- a large, non-empty
    slice. Callers must test emptiness with ``stop <= start`` (or
    :attr:`BevCrop.is_empty`) BEFORE slicing anything with it.
    """
    crop = build_bev_crop(center_h=-60.0, center_w=24.0, size_h=32, size_w=32, full_shape_hw=(48, 48))
    assert (crop.h_src.start, crop.h_src.stop) == (0, -44)
    assert crop.h_is_empty
    assert len(list(range(48))[crop.h_src]) == 4  # ... which is exactly the trap


def test_crop_size_is_clamped_into_the_grid():
    """Historical behaviour, preserved deliberately: an oversized field is silently clamped."""
    crop = build_bev_crop(center_h=24.0, center_w=24.0, size_h=999, size_w=999, full_shape_hw=(48, 52))
    assert crop.shape_hw == (48, 52)
    assert crop.full_shape_hw == (48, 52)
    assert (crop.h_src.start, crop.h_src.stop) == (0, 48)

    zero = build_bev_crop(center_h=24.0, center_w=24.0, size_h=0, size_w=-5, full_shape_hw=(48, 52))
    assert zero.shape_hw == (1, 1)


@pytest.mark.parametrize("name", list(GEOMETRIES))
def test_crop_invariants_over_a_sweep(name):
    """src/dst lengths agree, dst stays inside the window, target keeps the size."""
    shape, spacing, iso, field = GEOMETRIES[name]
    H, _D, W = shape
    rh, _rd, rw = spacing
    for center_h in [-60.0, -11.5, 0.0, 0.5, iso[0] / rh, H - 0.5, float(H), H + 60.0]:
        for center_w in [-60.0, -11.5, 0.0, iso[2] / rw, W - 0.5, float(W), W + 60.0]:
            crop = build_bev_crop(center_h, center_w, field[0], field[1], (H, W))
            assert crop.shape_hw == (field[0], field[1])
            assert crop.full_shape_hw == (H, W)
            assert crop.h_target.stop - crop.h_target.start == field[0]
            assert crop.w_target.stop - crop.w_target.start == field[1]
            assert crop.target_h_start == crop.h_target.start
            assert crop.target_w_start == crop.w_target.start
            for src, dst, size, full in (
                (crop.h_src, crop.h_dst, field[0], H),
                (crop.w_src, crop.w_dst, field[1], W),
            ):
                assert max(src.stop - src.start, 0) == max(dst.stop - dst.start, 0)
                if src.stop <= src.start:
                    continue  # empty crop: see test_fully_outside_crop_*
                assert 0 <= src.start <= full and 0 <= src.stop <= full
                assert 0 <= dst.start <= size and 0 <= dst.stop <= size


def test_crop_is_an_immutable_dataclass():
    crop = build_bev_crop(24.0, 24.0, 8, 8, (48, 48))
    assert isinstance(crop, BevCrop)
    with pytest.raises(dataclasses.FrozenInstanceError):
        crop.target_h_start = 0
