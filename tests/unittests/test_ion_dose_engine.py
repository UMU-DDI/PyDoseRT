"""Physics and contract tests for the ion BEV lattice dose engine.

Self-contained: the phantoms are analytic, the base data is the kernel table
shipped in the package, and nothing is read from outside the repository.

The tests here check *physics that must hold*, not values that happen to come
out: the depth dose must reproduce the commissioned integrated depth dose, a
denser slab must move the Bragg peak by its water-equivalent thickness, the
lateral kernel must conserve that integral at every depth, the per-beamlet
outputs must sum to the summed volume, a cylindrically symmetric phantom must
give a rotation-invariant result, and dose must be exactly linear in the beamlet
weight. Bit-level agreement with the previous implementation is pinned
separately by an out-of-tree regression capture.
"""

import math
from pathlib import Path

import pytest
import torch

from pydosert.data.ion_beam import IonBeamletBatch
from pydosert.data.ion_machine import IonMachineConfig
from pydosert.engine.ion_dose_engine import (
    MEV_CM2_PER_G_TO_GY_MM2,
    BeamletDose,
    IonDoseEngine,
    patient_dose_mask,
)
from pydosert.physics.ion_scattering import fermi_eyges_excess
from pydosert.physics.kernels.ion_kernel_table import IonKernelTable

TABLE_NPZ = (
    Path(__file__).parents[2] / "src" / "pydosert" / "data" / "machine_presets" / "protons_doserad.npz"
)

#: (H, D, W) voxels and (rh, rd, rw) mm. 300 mm deep, so a 120 MeV Bragg peak
#: (~110 mm) sits well inside the grid.
GRID = (32, 150, 32)
SPACING = (2.0, 2.0, 2.0)
#: Beam axis on the lateral voxel boundary at the grid centre, like the
#: water-phantom commissioning benchmark.
ISO = (31.0, 150.0, 31.0)
SAD_MM = 1000.0
WEIGHT = 1.0e7
SIGMA_MM = 4.6


@pytest.fixture(scope="module")
def table() -> IonKernelTable:
    """The packaged proton kernel table, float64 for a clean physics check."""
    return IonKernelTable.load(TABLE_NPZ, dtype=torch.float64)


@pytest.fixture(scope="module")
def energy(table: IonKernelTable) -> float:
    """A tabulated energy whose range is comfortably inside the test grid."""
    return min(table.available_energies, key=lambda e: abs(e - 120.0))


def make_engine(table: IonKernelTable, **kwargs) -> IonDoseEngine:
    """An engine on the default test geometry, overridable per test."""
    settings = dict(
        machine_config=IonMachineConfig(),
        kernel_table=table,
        dose_grid_spacing=SPACING,
        dose_grid_shape=GRID,
        field_size=(GRID[0], GRID[2]),
        lateral_model="gauss_double",
        heterogeneous_mcs=True,
    )
    settings.update(kwargs)
    return IonDoseEngine(**settings)


def make_beamlets(
    table: IonKernelTable,
    energy_mev: float,
    *,
    gantry_angle_deg=0.0,
    weight=WEIGHT,
    position_mm=(0.0, 0.0),
    iso_center_mm=ISO,
) -> IonBeamletBatch:
    """One or more beamlets on the default geometry."""
    angles = [float(gantry_angle_deg)] if not isinstance(gantry_angle_deg, (list, tuple)) else list(gantry_angle_deg)
    count = len(angles)
    return IonBeamletBatch.create(
        gantry_angle_deg=angles,
        position_mm=[list(position_mm)] * count,
        energy_mev=[float(energy_mev)] * count,
        sigma_mm=[[SIGMA_MM, SIGMA_MM]] * count,
        weight=[float(weight)] * count,
        iso_center_mm=list(iso_center_mm),
        sad_mm=SAD_MM,
        dtype=table.dtype,
    )


def water(dtype: torch.dtype, shape=GRID) -> torch.Tensor:
    """A uniform water phantom."""
    return torch.ones(shape, dtype=dtype)


def all_scored(shape=GRID) -> torch.Tensor:
    """A dose mask that scores every voxel."""
    return torch.ones(shape, dtype=torch.bool)


def integrated_depth_dose(dose: torch.Tensor, spacing=SPACING) -> torch.Tensor:
    """Undo the Gy conversion and integrate laterally: MeV per transport step."""
    res_h, res_d, res_w = spacing
    lateral_area_mm2 = (res_h * res_d * res_w) / res_d
    return dose[0].sum(dim=(0, 2)) * lateral_area_mm2 / MEV_CM2_PER_G_TO_GY_MM2


def depth_of_peak_mm(profile: torch.Tensor, res_d=SPACING[1]) -> float:
    """Depth in mm of the maximum of a per-voxel depth profile (voxel centres)."""
    return (float(torch.argmax(profile)) + 0.5) * res_d


# --------------------------------------------------------------------------- physics


def test_depth_dose_reproduces_the_kernel_table_curve(table, energy):
    """The laterally integrated dose in water is the table's IDD, at every depth.

    The lateral kernel is normalised to unit sum on every depth plane, so this
    is a statement about the engine's bookkeeping (splitting, halo weighting,
    Gy conversion) rather than about the kernel shape.
    """
    engine = make_engine(table)
    dose = engine.compute_dose(
        make_beamlets(table, energy), water(table.dtype), all_scored()
    )

    depths_mm = (torch.arange(GRID[1], dtype=table.dtype) + 0.5) * SPACING[1]
    kernel_depth = depths_mm - table.kernel_offset(energy)
    expected = WEIGHT * table.edep(energy, kernel_depth.clamp_min(0.0), beyond_range="edge")

    got = integrated_depth_dose(dose)
    inside_range = depths_mm < table.depth_max_mm(energy)
    assert torch.allclose(got[inside_range], expected[inside_range], rtol=1e-6)


def test_bragg_peak_lands_at_the_tabulated_peak_position(table, energy):
    """The on-axis depth dose peaks where the table's IDD peaks."""
    engine = make_engine(table)
    dose = engine.compute_dose(
        make_beamlets(table, energy), water(table.dtype), all_scored()
    )

    curves = table.row_curves(energy)
    tabulated_peak_mm = float(curves["depth_mm"][int(torch.argmax(curves["idd"]))])

    on_axis = dose[0, GRID[0] // 2, :, GRID[2] // 2]
    assert depth_of_peak_mm(on_axis) == pytest.approx(tabulated_peak_mm, abs=SPACING[1])
    assert depth_of_peak_mm(integrated_depth_dose(dose)) == pytest.approx(
        tabulated_peak_mm, abs=SPACING[1]
    )


def test_slab_shifts_the_peak_by_its_water_equivalent_thickness(table, energy):
    """A slab of density rho over thickness t pulls the peak forward by (rho-1)*t."""
    engine = make_engine(table)
    beamlets = make_beamlets(table, energy)
    mask = all_scored()

    water_peak = depth_of_peak_mm(
        integrated_depth_dose(engine.compute_dose(beamlets, water(table.dtype), mask))
    )

    density = water(table.dtype)
    first, last, rho = 10, 30, 1.5
    density[:, first:last, :] = rho
    thickness_mm = (last - first) * SPACING[1]
    expected_shift_mm = (rho - 1.0) * thickness_mm

    slab_peak = depth_of_peak_mm(
        integrated_depth_dose(engine.compute_dose(beamlets, density, mask))
    )
    assert water_peak - slab_peak == pytest.approx(expected_shift_mm, abs=SPACING[1])


def test_lateral_integral_conserves_the_idd_at_every_depth(table, energy):
    """Splitting into sub-beams and adding a halo must not move energy in depth.

    Compared against a single un-split reference: the *same* engine with one
    sub-beam and no halo has to integrate to the same depth profile.
    """
    reference = make_engine(table, lateral_model="gauss", n_sub_beams_per_dim=1)
    split_with_halo = make_engine(table, lateral_model="gauss_double", n_sub_beams_per_dim=9)
    beamlets = make_beamlets(table, energy)
    mask = all_scored()

    profile_reference = integrated_depth_dose(
        reference.compute_dose(beamlets, water(table.dtype), mask)
    )
    profile_split = integrated_depth_dose(
        split_with_halo.compute_dose(beamlets, water(table.dtype), mask)
    )
    assert torch.allclose(profile_split, profile_reference, rtol=1e-6)


def test_per_beamlet_doses_sum_to_the_summed_volume(table, energy):
    """The per-beamlet finaliser is the summed one, un-accumulated."""
    engine = make_engine(table)
    beamlets = IonBeamletBatch.create(
        gantry_angle_deg=[0.0, 45.0, 270.0],
        position_mm=[[0.0, 0.0], [6.0, -4.0], [-8.0, 10.0]],
        energy_mev=[energy] * 3,
        sigma_mm=[[SIGMA_MM, SIGMA_MM]] * 3,
        weight=[WEIGHT, 0.5 * WEIGHT, 2.0 * WEIGHT],
        iso_center_mm=list(ISO),
        sad_mm=SAD_MM,
        dtype=table.dtype,
    )
    density = water(table.dtype)
    mask = all_scored()

    summed = engine.compute_dose(beamlets, density, mask)
    per_beamlet = engine.compute_dose(beamlets, density, mask, return_per_beamlet=True)

    rebuilt = torch.zeros_like(summed)
    for item in per_beamlet:
        assert isinstance(item, BeamletDose)
        z, y, x = item.offset
        dz, dy, dx = item.dose.shape[1:]
        rebuilt[:, z : z + dz, y : y + dy, x : x + dx] += item.dose
    assert torch.allclose(rebuilt, summed, rtol=1e-10, atol=float(summed.max()) * 1e-12)


def test_rotation_invariance_on_a_cylindrical_phantom(table, energy):
    """Four cardinal gantry angles through a cylinder give the same dose, rotated."""
    shape = (16, 64, 64)
    spacing = (2.0, 2.0, 2.0)
    iso = (
        (shape[0] / 2 - 0.5) * spacing[0],
        (shape[1] / 2 - 0.5) * spacing[1],
        (shape[2] / 2 - 0.5) * spacing[2],
    )

    depth_mm = (torch.arange(shape[1], dtype=table.dtype) - (shape[1] / 2 - 0.5)) * spacing[1]
    width_mm = (torch.arange(shape[2], dtype=table.dtype) - (shape[2] / 2 - 0.5)) * spacing[2]
    radius_mm = torch.sqrt(depth_mm[:, None] ** 2 + width_mm[None, :] ** 2)
    density = torch.where(radius_mm < 50.0, 1.0, 0.2).to(table.dtype)
    density = density.unsqueeze(0).expand(shape[0], -1, -1).contiguous()

    engine = IonDoseEngine(
        machine_config=IonMachineConfig(),
        kernel_table=table,
        dose_grid_spacing=spacing,
        dose_grid_shape=shape,
        field_size=(shape[0], shape[2]),
        lateral_model="gauss_double",
        heterogeneous_mcs=True,
    )
    mask = torch.ones(shape, dtype=torch.bool)

    doses = {}
    for angle in (0.0, 90.0, 180.0, 270.0):
        beamlets = make_beamlets(table, energy, gantry_angle_deg=angle, iso_center_mm=iso)
        doses[angle] = engine.compute_dose(beamlets, density, mask)

    reference = doses[0.0]
    peak = float(reference.max())
    assert peak > 0.0
    for angle, dose in doses.items():
        # The gantry turns in the (D, W) plane, so a quarter turn of the gantry
        # is a quarter turn of the dose about the H axis.
        turns = int(angle // 90)
        rotated = torch.rot90(reference, k=-turns, dims=(2, 3))
        assert torch.allclose(dose, rotated, atol=peak * 1e-12), f"gantry {angle} deg"


def test_fermi_eyges_excess_vanishes_in_homogeneous_media(energy):
    """Geometric depth == WEQ depth => the lever-arm excess is identically zero."""
    steps = 200
    step_mm = 2.0
    weq = (torch.arange(steps, dtype=torch.float64) * step_mm).expand(5, steps).contiguous()
    excess = fermi_eyges_excess(weq, energy, step_mm)
    assert torch.equal(excess, torch.zeros_like(excess))

    # ...and it is strictly positive once the geometric path outruns the WEQ.
    lung = fermi_eyges_excess(0.3 * weq, energy, step_mm)
    assert float(lung.max()) > 0.0
    assert bool((lung >= 0.0).all())


def test_heterogeneous_mcs_is_a_no_op_in_water(table, energy):
    """Because the excess vanishes, the flag cannot move a water dose."""
    density, mask = water(table.dtype), all_scored()
    beamlets = make_beamlets(table, energy)
    off = make_engine(table, heterogeneous_mcs=False).compute_dose(beamlets, density, mask)
    on = make_engine(table, heterogeneous_mcs=True).compute_dose(beamlets, density, mask)
    assert torch.allclose(on, off, atol=float(off.max()) * 1e-9)


def test_dose_is_linear_in_the_beamlet_weight(table, energy):
    """Dose scales exactly with the weight; the tolerance sits below the dose."""
    engine = make_engine(table)
    density, mask = water(table.dtype), all_scored()
    single = engine.compute_dose(make_beamlets(table, energy, weight=WEIGHT), density, mask)
    tripled = engine.compute_dose(make_beamlets(table, energy, weight=3.0 * WEIGHT), density, mask)

    peak = float(single.max())
    assert peak > 1e-3, "the test would be vacuous if the dose were near zero"
    assert torch.allclose(tripled, 3.0 * single, rtol=1e-12, atol=peak * 1e-12)


def test_dose_past_the_range_is_the_documented_edge_pedestal(table, energy):
    """``beyond_range='edge'`` leaves a small constant tail distal of the peak.

    This is a known, documented artefact of the depth lookup, not a physical
    dose: the pedestal is what the commissioned corrections were fitted against
    and it is pinned here so that switching the lookup to ``'zero'`` is a
    deliberate, visible change.
    """
    engine = make_engine(table)
    dose = engine.compute_dose(make_beamlets(table, energy), water(table.dtype), all_scored())
    profile = integrated_depth_dose(dose)

    depths_mm = (torch.arange(GRID[1], dtype=table.dtype) + 0.5) * SPACING[1]
    past_range = depths_mm > table.depth_max_mm(energy) + 10.0
    tail = profile[past_range]
    assert tail.numel() > 0
    assert float(tail.min()) > 0.0, "an 'edge' lookup holds the last tabulated value"
    assert float(tail.max()) < 0.05 * float(profile.max())
    # It really is a constant pedestal, not a decaying tail.
    assert float(tail.max() - tail.min()) < 1e-9 * float(tail.max())


# --------------------------------------------------------------------------- hook


def test_bev_correction_hook_changes_the_result(table, energy):
    """The injection point is real: zeroing the BEV energy gives zero dose."""

    class Zero(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, payload, **_context):
            self.seen = dict(payload)
            return {**payload, "edep_bev": torch.zeros_like(payload["edep_bev"])}

    hook = Zero()
    engine = make_engine(table, bev_correction=hook)
    beamlets = make_beamlets(table, energy)
    dose = engine.compute_dose(beamlets, water(table.dtype), all_scored())

    assert float(dose.abs().max()) == 0.0
    # The payload contract the correction models rely on.
    for key in (
        "edep_bev",
        "density_bev",
        "weq_bev",
        "weq_depths",
        "density_image",
        "resolved_offset",
        "beamlets",
        "edep_to_gy",
        "bev_crop",
        "crop_centers_hw",
    ):
        assert key in hook.seen, key
    assert hook.seen["edep_bev"].shape == (1, 1, GRID[1], GRID[0], GRID[2])
    assert float(hook.seen["edep_bev"].abs().max()) > 0.0, "the hook must see uncorrected energy"

    without_hook = make_engine(table).compute_dose(beamlets, water(table.dtype), all_scored())
    assert float(without_hook.max()) > 0.0


def test_bev_correction_hook_scales_the_result(table, energy):
    """A hook that halves the BEV energy halves the dose."""

    class Half(torch.nn.Module):
        def forward(self, payload, **_context):
            return {**payload, "edep_bev": 0.5 * payload["edep_bev"]}

    beamlets = make_beamlets(table, energy)
    density, mask = water(table.dtype), all_scored()
    plain = make_engine(table).compute_dose(beamlets, density, mask)
    halved = make_engine(table, bev_correction=Half()).compute_dose(beamlets, density, mask)
    assert torch.allclose(halved, 0.5 * plain, atol=float(plain.max()) * 1e-12)


def test_gradients_flow_to_the_beamlet_weight(table, energy):
    """The engine never disables autograd on the caller's behalf."""
    engine = make_engine(table)
    beamlets = make_beamlets(table, energy).with_requires_grad(weight=True)
    dose = engine.compute_dose(beamlets, water(table.dtype), all_scored())
    dose.sum().backward()
    assert beamlets.weight.grad is not None
    assert float(beamlets.weight.grad.abs().max()) > 0.0


# --------------------------------------------------------------------------- masking


def test_dose_mask_is_required_and_explicit(table, energy):
    """No hidden density threshold: the mask is data, passed per call."""
    engine = make_engine(table)
    beamlets = make_beamlets(table, energy)
    density = water(table.dtype)

    with pytest.raises(TypeError):
        engine.compute_dose(beamlets, density)  # no mask

    cavity = density.clone()
    cavity[12:20, 60:70, 12:20] = 0.0012  # internal air, below any threshold

    kept = engine.compute_dose(beamlets, cavity, all_scored())
    topological = engine.compute_dose(beamlets, cavity, patient_dose_mask(cavity))
    thresholded = engine.compute_dose(beamlets, cavity, cavity > 0.03)

    assert float(kept[:, 12:20, 60:70, 12:20].max()) > 0.0
    # patient_dose_mask keeps the internal cavity; a density threshold zeroes it.
    assert float(topological[:, 12:20, 60:70, 12:20].max()) > 0.0
    assert float(thresholded[:, 12:20, 60:70, 12:20].max()) == 0.0



def test_cuda_without_an_index_matches_an_indexed_table(table):
    """`device="cuda"` must not read as a different device from the table's `cuda:0`."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    cuda_table = table.to(device="cuda")
    engine = IonDoseEngine(
        machine_config=IonMachineConfig(),
        kernel_table=cuda_table,
        dose_grid_spacing=(1.0, 1.0, 1.0),
        dose_grid_shape=(8, 16, 8),
        field_size=(4, 4),
        device="cuda",
    )
    assert engine.device == cuda_table.device


# --------------------------------------------------------------------------- crash loud


def test_oversized_field_raises_instead_of_being_clamped(table):
    """A field larger than the grid used to be silently shrunk to the grid."""
    with pytest.raises(ValueError, match="does not fit inside the dose grid"):
        make_engine(table, field_size=(GRID[0] + 2, GRID[2]))
    with pytest.raises(ValueError, match="must be positive"):
        make_engine(table, field_size=(0, GRID[2]))


def test_unknown_lateral_model_raises(table):
    """An unrecognised model used to fall back to a single Gaussian silently."""
    with pytest.raises(ValueError, match="lateral_model"):
        make_engine(table, lateral_model="double_gauss")


def test_double_gauss_is_never_silently_single(table, energy):
    """The halo is always available, so asking for it must change the answer."""
    assert table.has_double_gauss
    density, mask = water(table.dtype), all_scored()
    beamlets = make_beamlets(table, energy)
    single = make_engine(table, lateral_model="gauss").compute_dose(beamlets, density, mask)
    double = make_engine(table, lateral_model="gauss_double").compute_dose(beamlets, density, mask)
    assert not torch.allclose(single, double, rtol=1e-3)


def test_kernel_table_dtype_mismatch_raises(table):
    """The engine will not quietly compute in a dtype its base data is not in."""
    with pytest.raises(ValueError, match="kernel_table is on"):
        make_engine(table, dtype=torch.float32)


def test_beamlet_dtype_mismatch_raises(table, energy):
    """Beamlets in another dtype are a caller bug, not something to cast away."""
    engine = make_engine(table)
    beamlets = IonBeamletBatch.create(
        gantry_angle_deg=[0.0],
        position_mm=[[0.0, 0.0]],
        energy_mev=[energy],
        sigma_mm=[[SIGMA_MM, SIGMA_MM]],
        weight=[WEIGHT],
        iso_center_mm=list(ISO),
        sad_mm=SAD_MM,
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="beamlets are on"):
        engine.compute_dose(beamlets, water(table.dtype), all_scored())
    with pytest.raises(TypeError, match="IonBeamletBatch"):
        engine.compute_dose("not a batch", water(table.dtype), all_scored())


def test_volume_shape_mismatch_raises(table, energy):
    """Density and mask must match the dose grid exactly."""
    engine = make_engine(table)
    beamlets = make_beamlets(table, energy)
    with pytest.raises(ValueError, match="density_image has shape"):
        engine.compute_dose(beamlets, water(table.dtype, (16, 150, 32)), all_scored())
    with pytest.raises(ValueError, match="dose_mask has shape"):
        engine.compute_dose(beamlets, water(table.dtype), all_scored((16, 150, 32)))
    with pytest.raises(ValueError, match=r"\[H, D, W\]"):
        engine.compute_dose(beamlets, water(table.dtype).unsqueeze(0).expand(2, -1, -1, -1), all_scored())


def test_machine_config_ion_fields_are_read_directly(table, energy):
    """The nozzle geometry is a typed field, not a defensive ``getattr``."""
    engine = make_engine(table)
    beamlets = make_beamlets(table, energy)

    class PhotonLikeConfig:
        """A config that never declared the ion fields."""

    engine.machine_config = PhotonLikeConfig()
    with pytest.raises(AttributeError, match="bams_to_iso_dist_mm"):
        engine.compute_dose(beamlets, water(table.dtype), all_scored(), ssd_mm=800.0)


def test_ssd_offset_matches_the_nozzle_geometry(table, energy):
    """The SSD path is the only depth-offset path, and it is the documented one."""
    config = IonMachineConfig(bams_to_iso_dist_mm=1000.0, fit_air_offset_mm=0.0)
    engine = make_engine(table, machine_config=config)
    beamlets = make_beamlets(table, energy)
    ssd_mm = 800.0
    expected = 0.0011 * ((ssd_mm + config.bams_to_iso_dist_mm) - SAD_MM - config.fit_air_offset_mm)
    assert engine._resolve_depth_offset(beamlets, ssd_mm).tolist() == pytest.approx([expected])
    assert engine._resolve_depth_offset(beamlets, None).tolist() == [0.0]
    with pytest.raises(ValueError, match="ssd_mm must be scalar"):
        engine._resolve_depth_offset(beamlets, [800.0, 900.0])


def test_sub_beam_count_is_a_real_accuracy_knob(table, energy):
    """n=3 under-samples the splitting envelope and over-peaks the core."""
    density, mask = water(table.dtype), all_scored()
    beamlets = make_beamlets(table, energy)
    coarse = make_engine(table, n_sub_beams_per_dim=3).compute_dose(beamlets, density, mask)
    fine = make_engine(table, n_sub_beams_per_dim=9).compute_dose(beamlets, density, mask)
    assert float(coarse.max()) > 1.2 * float(fine.max())
    with pytest.raises(ValueError, match="n_sub_beams_per_dim"):
        make_engine(table, n_sub_beams_per_dim=0)


def test_finalize_chunk_size_does_not_change_the_result(table, energy):
    """Chunking is a memory knob only."""
    engine = make_engine(table)
    beamlets = IonBeamletBatch.create(
        gantry_angle_deg=[0.0, 30.0, 90.0],
        position_mm=[[0.0, 0.0], [4.0, 2.0], [-6.0, 3.0]],
        energy_mev=[energy] * 3,
        sigma_mm=[[SIGMA_MM, SIGMA_MM]] * 3,
        weight=[WEIGHT] * 3,
        iso_center_mm=list(ISO),
        sad_mm=SAD_MM,
        dtype=table.dtype,
    )
    density, mask = water(table.dtype), all_scored()
    one = engine.compute_dose(beamlets, density, mask, finalize_chunk_size=1)
    four = engine.compute_dose(beamlets, density, mask, finalize_chunk_size=4)
    assert torch.equal(one, four)


def test_beamlet_entirely_outside_the_grid_yields_none(table, energy):
    """An off-grid beamlet contributes nothing and is reported as ``None``."""
    engine = make_engine(table)
    beamlets = IonBeamletBatch.create(
        gantry_angle_deg=[0.0, 0.0],
        position_mm=[[0.0, 0.0], [0.0, 400.0]],
        energy_mev=[energy] * 2,
        sigma_mm=[[SIGMA_MM, SIGMA_MM]] * 2,
        weight=[WEIGHT] * 2,
        iso_center_mm=list(ISO),
        sad_mm=SAD_MM,
        dtype=table.dtype,
    )
    per_beamlet = engine.compute_dose(
        beamlets, water(table.dtype), all_scored(), return_per_beamlet=True
    )
    assert per_beamlet[0] is not None
    assert per_beamlet[1] is None


def test_engine_geometry_cache_survives_a_different_batch(table, energy):
    """Reusing an engine on other angles must not corrupt the next call."""
    engine = make_engine(table)
    density, mask = water(table.dtype), all_scored()
    beamlets = make_beamlets(table, energy)

    fresh = make_engine(table).compute_dose(beamlets, density, mask)
    engine.compute_dose(make_beamlets(table, energy, gantry_angle_deg=137.0), density, mask)
    reused = engine.compute_dose(beamlets, density, mask)
    assert torch.equal(reused, fresh)


def test_geometry_validation(table):
    """A malformed dose grid raises rather than being coerced."""
    with pytest.raises(ValueError, match="dose_grid_shape"):
        make_engine(table, dose_grid_shape=(32, 150))
    with pytest.raises(ValueError, match="dose_grid_spacing"):
        make_engine(table, dose_grid_spacing=(2.0, 0.0, 2.0))
    assert math.isclose(make_engine(table)._lateral_area_mm2(), SPACING[0] * SPACING[2])
