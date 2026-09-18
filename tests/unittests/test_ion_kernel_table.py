"""Tests for the ion kernel table and its commissioning converter.

Everything here stays inside the repository: the shipped
``pydosert/data/machine_presets/protons_doserad.npz`` is used for the real-table
behaviour, and the converter round-trip is exercised on synthetic ``.mat`` files
written into ``tmp_path``.
"""

import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from commissioning.conversion.convert_proton_mat_to_npz import (  # noqa: E402
    FORMAT_VERSION,
    convert_proton_mat_to_npz,
)
from pydosert.physics.kernels.ion_kernel_table import (  # noqa: E402
    DepthOutOfRangeError,
    EnergyNotInTableError,
    IonKernelTable,
)

PRESET_NPZ = Path(__file__).parents[2] / "src" / "pydosert" / "data" / "machine_presets" / "protons_doserad.npz"


# --------------------------------------------------------------------------- helpers


def _synthetic_row(energy: float, n_depths: int, n_focus: int = 4) -> dict:
    """One ``machine.data`` entry with reproducible, non-trivial curves."""
    rng = np.random.default_rng(int(energy * 1000))
    depths = np.cumsum(rng.uniform(0.1, 0.5, size=n_depths))
    return {
        "energy": energy,
        "offset": 0.25 * energy,
        "depths": depths,
        "Z": rng.uniform(0.5, 5.0, size=n_depths),
        "sigma": np.sort(rng.uniform(0.5, 8.0, size=n_depths)),
        "sigma1": np.sort(rng.uniform(0.4, 6.0, size=n_depths)),
        "sigma2": np.sort(rng.uniform(6.0, 40.0, size=n_depths)),
        "weight": rng.uniform(0.0, 0.4, size=n_depths),
        "initFocus": {
            "dist": np.linspace(9000.0, 11000.0, n_focus),
            "sigma": rng.uniform(2.0, 9.0, size=n_focus),
        },
    }


def _write_mat(path: Path, rows: list[dict], meta: dict | None = None) -> Path:
    """Write a matRad-shaped ``machine`` struct to ``path``."""
    sio = pytest.importorskip("scipy.io")
    meta = {"SAD": 10000.0, "BAMStoIsoDist": 1000.0} if meta is None else meta
    data = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        data[i] = row
    sio.savemat(str(path), {"machine": {"meta": meta, "data": data}})
    return path


@pytest.fixture(scope="module")
def synthetic_rows() -> list[dict]:
    # Deliberately ragged: the padded format must not resample or truncate.
    return [_synthetic_row(70.0, 11), _synthetic_row(100.0, 37), _synthetic_row(150.0, 23)]


@pytest.fixture(scope="module")
def synthetic_npz(tmp_path_factory, synthetic_rows) -> Path:
    tmp_path = tmp_path_factory.mktemp("ion_kernel_table")
    mat_path = _write_mat(tmp_path / "machine.mat", synthetic_rows)
    npz_path = tmp_path / "machine.npz"
    convert_proton_mat_to_npz(mat_path, npz_path)
    return npz_path


@pytest.fixture(scope="module")
def synthetic_table(synthetic_npz) -> IonKernelTable:
    return IonKernelTable.load(synthetic_npz, dtype=torch.float64)


@pytest.fixture(scope="module")
def preset_table() -> IonKernelTable:
    return IonKernelTable.load(PRESET_NPZ, dtype=torch.float64)


# ----------------------------------------------------------------- converter round-trip


class TestConverterRoundTrip:
    def test_shapes_and_padding(self, synthetic_npz, synthetic_rows):
        with np.load(synthetic_npz) as archive:
            assert int(archive["format_version"]) == FORMAT_VERSION
            n_valid = archive["n_valid"]
            depth = archive["depth_mm"]
            idd = archive["idd"]
            sigma = archive["sigma"]
            assert depth.dtype == np.float64
            assert n_valid.dtype == np.int32
            assert depth.shape == (3, max(r["depths"].size for r in synthetic_rows))
            for e, n in enumerate(n_valid):
                assert np.all(np.isinf(depth[e, n:]))  # depth padded with +inf
                assert np.all(idd[e, n:] == 0.0)  # idd padded with 0
                assert np.all(sigma[e, n:] == sigma[e, n - 1])  # sigma padded with the edge value

    @pytest.mark.parametrize("curve,field", [("idd", "Z"), ("sigma", "sigma"), ("sigma1", "sigma1"),
                                             ("sigma2", "sigma2"), ("weight", "weight")])
    def test_curves_are_bit_exact(self, synthetic_npz, synthetic_rows, curve, field):
        rows = sorted(synthetic_rows, key=lambda r: r["energy"])
        with np.load(synthetic_npz) as archive:
            stored = archive[curve]
            n_valid = archive["n_valid"]
            for e, row in enumerate(rows):
                order = np.argsort(row["depths"], kind="stable")
                assert np.array_equal(stored[e, : n_valid[e]], row[field][order])

    def test_scalars_and_focus_are_bit_exact(self, synthetic_npz, synthetic_rows):
        rows = sorted(synthetic_rows, key=lambda r: r["energy"])
        with np.load(synthetic_npz) as archive:
            assert float(archive["sad_mm"]) == 10000.0
            assert float(archive["bams_to_iso_mm"]) == 1000.0
            assert np.array_equal(archive["energy_mev"], [r["energy"] for r in rows])
            assert np.array_equal(archive["offset_mm"], [r["offset"] for r in rows])
            for e, row in enumerate(rows):
                assert np.array_equal(archive["focus_dist_mm"][e], row["initFocus"]["dist"])
                assert np.array_equal(archive["focus_sigma_mm"][e], row["initFocus"]["sigma"])

    def test_table_rows_match_the_mat_bit_exactly(self, synthetic_table, synthetic_rows):
        for row in synthetic_rows:
            curves = synthetic_table.row_curves(row["energy"])
            order = np.argsort(row["depths"], kind="stable")
            assert torch.equal(curves["depth_mm"], torch.from_numpy(row["depths"][order]))
            assert torch.equal(curves["idd"], torch.from_numpy(row["Z"][order]))
            assert torch.equal(curves["sigma_mm"], torch.from_numpy(row["sigma"][order]))
            assert torch.equal(curves["sigma1_mm"], torch.from_numpy(row["sigma1"][order]))
            assert torch.equal(curves["sigma2_mm"], torch.from_numpy(row["sigma2"][order]))
            assert torch.equal(curves["weight"], torch.from_numpy(row["weight"][order]))


class TestConverterFailsLoudly:
    @pytest.mark.parametrize("field", ["sigma1", "sigma2", "weight", "Z", "sigma", "depths", "offset", "initFocus"])
    def test_missing_entry_field(self, tmp_path, field):
        row = _synthetic_row(100.0, 9)
        row.pop(field)
        mat_path = _write_mat(tmp_path / f"missing_{field}.mat", [row])
        with pytest.raises(ValueError, match=f"'{field}'"):
            convert_proton_mat_to_npz(mat_path, tmp_path / "out.npz")

    @pytest.mark.parametrize("field", ["SAD", "BAMStoIsoDist"])
    def test_missing_meta_field(self, tmp_path, field):
        meta = {"SAD": 10000.0, "BAMStoIsoDist": 1000.0}
        meta.pop(field)
        mat_path = _write_mat(tmp_path / f"meta_{field}.mat", [_synthetic_row(100.0, 9)], meta=meta)
        with pytest.raises(ValueError, match=f"'{field}'"):
            convert_proton_mat_to_npz(mat_path, tmp_path / "out.npz")

    def test_missing_init_focus_subfield(self, tmp_path):
        row = _synthetic_row(100.0, 9)
        row["initFocus"].pop("sigma")
        mat_path = _write_mat(tmp_path / "focus.mat", [row])
        with pytest.raises(ValueError, match="'sigma'"):
            convert_proton_mat_to_npz(mat_path, tmp_path / "out.npz")

    def test_mismatched_curve_length(self, tmp_path):
        row = _synthetic_row(100.0, 9)
        row["sigma2"] = row["sigma2"][:-1]
        mat_path = _write_mat(tmp_path / "short.mat", [row])
        with pytest.raises(ValueError, match="mismatched 'sigma2'"):
            convert_proton_mat_to_npz(mat_path, tmp_path / "out.npz")

    def test_duplicate_energies(self, tmp_path):
        rows = [_synthetic_row(100.0, 9), _synthetic_row(100.0, 9)]
        mat_path = _write_mat(tmp_path / "dup.mat", rows)
        with pytest.raises(ValueError, match="duplicate energies"):
            convert_proton_mat_to_npz(mat_path, tmp_path / "out.npz")

    def test_non_monotonic_depths(self, tmp_path):
        row = _synthetic_row(100.0, 9)
        row["depths"][3] = row["depths"][2]
        mat_path = _write_mat(tmp_path / "flat.mat", [row])
        with pytest.raises(ValueError, match="non-monotonic 'depths'"):
            convert_proton_mat_to_npz(mat_path, tmp_path / "out.npz")

    def test_ragged_focus_grid(self, tmp_path):
        rows = [_synthetic_row(100.0, 9, n_focus=4), _synthetic_row(150.0, 9, n_focus=5)]
        mat_path = _write_mat(tmp_path / "focus_ragged.mat", rows)
        with pytest.raises(ValueError, match="differing lengths"):
            convert_proton_mat_to_npz(mat_path, tmp_path / "out.npz")

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            convert_proton_mat_to_npz(tmp_path / "nope.mat", tmp_path / "out.npz")


# ------------------------------------------------------------------------ row lookup


class TestRowIndex:
    def test_every_tabulated_energy_resolves(self, preset_table):
        energies = preset_table.available_energies
        assert len(energies) == preset_table.num_energies
        for i, energy in enumerate(energies):
            assert preset_table.row_index(energy) == i
            assert preset_table.row_index(torch.tensor(energy)) == i

    def test_lookup_is_constant_time(self, preset_table):
        """The old ``_bracket`` scanned linearly (4.5 us low, 494 us high)."""
        energies = preset_table.available_energies

        def timeit(energy, reps=20000):
            start = time.perf_counter()
            for _ in range(reps):
                preset_table.row_index(energy)
            return (time.perf_counter() - start) / reps

        low = timeit(energies[0])
        high = timeit(energies[-1])
        assert high < 10.0 * low + 1e-6, f"lookup scales with row index: {low:.3e}s vs {high:.3e}s"

    def test_untabulated_energy_raises_and_names_neighbours(self, preset_table):
        energies = preset_table.available_energies
        mid = 0.5 * (energies[10] + energies[11])
        with pytest.raises(EnergyNotInTableError) as excinfo:
            preset_table.row_index(mid)
        message = str(excinfo.value)
        assert f"{energies[10]:.6f}" in message
        assert f"{energies[11]:.6f}" in message

    def test_above_the_table_raises(self, preset_table):
        """The old loader silently returned the top row for 1000 MeV."""
        with pytest.raises(EnergyNotInTableError, match="above the table"):
            preset_table.row_index(1000.0)

    def test_below_the_table_raises(self, preset_table):
        """The old loader silently returned the bottom row for 5 MeV."""
        with pytest.raises(EnergyNotInTableError, match="below the table"):
            preset_table.row_index(5.0)

    @pytest.mark.parametrize("accessor", ["edep_curve", "sigma_curve", "double_gauss_curves", "kernel_offset"])
    def test_accessors_reject_untabulated_energies(self, preset_table, accessor):
        with pytest.raises(EnergyNotInTableError):
            getattr(preset_table, accessor)(1000.0)

    def test_depth_and_sample_helpers(self, preset_table, ):
        energy = preset_table.available_energies[40]
        index = preset_table.row_index(energy)
        n = preset_table.n_samples(energy)
        assert n == int(preset_table.n_valid[index])
        assert preset_table.depth_max_mm(energy) == float(preset_table.depth_mm[index, n - 1])


# --------------------------------------------------------------- depth interpolation


class TestDepthInterpolation:
    def test_tabulated_nodes_return_stored_values(self, preset_table):
        # Evaluating at a node still goes through the interpolation arithmetic
        # (y0 + slope * (x1 - x0)), so it can differ from the stored value by an
        # ulp; bit-exactness of the *stored* rows is covered by the round-trip tests.
        close = dict(rtol=1e-14, atol=0.0)
        for energy in preset_table.available_energies[::17]:
            curves = preset_table.row_curves(energy)
            depths = curves["depth_mm"]
            torch.testing.assert_close(preset_table.edep(energy, depths), curves["idd"], **close)
            torch.testing.assert_close(preset_table.sigma(energy, depths), curves["sigma_mm"], **close)
            s1, s2, w = preset_table.double_gauss(energy, depths)
            torch.testing.assert_close(s1, curves["sigma1_mm"], **close)
            torch.testing.assert_close(s2, curves["sigma2_mm"], **close)
            torch.testing.assert_close(w, curves["weight"], **close)

    def test_midpoint_is_the_mean_of_the_neighbours(self, preset_table):
        energy = preset_table.available_energies[40]
        curves = preset_table.row_curves(energy)
        depths = curves["depth_mm"]
        mid = 0.5 * (depths[:-1] + depths[1:])
        expected = 0.5 * (curves["idd"][:-1] + curves["idd"][1:])
        assert torch.allclose(preset_table.edep(energy, mid), expected, rtol=0, atol=1e-9)

    def test_last_valid_sample_ignores_the_padding(self, preset_table):
        """n_valid - 1 must return the stored edge value, never a padded one."""
        for energy in preset_table.available_energies[::13]:
            curves = preset_table.row_curves(energy)
            n = curves["depth_mm"].numel()
            assert preset_table.n_samples(energy) == n
            last = curves["depth_mm"][n - 1]
            torch.testing.assert_close(preset_table.edep(energy, last), curves["idd"][n - 1], rtol=1e-14, atol=0.0)
            torch.testing.assert_close(
                preset_table.sigma(energy, last), curves["sigma_mm"][n - 1], rtol=1e-14, atol=0.0
            )
            # a hair before the last node interpolates between the last two samples
            just_before = last - 1e-6
            value = preset_table.sigma(energy, just_before)
            lo, hi = sorted((float(curves["sigma_mm"][n - 2]), float(curves["sigma_mm"][n - 1])))
            assert lo <= float(value) <= hi

    def test_below_the_first_node_holds_the_first_value(self, preset_table):
        energy = preset_table.available_energies[3]
        curves = preset_table.row_curves(energy)
        assert preset_table.edep(energy, curves["depth_mm"][0] - 5.0) == curves["idd"][0]

    def test_accepts_arbitrary_trailing_shape(self, preset_table):
        energy = preset_table.available_energies[20]
        depth_max = preset_table.depth_max_mm(energy)
        for shape in [(7,), (3, 5), (2, 3, 4)]:
            depths = torch.rand(shape, dtype=torch.float64) * depth_max
            assert preset_table.edep(energy, depths).shape == depths.shape
            assert preset_table.sigma(energy, depths).shape == depths.shape
            s1, s2, w = preset_table.double_gauss(energy, depths)
            assert s1.shape == s2.shape == w.shape == depths.shape


class TestBeyondRange:
    def test_idd_is_zero_past_the_range(self, preset_table):
        energy = preset_table.available_energies[40]
        depth_max = preset_table.depth_max_mm(energy)
        depths = torch.tensor([depth_max, depth_max + 1e-3, depth_max + 500.0], dtype=torch.float64)
        values = preset_table.edep(energy, depths)
        assert values[0] > 0.0
        assert torch.equal(values[1:], torch.zeros(2, dtype=torch.float64))

    @pytest.mark.parametrize("accessor", ["sigma", "double_gauss"])
    def test_lateral_parameters_raise_past_the_range(self, preset_table, accessor):
        energy = preset_table.available_energies[40]
        depth_max = preset_table.depth_max_mm(energy)
        with pytest.raises(DepthOutOfRangeError, match="past the tabulated range"):
            getattr(preset_table, accessor)(energy, torch.tensor([1.0, depth_max + 1e-3], dtype=torch.float64))

    def test_at_the_last_node_does_not_raise(self, preset_table):
        energy = preset_table.available_energies[40]
        preset_table.sigma(energy, preset_table.depth_max_mm(energy))

    def test_edge_mode_reproduces_the_old_clamped_continuation(self, preset_table):
        energy = preset_table.available_energies[40]
        curves = preset_table.row_curves(energy)
        far = preset_table.depth_max_mm(energy) + 5000.0
        assert preset_table.sigma(energy, far, beyond_range="edge") == curves["sigma_mm"][-1]
        assert preset_table.edep(energy, far, beyond_range="edge") == curves["idd"][-1]

    def test_unknown_mode_raises(self, preset_table):
        energy = preset_table.available_energies[0]
        with pytest.raises(ValueError, match="Unknown beyond_range"):
            preset_table.edep(energy, 1e6, beyond_range="clamp")


# ------------------------------------------------------------------ other accessors


class TestAccessors:
    def test_kernel_offset(self, synthetic_table, synthetic_rows):
        for row in synthetic_rows:
            offset = synthetic_table.kernel_offset(row["energy"])
            assert offset.ndim == 0
            assert float(offset) == row["offset"]

    def test_initial_sigma_interpolates_and_clamps(self, synthetic_table, synthetic_rows):
        row = sorted(synthetic_rows, key=lambda r: r["energy"])[1]
        dist = torch.from_numpy(row["initFocus"]["dist"])
        sigma = torch.from_numpy(row["initFocus"]["sigma"])
        assert torch.equal(synthetic_table.initial_sigma(row["energy"], dist), sigma)
        # outside the tabulated distances the focus curve is held at its edges
        assert synthetic_table.initial_sigma(row["energy"], float(dist[0]) - 1000.0) == sigma[0]
        assert synthetic_table.initial_sigma(row["energy"], float(dist[-1]) + 1000.0) == sigma[-1]

    def test_capability_flags(self, preset_table):
        assert preset_table.has_double_gauss
        assert preset_table.has_initial_focus
        assert preset_table.sad_mm == 10000.0
        assert preset_table.bams_to_iso_dist_mm == 1000.0

    def test_accessors_are_pure_functions_of_the_tensors(self, synthetic_npz):
        """No caches: a mutated row must show up immediately (residuals depend on this)."""
        table = IonKernelTable.load(synthetic_npz, dtype=torch.float64)
        energy = table.available_energies[0]
        depth = table.row_curves(energy)["depth_mm"][2]
        before = table.edep(energy, depth).clone()
        with torch.no_grad():
            table.idd[0, 2] += 1.0
        assert table.edep(energy, depth) == before + 1.0

    def test_energy_can_be_a_tensor_carrying_grad(self, preset_table):
        energy = preset_table.available_energies[10]
        tensor = torch.tensor(energy, dtype=torch.float64, requires_grad=True)
        value = preset_table.edep(tensor, 10.0)
        assert value.shape == ()


# ------------------------------------------------------------------ device / dtype


class TestDeviceAndDtype:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_load_dtype_propagates(self, dtype):
        table = IonKernelTable.load(PRESET_NPZ, dtype=dtype)
        assert table.dtype == dtype
        assert table.device == torch.device("cpu")
        for name in ("depth_mm", "idd", "sigma_mm", "sigma1_mm", "sigma2_mm", "weight", "offset_mm"):
            assert getattr(table, name).dtype == dtype
        # energies stay float64: they are lookup keys, not curves
        assert table.energy_mev.dtype == torch.float64
        energy = table.available_energies[7]
        assert table.edep(energy, 10.0).dtype == dtype
        assert table.sigma(energy, 10.0).dtype == dtype
        assert table.kernel_offset(energy).dtype == dtype

    def test_float32_is_the_rounded_float64_table(self, preset_table):
        table32 = preset_table.to(dtype=torch.float32)
        assert torch.equal(table32.idd, preset_table.idd.to(torch.float32))
        assert table32.available_energies == preset_table.available_energies

    def test_to_returns_a_new_table(self, preset_table):
        moved = preset_table.to(dtype=torch.float32)
        assert moved is not preset_table
        assert preset_table.dtype == torch.float64

    def test_non_float_dtype_rejected(self):
        with pytest.raises(ValueError, match="floating point"):
            IonKernelTable.load(PRESET_NPZ, dtype=torch.int32)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            IonKernelTable.load(tmp_path / "nope.npz")

    def test_wrong_archive_rejected(self, tmp_path):
        path = tmp_path / "bogus.npz"
        np.savez(path, something_else=np.zeros(3))
        with pytest.raises(ValueError, match="missing arrays"):
            IonKernelTable.load(path)

    def test_unknown_format_version_rejected(self, tmp_path, synthetic_npz):
        with np.load(synthetic_npz) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["format_version"] = np.int64(FORMAT_VERSION + 1)
        path = tmp_path / "future.npz"
        np.savez(path, **arrays)
        with pytest.raises(ValueError, match="format version"):
            IonKernelTable.load(path)


# ------------------------------------------------------------- energy interpolation


class TestEnergyInterpolation:
    def test_disabled_by_default(self, preset_table):
        energies = preset_table.available_energies
        mid = 0.5 * (energies[30] + energies[31])
        with pytest.raises(EnergyNotInTableError):
            preset_table.edep(mid, 10.0)

    def test_enabled_per_call(self, preset_table):
        energies = preset_table.available_energies
        mid = 0.5 * (energies[30] + energies[31])
        depth, idd = preset_table.edep_curve(mid, allow_energy_interpolation=True)
        assert depth.numel() == idd.numel() > 1
        # the range-shifted peak sits between the neighbouring peaks
        def peak(energy):
            d, z = preset_table.edep_curve(energy)
            return float(d[int(torch.argmax(z))])
        assert peak(energies[30]) < float(depth[int(torch.argmax(idd))]) < peak(energies[31])

    def test_enabled_on_the_table(self):
        table = IonKernelTable.load(PRESET_NPZ, dtype=torch.float64, allow_energy_interpolation=True)
        energies = table.available_energies
        mid = 0.5 * (energies[30] + energies[31])
        # The ported interpolation rescales depth by the peak-shift ratio, so a
        # value at fixed depth is close to, but not strictly bracketed by, the
        # neighbouring rows.
        sigma = table.sigma(mid, 10.0)
        lo = float(table.sigma(energies[30], 10.0))
        hi = float(table.sigma(energies[31], 10.0))
        assert float(sigma) == pytest.approx(0.5 * (lo + hi), rel=0.05)
        s1, s2, w = table.double_gauss(mid, 10.0)
        assert float(s1) > 0.0 and float(s2) > 0.0 and 0.0 <= float(w) <= 1.0
        assert float(table.kernel_offset(mid)) == pytest.approx(0.0)
        assert float(table.initial_sigma(mid, 10000.0)) > 0.0

    def test_cannot_extrapolate_outside_the_table(self):
        table = IonKernelTable.load(PRESET_NPZ, dtype=torch.float64, allow_energy_interpolation=True)
        with pytest.raises(EnergyNotInTableError, match="cannot extrapolate"):
            table.edep(1000.0, 10.0)
        with pytest.raises(EnergyNotInTableError, match="cannot extrapolate"):
            table.edep(5.0, 10.0)

    def test_tabulated_energies_are_untouched_when_enabled(self, preset_table):
        table = preset_table.to()
        interpolating = IonKernelTable.load(PRESET_NPZ, dtype=torch.float64, allow_energy_interpolation=True)
        for energy in table.available_energies[::23]:
            assert torch.equal(table.edep_curve(energy)[1], interpolating.edep_curve(energy)[1])


def test_load_accepts_a_bundled_table_name(tmp_path):
    """A bare name resolves to the packaged table, like MachineConfig's presets,
    while paths keep working and a local file of the same name wins."""
    from pydosert.physics.kernels.ion_kernel_table import list_ion_kernel_tables

    assert "protons_doserad" in list_ion_kernel_tables()

    by_name = IonKernelTable.load("protons_doserad")
    by_name_ext = IonKernelTable.load("protons_doserad.npz")
    by_path = IonKernelTable.load(PRESET_NPZ)
    by_abs = IonKernelTable.load(Path(PRESET_NPZ).resolve())
    for other in (by_name_ext, by_path, by_abs):
        assert other.available_energies == by_name.available_energies

    # an existing file is never shadowed by the bundled table of the same name
    shutil.copy(PRESET_NPZ, tmp_path / "protons_doserad.npz")
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        assert IonKernelTable.load("protons_doserad.npz").available_energies == by_name.available_energies
    finally:
        os.chdir(cwd)


def test_load_reports_unknown_names_and_paths():
    with pytest.raises(FileNotFoundError, match="bundled tables"):
        IonKernelTable.load("not_a_table")
    with pytest.raises(FileNotFoundError, match="not found"):
        IonKernelTable.load("no/such/dir/protons_doserad.npz")
