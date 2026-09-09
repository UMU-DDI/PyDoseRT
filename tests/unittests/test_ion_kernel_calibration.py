"""Contract tests for the differentiable LUT calibration.

Self-contained: everything runs against the kernel table packaged with
``pydosert``, on analytic water phantoms. The Monte Carlo reference the real
calibration is fitted to is ~238 GiB and lives outside the repository (see
``commissioning/calibrate_ion_kernel_table.py``), so the only fit exercised here is
against a *synthetic* target produced by perturbing a table row.

What is pinned:

* the calibration is exactly the identity at initialisation -- an untrained
  calibration cannot change a single dose value;
* the optimiser cannot leave the physical envelope, whatever it does to the
  parameters (positive curves, ``sigma2 >= sigma1``, ``weight`` in ``[0, 1]``);
* padding and unselected rows are never written;
* gradients reach the parameters through a full ``compute_dose`` -- the
  end-to-end differentiability the engine's autograd-free forward unlocked;
* a fitted calibration round-trips through ``state_dict``.
"""

from pathlib import Path

import pytest
import torch

from pydosert.data.ion_beam import IonBeamletBatch
from pydosert.data.ion_machine import IonMachineConfig
from pydosert.engine.ion_dose_engine import IonDoseEngine
from pydosert.physics.kernels.ion_kernel_calibration import IonKernelCalibration, inv_sigmoid, inv_softplus
from pydosert.physics.kernels.ion_kernel_table import EnergyNotInTableError, IonKernelTable

TABLE_NPZ = (
    Path(__file__).parents[2] / "src" / "pydosert" / "data" / "machine_presets" / "protons_doserad.npz"
)

#: Every curve tensor the calibration may touch, plus the ones it must not.
CURVE_NAMES = ("idd", "sigma_mm", "sigma1_mm", "sigma2_mm", "weight", "depth_mm", "offset_mm")

#: A small water phantom: (H, D, W) voxels at 2 mm, deep enough for a 120 MeV peak.
GRID = (16, 80, 16)
SPACING = (2.0, 2.0, 2.0)
ISO = (15.0, 80.0, 15.0)
SAD_MM = 1000.0
SIGMA_MM = 4.6
WEIGHT = 1.0e7


@pytest.fixture(scope="module")
def table() -> IonKernelTable:
    """The packaged proton kernel table, float64 so the identity claim is sharp."""
    return IonKernelTable.load(TABLE_NPZ, dtype=torch.float64)


@pytest.fixture(scope="module")
def energy(table: IonKernelTable) -> float:
    """A tabulated energy whose range fits inside the test grid."""
    return min(table.available_energies, key=lambda e: abs(e - 120.0))


def make_engine(table: IonKernelTable, **kwargs) -> IonDoseEngine:
    """An engine on the small water geometry."""
    settings = dict(
        machine_config=IonMachineConfig(),
        kernel_table=table,
        dose_grid_spacing=SPACING,
        dose_grid_shape=GRID,
        field_size=(GRID[0], GRID[2]),
        lateral_model="gauss_double",
        heterogeneous_mcs=False,
        n_sub_beams_per_dim=3,
    )
    settings.update(kwargs)
    return IonDoseEngine(**settings)


def make_beamlets(energy_mev: float, dtype: torch.dtype = torch.float64) -> IonBeamletBatch:
    """A single beamlet down the middle of the phantom."""
    return IonBeamletBatch.create(
        gantry_angle_deg=[0.0],
        position_mm=[[0.0, 0.0]],
        energy_mev=[float(energy_mev)],
        sigma_mm=[[SIGMA_MM, SIGMA_MM]],
        weight=[WEIGHT],
        iso_center_mm=list(ISO),
        sad_mm=SAD_MM,
        dtype=dtype,
    )


def dose_of(table: IonKernelTable, energy_mev: float, **engine_kwargs) -> torch.Tensor:
    """Water-phantom dose of one beamlet through ``table``."""
    engine = make_engine(table, **engine_kwargs)
    return engine.compute_dose(
        make_beamlets(energy_mev, table.dtype),
        torch.ones(GRID, dtype=table.dtype),
        torch.ones(GRID, dtype=torch.bool),
    )


def fill_residuals(calibration: IonKernelCalibration, value: float) -> None:
    """Set every residual parameter to ``value`` (including its padding columns)."""
    with torch.no_grad():
        for parameter in calibration.parameters():
            parameter.fill_(value)


# ------------------------------------------------------------------- identity


class TestIdentityAtInitialisation:
    """An untrained calibration must be invisible, bit-for-bit."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_calibrated_table_is_bit_identical(self, dtype):
        table = IonKernelTable.load(TABLE_NPZ, dtype=dtype)
        calibrated = IonKernelCalibration(table).apply()
        for name in CURVE_NAMES:
            assert torch.equal(getattr(calibrated, name), getattr(table, name)), name

    def test_identity_holds_for_a_row_subset(self, table):
        energies = [table.available_energies[i] for i in (0, 13, 60, table.num_energies - 1)]
        calibrated = IonKernelCalibration(table, energies).apply()
        for name in CURVE_NAMES:
            assert torch.equal(getattr(calibrated, name), getattr(table, name)), name

    def test_unselected_rows_stay_identical_after_training(self, table, energy):
        """Rows outside the selection are copied, never recomputed."""
        calibration = IonKernelCalibration(table, [energy])
        fill_residuals(calibration, 0.7)
        calibrated = calibration.apply()
        row = table.row_index(energy)
        others = [i for i in range(table.num_energies) if i != row]
        for name in ("idd", "sigma_mm", "sigma1_mm", "sigma2_mm", "weight"):
            assert torch.equal(getattr(calibrated, name)[others], getattr(table, name)[others]), name
        assert not torch.equal(calibrated.idd[row], table.idd[row])

    def test_the_input_table_is_not_mutated(self, table, energy):
        before = {name: getattr(table, name).clone() for name in CURVE_NAMES}
        calibration = IonKernelCalibration(table, [energy])
        fill_residuals(calibration, -0.4)
        calibration.apply()
        for name, tensor in before.items():
            assert torch.equal(getattr(table, name), tensor), name

    def test_the_engine_is_unaffected_by_an_untrained_calibration(self, table, energy):
        """The point of exact identity: dose through a calibrated table is the same dose."""
        plain = dose_of(table, energy)
        calibrated = dose_of(IonKernelCalibration(table, [energy]).apply(), energy)
        assert torch.equal(plain, calibrated)

    def test_the_inverse_transform_round_trip_is_not_exact(self, table):
        """Why the residual is additive: re-deriving the base would perturb it."""
        base = table.idd[table.num_energies // 2]
        assert not torch.equal(torch.nn.functional.softplus(inv_softplus(base)), base)
        weight = table.weight[table.num_energies // 2]
        assert not torch.equal(torch.sigmoid(inv_sigmoid(weight)), weight)


# ----------------------------------------------------------------- constraints


class TestPhysicalConstraints:
    """The optimiser cannot produce an unphysical curve, however hard it tries."""

    EXTREMES = [-1e6, -1e3, -50.0, -1.0, 1.0, 50.0, 1e3, 1e6]

    @pytest.mark.parametrize("value", EXTREMES)
    def test_positive_curves_stay_positive(self, table, energy, value):
        calibration = IonKernelCalibration(table, [energy])
        fill_residuals(calibration, value)
        curves = calibration.calibrated_rows()
        for name in ("idd", "sigma_mm", "sigma1_mm", "sigma2_mm"):
            assert bool((curves[name] >= 0.0).all()), name
            assert bool(torch.isfinite(curves[name]).all()), name

    @pytest.mark.parametrize("value", EXTREMES)
    def test_sigma2_never_falls_below_sigma1(self, table, energy, value):
        calibration = IonKernelCalibration(table, [energy])
        fill_residuals(calibration, value)
        curves = calibration.calibrated_rows()
        assert bool((curves["sigma2_mm"] >= curves["sigma1_mm"]).all())

    @pytest.mark.parametrize("scale", [0.5, 2.0, 20.0])
    def test_sigma2_stays_above_sigma1_for_random_parameters(self, table, energy, scale):
        calibration = IonKernelCalibration(table, [energy])
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for parameter in calibration.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * scale)
        curves = calibration.calibrated_rows()
        assert bool((curves["sigma2_mm"] >= curves["sigma1_mm"]).all())
        if scale <= 2.0:
            # A residual has to overshoot by a factor of ten before it can close
            # the gap entirely; short of that the two components stay separated.
            assert bool((curves["sigma2_mm"] > curves["sigma1_mm"]).all())

    @pytest.mark.parametrize("value", EXTREMES)
    def test_weight_stays_in_the_unit_interval(self, table, energy, value):
        calibration = IonKernelCalibration(table, [energy])
        fill_residuals(calibration, value)
        weight = calibration.calibrated_rows()["weight"]
        assert bool((weight >= 0.0).all()) and bool((weight <= 1.0).all())
        assert bool(torch.isfinite(weight).all())

    def test_an_extreme_calibration_still_makes_a_usable_table(self, table, energy):
        """A saturated calibration must still load into the engine without NaN."""
        calibration = IonKernelCalibration(table, [energy])
        fill_residuals(calibration, 3.0)
        dose = dose_of(calibration.apply(), energy).detach()
        assert bool(torch.isfinite(dose).all())
        assert float(dose.max()) > 0.0


class TestPadding:
    """Rows are padded-rectangular; the residual lives inside ``n_valid`` only."""

    @pytest.mark.parametrize("value", [-1e3, 5.0, 1e3])
    def test_padding_is_never_written(self, table, energy, value):
        calibration = IonKernelCalibration(table, [energy])
        fill_residuals(calibration, value)
        curves = calibration.calibrated_rows()
        padding = ~calibration.valid_mask
        row = table.row_index(energy)
        for name in ("idd", "sigma_mm", "sigma1_mm", "sigma2_mm", "weight"):
            base = getattr(table, name)[row : row + 1]
            assert torch.equal(curves[name][padding], base[padding]), name

    def test_a_residual_beyond_n_valid_changes_nothing(self, table, energy):
        calibration = IonKernelCalibration(table, [energy])
        n_valid = int(calibration.n_valid[0])
        with torch.no_grad():
            for parameter in calibration.parameters():
                parameter[:, n_valid:] = 1234.0
        calibrated = calibration.apply()
        for name in CURVE_NAMES:
            assert torch.equal(getattr(calibrated, name), getattr(table, name)), name

    def test_padding_receives_no_gradient(self, table, energy):
        calibration = IonKernelCalibration(table, [energy])
        sum(curve.sum() for curve in calibration.calibrated_rows().values()).backward()
        n_valid = int(calibration.n_valid[0])
        for parameter in calibration.parameters():
            assert float(parameter.grad[:, n_valid:].abs().max()) == 0.0
            assert float(parameter.grad[:, :n_valid].abs().max()) > 0.0


# ------------------------------------------------------------------ gradients


class TestGradients:
    def test_gradients_reach_the_parameters_through_apply(self, table, energy):
        calibration = IonKernelCalibration(table, [energy])
        calibrated = calibration.apply()
        (calibrated.idd.sum() + calibrated.sigma2_mm.sum() + calibrated.weight.sum()).backward()
        for name in ("idd_residual", "sigma1_residual", "gap_residual", "weight_residual"):
            gradient = getattr(calibration, name).grad
            assert gradient is not None, name
            assert bool(torch.isfinite(gradient).all()), name
            assert float(gradient.abs().max()) > 0.0, name

    def test_gradients_reach_the_parameters_through_compute_dose(self, table, energy):
        """End-to-end: the engine disables autograd nowhere, so the table is trainable.

        The objective is deliberately *spatially structured* (the dose in a
        central column). A plain ``dose.sum()`` is nearly blind to the lateral
        parameters, because the lateral kernel is normalised to conserve the
        depth-dose integral -- the sum would only see the IDD.
        """
        calibration = IonKernelCalibration(table, [energy])
        dose = dose_of(calibration.apply(), energy)
        dose[0, 7:9, :, 7:9].sum().backward()
        for name in ("idd_residual", "sigma1_residual", "gap_residual", "weight_residual"):
            gradient = getattr(calibration, name).grad
            assert gradient is not None, name
            assert bool(torch.isfinite(gradient).all()), name
            assert float(gradient.abs().max()) > 0.0, name

    def test_the_single_gaussian_sigma_is_trainable_too(self, table, energy):
        """``sigma`` is only read by the single-Gaussian model, so it is checked there."""
        calibration = IonKernelCalibration(table, [energy])
        dose = dose_of(calibration.apply(), energy, lateral_model="gauss")
        dose[0, 7:9, :, 7:9].sum().backward()
        gradient = calibration.sigma_residual.grad
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        assert float(gradient.abs().max()) > 0.0

    def test_the_smoothness_penalty_is_differentiable(self, table, energy):
        calibration = IonKernelCalibration(table, [energy])
        penalty = calibration.smoothness_penalty()
        assert penalty.ndim == 0 and float(penalty.detach()) > 0.0
        penalty.backward()
        assert float(calibration.idd_residual.grad.abs().max()) > 0.0

    def test_a_rougher_residual_costs_more(self, table, energy):
        calibration = IonKernelCalibration(table, [energy])
        smooth = float(calibration.smoothness_penalty().detach())
        with torch.no_grad():
            n_valid = int(calibration.n_valid[0])
            comb = torch.zeros_like(calibration.idd_residual)
            comb[:, :n_valid:2] = 1.0
            calibration.idd_residual.copy_(comb)
        assert float(calibration.smoothness_penalty().detach()) > smooth


# ----------------------------------------------------------------- state dict


class TestStateDict:
    def test_round_trip_reproduces_the_calibrated_table(self, table, energy):
        fitted = IonKernelCalibration(table, [energy])
        generator = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for parameter in fitted.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * 0.3)
        expected = fitted.apply()

        state = {name: tensor.clone() for name, tensor in fitted.state_dict().items()}
        restored = IonKernelCalibration(table, [energy])
        assert not torch.equal(restored.apply().idd, expected.idd)  # a fresh module is the identity
        restored.load_state_dict(state)

        reloaded = restored.apply()
        for name in ("idd", "sigma_mm", "sigma1_mm", "sigma2_mm", "weight"):
            assert torch.equal(getattr(reloaded, name), getattr(expected, name)), name

    def test_a_state_dict_from_another_row_count_is_rejected(self, table):
        one = IonKernelCalibration(table, [table.available_energies[3]])
        two = IonKernelCalibration(table, table.available_energies[3:5])
        with pytest.raises(RuntimeError):
            one.load_state_dict(two.state_dict())


# ------------------------------------------------------------------ selection


class TestSelection:
    def test_every_row_by_default(self, table):
        calibration = IonKernelCalibration(table)
        assert calibration.num_rows == table.num_energies
        assert calibration.energies_mev == table.available_energies

    def test_energies_are_reported_in_parameter_order(self, table):
        energies = [table.available_energies[i] for i in (5, 2)]
        assert IonKernelCalibration(table, energies).energies_mev == energies

    def test_an_untabulated_energy_raises(self, table):
        with pytest.raises(EnergyNotInTableError):
            IonKernelCalibration(table, [table.available_energies[0] + 0.123])

    def test_an_empty_selection_raises(self, table):
        with pytest.raises(ValueError, match="empty"):
            IonKernelCalibration(table, [])

    def test_a_repeated_energy_raises(self, table):
        energy = table.available_energies[4]
        with pytest.raises(ValueError, match="same table row twice"):
            IonKernelCalibration(table, [energy, energy])

    def test_an_unphysical_table_raises(self, table):
        from dataclasses import replace

        broken = replace(table, sigma2_mm=table.sigma1_mm - 1.0)
        with pytest.raises(ValueError, match="sigma2 < sigma1"):
            IonKernelCalibration(broken)


# ------------------------------------------------------------------- the fit


class TestSyntheticFit:
    """The only fit that can live in the repository: the MC reference is not shipped.

    The target is the dose of a table whose IDD row carries a depth-dependent
    stretch. That perturbation is *not* removable by rescaling -- it changes the
    shape of the depth dose -- so the fit has to move the curve, not a constant.
    """

    def test_the_loss_drops_materially(self, table, energy):
        row = table.row_index(energy)
        n_valid = int(table.n_valid[row])

        from dataclasses import replace

        perturbed = table.idd.clone()
        ramp = torch.linspace(0.0, 1.0, n_valid, dtype=table.dtype)
        perturbed[row, :n_valid] *= 1.0 + 0.25 * ramp
        target = dose_of(replace(table, idd=perturbed), energy).detach()

        calibration = IonKernelCalibration(table, [energy])
        engine = make_engine(table)
        beamlets = make_beamlets(energy, table.dtype)
        density = torch.ones(GRID, dtype=table.dtype)
        scored = torch.ones(GRID, dtype=torch.bool)
        optimiser = torch.optim.Adam(calibration.parameters(), lr=0.2)

        losses = []
        for _ in range(25):
            optimiser.zero_grad(set_to_none=True)
            engine.kernel_table = calibration.apply()  # what used to be a monkey-patch
            dose = engine.compute_dose(beamlets, density, scored)
            loss = (dose - target).abs().mean()
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))

        assert losses[-1] < 0.25 * losses[0], f"loss barely moved: {losses[0]:.4e} -> {losses[-1]:.4e}"
        # and the fit moved the IDD curve towards the perturbed one, not away
        fitted = calibration.apply().idd[row, :n_valid].detach()
        before = (table.idd[row, :n_valid] - perturbed[row, :n_valid]).abs().sum()
        after = (fitted - perturbed[row, :n_valid]).abs().sum()
        assert float(after) < float(before)
