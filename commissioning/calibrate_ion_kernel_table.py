"""Calibrate an ion kernel table against water-phantom Monte Carlo, by backpropagation.

The commissioned proton table (``machine.data`` in matRad/pyRadPlan terms) is a
fit whose residual against Monte Carlo is a few per cent in depth dose and
lateral halo. Since :class:`~pydosert.engine.ion_dose_engine.IonDoseEngine` is
differentiable end to end, that residual can be removed at its source: run the
engine on a water phantom, compare with the MC dose, backpropagate into the
table's depth curves.

This is the driver for that fit. It optimises
:class:`~pydosert.physics.kernels.ion_kernel_calibration.IonKernelCalibration`
-- learnable residuals on ``idd``/``sigma``/``sigma1``/``sigma2``/``weight``,
zero at initialisation and unable to produce an unphysical curve -- and rebuilds
the table each step with ``engine.kernel_table = calibration.apply()``.

Usage
-----
::

    python commissioning/calibrate_ion_kernel_table.py \\
        --mc-dir /path/to/MC_proton_simulation_DoseRAD2026/output \\
        --out    commissioning/data/protons_doserad_calibrated.npz \\
        --energies 164.4532 --iters 600

``--energies all`` calibrates every energy for which an MC file exists. Each
energy is fitted on its own (the rows are independent) and merged into one
output table. ``--state-dict`` additionally writes the fitted
:class:`IonKernelCalibration` parameters, so a calibration is a reproducible
artefact and not just a rewritten table.

The Monte Carlo reference
-------------------------
**The MC data is not shipped with this repository** (~238 GiB). It is published
separately as the Hugging Face dataset ``zimmeryWo/MC_proton_simulation_DoseRAD2026``::

    hf download zimmeryWo/MC_proton_simulation_DoseRAD2026 --repo-type dataset --local-dir <dir>

``--mc-dir`` must point at a directory of per-energy water-phantom edep volumes
in an ITK-readable format (``.mhd``/``.raw``, as written by the Geant4/TOPAS
scoring)::

    <anything>_<energy>MeV<anything>__edep.mhd     preferred (energy deposit)
    <anything>_<energy>MeV<anything>__dose.mhd     fallback (dose)

The energy in MeV is parsed out of the file name and must match a tabulated
energy of the kernel table. Each volume is a **water phantom irradiated by a
single pencil beam entering at the lateral centre**, shaped
``(500, 500, 1500)``: two lateral axes and one depth axis, all at 0.2 mm, i.e.
a +/-50 mm lateral extent and 300 mm of depth. The loss is peak-normalised, so
edep and dose are equivalent here (in a uniform phantom they differ by one
global scalar).

If the directory is missing, empty, or has no file for a requested energy, this
script *fails* rather than falling back to a synthetic reference -- a
calibration fitted to the model's own output looks like it worked.

Recipe notes
------------
* The lateral grid (``--lat-half-mm``) and the IDD normalisation window
  (``--widx-half-mm``) are separate knobs on purpose. The window must match half
  the kernel width the table was built with; leaving it narrow while widening
  the grid re-fits the IDD back down to the narrow integral and cancels the
  wider halo the table was given.
* ``--mask-floor`` hides MC voxels below that fraction of the peak from the
  loss. The default (0.5 %) excludes the far halo; raise it to fit the core
  harder, lower it to let the halo into the objective.
* ``--smooth`` weights the second-difference prior of the calibration. Without
  it a per-sample residual happily fits MC noise.
* ``--weight`` is **not** the MC primary count (1e9 for the published dataset).
  Both doses are peak-normalised before comparison, so it cancels exactly; the
  default is the value the shipped table was fitted with.
* ``--sad-mm`` is 100 m because the MC phantom is irradiated by a parallel pencil
  beam, not by a beam diverging from the machine's own SAD (10 m). It is inert
  here in any case: ``sad_mm`` reaches the engine only through the SSD air-gap
  depth offset, which needs an ``ssd_mm`` this script never passes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

from pydosert.data.ion_beam import IonBeamletBatch
from pydosert.data.ion_machine import IonMachineConfig
from pydosert.engine.ion_dose_engine import IonDoseEngine
from pydosert.physics.kernels.ion_kernel_calibration import IonKernelCalibration
from pydosert.physics.kernels.ion_kernel_table import (
    FORMAT_VERSION,
    EnergyNotInTableError,
    IonKernelTable,
)

#: Default table to calibrate: the one packaged with pydosert.
DEFAULT_TABLE = Path(__file__).parents[1] / "src" / "pydosert" / "data" / "machine_presets" / "protons_doserad.npz"

#: Where to get the Monte Carlo reference. Quoted in every failure message.
MC_DATASET = "zimmeryWo/MC_proton_simulation_DoseRAD2026 (Hugging Face dataset)"

#: Voxel size (mm) of the MC volumes, on all three axes.
MC_VOXEL_MM = 0.2

#: Expected MC volume shape: (lateral, lateral, depth) at :data:`MC_VOXEL_MM`.
MC_SHAPE = (500, 500, 1500)

#: ``_<energy>MeV`` in the file name is the energy of the run.
ENERGY_IN_NAME = re.compile(r"_([0-9]+(?:\.[0-9]+)?)MeV")


class MissingMonteCarloData(SystemExit):
    """Raised (as an exit) when the MC reference is absent or incomplete."""

    def __init__(self, message: str) -> None:
        super().__init__(f"{message}\nThe Monte Carlo reference is published separately as {MC_DATASET}.")


# --------------------------------------------------------------------- MC input


def find_mc_files(mc_dir: str | Path) -> dict[float, Path]:
    """Index the water-phantom MC volumes of a directory by energy.

    Args:
        mc_dir: Directory of per-energy ``*__edep.mhd`` (preferred) or
            ``*__dose.mhd`` volumes.

    Returns:
        ``{energy_mev: path}``, ascending in energy.

    Raises:
        MissingMonteCarloData: If the directory does not exist or holds no
            recognisable MC volume.
    """
    directory = Path(mc_dir)
    if not directory.is_dir():
        raise MissingMonteCarloData(f"--mc-dir {directory} does not exist.")

    files = sorted(directory.glob("*__edep.mhd")) or sorted(directory.glob("*__dose.mhd"))
    if not files:
        raise MissingMonteCarloData(f"--mc-dir {directory} holds no *__edep.mhd or *__dose.mhd volume.")

    indexed: dict[float, Path] = {}
    for path in files:
        match = ENERGY_IN_NAME.search(path.name)
        if match is None:
            raise MissingMonteCarloData(
                f"{path.name} does not carry an energy: the file name must contain '_<energy>MeV'."
            )
        indexed[float(match.group(1))] = path
    return dict(sorted(indexed.items()))


def load_mc_volume(path: str | Path, *, lat_half_mm: int, depth_factor: int) -> np.ndarray:
    """Read one MC volume and rebin it onto the calibration grid.

    The central ``2 * lat_half_mm`` mm of both lateral axes are kept and rebinned
    from 0.2 mm to 1 mm; the depth axis is rebinned by ``depth_factor``
    (``0.2 * depth_factor`` mm per bin).

    Args:
        path: The ``.mhd`` volume.
        lat_half_mm: Half-width in mm of the lateral crop (at most 50).
        depth_factor: Depth rebinning factor; must divide 1500.

    Returns:
        ``(2*lat_half_mm, 2*lat_half_mm, 1500 // depth_factor)`` float32 array.

    Raises:
        MissingMonteCarloData: If SimpleITK cannot be imported or the volume does
            not have the documented shape.
    """
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - SimpleITK is a hard dependency
        raise MissingMonteCarloData(f"SimpleITK is needed to read {path}: {exc}") from exc

    array = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    if tuple(array.shape) != MC_SHAPE:
        raise MissingMonteCarloData(
            f"{path} has shape {tuple(array.shape)}, expected {MC_SHAPE} "
            f"(two lateral axes and one depth axis at {MC_VOXEL_MM} mm)."
        )

    per_mm = int(round(1.0 / MC_VOXEL_MM))
    half = int(lat_half_mm) * per_mm
    centre = MC_SHAPE[0] // 2
    cropped = array[centre - half : centre + half, centre - half : centre + half, : MC_SHAPE[2]]

    n_lat = 2 * int(lat_half_mm)
    n_depth = MC_SHAPE[2] // int(depth_factor)
    return (
        cropped.astype(np.float32)
        .reshape(n_lat, per_mm, n_lat, per_mm, n_depth, int(depth_factor))
        .mean(axis=(1, 3, 5))
    )


# ------------------------------------------------------------------ the fit


def windowed_idd(volume: torch.Tensor, *, centre: int, half: int) -> torch.Tensor:
    """Depth profile of ``volume`` integrated over a lateral window.

    Args:
        volume: ``(lat, lat, depth)`` dose or edep.
        centre: Index of the beam axis on both lateral axes.
        half: Half-width of the integration window in lateral bins (mm).

    Returns:
        ``(depth,)`` windowed integrated depth dose.
    """
    return volume[centre - half : centre + half, centre - half : centre + half, :].sum(dim=(0, 1))


def build_engine(
    table: IonKernelTable,
    *,
    n_lat: int,
    n_depth: int,
    depth_spacing_mm: float,
    lateral_model: str,
    n_sub_beams_per_dim: int,
) -> IonDoseEngine:
    """The water-phantom engine the calibration is fitted through.

    Built once per energy rather than per optimiser step: the constructor
    validates geometry and the engine caches its BEV sampling grids, and neither
    depends on the kernel table. The table is swapped in on every step instead.
    """
    return IonDoseEngine(
        machine_config=IonMachineConfig(),
        kernel_table=table,
        dose_grid_spacing=(1.0, depth_spacing_mm, 1.0),
        dose_grid_shape=(n_lat, n_depth, n_lat),
        field_size=(n_lat, n_lat),
        lateral_model=lateral_model,
        heterogeneous_mcs=False,  # identically zero in water; pure cost here
        n_sub_beams_per_dim=n_sub_beams_per_dim,
    )


def fit_energy(
    table: IonKernelTable,
    energy_mev: float,
    mc_path: Path,
    args: argparse.Namespace,
) -> tuple[IonKernelCalibration, float, float]:
    """Calibrate one energy row against its water-phantom MC volume.

    Args:
        table: The table to calibrate (only this energy's row is touched).
        energy_mev: The tabulated energy to fit.
        mc_path: Its MC volume.
        args: Parsed command line (see :func:`build_parser`).

    Returns:
        ``(calibration, initial_loss, best_loss)``; the calibration holds the
        best-seen parameters, not the last ones.

    Raises:
        RuntimeError: If no gradient reaches the calibration -- the engine would
            then not be differentiable here and the fit would be a no-op.
    """
    device, dtype = table.device, table.dtype
    n_lat = 2 * int(args.lat_half_mm)
    centre = int(args.lat_half_mm)
    window = int(args.widx_half_mm)
    depth_spacing_mm = MC_VOXEL_MM * int(args.depth_factor)
    n_depth = MC_SHAPE[2] // int(args.depth_factor)

    mc = torch.from_numpy(load_mc_volume(mc_path, lat_half_mm=centre, depth_factor=args.depth_factor)).to(
        device=device, dtype=dtype
    )
    mc_normalised = mc / windowed_idd(mc, centre=centre, half=window).max()
    mask = mc_normalised > float(args.mask_floor) * mc_normalised.max()

    calibration = IonKernelCalibration(table, [energy_mev])
    engine = build_engine(
        table,
        n_lat=n_lat,
        n_depth=n_depth,
        depth_spacing_mm=depth_spacing_mm,
        lateral_model=args.lateral_model,
        n_sub_beams_per_dim=args.n_sub_beams_per_dim,
    )
    spot_sigma_mm = (
        float(args.spot_sigma_mm)
        if args.spot_sigma_mm is not None
        else float(table.initial_sigma(energy_mev, table.sad_mm))
    )
    beamlets = IonBeamletBatch.create(
        gantry_angle_deg=[0.0],
        position_mm=[[0.0, 0.0]],
        energy_mev=[float(energy_mev)],
        sigma_mm=[[spot_sigma_mm, spot_sigma_mm]],
        weight=[float(args.weight)],
        iso_center_mm=[centre - 0.5, 0.5 * n_depth * depth_spacing_mm, centre - 0.5],
        sad_mm=float(args.sad_mm),
        device=device,
        dtype=dtype,
    )
    water = torch.ones((n_lat, n_depth, n_lat), device=device, dtype=dtype)
    scored = torch.ones((n_lat, n_depth, n_lat), device=device, dtype=torch.bool)

    def data_and_total_loss() -> tuple[torch.Tensor, torch.Tensor]:
        # This is what used to be a monkey-patch of the table's bound methods: the
        # calibrated table is rebuilt from the residuals and handed to the engine.
        engine.kernel_table = calibration.apply()
        dose = engine.compute_dose(beamlets, water, scored)[0].permute(0, 2, 1)  # (lat, lat, depth)
        normalised = dose / windowed_idd(dose, centre=centre, half=window).max().clamp_min(1e-12)
        data = (normalised - mc_normalised).abs()[mask].mean()
        return data, data + float(args.smooth) * calibration.smoothness_penalty()

    optimiser = torch.optim.Adam(calibration.parameters(), lr=float(args.lr))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=int(args.iters), eta_min=float(args.lr) * 0.02
    )
    initial = float("nan")
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    checked_gradient = False

    for iteration in range(int(args.iters)):
        optimiser.zero_grad(set_to_none=True)
        data, loss = data_and_total_loss()
        # Skip a non-finite step rather than letting one unstable energy poison its
        # row with NaN. If every iteration is non-finite the parameters stay at
        # zero, and the row is written out exactly as it came in.
        if not torch.isfinite(loss):
            print(f"  iter {iteration:4d}  non-finite loss; step skipped", flush=True)
            schedule.step()
            continue
        loss.backward()
        if not checked_gradient:
            gradient = calibration.idd_residual.grad
            if gradient is None or float(gradient.norm()) == 0.0:
                raise RuntimeError(
                    "no gradient reached the calibration: the engine forward is not "
                    "differentiable in this configuration, so the fit would be a no-op"
                )
            checked_gradient = True
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in calibration.parameters()):
            optimiser.zero_grad(set_to_none=True)
            schedule.step()
            continue
        optimiser.step()
        schedule.step()

        value = float(data.detach())
        if np.isnan(initial):
            initial = value
        if value < best:  # the loss is noisy near convergence; keep the best
            best = value
            best_state = {k: v.detach().clone() for k, v in calibration.state_dict().items()}
        if iteration % 50 == 0 or iteration == int(args.iters) - 1:
            print(
                f"  iter {iteration:4d}  masked_L1={value:.6f}  best={best:.6f}  "
                f"lr={schedule.get_last_lr()[0]:.5f}",
                flush=True,
            )

    if best_state is not None:
        calibration.load_state_dict(best_state)
    return calibration, initial, best


# ------------------------------------------------------------------- output


def write_kernel_table_npz(table: IonKernelTable, path: str | Path) -> None:
    """Write a table back out in the archive layout the converter produces.

    Args:
        table: The table to write; curves are stored as float64 whatever the
            table's compute dtype is, exactly like
            ``commissioning/conversion/convert_proton_mat_to_npz.py``.
        path: Destination ``.npz``.
    """

    def array(name: str, dtype=np.float64) -> np.ndarray:
        return getattr(table, name).detach().cpu().numpy().astype(dtype)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            format_version=np.int64(FORMAT_VERSION),
            energy_mev=array("energy_mev"),
            n_valid=array("n_valid", np.int32),
            depth_mm=array("depth_mm"),
            idd=array("idd"),
            sigma=array("sigma_mm"),
            sigma1=array("sigma1_mm"),
            sigma2=array("sigma2_mm"),
            weight=array("weight"),
            offset_mm=array("offset_mm"),
            focus_dist_mm=array("focus_dist_mm"),
            focus_sigma_mm=array("focus_sigma_mm"),
            sad_mm=np.float64(table.sad_mm),
            bams_to_iso_mm=np.float64(table.bams_to_iso_dist_mm),
        )


# ---------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    """The command line of the calibration driver."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mc-dir", required=True, help=f"Directory of water-phantom MC volumes; see {MC_DATASET}")
    parser.add_argument("--table", default=str(DEFAULT_TABLE), help="Kernel table .npz to calibrate")
    parser.add_argument("--out", required=True, help="Calibrated kernel table .npz to write")
    parser.add_argument("--state-dict", default=None, help="Also write the fitted calibration parameters here (.pt)")
    parser.add_argument(
        "--energies",
        default="all",
        help="Comma-separated tabulated energies in MeV to calibrate, or 'all' for every energy with an MC file",
    )
    parser.add_argument("--iters", type=int, default=600, help="Optimiser steps per energy")
    parser.add_argument("--lr", type=float, default=0.03, help="Adam learning rate")
    parser.add_argument("--smooth", type=float, default=2e-4, help="Weight of the depth-smoothness prior")
    parser.add_argument(
        "--depth-factor", type=int, default=2, help="MC depth rebin factor; the depth bin is 0.2 * factor mm"
    )
    parser.add_argument(
        "--lat-half-mm", type=int, default=40, help="Half-width (mm) of the lateral grid; at most 50 (the phantom)"
    )
    parser.add_argument(
        "--widx-half-mm",
        type=int,
        default=37,
        help="Half-width (mm) of the IDD normalisation window; must match half the table's kernel width",
    )
    parser.add_argument(
        "--mask-floor", type=float, default=0.005, help="Exclude MC voxels below this fraction of the peak"
    )
    parser.add_argument("--lateral-model", default="gauss_double", choices=["gauss", "gauss_double"])
    parser.add_argument("--n-sub-beams-per-dim", type=int, default=9, help="Beamlet splitting, n**2 sub-beams")
    parser.add_argument(
        "--weight",
        type=float,
        default=1e7,
        help="Beamlet weight; an arbitrary scale that cancels in the loss, NOT the MC primary count",
    )
    parser.add_argument(
        "--sad-mm",
        type=float,
        default=1e5,
        help="Source-to-axis distance; 1e5 mm is a parallel beam like the MC, and is inert here anyway",
    )
    parser.add_argument(
        "--spot-sigma-mm", type=float, default=None, help="Initial spot sigma; default: the table's initFocus curve"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    return parser


def main(argv: list[str] | None = None) -> int:
    """Fit every requested energy and write the calibrated table."""
    args = build_parser().parse_args(argv)

    if not 1 <= int(args.lat_half_mm) <= MC_SHAPE[0] * MC_VOXEL_MM / 2:
        raise SystemExit(f"--lat-half-mm must be in 1..50 (the phantom is +/-50 mm), got {args.lat_half_mm}")
    if int(args.widx_half_mm) > int(args.lat_half_mm):
        raise SystemExit(
            f"--widx-half-mm ({args.widx_half_mm}) cannot exceed --lat-half-mm ({args.lat_half_mm}): "
            "the normalisation window has to fit inside the grid it is measured on"
        )
    if MC_SHAPE[2] % int(args.depth_factor) != 0:
        raise SystemExit(f"--depth-factor must divide {MC_SHAPE[2]}, got {args.depth_factor}")

    mc_files = find_mc_files(args.mc_dir)
    table = IonKernelTable.load(
        args.table, device=args.device, dtype=torch.float64 if args.dtype == "float64" else torch.float32
    )

    if args.energies.strip().lower() == "all":
        energies = [e for e in mc_files if _is_tabulated(table, e)]
        if not energies:
            raise MissingMonteCarloData(
                f"none of the {len(mc_files)} MC energies in {args.mc_dir} is tabulated in {args.table}."
            )
    else:
        energies = [float(token) for token in args.energies.split(",") if token.strip()]
        missing = [e for e in energies if not any(abs(e - k) < 1e-6 for k in mc_files)]
        if missing:
            raise MissingMonteCarloData(
                f"no MC volume in {args.mc_dir} for energies {missing}; "
                f"available: {[f'{e:.4f}' for e in mc_files]}"
            )

    print(
        f"calibrating {len(energies)} energies of {args.table}\n"
        f"  lateral grid +/-{args.lat_half_mm} mm | IDD window +/-{args.widx_half_mm} mm | "
        f"mask floor {args.mask_floor:g} | {args.device}/{args.dtype}",
        flush=True,
    )

    fitted: dict[str, dict[str, torch.Tensor]] = {}
    for number, energy in enumerate(energies, 1):
        mc_path = mc_files[min(mc_files, key=lambda k: abs(k - energy))]
        print(f"\n=== [{number}/{len(energies)}] {energy:.4f} MeV  <- {mc_path.name} ===", flush=True)
        calibration, initial, best = fit_energy(table, energy, mc_path, args)
        # Each energy owns one row, so chaining the calibrated table accumulates
        # the fits without any of them touching another energy's row. Applied
        # under no_grad so the next energy starts from plain tensors rather than
        # from the previous energy's autograd graph.
        with torch.no_grad():
            table = calibration.apply()
        fitted[f"{energy:.4f}"] = {k: v.detach().cpu() for k, v in calibration.state_dict().items()}
        print(f"  masked L1 {initial:.6f} -> {best:.6f}", flush=True)

    write_kernel_table_npz(table, args.out)
    print(f"\nwrote {args.out} ({len(fitted)} energies calibrated)")
    if args.state_dict:
        torch.save(fitted, args.state_dict)
        print(f"wrote {args.state_dict}")
    return 0


def _is_tabulated(table: IonKernelTable, energy_mev: float) -> bool:
    """Whether ``energy_mev`` is one of the table's energies."""
    try:
        table.row_index(energy_mev)
    except EnergyNotInTableError:
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
