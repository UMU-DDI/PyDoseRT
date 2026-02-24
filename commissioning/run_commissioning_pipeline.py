"""Commissioning pipeline using the PyDoseRT dose engine.

Run from the repository root:

    python commissioning/run_commissioning_pipeline.py

Configuration state is held in memory throughout the pipeline.  Toggle
``run_stepN`` flags in SETTINGS to run individual steps.  The final output
is one machine-config JSON per energy written to ``output_dir``.
"""
import os

from toolkit.commissioning_parser import MeasurementParser
from toolkit.commissioning_toolkit import CommissioningToolkit
from toolkit.commissioning_plotter import CommissioningPlotter


SETTINGS = {
    "config": "commissioning/machine_config_base.json",
    "profiles": "commissioning/data/measurements_10MV/measurements_10_profiles.asc",
    "diagonals": "commissioning/data/measurements_10MV/measurements_10_diagonals.asc",
    # JSON file produced by the clinic's OF measurement workflow
    "output_factors": "commissioning/data/measurements_10MV/measurements_10_of_sp.json",
    "energy": "10MV",
    "report_dir": "commissioning/reports/commissioning",
    "output_dir": "commissioning",
    "run_step1": True,
    "run_step2": True,
    "run_step3": True,
    "run_report": True,
    "plots": True,
    "verbose": True,
    "hs_bands_pct": ["40-90", "110-150"],
    "hs_band_weights": [100.0, 5.0],
    "hs_axes": ["X", "Y"],
    "hs_depths_mm": [100.0],
    "hs_fields_cm": ["10x10", "20x20"],
    "hs_plateau_window": 6,
    "hs_plateau_rtol": 1e-4,
    "hs_plateau_max_restarts": 5,
    "hs_jitter_amp": 0.005,
    "hs_jitter_sigma_mm": 1.0,
}


def _parse_field_pairs_cm(values):
    if not values:
        return None
    pairs = []
    for raw in values:
        parts = raw.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid field size format: {raw!r}. Use XxY in cm (e.g. 20x20).")
        pairs.append((float(parts[0]) * 10.0, float(parts[1]) * 10.0))
    return pairs


def _parse_bands_pct(values):
    if not values:
        return None
    bands = []
    for raw in values:
        parts = raw.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid band format: {raw!r}. Use start-end in percent (e.g. 40-90)."
            )
        start, end = float(parts[0]), float(parts[1])
        if end < start:
            start, end = end, start
        bands.append((start, end))
    return bands


def main() -> int:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    plotter = CommissioningPlotter(show=SETTINGS["plots"])
    toolkit = CommissioningToolkit(
        SETTINGS["config"],
        verbose=SETTINGS["verbose"],
        log_callback=plotter.log if SETTINGS["plots"] else None,
    )

    def log_section(title: str) -> None:
        line = "*" * 34
        print(line)
        print(title)
        print(line)
        if SETTINGS["plots"]:
            plotter.log(line)
            plotter.log(title)
            plotter.log(line)

    # ── Step 1: geometric penumbra ────────────────────────────────────────────
    if SETTINGS["run_step1"]:
        log_section("Tuning geometric penumbra")
        profiles = MeasurementParser.parse_rfa300(SETTINGS["profiles"])
        pen_res = toolkit.fit_geometric_penumbra(
            profiles,
            target_field_mm=(100.0, 100.0),
            target_depth_mm=100.0,
            plotter=plotter if SETTINGS["plots"] else None,
        )
        toolkit._log(
            f"Penumbra final: [{pen_res.geometric_penumbra_mm[0]:.2f}, "
            f"{pen_res.geometric_penumbra_mm[1]:.2f}]"
        )

    # ── Step 2: off-axis profile correction ───────────────────────────────────
    if SETTINGS["run_step2"]:
        log_section("Tuning profile correction")
        diagonals = MeasurementParser.parse_rfa300(SETTINGS["diagonals"])
        pc_res = toolkit.fit_profile_correction(
            diagonals,
            plotter=plotter if SETTINGS["plots"] else None,
        )
        toolkit._log(f"Profile correction curve points: {len(pc_res.profile_curve)}")

    # ── Step 3: head scatter / output factors ─────────────────────────────────
    if SETTINGS["run_step3"]:
        log_section("Tuning head scatter")

        # Accept both JSON and CSV output-factor files automatically.
        of_path = SETTINGS["output_factors"]
        if of_path.endswith(".json"):
            of_meas = MeasurementParser.parse_output_factors_json(of_path)
        else:
            of_meas = MeasurementParser.parse_output_factors_csv(of_path)

        tail_profiles = MeasurementParser.parse_rfa300(SETTINGS["profiles"])

        of_res = toolkit.fit_output_factors(
            of_meas,
            energy=SETTINGS["energy"],
            tail_profiles=tail_profiles,
            axes=SETTINGS["hs_axes"],
            depths_mm=SETTINGS["hs_depths_mm"],
            fields_mm=_parse_field_pairs_cm(SETTINGS["hs_fields_cm"]),
            bands_pct=_parse_bands_pct(SETTINGS["hs_bands_pct"]),
            band_weights=SETTINGS["hs_band_weights"],
            plateau_window=SETTINGS["hs_plateau_window"],
            plateau_rtol=SETTINGS["hs_plateau_rtol"],
            plateau_max_restarts=SETTINGS["hs_plateau_max_restarts"],
            jitter_amp=SETTINGS["hs_jitter_amp"],
            jitter_sigma_mm=SETTINGS["hs_jitter_sigma_mm"],
            plotter=plotter if SETTINGS["plots"] else None,
        )
        toolkit._log(
            f"HS final: Amplitude: {of_res.head_scatter_magnitude:.4f}, "
            f"Sigma@iso: [{of_res.head_scatter_sigma_mm[0]:.2f}, "
            f"{of_res.head_scatter_sigma_mm[1]:.2f}]"
        )

        toolkit.finalize_config()
        if SETTINGS["run_report"]:
            plotter.generate_report(
                toolkit=toolkit,
                measurement_files=[SETTINGS["profiles"], SETTINGS["diagonals"]],
                output_dir=SETTINGS["report_dir"],
            )

        machine_config_paths = toolkit.export_config(
            output_dir=SETTINGS["output_dir"],
        )
        for energy, path in machine_config_paths.items():
            toolkit._log(f"Machine config ({energy}): {path}")

    if SETTINGS["plots"]:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
