import os

from HeroDoseCalc.commissioning_parser import MeasurementParser
from HeroDoseCalc.commissioning_toolkit import CommissioningToolkit
from HeroDoseCalc.commissioning_plotter import CommissioningPlotter


SETTINGS = {
    "config": "examples/commissioning/machine_config_base.json",
    "profiles": "examples/commissioning/data/measurements_10MV/measurements_10_profiles.asc",
    "diagonals": "examples/commissioning/data/measurements_10MV/measurements_10_diagonals.asc",
    "output_factors": "examples/commissioning/data/measurements_10MV/measurements_10_of.csv",
    "energy": "10MV",
    "report_dir": "examples/commissioning/reports/commissioning",
    "step1": "examples/commissioning/machine_config_step1.json",
    "step2": "examples/commissioning/machine_config_step2.json",
    "final": "examples/commissioning/machine_config_complete.json",
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
    "hs_jitter_amp": 0.01,
    "hs_jitter_sigma_mm": 2.0,
}


def _parse_field_pairs_cm(values):
    if not values:
        return None
    pairs = []
    for raw in values:
        parts = raw.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid field size format: {raw}. Use XxY in cm (e.g. 20x20).")
        pairs.append((float(parts[0]) * 10.0, float(parts[1]) * 10.0))
    return pairs


def _parse_bands_pct(values):
    if not values:
        return None
    bands = []
    for raw in values:
        parts = raw.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid band format: {raw}. Use start-end in percent (e.g. 40-90).")
        start = float(parts[0])
        end = float(parts[1])
        if end < start:
            start, end = end, start
        bands.append((start, end))
    return bands


def main() -> int:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    def log_section(title: str) -> None:
        line = "*" * 34
        print(line)
        print(title)
        print(line)
        if SETTINGS["plots"]:
            plotter.log(line)
            plotter.log(title)
            plotter.log(line)

    meas = MeasurementParser()
    plotter = CommissioningPlotter(show=SETTINGS["plots"])
    toolkit = CommissioningToolkit(
        SETTINGS["config"],
        verbose=SETTINGS["verbose"],
        log_callback=plotter.log if SETTINGS["plots"] else None,
    )

    # Settings print suppressed; keep console focused on iteration logs.
    if SETTINGS["run_step1"]:
        log_section("Tuning geometric penumbra")
        profiles = meas.parse_rfa300(SETTINGS["profiles"])
        pen_res = toolkit.fit_geometric_penumbra(
            profiles,
            target_field_mm=(100.0, 100.0),
            target_depth_mm=100.0,
            output_json=SETTINGS["step1"],
            plotter=plotter if SETTINGS["plots"] else None,
        )
        if SETTINGS["verbose"]:
            toolkit._log(
                f"Penumbra final: [{pen_res.geometric_penumbra_mm[0]:.2f}, "
                f"{pen_res.geometric_penumbra_mm[1]:.2f}]"
            )
        toolkit.config_path = SETTINGS["step1"]

    if SETTINGS["run_step2"]:
        log_section("Tuning profile correction")
        diagonals = meas.parse_rfa300(SETTINGS["diagonals"])
        pc_res = toolkit.fit_profile_correction(
            diagonals,
            output_json=SETTINGS["step2"],
            plotter=plotter if SETTINGS["plots"] else None,
        )
        if SETTINGS["verbose"]:
            toolkit._log(f"Profile correction curve points: {len(pc_res.profile_curve)}")
        toolkit.config_path = SETTINGS["step2"]

    if SETTINGS["run_step3"]:
        log_section("Tuning head scatter")
        toolkit.config_path = SETTINGS["step2"]
        of_meas = meas.parse_output_factors_csv(SETTINGS["output_factors"])
        tail_profile_paths = [SETTINGS["profiles"]]
        tail_profiles = []
        for path in tail_profile_paths:
            tail_profiles.extend(meas.parse_rfa300(path))

        of_res = toolkit.fit_output_factors(
            of_meas,
            energy=SETTINGS["energy"],
            output_json=SETTINGS["final"],
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
        if SETTINGS["verbose"]:
            toolkit._log(
                f"HS final: Amplitude: {of_res.head_scatter_magnitude:.4f}, "
                f"Sigma@iso:[{of_res.head_scatter_sigma_mm[0]:.2f}, "
                f"{of_res.head_scatter_sigma_mm[1]:.2f}]"
            )

        if SETTINGS["run_report"]:
            toolkit.config_path = SETTINGS["final"]
            toolkit.finalize_config(SETTINGS["final"], intermediate_files=None)
            plotter.generate_report(
                toolkit=toolkit,
                measurement_files=[SETTINGS["profiles"], SETTINGS["diagonals"]],
                output_dir=SETTINGS["report_dir"],
            )
        else:
            toolkit.finalize_config(SETTINGS["final"], intermediate_files=None)

    if SETTINGS["plots"]:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
