"""Commissioning pipeline using the PyDoseRT dose engine.

Run from the repository root:

    python commissioning/run_commissioning_pipeline.py

Edit the configuration variables below, then run.  All three steps
(penumbra → profile correction → head scatter / output factors) are
always executed in order.  The final output is one machine-config JSON
per energy written to OUTPUT_DIR.
"""
import os

from toolkit.commissioning_parser import MeasurementParser
from toolkit.commissioning_toolkit import CommissioningToolkit
from toolkit.commissioning_plotter import CommissioningPlotter

# ---------------------------------------------------------------------------
# Input files - .json is expected. To convert raw data, there are functions 
# `convert_asc_to_json`, `convert_mcc_to_json`, `convert_of_csv_to_json`.
# ---------------------------------------------------------------------------
BASE_CONFIG         = "commissioning/machine_config_base_varian.json"
PROFILES_FILE       = "commissioning/data/umea/profiles_10MV.json"
DIAGONALS_FILE      = "commissioning/data/umea/diagonals_10MV.json"
OUTPUT_FACTORS_FILE = "commissioning/data/umea/output_factors_10MV.json"

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR  = "commissioning"
REPORT_DIR  = "commissioning/reports/commissioning"

# ---------------------------------------------------------------------------
# General options
# ---------------------------------------------------------------------------
ENERGY      = "10MV"
SHOW_PLOTS  = True
VERBOSE     = True

# ---------------------------------------------------------------------------
# Kernel size
# The dose engine kernel is commissioned with a large kernel size which vcan be used for computations with smaller kernel sizes. This is a known limitation, but has empirically shown good results.
# ---------------------------------------------------------------------------
KERNEL_SIZE_MM = 400.0

# ---------------------------------------------------------------------------
# Step 1 – geometric penumbra
# Fits geometric_penumbra_mm to match the measured 20–80 % penumbra width
# on the specified reference field and depth.
# ---------------------------------------------------------------------------
PENUMBRA_FIELD_MM = (100.0, 100.0)
PENUMBRA_DEPTH_MM = 100.0

# ---------------------------------------------------------------------------
# Step 2 – off-axis profile correction
# Builds a radial correction curve from the largest diagonal (or crossline)
# profile.  Only plateau points are used for the fit:
#   - dose must exceed PROFILE_PLATEAU_DOSE_THRESHOLD (fraction of CAX)
#   - position must be within PROFILE_PLATEAU_POSITION_FRACTION of the
#     reference field half-width
# For diagonal profiles the curve is tapered to zero beyond the field corner
# defined by PROFILE_DIAGONAL_CUTOFF_DEG (beam's-eye-view half-angle).
# ---------------------------------------------------------------------------
PROFILE_PLATEAU_DOSE_THRESHOLD    = 0.75   # fraction of CAX dose
PROFILE_PLATEAU_POSITION_FRACTION = 0.85   # fraction of field half-width
PROFILE_DIAGONAL_CUTOFF_DEG       = 13.0   # beam corner half-angle for diagonal taper

# ---------------------------------------------------------------------------
# Step 3 – head scatter / output factors
# ---------------------------------------------------------------------------
HS_AXES              = ["X", "Y"]
HS_DEPTHS_MM         = [100.0]
HS_FIELDS_CM         = ["10x10", "20x20"]   # field sizes used for tail-profile matching
HS_BANDS_PCT         = ["40-90", "110-150"]  # dose-range bands for the scatter fit
HS_BAND_WEIGHTS      = [100.0, 5.0]
HS_PLATEAU_WINDOW    = 6
HS_PLATEAU_RTOL      = 1e-4
HS_PLATEAU_MAX_RESTARTS = 5
HS_JITTER_AMP        = 0.005
HS_JITTER_SIGMA_MM   = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_field_pairs_cm(values):
    pairs = []
    for raw in values:
        parts = raw.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid field size format: {raw!r}. Use XxY in cm (e.g. 20x20).")
        pairs.append((float(parts[0]) * 10.0, float(parts[1]) * 10.0))
    return pairs


def _parse_bands_pct(values):
    bands = []
    for raw in values:
        parts = raw.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid band format: {raw!r}. Use start-end in percent (e.g. 40-90).")
        start, end = float(parts[0]), float(parts[1])
        if end < start:
            start, end = end, start
        bands.append((start, end))
    return bands


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    plotter = CommissioningPlotter(show=SHOW_PLOTS)
    toolkit = CommissioningToolkit(
        BASE_CONFIG,
        verbose=VERBOSE,
        log_callback=plotter.log if SHOW_PLOTS else None,        kernel_size_mm=KERNEL_SIZE_MM,
    )

    def log_section(title: str) -> None:
        line = "*" * 34
        print(line)
        print(title)
        print(line)
        if SHOW_PLOTS:
            plotter.log(line)
            plotter.log(title)
            plotter.log(line)

    # ── Step 1: geometric penumbra ────────────────────────────────────────────
    log_section("Tuning geometric penumbra")
    profiles = MeasurementParser.parse_json_profiles(PROFILES_FILE)
    pen_res = toolkit.fit_geometric_penumbra(
        profiles,
        target_field_mm=PENUMBRA_FIELD_MM,
        target_depth_mm=PENUMBRA_DEPTH_MM,
        plotter=plotter if SHOW_PLOTS else None,
    )
    toolkit._log(
        f"Penumbra final: [{pen_res.geometric_penumbra_mm[0]:.2f}, "
        f"{pen_res.geometric_penumbra_mm[1]:.2f}]"
    )

    # ── Step 2: off-axis profile correction ───────────────────────────────────
    log_section("Tuning profile correction")
    diagonals = MeasurementParser.parse_json_profiles(DIAGONALS_FILE)
    pc_res = toolkit.fit_profile_correction(
        diagonals,
        plateau_dose_threshold=PROFILE_PLATEAU_DOSE_THRESHOLD,
        plateau_position_fraction=PROFILE_PLATEAU_POSITION_FRACTION,
        diagonal_cutoff_deg=PROFILE_DIAGONAL_CUTOFF_DEG,
        plotter=plotter if SHOW_PLOTS else None,
    )
    toolkit._log(f"Profile correction curve points: {len(pc_res.profile_curve)}")

    # ── Step 3: head scatter / output factors ─────────────────────────────────
    log_section("Tuning head scatter")
    if OUTPUT_FACTORS_FILE.endswith(".json"):
        of_meas = MeasurementParser.parse_output_factors_json(OUTPUT_FACTORS_FILE)
    else:
        of_meas = MeasurementParser.parse_output_factors_csv(OUTPUT_FACTORS_FILE)

    of_res = toolkit.fit_output_factors(
        of_meas,
        energy=ENERGY,
        tail_profiles=profiles,
        axes=HS_AXES,
        depths_mm=HS_DEPTHS_MM,
        fields_mm=_parse_field_pairs_cm(HS_FIELDS_CM),
        bands_pct=_parse_bands_pct(HS_BANDS_PCT),
        band_weights=HS_BAND_WEIGHTS,
        plateau_window=HS_PLATEAU_WINDOW,
        plateau_rtol=HS_PLATEAU_RTOL,
        plateau_max_restarts=HS_PLATEAU_MAX_RESTARTS,
        jitter_amp=HS_JITTER_AMP,
        jitter_sigma_mm=HS_JITTER_SIGMA_MM,
        plotter=plotter if SHOW_PLOTS else None,
    )
    toolkit._log(
        f"HS final: Amplitude: {of_res.head_scatter_magnitude:.4f}, "
        f"Sigma@iso: [{of_res.head_scatter_sigma_mm[0]:.2f}, "
        f"{of_res.head_scatter_sigma_mm[1]:.2f}]"
    )

    toolkit.finalize_config()
    plotter.generate_report(
        toolkit=toolkit,
        measurement_files=[PROFILES_FILE, DIAGONALS_FILE],
        output_dir=REPORT_DIR,
    )

    machine_config_paths = toolkit.export_config(output_dir=OUTPUT_DIR)
    for energy, path in machine_config_paths.items():
        toolkit._log(f"Machine config ({energy}): {path}")

    if SHOW_PLOTS:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
