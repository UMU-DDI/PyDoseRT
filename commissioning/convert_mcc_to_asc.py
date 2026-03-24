#!/usr/bin/env python3
"""
Convert PTW .mcc (CC-Export V1.9) measurement files to RFA300 .asc (BDS) format.

This script reads .mcc profile files from the '10 MV CC/Profiles/' directory
and converts them to the RFA300 ASCII format expected by the
commissioning code (MeasurementParser.parse_rfa300).
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class MccScan:
    """One BEGIN_SCAN … END_SCAN block from a .mcc file."""

    scan_number: int
    meas_date: str            # e.g. "21-Mar-2014 14:33:04"
    energy: float             # MV
    ssd: float                # mm
    field_inplane: float      # mm
    field_crossplane: float   # mm
    scan_curvetype: str       # INPLANE_PROFILE | CROSSPLANE_PROFILE
    scan_depth: float         # mm
    scan_diagonal: str        # NOT_DIAGONAL | FIRST_DIAGONAL | SECOND_DIAGONAL
    scan_direction: str       # POSITIVE | NEGATIVE
    detector_name: str
    wedge_angle: float
    inplane_axis: str         # usually "Y"
    crossplane_axis: str      # usually "X"
    positions: List[float] = field(default_factory=list)   # 1-D positions (mm)
    doses: List[float]     = field(default_factory=list)   # raw dose values
    source_file: str       = ""

def parse_mcc_file(filepath: str) -> List[MccScan]:
    """Parse a PTW CC-Export .mcc file and return all scan blocks."""

    with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
        content = fh.read()

    # Split into individual scan blocks
    block_pattern = re.compile(
        r"BEGIN_SCAN\s+(\d+)(.*?)END_SCAN\s+\d+", re.DOTALL
    )
    scans: List[MccScan] = []
    for m in block_pattern.finditer(content):
        scan = _parse_scan_block(int(m.group(1)), m.group(2), filepath)
        if scan is not None:
            scans.append(scan)
    return scans


def _parse_scan_block(
    scan_number: int, block: str, source_file: str
) -> Optional[MccScan]:
    """Parse a single scan block into an MccScan."""

    def _val(key: str, default: str = "") -> str:
        m = re.search(rf"{key}=(.*)", block)
        return m.group(1).strip() if m else default

    def _fval(key: str, default: float = 0.0) -> float:
        try:
            return float(_val(key))
        except ValueError:
            return default

    data_match = re.search(r"BEGIN_DATA\s*\n(.*?)END_DATA", block, re.DOTALL)
    if not data_match:
        return None

    positions: List[float] = []
    doses: List[float] = []
    for line in data_match.group(1).strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                positions.append(float(parts[0]))
                doses.append(float(parts[1]))
            except ValueError:
                continue
    if not positions:
        return None

    try:
        return MccScan(
            scan_number=scan_number,
            meas_date=_val("MEAS_DATE"),
            energy=_fval("ENERGY"),
            ssd=_fval("SSD"),
            field_inplane=_fval("FIELD_INPLANE"),
            field_crossplane=_fval("FIELD_CROSSPLANE"),
            scan_curvetype=_val("SCAN_CURVETYPE"),
            scan_depth=_fval("SCAN_DEPTH"),
            scan_diagonal=_val("SCAN_DIAGONAL"),
            scan_direction=_val("SCAN_DIRECTION"),
            detector_name=_val("DETECTOR_NAME"),
            wedge_angle=_fval("WEDGE_ANGLE"),
            inplane_axis=_val("INPLANE_AXIS"),
            crossplane_axis=_val("CROSSPLANE_AXIS"),
            positions=positions,
            doses=doses,
            source_file=source_file,
        )
    except Exception as e:
        print(f"  ERROR parsing scan {scan_number} in {source_file}: {e}")
        return None

def mcc_scan_to_xyzd(
    scan: MccScan,
) -> List[Tuple[float, float, float, float]]:
    """
    Map the 1-D position array of an MccScan to (X, Y, Z, Dose) tuples.

    See module docstring for the coordinate conventions.
    """
    depth = scan.scan_depth
    points: List[Tuple[float, float, float, float]] = []

    for pos, dose in zip(scan.positions, scan.doses):
        if scan.scan_diagonal != "NOT_DIAGONAL":
            # ---------- diagonal ----------
            # pos is the distance along the 45° diagonal; decompose into
            # Cartesian X, Y components (each = pos / √2), preserving sign.
            component = pos / 2**0.5
            if scan.scan_curvetype == "INPLANE_PROFILE":
                x, y = component, component
            else:
                x, y = component, -component
        else:
            # ---------- regular profile ----------
            if scan.scan_curvetype == "INPLANE_PROFILE":
                x, y = 0.0, pos
            else:
                x, y = pos, 0.0

        points.append((x, y, depth, dose))

    return points

_MONTH_MAP: Dict[str, str] = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}


def _convert_date(mcc_date: str) -> Tuple[str, str]:
    """
    Convert MCC date  'DD-Mon-YYYY HH:MM:SS'
    to     ASC date   ('DD-MM-YYYY', 'HH:MM:SS').
    """
    try:
        parts = mcc_date.split()
        d, mon, yr = parts[0].split("-")
        time_str = parts[1] if len(parts) > 1 else "00:00:00"
        return f"{d}-{_MONTH_MAP.get(mon, '01')}-{yr}", time_str
    except (IndexError, KeyError, ValueError):
        return "01-01-2000", "00:00:00"

def write_asc_file(
    output_path: str,
    scans: List[MccScan],
    scan_type: str = "PRO",
) -> None:
    """
    Write scan data in the RFA300 BDS ASCII format.

    Parameters
    ----------
    output_path : str
        Destination file path.
    scans : list[MccScan]
        Ordered list of scans to write.
    scan_type : str
        "PRO" for line profiles, "DIA" for diagonal profiles.
    """
    n = len(scans)
    if n == 0:
        print(f"  WARNING: no scans to write for {output_path}")
        return

    with open(output_path, "w", encoding="utf-8") as f:
        # ---- file header ----
        f.write(f":MSR \t{n}\t # No. of measurement in file\n")
        f.write(":SYS BDS 0 # Beam Data Scanner System\n")

        for meas_idx, scan in enumerate(scans, start=1):
            points = mcc_scan_to_xyzd(scan)
            if not points:
                continue

            date_str, time_str = _convert_date(scan.meas_date)

            # %PRD = depth in tenths of mm
            prd = int(round(scan.scan_depth * 10))

            # Detector tag: SEM for semiconductor/diode, ION for ionisation
            fld_tag = (
                "SEM"
                if any(kw in scan.detector_name.lower() for kw in ("diode", "sem"))
                else "ION"
            )

            start_pt = points[0]
            end_pt = points[-1]

            # ---- measurement block ----
            f.write("#\n")
            f.write("# RFA300 ASCII Measurement Dump ( BDS format )\n")
            f.write("#\n")
            f.write(f"# Measurement number \t{meas_idx}\n")
            f.write("#\n")
            f.write("%VNR 1.0\n")
            f.write("%MOD \tRAT\n")
            f.write("%TYP \tSCN \n")
            f.write(f"%SCN \t{scan_type} \n")
            f.write(f"%FLD \t{fld_tag} \n")
            f.write(f"%DAT \t{date_str} \n")
            f.write(f"%TIM \t{time_str} \n")
            f.write(
                f"%FSZ \t{scan.field_inplane:.0f}\t{scan.field_crossplane:.0f}\n"
            )
            f.write(f"%BMT \tPHO\t   {scan.energy:.1f}\n")
            f.write(f"%SSD \t{scan.ssd:.0f}\n")
            f.write("%BUP \t0\n")
            f.write("%BRD \t1000\n")
            f.write("%FSH \t-1\n")
            f.write("%ASC \t0\n")
            f.write("%WEG \t0\n")
            f.write("%GPO \t0\n")
            f.write("%CPO \t0\n")
            f.write(f"%MEA \t{meas_idx}\n")
            f.write(f"%PRD \t{prd}\n")
            f.write(f"%PTS \t{len(points)}\n")
            f.write(
                f"%STS \t{start_pt[0]:8.1f}\t{start_pt[1]:8.1f}"
                f"\t{start_pt[2]:8.1f}"
                " # Start Scan values in mm ( X , Y , Z )\n"
            )
            f.write(
                f"%EDS \t{end_pt[0]:8.1f}\t{end_pt[1]:8.1f}"
                f"\t{end_pt[2]:8.1f}"
                " # End Scan values in mm ( X , Y , Z )\n"
            )
            f.write("#\n")
            f.write("#\t  X      Y      Z     Dose\n")
            f.write("#\n")

            # ---- data points ----
            for x, y, z, dose in points:
                f.write(f"= \t{x:8.1f}\t{y:8.1f}\t{z:8.1f}\t{dose:8.4f}\n")

            f.write(":EOM  # End of Measurement\n")

    print(f"  Written {n} measurements to {output_path}")

def collect_mcc_scans(
    profiles_dir: str,
) -> Tuple[List[MccScan], List[MccScan]]:
    """
    Walk `profiles_dir`, parse every .mcc file, and partition scans into
    regular profiles (NOT_DIAGONAL) and diagonal profiles.

    Within each group the scans are sorted by
        (field_inplane, field_crossplane, scan_curvetype, scan_depth)
    so that the output .asc file is ordered consistently.

    Returns
    -------
    profiles, diagonals : tuple[list[MccScan], list[MccScan]]
    """
    profiles: List[MccScan] = []
    diagonals: List[MccScan] = []

    mcc_files = sorted(Path(profiles_dir).glob("*.mcc"))
    if not mcc_files:
        print(f"  WARNING: no .mcc files found in {profiles_dir}")
        return profiles, diagonals

    for mcc_path in mcc_files:
        print(f"  Parsing {mcc_path.name} …")
        scans = parse_mcc_file(str(mcc_path))
        for s in scans:
            # Skip wedged fields (only open beams)
            if s.wedge_angle > 0.01:
                continue
            if s.scan_diagonal != "NOT_DIAGONAL":
                diagonals.append(s)
            else:
                profiles.append(s)

    # Sort: group by field size, then inplane before crossplane, then depth
    def _sort_key(s: MccScan) -> Tuple:
        curve_order = 0 if s.scan_curvetype == "INPLANE_PROFILE" else 1
        return (s.field_inplane, s.field_crossplane, curve_order, s.scan_depth)

    profiles.sort(key=_sort_key)
    diagonals.sort(key=_sort_key)

    return profiles, diagonals


def convert_profiles_directory(
    profiles_dir: str,
    output_dir: str,
    *,
    profiles_filename: str = "converted_10_profiles.asc",
    diagonals_filename: str = "converted_10_diagonals.asc",
) -> None:
    """
    Main entry point: convert all .mcc files in *profiles_dir* into two
    RFA300 .asc files placed in *output_dir*.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MCC → ASC converter")
    print(f"  Source : {profiles_dir}")
    print(f"  Output : {output_dir}")
    print(f"{'='*60}\n")

    profiles, diagonals = collect_mcc_scans(profiles_dir)

    print(f"\n  Found {len(profiles)} profile scans, "
          f"{len(diagonals)} diagonal scans.\n")

    # --- write profiles ---
    if profiles:
        out_profiles = os.path.join(output_dir, profiles_filename)
        write_asc_file(out_profiles, profiles, scan_type="PRO")
    else:
        print("  No profiles to write.")

    # --- write diagonals ---
    if diagonals:
        out_diags = os.path.join(output_dir, diagonals_filename)
        write_asc_file(out_diags, diagonals, scan_type="DIA")
    else:
        print("  No diagonals to write.")

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Default paths (relative to this script's location)
    script_dir = Path(__file__).resolve().parent
    default_profiles_dir = script_dir / "data" / "Profiles"
    default_output_dir   = script_dir / "data" / "test"

    profiles_dir = sys.argv[1] if len(sys.argv) > 1 else str(default_profiles_dir)
    output_dir   = sys.argv[2] if len(sys.argv) > 2 else str(default_output_dir)

    convert_profiles_directory(profiles_dir, output_dir)