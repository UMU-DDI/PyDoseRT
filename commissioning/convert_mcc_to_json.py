"""Convert PTW .mcc measurement files to JSON format.

Usage:
    python convert_mcc_to_json.py <profiles_dir> <output_dir>

Example:
    python convert_mcc_to_json.py data/Profiles data/

This script reads .mcc profile files and converts them to JSON format,
partitioning into profiles and diagonals.
"""
import json
import os
import sys
from typing import Any, Dict, List

from convert_mcc_to_asc import (
    MccScan,
    collect_mcc_scans,
    mcc_scan_to_xyzd,
)


def mcc_scans_to_json(
    scans: List[MccScan],
    energy: str = None,
    ssd_mm: float = None,
    source: str = "MCC",
) -> Dict[str, Any]:
    """Convert a list of MccScan objects to JSON-compatible dict.
    
    Args:
        scans: List of MccScan objects
        energy: Energy string (e.g., "10MV"). If None, extracted from first scan.
        ssd_mm: SSD in mm. If None, extracted from first scan.
        source: Source of the data (default: "MCC")
    
    Returns:
        Dictionary ready to be serialized to JSON
    """
    if not scans:
        return {"metadata": {}, "measurements": []}
    
    # Extract metadata from scans
    if energy is None:
        energy = f"{scans[0].energy:.0f}MV"
    if ssd_mm is None:
        ssd_mm = scans[0].ssd
    
    measurements = []
    for scan_idx, scan in enumerate(scans):
        # Convert MCC coordinates to (X, Y, Z, Dose) tuples
        points = mcc_scan_to_xyzd(scan)
        
        if not points:
            continue
        
        # Determine scan type (PRO or DIA) and axis from scan metadata
        if scan.scan_diagonal != "NOT_DIAGONAL":
            scan_type = "DIA"
            axis = "D"
        else:
            scan_type = "PRO"
            # Determine axis from scan curve type
            if scan.scan_curvetype == "INPLANE_PROFILE":
                axis = "Y"  # PTW convention: inplane = Y axis
            else:
                axis = "X"  # PTW convention: crossplane = X axis
        
        # Extract positions and doses from converted points
        positions = [pt[0] if axis == "X" else pt[1] if axis == "Y" else None for pt in points]
        doses = [pt[3] for pt in points]
        
        # For diagonal, we need radial distance
        if scan_type == "DIA":
            positions = []
            for pt in points:
                # Recalculate radial distance with sign preservation
                x, y = pt[0], pt[1]
                r = (x**2 + y**2)**0.5
                # Preserve sign: positive if x >= 0, negative otherwise
                positions.append(r if x >= 0 else -r)
        
        # Determine field size (use inplane and crossplane)
        field_size_mm = (scan.field_inplane, scan.field_crossplane)
        
        measurement = {
            "id": scan_idx + 1,
            "field_size_mm": list(field_size_mm),
            "depth_mm": scan.scan_depth,
            "scan_type": scan_type,
            "axis": axis,
            "positions": positions,
            "doses": doses,
        }
        measurements.append(measurement)
    
    return {
        "metadata": {
            "energy": energy,
            "ssd_mm": ssd_mm,
            "source": source,
        },
        "measurements": measurements,
    }


def convert_mcc_to_json(
    profiles_dir: str,
    output_dir: str,
    *,
    profiles_filename: str = "profiles_10MV.json",
    diagonals_filename: str = "diagonals_10MV.json",
) -> None:
    """Main entry point: convert all .mcc files in profiles_dir to JSON.
    
    Args:
        profiles_dir: Directory containing .mcc files
        output_dir: Directory where JSON files will be written
        profiles_filename: Output filename for profiles
        diagonals_filename: Output filename for diagonals
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  MCC → JSON converter")
    print(f"  Source : {profiles_dir}")
    print(f"  Output : {output_dir}")
    print(f"{'='*60}\n")
    
    # Collect and partition scans
    profiles, diagonals = collect_mcc_scans(profiles_dir)
    
    print(f"\n  Found {len(profiles)} profile scans, "
          f"{len(diagonals)} diagonal scans.\n")
    
    # Convert profiles to JSON
    if profiles:
        json_data = mcc_scans_to_json(profiles, source="MCC")
        out_path = os.path.join(output_dir, profiles_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        print(f"  Written {len(profiles)} profiles to {out_path}")
    else:
        print("  No profiles to write.")
    
    # Convert diagonals to JSON
    if diagonals:
        json_data = mcc_scans_to_json(diagonals, source="MCC")
        out_path = os.path.join(output_dir, diagonals_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        print(f"  Written {len(diagonals)} diagonals to {out_path}")
    else:
        print("  No diagonals to write.")
    
    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}\n")


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python convert_mcc_to_json.py <profiles_dir> [output_dir]")
        print("Example: python convert_mcc_to_json.py data/Profiles data/")
        return 1
    
    profiles_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    
    try:
        convert_mcc_to_json(profiles_dir, output_dir)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
