"""Convert ASC measurement files to JSON format.

Usage:
    python convert_asc_to_json.py <input_asc> <output_json>

Example:
    python convert_asc_to_json.py commissioning/data/measurements_10MV/measurements_10_profiles.asc commissioning/data/profiles_10MV.json
"""
import json
import os
import sys
from typing import Any, Dict, List

from toolkit.commissioning_parser import MeasurementParser
from toolkit.commissioning_types import MeasuredProfile


def profiles_to_json(profiles: List[MeasuredProfile], energy: str = None, ssd_mm: float = None, source: str = "ASC") -> Dict[str, Any]:
    """Convert a list of MeasuredProfile objects to JSON-compatible dict.
    
    Args:
        profiles: List of MeasuredProfile objects
        energy: Energy string (e.g., "10MV"). If None, extracted from first profile.
        ssd_mm: SSD in mm. If None, extracted from first profile.
        source: Source of the data (e.g., "ASC", "MCC")
    
    Returns:
        Dictionary ready to be serialized to JSON
    """
    if not profiles:
        return {"metadata": {}, "measurements": []}
    
    # Extract metadata from profiles
    if energy is None:
        energy = profiles[0].energy
    if ssd_mm is None:
        ssd_mm = profiles[0].ssd_mm
    
    measurements = []
    for profile in profiles:
        measurement = {
            "id": profile.id,
            "field_size_mm": list(profile.field_size_mm),
            "depth_mm": profile.depth_mm,
            "scan_type": profile.scan_type,
            "axis": profile.axis,
            "positions": profile.position_mm.tolist(),
            "doses": profile.dose_values.tolist(),
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


def convert_asc_to_json(input_asc: str, output_json: str, source: str = "ASC") -> None:
    """Convert an ASC file to JSON format.
    
    Args:
        input_asc: Path to input ASC file
        output_json: Path to output JSON file
        source: Source identifier (default: "ASC")
    """
    if not os.path.exists(input_asc):
        raise FileNotFoundError(f"Input file not found: {input_asc}")
    
    # Parse ASC file
    profiles = MeasurementParser.parse_rfa300(input_asc)
    
    # Convert to JSON
    json_data = profiles_to_json(profiles, source=source)
    
    # Write JSON file
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Converted {len(profiles)} profiles from {input_asc} to {output_json}")


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python convert_asc_to_json.py <input_asc> <output_json>")
        print("Example: python convert_asc_to_json.py measurements_10_profiles.asc profiles_10MV.json")
        return 1
    
    input_asc = sys.argv[1]
    output_json = sys.argv[2]
    
    try:
        convert_asc_to_json(input_asc, output_json)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
