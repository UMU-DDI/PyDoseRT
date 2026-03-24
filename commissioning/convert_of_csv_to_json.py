"""Convert output factors CSV to JSON format."""
import csv
import json
import sys


def convert_csv_to_json(csv_file: str, json_file: str) -> None:
    """Convert output factors CSV to JSON."""
    measurements = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # CSV has X, Y (in cm), Z (the output factor value)
            field_x_mm = float(row['X']) * 10.0  # Convert cm to mm
            field_y_mm = float(row['Y']) * 10.0  # Convert cm to mm
            output_factor = float(row['Z'])
            
            measurements.append({
                "field_x_mm": field_x_mm,
                "field_y_mm": field_y_mm,
                "output_factor": output_factor,
            })
    
    with open(json_file, 'w') as f:
        json.dump({"measurements": measurements}, f, indent=2)
    
    print(f"Converted {len(measurements)} output factors from {csv_file} to {json_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_of_csv_to_json.py <input.csv> <output.json>")
        sys.exit(1)
    
    convert_csv_to_json(sys.argv[1], sys.argv[2])
