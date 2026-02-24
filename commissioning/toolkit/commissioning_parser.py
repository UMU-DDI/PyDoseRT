"""Commissioning parser: water-tank scans and output factor measurements."""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple, Union

import numpy as np

from .commissioning_types import MeasuredProfile, OutputFactorMeasurement


class MeasurementParser:
    """Parses water tank scans into commissioning types."""

    @staticmethod
    def parse_rfa300(file_path: str) -> List[MeasuredProfile]:
        profiles: List[MeasuredProfile] = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        current_meta = {}
        data_points: List[List[float]] = []

        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("%VNR"):
                if current_meta and data_points:
                    profiles.append(MeasurementParser._build_profile(current_meta, data_points))
                current_meta = {}
                data_points = []
                continue

            if line.startswith("%"):
                parts = line.split()
                tag = parts[0]
                vals = parts[1:]

                if tag == "%FSZ":
                    current_meta["field_size"] = (float(vals[0]), float(vals[1]))
                elif tag == "%SSD":
                    current_meta["ssd"] = float(vals[0])
                elif tag == "%BMT":
                    current_meta["energy"] = f"{float(vals[1]):.0f}MV"
                elif tag == "%SCN":
                    current_meta["type"] = vals[0]
                elif tag == "%STS":
                    current_meta["start"] = [float(v) for v in vals[:3]]
                elif tag == "%EDS":
                    current_meta["end"] = [float(v) for v in vals[:3]]
                elif tag == "%MEA":
                    current_meta["id"] = int(vals[0])
                continue

            if line.startswith("="):
                parts = line.replace("=", "").split()
                if len(parts) >= 4:
                    data_points.append([float(p) for p in parts])

        if current_meta and data_points:
            profiles.append(MeasurementParser._build_profile(current_meta, data_points))

        return profiles

    @staticmethod
    def _build_profile(meta, data) -> MeasuredProfile:
        arr = np.array(data)
        start = np.array(meta.get("start", [0, 0, 0]))
        end = np.array(meta.get("end", [0, 0, 0]))

        diff = np.abs(start - end)
        scan_type_raw = meta.get("type", "PRO")

        if diff[0] > 5.0 and diff[1] > 5.0 and abs(diff[0] - diff[1]) < 10.0:
            scan_type = "DIA"
            axis_name = "D"
            depth = float(start[2])

            r = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
            signs = np.sign(arr[:, 0])
            signs[signs == 0] = 1
            position = r * signs

        elif diff[2] > 10.0 or scan_type_raw == "DPT":
            scan_type = "PDD"
            axis_name = "Z"
            depth = None
            position = arr[:, 2]

        else:
            scan_type = "PRO"
            depth = float(start[2])
            if diff[0] > diff[1]:
                axis_name = "X"
                position = arr[:, 0]
            else:
                axis_name = "Y"
                position = arr[:, 1]

        return MeasuredProfile(
            id=int(meta.get("id", 0)),
            field_size_mm=meta.get("field_size", (100.0, 100.0)),
            depth_mm=depth,
            ssd_mm=float(meta.get("ssd", 1000.0)),
            energy=str(meta.get("energy", "6MV")),
            scan_type=scan_type,
            axis=axis_name,
            position_mm=position,
            dose_values=arr[:, 3],
        )

    @staticmethod
    def find_profile(
        profiles: List[MeasuredProfile],
        field_size: Union[float, Tuple[float, float]],
        depth_mm: float,
        *,
        axis: str = "X",
    ) -> Optional[MeasuredProfile]:
        if isinstance(field_size, (float, int)):
            fs = (float(field_size), float(field_size))
        else:
            fs = field_size
        target_axis = axis.upper()

        for p in profiles:
            if p.axis != target_axis:
                continue
            if target_axis == "Z" and p.scan_type == "PDD":
                pass
            else:
                if p.depth_mm is None or abs(p.depth_mm - depth_mm) > 1.0:
                    continue

            if abs(p.field_size_mm[0] - fs[0]) < 2.0 and abs(p.field_size_mm[1] - fs[1]) < 2.0:
                return p
        return None

    @staticmethod
    def parse_output_factors_csv(file_path: str) -> List[OutputFactorMeasurement]:
        measurements: List[OutputFactorMeasurement] = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            raw = line.strip()
            if not raw:
                continue
            if "X,Y,Z" in raw.upper():
                continue
            parts = raw.split(",")
            try:
                x_cm = float(parts[0])
                y_cm = float(parts[1])
                value = float(parts[2])
            except (ValueError, IndexError):
                continue

            measurements.append(
                OutputFactorMeasurement(
                    field_x_mm=x_cm * 10.0,
                    field_y_mm=y_cm * 10.0,
                    value=value,
                )
            )

        return measurements

    @staticmethod
    def parse_output_factors_json(file_path: str) -> List[OutputFactorMeasurement]:
        """Parse output factor measurements from a JSON file.

        Supported formats:
          - List of objects with ``field_x_mm``, ``field_y_mm``, and ``value``
            (or ``sp`` as a fallback for the value field).
          - Object with a top-level ``"output_factors"`` list using the same
            per-entry schema.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Unwrap optional top-level key
        if isinstance(data, dict):
            data = data.get("output_factors", data)

        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON list (or dict with 'output_factors' list) in {file_path}"
            )

        measurements: List[OutputFactorMeasurement] = []
        for entry in data:
            try:
                fx = float(entry["field_x_mm"])
                fy = float(entry["field_y_mm"])
                # Accept either "value" or "sp" as the OF value
                value = float(entry.get("value", entry.get("sp", 1.0)))
            except (KeyError, TypeError, ValueError):
                continue
            measurements.append(
                OutputFactorMeasurement(field_x_mm=fx, field_y_mm=fy, value=value)
            )

        return measurements
