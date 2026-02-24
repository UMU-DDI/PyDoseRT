"""Minimal commissioning types for step 1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class MeasuredProfile:
    id: int
    field_size_mm: Tuple[float, float]
    depth_mm: float | None
    ssd_mm: float
    energy: str
    scan_type: str
    axis: str
    position_mm: np.ndarray
    dose_values: np.ndarray


@dataclass
class OutputFactorMeasurement:
    field_x_mm: float
    field_y_mm: float
    value: float
    sp: float = 1.0
    sc_meas: float = 1.0
    sc_model: float = 1.0
    residual: float = 1.0
