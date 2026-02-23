"""Machine configuration dataclasses for commissioning, matching the JSON schema."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class MLCConfig:
    model: str
    leaf_boundaries: List[float]
    transmission: float
    dosimetric_leaf_gap_mm: float = 0.0


@dataclass
class MachineConfig:
    energy: str
    gy_per_mu: float
    tpr20_10: float
    mlc: Optional[MLCConfig] = None

    reference_mu: float = 100.0
    reference_dose_gy: float = 1.0

    geometric_penumbra_mm: Tuple[float, float] = (0.0, 0.0)
    head_scatter_sigma_mm: Tuple[float, float] = (0.0, 0.0)
    head_scatter_magnitude: float = 0.0

    profile_curve: Optional[List[Tuple[float, float]]] = None
    output_factor_curve: Optional[List[Tuple[float, float]]] = None

    mlc_transmission: float = 0.0

    @classmethod
    def _from_dict(cls, data: dict, energy: str = "10MV") -> "MachineConfig":
        mlc = None
        if "mlc" in data:
            md = data["mlc"]
            mlc = MLCConfig(
                model=md.get("model", "Unknown"),
                leaf_boundaries=md.get("leaf_boundaries", []),
                transmission=md.get("transmission", 0.0),
                dosimetric_leaf_gap_mm=md.get("dosimetric_leaf_gap_mm", 0.0),
            )

        energies = data.get("energies", {})
        if energy not in energies:
            energy = energy.replace(" ", "")
        if energy not in energies:
            raise ValueError(f"Energy '{energy}' not found in config data")

        e = energies[energy]
        src = e.get("source", {})

        prof_data = e.get("profile", {}).get("curve")
        of_data = e.get("output_factors", {}).get("curve")

        return cls(
            energy=energy,
            gy_per_mu=e.get("gy_per_mu", 1.0),
            tpr20_10=e.get("tpr20_10", 0.7),
            reference_mu=e.get("reference_mu", 100.0),
            reference_dose_gy=e.get("reference_dose_gy", 1.0),
            mlc=mlc,
            geometric_penumbra_mm=tuple(src.get("geometric_penumbra_mm", [0.0, 0.0])),
            head_scatter_sigma_mm=tuple(src.get("head_scatter_sigma_mm", [0.0, 0.0])),
            head_scatter_magnitude=float(src.get("head_scatter_magnitude", 0.0)),
            profile_curve=[tuple(p) for p in prof_data] if prof_data else None,
            output_factor_curve=[tuple(p) for p in of_data] if of_data else None,
            mlc_transmission=mlc.transmission if mlc else 0.0,
        )

    @staticmethod
    def load_from_json(file_path: str, energy: str = "10MV") -> "MachineConfig":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return MachineConfig._from_dict(data, energy)
