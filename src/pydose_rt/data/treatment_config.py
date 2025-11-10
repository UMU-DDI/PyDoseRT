import json
from pathlib import Path
from pydantic import Field, computed_field, model_validator
from pydantic_settings import SettingsConfigDict, BaseSettings
import numpy as np
import torch
import math
from typing import Any, Optional
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from enum import Enum
import json
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
_PRESET_DIR_DEFAULT = _THIS_DIR / "treatment_presets"   # <--- now relative to this module file



class StructureConstraints(BaseModel):
    """
    Dose constraints for a single structure.
    
    Maps directly to your constraint dictionary structure!
    """
    
    # Dose bounds
    lower_bound_gy: float = Field(
        default=0.0,
        description="Minimum dose constraint (Gy). For targets: prescription dose."
    )
    higher_bound_gy: float = Field(
        default=100.0,
        description="Maximum dose constraint (Gy). For OARs: tolerance dose."
    )
    
    # Volume constraints (DVH)
    lower_bound_target_percent: float = Field(
        default=0.0,
        description="Percentage of volume that must receive ≥ lower_bound_gy. E.g., 95 = D95% ≥ prescription"
    )
    higher_bound_target_percent: float = Field(
        default=100.0,
        description="Percentage of volume that must be ≤ higher_bound_gy. E.g., 75 = V70Gy < 25%"
    )
    
    # Optimization weight
    weight: float = Field(
        default=1.0,
        description="Optimization weight for this structure"
    )


class StructureTemplate(BaseModel):
    """Template for a single structure with all constraints."""
    
    # Identity
    name: str = Field(..., description="Structure name (PTV, Rectum, etc.)")
    color: str = Field(default=None, description="Optional color for plotting")
    
    # Constraints (embedded)
    constraints: StructureConstraints = Field(
        default_factory=StructureConstraints,
        description="Dose constraints for this structure"
    )


class TreatmentConfig(BaseModel):
    """
    Treatment site configuration with constraints.
    
    Can be created from your legacy constraint dict or loaded from JSON.
    """
    
    preset: str
    prescription_gy: float
    structures: List[StructureTemplate]

    @staticmethod
    def _load_preset_json(name: str, base_dir: Path) -> dict[str, Any]:
        """
        Read presets/{name}.json and return its dict. Raise a nice error if missing.
        """
        path = base_dir / f"{name}.json"
        if not path.is_file():
            # Build a helpful error listing available preset files
            available = sorted(p.stem for p in base_dir.glob("*.json"))
            valid = ", ".join(available) if available else "(no presets found)"
            raise ValueError(
                f"Unknown preset '{name}'. Expected a JSON at '{path}'. "
                f"Valid presets: {valid}"
            )
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Preset file '{path}' must contain a JSON object at the top level.")
        return data

    @computed_field(repr=False)
    @property
    def weights(self) -> dict[str, float]:
        return {struct.name: struct.constraints.weight for struct in self.structures}

    @computed_field(repr=False)
    @property
    def lower_bound_gys(self) -> dict[str, float]:
        return {struct.name: struct.constraints.lower_bound_gy for struct in self.structures}
    
    @computed_field(repr=False)
    @property
    def higher_bound_gys(self) -> dict[str, float]:
        return {struct.name: struct.constraints.higher_bound_gy for struct in self.structures}
    
    @computed_field(repr=False)
    @property
    def lower_bound_percents(self) -> dict[str, float]:
        return {struct.name: struct.constraints.lower_bound_target_percent for struct in self.structures}
    
    @computed_field(repr=False)
    @property
    def higher_bound_percents(self) -> dict[str, float]:
        return {struct.name: struct.constraints.higher_bound_target_percent for struct in self.structures}

    @model_validator(mode="before")
    @classmethod
    def _apply_preset(cls, data: Any) -> Any:
        """
        Merge selected preset values from JSON into incoming data before validation.

        Precedence (highest → lowest):
            1) Explicit kwargs (passed to MachineConfig(...))
            2) Environment variables (handled by BaseSettings later)
            3) Preset values (from presets/{name}.json)
            4) Field defaults
        """
        if not isinstance(data, dict):
            # nothing to do if the source isn’t a dict (pydantic internals)
            return data

        name = data.get("preset")
        if not name:
            return data

        preset_values = cls._load_preset_json(name, _PRESET_DIR_DEFAULT)

        # Merge so explicit kwargs in `data` override preset entries.
        # (Env vars will still override later because BaseSettings.)
        merged = {**preset_values, **data}
        return merged

    def randomize_weights(self):
        for struct in self.structures:
            if "PTV" in struct.name or "CTV" in struct.name:
                struct.constraints.weight = 1000
            else:
                struct.constraints.weight = 10**np.random.randint(-3, 3)
        