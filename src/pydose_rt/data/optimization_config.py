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

import json
from pathlib import Path
from pydantic import Field, computed_field, model_validator
from pydantic_settings import SettingsConfigDict, BaseSettings
import numpy as np
import torch
import math
from typing import Any, Optional


class ClinicalCriterion(BaseModel):
    """
    A single clinical acceptance criterion for dose-volume analysis.
    Examples:
    - Bladder V38.5Gy <= 15%: {"criterion_type": "volume_at_dose", "dose_gy": 38.5, "volume_percent": 15, "constraint_type": "at_most"}
    - PTV D99% >= 38.43 Gy: {"criterion_type": "dose_at_volume", "volume_percent": 99, "dose_gy": 38.43, "constraint_type": "at_least"}
    - Rectum D0.01cc <= 45 Gy: {"criterion_type": "dose_at_volume_cc", "volume_cc": 0.01, "dose_gy": 45, "constraint_type": "at_most"}
    """

    criterion_type: str = Field(
        ...,
        description="Type of criterion: 'dose_at_volume', 'dose_at_volume_cc', or 'volume_at_dose'"
    )

    constraint_type: str = Field(
        ...,
        description="Constraint direction: 'at_most' or 'at_least'"
    )

    # Dose parameters
    dose_gy: Optional[float] = Field(
        default=None,description="Dose threshold or value in Gy (absolute)"
    )

    dose_percent: Optional[float] = Field(
        default=None,
        description="Dose as percentage of prescription (e.g., 110 for 110% of prescription)"
    )
    
    # Volume parameters
    volume_percent: Optional[float] = Field(
        default=None,
        description="Volume as percentage (0-100)"
    )

    volume_cc: Optional[float] = Field(
        default=None,
        description="Volume in cubic centimeters"
    )

    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the criterion"
    )

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

    # Clinical acceptance criteria (optional list of explicit criteria)
    clinical_criteria: List[ClinicalCriterion] = Field(
        default_factory=list,
        description="List of clinical acceptance criteria for validation"
    )


class OptimizationConfig(BaseModel):
    """
    Treatment site configuration with constraints.
    
    Can be created from your legacy constraint dict or loaded from JSON.
    """
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    preset: Optional[str] = None
    prescription_gy: Optional[float] = None
    structures: Optional[List[StructureTemplate]] = []

    @staticmethod
    def _load_preset_json(path_str: str) -> dict[str, Any]:
        """
        Read presets/{name}.json and return its dict. Raise a nice error if missing.
        """
        path = Path(path_str)
        name = path.stem
        if not path.is_file():
            # Build a helpful error listing available preset files
            raise ValueError(
                f"Unknown preset '{name}' at path {path}"
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

        preset_values = cls._load_preset_json(name)

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
        