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


class TreatmentConfig(BaseModel):
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

    plan_iso_center: Optional[tuple[float, float, float]] = None
    plan_mlcs: Optional[np.array] = None
    plan_mus: Optional[np.array] = None
    plan_jaws: Optional[np.array] = None
    plan_clockwise: Optional[bool] = None
    plan_starting_angle: Optional[float] = None

    kernel_size: int = Field(
        default=15,
        description="Kernel size to use during convolution",
    )
    fluence_kernel_size: int = Field(
        default=0,
        description="Kernel size for trainable fluence map",
    )
    field_size: tuple[int, int] = Field(
        default=(400, 400), description="The field size in the plane given in mm (H,W)"
    )
    downsampling_factor: tuple[int, int, int] = Field(
        default=(1, 1, 1),
        description="The downsampling factor in the order z,y,x",
    )
    dtype: torch.dtype = Field(
        default=torch.float32, description="The data type used for the calculations"
    )

    device: torch.device = Field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        description="The device used for the calculations",
    )
    @property
    def lookup_table(self) -> np.ndarray:
        return np.array(
            [
                [-1000, 0.0],  # SAT: Added for safety
                [-992, 0.00109],
                [-960, 0.00109],
                [-500, 0.5],
                [-75, 0.95],
                [42, 1.04],
                [85, 1.08],
                [490, 1.29],
                [890, 1.52],
                [1240, 1.72],
                [1670, 1.95],
                [2155, 2.15],
                [2640, 2.34],
                [2832, 2.46],
                [2840, 6.6],
            ],
            dtype=np.float32,
        )

    number_of_cps: int = Field(
        description="The number of beams for the plan"
    )
    starting_angle: Optional[float] = Field(
        default=0.0,
        description="The beam angle of the first beam in the series in degrees",
    )
    beam_limiting_device_angle: Optional[float] = Field(
        default=0.0,
        description="The beam limiting device angle in degrees for the in-plane fluence rotation.",
    )
    clockwise: Optional[bool] = Field(
        default=False,
        description="Determines whether the arc moves clockwise or not.",
    )
    iso_center: Optional[tuple[float, float, float]] = Field(
        default=(0.0, 0.0, 0.0),
        description="The distance of the isocenter from the center of the CT volume in mm",
    )
    SID: float = Field(default=1000, description="Source-to-isocenter distance in mm")

    @computed_field(repr=False)
    @property
    def gantry_angles(self) -> np.ndarray:
        start = math.radians(self.starting_angle)
        if self.number_of_cps == 1:
            return [ start ]
        
        if self.clockwise:
            end = math.radians(self.starting_angle) + math.radians(360)  
        else:
            end = math.radians(self.starting_angle) - math.radians(360)

        return np.linspace(
            start, 
            end,
            self.number_of_cps + 2,
            endpoint=False,
        )[:-2] % (2 * math.pi)

    @computed_field(repr=False)
    @property
    def depth_offset(self) -> np.ndarray:
        return self.SID - self.iso_center[1]

    @computed_field(repr=False)
    @property
    def gantry_diff(self) -> float:
        return float(math.radians(360) / self.number_of_cps)

    @computed_field(repr=False)
    @property
    def gantry_diff_deg(self) -> float:
        return float(360.0 / self.number_of_cps)

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
        