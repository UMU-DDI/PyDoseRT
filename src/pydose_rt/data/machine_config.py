import json
from pathlib import Path
from pydantic import Field, computed_field, model_validator
from pydantic_settings import SettingsConfigDict, BaseSettings
import numpy as np
from typing import Any, Optional

class MachineConfig(BaseSettings):
    preset: Optional[str] = Field(
        default=None,
        description="Optional preset name whose values are merged before validation.",
    )
    tpr_20_10: float = Field(
        description="The tissue phantom ratio TPR20/10"
    )
    number_of_leaf_pairs: int = Field(
        description="The number of leafs"
    )
    minimum_leaf_opening: float = Field(
        default=5.0, description="The minimum opening of the leafs, given in mm."
    )
    minimum_jaw_opening: float = Field(
        default=5.0, description="The minimum opening of the jaws, given in mm."
    )
    maximum_leaf_tip_overlap: float = Field(
        default=150.0, description="The minimum opening of the leafs, given in mm."
    )
    maximum_jaw_speed: float = Field(
        default=22.5, description="The maximum speed of the leafs, given in mm / s."
    )
    maximum_leaf_speed: float = Field(
        default=22.5, description="The maximum speed of the leafs, given in mm / s."
    )
    minimum_gantry_angle_speed: float = Field(
        default=0.1, description="The minimum gantry angle speed defined in deg/s."
    )
    maximum_gantry_angle_speed: float = Field(
        default=6.0, description="The maximum gantry angle speed defined in deg/s."
    )
    maximum_gantry_angle_speed_variation: float = Field(
        default=0.75, description="The maximum gantry angle speed defined in deg/s."
    )
    minimum_dose_rate: float = Field(
        default=0.833, description="The minimum dynamic arc dose rate defined in MU/s."
    )
    maximum_dose_rate: float = Field(
        default=10.0,
        description="The maximum dynamic arc dose rate defined in MU/s.",
    )
    mlc_transmission: float = Field(
        default=0.0,
        description="Transmission rate for closed MLCs as a percentage of the open fluence.",
    )
    penumbra_fwhm: Optional[list[float]] = Field(
        default=None,
        description="Modelled penumbra width in mm. Use two values for different fwhm in MLC and jaw directions, respectively.",
    )
    head_scatter_amplitude: Optional[list[float]] = Field(
        default=None,
        description="Head scatter amplitude as fraction of dose. Use two vales for different amplitudes in MLC and Jaw directions.",
    )
    head_scatter_sigma: Optional[list[float]] = Field(
        default=None,
        description="Head scatter Gaussian sigma in MLC direction in mm",
    )
    head_scatter_ssd_mm: float = Field(
        default=50.0,
        description="Source to scatter-source distance (flattening filter depth) in mm for head scatter model",
    )
    calibration_mu: float = Field(
        default=100,
        description="The mu value for dose calibration in water."
    )
    mean_photon_energy_MeV: float = Field(
        default=10.0, description="Mean photon energy in MeV"
    )
    leaf_widths: Optional[list[float]] = Field(
        default=None, description="A list of the leaf widths" 
    )
    profile_corrections: Optional[list[list[float]]] = Field(
        default=None,
        description="Off-axis correction data: [distances_mm, correction_ratios]"
    )
    output_factors: Optional[list[list[float]]] = Field(
        default=None,
        description="Off-axis correction data: [distances_mm, correction_ratios]"
    )

    
    @staticmethod
    def _load_preset_json(path_str: str) -> dict[str, Any]:
        """
        Read presets/{name}.json and return its dict. Raise a nice error if missing.
        """
        path = Path(path_str)
        name = path.stem
        if not path.is_file():
            raise ValueError(
                f"Unknown preset '{name}' at path '{path}'. "
            )
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Preset file '{path}' must contain a JSON object at the top level.")
        return data

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