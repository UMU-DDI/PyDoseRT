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
        default=50.0, description="The minimum dynamic arc dose rate defined in MU/min."
    )
    maximum_dose_rate: float = Field(
        default=600.0,
        description="The maximum dynamic arc dose rate defined in MU/min.",
    )
    penumbra_fwhm_mlc: float = Field(
        default=1.5,
        description="Modelled penumbra width (FWHM) in MLC direction (horizontal/width) in mm.",
    )
    penumbra_fwhm_jaw: float = Field(
        default=1.5,
        description="Modelled penumbra width (FWHM) in JAW direction (vertical/height) in mm.",
    )
    head_scatter_amplitude_mlc: float = Field(
        default=0.0,
        description="Head scatter amplitude in MLC direction as fraction of dose (e.g., 0.04 = 4%)",
    )
    head_scatter_amplitude_jaw: float = Field(
        default=0.0,
        description="Head scatter amplitude in JAW direction as fraction of dose (e.g., 0.06 = 6%)",
    )
    head_scatter_sigma_mlc_mm: Optional[float] = Field(
        default=45.0,
        description="Head scatter Gaussian sigma in MLC direction in mm",
    )
    head_scatter_sigma_jaw_mm: float = Field(
        default=49.0,
        description="Head scatter Gaussian sigma in JAW direction in mm",
    )
    tpr_20_10: float = Field(
        description="The tissue phantom ratio TPR20/10"
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
    number_of_leaf_pairs: int = Field(description="The number of leafs")
    profile_corrections: Optional[list[list[float]]] = Field(
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


    @computed_field(repr=False)
    @property
    def fluence_profile(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array(
                [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    7.5,
                    10.0,
                    12.5,
                    15.0,
                    17.5,
                    20.0,
                    25.0,
                    25.25,
                    25.75,
                    26.0,
                    50.0,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    1.0,
                    0.989,
                    0.949,
                    0.9,
                    0.85,
                    0.795,
                    0.685,
                    0.598,
                    0.522,
                    0.465,
                    0.415,
                    0.372,
                    0.294,
                    0.22,
                    0.1,
                    0.03,
                    0.03,
                ],
                dtype=np.float32,
            ),
        )