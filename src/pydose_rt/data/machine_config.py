import json
from pathlib import Path
from pydantic import Field, computed_field, model_validator
from pydantic_settings import SettingsConfigDict, BaseSettings
import numpy as np
import torch
import math
from typing import Any, Optional
_THIS_DIR = Path(__file__).resolve().parent
_PRESET_DIR_DEFAULT = _THIS_DIR / "machine_presets"   # <--- now relative to this module file

class MachineConfig(BaseSettings):
    preset: Optional[str] = Field(
        default=None,
        description="Optional preset name whose values are merged before validation.",
    )
    ct_array_shape: tuple[int, int, int] = Field(
        default=(320, 128, 128),
        description="Shape of the CT array defining the array voxels",
    )
    resolution: tuple[float, float, float] = Field(
        default=(1.25, 3.125, 3.125),
        description="The resolution in mm of the CT array in the order z,y,x",
    )
    field_size: tuple[int, int] = Field(
        default=(400, 400), description="The field size in the plane given in mm (H,W)"
    )
    downsampling_factor: tuple[int, int, int] = Field(
        default=(1, 1, 1),
        description="The downsampling factor in the order z,y,x",
    )
    # For now, only the first value of the isocenter is used.
    iso_center: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="The distance of the isocenter from the center of the CT volume in mm",
    )
    minimum_leaf_overlap: float = Field(
        default=5.0, description="The minimum opening of the leafs, given in mm."
    )
    minimum_jaw_overlap: float = Field(
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
    is_fff: bool = Field(
        default=True,
        description="Boolean to define if the setup is flattening filter free. (Flattening filters not yet implemented)",
    )
    focal_spot_sigma: float = Field(
        default=0.15,
        description="Effective sigma of the focal spot (source blur) in cm",
    )
    focus_to_collimator: float = Field(
        default=49.7,
        description="Distance from source to collimator (or isocenter) in cm",
    )
    oar_coeffs: tuple[float, float, float] = Field(
        default=(1.0, -1e-4, 2.5e-7),
        description="Off-axis ratio polynomial coefficients (c0, c2, c4)",
    )
    mlc_thickness: float = Field(
        default=6.8, description="MLC physical thickness along beam axis in cm"
    )
    mlc_mu: float = Field(
        default=0.7,
        description="Linear attenuation coefficient (1/cm) for the MLC material",
    )
    dtype: torch.dtype = Field(
        default=torch.float32, description="The data type used for the calculations"
    )

    device: torch.device = Field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        description="The device used for the calculations",
    )

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

    @computed_field(repr=False)
    @property
    def mlc_transmission(self) -> float:
        return math.exp(-self.mlc_mu * self.mlc_thickness)

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

    number_of_leaf_pairs: int = Field(description="The number of leafs")
    tpr_20_10: float = Field(description="The tissue phantom ratio TPR20/10")
    mean_photon_energy_MeV: float = Field(
        default=10.0, description="Mean photon energy in MeV"
    )
    SID: float = Field(default=1000, description="Source-to-isocenter distance in mm")
    number_of_cps: int = Field(
        description="The number of beams for the plan"
    )
    starting_angle: float = Field(
        default=0.0,
        description="The beam angle of the first beam in the series in degrees",
    )
    clockwise: bool = Field(
        default=False,
        description="Determines whether the arc moves clockwise or not.",
    )
    # Private flag to prevent double-validation
    _shapes_adjusted: bool = False

    @model_validator(mode="after")
    def adjust_shapes(self) -> "MachineConfig":
        if self._shapes_adjusted:
            return self
        
        self.ct_array_shape = (
            self.ct_array_shape[0] // self.downsampling_factor[0],
            self.ct_array_shape[1] // self.downsampling_factor[1],
            self.ct_array_shape[2] // self.downsampling_factor[2],
        )

        self.resolution = (
            self.resolution[0] * self.downsampling_factor[0],
            self.resolution[1] * self.downsampling_factor[1],
            self.resolution[2] * self.downsampling_factor[2],
        )

        self._shapes_adjusted = True

        return self

    @computed_field(repr=False)
    @property
    def gantry_angles(self) -> np.ndarray:
        start = math.radians(self.starting_angle)
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

    @computed_field(repr=False)
    @property
    def iso_center_in_pixels(self) -> int:
        return np.ceil(np.divide(self.iso_center, self.resolution)).astype(np.int32)

    @computed_field(repr=False)
    @property
    def field_size_in_pixels(self) -> tuple[int, int]:
        return (
            int(np.ceil(self.field_size[0] / self.resolution[0])),
            int(np.ceil(self.field_size[1] / self.resolution[2])),
        )

    @computed_field(repr=False)
    @property
    def leaf_size(self) -> float:
        return float(self.field_size[1] / self.number_of_leaf_pairs)

    @computed_field(repr=False)
    @property
    def leaf_widths(self) -> np.ndarray:
        if (self.preset == "umea") and (self.number_of_leaf_pairs == 60):
            return np.array(
                [
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                ],
                dtype=np.float32,
            )
        else:
            return np.ones(self.number_of_leaf_pairs, dtype=np.float32) * self.leaf_size

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

    @computed_field(repr=False)
    @property
    def shape_mlc(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            (1, 2, self.number_of_cps, self.number_of_leaf_pairs),
            (1, self.number_of_cps),
        )

    @computed_field(repr=False)
    @property
    def shape_jaws(self) -> np.ndarray:
        return (1, 2, self.number_of_cps)

    @computed_field(repr=False)
    @property
    def shape_fluence_map(self) -> np.ndarray:
        return (
            self.number_of_cps,
            self.field_size_in_pixels[0],
            self.field_size_in_pixels[1],
            1,
        )

    @computed_field(repr=False)
    @property
    def shape_fluence_volume(self) -> np.ndarray:
        return (
            self.number_of_cps,
            self.ct_array_shape[0],
            self.ct_array_shape[1],
            self.ct_array_shape[2],
            1,
        )

    @computed_field(repr=False)
    @property
    def shape_radiological_depth(self) -> np.ndarray:
        return (self.number_of_cps, self.ct_array_shape[0], 1)
    
    @computed_field(repr=False)
    @property
    def physical_size_ct(self) -> np.ndarray:
        return np.multiply(
            np.array(self.ct_array_shape, dtype=np.float32),
            np.array(self.resolution, dtype=np.float32),
        )

    model_config = SettingsConfigDict(
        env_prefix="AUTOPLAN_DM_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
