from pydantic import Field, computed_field, model_validator
from pydantic_settings import SettingsConfigDict, BaseSettings
import numpy as np
import torch
import math
from typing import Any, Dict, ClassVar, Optional


class ModelConfig(BaseSettings):
    preset: Optional[str] = Field(
        default=None,
        description="Optional preset name whose values are merged before validation.",
    )
    ct_array_shape: tuple[int, int, int] = Field(
        default=(320, 128, 128),
        description="Shape of the CT array defining the array voxels",
    )
    resolution: tuple[float, float, float] = Field(
        default=(0.125, 0.3125, 0.3125),
        description="The resolution in cm of the CT array in the order z,y,x",
    )
    field_size: tuple[int, int] = Field(
        default=(40, 40), description="The field size in the plane given in cm (H,W)"
    )
    downsampling_factor: tuple[int, int, int] = Field(
        default=(1, 1, 1),
        description="The downsampling factor in the order z,y,x",
    )
    # For now, only the first value of the isocenter is used.
    iso_center: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="The distance of the isocenter from the center of the CT volume in cm",
    )
    minimum_leaf_overlap: float = Field(
        default=0.05, description="The minimum opening of the leafs, given in cm."
    )
    minimum_jaw_overlap: float = Field(
        default=0.5, description="The minimum opening of the jaws, given in cm."
    )
    maximum_jaw_speed: float = Field(
        default=2.25, description="The maximum speed of the leafs, given in cm / s."
    )
    maximum_leaf_speed: float = Field(
        default=2.25, description="The maximum speed of the leafs, given in cm / s."
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
    mu_scaling: float = Field(
        default=18.78, description="Beam scaling so that 130MU is 1Gy at the isocenter"
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

    # Registry of presets. Add as many as you like.
    PRESETS: ClassVar[dict[str, dict[str, Any]]] = {
        "test": {
            "ct_array_shape": (64, 64, 64),
            "resolution": (0.3, 0.3, 0.3),
            "field_size": (40, 40),
            "number_of_leaf_pairs": 10,
            "tpr_20_10": 0.72,
            "number_of_cps": 24,
        },
        "lund-probe": {
            "ct_array_shape": (96, 256, 256),
            "resolution": (0.416, 0.15625, 0.15625),
            "mean_photon_energy_MeV": 0.8686360716819763, # Temporary, needs to be calibrated
            "downsampling_factor": (1, 1, 1),
            "field_size": (40, 40),
            "number_of_leaf_pairs": 60,
            "tpr_20_10": 0.72,
            "number_of_cps": 240,
        },
        "umea": {
            "ct_array_shape": (320, 128, 128),
            "resolution": (0.25, 0.5, 0.5),
            "field_size": (40, 40),
            "number_of_leaf_pairs": 60,
            "number_of_cps": 240,
            "mean_photon_energy_MeV": 0.8686360716819763,
            "tpr_20_10": 0.72, 
            "starting_angle": 0.0,
            "iso_center": (0.0, 0.0, 0.0),
        },
    }

    @model_validator(mode="before")
    @classmethod
    def _apply_preset(cls, data: Any) -> Any:
        """
        Merge selected preset values into the incoming data before validation.
        Precedence (highest → lowest):
            1) Explicit kwargs (passed to ModelConfig(...))
            2) Environment variables (handled by BaseSettings later)
            3) Preset values (from PRESETS)
            4) Field defaults
        """
        if not isinstance(data, dict):
            return data
        name = data.get("preset")
        if not name:
            return data
        try:
            preset_values = cls.PRESETS[name]
        except KeyError as e:
            valid = ", ".join(sorted(cls.PRESETS))
            raise ValueError(f"Unknown preset '{name}'. Valid presets: {valid}") from e

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
    SID: float = Field(default=100, description="Source-to-isocenter distance in cm")
    number_of_cps: int = Field(
        description="The number of beams in the plane given in cm"
    )
    starting_angle: float = Field(
        default=0.0,
        description="The beam angle of the first beam in the series in degrees",
    )
    clockwise: bool = Field(
        default=False,
        description="Determines whether the arc moves clockwise or not.",
    )

    @model_validator(mode="after")
    def adjust_shapes(self) -> "ModelConfig":
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

        return self

    @computed_field(repr=False)
    @property
    def gantry_angles(self) -> np.ndarray:
        return (1 if self.clockwise else -1) * np.linspace(
            math.radians(self.starting_angle),
            math.radians(self.starting_angle) + math.radians(360),
            self.number_of_cps,
            endpoint=False,
        )

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
            int(np.ceil(self.field_size[1] / self.resolution[1])),
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
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
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
