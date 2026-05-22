import json
from importlib import resources
from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings
from typing import Any, Optional


def list_machine_presets() -> list[str]:
    """Return the names of all built-in machine presets (without .json extension)."""
    preset_dir = resources.files("pydosert.data").joinpath("machine_presets")
    return sorted(
        p.name[:-5]  # strip .json
        for p in preset_dir.iterdir()
        if p.name.endswith(".json")
    )

class MachineConfig(BaseSettings):
    """
    Linac/MLC machine parameters used by the dose engine.
    A pydantic ``BaseSettings`` model: fields can be supplied as kwargs, via
    environment variables, or merged from a named JSON preset (see ``preset``).
    Each field's meaning and units are given in its ``Field`` description; a
    summary of the key fields:
    Attributes:
        preset (Optional[str]): Name/path of a preset whose values are merged
            before validation (explicit kwargs and env vars take precedence).
        tpr_20_10 (float): Tissue phantom ratio TPR20/10 (dimensionless).
        number_of_leaf_pairs (int): Number of MLC leaf pairs (N).
        minimum_leaf_opening (float): Minimum MLC leaf-pair opening (mm).
        minimum_jaw_opening (float): Minimum jaw opening (mm).
        maximum_leaf_tip_overlap (float): Maximum opposing-leaf tip overlap (mm).
        maximum_jaw_speed (float): Maximum jaw speed (mm/s).
        maximum_leaf_speed (float): Maximum leaf speed (mm/s).
        minimum_gantry_angle_speed (float): Minimum gantry rotation speed (deg/s).
        maximum_gantry_angle_speed (float): Maximum gantry rotation speed (deg/s).
        maximum_gantry_angle_speed_variation (float): Max gantry speed variation (deg/s).
        minimum_dose_rate (float): Minimum dynamic-arc dose rate (MU/s).
        maximum_dose_rate (float): Maximum dynamic-arc dose rate (MU/s).
        mlc_transmission (float): Closed-MLC transmission as a fraction of open fluence.
        penumbra_fwhm (Optional[list[float]]): Penumbra FWHM (mm); one value, or two
            for the MLC and jaw directions respectively.
        head_scatter_amplitude (Optional[list[float]]): Head-scatter amplitude as a
            fraction of dose; one value or two (MLC, jaw directions).
        head_scatter_sigma (Optional[list[float]]): Head-scatter Gaussian sigma (mm).
        head_scatter_ssd_mm (float): Source-to-scatter-source distance (mm).
        calibration_mu (float): MU value for dose calibration in water.
        mean_photon_energy_MeV (float): Mean photon energy (MeV).
        leaf_widths (Optional[list[float]]): Per-leaf widths (mm), length N.
        profile_corrections (Optional[list[list[float]]]): Off-axis correction data
            as [distances_mm, correction_ratios].
        output_factors (Optional[list[list[float]]]): Output-factor LUT data as
            [distances_mm, correction_ratios].
        dlg_mm (Optional[float]): Dosimetric leaf gap (mm); each MLC bank is shifted
            outward by half this value.
        sc_source_sigma_mm (Optional[list[float]]): Effective source sigma at
            isocentre (mm) for the analytical Sc collimator-scatter model; one
            isotropic value or two [sigma_x_mm, sigma_y_mm].
    """

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
    dlg_mm: Optional[float] = Field(
        default=None,
        description=(
            "Dosimetric Leaf Gap in mm. Each MLC bank is shifted outward by half "
            "this value so that the effective aperture is wider than the nominal "
            "leaf positions, matching the sub-field radiation leakage observed in "
            "physical measurements."
        ),
    )
    sc_source_sigma_mm: Optional[list[float]] = Field(
        default=None,
        description=(
            "Effective source sigma at isocentre (mm) for the analytical "
            "Sc(field_size) collimator-scatter model.  Provide a single value "
            "[sigma_mm] for an isotropic source, or two values "
            "[sigma_x_mm, sigma_y_mm] for crossline / inline directions.  "
            "When set, Sc is computed via the erf-integral formula instead of "
            "the empirical output-factor LUT."
        ),
    )

    
    @staticmethod
    def _load_preset_json(name_or_path: str) -> dict[str, Any]:
        """
        Load a preset by name or file path.

        - If ``name_or_path`` is an existing file, load it directly.
        - Otherwise treat it as a built-in preset name (with or without
          the ``.json`` extension) and load from the bundled package data.
          This works both during development and after ``pip install``.
        
        Args:
            name_or_path (str): Filesystem path to a JSON file, or a built-in
                preset name (``.json`` extension optional).
        Returns:
            dict[str, Any]: The decoded JSON object of preset field values.
        """
        path = Path(name_or_path)
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            # Resolve as a built-in preset name
            stem = path.stem  # strips .json if present
            filename = stem + ".json"
            try:
                preset_file = resources.files("pydosert.data").joinpath("machine_presets").joinpath(filename)
                data = json.loads(preset_file.read_text(encoding="utf-8"))
            except (FileNotFoundError, TypeError):
                available = list_machine_presets()
                raise ValueError(
                    f"Unknown machine preset '{stem}'. "
                    f"Available built-in presets: {available}. "
                    "You can also pass an absolute path to a custom JSON file."
                )
        if not isinstance(data, dict):
            raise ValueError(f"Preset '{name_or_path}' must contain a JSON object at the top level.")
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

        Args:
            data (Any): Raw input passed to the validator; only acted upon when it
                is a dict containing a non-empty ``preset`` key.
        Returns:
            Any: ``data`` unchanged, or a dict of preset values merged under the
                explicit input values.
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