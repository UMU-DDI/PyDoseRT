"""Machine parameters for the ion dose engine.

The engine's only machine-level input is the nozzle geometry that turns an SSD
into a radiological-depth offset::

    rad_depth_offset_mm = 0.0011 * ((ssd_mm + bams_to_iso_dist_mm) - sad_mm - fit_air_offset_mm)

Everything else about the beam is per-beamlet on
:class:`~pydosert.data.ion_beam.IonBeamletBatch` -- SAD included -- or is
commissioned base data on
:class:`~pydosert.physics.kernels.ion_kernel_table.IonKernelTable`.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

__all__ = ["IonMachineConfig", "list_ion_machine_presets"]

#: Built-in ion machine presets; separate from the photon ``machine_presets``.
_PRESET_PACKAGE = "pydosert.data"
_PRESET_DIR = "ion_machine_presets"


def list_ion_machine_presets() -> list[str]:
    """Return the names of all built-in ion machine presets (without ``.json``)."""
    preset_dir = resources.files(_PRESET_PACKAGE).joinpath(_PRESET_DIR)
    return sorted(p.name[:-5] for p in preset_dir.iterdir() if p.name.endswith(".json"))


class IonMachineConfig(BaseSettings):
    """Nozzle geometry of an ion beam line.

    A pydantic ``BaseSettings`` model: fields can be supplied as kwargs, via
    environment variables, or merged from a named JSON preset (``preset``). Both
    distances default to pyRadPlan's ``Generic`` proton machine.

    Attributes:
        preset (Optional[str]): Name or path of a preset whose values are merged
            before validation (explicit kwargs and env vars take precedence).
        bams_to_iso_dist_mm (float): Beam-application-monitor-system (nozzle
            exit) to isocentre distance, mm.
        fit_air_offset_mm (float): Air gap, mm, already folded into the
            commissioned pencil-beam kernels.
    """

    preset: Optional[str] = Field(
        default=None,
        description="Optional preset name or JSON path whose values are merged before validation.",
    )
    bams_to_iso_dist_mm: float = Field(
        default=1000.0,
        ge=0.0,
        description=(
            "Distance from the beam-application-monitor system (nozzle exit) to "
            "isocentre, in mm. Default: pyRadPlan's Generic protons machine."
        ),
    )
    fit_air_offset_mm: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Air gap, in mm, already included when the pencil-beam kernels were "
            "fitted, and therefore subtracted again from the SSD."
        ),
    )

    @staticmethod
    def _load_preset_json(name_or_path: str) -> dict[str, Any]:
        """Load a preset from a JSON file path, or by built-in name (``.json`` optional).

        Raises:
            ValueError: Unknown preset name, or the file is not a JSON object.
        """
        path = Path(name_or_path)
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            stem = path.stem  # strips .json if present
            try:
                preset_file = (
                    resources.files(_PRESET_PACKAGE).joinpath(_PRESET_DIR).joinpath(stem + ".json")
                )
                data = json.loads(preset_file.read_text(encoding="utf-8"))
            except (FileNotFoundError, TypeError, OSError) as exc:
                available = list_ion_machine_presets()
                raise ValueError(
                    f"Unknown ion machine preset '{stem}'. "
                    f"Available built-in presets: {available}. "
                    "You can also pass an absolute path to a custom JSON file."
                ) from exc
        if not isinstance(data, dict):
            raise ValueError(f"Preset '{name_or_path}' must contain a JSON object at the top level.")
        return data

    @model_validator(mode="before")
    @classmethod
    def _apply_preset(cls, data: Any) -> Any:
        """Merge preset values underneath the incoming data.

        Precedence, highest first: explicit kwargs, environment variables,
        preset values, field defaults.
        """
        if not isinstance(data, dict):
            return data
        name = data.get("preset")
        if not name:
            return data
        preset_values = cls._load_preset_json(name)
        return {**preset_values, **data}
