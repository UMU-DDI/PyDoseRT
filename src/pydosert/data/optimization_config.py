"""
Simplified optimization configuration for dose planning.

Stores structure constraints and clinical criteria as simple dictionaries,
providing a clean API for programmatic setup and validation.
"""

import json
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import torch


def list_optimization_presets() -> list[str]:
    """Return the names of all built-in optimization presets (without .json extension)."""
    preset_dir = resources.files("pydosert.data").joinpath("optimization_presets")
    return sorted(
        p.name[:-5]  # strip .json
        for p in preset_dir.iterdir()
        if p.name.endswith(".json")
    )


class OptimizationConfig:
    """
    Optimization configuration for treatment planning.

    Simple dict-based storage with a clean API for setting up
    structure constraints and clinical criteria.

    Example:
        # Load from JSON
        config = OptimizationConfig.from_json("varian_10MV.json")

        # Or create programmatically
        config = OptimizationConfig(prescription_gy=42.7)
        config.add_structure("PTV", lower_bound_gy=42.7, weight=1000.0)
        config.add_criterion("PTV", "dose_at_volume",
                           volume_percent=95.0, dose_percent=100.0,
                           constraint_type="at_least")

        # Validate dose
        results = config.validate(pred_dose, patient)
    """

    def __init__(self, prescription_gy: Optional[float] = None):
        """
        Initialize optimization config.

        Args:
            prescription_gy (Optional[float]): Prescription dose in Gy.
        """
        self.prescription_gy = prescription_gy
        self.structures: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'OptimizationConfig':
        """
        Load configuration from JSON file.

        Args:
            path (Union[str, Path]): Path to JSON file.

        Returns:
            OptimizationConfig: Instance populated from the file's
                ``prescription_gy`` and ``structures`` entries.
        """
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        config = cls(prescription_gy=data.get("prescription_gy"))
        config.structures = data.get("structures", {})

        return config

    @classmethod
    def from_preset(cls, name: str) -> 'OptimizationConfig':
        """
        Load a built-in optimization preset by name.

        Works after ``pip install`` because the presets are bundled with the
        package and resolved via ``importlib.resources``.

        Args:
            name (str): Preset name (with or without ``.json`` extension).
                  Call ``list_optimization_presets()`` to see available names.

        Returns:
            OptimizationConfig: Instance populated from the preset's
                ``prescription_gy`` and ``structures`` entries.

        Example::

            from pydosert.data import OptimizationConfig, list_optimization_presets
            print(list_optimization_presets())   # ['gold-atlas', 'lund-probe', 'vienna']
            config = OptimizationConfig.from_preset("vienna")
        """
        stem = Path(name).stem  # strips .json if present
        filename = stem + ".json"
        try:
            preset_file = resources.files("pydosert.data").joinpath("optimization_presets").joinpath(filename)
            data = json.loads(preset_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, TypeError):
            available = list_optimization_presets()
            raise ValueError(
                f"Unknown optimization preset '{stem}'. "
                f"Available built-in presets: {available}. "
                "You can also use from_json() with an absolute path to a custom file."
            )

        config = cls(prescription_gy=data.get("prescription_gy"))
        config.structures = data.get("structures", {})
        return config

    def to_json(self, path: Union[str, Path]):
        """
        Save configuration to JSON file.

        Args:
            path (Union[str, Path]): Path to save JSON file.
        """
        path = Path(path)
        data = {
            "prescription_gy": self.prescription_gy,
            "structures": self.structures
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_structure(self,
                     name: str,
                     color: Optional[str] = None,
                     lower_bound_gy: float = 0.0,
                     higher_bound_gy: float = 100.0,
                     lower_bound_target_percent: float = 0.0,
                     higher_bound_target_percent: float = 100.0,
                     weight: float = 1.0):
        """
        Add a structure with optimization constraints.

        Args:
            name (str): Structure name.
            color (Optional[str]): Color for plotting.
            lower_bound_gy (float): Minimum dose constraint (Gy).
            higher_bound_gy (float): Maximum dose constraint (Gy).
            lower_bound_target_percent (float): % of volume that must receive >= lower_bound_gy.
            higher_bound_target_percent (float): % of volume that must be <= higher_bound_gy.
            weight (float): Optimization weight.
        """
        self.structures[name] = {
            "color": color,
            "lower_bound_gy": lower_bound_gy,
            "higher_bound_gy": higher_bound_gy,
            "lower_bound_target_percent": lower_bound_target_percent,
            "higher_bound_target_percent": higher_bound_target_percent,
            "weight": weight,
            "clinical_criteria": []
        }

    def add_criterion(self,
                     structure: str,
                     criterion_type: str,
                     constraint_type: str,
                     dose_gy: Optional[float] = None,
                     dose_percent: Optional[float] = None,
                     volume_percent: Optional[float] = None,
                     volume_cc: Optional[float] = None,
                     description: Optional[str] = None):
        """
        Add a clinical criterion to a structure.

        Args:
            structure (str): Structure name.
            criterion_type (str): 'dose_at_volume', 'dose_at_volume_cc', or 'volume_at_dose'.
            constraint_type (str): 'at_least' or 'at_most'.
            dose_gy (Optional[float]): Dose in Gy (absolute).
            dose_percent (Optional[float]): Dose as % of prescription.
            volume_percent (Optional[float]): Volume percentage (0-100).
            volume_cc (Optional[float]): Volume in cubic centimeters.
            description (Optional[str]): Human-readable description.
        """
        if structure not in self.structures:
            raise ValueError(f"Structure '{structure}' not found. Add it first with add_structure().")

        criterion = {
            "criterion_type": criterion_type,
            "constraint_type": constraint_type,
            "dose_gy": dose_gy,
            "dose_percent": dose_percent,
            "volume_percent": volume_percent,
            "volume_cc": volume_cc,
            "description": description
        }

        self.structures[structure]["clinical_criteria"].append(criterion)

    def get_parameters(self, parameter_name: str) -> Dict[str, float]:
        """
        Collect one named parameter value across all structures.

        Args:
            parameter_name (str): Structure field to read (e.g. "weight",
                "lower_bound_gy").

        Returns:
            Dict[str, float]: Mapping of structure name to that field's value
                (None for structures lacking the field).
        """
        return {name: struct.get(parameter_name)
                for name, struct in self.structures.items()}

    def validate(self, pred_dose: Union[np.ndarray, torch.Tensor], patient) -> Dict[str, Dict]:
        """
        Validate predicted dose against clinical criteria.

        Args:
            pred_dose (Union[np.ndarray, torch.Tensor]): Predicted dose
                distribution (Gy), shape [1, D, H, W] or [D, H, W]; a leading
                batch axis of 1 is squeezed out internally.
            patient: Patient object exposing ``.resolution`` (mm voxel spacing)
                and ``.structures`` (name -> [D, H, W] mask).

        Returns:
            Dictionary with structure names as keys, containing:
                - 'criteria': List of criterion results
                - Each criterion has: 'type', 'description', 'value', 'threshold', 'ratio', 'passed'
        """
        from ..objectives.metrics import (
            dose_at_volume_percent,
            dose_at_volume_cc,
            volume_at_dose
        )

        # Handle dose array shape
        if isinstance(pred_dose, torch.Tensor):
            dose = pred_dose.cpu().detach().numpy()
        else:
            dose = pred_dose

        if dose.ndim == 4:
            dose = dose[0, ...]

        # Calculate voxel volume in cc
        resolution = patient.resolution
        voxel_volume_cc = np.prod(resolution) / 1000.0  # Convert mm³ to cc

        results = {}

        # Process each structure
        for structure_name, struct_data in self.structures.items():
            # Skip if structure not in patient masks
            if structure_name not in patient.structures:
                continue

            structure_mask = patient.structures[structure_name]
            if isinstance(structure_mask, torch.Tensor):
                structure_mask = structure_mask.cpu().detach().numpy() > 0
            else:
                structure_mask = structure_mask > 0

            structure_results = {'criteria': []}

            # Process explicit clinical criteria if defined
            clinical_criteria = struct_data.get("clinical_criteria", [])

            if clinical_criteria:
                for criterion in clinical_criteria:
                    criterion_result = self._evaluate_criterion(
                        dose, structure_mask, criterion, voxel_volume_cc
                    )
                    structure_results['criteria'].append(criterion_result)

            # Otherwise, fall back to generating criteria from constraints
            else:
                # Lower bound criterion: D_x% >= threshold
                lower_bound_gy = struct_data.get("lower_bound_gy", 0.0)
                lower_bound_percent = struct_data.get("lower_bound_target_percent", 0.0)

                if lower_bound_percent > 0 and lower_bound_gy > 0:
                    actual_dose = dose_at_volume_percent(
                        dose, structure_mask, lower_bound_percent
                    )
                    threshold_dose = lower_bound_gy

                    if actual_dose > 0:
                        ratio = threshold_dose / actual_dose
                    else:
                        ratio = float('inf') if threshold_dose > 0 else 1.0

                    structure_results['criteria'].append({
                        'type': f'D{lower_bound_percent:.2f}%',
                        'description': f'At least {threshold_dose:.2f} Gy at {lower_bound_percent:.2f}% volume',
                        'value': actual_dose,
                        'threshold': threshold_dose,
                        'ratio': ratio,
                        'passed': ratio <= 1.0
                    })

                # Higher bound criterion: D_x% <= threshold
                higher_bound_gy = struct_data.get("higher_bound_gy", 100.0)
                higher_bound_percent = struct_data.get("higher_bound_target_percent", 100.0)

                if higher_bound_percent < 100 and higher_bound_gy > 0:
                    actual_dose = dose_at_volume_percent(
                        dose, structure_mask, 100 - higher_bound_percent
                    )
                    threshold_dose = higher_bound_gy

                    if threshold_dose > 0:
                        ratio = actual_dose / threshold_dose
                    else:
                        ratio = float('inf') if actual_dose > 0 else 1.0

                    structure_results['criteria'].append({
                        'type': f'D{100 - higher_bound_percent:.2f}%',
                        'description': f'At most {threshold_dose:.2f} Gy at {100 - higher_bound_percent:.2f}% volume',
                        'value': actual_dose,
                        'threshold': threshold_dose,
                        'ratio': ratio,
                        'passed': ratio <= 1.0
                    })

                # Volume criterion: V_x Gy <= threshold %
                if higher_bound_gy > 0 and higher_bound_percent < 100:
                    actual_volume_percent = volume_at_dose(
                        dose, structure_mask, higher_bound_gy
                    )
                    threshold_volume_percent = higher_bound_percent

                    if threshold_volume_percent > 0:
                        ratio = actual_volume_percent / threshold_volume_percent
                    else:
                        ratio = float('inf') if actual_volume_percent > 0 else 1.0

                    structure_results['criteria'].append({
                        'type': f'V{higher_bound_gy:.2f}Gy',
                        'description': f'At most {threshold_volume_percent:.2f}% volume at {higher_bound_gy:.2f} Gy',
                        'value': actual_volume_percent,
                        'threshold': threshold_volume_percent,
                        'ratio': ratio,
                        'passed': ratio <= 1.0
                    })

            results[structure_name] = structure_results

        return results

    def _evaluate_criterion(self,
                           dose: np.ndarray,
                           structure_mask: np.ndarray,
                           criterion: Dict,
                           voxel_volume_cc: float) -> Dict:
        """
        Evaluate a single clinical criterion.

        Args:
            dose (np.ndarray): 3D dose distribution (Gy), shape [D, H, W].
            structure_mask (np.ndarray): 3D binary mask, shape [D, H, W].
            criterion (Dict): Criterion dictionary (criterion_type,
                constraint_type, and the relevant dose/volume keys).
            voxel_volume_cc (float): Volume of a single voxel in cubic centimetres.

        Returns:
            Dict: Evaluation result with keys 'type', 'description', 'value',
                'threshold', 'ratio', 'passed'.
        """
        from ..objectives.metrics import (
            dose_at_volume_percent,
            dose_at_volume_cc,
            volume_at_dose
        )

        criterion_type = criterion["criterion_type"]
        constraint_type = criterion["constraint_type"]

        # Resolve dose threshold
        def get_dose_threshold() -> float:
            """Resolve the criterion's dose threshold in Gy.

            Returns:
                float: ``dose_percent`` scaled by ``prescription_gy``, else the
                    absolute ``dose_gy``.

            Raises:
                ValueError: If neither ``dose_gy`` nor ``dose_percent`` is given,
                    or ``dose_percent`` is used without ``prescription_gy`` set.
            """
            if criterion.get("dose_percent") is not None:
                if self.prescription_gy is None:
                    raise ValueError("Criterion uses dose_percent but prescription_gy not set")
                return criterion["dose_percent"] * self.prescription_gy / 100.0
            elif criterion.get("dose_gy") is not None:
                return criterion["dose_gy"]
            else:
                raise ValueError("Criterion must specify either dose_gy or dose_percent")

        if criterion_type == "dose_at_volume":
            # Dx% - dose at x% of volume
            actual_value = dose_at_volume_percent(
                dose, structure_mask, criterion["volume_percent"]
            )
            threshold = get_dose_threshold()

            if constraint_type == "at_least":
                ratio = threshold / actual_value if actual_value > 0 else float('inf')
                type_str = f'D{criterion["volume_percent"]:.2f}%'
                desc = criterion.get("description") or f'At least {threshold:.2f} Gy at {criterion["volume_percent"]:.2f}% volume'
            else:  # at_most
                ratio = actual_value / threshold if threshold > 0 else float('inf')
                type_str = f'D{criterion["volume_percent"]:.2f}%'
                desc = criterion.get("description") or f'At most {threshold:.2f} Gy at {criterion["volume_percent"]:.2f}% volume'

        elif criterion_type == "dose_at_volume_cc":
            # Dx cc - dose at x cubic centimeters
            actual_value = dose_at_volume_cc(
                dose, structure_mask, criterion["volume_cc"], voxel_volume_cc
            )
            threshold = get_dose_threshold()

            if constraint_type == "at_most":
                ratio = actual_value / threshold if threshold > 0 else float('inf')
                type_str = f'D{criterion["volume_cc"]:.2f}cc'
                desc = criterion.get("description") or f'At most {threshold:.2f} Gy at {criterion["volume_cc"]:.2f} cm³ volume'
            else:  # at_least
                ratio = threshold / actual_value if actual_value > 0 else float('inf')
                type_str = f'D{criterion["volume_cc"]:.2f}cc'
                desc = criterion.get("description") or f'At least {threshold:.2f} Gy at {criterion["volume_cc"]:.2f} cm³ volume'

        elif criterion_type == "volume_at_dose":
            # Vx Gy - volume % receiving at least x Gy
            dose_threshold = get_dose_threshold()
            actual_value = volume_at_dose(
                dose, structure_mask, dose_threshold
            )
            threshold = criterion["volume_percent"]

            if constraint_type == "at_most":
                ratio = actual_value / threshold if threshold > 0 else float('inf')
                type_str = f'V{dose_threshold:.2f}Gy'
                desc = criterion.get("description") or f'At most {threshold:.2f}% volume at {dose_threshold:.2f} Gy'
            else:  # at_least
                ratio = threshold / actual_value if actual_value > 0 else float('inf')
                type_str = f'V{dose_threshold:.2f}Gy'
                desc = criterion.get("description") or f'At least {threshold:.2f}% volume at {dose_threshold:.2f} Gy'

        else:
            raise ValueError(f"Unknown criterion type: {criterion_type}")

        return {
            'type': type_str,
            'description': desc,
            'value': actual_value,
            'threshold': threshold,
            'ratio': ratio,
            'passed': ratio <= 1.0
        }