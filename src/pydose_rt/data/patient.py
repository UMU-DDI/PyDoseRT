"""
Patient configuration - CT dimensions and geometric parameters.
"""
# from pydantic import BaseModel, Field, model_validator
from dataclasses import dataclass, field
from token import OP
from typing import Optional, TYPE_CHECKING, List
import torch
import numpy as np
from pydose_rt.data.utils.dicom_utils import load_ct_series, load_structures, load_dose, fetch_plan_data, resample_based_on_plan, resample_based_on_dose
from .utils.nifti_utils import load_files
import SimpleITK as sitk


if TYPE_CHECKING:
    from pydose_rt.data import Patient

@dataclass
class Patient:
    """
    Patient-specific configuration.

    Defines CT dimensions and geometric parameters for dose calculation.
    """

    # CT dimensions
    ct_array: torch.Tensor
    structures: Optional[dict[str, torch.Tensor]] = field(default_factory=dict)
    dose: Optional[torch.Tensor] = None
    voxel_spacing_mm: Optional[tuple[float, float, float]] = None

    plan_iso_center: Optional[tuple[float, float, float]] = None
    plan_clockwise: Optional[bool] = None
    plan_starting_angle: Optional[float] = None
    

@dataclass
class Phantom(Patient):
    """
    Phantom patient configuration for testing.

    Inherits from Patient.
    """

    def __init__(
        self,
        ct_array: np.array,
        voxel_spacing_mm: tuple[float, float, float],
        patient_id: Optional[str] = "Phantom"
    ):
        super().__init__(
            ct_array=ct_array,
            structures={},
            dose=None,
            voxel_spacing_mm=voxel_spacing_mm,
            patient_id=patient_id
        )
    
    @classmethod
    def from_uniform_water(
        cls,
        shape: tuple[int, int, int],
        spacing: tuple[float, float, float]
    ) -> "Phantom":
        """
        Alternate constructor: create a Phantom directly from a spherical phantom.
        """
        ct_array = torch.from_numpy(np.expand_dims(np.ones(shape), 0))

        return cls(
            ct_array=ct_array,
            voxel_spacing_mm=spacing,
            patient_id="",
        )


    @classmethod
    def from_sphere_water(
        cls,
        shape: tuple[int, int, int],
        spacing: tuple[float, float, float],
        radius_mm: float,
        ct_value: float = 0.0,
        background_value: float = -1000.0
    ) -> "Phantom":
        """
        Alternate constructor: create a Phantom directly from a spherical phantom.
        """
        z = np.arange(shape[0]) * spacing[0]
        y = np.arange(shape[1]) * spacing[1]
        x = np.arange(shape[2]) * spacing[2]
        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

        center = (np.array(shape) * np.array(spacing)) / 2.0
        distances = np.sqrt(
            (X - center[2]) ** 2 +
            (Y - center[1]) ** 2 +
            (Z - center[0]) ** 2
        )

        ct_array = torch.from_numpy(np.expand_dims(np.where(distances <= radius_mm, ct_value, background_value), 0))

        return cls(
            ct_array=ct_array,
            voxel_spacing_mm=spacing,
            patient_id="",
        )
