"""
Patient configuration - CT dimensions and geometric parameters.
"""
# from pydantic import BaseModel, Field, model_validator
from dataclasses import dataclass
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
    structures: dict[str, np.array]
    dose: Optional[np.array] = None
    voxel_spacing_mm: Optional[tuple[float, float, float]] = None

    plan_iso_center: Optional[tuple[float, float, float]] = None
    plan_mlcs: Optional[np.array] = None
    plan_mus: Optional[np.array] = None
    plan_jaws: Optional[np.array] = None
    plan_clockwise: Optional[bool] = None
    plan_starting_angle: Optional[float] = None
    
    # Optional metadata
    patient_id: Optional[str] = None
    
    @classmethod
    def from_dicom(
        cls, 
        ct_folder: str, 
        dose_path: str | None, 
        plan_path: str | None, 
        struct_names: List[str] | None = None, 
        recenter: bool = True) -> 'Patient':
        """
        Create PatientCoPatientnfig from Patient.
        
        Args:
            patient: Patient instance
            
        Returns:
            Patient with CT dimensions from patient
        """
        ct_series, ref = load_ct_series(ct_folder)
        structures = load_structures(ct_series, ct_folder, struct_names=struct_names)
        dose = load_dose(dose_path)
        scaling = 400

        # If RTPLAN is available, use it to determine isocenter
        clockwise = True
        starting_angle = 0.0
        if plan_path is not None:
            mlcs, jaws, mus, clockwise, starting_angle = fetch_plan_data(plan_path, scaling)
            # Use the first dose as reference
            ct_series, structures, dose, iso_center = resample_based_on_plan(ct_series, structures, dose, recenter, plan_path)
            



        else:
            # No plan, just match to first dose
            mlcs = None
            ct_series, structures = resample_based_on_dose(ct_series, dose)
        
        return cls(
            ct_array=sitk.GetArrayFromImage(ct_series),
            structures={k: sitk.GetArrayFromImage(v) for k, v in structures.items()},            voxel_spacing_mm=ct_series.GetSpacing(),
            dose=sitk.GetArrayFromImage(dose),
            plan_mlcs=mlcs,
            plan_jaws=jaws,
            plan_mus=mus,
            plan_iso_center=iso_center,
            plan_clockwise=clockwise,
            plan_starting_angle=starting_angle,
        )
    
    @classmethod
    def from_nifti(
        cls,
        folder_path
        ) -> 'Patient':
        ct, structures, dose = load_files(folder_path)

        return cls(
            ct_array=ct,
            structures=structures,
            dose=dose,
            patient_id="",
        )

    @classmethod
    def from_ct_array(
        cls,
        ct_array: torch.Tensor,
        voxel_spacing_mm: tuple,
        patient_id: Optional[str] = None
    ) -> 'Patient':
        """Create from CT array directly."""
        return cls(
            ct_shape=tuple(ct_array.shape),
            voxel_spacing_mm=voxel_spacing_mm,
            patient_id=patient_id
        )
    

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
    def from_sphere(
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
