"""
Patient configuration - CT dimensions and geometric parameters.
"""
from pydantic import BaseModel, Field, model_validator
from typing import Optional, TYPE_CHECKING, List
import torch
import numpy as np
from .utils.dicom_utils import load_ct_series, load_structures, load_dose, fetch_plan_data, resample_based_on_plan, resample_based_on_dose
from .utils.nifti_utils import load_files
import SimpleITK as sitk


if TYPE_CHECKING:
    from pydose_rt.data import PatientData


class PatientConfig(BaseModel):
    """
    Patient-specific configuration.
    
    Defines CT dimensions and geometric parameters for dose calculation.
    """
    
    # CT dimensions
    ct_array: np.array = Field(..., description="CT array shape [D, H, W]")
    structures: dict[str, np.array] = Field(..., description="Dictionary of all the structures")
    dose: np.array = Field(
        default=None,
        description="CT array shape [D, H, W]")
    voxel_spacing_mm: Optional[tuple[float, float, float]] = Field(
        default=None,
        description="Voxel spacing [dz, dy, dx] in mm")

    plan_iso_center: Optional[tuple[float, float, float]] = Field(
        default=None,
        description="isocenter")
    plan_mlcs: Optional[np.array] = Field(
        default=None,
        description="mlc positions")
    plan_clockwise: Optional[bool] = Field(
        default=None,
        description="Whether the plan is clockwise or not")
    plan_starting_angle: Optional[float] = Field(
        default=None, 
        description="starting angle of the plan")
    
    # Optional metadata
    patient_id: Optional[str] = None
    
    @classmethod
    def from_dicom(cls, ct_folder: str, dose_path: str | None, plan_path: str | None, struct_names: List[str] | None = None, recenter: bool = True) -> 'PatientConfig':
        """
        Create PatientConfig from PatientData.
        
        Args:
            patient: PatientData instance
            
        Returns:
            PatientConfig with CT dimensions from patient
        """
        ct_series, ref = load_ct_series(ct_folder)
        structures = load_structures(ct_series, ct_folder, struct_names=struct_names)
        dose = load_dose(dose_path)
        scaling = 400

        # If RTPLAN is available, use it to determine isocenter
        clockwise = True
        starting_angle = 0.0
        if plan_path is not None:
            mlcs, clockwise, starting_angle = fetch_plan_data(plan_path, scaling)
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
            plan_iso_center=iso_center,
            plan_clockwise=clockwise,
            plan_starting_angle=starting_angle,
        )
    
    @classmethod
    def from_nifti(
        cls,
        folder_path
        ) -> 'PatientConfig':
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
    ) -> 'PatientConfig':
        """Create from CT array directly."""
        return cls(
            ct_shape=tuple(ct_array.shape),
            voxel_spacing_mm=voxel_spacing_mm,
            patient_id=patient_id
        )
    
    class Config:
        arbitrary_types_allowed = True