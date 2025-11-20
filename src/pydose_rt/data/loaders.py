"""
Patient configuration - CT dimensions and geometric parameters.
"""
# from pydantic import BaseModel, Field, model_validator
from dataclasses import dataclass
from token import OP
from typing import Optional, TYPE_CHECKING, List
import torch
import math
import numpy as np
from pydose_rt.data.utils.dicom_utils import load_ct_series, load_structures, load_dose, fetch_plan_data, resample_based_on_plan, resample_based_on_dose
from pydose_rt.data import TreatmentConfig, Patient
from .utils.nifti_utils import load_files
import SimpleITK as sitk
from typing import List, Dict, Any, Tuple

def load_dicom(
    ct_folder: str, 
    dose_path: str | None, 
    plan_path: str | None, 
    struct_names: List[str] | None = None, 
    treatment_preset: str | None = None,
    recenter: bool = True) -> tuple['Patient', "TreatmentConfig"]:
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
        plans = fetch_plan_data(plan_path, scaling)
        mlcs, jaws, mus, clockwise, starting_angle, final_angle, bld_angle, num_fractions = list(plans.values())[0]
        # Use the first dose as reference
        ct_series, structures, dose, iso_center = resample_based_on_plan(ct_series, structures, dose, recenter, plan_path)

    else:
        # No plan, just match to first dose
        mlcs = None
        ct_series, structures = resample_based_on_dose(ct_series, dose)

    num_of_cps = max(mus.shape[1] - 1, 1)
    patient = Patient(ct_array=sitk.GetArrayFromImage(ct_series),
        structures={k: sitk.GetArrayFromImage(v) for k, v in structures.items()},            voxel_spacing_mm=ct_series.GetSpacing(),
        dose=sitk.GetArrayFromImage(dose) / num_fractions)
    
    treatment = TreatmentConfig(
        preset=treatment_preset,
        number_of_cps=num_of_cps,
        iso_center=(0, 0, 0) if recenter else iso_center,
        plan_mlcs=mlcs,
        plan_jaws=jaws,
        plan_mus=mus,
        clockwise=clockwise,
        starting_angle=starting_angle,
        beam_limiting_device_angle=math.radians(bld_angle),
    )

    return patient, treatment

def load_nifti(
    folder_path
    ) -> 'Patient':
    ct, structures, dose = load_files(folder_path)
    patient = Patient(
        ct_array=ct,
        structures=structures,
        dose=dose,
        patient_id="",
    )

    return patient