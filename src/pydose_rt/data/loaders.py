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



def load_asc_measurements(path: str,
                          coord_map: Tuple[str, str, str] = ("X", "Y", "Z")):
    """
    Load a BDS-style .asc file and split it into measurements.

    coord_map:
        Mapping from engine (x,y,z) to ASC axes.
        Each entry must be one of "X", "Y", "Z".

        Example:
            coord_map=("X", "Z", "Y")
            -> engine_x = ASC.X
               engine_y = ASC.Z
               engine_z = ASC.Y

    Returns:
        measurements: list of dicts, each with:
            - 'measurement_number': int or None
            - 'header_dict': parsed % / : lines, e.g. {'DAT': '09-07-2015', ...}
            - 'header_lines': raw header lines
            - 'data_raw': np.ndarray of shape (N, 4) [X_file, Y_file, Z_file, Dose]
            - 'coords_asc': np.ndarray of shape (N, 3) [X_file, Y_file, Z_file]
            - 'coords_engine': np.ndarray of shape (N, 3) [x_eng, y_eng, z_eng]
            - 'dose': np.ndarray of shape (N,)
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    # validate coord_map
    valid_axes = {"X", "Y", "Z"}
    if set(coord_map) != valid_axes:
        raise ValueError(
            f"coord_map must be a permutation of ('X','Y','Z'), got {coord_map}"
        )

    measurements: List[Dict[str, Any]] = []

    current_number = None
    current_data_lines: List[str] = []
    current_header_lines: List[str] = []
    current_header_dict: Dict[str, str] = {}

    def finalize_block():
        """Finalize current measurement block into measurements list."""
        if current_number is None:
            return

        if current_data_lines:
            data = np.loadtxt(current_data_lines, usecols=(1, 2, 3, 4))
            if data.ndim == 1:  # single row special case
                data = data[None, :]
        else:
            data = np.empty((0, 4), dtype=float)

        coords_asc = data[:, :3]          # [X_file, Y_file, Z_file]
        dose = data[:, 3]

        # map ASC -> engine coords
        name_to_idx = {"X": 0, "Y": 1, "Z": 2}
        idxs = [name_to_idx[name] for name in coord_map]
        coords_engine = coords_asc[:, idxs]

        measurements.append(
            {
                "measurement_number": current_number,
                "header_dict": current_header_dict.copy(),
                "header_lines": current_header_lines.copy(),
                "data_raw": data,
                "coords_asc": coords_asc,
                "coords_engine": coords_engine,
                "dose": dose,
            }
        )

    for line in lines:
        if "Measurement number" in line:
            # close previous measurement
            finalize_block()

            # extract number from line
            num = None
            for token in line.split():
                if token.isdigit():
                    num = int(token)

            current_number = num
            current_data_lines = []
            current_header_lines = [line]
            current_header_dict = {}
            if num is not None:
                current_header_dict["MeasurementNumber"] = str(num)

        else:
            if current_number is None:
                # global header: ignore
                continue

            stripped = line.lstrip()

            if stripped.startswith("="):
                # data row
                current_data_lines.append(line)
            else:
                # header/meta row
                current_header_lines.append(line)

                # strip inline comments after '#'
                stripped_comment = stripped.split("#", 1)[0].rstrip()
                if not stripped_comment:
                    continue

                if stripped_comment[0] in ("%", ":"):
                    body = stripped_comment[1:].strip()
                    if not body:
                        continue
                    parts = body.split(None, 1)
                    key = parts[0]
                    value = parts[1].strip() if len(parts) > 1 else ""
                    current_header_dict[key] = [val.strip() for val in value.split("\t")]

    # last measurement
    finalize_block()

    return measurements
