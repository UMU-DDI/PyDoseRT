import json
from pathlib import Path
from pydantic import Field, computed_field, model_validator
from pydantic_settings import SettingsConfigDict, BaseSettings
import numpy as np
import torch
import math
from pydose_rt.data import PatientConfig, MachineConfig, TreatmentConfig
from typing import Any, Optional, List
_THIS_DIR = Path(__file__).resolve().parent
_PRESET_DIR_DEFAULT = _THIS_DIR / "machine_presets"   # <--- now relative to this module file

class DoseConfig(BaseSettings):
    machine_preset: Optional[str] = Field(
        default=None,
        description="Optional preset name whose values are merged before validation.",
    )
    treatment_preset: Optional[str] = Field(
        default=None,
        description="Optional preset name whose values are merged before validation.",
    )
    downsampling_factor: tuple[int, int, int] = Field(
        default=(1, 1, 1),
        description="The downsampling factor in the order z,y,x",
    )
    dtype: torch.dtype = Field(
        default=torch.float32, description="The data type used for the calculations"
    )

    device: torch.device = Field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        description="The device used for the calculations",
    )
    patient: Optional[PatientConfig] = Field(
        default=None,
        description="Patient information"
    )
    machine: Optional[MachineConfig] = Field(
        default=None,
        description="Machine information"
    )
    treatment: Optional[TreatmentConfig] = Field(
        default=None,
        description="Treatment information"
    )

    @model_validator(mode="before")
    @classmethod
    def create_machine_from_preset(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        
        machine_data = data.copy()
        machine_data["preset"] = data.get("machine_preset")
        if "patient" in machine_data:
            patient = machine_data.get("patient")
            machine_data["ct_array_shape"] = patient.ct_array.shape
            if patient.voxel_spacing_mm is not None:
                machine_data["resolution"] = patient.voxel_spacing_mm
            if (patient.plan_mlcs is not None):
                machine_data["clockwise"] = patient.plan_clockwise
                machine_data["starting_angle"] = patient.plan_starting_angle
                machine_data["number_of_cps"] = patient.plan_mlcs[0][0].shape[2]
                machine_data["number_of_leafs"] = patient.plan_mlcs[0][0].shape[3]

        if "machine" not in data:
            data["machine"] = MachineConfig(**machine_data)

        if (data.get("treatment_preset") is not None):
            treatment_data = data.copy()
            treatment_data["preset"] = treatment_data.get("treatment_preset")
            if "treatment" not in treatment_data:
                data["treatment"] = TreatmentConfig(**treatment_data)
        
        return data
    
    def load_patient(self, ct_folder: str, dose_path: str | None, plan_path: str | None, struct_names: List[str] | None = None, recenter: bool = True) -> dict[str, Any]:
        self.patient = PatientConfig.from_dicom(ct_folder, dose_path, plan_path, struct_names, recenter)

    @classmethod
    def from_nifti(
        cls,
        folder_path,
        **dose_config_fields
        ) -> 'PatientConfig':
        patient = PatientConfig.from_nifti(folder_path)

        return cls(
            patient=patient,
            **dose_config_fields
        )
    
    @classmethod
    def from_dicom(
        cls,
        ct_folder: str, 
        dose_path: str | None, 
        plan_path: str | None, 
        struct_names: List[str] | None = None, 
        recenter: bool = True,
        **dose_config_fields
        ) -> 'PatientConfig':
        patient = PatientConfig.from_dicom(
            ct_folder=ct_folder, 
            dose_path=dose_path, 
            plan_path=plan_path, 
            struct_names=struct_names, 
            recenter=recenter)

        return cls(
            patient=patient,
            **dose_config_fields
        )
    
    model_config = SettingsConfigDict(
        env_prefix="AUTOPLAN_DM_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
