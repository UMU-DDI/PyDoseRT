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

    @property
    def shape(self) -> torch.Size:
        """
        Shape of the patient data.

        Returns the shape of `ct_array` and verifies that all structures
        (and dose, if present) have the same shape. Raises an error if any
        mismatch is found.
        """
        base_shape = self.ct_array.shape

        # Check structures
        for name, struct in self.structures.items():
            if struct.shape != base_shape:
                raise ValueError(
                    f"Structure '{name}' has shape {struct.shape}, "
                    f"but expected {base_shape} (same as ct_array)."
                )

        # Optionally also enforce dose shape consistency
        if self.dose is not None and self.dose.shape != base_shape:
            raise ValueError(
                f"Dose has shape {self.dose.shape}, "
                f"but expected {base_shape} (same as ct_array)."
            )

        return base_shape
    
    def to(self, target: torch.device | str | torch.dtype) -> 'Patient':
        """Move all tensors to a different device or dtype."""
        return Patient(
            ct_array=self.ct_array.to(target),
            structures={k: v.to(target) > 0 for k, v in self.structures.items()} if self.structures else {},
            dose=self.dose.to(target) if self.dose is not None else None,
            voxel_spacing_mm=self.voxel_spacing_mm
        )
    
    @property
    def physical_size(self) -> torch.Size:
        return np.multiply(
            np.array(self.ct_array.shape, dtype=np.float32),
            np.array(self.voxel_spacing_mm, dtype=np.float32),
        )

    def get_masked_dose(self, mask_name=None) -> torch.Tensor:
        """Returns the dose where the provided mask is true."""
        if mask_name is None:
            raise Exception("Mask name not provided")
        
        if mask_name not in self.structures:
            raise Exception(f"Mask {mask_name} does not exist in structures ({list(self.structures.keys())})")
        
        return torch.where(self.structures[mask_name], self.dose, 0.0)
    
    def get_masked_ct(self, mask_name=None) -> torch.Tensor:
        """Returns the CT array where the provided mask is true."""
        if mask_name is None:
            raise Exception("Mask name not provided")
        
        if mask_name not in self.structures:
            raise Exception(f"Mask {mask_name} does not exist in structures ({list(self.structures.keys())})")
        
        return torch.where(self.structures[mask_name], self.ct_array, -1000.0)
    
    def add_mask(self, mask_name: str, mask: np.ndarray | torch.Tensor, overwrite: bool = False):
        if not overwrite and (mask_name in self.structures):
            raise Exception(
                f"Mask {mask_name} already exists for the patient. "
                f"If you want to overwrite, set overwrite to True."
            )
        
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask) > 0
        elif isinstance(mask, torch.Tensor):
            mask = mask > 0
        else:
            raise Exception(f"Mask type {type(mask)} not supported.")

        # Enforce same shape as ct_array
        if mask.shape != self.ct_array.shape:
            raise ValueError(
                f"Mask '{mask_name}' has shape {mask.shape}, "
                f"but expected {self.ct_array.shape} (same as ct_array)."
            )
        
        self.structures[mask_name] = mask

@dataclass
class Phantom(Patient):
    """
    Phantom patient configuration for testing.

    Inherits from Patient.
    """

    def __init__(
        self,
        ct_array: np.ndarray,
        voxel_spacing_mm: tuple[float, float, float]
    ):
        super().__init__(
            ct_array=ct_array,
            structures={},
            dose=None,
            voxel_spacing_mm=voxel_spacing_mm
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
        ct_array = torch.ones(shape)

        return cls(
            ct_array=ct_array,
            voxel_spacing_mm=spacing
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
            voxel_spacing_mm=spacing
        )
