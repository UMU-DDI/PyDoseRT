"""
Patient configuration - CT dimensions and geometric parameters.
"""
# from pydantic import BaseModel, Field, model_validator
from dataclasses import dataclass, field
from token import OP
from typing import Optional, TYPE_CHECKING
from pydose_rt.physics.attenuation.hu_density_conversion import convert_HU_to_density
import torch
import numpy as np

if TYPE_CHECKING:
    from pydose_rt.data import Patient

@dataclass
class Patient:
    """
    Patient-specific configuration.

    """

    _ct_tensor: torch.Tensor | None = None
    _attenuation_tensor: torch.Tensor | None = None
    _resolution: tuple[float, float, float] = None
    structures: Optional[dict[str, torch.Tensor]] = field(default_factory=dict)
    dose: Optional[torch.Tensor] = None

    def __init__(self, ct_tensor=None, attenuation_tensor = None, structures: Optional[dict[str, torch.Tensor]] = dict(), dose: torch.Tensor = None, resolution=None) -> 'Patient':
        self._resolution = resolution
        self._ct_tensor = self.set_resolution(ct_tensor) if ct_tensor is not None else None
        self._attenuation_tensor = self.set_resolution(attenuation_tensor) if attenuation_tensor is not None else None
        self.structures = structures
        self.dose = self.set_resolution(dose) if dose is not None else None

    def __post_init__(self):
        # Enforce that structures and dose have same shape as density_image
        base_shape = self.density_image.shape

        for name, struct in self.structures.items():
            if struct.shape != base_shape:
                raise ValueError(
                    f"Structure '{name}' has shape {struct.shape}, "
                    f"but expected {base_shape} (same as density_image)."
                )

        if self.dose is not None and self.dose.shape != base_shape:
            raise ValueError(
                f"Dose has shape {self.dose.shape}, "
                f"but expected {base_shape} (same as density_image)."
            )
        
    def set_resolution(self, x: torch.Tensor):
        x.resolution = self._resolution
        return x
    
    @property
    def density_image(self) -> torch.Tensor:
        """

        """
        if self._ct_tensor is not None:
            density_image = convert_HU_to_density(self._ct_tensor)
        elif self._attenuation_tensor is not None:
            density_image = self._attenuation_tensor
            
        return self.set_resolution(density_image)
    
    def to(self, target: torch.device | str | torch.dtype) -> 'Patient':
        """Move all tensors to a different device or dtype."""
        return Patient(
            ct_tensor=self.set_resolution(self._ct_tensor.to(target)) if self._ct_tensor is not None else None, 
            attenuation_tensor = self.set_resolution(self._attenuation_tensor.to(target)) if self._attenuation_tensor is not None else None, 
            structures={k: v.to(target) > 0 for k, v in self.structures.items()} if self.structures else {},
            dose=self.set_resolution(self.dose.to(target)) if self.dose is not None else None,
            resolution=self._resolution
        )
    
    @property
    def device(self) -> torch.device:
        return self.density_image.attenuation.data.device
    
    @property
    def dtype(self) -> type:
        return self.density_image.attenuation.data.dtype
    

    @property
    def physical_size(self) -> torch.Size:
        return np.multiply(
            np.array(self.density_image.shape, dtype=np.float32),
            np.array(self._resolution, dtype=np.float32),
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
        
        return torch.where(self.structures[mask_name], self.density_image, -1000.0)
    
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

        # Enforce same shape as density_image
        if mask.shape != self.density_image.shape:
            raise ValueError(
                f"Mask '{mask_name}' has shape {mask.shape}, "
                f"but expected {self.density_image.shape} (same as density_image)."
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
        density_image: np.ndarray,
        resolution: tuple[float, float, float]
    ):
        super().__init__(
            ct_tensor=density_image,
            structures={},
            dose=None,
            resolution=resolution
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
        density_image = torch.ones(shape)

        return cls(
            density_image=density_image,
            resolution=spacing
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

        density_image = torch.from_numpy(np.expand_dims(np.where(distances <= radius_mm, ct_value, background_value), 0))

        return cls(
            density_image=density_image,
            resolution=spacing
        )
