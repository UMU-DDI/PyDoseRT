"""
Patient configuration - CT dimensions and geometric parameters.
"""
# from pydantic import BaseModel, Field, model_validator
from dataclasses import dataclass, field
from token import OP
from typing import Optional, TYPE_CHECKING
from pydosert.physics.attenuation.hu_density_conversion import convert_HU_to_density
import torch
import numpy as np

if TYPE_CHECKING:
    from pydosert.data import Patient

@dataclass
class Patient:
    """
    Patient-specific configuration holding CT/attenuation, structures and dose.

    Attributes:
        _ct_tensor (torch.Tensor | None): CT image in Hounsfield units [D, H, W].
        _attenuation_tensor (torch.Tensor | None): Precomputed density/attenuation
            image [D, H, W], used when no CT tensor is provided.
        resolution (tuple[float, float, float]): Voxel spacing (z, y, x) in mm.
        structures (dict[str, torch.Tensor]): Binary masks keyed by name, each [D, H, W].
        dose (torch.Tensor | None): Dose grid [D, H, W].
        number_of_fractions (int): Number of treatment fractions.
    """

    _ct_tensor: torch.Tensor | None = None
    _attenuation_tensor: torch.Tensor | None = None
    resolution: tuple[float, float, float] = None
    structures: Optional[dict[str, torch.Tensor]] = field(default_factory=dict)
    dose: Optional[torch.Tensor] = None
    number_of_fractions: int = 1

    def __init__(self, 
                 ct_tensor=None, 
                 attenuation_tensor = None,         
                 structures: Optional[dict[str, torch.Tensor | np.ndarray]] = None,
                 dose: torch.Tensor = None, 
                 resolution=None,
                 number_of_fractions: int = 1) -> 'Patient':
        """
        Initialize a Patient.

        Args:
            ct_tensor (torch.Tensor | None): CT image in Hounsfield units [D, H, W].
            attenuation_tensor (torch.Tensor | None): Precomputed density/attenuation
                image [D, H, W], used when ct_tensor is None.
            structures (dict[str, torch.Tensor | np.ndarray] | None): Binary masks
                keyed by name, each [D, H, W]; None defaults to empty dict.
            dose (torch.Tensor): Dose grid [D, H, W].
            resolution (tuple[float, float, float]): Voxel spacing (z, y, x) in mm.
            number_of_fractions (int): Number of treatment fractions.
        """
        self.resolution = resolution
        self._ct_tensor = ct_tensor if ct_tensor is not None else None
        self._attenuation_tensor = attenuation_tensor if attenuation_tensor is not None else None
        self.dose = dose if dose is not None else None
        self.number_of_fractions = number_of_fractions

        if structures is None:
            self.structures = {}
        else:
            self.structures = structures

    def __post_init__(self):
        """
        Validate that every structure mask and the dose share density_image shape [D, H, W].

        Raises:
            ValueError: If any structure or the dose has a shape different from
                density_image.
        """
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
    
    @property
    def density_image(self) -> torch.Tensor:
        """
        Density image [D, H, W] derived from the CT (HU converted to density) or
        the attenuation tensor.

        Returns:
            torch.Tensor: Density image [D, H, W].
        """
        if self._ct_tensor is not None:
            density_image = convert_HU_to_density(self._ct_tensor)
        elif self._attenuation_tensor is not None:
            density_image = self._attenuation_tensor
            
        return density_image
    
    def to(self, target: torch.device | str | torch.dtype) -> 'Patient':
        """
        Move all tensors to a different device or dtype.

        Args:
            target (torch.device | str | torch.dtype): Target device or dtype.

        Returns:
            Patient: New patient with CT/attenuation [D, H, W], structures [D, H, W]
                (re-binarized) and dose [D, H, W] on the target device/dtype.
        """
        return Patient(
            ct_tensor=self._ct_tensor.to(target) if self._ct_tensor is not None else None, 
            attenuation_tensor = self._attenuation_tensor.to(target) if self._attenuation_tensor is not None else None, 
            structures={k: v.to(target) > 0 for k, v in self.structures.items()} if self.structures else {},
            dose=self.dose.to(target) if self.dose is not None else None,
            resolution=self.resolution,
            number_of_fractions=self.number_of_fractions
        )
    
    @property
    def device(self) -> torch.device:
        """Device of the underlying density image tensor."""
        return self.density_image.attenuation.data.device

    @property
    def dtype(self) -> type:
        """Data type of the underlying density image tensor."""
        return self.density_image.attenuation.data.dtype


    @property
    def physical_size(self) -> torch.Size:
        """
        Physical extent of the volume in mm, per axis.

        Returns:
            np.ndarray: Physical size (z, y, x) in mm [3], i.e. density_image
                shape [D, H, W] times resolution.
        """
        return np.multiply(
            np.array(self.density_image.shape, dtype=np.float32),
            np.array(self.resolution, dtype=np.float32),
        )

    def get_masked_dose(self, mask_name=None) -> torch.Tensor:
        """
        Return the dose inside the named structure, zero elsewhere.

        Args:
            mask_name (str): Name of a structure mask in self.structures.

        Returns:
            torch.Tensor: Dose [D, H, W] kept where the mask is True, else 0.

        Raises:
            Exception: If mask_name is None or not present in structures.
        """
        if mask_name is None:
            raise Exception("Mask name not provided")
        
        if mask_name not in self.structures:
            raise Exception(f"Mask {mask_name} does not exist in structures ({list(self.structures.keys())})")
        
        return torch.where(self.structures[mask_name], self.dose, 0.0)
    
    def get_masked_ct(self, mask_name=None) -> torch.Tensor:
        """
        Return the density image inside the named structure, -1000 elsewhere.

        Args:
            mask_name (str): Name of a structure mask in self.structures.

        Returns:
            torch.Tensor: density_image [D, H, W] kept where the mask is True,
                else -1000.0.

        Raises:
            Exception: If mask_name is None or not present in structures.
        """
        if mask_name is None:
            raise Exception("Mask name not provided")
        
        if mask_name not in self.structures:
            raise Exception(f"Mask {mask_name} does not exist in structures ({list(self.structures.keys())})")
        
        return torch.where(self.structures[mask_name], self.density_image, -1000.0)
    
    def add_mask(self, mask_name: str, mask: np.ndarray | torch.Tensor, overwrite: bool = False):
        """
        Add a binary structure mask, validating it matches the volume shape.

        Args:
            mask_name (str): Name to store the mask under.
            mask (np.ndarray | torch.Tensor): Mask [D, H, W]; binarized as (mask > 0).
            overwrite (bool): If False, raise when mask_name already exists.

        Raises:
            Exception: If the mask already exists (and overwrite is False) or the
                mask type is unsupported.
            ValueError: If the mask shape differs from density_image [D, H, W].
        """
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
        ct_image: np.ndarray,
        resolution: tuple[float, float, float]
    ):
        """
        Initialize a Phantom from a CT image.

        Args:
            ct_image (np.ndarray): CT/HU image [D, H, W].
            resolution (tuple[float, float, float]): Voxel spacing (z, y, x) in mm.
        """
        super().__init__(
            ct_tensor=ct_image,
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
        Create a uniform (all-zero HU) Phantom.

        Args:
            shape (tuple[int, int, int]): Volume shape (D, H, W) in voxels.
            spacing (tuple[float, float, float]): Voxel spacing (z, y, x) in mm.

        Returns:
            Phantom: Phantom whose CT image is zeros [D, H, W].
        """
        ct_image = torch.zeros(shape)

        return cls(
            ct_image=ct_image,
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
        Create a Phantom containing a centered water sphere in a background medium.

        Args:
            shape (tuple[int, int, int]): Volume shape (D, H, W) in voxels.
            spacing (tuple[float, float, float]): Voxel spacing (z, y, x) in mm.
            radius_mm (float): Sphere radius in mm.
            ct_value (float): CT/HU value assigned inside the sphere.
            background_value (float): CT/HU value assigned outside the sphere.

        Returns:
            Phantom: Phantom whose CT image is [1, D, H, W] (sphere=ct_value,
                background=background_value).
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

        ct_image = torch.from_numpy(np.expand_dims(np.where(distances <= radius_mm, ct_value, background_value), 0))

        return cls(
            ct_image=ct_image,
            resolution=spacing
        )