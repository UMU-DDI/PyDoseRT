"""
Core data structures for machine geometry, control points, and phantoms.
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Union
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import datetime

from .hardware import DEVICE

# Optional import for DICOM support
try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    print("⚠️ pydicom not installed. DICOM loading will be disabled.")


@dataclass
class MLCConfig:
    model: str
    leaf_boundaries: List[float]
    transmission: float
    dosimetric_leaf_gap_mm: float = 0.0


@dataclass
class ControlPoint:
    gantry_angle_deg: float
    collimator_angle_deg: float
    couch_angle_deg: float
    source_distance_mm: float
    isocenter_mm: Tuple[float, float, float]
    jaw_x_mm: Tuple[float, float]
    jaw_y_mm: Tuple[float, float]
    mlc_positions_mm: Optional[np.ndarray] = None
    monitor_units: float = 0.0

    @staticmethod
    def create_manual(
        gantry: float,
        field_size_mm: Tuple[float, float],
        iso: Tuple[float, float, float],
        mu: float = 100.0,
        coll: float = 0.0,
        couch: float = 0.0,
        sad: float = 1000.0,
    ):
        """Convenience factory for symmetric open fields."""
        hw = field_size_mm[0] / 2.0
        hh = field_size_mm[1] / 2.0
        return ControlPoint(
            gantry_angle_deg=gantry,
            collimator_angle_deg=coll,
            couch_angle_deg=couch,
            source_distance_mm=sad,
            isocenter_mm=iso,
            jaw_x_mm=(-hw, hw),
            jaw_y_mm=(-hh, hh),
            mlc_positions_mm=None,
            monitor_units=mu,
        )


@dataclass
class MachineConfig:
    energy: str
    gy_per_mu: float
    tpr20_10: float
    mlc: Optional[MLCConfig] = None
    
    # Reference Conditions (Added)
    reference_mu: float = 100.0
    reference_dose_gy: float = 1.0
    
    # Source Parameters
    geometric_penumbra_mm: Tuple[float, float] = (0.0, 0.0)
    head_scatter_sigma_mm: Tuple[float, float] = (0.0, 0.0)
    head_scatter_magnitude: float = 0.0
    
    # Curves
    profile_curve: Optional[List[Tuple[float, float]]] = None
    output_factor_curve: Optional[List[Tuple[float, float]]] = None
    
    # Deprecated but kept
    mlc_transmission: float = 0.0

    @staticmethod
    def load_from_json(file_path: str, energy: str = "10MV"):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 1. Parse MLC Config
        mlc = None
        if "mlc" in data:
            mlc_data = data["mlc"]
            mlc = MLCConfig(
                model=mlc_data.get("model", "Unknown"),
                leaf_boundaries=mlc_data.get("leaf_boundaries", []),
                transmission=mlc_data.get("transmission", 0.0),
                dosimetric_leaf_gap_mm=mlc_data.get("dosimetric_leaf_gap_mm", 0.0)
            )

        # 2. Parse Energy Specifics
        if energy not in data["energies"]:
            energy = energy.replace(" ", "")
            if energy not in data["energies"]:
                raise ValueError(f"Energy {energy} not found in {file_path}")

        e_data = data["energies"][energy]
        
        src = e_data.get("source", {})
        geo_pen = tuple(src.get("geometric_penumbra_mm", [0.0, 0.0]))
        sc_sigma = tuple(src.get("head_scatter_sigma_mm", [0.0, 0.0]))
        sc_mag = src.get("head_scatter_magnitude", 0.0)
        
        prof_data = e_data.get("profile", {}).get("curve", None)
        prof_curve = [tuple(p) for p in prof_data] if prof_data else None
        
        of_data = e_data.get("output_factors", {}).get("curve", None)
        of_curve = [tuple(p) for p in of_data] if of_data else None

        return MachineConfig(
            energy=energy,
            gy_per_mu=e_data.get("gy_per_mu", 1.0),
            tpr20_10=e_data.get("tpr20_10", 0.7),
            
            # FIX: Extract Reference values here
            reference_mu=e_data.get("reference_mu", 100.0),
            reference_dose_gy=e_data.get("reference_dose_gy", 1.0),
            
            mlc=mlc,
            geometric_penumbra_mm=geo_pen,
            head_scatter_sigma_mm=sc_sigma,
            head_scatter_magnitude=sc_mag,
            profile_curve=prof_curve,
            output_factor_curve=of_curve,
            mlc_transmission=mlc.transmission if mlc else 0.0
        )


class Phantom:
    """Simple voxelized phantom supporting primitive shapes."""

    def __init__(self, size_mm: Tuple[int, int, int], resolution_mm: float, device=DEVICE):
        self.res = resolution_mm
        
        # 1. Calculate Grid Dimensions (Indices)
        self.shape = (
            int(size_mm[0] / self.res),
            int(size_mm[1] / self.res),
            int(size_mm[2] / self.res),
        )
        
        # 2. Calculate Origin (Physical Coordinate of corner 0,0,0)
        # We calculate this from the *actual* shape to avoid rounding shifts.
        # This ensures physical (0,0,0) is exactly in the center of the grid.
        phys_x = self.shape[0] * self.res
        phys_y = self.shape[1] * self.res
        phys_z = self.shape[2] * self.res
        
        self.origin = torch.tensor(
            [-phys_x / 2.0, -phys_y / 2.0, -phys_z / 2.0],
            dtype=torch.float32,
            device=device,
        )
        
        self.device = device
        self.data = torch.zeros(self.shape, dtype=torch.float32, device=device)

    def get_geometrical_center(self) -> Tuple[float, float, float]:
        dims = torch.tensor(self.shape, device=self.device, dtype=torch.float32)
        center_vec = self.origin + (dims * self.res) / 2.0
        return tuple(center_vec.cpu().tolist())

    def add_cylinder(self, radius_mm, center_mm, density=1.0):
        """Insert a uniform-density cylinder along +z."""
        nx, ny, nz = self.shape
        # Coordinate grids based on the calculated origin
        x = self.origin[0] + (torch.arange(nx, device=self.device) + 0.5) * self.res
        y = self.origin[1] + (torch.arange(ny, device=self.device) + 0.5) * self.res
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        
        mask = ((grid_x - center_mm[0]) ** 2 + (grid_y - center_mm[1]) ** 2) <= radius_mm**2
        self.data[mask.unsqueeze(-1).expand(-1, -1, nz)] = density

    def get_density_tensor(self):
        """Return density tensor as (1,1,Z,Y,X) for grid_sample usage."""
        return self.data.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)


class DicomPhantom(Phantom):
    def __init__(self, dicom_folder: str, target_res_mm: float = 2.0, device=DEVICE):
        if not HAS_PYDICOM: raise ImportError("pydicom required")
        print(f"📂 Loading DICOM from: {dicom_folder}")
        
        # 1. Load & Sort Slices
        slices = []
        for fname in os.listdir(dicom_folder):
            if fname.lower().endswith(".dcm"):
                try:
                    ds = pydicom.dcmread(os.path.join(dicom_folder, fname))
                    # Check for CT Image Storage UID or just 'CT' modality
                    if ds.Modality != 'CT': continue 
                    if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "PixelData"):
                        slices.append(ds)
                except: pass
        
        if not slices: raise ValueError("No valid CT slices found.")
        
        # Sort by Z-position (ImagePositionPatient[2])
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        ref = slices[0]
        
        # 2. Extract Resolution (DICOM is usually [RowSpacing, ColSpacing])
        # PixelSpacing is [Row(Y), Col(X)]
        raw_y_res, raw_x_res = map(float, ref.PixelSpacing)
        raw_z_res = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))

        # 3. Stack Volume -> Shape is (Z, Y, X)
        raw_vol_zyx = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
        
        # Apply RescaleSlope and RescaleIntercept (HU Conversion)
        raw_vol_zyx = raw_vol_zyx * getattr(ref, "RescaleSlope", 1.0) + getattr(ref, "RescaleIntercept", -1024.0)

        # --- FIX: PERMUTE TO (X, Y, Z) ---
        # Your engine expects: Dim 0: X (Left-Right), Dim 1: Y (Ant-Post), Dim 2: Z (Sup-Inf)
        # Current: (Z, Y, X) -> Transpose to (X, Y, Z) i.e., (2, 1, 0)
        raw_vol_xyz = raw_vol_zyx.transpose(2, 1, 0) 
        
        # Calculate Physical Size based on new (X, Y, Z) order
        phys_size = [
            raw_vol_xyz.shape[0] * raw_x_res, 
            raw_vol_xyz.shape[1] * raw_y_res, 
            raw_vol_xyz.shape[2] * raw_z_res
        ]

        # 4. Setup Target Grid
        self.device = device
        self.res = target_res_mm
        self.shape = (
            int(phys_size[0] / self.res),
            int(phys_size[1] / self.res),
            int(phys_size[2] / self.res),
        )
        
        # Origin: DICOM ImagePositionPatient is already (X, Y, Z). 
        # Since we permuted our data to (X, Y, Z), this matches perfectly.
        self.origin = torch.tensor(list(map(float, ref.ImagePositionPatient)), dtype=torch.float32, device=device)
        
        print(f"   Raw CT (Permuted): {raw_vol_xyz.shape} | Target Grid: {self.shape}")

        # 5. Resample
        # We need to grid_sample the raw volume to the new resolution.
        # Input Tensor: (X, Y, Z). We treat D=X, H=Y, W=Z.
        raw_t = torch.tensor(raw_vol_xyz, dtype=torch.float32, device="cpu").unsqueeze(0).unsqueeze(0)

        def make_grid(idx, res, size):
            # Create normalized coordinates (-1 to 1)
            return (((torch.arange(idx, device="cpu") + 0.5) * res) / size) * 2.0 - 1.0

        gx = make_grid(self.shape[0], self.res, phys_size[0]) # Target X
        gy = make_grid(self.shape[1], self.res, phys_size[1]) # Target Y
        gz = make_grid(self.shape[2], self.res, phys_size[2]) # Target Z
        
        # Meshgrid: indexing='ij' gives output (X, Y, Z)
        X_grid, Y_grid, Z_grid = torch.meshgrid(gx, gy, gz, indexing="ij")
        
        # Stack for sampling: (Z, Y, X) -> maps to (W, H, D) -> maps to input (Z, Y, X) dims 2,1,0
        grid = torch.stack([Z_grid, Y_grid, X_grid], dim=-1).unsqueeze(0)
        
        resampled_hu = F.grid_sample(raw_t, grid, align_corners=False, mode="bilinear").squeeze()

        # 6. Density Conversion (USING LUT)
        # Move HU to device for fast lookup
        hu_device = resampled_hu.to(device)
        self.data = self._hu_to_density(hu_device)
        
        print("   ✅ DICOM Loaded, Permuted & Resampled (with LUT).")

    def _hu_to_density(self, hu: torch.Tensor) -> torch.Tensor:
        """
        Applies Piecewise Linear Interpolation using the specific LUT.
        HU:       [-1000, -976, -480, -96, 48, 230, 1170, 1850, 6400]
        Density:  [0.001, 0.001, 0.5, 0.95, 1.05, 1.15, 1.82, 2.7, 4.51]
        """
        # LUT Definition
        x = torch.tensor([-1000., -976., -480., -96., 48., 230., 1170., 1850., 6400.], device=hu.device)
        y = torch.tensor([0.00121, 0.00121, 0.5, 0.95, 1.05, 1.15, 1.82, 2.7, 4.51], device=hu.device)
        
        # 1. Identify intervals
        indices = torch.bucketize(hu, x)
        indices = torch.clamp(indices, 1, len(x)-1)
        
        x_left = x[indices - 1]
        x_right = x[indices]
        y_left = y[indices - 1]
        y_right = y[indices]
        
        # 2. Linear Interpolation
        denom = x_right - x_left + 1e-6
        t = (hu - x_left) / denom
        density = y_left + t * (y_right - y_left)
        
        # 3. Handle Out of Bounds
        # Below -1000 -> Clamp to min density (0.00121)
        density = torch.where(hu < x[0], y[0], density)
        # Above 6400 -> Extrapolate using the slope of the last segment
        # (Already handled by the linear formula as long as we use the last bucket parameters)
        
        return density


class ImportRTPlan:
    def __init__(self, mlc_config: Optional[MLCConfig] = None):
        self.mlc_config = mlc_config

    def load_plan(self, file_path: str) -> List[ControlPoint]:
        if not HAS_PYDICOM: raise ImportError("pydicom is required.")
        if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")

        ds = pydicom.dcmread(file_path)
        beams = []

        for beam in ds.BeamSequence:
            if beam.BeamType not in ["STATIC", "DYNAMIC"]: continue
            if beam.get("TreatmentDeliveryType") == "SETUP": continue

            # --- FIX: LOOKUP BEAM METERSET (MU) ---
            beam_mu = 100.0 # Default fallback
            found_mu = False
            
            if hasattr(ds, "FractionGroupSequence"):
                for fg in ds.FractionGroupSequence:
                    # Check if this Fraction Group references our beam
                    if hasattr(fg, "ReferencedBeamSequence"):
                        for rb in fg.ReferencedBeamSequence:
                            if rb.ReferencedBeamNumber == beam.BeamNumber:
                                if hasattr(rb, "BeamMeterset"):
                                    beam_mu = float(rb.BeamMeterset)
                                    found_mu = True
                                break
                    if found_mu: break # Stop searching if found
            
            if not found_mu:
                print(f"⚠️ Warning: MU not found for Beam {beam.BeamNumber}. Using 100.0")

            # Get Isocenter
            cps = beam.ControlPointSequence
            iso = cps[0].IsocenterPosition
            iso_tuple = (float(iso[0]), float(iso[1]), float(iso[2]))

            current_gantry = 0.0
            current_coll = 0.0
            current_couch = 0.0
            current_jaw_x = (-200.0, 200.0)
            current_jaw_y = (-200.0, 200.0)
            current_mlc = None
            prev_cum_weight = 0.0

            for i, cp in enumerate(cps):
                val_g = float(cp.get("GantryAngle", current_gantry))
                if "GantryAngle" in cp:
                    current_gantry = val_g

                val_coll = float(cp.get("BeamLimitingDeviceAngle", current_coll))
                if "BeamLimitingDeviceAngle" in cp:
                    current_coll = -val_coll

                val_c = float(cp.get("PatientSupportAngle", current_couch))
                if "PatientSupportAngle" in cp:
                    current_couch = val_c

                sad = float(cp.get("SourceToIsocenterDistance", 1000.0))

                if "BeamLimitingDevicePositionSequence" in cp:
                    for bld in cp.BeamLimitingDevicePositionSequence:
                        dev_type = bld.RTBeamLimitingDeviceType
                        pos = np.array(bld.LeafJawPositions, dtype=float)

                        if dev_type in ["ASYMX", "X"]:
                            current_jaw_x = (pos[0], pos[1])
                        elif dev_type in ["ASYMY", "Y"]:
                            current_jaw_y = (pos[0], pos[1])
                        elif "MLC" in dev_type:
                            n_leaves = len(pos) // 2
                            current_mlc = np.column_stack((pos[:n_leaves], pos[n_leaves:]))

                curr_cum_weight = float(cp.get("CumulativeMetersetWeight", 0.0))
                weight_delta = curr_cum_weight - prev_cum_weight
                prev_cum_weight = curr_cum_weight
                
                # Calculate MU for this specific segment
                segment_mu = weight_delta * beam_mu

                if segment_mu > 0.001 or i == 0:
                    beams.append(ControlPoint(
                        gantry_angle_deg=current_gantry,
                        collimator_angle_deg=current_coll,
                        couch_angle_deg=current_couch,
                        source_distance_mm=sad,
                        isocenter_mm=iso_tuple,
                        jaw_x_mm=current_jaw_x,
                        jaw_y_mm=current_jaw_y,
                        mlc_positions_mm=current_mlc,
                        monitor_units=segment_mu
                    ))
        return beams
    
class RTDoseExporter:
    """
    Exports the calculated dose grid to a DICOM RT Dose file.
    """
    @staticmethod
    def save(dose_grid: np.ndarray, 
             phantom: Phantom, 
             filename: str = "rtdose.dcm",
             patient_id: str = "123456",
             referenced_plan_uid: Optional[str] = None):
        
        if not HAS_PYDICOM:
            raise ImportError("pydicom is required to export RT Dose.")
        
        from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
        # --- FIX: Import UID class ---
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid, UID 

        print(f"💾 Exporting RT Dose to: {filename}...")

        # 1. Prepare File Meta
        file_meta = FileMetaDataset()
        # --- FIX: Wrap strings in UID() ---
        file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.481.2') # RT Dose Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = UID('1.2.3.4.5') 
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

        # 2. Patient / Study Info (Generic)
        ds.PatientName = "Hero^Dose^Phantom"
        ds.PatientID = patient_id
        ds.Modality = "RTDOSE"
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        
        dt = datetime.datetime.now()
        ds.ContentDate = dt.strftime('%Y%m%d')
        ds.ContentTime = dt.strftime('%H%M%S')
        ds.InstanceCreationDate = ds.ContentDate
        ds.InstanceCreationTime = ds.ContentTime

        # 3. RT Dose Specific Tags
        ds.DoseSummationType = "PLAN"
        ds.DoseUnits = "GY"
        ds.DoseType = "PHYSICAL"
        
        # Link to Plan (if provided)
        if referenced_plan_uid:
            ref_plan = Dataset()
            ref_plan.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.5' # RT Plan
            ref_plan.ReferencedSOPInstanceUID = referenced_plan_uid
            ds.ReferencedRTPlanSequence = [ref_plan]

        # 4. Geometry
        # Grid dimensions
        nx, ny, nz = dose_grid.shape
        ds.Columns = nx
        ds.Rows = ny
        ds.NumberOfFrames = nz
        
        # Pixel Spacing (X, Y) - DICOM uses [RowSpacing, ColSpacing] -> [Y_res, X_res]
        # Our engine uses isotropic resolution 'res'
        ds.PixelSpacing = [phantom.res, phantom.res]
        ds.SliceThickness = phantom.res
        
        # Origin: ImagePositionPatient
        # This defines the center of the first voxel (0,0,0).
        # In DoseEngine, physical coords are: origin + (idx + 0.5) * res.
        # So IPP = phantom.origin + 0.5 * res
        # Note: phantom.origin is a Tensor.
        origin_cpu = phantom.origin.cpu().numpy()
        ipp = origin_cpu + 0.5 * phantom.res
        ds.ImagePositionPatient = ipp.tolist()
        
        # Orientation: Standard HFS (1,0,0,0,1,0)
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        # Grid Frame Offset Vector (Z positions relative to IPP Z)
        # 0, res, 2*res ...
        offsets = np.arange(nz) * phantom.res
        ds.GridFrameOffsetVector = offsets.tolist()

        # 5. Pixel Data (Scaling)
        # DICOM Dose is usually uint16 or uint32. We scale Float -> Int.
        # Value = PixelData * GridScaling
        # We map Max Dose to ~60000 (uint16 range is 65535) for precision.
        
        max_dose = np.max(dose_grid)
        if max_dose > 0:
            scale_factor = max_dose / 60000.0
            pixel_data = (dose_grid / scale_factor).astype(np.uint16)
        else:
            scale_factor = 1.0
            pixel_data = dose_grid.astype(np.uint16)
            
        ds.DoseGridScaling = scale_factor
        
        # Permute for DICOM storage: (Frames, Rows, Cols) -> (Z, Y, X)
        # Our dose_grid is (X, Y, Z).
        # Transpose to (Z, Y, X).
        pixel_data_dicom = pixel_data.transpose(2, 1, 0)
        
        ds.PixelData = pixel_data_dicom.tobytes()
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0 # Unsigned integer
        
        ds.save_as(filename)
        print("   ✅ Export complete.")
