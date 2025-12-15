import pydicom
import os
import numpy as np
import SimpleITK as sitk
from rt_utils import RTStructBuilder
from typing import Any, Optional, List
import math
from pydose_rt.data.beam import Beam
import torch

def resample_based_on_dose(ct_series, structures, dose):
    
    reference_dose = dose
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_dose)
    ct_series = resample.Execute(ct_series)

    for k in structures:
        structures[k] = resample.Execute(structures[k])
    return ct_series, structures

def resample_based_on_plan(ct_series, structures, dose, recenter, plan_path):
    reference_dose = dose
    reference_spacing = reference_dose.GetSpacing()
    reference_dose_size = reference_dose.GetSize()
    reference_origin = reference_dose.GetOrigin()


    if recenter:

        max_slice_size = np.max(reference_dose_size[0:2])
        max_slice_size = 2 * (max_slice_size // 2)
        reference_size = tuple(int(x) for x in [
                max_slice_size,
                max_slice_size,
                2 * (reference_dose_size[2] // 2)
            ])
        iso_center = np.array(get_iso_from_rtplan(plan_path), dtype=np.float64)
        # Resample CT
        ct_series, _ = resample_to_iso_center(ct_series, iso_center, reference_spacing, reference_size, -1000)

        # Resample all dose volumes
        dose, _ = resample_to_iso_center(dose, iso_center, reference_spacing, reference_size, 0)

        for k in structures:
            structures[k], _ = resample_to_iso_center(structures[k], iso_center, reference_spacing, reference_size, 0, sitk.sitkNearestNeighbor)
    else:
        iso_center = reference_origin + (np.array(reference_dose_size) - 1) / 2.0 * np.array(reference_spacing) # reference_origin + np.array(reference_dose_size) / 2.0 * np.array(reference_spacing)
    return ct_series, structures, dose, iso_center

def load_ct_images(folder_path):
    """Loads all CT DICOM files from a specified folder."""
    ct_images = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        try:
            ds = pydicom.dcmread(file_path)  # Read DICOM file

            # Check if the file is a CT image
            if hasattr(ds, "Modality") and ds.Modality == "CT":
                ct_images.append(ds)

        except Exception as e:
            print(f"Skipping {filename}: {e}")  # Handle errors gracefully

    return ct_images

def load_ct_series(ct_folder):
    # Load and sort CT slices
    slices = load_ct_images(ct_folder)
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))  # sort by Z position

    # Extract metadata from first slice
    origin = np.array(slices[0].ImagePositionPatient, dtype=np.float64)
    spacing = list(map(float, slices[0].PixelSpacing))
    slice_thickness = float(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
    spacing.append(abs(slice_thickness))

    # Direction cosines (DICOM uses row, column, slice direction vectors)
    orientation = slices[0].ImageOrientationPatient  # [row_x, row_y, row_z, col_x, col_y, col_z]
    row_dir = np.array(orientation[:3])
    col_dir = np.array(orientation[3:])
    slice_dir = np.cross(row_dir, col_dir)
    direction = np.concatenate([row_dir, col_dir, slice_dir])

    # Create numpy volume
    volume = np.stack([s.pixel_array for s in slices], axis=-1)  # shape: (rows, cols, slices)

    # Convert to float32 for proper intensity scaling
    volume = volume.astype(np.int16)

    # Apply rescale intercept/slope
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope
    volume = volume * slope + intercept

    # Convert to sitk.Image
    sitk_img = sitk.GetImageFromArray(np.transpose(volume, (2, 0, 1)))  # Transpose to (z, y, x)
    sitk_img.SetSpacing(spacing)
    sitk_img.SetOrigin(origin)
    sitk_img.SetDirection(direction)

    return sitk_img, slices[0]

def get_iso_from_rtplan(rtplan_path):
    ds = pydicom.dcmread(rtplan_path)
    # Assuming single beam
    beam = ds.BeamSequence[0]
    iso = np.array(beam.ControlPointSequence[0].IsocenterPosition, dtype=np.float32)  # [x, y, z]
    return iso

def fetch_plan_data(plan_path: str) -> str:
    """Summarizes the RTPLAN beam information in the dataset."""
    ds = pydicom.dcmread(plan_path)
    data = dict()
    beam_metersets = dict()
    
    for ref_seq in ds.FractionGroupSequence[0].ReferencedBeamSequence:
        if hasattr(ref_seq, "BeamMeterset"):
            beam_metersets[str(ref_seq.ReferencedBeamNumber)] = ref_seq.BeamMeterset
            number_of_fractions = int(ds.FractionGroupSequence[0].NumberOfFractionsPlanned)
            
    parameters = dict()
    for beam in ds.BeamSequence:
        beam_data = []
        bld_angle = 0.0
        old_mu_value = 0.0
        iso_center = None
        for cps in beam.ControlPointSequence:
            if "BeamLimitingDevicePositionSequence" in cps:
                asymy_seqs = [seq for seq in cps.BeamLimitingDevicePositionSequence if seq.RTBeamLimitingDeviceType == "ASYMY"]
                if (len(asymy_seqs) > 0):
                    jaw_positions = np.stack([float(asymy_seqs[0].LeafJawPositions[0]), 
                                              float(asymy_seqs[0].LeafJawPositions[1])], 0)
                    
                for sequence in cps.BeamLimitingDevicePositionSequence:
                    if sequence.RTBeamLimitingDeviceType == "MLCX":
                        beam_meterset = beam_metersets[str(beam.BeamNumber)]
                        if hasattr(cps, "BeamLimitingDeviceAngle"):
                            bld_angle = float(cps.BeamLimitingDeviceAngle)
                        if hasattr(cps, "CumulativeMetersetWeight"):
                            if (len(beam.ControlPointSequence) == 2):
                                mu_value = beam_meterset
                            else:
                                mu_value = beam_meterset * cps.CumulativeMetersetWeight
                        if hasattr(cps, "IsocenterPosition"):
                            if iso_center is None:
                                iso_center = (float(cps.IsocenterPosition[2]), float(cps.IsocenterPosition[1]), float(cps.IsocenterPosition[0]))
                        beam_data.append(Beam(gantry_angle=math.radians(cps.GantryAngle), 
                            collimator_angle=math.radians(bld_angle), 
                            ssd=cps.SourceToSurfaceDistance,
                            mu=torch.from_numpy(np.array(mu_value - old_mu_value)),
                            leaf_positions=torch.from_numpy(np.stack(
                                [np.array(sequence.LeafJawPositions[:int(len(sequence.LeafJawPositions) / 2)]), 
                                 sequence.LeafJawPositions[int(len(sequence.LeafJawPositions) / 2):]], 1)),
                            jaw_positions=torch.from_numpy(jaw_positions),
                            field_size=(400, 400),
                            sid=float(beam.SourceAxisDistance),
                            iso_center=iso_center
                        ))
                        old_mu_value = mu_value
        
        if len(beam_data) > 0:
            parameters[f"{ds.SOPInstanceUID}_{beam.BeamNumber}"] = (beam_data, number_of_fractions)

    return parameters

def load_structures(ct_series, ct_folder_path, struct_path, struct_names: List[str] | None = None):
    
    masks = dict()
    if struct_path is not None:
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=ct_folder_path, 
            rt_struct_path=struct_path
        )
        
        available_names = rtstruct.get_roi_names()
        available_names = [name for name in available_names if not(name.startswith("z")) and not(name.startswith("_"))]
        
        if struct_names is None:
            matched_names = available_names
        else:
            matched_names = []
            for pattern in struct_names:
                matches = [name for name in available_names if pattern.upper() in name.upper()]
                
                if len(matches) == 0:
                    print(f"Warning: No ROI matching '{pattern}' found. Available: {available_names}")
                else:
                    if len(matches) > 1:
                        print(f"Ambiguous pattern '{pattern}' matches multiple ROIs: {matches}. Adding first one")
                    matched_names.append(matches[0])

        masks = dict()
        for idx, struct_name in enumerate(matched_names):
            mask_np = rtstruct.get_roi_mask_by_name(struct_name)
            mask = sitk.GetImageFromArray(np.transpose(mask_np.astype(np.float32), (2, 0, 1)))
            mask.SetOrigin(ct_series.GetOrigin())
            mask.SetDirection(ct_series.GetDirection())
            mask.SetSpacing(ct_series.GetSpacing())
            masks[struct_names[idx]] = mask
    return masks

def load_dose(path):
    # Load with pydicom for DoseGridScaling
    dataset = pydicom.dcmread(path)
    scaling = float(dataset.DoseGridScaling)

    # Load SimpleITK image
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    dose = reader.Execute()
    dose = sitk.Cast(dose, sitk.sitkFloat32) * scaling

    # Beam name
    beam_name = dataset.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
    beam_number = safe_get_beam_number(dataset)
    if beam_number is not None:
        beam_name = f"{beam_name}_{beam_number}"

    return dose, beam_name

def safe_get_beam_number(ds):
    """
    Safely extract:
    ReferencedRTPlanSequence[0]
      -> ReferencedFractionGroupSequence[0]
         -> ReferencedBeamSequence[0]
            -> ReferencedBeamNumber

    Returns:
        int | None
    """
    try:
        rtplan_seq = getattr(ds, "ReferencedRTPlanSequence", None)
        if not rtplan_seq or len(rtplan_seq) == 0:
            return None

        frac_seq = getattr(rtplan_seq[0], "ReferencedFractionGroupSequence", None)
        if not frac_seq or len(frac_seq) == 0:
            return None

        beam_seq = getattr(frac_seq[0], "ReferencedBeamSequence", None)
        if not beam_seq or len(beam_seq) == 0:
            return None

        return getattr(beam_seq[0], "ReferencedBeamNumber", None)

    except Exception:
        return None


def center_crop_or_pad_to_cube(img, cube_size_mm=400.0):
    """
    Make the image a cube of cube_size_mm (e.g. 400 mm = 40 cm) on each side,
    by center-cropping or padding as needed.
    """
    spacing = img.GetSpacing()  # (sx, sy, sz) in mm
    current_size = img.GetSize()  # (nx, ny, nz)

    # Target size in voxels for each dimension
    target_size = [
        int(round(cube_size_mm / s)) for s in spacing
    ]

    # ----- Step 1: center-crop if image is larger than target -----
    crop_lower = [0, 0, 0]
    crop_upper = [0, 0, 0]

    for i in range(3):
        diff = current_size[i] - target_size[i]
        if diff > 0:
            # We need to crop 'diff' voxels along this axis
            crop_lower[i] = diff // 2
            crop_upper[i] = diff - crop_lower[i]

    if any(c > 0 for c in crop_lower + crop_upper):
        img = sitk.Crop(img, lowerBoundaryCropSize=crop_lower,
                             upperBoundaryCropSize=crop_upper)
        current_size = img.GetSize()

    # ----- Step 2: center-pad if image is smaller than target -----
    pad_lower = [0, 0, 0]
    pad_upper = [0, 0, 0]

    for i in range(3):
        diff = target_size[i] - current_size[i]
        if diff > 0:
            # We need to pad 'diff' voxels along this axis
            pad_lower[i] = diff // 2
            pad_upper[i] = diff - pad_lower[i]

    if any(p > 0 for p in pad_lower + pad_upper):
        img = sitk.ConstantPad(img,
                               padLowerBound=pad_lower,
                               padUpperBound=pad_upper,
                               constant=0.0)

    return img

# def load_dose(path):
#     dataset = pydicom.dcmread(path)

#     scaling = float(dataset.DoseGridScaling)
#     reader = sitk.ImageFileReader()
#     reader.SetFileName(path)
#     dose = reader.Execute()
#     dose = sitk.Cast(dose, sitk.sitkFloat32)
#     dose = scaling * dose
#     beam_name = path.name

#     # plan_sequence = dataset.ReferencedRTPlanSequence
#     # if len(plan_sequence) == 0:
#     #     beam_name = path.name
#     # else:
#     #     beam_name = plan_sequence[0].ReferencedSOPInstanceUID
#     #     plan_sequence[0].ReferencedFractionGroupSequence[0].ReferencedBeamSequence[0].ReferencedBeamNumber
#     return dose, beam_name


def resample_to_iso_center(image, iso_center, spacing, size, pixel_value=0, interpolation=sitk.sitkLinear):
    dim = image.GetDimension()
    direction = np.eye(dim).flatten()

    center_index = (np.array(size) - 1) / 2.0 # np.array(size) / 2.0
    origin = iso_center - center_index * np.array(spacing)

    ref_img = sitk.Image(size, image.GetPixelIDValue())
    ref_img.SetSpacing(spacing)
    ref_img.SetOrigin(origin.tolist())
    ref_img.SetDirection(direction.tolist())

    return sitk.Resample(image, ref_img, sitk.Transform(), interpolation, pixel_value), ref_img