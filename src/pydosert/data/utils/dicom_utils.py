import pydicom
import os
import numpy as np
import SimpleITK as sitk
from rt_utils import RTStructBuilder
from typing import Any, Optional, List
import math
from pydosert.data.beam import Beam
import torch

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