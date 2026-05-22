import pydicom
import os
import numpy as np
import SimpleITK as sitk
from rt_utils import RTStructBuilder
from typing import Any, Optional, List
import math
from pydosert.data.beam import Beam
import torch

ROI_SYNONYM_CONFIG = {    
      "CTV": [
        "CTV", "CTV_Prostata", "CTV_Prostata_6000", "CTV_Prostata_gol_6000",
        "CTV_Samenblasen_gol_6000", "CTV_Prostata", "CTV:Prostata_6000", "zaa_CTV_Prostata_6000",
        "CTV_Prostata_gol_6000", "CTV_Prostata_gol", "CTV_Prostata60_a", "CTV_Prostata60_auto"
      ],
      "PTV": [
        "PTV", "PTV_Prostata", "PTV_Prostata_6000", "PTV_Prostata_gol_6000",
        "PTV_Samenblasen_gol_6000", "PTV_Prostata", "PTV:Prostata_6000",
        "PTV_Prostata_gol_6000", "PTV_Prostata_gol", "PTV_Prostata60_a", "PTV_Prostata60_auto"
      ],
      "Bladder": ["Blase", "Bladder", "Blad"],
      "Rectum": ["Rektum", "Rectum", "Colon", "Rect"],
      "FemoralHead_L": ["FemoralHead_L", "Femurkopf_Links", "Femoral Head_Left", "Hüftkopf_Links", "Hüftkopf links"],
      "FemoralHead_R": ["FemoralHead_R", "Femurkopf_Rechts", "Femoral Head_Right", "Hüftkopf_Rechts", "Hüftkopf rechts"],
      "Body": ["Körper", "Body", "Torso"]
    }

def load_ct_images(folder_path):
    """
    Load all CT-modality DICOM datasets from a folder.

    Args:
        folder_path (str): Folder containing DICOM files.

    Returns:
        list[pydicom.Dataset]: Datasets whose Modality is "CT" (unsorted).
    """
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
    """
    Load a CT series into a SimpleITK volume, applying rescale slope/intercept.

    Slices are sorted by Z position and stacked; the resulting array is
    transposed to (z, y, x) before being wrapped as a SimpleITK image with
    spacing, origin and direction set from the DICOM metadata.

    Args:
        ct_folder (str): Folder containing CT DICOM files.

    Returns:
        tuple[sitk.Image, pydicom.Dataset]: The CT volume (HU, axes z, y, x) and
            the first (reference) slice dataset.
    """
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
    """
    Parse an RTPLAN into per-beam control-point Beam objects.

    For each MLCX control point a Beam is built: leaf_positions [N, 2] from the
    split LeafJawPositions (left, right), jaw_positions [2] from the ASYMY jaw,
    a scalar incremental mu (difference of cumulative meterset weights), gantry
    and collimator angles in radians, plus isocenter (z, y, x) and SID/SSD.

    Args:
        plan_path (str): Path to the RTPLAN DICOM file.

    Returns:
        dict[str, tuple[list[Beam], int]]: Keyed by
            "{SOPInstanceUID}_{BeamNumber}", each value is the list of per
            control-point Beam objects and the planned number of fractions.
    """
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
    """
    Load RTSTRUCT ROIs as SimpleITK masks resampled onto the CT grid.

    Requested names are matched against available ROI names by substring, then
    by the ROI_SYNONYM_CONFIG synonym table; unmatched names are skipped.

    Args:
        ct_series (sitk.Image): Reference CT volume (z, y, x) for origin,
            direction and spacing.
        ct_folder_path (str): Folder of CT DICOM files (for RTStructBuilder).
        struct_path (str | None): Path to the RTSTRUCT file; None returns {}.
        struct_names (List[str] | None): Requested structure names; None loads all.

    Returns:
        dict[str, sitk.Image]: Mask images keyed by requested name, each (z, y, x)
            on the CT grid.
    """
    masks = dict()
    if struct_path is not None:
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=ct_folder_path, 
            rt_struct_path=struct_path
        )
        
        # Available ROI names are those present in the RTSTRUCT
        available_names = rtstruct.get_roi_names()
        available_names = [name for name in available_names]

        if struct_names is None:
            # If no specific structure name set is provided, take all available names
            matched_names = available_names
        else:
            # If specific structure names are provided, try to match them with available names using synonyms
            matched_names = {}
            for name in struct_names:
                # If name pattern is present directly, take that
                temp_names = []
                for available_name in available_names:
                    if name.lower() in available_name.lower():
                        temp_names.append(available_name)
                if len(temp_names) == 1:
                    matched_names[name] = temp_names[0]
                    continue
                elif len(temp_names) > 1:
                    print(f"Warning: Multiple matches found for '{name}' in RTSTRUCT: {temp_names}. Using the first match.")
                    matched_names[name] = temp_names[0]
                    continue
                # If len(temp_names) == 0, try to find synonyms

                found = False
                for canonical_name, synonyms in ROI_SYNONYM_CONFIG.items():
                    # Check if there are synonyms for this requested structure
                    if name.lower() == canonical_name.lower():
                        for synonym in synonyms:
                            if synonym.lower() in available_names:
                                matched_names[name] = synonym
                                found = True
                                break
                    if found:
                        break
                if not found:
                    print(f"Warning: Structure '{name}' not found in RTSTRUCT and no suitable synonym found. Skipping.")

        masks = dict()
        for req_name, matched_name in matched_names.items():
            mask_np = rtstruct.get_roi_mask_by_name(matched_name)
            mask = sitk.GetImageFromArray(np.transpose(mask_np.astype(np.float32), (2, 0, 1)))
            mask.SetOrigin(ct_series.GetOrigin())
            mask.SetDirection(ct_series.GetDirection())
            mask.SetSpacing(ct_series.GetSpacing())
            masks[req_name] = mask
    return masks

def load_dose(path):
    """
    Load an RTDOSE file and apply its DoseGridScaling.

    Args:
        path (str | Path): Path to the RTDOSE DICOM file.

    Returns:
        tuple[sitk.Image, str]: The scaled dose volume (z, y, x) and a beam name
            built from the referenced RTPLAN SOPInstanceUID (suffixed with the
            referenced beam number when available).
    """
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

    Args:
        ds (pydicom.Dataset): RTDOSE dataset.

    Returns:
        int | None: Referenced beam number, or None if any link is missing.
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