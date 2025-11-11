import pydicom
import os
import numpy as np
import SimpleITK as sitk
from rt_utils import RTStructBuilder
from typing import Any, Optional, List

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
        iso_center = reference_origin + np.array(reference_dose_size) / 2.0 * np.array(reference_spacing)
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

def get_rtdose_info(rtdose_path):
    ds = pydicom.dcmread(rtdose_path)
    origin = np.array(ds.ImagePositionPatient, dtype=np.float32)
    spacing = list(map(float, ds.PixelSpacing))
    grid_frame_offset = np.array(ds.GridFrameOffsetVector, dtype=np.float32)
    slice_thickness = np.abs(grid_frame_offset[1] - grid_frame_offset[0])
    spacing.append(slice_thickness)
    shape = (ds.Rows, ds.Columns, len(grid_frame_offset))
    return origin, spacing, shape

def fetch_plan_data(plan_path: str, scaling: float) -> str:
    """Summarizes the RTPLAN beam information in the dataset."""
    ds = pydicom.dcmread(plan_path)
    data = dict()
    beam_metersets = dict()
    for ref_seq in ds.FractionGroupSequence[0].ReferencedBeamSequence:
        if hasattr(ref_seq, "BeamMeterset"):
            beam_metersets[str(ref_seq.ReferencedBeamNumber)] = ref_seq.BeamMeterset
            
    for beam in ds.BeamSequence:
        beam_data = []
        jaw_data = []
        for index, cps in enumerate(beam.ControlPointSequence):
            if "BeamLimitingDevicePositionSequence" in cps:
                for sequence in cps.BeamLimitingDevicePositionSequence:
                    if sequence.RTBeamLimitingDeviceType == "MLCX":
                        beam_meterset = beam_metersets[str(beam.BeamNumber)]
                        if hasattr(cps, "CumulativeMetersetWeight"):
                            if (len(beam.ControlPointSequence) == 2):
                                mu_value = beam_meterset
                            else:
                                mu_value = beam_meterset * cps.CumulativeMetersetWeight
                        seq_data = {
                            "clockwise": cps.GantryRotationDirection,
                            "angle": cps.GantryAngle,
                            "ssd": cps.SourceToSurfaceDistance,
                            "mu": mu_value,
                            "lower": sequence.LeafJawPositions[int(len(sequence.LeafJawPositions) / 2):],
                            "higher": sequence.LeafJawPositions[:int(len(sequence.LeafJawPositions) / 2)],
                            }
                        
                        beam_data.append(seq_data)
                    elif sequence.RTBeamLimitingDeviceType == "ASYMY":
                        jaw = {
                            "lower": sequence.LeafJawPositions[0],
                            "higher": sequence.LeafJawPositions[1],
                        }
                        jaw_data.append(jaw)
        if (len(beam_data) > 0):
            for _beam in beam_data:
                _beam["jaw_lower"] = jaw_data[0]["lower"]
                _beam["jaw_higher"] = jaw_data[0]["higher"]
            data[str(beam.BeamNumber)] = beam_data
    
    parameters = []
    for index, beam_data in enumerate(data):
        beams = data[beam_data]

        if ((len(beams) == 0)):
            continue

        multi_cp = len(beams) > 1
        if (multi_cp and beams[0]['angle'] == 0):
            continue

        mus =  np.array([beam["mu"] for beam in beams])
        if (multi_cp):
            mus = np.abs(np.diff(mus))
        mus = np.expand_dims(mus, axis=0)

        beam_higher = np.array([beam["higher"] for beam in beams[1:]])
        beam_lower = np.array([beam["lower"] for beam in beams[1:]])
        jaw_higher = np.array([beam["jaw_higher"] for beam in beams[1:]])
        jaw_lower = np.array([beam["jaw_lower"] for beam in beams[1:]])

        # beam_higher_start = np.array([beam["higher"] for beam in beams[:-1]])
        # beam_higher_end = np.array([beam["higher"] for beam in beams[1:]])
        # beam_higher = (beam_higher_start + beam_higher_end) / 2.0

        # beam_lower_start = np.array([beam["lower"] for beam in beams[:-1]])
        # beam_lower_end = np.array([beam["lower"] for beam in beams[1:]])
        # beam_lower = (beam_lower_start + beam_lower_end) / 2.0

        leafs = np.stack([beam_higher, beam_lower], axis=0)
        leafs = np.expand_dims(leafs, axis=0)

        jaws = np.stack([jaw_lower, jaw_higher], axis=0)
        jaws = np.expand_dims(jaws, axis=0)

        parameters.append((leafs, jaws, mus))
    clockwise = beams[0]["clockwise"] != "CC"
    starting_angle = beams[1]["angle"] # (beams[0]["angle"] + beams[1]["angle"]) / 2.0

    return leafs, jaws, mus, clockwise, starting_angle


def load_structures(ct_series, folder_path, struct_names: List[str] | None = None):
    struct_path = [os.path.join(folder_path, path) for path in os.listdir(folder_path) if ("RTSTRUCT" in path or "RS" in path)]
    
    masks = dict()
    if (len(struct_path) > 0):
        rtstruct = RTStructBuilder.create_from(
        dicom_series_path=folder_path, 
        rt_struct_path=struct_path[0]
        )
        if struct_names is None:
            struct_names = rtstruct.get_roi_names()

        masks = dict()
        for struct_name in struct_names:
            mask_np = rtstruct.get_roi_mask_by_name(struct_name)
            mask = sitk.GetImageFromArray(np.transpose(mask_np.astype(np.float32), (2, 0, 1)))
            mask.SetOrigin(ct_series.GetOrigin())
            mask.SetDirection(ct_series.GetDirection())
            mask.SetSpacing(ct_series.GetSpacing())
            masks[struct_name] = mask
    return masks

def load_dose(path):
    # Load dose volumes
    doses = dict()
    dose_idx = 0

    scaling = float(pydicom.dcmread(path).DoseGridScaling)
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    dose = reader.Execute()
    dose = sitk.Cast(dose, sitk.sitkFloat32)
    dose = scaling * dose
    
    dataset = pydicom.dcmread(path)
    plan_sequence = dataset.ReferencedRTPlanSequence

    if len(plan_sequence) == 0 or not hasattr(plan_sequence[0], "ReferencedFractionGroupSequence"):
        beam_name = "dose_" + str(dose_idx)
    else:
        beam_name = str(plan_sequence[0].ReferencedFractionGroupSequence[0].ReferencedBeamSequence[0].ReferencedBeamNumber)

    return dose


def resample_to_iso_center(image, iso_center, spacing, size, pixel_value=0, interpolation=sitk.sitkLinear):
    dim = image.GetDimension()
    direction = np.eye(dim).flatten()

    center_index = np.array(size) / 2.0
    origin = iso_center - center_index * np.array(spacing)

    ref_img = sitk.Image(size, image.GetPixelIDValue())
    ref_img.SetSpacing(spacing)
    ref_img.SetOrigin(origin.tolist())
    ref_img.SetDirection(direction.tolist())

    return sitk.Resample(image, ref_img, sitk.Transform(), interpolation, pixel_value), ref_img