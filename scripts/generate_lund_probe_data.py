import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from pathlib import Path
import nibabel as nib
import pandas as pd
import SimpleITK as sitk
import scipy
import itertools
import pydose_rt.utils.path_utils as path_utils
from typing import Dict, List, Tuple

### Make 256x256 and 128x128 versions of the same dataset.
struct_dict = {
    "dose_interpolated.nii.gz": "dose",
    "image_reg2MRI.nii.gz": "CT",
    "mask_PTVT_427.nii.gz": "PTV",
    "mask_PenileBulb.nii.gz": "ROI1",
    "mask_FemoralHead_L.nii.gz": "ROI2",
    "mask_FemoralHead_R.nii.gz": "ROI3",
    "mask_Bladder.nii.gz": "ROI4",
    "mask_Rectum.nii.gz": "ROI5",
    "mask_BODY.nii.gz": "ROI7",
}


def _align_and_validate_shapes(
    rois: Dict[str, np.ndarray], names: List[str]
) -> Tuple[Dict[str, np.ndarray], Tuple[int, ...]]:
    """
    Ensure all requested rois exist, convert to bool arrays, and if shapes mismatch
    crop them to the minimum common shape (with a printed warning).
    Returns (normalized_rois, common_shape)
    """
    # check presence
    missing = [n for n in names if n not in rois]
    if missing:
        raise ValueError(f"_align_and_validate_shapes: missing ROIs: {missing}")

    # convert to bool and collect shapes
    shapes = [tuple(rois[n].shape) for n in names]
    # compute min shape across dimensions
    common_shape = tuple(min(s[d] for s in shapes) for d in range(len(shapes[0])))

    # If shapes differ, crop and print warning
    if not all(s == common_shape for s in shapes):
        print(
            f"WARNING: ROI shapes differ; cropping to common shape {common_shape}. Original shapes: {dict(zip(names, shapes))}"
        )

    normalized = {}
    for n in names:
        arr = np.asarray(rois[n]).astype(bool)
        if arr.shape != common_shape:
            slices = tuple(slice(0, cs) for cs in common_shape)
            arr = arr[slices]
        normalized[n] = arr
    return normalized, common_shape


def check_pairwise_rois_no_overlap(
    rois: Dict[str, np.ndarray], roi_names: List[str] = None
) -> None:
    """
    Print WARNING lines if any pair among the provided ROIs overlaps.
    rois: dict mapping roi name -> boolean numpy array
    roi_names: optional list of names to check; defaults to ['ROI1','ROI2','ROI3','ROI4','ROI5']
    """
    if roi_names is None:
        roi_names = ["ROI1", "ROI2", "ROI3", "ROI4", "ROI5"]

    # validate and normalize shapes (may raise if missing)
    normalized, _ = _align_and_validate_shapes(rois, roi_names)

    found_any = False
    for a, b in itertools.combinations(roi_names, 2):
        A = normalized[a]
        B = normalized[b]
        overlap_vox = int(np.count_nonzero(A & B))
        if overlap_vox > 0:
            found_any = True
            size_a = int(np.count_nonzero(A))
            size_b = int(np.count_nonzero(B))
            smaller = min(size_a, size_b) if min(size_a, size_b) > 0 else 1
            pct = overlap_vox / smaller * 100.0
            print(
                f"WARNING: Pairwise ROI overlap detected: {a} & {b} | overlap_vox={overlap_vox} | "
                f"size_{a}={size_a} | size_{b}={size_b} | % of smaller={pct:.2f}%"
            )
    if not found_any:
        print("INFO: No pairwise overlaps detected among checked ROIs.")


def prioritize_by_priority_list(
    priority_list: List[str], rois: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Resolve overlaps among ROIs according to priority_list (highest priority first).
    For any overlapping voxel between two ROIs, the ROI earlier in priority_list keeps the voxel,
    and the lower-priority ROI(s) have those voxels removed.

    Args:
      priority_list: ordered list of ROI names, e.g. ['PTV','ROI1','ROI2','ROI3','ROI4','ROI5']
      rois: dict mapping roi name -> boolean numpy array (may contain more names; only those in priority_list are processed)

    Returns:
      updated_rois: dict mapping the processed ROI names -> boolean numpy arrays (cropped to common shape if necessary)
    """
    # validate presence and normalize shapes
    normalized, common_shape = _align_and_validate_shapes(rois, priority_list)

    # Start with a copy so we don't alter original arrays unexpectedly
    updated = {name: normalized[name].copy() for name in priority_list}

    # We'll iterate in priority order: high -> low.
    # For each higher-priority ROI, remove its voxels from all lower-priority ROIs.
    for i, high_name in enumerate(priority_list):
        high_mask = updated[high_name].astype(bool)
        if not np.any(high_mask):
            # nothing to do if high_priority mask empty
            continue
        for low_name in priority_list[i + 1 :]:
            low_mask = updated[low_name].astype(bool)
            if not np.any(low_mask):
                continue
            # voxels to remove from low = intersection(high, low)
            intersect = np.logical_and(high_mask, low_mask)
            if np.any(intersect):
                removed = int(np.count_nonzero(intersect))
                size_low_before = int(np.count_nonzero(low_mask))
                updated_low = np.logical_and(low_mask, np.logical_not(high_mask))
                size_low_after = int(np.count_nonzero(updated_low))
                pct_removed_of_low = (
                    (removed / size_low_before * 100.0) if size_low_before > 0 else 0.0
                )
                print(
                    f"INFO: Priority resolution: '{high_name}' > '{low_name}' | removed {removed} voxels "
                    f"from {low_name} ({size_low_before} -> {size_low_after}, removed {pct_removed_of_low:.2f}% of original {low_name})"
                )
                updated[low_name] = updated_low.astype(np.bool8)

    # Return updated masks (kept as bool8)
    return {k: v.astype(np.bool8) for k, v in updated.items()}


def make_square(arr, pad_value=0):
    # arr.shape ==> (y, x, ...)
    y, x = arr.shape[:2]
    size = max(y, x)

    pad_y = size - y
    pad_x = size - x

    pad_before_y = pad_y // 2
    pad_after_y = pad_y - pad_before_y
    pad_before_x = pad_x // 2
    pad_after_x = pad_x - pad_before_x

    # Build pad widths: (for each axis) -> (before, after)
    pad_widths = [
        (pad_before_y, pad_after_y),  # y axis
        (pad_before_x, pad_after_x),  # x axis
    ]

    # For remaining axes, no padding
    for _ in range(arr.ndim - 2):
        pad_widths.append((0, 0))

    padded = np.pad(arr, pad_widths, mode="constant", constant_values=pad_value)
    return padded


def process_patient(folder_path, prefix):
    print(f">> processing {folder_path}")
    path = Path(folder_path)
    patient_name = path.name
    csv_path = os.path.join(
        Path(folder_path).parent.parent, "patGeometryInformation_" + prefix + "Part.csv"
    )
    sCT_path = os.path.join(folder_path, "sCT")
    patient_info = pd.read_csv(csv_path, delimiter=";")
    patient_info = patient_info[patient_info.iloc[:, 0] == patient_name]
    spacing = patient_info["sCTVoxelSizeOrig (mm)"]
    spacing = tuple(float(p) for p in str(spacing.values[0]).strip("()").split(","))
    target_shape = (256, 256, 96)

    patient_data = {}
    for file_name in os.listdir(sCT_path):
        if file_name not in struct_dict:
            continue
        struct_name = struct_dict[file_name]
        data = nib.load(os.path.join(sCT_path, file_name)).get_fdata()

        order = 2
        padding_value = -1000.0
        if file_name.startswith("mask"):
            order = 0
            padding_value = 0.0

        data = make_square(data, padding_value)
        data = scipy.ndimage.zoom(
            data,
            (
                target_shape[0] / data.shape[0],
                target_shape[1] / data.shape[1],
                target_shape[2] / data.shape[2],
            ),
            order=order,
            mode="constant",
            cval=padding_value,
        )

        if struct_name == "dose":
            data = np.interp(np.clip(data, 0.0, 50.0), (0.0, 50.0), (0, 255)).astype(
                np.uint8
            )
        elif struct_name == "CT":
            data = np.interp(
                np.clip(data, -1000.0, 1000.0), (-1000.0, 1000.0), (0, 255)
            ).astype(np.uint8)
        else:
            # masks -> boolean
            data = data.astype(np.bool8)
        data = np.transpose(data, (1, 0, 2))
        patient_data[struct_name] = data

    # Ensure we have all expected structures (including ROI7 from file)
    required_structs = {
        "dose",
        "CT",
        "PTV",
        "ROI1",
        "ROI2",
        "ROI3",
        "ROI4",
        "ROI5",
        "ROI7",
    }
    if not required_structs.issubset(set(patient_data.keys())):
        # missing at least one required struct -> skip
        return

    # 1) Print warnings if any pair among ROI1..ROI5 overlaps (does not raise)
    print("     > check_pairwise_rois_no_overlap")
    check_pairwise_rois_no_overlap(
        patient_data, roi_names=["ROI1", "ROI2", "ROI3", "ROI4", "ROI5"]
    )

    print("     > prioritize_by_priority_list")
    # 2) Enforce priority ordering (PTV first), resolving overlaps accordingly
    priority_order = ["PTV", "ROI1", "ROI2", "ROI3", "ROI4", "ROI5"]
    updated = prioritize_by_priority_list(priority_order, patient_data)
    for k, v in updated.items():
        patient_data[k] = v

    # Make ROI6 same shape as PTV, set all voxels to 1, then zero-out the union of PTV and ROI1..ROI5
    print("     > make ROI6")
    roi_shape = patient_data["PTV"].shape
    roi6 = np.ones(roi_shape, dtype=np.bool8)
    union_masks = (
        patient_data["PTV"]
        | patient_data["ROI1"]
        | patient_data["ROI2"]
        | patient_data["ROI3"]
        | patient_data["ROI4"]
        | patient_data["ROI5"]
    )
    roi6[union_masks] = False
    patient_data["ROI6"] = roi6

    # --- Consistency check: masks should be mutually exclusive and fully cover space ---
    mask_sum = (
        patient_data["PTV"].astype(np.uint8)
        + patient_data["ROI1"].astype(np.uint8)
        + patient_data["ROI2"].astype(np.uint8)
        + patient_data["ROI3"].astype(np.uint8)
        + patient_data["ROI4"].astype(np.uint8)
        + patient_data["ROI5"].astype(np.uint8)
        + patient_data["ROI6"].astype(np.uint8)
    )

    assert np.all(
        mask_sum == 1
    ), f"Mask sum check failed: min={mask_sum.min()}, max={mask_sum.max()}"

    # Randomly assign to validation with 20% chance
    out_prefix = prefix
    if np.random.rand() < 0.2:
        out_prefix = "val_" + prefix

    # Save -- include ROI6
    np.savez(
        save_path + out_prefix + "_" + patient_name + ".npz",
        **{f"{key}": value for key, value in patient_data.items()},
    )


# save_path = "database/lund-probe-processed/"
# path_utils.make_dir(save_path)
# base_path = "database/lund-probe/"
# base_part = base_path + "basePart"
# extended_part = base_path + "extendedPart"


# for path in os.listdir(base_part):
#     process_patient(os.path.join(base_part, path), "base")

# for path in os.listdir(extended_part):
#     process_patient(os.path.join(extended_part, path), "extended")


if __name__ == "__main__":
    save_path = "database/lund-probe-processed/"
    path_utils.make_dir(save_path)

    base_path = "database/lund-probe/"
    base_part = base_path + "basePart"
    extended_part = base_path + "extendedPart"

    # build tasks list: tuples of (folder_path, prefix)
    tasks = []
    for p in os.listdir(base_part):
        tasks.append((os.path.join(base_part, p), "base"))
    for p in os.listdir(extended_part):
        tasks.append((os.path.join(extended_part, p), "extended"))

    # choose number of worker processes (leave one core free by default)
    n_cpus = os.cpu_count() or 2
    n_workers = max(1, n_cpus - 1)

    print(f"Starting multiprocessing with {n_workers} workers, {len(tasks)} tasks...")

    # run in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        future_to_task = {
            exe.submit(process_patient, folder, prefix): (folder, prefix)
            for folder, prefix in tasks
        }

        for fut in as_completed(future_to_task):
            folder, prefix = future_to_task[fut]
            try:
                fut.result()
            except Exception as e:
                # Print the patient and the exception. You can adapt this to write to a log file.
                print(f"ERROR processing {folder} ({prefix}): {repr(e)}")
            else:
                print(f"DONE processing {folder} ({prefix})")
