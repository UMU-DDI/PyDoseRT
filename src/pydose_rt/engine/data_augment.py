import os
import itertools
import time, random
import numpy as np
import torch
import torch.nn.functional as F
import copy
import json
import scipy.ndimage
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter
from pydose_rt.engine.config import config as PARAMS

# Optional: import torchio only if user passes transform; safe to import here
import torchio as tio


# -------------------------
# Utility functions (adapted for PyTorch)
# -------------------------


def conv3d_pytorch_same_padding(input_volume_np, kernel_np):
    """
    Performs 3D convolution of a 3D input volume with a 3D kernel using PyTorch,
    with 'same' padding to ensure the output volume has the same shape as the input volume.

    Args:
        input_volume_np (numpy.ndarray): A 3D NumPy array representing the input volume (shape: (depth, height, width)).
        kernel_np (numpy.ndarray): A 3D NumPy array representing the convolution kernel (shape: (kernel_depth, kernel_height, kernel_width)).

    Returns:
        numpy.ndarray: A 3D NumPy array representing the convolved output volume with 'same' padding (shape: same as input_volume_np).
    """
    # Convert NumPy arrays to PyTorch tensors
    # PyTorch conv3d expects input: (N, C_in, D_in, H_in, W_in)
    # PyTorch conv3d expects kernel: (C_out, C_in, D_k, H_k, W_k)

    # Add batch and channel dimensions to input (1, 1, D, H, W)
    input_tensor = torch.from_numpy(input_volume_np).float().unsqueeze(0).unsqueeze(0)

    # Add input and output channel dimensions to kernel (1, 1, D_k, H_k, W_k)
    kernel_tensor = torch.from_numpy(kernel_np).float().unsqueeze(0).unsqueeze(0)

    # Get dimensions
    input_depth, input_height, input_width = input_volume_np.shape
    kernel_depth, kernel_height, kernel_width = kernel_np.shape

    # Calculate padding for 'same' convolution
    # Pytorch's padding is for (W, H, D)
    pad_d = kernel_depth // 2
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    output_tensor = F.conv3d(
        input=input_tensor,
        weight=kernel_tensor,
        padding=(pad_d, pad_h, pad_w),  # PyTorch padding is (D, H, W)
    )

    # Remove batch and channel dimensions and convert back to NumPy array
    # output_tensor shape will be (1, 1, D, H, W)
    output_volume = output_tensor.squeeze().numpy()

    return output_volume


def fix_roi6_randomize_weights_and_swap_rows(constraints):  #
    """
    Fixes ROI6 values, randomizes weights with specific ranges, and swaps the entire row of constraint values
    between other keys using index shuffling.

    Args:
        constraints (dict): The dictionary containing the constraints.

    Returns:
        dict: The modified constraints dictionary.

    the loss appears to be unstable if weight is random
    """
    rois_to_swap = [roi for roi in constraints["weight"].keys() if roi != "ROI6"]
    constraint_types_to_swap = [
        "lower_bound_gy",
        "higher_bound_gy",
        "lower_bound_target_percent",
        "higher_bound_target_percent",
        "weight",
    ]

    # Randomize weights (excluding ROI6) with specific ranges
    for roi in rois_to_swap:
        if roi == "PTV":
            constraints["weight"][roi] = random.randint(10, 1000)
        else:
            constraints["weight"][roi] = random.randint(1, 10)

    # Swap the whole row of values between ROIs (excluding ROI6) using index shuffling
    num_rois_to_swap = len(rois_to_swap)
    indices = list(range(num_rois_to_swap))
    random.shuffle(indices)

    for i in range(num_rois_to_swap):
        # Get the original ROI name using the current index
        roi1 = rois_to_swap[i]
        # Get the ROI name to swap with using the shuffled index
        roi2_index = indices[i]
        roi2 = rois_to_swap[roi2_index]

        # Perform the swap for all constraint types
        for constraint_type in constraint_types_to_swap:
            temp_value = constraints[constraint_type][roi1]
            constraints[constraint_type][roi1] = constraints[constraint_type][roi2]
            constraints[constraint_type][roi2] = temp_value

    return constraints


def fix_roi6_and_swap_rows(constraints):  #
    """
    Fixes ROI6 values, randomizes weights with specific ranges, and swaps the entire row of constraint values
    between other keys using index shuffling.

    Args:
        constraints (dict): The dictionary containing the constraints.

    Returns:
        dict: The modified constraints dictionary.
    """
    rois_to_swap = [roi for roi in constraints["weight"].keys() if roi != "ROI6"]
    constraint_types_to_swap = [
        "lower_bound_gy",
        "higher_bound_gy",
        "lower_bound_target_percent",
        "higher_bound_target_percent",
        "weight",
    ]

    # Swap the whole row of values between ROIs (excluding ROI6) using index shuffling
    num_rois_to_swap = len(rois_to_swap)
    indices = list(range(num_rois_to_swap))
    random.shuffle(indices)

    for i in range(num_rois_to_swap):
        # Get the original ROI name using the current index
        roi1 = rois_to_swap[i]
        # Get the ROI name to swap with using the shuffled index
        roi2_index = indices[i]
        roi2 = rois_to_swap[roi2_index]

        # Perform the swap for all constraint types
        for constraint_type in constraint_types_to_swap:
            temp_value = constraints[constraint_type][roi1]
            constraints[constraint_type][roi1] = constraints[constraint_type][roi2]
            constraints[constraint_type][roi2] = temp_value

    return constraints


def randomize_weights(constraints):  #
    """
    Creates a new dictionary with the same structure as the input constraints,
    but with randomized weight values (between 1 and 100).

    Args:
        constraints (dict): The original constraints dictionary.

    Returns:
        dict: A new dictionary with randomized weights.
    """
    new_constraints = copy.deepcopy(constraints)
    for roi in new_constraints["weight"]:
        new_constraints["weight"][roi] = random.uniform(0.01, 1.0)
    return new_constraints


def normalize_weights(constraints, sum_value=100):  #
    """
    Normalizes the values in the 'weight' sub-dictionary of the constraints
    so that their sum is 100.

    Args:
        constraints (dict): The constraints dictionary containing the 'weight' key.

    Returns:
        dict: The modified constraints dictionary with normalized weights.
    """
    weights = constraints.get("weight")
    if not weights:
        return constraints  # Return original if 'weight' key is missing

    total_weight = sum(weights.values())
    if total_weight == 0:
        total_weight = 1e-6

    normalized_weights = {}
    for roi, weight in weights.items():
        normalized_weights[roi] = (weight / total_weight) * sum_value

    constraints["weight"] = normalized_weights
    return constraints

def prune_patients(patient_list):
    pruned_list = []
    for patient in patient_list:
        if not os.path.isdir(patient):
            continue

        if (("CT.npy" in os.listdir(patient)) and ("StructureSet.npy" in os.listdir(patient))):
            pruned_list.append(patient)
    return pruned_list

class DataGenerator(Dataset):
    """
    Torch Dataset replacement for your previous Keras Sequence.
    - Supports optional TorchIO transforms (passed via `transform` argument).
    - Uses torch.interpolate for CT downsampling (trilinear) and masks (nearest).
    """

    def __init__(
        self,
        data_path,
        cohort,
        transform=None,  # <--- new: pass a torchio.Compose(...) if you want augmentation
        transform_mask=None,
        shuffle=False,
        batch_size=1,
        synthetic_dose=False,
        downsampling_factor=1,
        is_debug=0,
        constraints=PARAMS.constraints,
        is_return_gaussian_ptv_oars=False,
        weight_ptv=None,
        constraint_mode="fixed",
        is_normalize_weight=False,
        verbose=False,
        is_return_file_name=False,
    ):
        self.synthetic_dose = synthetic_dose
        self.data_path = data_path
        self.downsampling_factor = downsampling_factor
        self.transform = transform  # torchio transform or None
        self.transform_mask = transform_mask  # torchio transform or None

        self.patient_list = prune_patients([os.path.join(data_path, name) for name in os.listdir(data_path)])
        self.patient_list.sort()
        self.cohort = cohort
        if cohort == "training":
            self.patient_list = [
                file for file in self.patient_list if (not (file.startswith("val_")))
            ]
            if is_debug == 1:
                self.patient_list = self.patient_list[:10]
            elif is_debug == 2:
                self.patient_list = self.patient_list[:1]
            else:
                pass
        elif cohort == "validating":
            self.patient_list = [
                file for file in self.patient_list if (file.startswith("val_"))
            ]
            if is_debug == 1:
                self.patient_list = self.patient_list[:10]
            elif is_debug == 2:
                self.patient_list = self.patient_list[:1]
            else:
                pass
        elif cohort == "plotting":
            self.patient_list = [
                file for file in self.patient_list if (file.startswith("val_"))
            ][0:1]
        elif cohort == "all":
            pass

        self.indexes = np.arange(len(self.patient_list))
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.is_return_gaussian_ptv_oars = is_return_gaussian_ptv_oars
        self.constraints_template = copy.deepcopy(constraints)
        self.weight_ptv = weight_ptv
        self.constraint_mode = constraint_mode
        self.is_normalize_weight = is_normalize_weight
        self.verbose = verbose
        self.is_return_file_name = is_return_file_name

        self.epoch_number = 0
        self.gaussian_kernel_3d = self.gaussian_kernel_3d(size=9, sigma=1.5)

        print(f"Number of files: {len(self.patient_list)} in {cohort} cohort")

    def gaussian_kernel_3d(self, size=9, sigma=1.5):
        ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
        xx, yy, zz = np.meshgrid(ax, ax, ax)
        kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
        return kernel / np.sum(kernel)

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, index):
        file_ID = self.patient_list[self.indexes[index]]
        return self._data_generation_single(file_ID)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def set_epoch(self, epoch):
        self.epoch_number = epoch

    def create_bound_weight_matrix(self, structures, bound, factor=1):
        first_structure = next(iter(structures.values()))
        bound_matrix = np.zeros_like(first_structure, dtype=np.float32)
        for structure_id, array in structures.items():
            if structure_id in bound:
                bound_matrix += array * bound[structure_id]
        return bound_matrix / factor

    def downsample_ct(self, ct_data, downsample_factor=(1, 1, 1), mode="trilinear"):
        """
        Downsample CT using torch.interpolate (trilinear).
        ct_data: numpy array shape [W, D, H] (same as your current code)
        returns: numpy array downsampled (same axis order)
        """
        if downsample_factor == (1, 1, 1):
            return ct_data

        # convert to torch tensor: shape (1, 1, W, D, H)
        x = torch.from_numpy(ct_data.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        zoom_factors = tuple(1 / x for x in downsample_factor)
        # interpolate expects (N,C,D,H,W) — we keep the same order (W,D,H) as "D,H,W"
        x_ds = F.interpolate(
            x,
            scale_factor=zoom_factors,
            mode=mode,
            align_corners=False,
        )
        return x_ds.squeeze().cpu().numpy()

    def downsample_masks(self, mask_list, downsample_factor=2):
        """
        mask_list: list of numpy arrays each shape [W,D,H] (binary)
        returns stacked numpy array shape (C_mask, W', D', H')
        Uses nearest interpolation to preserve binary labels.
        """
        if downsample_factor == (1, 1, 1):
            return np.stack(mask_list, axis=0).astype(np.float32)

        masks = np.stack(mask_list, axis=0).astype(np.float32)  # (C, W, D, H)
        # convert to torch (1, C, W, D, H)
        t = torch.from_numpy(masks).unsqueeze(0)
        zoom_factors = tuple(1 / x for x in downsample_factor)
        t_ds = F.interpolate(
            t,
            scale_factor=zoom_factors,
            mode="nearest",
        )
        return t_ds.squeeze(0).cpu().numpy()

    def _apply_single_mask_transform(self, mask, ct_shape, max_tries=10):
        """
        Apply per-mask transform around the mask centroid:
        1) shift mask so centroid -> image center,
        2) apply TorchIO RandomAffine (center='image'),
        3) inverse-shift back to original coordinates,
        4) ensure mask lies inside ct_shape (retry / clip fallback).

        mask: np.ndarray, shape (D,H,W) or (W,D,H) — use same ordering as ct_shape.
        ct_shape: tuple of ints (same shape as mask)
        """
        tries = 0
        orig_mask = (mask > 0).astype(np.uint8)

        while True:
            tries += 1

            # find centroid (in voxel coords)
            nz = np.argwhere(orig_mask)
            if nz.size == 0:
                # empty mask stays empty
                return np.zeros(ct_shape, dtype=np.uint8)

            centroid = nz.mean(axis=0)  # float centroid (z,y,x)
            shape = np.array(orig_mask.shape)
            image_center = shape / 2.0
            # shift needed to move centroid -> image center
            shift = image_center - centroid  # can be fractional

            # 1) shift mask to center (no-wrap; pad with 0)
            mask_centered = scipy.ndimage.shift(
                orig_mask.astype(np.float32),
                shift=shift,
                order=0,
                mode="constant",
                cval=0.0,
            )

            # build TorchIO subject with centered mask (channel-first)
            mask_t = torch.from_numpy(mask_centered[np.newaxis].astype(np.float32))
            subject = tio.Subject(mask=tio.LabelMap(tensor=mask_t))

            # --- define per-mask transform ---
            transform = tio.Compose(
                [
                    # tio.RandomFlip(axes=(0,), p=0.5),  # flip left/right
                    # tio.RandomAffine(
                    #     scales=(0.90, 1.10),
                    #     # degrees=(0, 0, 360),  # free rotation
                    #     # translation=(0.05, 0.10),  # 20–35% shift (fraction of size)
                    #     center="image",  # now effectively mask center
                    #     image_interpolation="nearest",
                    #     p=1.0,
                    # )
                    tio.RandomAffine(
                        scales=(0.85, 1.25),  # zoom in/out 15–25%
                        degrees=(5, 5, 5),  # small rotations
                        translation=(0.05, 0.1),  # 10–20% shift along each axis
                        center="image",  # use image center
                    ),
                ]
            )

            # 2) apply transform
            out_subject = transform(subject)
            out_mask_centered = (
                out_subject["mask"].data.cpu().numpy().squeeze(0)
            )  # same shape as mask_centered

            # 3) inverse shift back to original coords
            inv_shift = -shift
            out_mask = scipy.ndimage.shift(
                (out_mask_centered >= 0.5).astype(np.uint8),
                shift=inv_shift,
                order=0,
                mode="constant",
                cval=0,
            )

            # 4) binarize & check bbox in bounds
            out_mask_bin = (out_mask > 0).astype(np.uint8)
            nz2 = np.argwhere(out_mask_bin)
            if nz2.size == 0:
                return np.zeros(ct_shape, dtype=np.uint8)

            mins = nz2.min(axis=0)
            maxs = nz2.max(axis=0)

            # in-bounds check
            if (mins >= 0).all() and (maxs < np.array(ct_shape)).all():
                return out_mask_bin

            # retry until max_tries
            if tries >= max_tries:
                # fallback: clip bounding box to be inside by cropping/pasting
                clipped = np.zeros(ct_shape, dtype=np.uint8)
                src_min = np.maximum(mins, 0)
                src_max = np.minimum(maxs, np.array(ct_shape) - 1)
                src_slices = tuple(slice(src_min[d], src_max[d] + 1) for d in range(3))
                dst_slices = tuple(slice(src_min[d], src_max[d] + 1) for d in range(3))
                try:
                    clipped[dst_slices] = out_mask_bin[src_slices]
                except Exception:
                    clipped = out_mask_bin[: ct_shape[0], : ct_shape[1], : ct_shape[2]]
                return clipped

    def _apply_ct_transform(self, ct_np, scale_range=(0.75, 1.25)):
        """
        Apply random flip + affine (scale only/optional small rotation) to CT using TorchIO.
        Returns transformed CT as numpy array same shape.
        """
        ct_t = torch.from_numpy(ct_np[np.newaxis].astype(np.float32))  # (1, W, D, H)
        subject = tio.Subject(img=tio.ScalarImage(tensor=ct_t))

        transform = tio.Compose(
            [
                tio.RandomFlip(axes=(0,), p=0.5),
                tio.RandomAffine(
                    scales=scale_range,
                    degrees=(
                        0,
                        10,
                    ),  # keep CT rotation mild; masks rotate freely per requirement
                    translation=(0, 0, 0),
                    image_interpolation="linear",
                    p=1.0,
                ),
            ]
        )
        out_subject = transform(subject)
        out_ct = out_subject["img"].data.cpu().numpy().squeeze(0)
        return out_ct.astype(np.float32)

    def _clean_augmented_masks(
        self, augmented_masks, hole_size_threshold=8, do_close=True
    ):
        """
        Clean post-augmented masks and guarantee a proper partition.
        Input: augmented_masks: dict with keys "PTV","ROI1","ROI2","ROI3","ROI4","ROI5"
            values are numpy arrays (any dtype) same shape.
        Returns: resolved: dict with keys "PTV".."ROI6" each a uint8 array,
                and guaranteed mask_sum == 1 everywhere.
        Steps:
        - Binarize (>=0.5)
        - Resolve overlaps by priority PTV > ROI1 > ROI2 > ...
        - Optional morphological closing and removal of very small speckles
        - Recompute ROI6 = complement of union
        - Final sweep to enforce exclusivity if needed
        """
        keys = ["PTV", "ROI1", "ROI2", "ROI3", "ROI4", "ROI5"]
        shape = next(iter(augmented_masks.values())).shape

        # 1) Binarize strictly
        bin_masks = {}
        for k in keys:
            bin_masks[k] = (augmented_masks[k] >= 0.5).astype(np.uint8)

        # 2) Resolve overlaps by priority (PTV highest)
        assigned = np.zeros(shape, dtype=np.uint8)
        resolved = {}
        for k in keys:
            m = bin_masks[k].copy()
            m = m * (1 - assigned)  # remove voxels already assigned
            resolved[k] = m
            assigned = np.clip(assigned + m, 0, 1)

        # 3) Optional morphological closing + remove tiny speckles
        if do_close:
            structure = np.ones((3, 3, 3), dtype=np.uint8)
            for k in keys:
                m = resolved[k]
                if m.sum() == 0:
                    continue
                # morphological closing to fill small holes
                m_closed = ndimage.binary_closing(m, structure=structure).astype(
                    np.uint8
                )

                # remove tiny connected components (noise)
                labeled, ncomp = ndimage.label(m_closed)
                if ncomp == 0:
                    resolved[k] = m_closed
                    continue
                comp_sizes = ndimage.sum(
                    m_closed, labeled, index=np.arange(1, ncomp + 1)
                )
                # keep only components larger than threshold
                keep_comps = np.where(comp_sizes > hole_size_threshold)[0] + 1
                if keep_comps.size == 0:
                    # nothing survived -> empty mask
                    resolved[k] = np.zeros_like(m_closed, dtype=np.uint8)
                else:
                    keep_mask = np.isin(labeled, keep_comps)
                    resolved[k] = keep_mask.astype(np.uint8)

            # recompute assigned and ensure exclusivity again
            assigned = np.zeros(shape, dtype=np.uint8)
            for k in keys:
                m = resolved[k]
                m = m * (1 - assigned)
                resolved[k] = m
                assigned = np.clip(assigned + m, 0, 1)

        # 4) background = complement of assigned
        resolved["ROI6"] = (assigned == 0).astype(np.uint8)

        # 5) final safety sweep: enforce strict partition if something odd remains
        all_stack = np.stack(
            [
                resolved[k]
                for k in ["PTV", "ROI1", "ROI2", "ROI3", "ROI4", "ROI5", "ROI6"]
            ],
            axis=0,
        )
        mask_sum = np.sum(all_stack, axis=0)
        if not np.all(mask_sum == 1):
            # force exclusivity by priority order
            final = {}
            assigned = np.zeros(shape, dtype=np.uint8)
            for k in ["PTV", "ROI1", "ROI2", "ROI3", "ROI4", "ROI5"]:
                m = (resolved[k] > 0).astype(np.uint8)
                m = m * (1 - assigned)
                final[k] = m
                assigned = np.clip(assigned + m, 0, 1)
            final["ROI6"] = (assigned == 0).astype(np.uint8)
            resolved = final

        # sanity check (should pass)
        all_stack = np.stack(
            [
                resolved[k]
                for k in ["PTV", "ROI1", "ROI2", "ROI3", "ROI4", "ROI5", "ROI6"]
            ],
            axis=0,
        )
        mask_sum = np.sum(all_stack, axis=0)
        # assert np.all(
        #     mask_sum == 1
        # ), f"Final mask sum invalid: min {mask_sum.min()} max {mask_sum.max()}"

        # cast to uint8 and return
        for k in resolved:
            resolved[k] = resolved[k].astype(np.uint8)

        return resolved

    def _data_generation_single(self, file_ID):

        # constraints logic (unchanged)
        epoch = self.epoch_number
        p_augment = min(epoch * 0.1, 1.0)
        if self.constraint_mode == "fixed":
            constraints = copy.deepcopy(self.constraints_template)
        elif self.constraint_mode == "rconstraint":
            if random.random() < p_augment:
                constraints = copy.deepcopy(random.choice(PARAMS.list_constraints))
            else:
                constraints = copy.deepcopy(self.constraints_template)
            if self.weight_ptv is not None:
                for key, value in constraints["weight"].items():
                    if value == 10:
                        constraints["weight"][key] = self.weight_ptv
        elif self.constraint_mode == "rweight":
            if random.random() < p_augment:
                constraints = randomize_weights(
                    copy.deepcopy(self.constraints_template)
                )
            else:
                constraints = copy.deepcopy(self.constraints_template)

        if self.is_normalize_weight:
            constraints = normalize_weights(constraints)

        constraints["weight"]["ROI1"] = 10**np.random.randint(0, 2)
        constraints["weight"]["ROI2"] = 10**np.random.randint(0, 3)
        constraints["weight"]["ROI3"] = 10**np.random.randint(0, 3)
        constraints["weight"]["ROI4"] = 10**np.random.randint(0, 3)
        constraints["weight"]["ROI5"] = 10**np.random.randint(0, 3)
        constraints["weight"]["ROI6"] = 10**np.random.randint(0, 3)

        ct = None       
        while ct is None:
            try:
                ct = np.transpose((np.load(os.path.join(file_ID, "CT.npy"), allow_pickle=True) / 1000), (1, 2, 0))
            except Exception:
                time.sleep(random.random())

        ptv = None
        while ptv is None:
            try:
                structs = np.load(os.path.join(file_ID, "StructureSet.npy"), allow_pickle=True).astype(np.float32)
                ptv = np.transpose(structs[0, ...], (1, 2, 0))
                oar_penile = np.clip(np.transpose(structs[1, ...], (1, 2, 0)) - (ptv), 0, 1)
                oar_fem_l = np.clip(np.transpose(structs[2, ...], (1, 2, 0)) - (ptv + oar_penile), 0, 1)
                oar_fem_r = np.clip(np.transpose(structs[3, ...], (1, 2, 0)) - (ptv + oar_penile + oar_fem_l), 0, 1)
                oar_bladder = np.clip(np.transpose(structs[4, ...], (1, 2, 0)) - (ptv + oar_penile + oar_fem_l + oar_fem_r), 0, 1)
                oar_rectum = np.clip(np.transpose(structs[5, ...], (1, 2, 0)) - (ptv + oar_penile + oar_fem_l + oar_fem_r + oar_bladder), 0, 1)
                background = np.clip(np.transpose(structs[6, ...], (1, 2, 0)) - (ptv + oar_penile + oar_fem_l + oar_fem_r + oar_bladder + oar_rectum), 0, 1)
            except Exception:
                time.sleep(random.random())

        # Dose (optional)
        dose = None
        if (os.path.exists(os.path.join(file_ID, "Dose.npy"))):
            while dose is None:
                try:
                        dose = np.transpose(np.load(os.path.join(file_ID, "Dose.npy"), allow_pickle=True), (1, 2, 0))
                except Exception:
                    time.sleep(random.random())
        else:
            dose = np.zeros_like(ct)
        dose_tensor = torch.from_numpy(dose)
        # if "Dose" in npzdata:
        #     dose = npzdata["Dose"]
        # if "dose" in npzdata:
        #     dose = 50 * (npzdata["dose"] / 255.0)
        # if dose.size == 1:
        #     dose = np.zeros_like(ct)
        # dose_tensor = torch.from_numpy(dose.astype(np.float32))

        # Downsample if requested (CT: trilinear, masks: nearest)
        if self.downsampling_factor and self.downsampling_factor != (1, 1, 1):
            factor = self.downsampling_factor
            ct = self.downsample_ct(ct, factor)
            (
                ptv,
                oar_penile,
                oar_fem_l,
                oar_fem_r,
                oar_bladder,
                oar_rectum,
                background,
            ) = self.downsample_masks(
                [
                    ptv,
                    oar_penile,
                    oar_fem_l,
                    oar_fem_r,
                    oar_bladder,
                    oar_rectum,
                    background,
                ],
                factor,
            )

            # Ensure binary masks remain 0/1
            for mask_ in [
                ptv,
                oar_penile,
                oar_fem_l,
                oar_fem_r,
                oar_bladder,
                oar_rectum,
                background,
            ]:
                assert np.all(
                    np.logical_or(mask_ == 0, mask_ == 1)
                ), "Mask contains non-binary values"

            masks = [
                ptv,
                oar_penile,
                oar_fem_l,
                oar_fem_r,
                oar_bladder,
                oar_rectum,
                background,
            ]
            mask_sum = np.sum(masks, axis=0)
            # assert np.all(
            #     mask_sum == 1
            # ), f"Mask sum is not 1 everywhere: min={mask_sum.min()}, max={mask_sum.max()}"

        # gaussian ptv/oar (optional)
        gaussian_ptv_oar = None
        if self.is_return_gaussian_ptv_oars:
            gaussian_ptv_oar = conv3d_pytorch_same_padding(
                input_volume_np=ptv
                + oar_penile
                + oar_fem_l
                + oar_fem_r
                + oar_bladder
                + oar_rectum,
                kernel_np=self.gaussian_kernel_3d,
            )

        # --- AUGMENTATION: MUST HAPPEN BEFORE generating weight maps ---
        # If no augmentation is required, set transform=None and skip
        if self.transform is None:
            # No augmentation -> proceed to generate bound maps using original masks
            pass
        else:
            """
            # --- 1. Global augmentation (ct + masks together) ---
            subject = tio.Subject(
                ct=tio.ScalarImage(tensor=ct[np.newaxis, ...]),  # add channel dim
                ptv=tio.LabelMap(tensor=ptv[np.newaxis, ...]),
                oar_penile=tio.LabelMap(tensor=oar_penile[np.newaxis, ...]),
                oar_fem_l=tio.LabelMap(tensor=oar_fem_l[np.newaxis, ...]),
                oar_fem_r=tio.LabelMap(tensor=oar_fem_r[np.newaxis, ...]),
                oar_bladder=tio.LabelMap(tensor=oar_bladder[np.newaxis, ...]),
                oar_rectum=tio.LabelMap(tensor=oar_rectum[np.newaxis, ...]),
                background=tio.LabelMap(tensor=background[np.newaxis, ...]),
            )

            global_transform = self.transform

            subject = global_transform(subject)

            ct = subject.ct.data[0].numpy()
            ptv = subject.ptv.data[0].numpy()
            oar_penile = subject.oar_penile.data[0].numpy()
            oar_fem_l = subject.oar_fem_l.data[0].numpy()
            oar_fem_r = subject.oar_fem_r.data[0].numpy()
            oar_bladder = subject.oar_bladder.data[0].numpy()
            oar_rectum = subject.oar_rectum.data[0].numpy()
            background = subject.background.data[0].numpy()

            # --- 2. Per-mask augmentation (applied independently) ---
            W, D, H = ct.shape  # volume shape

            # Prepare masks dict (binary)
            masks_dict = {
                "PTV": (ptv > 0.5).astype(np.uint8),
                "ROI1": (oar_penile > 0.5).astype(np.uint8),
                "ROI2": (oar_fem_l > 0.5).astype(np.uint8),
                "ROI3": (oar_fem_r > 0.5).astype(np.uint8),
                "ROI4": (oar_bladder > 0.5).astype(np.uint8),
                "ROI5": (oar_rectum > 0.5).astype(np.uint8),
                "ROI6": (background > 0.5).astype(np.uint8),
            }

            # For each structure, compute bbox size (voxels) for shift magnitude
            augmented_masks = {}
            for name, mask_np in masks_dict.items():
                if self.transform_mask is not None:
                    # compute bbox size
                    nz = np.argwhere(mask_np)
                    if nz.size == 0:
                        # empty mask remains empty
                        augmented_masks[name] = mask_np
                        continue
                    mins = nz.min(axis=0)
                    maxs = nz.max(axis=0)
                    bbox_size = (maxs - mins + 1).astype(float)  # voxels per axis

                    # determine translation limit in voxels:
                    if name == "PTV" or name.startswith("ROI"):
                        # shift for PTV and OARs: 20-35% of structure's own size
                        shift_frac = np.random.uniform(0.20, 0.35)
                    else:
                        shift_frac = 0.0
                    translate_max_vox = tuple((bbox_size * shift_frac).tolist())

                    # For masks we allow free rotation; call helper
                    aug_mask = self._apply_single_mask_transform(
                        mask_np.astype(np.float32),
                        ct_shape=(W, D, H),
                        # translate_max_vox=translate_max_vox,
                        # scale_range=(0.75, 1.25),
                        max_tries=5,
                    )
                    augmented_masks[name] = aug_mask
                else:
                    augmented_masks[name] = mask_np

            # background = 1 if all others are 0
            union_mask = (
                augmented_masks["PTV"]
                | augmented_masks["ROI1"]
                | augmented_masks["ROI2"]
                | augmented_masks["ROI3"]
                | augmented_masks["ROI4"]
                | augmented_masks["ROI5"]
            )

            augmented_masks["ROI6"] = (union_mask == 0).astype(np.uint8)
            """

            # # After all masks transformed, ensure no overlaps between OARs and PTV:
            # ptv_aug = augmented_masks["PTV"].astype(np.uint8)
            # for key in ["PTV", "ROI1", "ROI2", "ROI3", "ROI4", "ROI5", "ROI6"]:
            #     oar = augmented_masks[key].astype(np.uint8)
            #     # Remove OAR voxels that overlap PTV
            #     oar = oar * (1 - ptv_aug)
            #     augmented_masks[key] = oar

            # Build a TorchIO Subject with CT and all masks
            ct_min = float(ct.min())
            subject = tio.Subject(
                ct=tio.ScalarImage(
                    tensor=ct[np.newaxis, ...], default_pad_value=ct_min
                ),
                PTV=tio.LabelMap(tensor=ptv[np.newaxis, ...], default_pad_value=0),
                ROI1=tio.LabelMap(
                    tensor=oar_penile[np.newaxis, ...], default_pad_value=0
                ),
                ROI2=tio.LabelMap(
                    tensor=oar_fem_l[np.newaxis, ...], default_pad_value=0
                ),
                ROI3=tio.LabelMap(
                    tensor=oar_fem_r[np.newaxis, ...], default_pad_value=0
                ),
                ROI4=tio.LabelMap(
                    tensor=oar_bladder[np.newaxis, ...], default_pad_value=0
                ),
                ROI5=tio.LabelMap(
                    tensor=oar_rectum[np.newaxis, ...], default_pad_value=0
                ),
            )

            # Define your transform (TorchIO will use linear for ct, nearest for masks)
            aug_subject = self.transform(subject)

            # Extract the 6 output masks from the augmented subject (numpy arrays)
            augmented_masks = {
                "PTV": aug_subject["PTV"].data[0].numpy(),
                "ROI1": aug_subject["ROI1"].data[0].numpy(),
                "ROI2": aug_subject["ROI2"].data[0].numpy(),
                "ROI3": aug_subject["ROI3"].data[0].numpy(),
                "ROI4": aug_subject["ROI4"].data[0].numpy(),
                "ROI5": aug_subject["ROI5"].data[0].numpy(),
            }

            # Clean & resolve overlaps / holes and compute ROI6
            resolved = self._clean_augmented_masks(
                augmented_masks, hole_size_threshold=8, do_close=True
            )

            # Replace original masks with cleaned ones (float32 for downstream)
            ptv = resolved["PTV"].astype(np.float32)
            oar_penile = resolved["ROI1"].astype(np.float32)
            oar_fem_l = resolved["ROI2"].astype(np.float32)
            oar_fem_r = resolved["ROI3"].astype(np.float32)
            oar_bladder = resolved["ROI4"].astype(np.float32)
            oar_rectum = resolved["ROI5"].astype(np.float32)
            background = resolved["ROI6"].astype(np.float32)

            # for k in ["PTV", "ROI1", "ROI2", "ROI3", "ROI4", "ROI5"]:
            #     m = aug_subject[k].data[0].numpy().astype(np.uint8)
            #     m = ndimage.binary_closing(m, structure=np.ones((3, 3, 3))).astype(
            #         np.uint8
            #     )
            #     aug_subject[k].set_data(torch.from_numpy(m[np.newaxis, ...]))

            all_masks = np.stack(
                [
                    ptv,
                    oar_penile,
                    oar_fem_l,
                    oar_fem_r,
                    oar_bladder,
                    oar_rectum,
                    background,
                ],
                axis=0,  # channel dimension
            )

            mask_sum = np.sum(all_masks, axis=0)
            # assert np.all(
            #     mask_sum == 1
            # ), f"Mask sum check failed: min={mask_sum.min()}, max={mask_sum.max()}"

            # # Replace original masks with augmented ones
            # ptv = augmented_masks["PTV"].astype(np.float32)
            # oar_penile = augmented_masks["ROI1"].astype(np.float32)
            # oar_fem_l = augmented_masks["ROI2"].astype(np.float32)
            # oar_fem_r = augmented_masks["ROI3"].astype(np.float32)
            # oar_bladder = augmented_masks["ROI4"].astype(np.float32)
            # oar_rectum = augmented_masks["ROI5"].astype(np.float32)
            # background = augmented_masks["ROI6"].astype(np.float32)

        # --- Now generate bound/weight matrices (AFTER augmentation) ---
        structures = {
            "PTV": ptv,
            "ROI1": oar_penile,
            "ROI2": oar_fem_l,
            "ROI3": oar_fem_r,
            "ROI4": oar_bladder,
            "ROI5": oar_rectum,
            "ROI6": background,
        }

        lower_bound_gy = self.create_bound_weight_matrix(
            structures, bound=constraints["lower_bound_gy"]
        )
        higher_bound_gy = self.create_bound_weight_matrix(
            structures, bound=constraints["higher_bound_gy"]
        )
        lower_bound_target_percent = self.create_bound_weight_matrix(
            structures, bound=constraints["lower_bound_target_percent"]
        )
        higher_bound_target_percent = self.create_bound_weight_matrix(
            structures, bound=constraints["higher_bound_target_percent"]
        )
        region_weight = self.create_bound_weight_matrix(
            structures, bound=constraints["weight"], factor=1
        )

        # Stack inputs (channel-first): expected shape -> (C, W, D, H)
        images = np.stack(
            [
                ct,
                lower_bound_gy,
                higher_bound_gy,
                lower_bound_target_percent,
                higher_bound_target_percent,
                region_weight,
            ],
            axis=0,
        )

        mask = np.stack(
            [
                ptv,
                oar_penile,
                oar_fem_l,
                oar_fem_r,
                oar_bladder,
                oar_rectum,
                background,
            ],
            axis=0,
        )

        mask_names = [
            "ptv",
            "oar_penile",
            "oar_fem_l",
            "oar_fem_r",
            "oar_bladder",
            "oar_rectum",
            "background",
        ]

        mask_sum = np.sum(mask, axis=0)

        # if not np.all(mask_sum == 1):
        #     print(
        #         f"[ERROR] Mask sum check failed: min={mask_sum.min()}, max={mask_sum.max()}"
        #     )

        #     overlap_found = False

        #     # --- Check pairwise overlaps ---
        #     print("\n[Pairwise overlaps]:")
        #     for i, j in itertools.combinations(range(len(mask_names)), 2):
        #         overlap = np.logical_and(mask[i], mask[j])
        #         n = np.count_nonzero(overlap)
        #         if n > 0:
        #             overlap_found = True
        #             print(f"  {mask_names[i]} + {mask_names[j]} -> {n} voxels overlap")

        #     # --- Check triple overlaps (optional but useful) ---
        #     print("\n[Triple overlaps]:")
        #     for i, j, k in itertools.combinations(range(len(mask_names)), 3):
        #         overlap = mask[i] & mask[j] & mask[k]
        #         n = np.count_nonzero(overlap)
        #         if n > 0:
        #             overlap_found = True
        #             print(
        #                 f"  {mask_names[i]} + {mask_names[j]} + {mask_names[k]} -> {n} voxels overlap"
        #             )

        #     if overlap_found:
        #         raise AssertionError("Mask overlaps detected between structures.")
        #     else:
        #         print("No pairwise or triple overlaps detected.")
        # else:
        #     print("All masks are disjoint (sum==1 everywhere).")

        # assert np.all(
        #     mask_sum == 1
        # ), f"Mask sum check failed: min={mask_sum.min()}, max={mask_sum.max()}"

        images_tensor = torch.from_numpy(images.astype(np.float32))
        masks_tensor = torch.from_numpy(mask.astype(np.float32))

        region_weights_tensor = torch.from_numpy(region_weight.astype(np.float32))

        # gaussian ptv oars
        if self.is_return_gaussian_ptv_oars:
            gaussian_ptv_oars_tensor = torch.from_numpy(
                gaussian_ptv_oar.astype(np.float32)
            )
            if self.is_return_file_name:
                return (
                    images_tensor,
                    dose_tensor,
                    masks_tensor,
                    region_weights_tensor,
                    constraints,
                    gaussian_ptv_oars_tensor,
                    file_ID,
                )
            else:
                return (
                    images_tensor,
                    dose_tensor,
                    masks_tensor,
                    region_weights_tensor,
                    constraints,
                    gaussian_ptv_oars_tensor,
                )
        else:
            if self.is_return_file_name:
                return (
                    images_tensor,
                    dose_tensor,
                    masks_tensor,
                    region_weights_tensor,
                    constraints,
                    file_ID,
                )
            else:
                return (
                    images_tensor,
                    dose_tensor,
                    masks_tensor,
                    region_weights_tensor,
                    constraints,
                )


# Example Usage (adapted for PyTorch DataLoader):
if __name__ == "__main__":
    # Instantiate the Dataset
    train_dataset = DataGenerator(
        "database/AUTORPT/",
        "training",
        # transform=tio.Compose(
        #     [
        #         tio.RandomFlip(axes=(1,), flip_probability=0.5),  # left-right flip
        #         tio.RandomAffine(
        #             scales=(0.85, 1.25),  # zoom in/out 15–25%
        #             degrees=(5, 5, 5),  # small rotations
        #             translation=(0.1, 0.2),  # 10–20% shift along each axis
        #             center="image",  # use image center
        #         ),
        #     ]
        # ),
        shuffle=True,  # Shuffle here for the DataLoader
        batch_size=1,  # This will be ignored by DataLoader
        constraints=PARAMS.constraints,
        is_debug=0,
        weight_ptv=1000,
        # constraint_mode="rweight",  # Using rweight from original example
        constraint_mode="fixed",
        downsampling_factor=1,
        verbose=1,
        is_normalize_weight=True,  # Example from original
        is_return_file_name=True,
    )

    # Instantiate the DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        # num_workers=0,
        num_workers=8,  # try 2, 4, or higher depending on CPU cores
        # pin_memory=True,
        # prefetch_factor=2,  # each worker prefetches batches
        # persistent_workers=True,  # keep workers alive between epochs
    )

    print(f"\nNumber of batches per epoch: {len(train_loader)}")

    for epoch in range(20):
        print(f"\n--- Epoch {epoch+1} ---")
        train_dataset.set_epoch(epoch)  # Call set_epoch for constraint logic

        patients = []
        for i, batch_data in enumerate(train_loader):
            images, y_dose, masks, region_weights, constraints, patient = batch_data
            patients.append(patient)

            print(f"Batch {i+1}:")
            print("  images shape:", images.shape)  # (Batch, C, D, H, W)
            print("  y_dose shape:", y_dose.shape)  # (Batch, D, H, W)
            print("  masks shape:", masks.shape)  # (Batch, C, D, H, W)
            print("  region_weights shape:", region_weights.shape)  # (Batch, D, H, W)
            # Constraints is now a list of dicts or a dict depending on collate_fn
            # Default collate_fn will make it a list of dicts for `constraints`
            # print("  constraints (first item):", json.dumps(constraints[0], indent=4))

            # # Break after a few batches for demonstration
            # if i >= 16:
            #     break

        patients.sort()

        a = 2

        # if epoch >= 0:  # Only run one epoch for quick test
        #     break
