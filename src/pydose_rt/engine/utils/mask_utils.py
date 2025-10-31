import os
import json
import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader
from pydose_rt.engine.config import config as PARAMS

# adapt import to your project structure:
from pydose_rt.engine.data_augment import DataGenerator


import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import SimpleITK as sitk
from typing import Union, List, Tuple


def _to_numpy_ct(
    input_ct: Union[np.ndarray, torch.Tensor, sitk.Image],
) -> Tuple[np.ndarray, dict]:
    """
    Convert various input types to a numpy array with shape (..., Z) where last axis is Z (depth).
    Returns (array, meta) where meta keeps the original type info for restoring result if needed.
    """
    meta = {}
    if isinstance(input_ct, sitk.Image):
        arr = sitk.GetArrayFromImage(input_ct)  # SimpleITK returns (Z, Y, X)
        # move to (Y, X, Z) so last axis is Z to match the function convention
        arr = np.transpose(arr, (1, 2, 0))
        meta["sitk"] = True
        meta["sitk_info"] = input_ct
    elif isinstance(input_ct, torch.Tensor):
        arr = input_ct.detach().cpu().numpy()
        meta["torch"] = True
    elif isinstance(input_ct, np.ndarray):
        arr = input_ct
    else:
        raise ValueError(
            "Unsupported ct input type. Provide numpy, torch tensor or SimpleITK.Image."
        )

    # Ensure last axis is depth (Z). If arr.ndim == 3 we accept any layout but treat last axis as Z.
    if arr.ndim != 3:
        raise ValueError(
            "Expected a 3D volume as numpy/torch/simpleitk -> shape (..., Z). Got shape: {}".format(
                arr.shape
            )
        )

    return arr.astype(np.float32), meta


def _from_numpy_mask(mask_np: np.ndarray, meta: dict):
    """Return mask in original format if requested (SimpleITK) else numpy."""
    if meta.get("sitk"):
        # mask_np is (Y, X, Z) convert back to (Z, Y, X) for SimpleITK
        arr_for_sitk = np.transpose(mask_np, (2, 0, 1)).astype(np.uint8)
        img = sitk.GetImageFromArray(arr_for_sitk)
        img.CopyInformation(meta["sitk_info"])
        return img
    else:
        return mask_np


def make_spherical_struct(radius: int) -> np.ndarray:
    """Return a 3D spherical structuring element with given radius (in voxels)."""
    L = np.arange(-radius, radius + 1)
    X, Y, Z = np.meshgrid(L, L, L, indexing="ij")
    sphere = (X**2 + Y**2 + Z**2) <= radius**2
    return sphere


def get_body_mask_from_normalized_ct(
    ct_normalized: Union[np.ndarray, torch.Tensor, sitk.Image],
    hu_min: int = -1024,
    hu_max: int = 3071,
    hu_threshold: int = -700,
    min_component_voxels: int = 10000,
    closing_radius_voxels: int = 5,
    fill_holes: bool = True,
    remove_table: bool = True,
    depth_axis: int = -1,
):
    """
    ct_normalized : volume in range [-1, 1], shape (..., Z) (last axis treated as Z/depth)
    Returns: final mask in same format as input (SimpleITK if input was SimpleITK, else numpy)
    Also returns a dict `intermediates` with stored intermediate numpy arrays (dtype float32 for images, bool for masks).
    """
    arr, meta = _to_numpy_ct(ct_normalized)  # arr shape (Y, X, Z) — last axis is depth
    # convert normalized [-1,1] -> HU
    hu = ((arr + 1.0) / 2.0) * (hu_max - hu_min) + hu_min
    intermediates = {}
    intermediates["normalized"] = arr.copy()  # normalized [-1,1]
    intermediates["hu"] = hu.copy()  # HU values

    # threshold in HU -> candidate body mask
    thresh_mask = hu > hu_threshold
    intermediates["threshold_mask"] = thresh_mask.copy()

    # remove tiny connected components (3D along axes (Y,X,Z))
    labeled, n = ndi.label(thresh_mask)
    if n == 0:
        # fallback: relax threshold
        thresh_mask = hu > (hu_threshold - 200)
        labeled, n = ndi.label(thresh_mask)
    intermediates["labeled_initial"] = labeled.copy()

    if n > 0:
        counts = np.bincount(labeled.ravel())
        # zero label is background; remove components smaller than min_component_voxels
        too_small = counts < min_component_voxels
        too_small_mask = too_small[labeled]
        after_small = thresh_mask.copy()
        after_small[too_small_mask] = False
    else:
        after_small = thresh_mask.copy()
    intermediates["after_small_removal"] = after_small.copy()

    # Keep largest connected component (typical for torso). If you want arms too, skip this.
    labeled2, n2 = ndi.label(after_small)
    if n2 == 0:
        largest_comp = after_small.copy()  # nothing to do
    else:
        counts2 = np.bincount(labeled2.ravel())
        counts2[0] = 0
        largest_label = counts2.argmax()
        largest_comp = labeled2 == largest_label
    intermediates["largest_component"] = largest_comp.copy()

    # --- NEW: perform a binary opening first to remove thin flat objects like the bed ---
    struct = make_spherical_struct(radius=2)
    # Opening removes small/thin attachments (like a flat bed) while preserving bulk shape.
    opened = ndi.binary_opening(largest_comp, structure=struct, iterations=1)
    intermediates["opened"] = opened.copy()

    # morphological closing to connect limbs / fill small gaps (apply to the opened result)
    if closing_radius_voxels > 0:
        closed = ndi.binary_closing(
            opened,
            structure=struct,
            iterations=closing_radius_voxels,
        )
    else:
        closed = opened.copy()
    intermediates["closed"] = closed.copy()

    # fill holes (3D)
    if fill_holes:
        filled = ndi.binary_fill_holes(closed)
    else:
        filled = closed.copy()
    intermediates["filled"] = filled.copy()

    # optional table removal heuristic
    table_removed = filled.copy()
    if remove_table:
        labeled3, n3 = ndi.label(filled)
        if n3 > 1:
            keep_mask = np.zeros_like(filled, dtype=bool)
            for lab in range(1, n3 + 1):
                comp = labeled3 == lab
                voxels = int(comp.sum())
                # detect touches on depth-first/last slices and XY borders
                touches_front = comp[..., 0].any()
                touches_back = comp[..., -1].any()
                touches_xy_border = (
                    comp[0, :, :].any()
                    or comp[-1, :, :].any()
                    or comp[:, 0, :].any()
                    or comp[:, -1, :].any()
                )
                # keep criteria: not touching both ends or very large component (probably patient)
                if (
                    (not touches_front and not touches_back)
                    or voxels > 5 * min_component_voxels
                    or (not touches_xy_border)
                ):
                    keep_mask |= comp
            if keep_mask.sum() == 0:
                # fallback: keep largest
                counts3 = np.bincount(labeled3.ravel())
                counts3[0] = 0
                if counts3.size > 1:
                    largest_lab = counts3.argmax()
                    keep_mask = labeled3 == largest_lab
            table_removed = keep_mask
    intermediates["table_removed"] = table_removed.copy()

    # final cleaning: remove small components again
    labeled_final, n_final = ndi.label(table_removed)
    if n_final > 1:
        counts_final = np.bincount(labeled_final.ravel())
        counts_final[0] = 0
        keep_labels = np.where(counts_final >= min_component_voxels)[0]
        if len(keep_labels) == 0:
            # keep largest
            keep_labels = [counts_final.argmax()]
        mask_final = np.isin(labeled_final, keep_labels)
    else:
        mask_final = table_removed.copy()
    intermediates["final_mask"] = mask_final.copy()

    # Return final mask in original format if needed
    out_mask = _from_numpy_mask(mask_final.astype(np.uint8), meta)
    return out_mask, intermediates


def get_batched_body_mask_from_normalized_ct(
    x_batch: np.ndarray,
    **kwargs,
):
    """
    Compute body mask for a batch of normalized CT volumes.

    Parameters
    ----------
    x_batch : np.ndarray or torch.Tensor
        Array of shape [B, W, H, D] or [B, Y, X, Z] with normalized CT values in [-1, 1].
    **kwargs :
        Extra arguments passed to body_mask_from_normalized_ct
        (e.g., closing_radius_voxels=5, fill_holes=True, etc.)

    Returns
    -------
    body_mask : np.ndarray
        Boolean mask of shape [B, W, H, D] corresponding to each batch element.
    intermediates_list : list[dict]
        List of intermediates dictionaries (one per batch item).
    """
    import torch

    # Move to numpy if necessary
    if isinstance(x_batch, torch.Tensor):
        x_np = x_batch.detach().cpu().numpy()
    else:
        x_np = np.asarray(x_batch)

    B = x_np.shape[0]
    body_masks = []
    intermediates_list = []

    for b in range(B):
        mask_img_or_np, intermediates = get_body_mask_from_normalized_ct(
            x_np[b, ...], **kwargs
        )
        # make sure mask is numpy array (not sitk.Image)
        if not isinstance(mask_img_or_np, np.ndarray):
            mask_np = intermediates["final_mask"].astype(np.uint8)
        else:
            mask_np = np.asarray(mask_img_or_np).astype(np.uint8)
        body_masks.append(mask_np)
        intermediates_list.append(intermediates)

    body_masks = np.stack(body_masks, axis=0)  # shape [B, W, H, D]
    return body_masks, intermediates_list


def plot_all_intermediates(
    intermediates: dict,
    final_mask: np.ndarray,
    slice_indices: list = None,
    depth_axis: int = -1,
    figsize_per_panel=(5, 5),
    cmap_ct="gray",
    cmap_mask="gray",
):
    """
    Plot every intermediate plus the final mask for selected Z slices.

    Rows: each intermediate + final mask
    Columns: selected slices along depth
    """
    # choose representative depth
    sample = next(iter(intermediates.values()))
    depth = sample.shape[depth_axis]

    if slice_indices is None:
        mid = depth // 2
        slice_indices = sorted(
            list(set([mid, max(0, mid // 2), min(depth - 1, mid + max(1, depth // 4))]))
        )

    # rows = intermediates + final mask
    intermediate_names = list(intermediates.keys())
    intermediate_names.append("final_mask")
    n_rows = len(intermediate_names)
    n_cols = len(slice_indices)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.expand_dims(axes, 0)
    elif n_cols == 1:
        axes = np.expand_dims(axes, 1)

    for r, name in enumerate(intermediate_names):
        if name == "final_mask":
            arr = final_mask
            cmap = cmap_mask
        else:
            arr = intermediates[name]
            # use gray for CT/HU, red for masks
            cmap = cmap_ct if name in ["hu", "normalized"] else cmap_mask

        for c, si in enumerate(slice_indices):
            slice_arr = np.take(arr, si, axis=depth_axis)
            ax = axes[r, c]
            ax.imshow(np.rot90(slice_arr.T), cmap=cmap, origin="lower")
            if c == 0:
                ax.set_ylabel(name, rotation=0, labelpad=40, va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"z={si}")

    plt.tight_layout()
    plt.show()


def do():
    # instantiate as user example
    train_dataset = DataGenerator(
        data_path="database/AUTORPT/",
        cohort="training",
        shuffle=True,
        batch_size=2,
        constraints=PARAMS.constraints,
        is_debug=2,  # only load our one case
        weight_ptv=1000,  # override PTV weight
        constraint_mode="fixed",
        downsampling_factor=(1, 1, 1),
        verbose=False,
        is_normalize_weight=True,
        transform=None,
        transform_mask=None,
    )
    train_dataset.set_epoch(0)  # so normalization logic runs
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # get one batch
    x, dose_tensor, masks, region_weights, constraints = next(iter(train_loader))
    x_ct = x[0, 0, ...]

    mask_img_or_np, intermediates = get_body_mask_from_normalized_ct(x_ct)
    # pick slices to plot (or let function pick defaults)
    final_mask = intermediates["final_mask"]
    plot_all_intermediates(intermediates, final_mask)

    a = 2


if __name__ == "__main__":
    do()
