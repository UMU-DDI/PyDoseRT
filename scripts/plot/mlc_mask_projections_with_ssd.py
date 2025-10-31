#!/usr/bin/env python3
"""
mlc_mask_projection.py

Project 3D structure masks (PTV, OARs) into the MLC (u,v) planes for multiple gantry angles,
accounting for a finite source-to-isocenter distance (SSD), compute convex-hull contours,
and plot them in a grid of subplots.

Usage:
    python mlc_mask_projection.py

Dependencies:
    - numpy
    - torch
    - scipy
    - scikit-image
    - matplotlib
    - engine.utils.test_utils.TestSetup (for dummy data example)
"""

import numpy as np
import torch
from skimage import measure
import matplotlib.pyplot as plt
import math
from scipy.spatial import ConvexHull
from engine.utils.test_utils import TestSetup


def get_beam_axes(beam_dir: torch.Tensor):
    """
    Given a unit-vector beam direction, return two orthonormal unit vectors (u, v)
    that span the plane perpendicular to 'beam_dir'. These axes define the MLC plane.

    Args:
        beam_dir (torch.Tensor): 3-element tensor [dx, dy, dz] (unit vector).

    Returns:
        (u, v): Each is a 3-element torch.Tensor, orthonormal to beam_dir and each other.
    """
    device = beam_dir.device

    # Special case: if beam_dir is exactly pointing along +Y, pick u = +X, v = +Z
    if torch.allclose(
        beam_dir, torch.tensor([0.0, 1.0, 0.0], device=device), atol=1e-6
    ):
        u = torch.tensor([1.0, 0.0, 0.0], device=device)
        v = torch.tensor([0.0, 0.0, 1.0], device=device)
    else:
        # Otherwise pick a reference vector not colinear with beam_dir
        if abs(beam_dir[2]) < 0.99:
            ref = torch.tensor([0.0, 0.0, 1.0], device=device)
        else:
            ref = torch.tensor([0.0, 1.0, 0.0], device=device)

        # u = normalized(cross(beam_dir, ref))
        u = torch.cross(beam_dir, ref)
        u = u / (u.norm() + 1e-8)

        # v = normalized(cross(beam_dir, u))
        v = torch.cross(beam_dir, u)
        v = v / (v.norm() + 1e-8)

    return u, v


def project_mask_to_uv_divergent(
    mask_3d: torch.Tensor,
    voxel_sizes: tuple,
    isocenter: tuple,
    beam_dir: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    ssd: float,
):
    """
    Project a 3D binary mask into the (u,v) MLC plane using a diverging beam model:
    - The source is located at 'source = isocenter - beam_dir * ssd'.
    - For each nonzero voxel P in mask_3d, trace the ray from source through P,
      find intersection with the plane (perpendicular to beam_dir) at isocenter.
    - Finally compute (u,v) = dot( (X_int - isocenter), [u,v] ).

    Args:
        mask_3d (torch.Tensor): [W, D, H], binary mask (dtype=bool or uint8).
        voxel_sizes (tuple): (dx, dy, dz) in mm.
        isocenter (tuple): (x0, y0, z0) in mm.
        beam_dir (torch.Tensor): 3-element unit vector along beam direction.
        u (torch.Tensor): first in-plane axis (3-element).
        v (torch.Tensor): second in-plane axis (3-element).
        ssd (float): source-to-isocenter distance in mm.

    Returns:
        np.ndarray: N×2 array of projected points [u_coord, v_coord].
    """
    device = mask_3d.device
    # 1) Find all nonzero voxel indices: [N, 3] = (i, j, k)
    coords = torch.nonzero(mask_3d, as_tuple=False).float()  # float for mm conversion

    if coords.shape[0] < 3:
        # Not enough points to form a meaningful hull; return empty
        return np.zeros((0, 2), dtype=np.float32)

    # 2) Convert voxel indices to mm: (i*dx, j*dy, k*dz)
    spacing = torch.tensor(voxel_sizes, dtype=torch.float32, device=device)  # [3]
    pts_mm = coords * spacing  # [N, 3]

    # 3) Build source location: source = isocenter - ssd * beam_dir
    iso = torch.tensor(isocenter, dtype=torch.float32, device=device)  # [3]
    source = iso - beam_dir * ssd  # [3]

    # 4) For each point P, compute intersection R of ray S→P with plane at isocenter:
    #    Ray param: R(t) = S + t*(P - S).  Solve (R(t) - I)·D = 0  =>  t = ((I - S)·D)/((P - S)·D)
    D = beam_dir  # [3]
    I_minus_S_dot_D = (iso - source).dot(D)  # scalar
    P_minus_S = pts_mm - source.unsqueeze(0)  # [N,3]
    denom = (P_minus_S * D.unsqueeze(0)).sum(dim=1)  # [N]
    # Avoid division by zero: mask out voxels nearly colinear with beam direction
    eps = 1e-8
    mask_valid = denom.abs() > eps
    if mask_valid.sum() < 3:
        return np.zeros((0, 2), dtype=np.float32)

    t_vals = torch.zeros_like(denom)
    t_vals[mask_valid] = I_minus_S_dot_D / denom[mask_valid]  # [N]
    # Intersection points: R = S + t*(P - S)
    R = source.unsqueeze(0) + (P_minus_S * t_vals.unsqueeze(1))  # [N,3]

    # 5) Now project R onto (u, v) by dot product of (R - I) with u and v
    rel = R - iso.unsqueeze(0)  # [N,3]
    u_coords = (rel * u.unsqueeze(0)).sum(dim=1).cpu().numpy()  # [N]
    v_coords = (rel * v.unsqueeze(0)).sum(dim=1).cpu().numpy()  # [N]

    return np.stack([u_coords, v_coords], axis=-1)  # [N,2]


def plot_struct_contours_for_beams(
    masks_3d: torch.Tensor,
    struct_keys: list,
    voxel_sizes: tuple,
    isocenter: tuple,
    ssd: float,
    number_of_cps: int = 8,
    cols: int = 4,
    structure_names: dict = None,
    roi_colors: dict = None,
):
    """
    Plot 2D convex-hull contours of multiple 3D structure masks for a set of beam angles,
    using a diverging beam projection (finite SSD).

    For each beam angle (evenly spaced around 360°):
      - Compute beam_dir, then source = isocenter - beam_dir * ssd.
      - Compute MLC‐plane axes (u, v).
      - Project each structure’s mask via project_mask_to_uv_divergent → (u,v) points.
      - Compute ConvexHull, close the polygon, and plot with a label & color.

    Args:
        masks_3d (torch.Tensor): [B, W, D, H, C] binary masks.  (We assume B=1 here.)
        struct_keys (list): length‐C list of keys identifying each mask (e.g., ["PTV","ROI1",…]).
        voxel_sizes (tuple): (dx, dy, dz) in mm per voxel.
        isocenter (tuple): (x0, y0, z0) isocenter coordinates in mm.
        ssd (float): source‐to‐isocenter distance in mm.
        number_of_cps (int): number of gantry angles to plot (evenly spaced).
        cols (int): number of columns in the subplot grid; rows = ceil(number_of_cps/cols).
        structure_names (dict): maps each struct_key → display name in legend.
        roi_colors (dict): maps each struct_key → color string (e.g. "red").

    Returns:
        None (displays a matplotlib Figure)
    """
    device = masks_3d.device
    B, W, D, H, C = masks_3d.shape
    assert B == 1, "Batch size B must be 1 for plotting"
    assert len(struct_keys) == C

    # Default display names = keys, default colors = None
    if structure_names is None:
        structure_names = {k: k for k in struct_keys}
    if roi_colors is None:
        roi_colors = {k: None for k in struct_keys}

    # Convert isocenter to a torch.Tensor on correct device
    iso = torch.tensor(isocenter, dtype=torch.float32, device=device)

    # Determine grid layout
    rows = math.ceil(number_of_cps / cols)
    fig, axs = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True
    )
    axs = axs.flatten()

    for i in range(number_of_cps):
        angle_deg = 360.0 * i / number_of_cps
        angle_rad = torch.tensor(angle_deg * math.pi / 180.0, device=device)
        beam_dir = torch.tensor(
            [torch.sin(angle_rad), torch.cos(angle_rad), 0.0], device=device
        )
        beam_dir = beam_dir / (beam_dir.norm() + 1e-8)

        # Compute MLC‐plane axes
        u, v = get_beam_axes(beam_dir)

        ax = axs[i]
        ax.set_title(f"Gantry {int(angle_deg)}°")
        ax.set_xlabel("u (mm)")
        ax.set_ylabel("v (mm)")

        # For each structure channel
        for c in range(C - 1):
            key = struct_keys[c]
            mask = masks_3d[0, ..., c]  # [W, D, H]

            # Divergent projection into UV plane
            proj = project_mask_to_uv_divergent(
                mask, voxel_sizes, isocenter, beam_dir, u, v, ssd
            )
            if proj.shape[0] < 3:
                # Not enough points to form a hull
                continue

            try:
                hull = ConvexHull(proj)  # compute 2D convex hull
                poly = proj[hull.vertices]  # extract vertices
                poly = np.vstack([poly, poly[0]])  # close the polygon
                ax.plot(
                    poly[:, 0],
                    poly[:, 1],
                    label=structure_names[key],
                    color=roi_colors.get(key, None),
                    linewidth=2,
                )
            except Exception as e:
                print(f"Warning: Failed ConvexHull for {key} at {int(angle_deg)}°: {e}")

        # Show legend inside this subplot
        ax.legend(fontsize=8)

    # Turn off unused subplots
    for j in range(number_of_cps, len(axs)):
        axs[j].axis("off")

    plt.suptitle("MLC‐Plane Projections of Structure Masks (with SSD)", fontsize=14)
    plt.show()


if __name__ == "__main__":
    # ---------------------------------------------
    # Example / Test using TestSetup dummy data
    # ---------------------------------------------
    # Create dummy CT and masks via TestSetup
    T = TestSetup()
    T.create_dummy(number_of_leaf_pairs=128, number_of_cps=15)

    # Retrieve 3D masks: shape [1, W, D, H, C]
    masks = T.data["masks"]  # numpy array
    # Convert to torch.Tensor on GPU/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    masks_3d = torch.tensor(masks, device=device)

    # Define structure keys and display names
    struct_keys = ["PTV", "ROI1", "ROI2", "ROI3", "ROI4", "ROI5", "ROI6"]
    structure_names = {
        "PTV": "PTV",
        "ROI1": "PenileBulb",
        "ROI2": "FemoralHead_L",
        "ROI3": "FemoralHead_R",
        "ROI4": "Bladder",
        "ROI5": "Rectum",
        "ROI6": "Background",
    }

    # Define colors for each ROI
    roi_colors = {
        "PTV": "orange",
        "ROI1": "red",
        "ROI2": "green",
        "ROI3": "blue",
        "ROI4": "purple",
        "ROI5": "brown",
        "ROI6": "black",
    }

    # Voxel spacing (dx, dy, dz) in mm
    voxel_sizes = (2.5, 2.5, 2.5)

    # Compute isocenter at center of volume in mm
    B, W, D, H, C = masks_3d.shape
    isocenter = (
        W * voxel_sizes[0] / 2.0,
        D * voxel_sizes[1] / 2.0,
        H * voxel_sizes[2] / 2.0,
    )

    # Set a realistic SSD (e.g. 1000 mm)
    ssd = 1000.0

    # Plot for 8 beams with 4 columns
    plot_struct_contours_for_beams(
        masks_3d=masks_3d,
        struct_keys=struct_keys,
        voxel_sizes=voxel_sizes,
        isocenter=isocenter,
        ssd=ssd,
        number_of_cps=8,
        cols=4,
        structure_names=structure_names,
        roi_colors=roi_colors,
    )
