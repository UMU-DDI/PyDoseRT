#!/usr/bin/env python3
"""
mlc_mask_projection.py

Project 3D structure masks (PTV, OARs) into MLC (u,v) planes for multiple gantry angles,
compute convex hull contours, and plot them in a grid of subplots.

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
from pydose_rt.engine.utils.test_utils import TestSetup


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

    # Special case: if beam_dir is exactly pointing along +Y, choose u = +X, v = +Z
    if torch.allclose(
        beam_dir, torch.tensor([0.0, 1.0, 0.0], device=device), atol=1e-6
    ):
        u = torch.tensor([1.0, 0.0, 0.0], device=device)  # along +X
        v = torch.tensor([0.0, 0.0, 1.0], device=device)  # along +Z
    else:
        # Otherwise pick a reference vector that is not colinear with beam_dir.
        # If beam_dir's Z component is small (<0.99), use +Z as reference; else use +Y.
        if abs(beam_dir[2]) < 0.99:
            ref = torch.tensor([0.0, 0.0, 1.0], device=device)
        else:
            ref = torch.tensor([0.0, 1.0, 0.0], device=device)

        # Compute u = normalized(cross(beam_dir, ref))
        u = torch.cross(beam_dir, ref)
        u = u / (u.norm() + 1e-8)

        # Compute v = normalized(cross(beam_dir, u))
        v = torch.cross(beam_dir, u)
        v = v / (v.norm() + 1e-8)

    return u, v


def project_mask_to_uv(
    mask_3d: torch.Tensor,
    voxel_sizes: tuple,
    isocenter: tuple,
    beam_dir: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
):
    """
    Project the nonzero voxels of a 3D binary mask into the (u,v) plane.

    1) Find all voxel indices where mask==1.
    2) Convert voxel indices (i,j,k) to physical mm coordinates by multiplying by voxel_sizes.
    3) Subtract isocenter (in mm) to get coordinates relative to isocenter.
    4) Dot-product with u and v axes to get 2D (u,v) coordinates.

    Args:
        mask_3d (torch.Tensor): Binary 3D mask of shape [W, D, H], dtype=bool or uint8.
        voxel_sizes (tuple): (dx, dy, dz) in mm for each voxel dimension.
        isocenter (tuple): (x0, y0, z0) isocenter position in mm.
        beam_dir (torch.Tensor): 3-element unit vector along beam, dtype=float32.
        u (torch.Tensor): First in-plane axis (3-element).
        v (torch.Tensor): Second in-plane axis (3-element).

    Returns:
        np.ndarray: N×2 array of [u_coord, v_coord] for each nonzero voxel in mask.
    """
    # 1) Get coordinates of all nonzero voxels: shape [N, 3] = (i, j, k)
    coords = torch.nonzero(
        mask_3d, as_tuple=False
    ).float()  # convert to float for mm conversion

    # 2) Build a spacing tensor [dx, dy, dz] on the same device
    spacing = torch.tensor(voxel_sizes, dtype=torch.float32, device=mask_3d.device)

    # Convert voxel indices to mm: (i*dx, j*dy, k*dz)
    pts_mm = coords * spacing  # [N, 3]

    # 3) Compute coordinates relative to isocenter: (x,y,z) - (x0,y0,z0)
    isoc = torch.tensor(isocenter, dtype=torch.float32, device=mask_3d.device)
    rel = pts_mm - isoc.unsqueeze(0)  # broadcast isocenter to [N,3]

    # 4) Project onto (u,v) by dot-products; result is CPU numpy arrays
    u_coords = (rel @ u).cpu().numpy()  # shape [N]
    v_coords = (rel @ v).cpu().numpy()  # shape [N]

    # Stack into N×2 array
    return np.stack([u_coords, v_coords], axis=-1)  # shape [N,2]


def plot_struct_contours_for_beams(
    masks_3d: torch.Tensor,
    struct_keys: list,
    voxel_sizes: tuple,
    isocenter: tuple,
    number_of_cps: int = 8,
    cols: int = 4,
    structure_names: dict = None,
    roi_colors: dict = None,
):
    """
    Plot 2D convex-hull contours of multiple 3D structure masks for a set of beam angles.

    For each beam angle (evenly spaced around 360°), compute beam_dir,
    project each mask into the MLC (u,v) plane, compute its convex hull,
    and draw a closed contour in a grid of subplots (rows = ceil(number_of_cps/cols), cols fixed).

    Args:
        masks_3d (torch.Tensor): [B, W, D, H, C] 5D tensor of binary masks (batch size B assumed 1).
        struct_keys (list): list of C keys indexing the masks, e.g. ["PTV","ROI1",...].
        voxel_sizes (tuple): (dx, dy, dz) in mm per voxel.
        isocenter (tuple): (x0, y0, z0) isocenter in mm.
        number_of_cps (int): number of gantry angles to plot (evenly spaced).
        cols (int): number of columns in the subplot grid.
        structure_names (dict): maps each struct_key to a display name in legend.
        roi_colors (dict): maps each struct_key to a color string for plotting.

    Returns:
        None (displays a matplotlib Figure with subplots)
    """
    device = masks_3d.device
    B, W, D, H, C = masks_3d.shape
    assert B == 1, "Batch size B must be 1 for this plotting utility"
    assert (
        len(struct_keys) == C
    ), "Length of struct_keys must equal the # of mask channels"

    # If no custom display names provided, use keys directly
    if structure_names is None:
        structure_names = {k: k for k in struct_keys}
    # If no custom colors provided, default to None (matplotlib picks automatically)
    if roi_colors is None:
        roi_colors = {k: None for k in struct_keys}

    # Convert isocenter to a torch.Tensor on the same device
    isoc = torch.tensor(isocenter, dtype=torch.float32, device=device)

    # Compute number of rows needed for given #beams and columns
    rows = math.ceil(number_of_cps / cols)

    # Create a figure with subplots: rows × cols
    fig, axs = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True
    )
    axs = axs.flatten()  # flatten to 1D list for easy indexing

    # Loop over each beam index
    for i in range(number_of_cps):
        # Compute gantry angle in degrees and radians
        angle_deg = 360.0 * i / number_of_cps
        angle_rad = torch.tensor(angle_deg * math.pi / 180.0, device=device)

        # Beam direction vector (sinθ, cosθ, 0) in patient coordinates
        beam_dir = torch.tensor(
            [torch.sin(angle_rad), torch.cos(angle_rad), 0.0], device=device
        )
        beam_dir = beam_dir / (beam_dir.norm() + 1e-8)  # normalize to unit length

        # Compute MLC‐plane axes u, v
        u, v = get_beam_axes(beam_dir)

        # Select the corresponding subplot
        ax = axs[i]
        ax.set_title(f"Gantry {int(angle_deg)}°")
        ax.set_xlabel("u (mm)")
        ax.set_ylabel("v (mm)")

        # For each structure channel (except the last “background” if you want to skip)
        for c in range(C - 1):
            key = struct_keys[c]
            mask = masks_3d[0, ..., c]  # [W, D, H]

            # Project mask into UV plane, get Nx2 array of points
            proj = project_mask_to_uv(mask, voxel_sizes, isoc, beam_dir, u, v)
            if proj.shape[0] < 3:
                # Need at least 3 points to form a convex hull
                continue
            try:
                # Compute 2D convex hull of the projected points
                hull = ConvexHull(proj)
                poly = proj[hull.vertices]  # vertices in hull order

                # Close the polygon by appending the first point at the end
                poly = np.vstack([poly, poly[0]])

                # Plot with custom color and label
                ax.plot(
                    poly[:, 0],
                    poly[:, 1],
                    label=structure_names[key],
                    color=roi_colors.get(key, None),
                    linewidth=2,
                )
            except Exception as e:
                print(f"Warning: Failed ConvexHull for {key} at {int(angle_deg)}°: {e}")

        # Show legend (one per subplot)
        ax.legend(fontsize=8)

    # Turn off any unused subplots
    for j in range(number_of_cps, len(axs)):
        axs[j].axis("off")

    plt.suptitle("MLC Plane Projections of Structure Masks", fontsize=14)
    plt.show()


if __name__ == "__main__":
    # ---------------------------------------------
    # Example / Test using TestSetup dummy data
    # ---------------------------------------------
    # Create dummy CT and masks via TestSetup
    T = TestSetup()
    T.create_dummy(number_of_leaf_pairs=128, number_of_cps=15)

    # Retrieve 3D masks: shape [1, W, D, H, C]
    masks = T.data["masks"]  # numpy array, dtype e.g. uint8 or bool
    masks_3d = torch.tensor(
        masks, device=("cuda" if torch.cuda.is_available() else "cpu")
    )

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

    # Voxel spacing and isocenter in mm
    voxel_sizes = (1.0, 1.0, 1.0)
    W, D, H, C = masks_3d.shape[1:]
    # Place isocenter at center of volume
    isocenter = (
        W * voxel_sizes[0] / 2.0,
        D * voxel_sizes[1] / 2.0,
        H * voxel_sizes[2] / 2.0,
    )

    # Plot for 8 beams with 4 columns
    plot_struct_contours_for_beams(
        masks_3d=masks_3d,
        struct_keys=struct_keys,
        voxel_sizes=voxel_sizes,
        isocenter=isocenter,
        number_of_cps=15,
        cols=5,
        structure_names=structure_names,
        roi_colors=roi_colors,
    )
