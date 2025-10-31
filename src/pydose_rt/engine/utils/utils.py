import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.ndimage as ndi
from pydose_rt.engine.utils.test_utils import TestSetup

# from pydose_rt import ModelConfig


def downsample_ct_by_2(ct_array):
    """
    Downsamples a 3D CT array (NumPy array) by a factor of 2 along all three axes.

    Args:
        ct_array: A 3D NumPy array representing the CT volume. The array must
                  have an even number of elements along each axis
                  (i.e., ct_array.shape must be (d1, d2, d3) where d1, d2,
                  and d3 are even).

    Returns:
        A new 3D NumPy array that is downsampled by a factor of 2.
        The output array has shape (d1//2, d2//2, d3//2).
    """
    # Check that the input array is 3D
    if ct_array.ndim != 3:
        raise ValueError("Input array must be 3-dimensional.")

    # Check that the input array has an even number of elements along each axis.
    for i, dim_size in enumerate(ct_array.shape):
        if dim_size % 2 != 0:
            raise ValueError(
                f"CT array must have an even number of elements along axis {i} (size was {dim_size})"
            )

    # Use slicing to downsample.  The ::2 syntax selects every other element
    # along each axis.
    return ct_array[::2, ::2, ::2]


def prepare_to_plot(x):
    try:
        x = x.cpu().detach()
    except:
        pass
    x_np = x.numpy()
    x_np = x_np.transpose(0, 3, 2, 1)  # → (B, X, Y, Z)
    x_np = x_np.squeeze(0)  # now (Z, Y, X)
    return x_np


def plot_ct_and_doses(ct_np, dose_list, slice_idx=None, extent=None):
    """
    Plot CT, multiple dose distributions, and overlays in a 3×N grid.

    Args:
        ct_np:      np.ndarray, shape (Z, Y, X)
        dose_list:  list of np.ndarray, each shape (Z, Y, X)
        slice_idx:  int or None → which Z-slice to plot. None→middle slice.
        extent:     [xmin, xmax, ymin, ymax] passed to imshow.
    """
    X, Y, Z = ct_np.shape
    N = len(dose_list)
    if slice_idx is None:
        slice_idx = Z // 2

    ct_slice = ct_np[:, :, slice_idx]
    ct_slice = np.rot90(ct_slice, k=2)

    # Prepare figure
    fig, axes = plt.subplots(3, N, figsize=(4 * N, 12))
    # If N==1, axes will be (3,), so we reshape
    if N == 1:
        axes = axes.reshape(3, 1)

    for col in range(N):
        dose_np = dose_list[col]
        dose_slice = dose_np[:, :, slice_idx]
        dose_slice = np.rot90(dose_slice, k=2)

        # Row 1: CT
        ax0 = axes[0, col]
        im0 = ax0.imshow(ct_slice, cmap="gray", origin="lower", extent=extent)
        ax0.set_title("CT")
        ax0.axis("off")

        # Row 2: Dose
        ax1 = axes[1, col]
        im1 = ax1.imshow(dose_slice, cmap="jet", origin="lower", extent=extent)
        ax1.set_title(f"Dose {col}")
        ax1.axis("off")

        # Row 3: Overlay
        ax2 = axes[2, col]
        ax2.imshow(ct_slice, cmap="gray", origin="lower", extent=extent)
        ax2.imshow(dose_slice, cmap="jet", alpha=0.5, origin="lower", extent=extent)
        ax2.set_title(f"Dose {col} over CT")
        ax2.axis("off")

    plt.tight_layout()
    plt.show()


def animate_ct_and_doses(
    ct_np,  # np.ndarray, shape (Z, Y, X)
    dose_list,  # list of np.ndarray, each shape (Z, Y, X)
    out_path,  # path to save mp4, e.g. "dose_movie.mp4"
    slice_indices=None,  # list or range of Z indices; None→all
    extent=None,  # [xmin, xmax, ymin, ymax]
    fps=5,
    dpi=100,
):
    """
    Sweep through Z-slices and animate the same 3×N grid as plot_ct_and_doses.
    Row 0: CT slice
    Row 1: Dose slice
    Row 2: Overlay
    """
    X, Y, Z = ct_np.shape
    N = len(dose_list)
    if slice_indices is None:
        slice_indices = range(Z)

    # Prepare the figure and axes
    fig, axes = plt.subplots(3, N, figsize=(4 * N, 12))
    if N == 1:
        axes = axes.reshape(3, 1)

    # Initialize each subplot with slice 0
    ims = []
    ct_min, ct_max = ct_np.min(), ct_np.max()

    for col in range(N):
        # Row 0: CT
        ax0 = axes[0, col]
        ct0 = np.rot90(ct_np[:, :, slice_indices[0]], k=2)
        im_ct = ax0.imshow(
            ct0, cmap="gray", origin="lower", extent=extent, vmin=ct_min, vmax=ct_max
        )
        ax0.set_title("CT")
        ax0.axis("off")
        ims.append(im_ct)

        # Row 1: Dose
        ax1 = axes[1, col]
        dose_min, dose_max = dose_list[col].min(), dose_list[col].max()
        dose0 = np.rot90(dose_list[col][:, :, slice_indices[0]], k=2)

        im_d = ax1.imshow(
            dose0,
            cmap="jet",
            origin="lower",
            extent=extent,
            vmin=dose_min,
            vmax=dose_max,
        )
        ax1.set_title(f"Dose {col}")
        ax1.axis("off")
        ims.append(im_d)

        # Row 2: Overlay
        ax2 = axes[2, col]
        # First plot CT background
        ax2.imshow(
            ct0,
            cmap="gray",
            origin="lower",
            extent=extent,
            vmin=ct_min,
            vmax=ct_max,
        )
        # Then dose overlay
        im_ov = ax2.imshow(
            dose0,
            cmap="jet",
            alpha=0.5,
            origin="lower",
            extent=extent,
            vmin=dose_min,
            vmax=dose_max,
        )
        ax2.set_title(f"Dose {col} over CT")
        ax2.axis("off")
        ims.append(im_ov)

    plt.tight_layout()

    def update(frame):
        z = slice_indices[frame]
        for col in range(N):
            ct_slice = np.rot90(ct_np[:, :, z], k=2)
            dose_slice = np.rot90(dose_list[col][:, :, z], k=2)

            # CT image artist at 3*col + 0
            ims[3 * col + 0].set_array(ct_slice)
            # Dose-only at 3*col + 1
            ims[3 * col + 1].set_array(dose_slice)
            # Overlay at 3*col + 2 (the second artist in ax2)
            ims[3 * col + 2].set_array(dose_slice)

        return ims

    ani = animation.FuncAnimation(fig, update, frames=len(slice_indices), blit=True)
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)


# Example usage:
# animate_ct_and_doses(ct_np, [dose0, dose1, dose2], "dose_movie.mp4", extent=[-30,30,-30,30])


def compute_valid_leaf_mask_minh(
    ptv_mask,  # [B, W, D, H] boolean PTV mask in voxel-indices
    config,
    leaf_width=1,
    voxel_sizes=(1, 1, 1),
    margin_mm: float = 0,
) -> torch.BoolTensor:
    """
    Returns a (B, number_of_cps, num_leafs) mask marking which leaves ever intercept the PTV for each batch.
    Assumes leaves move along the z-axis (H axis).
    """
    if ptv_mask.ndim == 3:
        ptv_mask = ptv_mask.unsqueeze(0)  # [1, W, D, H]
    B = ptv_mask.shape[0]
    number_of_cps = config.number_of_cps
    num_leafs = config.number_of_leaf_pairs

    (W, D, H) = config.ct_array_shape
    dx, dy, dz = voxel_sizes

    iso_x = (W // 2) * dx
    iso_y = (D // 2) * dy
    iso_z = (H // 2) * dz

    isocenter = (iso_x, iso_y, iso_z)

    device = ptv_mask.device

    all_valid_leaf = torch.zeros(
        (B, number_of_cps, num_leafs), dtype=torch.uint8, device=device
    )

    for b in range(B):
        # 1) Gather PTV voxel centers (in mm)
        coords = torch.nonzero(
            ptv_mask[b], as_tuple=False
        ).float()  # [N, 3] indices: [w, d, h]
        if coords.shape[0] == 0:
            continue  # No PTV in this batch
        pts_mm = coords * torch.tensor([dx, dy, dz], device=device)  # [N,3] in mm

        # 2) Project all PTV points onto the z-axis (leaf direction)
        v_coord = pts_mm[:, 2]  # z in mm

        # 3) Compute leaf centers along z-axis (centered at isocenter z)
        z_leaf_centers = (
            torch.linspace(
                -(num_leafs / 2 - 0.5) * leaf_width,
                (num_leafs / 2 - 0.5) * leaf_width,
                num_leafs,
                device=device,
            )
            + iso_z
        )  # [num_leafs]

        # 4) For each beam, mark leaves whose center is within the PTV z-range
        z_min = v_coord.min().item()
        z_max = v_coord.max().item()

        valid_leaf_1d = (z_leaf_centers >= (z_min - margin_mm)) & (
            z_leaf_centers <= (z_max + margin_mm)
        )
        valid_leaf_per_beam = (
            valid_leaf_1d.unsqueeze(0).expand(number_of_cps, -1).clone()
        )
        all_valid_leaf[b] = valid_leaf_per_beam

    return all_valid_leaf  # shape: (B, number_of_cps, num_leafs)


def _compute_valid_leaf_mask_attila(ptv_mask, config) -> torch.BoolTensor:
    device = ptv_mask.device
    B = ptv_mask.shape[0]
    number_of_cps = config.number_of_cps
    num_leafs = config.number_of_leaf_pairs

    all_valid_leaf = torch.ones(
        (B, number_of_cps, num_leafs), dtype=torch.uint8, device=device
    )
    # all_valid_leaf[:, :, 20:40] = 1
    return all_valid_leaf  # shape: (B, number_of_cps, num_leafs)


def compute_valid_leaf_mask(
    dose_engine,
    dose_model,
    ct,  # Tensor of shape [1, Z, Y, X, 1]
    ptv_mask,  # Tensor of shape [1, Z, Y, X, 1], binary {0,1}
    n_cps: int,
    n_leafs: int,
    eps=1e-6,
    device=None,
):
    """
    Identifies MLC leaves that do not affect the PTV dose.

    Args:
        dose_model: A PyTorch module that computes dose from CT, MLC, and MU.
        ct: 5D CT image tensor [1, Z, Y, X, 1].
        ptv_mask: Binary mask of PTV region, same shape as ct.
        mlc: MLC positions [1, n_cps, n_leafs, 2] (left/right).
        mus: Monitor units per control point [1, n_cps].
        eps: Threshold for considering a gradient to be effectively zero.

    Returns:
        out_of_range: Boolean mask [1, n_leafs] — True if the leaf does not affect the PTV dose.
    """
    # Prevent gradients for dose_engine parameters
    for param in dose_model.parameters():
        param.requires_grad = False

    B = ct.shape[0]
    n_sides = 2  # MLC has 2 sides

    if device is None:
        device = ct.device if ct.device is not None else torch.device("cpu")

    # --- Create dummy mlc and mus ---
    mlc = torch.zeros((B, n_cps, n_leafs, n_sides), device=device)
    mlc[:, :, :, 1] = 1
    if dose_engine in ["attila", "matthias"]:
        mlc = mlc.permute(0, 3, 1, 2)
    elif dose_engine == "minh":  # does not work for now
        pass
    mus = torch.ones((B, n_cps), device=device)

    # Clone mlc and set requires_grad=True
    mlc = mlc.clone().requires_grad_(True)

    # Perform the forward+backward in an enabled-grad block even if outer context is no_grad()
    with torch.enable_grad():
        # Forward pass
        if dose_engine == "attila":
            dose_pred = dose_model(ct, mlc, mus)  # Predict 3D dose
        elif dose_engine == "matthias":
            dose_pred = dose_model(mlc, mus, jaw_positions=None, ct_image=ct * 1000)

        ptv_dose = dose_pred * ptv_mask  # Isolate PTV dose
        ptd = torch.sum(ptv_dose)  # Total dose in PTV

        # Backward pass
        ptd.backward()

        # Get gradients
        grads = mlc.grad  # [1, 2, n_cps, n_leafs]

        grads = grads.sum(axis=1)

        out_of_range = grads < eps

    valid_leaf = ~out_of_range
    return valid_leaf


def compute_leaf_bounds(
    ptv_mask: np.ndarray,
    beam_angles: np.ndarray,
    num_leafs: int,
    leaf_width: float,
    voxel_sizes: tuple = (1.0, 1.0, 1.0),
    margin_mm: float = 0.0,
):
    """
    Args:
        ptv_mask:    (H, W, D) binary PTV volume.
                     H = rows (z direction), W = columns (x direction), D = depth (y).
        beam_angles: (B,) array of gantry angles in degrees.
        num_leafs:   number of leaf rows.
        leaf_width:  physical height of each leaf (mm).
        voxel_sizes: tuple(dx, dy, dz) in mm for (x, y, z).
        margin_mm:   extra margin around PTV (mm).
    Returns:
        bounds: np.ndarray of shape (B, num_leafs, 2) with normalized [l,r].
    """
    dx, dy, dz = voxel_sizes
    H, W, D = ptv_mask.shape
    flat_angles = np.atleast_1d(beam_angles).ravel()
    B = len(flat_angles)
    bounds = np.zeros((B, num_leafs, 2), dtype=np.float32)

    # 1) Collapse depth (D) → 2D projection in (H,W)
    proj = ptv_mask.max(axis=2).astype(np.uint8)  # shape (H, W)

    # 2) Compute physical z‐centers of each leaf (along H axis)
    #    voxel index h=0..H-1 maps to z = h*dz, with isocenter at H/2*dz
    iso_z = (H / 2) * dz
    leaf_centers_z = (np.arange(num_leafs) - (num_leafs / 2 - 0.5)) * leaf_width + iso_z

    for i, angle in enumerate(flat_angles):
        # 3) Rotate to beam‐eye view so that beam travel (x-axis) is horizontal
        rot = ndi.rotate(
            proj, -float(angle), reshape=False, order=0, mode="constant", cval=0
        )
        # rot still shape (H, W)

        for leaf_idx, zc in enumerate(leaf_centers_z):
            # 4) Determine which rows in rot correspond to this leaf's z‐span
            zmin = zc - leaf_width / 2 - margin_mm
            zmax = zc + leaf_width / 2 + margin_mm
            # Convert zmin/zmax back to row indices
            jmin = int(np.floor(zmin / dz))
            jmax = int(np.ceil(zmax / dz))
            jmin = max(jmin, 0)
            jmax = min(jmax, H - 1)

            # 5) Extract stripe of rows and collapse to 1D along x (W)
            stripe = rot[jmin : jmax + 1, :]  # shape (~rows_per_leaf, W)

            if stripe.size:
                occupied = stripe.max(axis=0)  # (W,)
                xs = np.nonzero(occupied)[0]
                if xs.size > 0:
                    l = xs.min() / float(W - 1)
                    r = xs.max() / float(W - 1)
                else:
                    # no PTV under this leaf → close it
                    l = 0.0
                    r = 0.0
            else:
                l = 0.0
                r = 0.0

            bounds[i, leaf_idx, 0] = l
            bounds[i, leaf_idx, 1] = r

    return bounds


def prepare_real(is_hu=False):
    T = TestSetup()
    T.create_dummy(number_of_leaf_pairs=128, number_of_cps=15)

    ct = T.ct  # numpy array

    # ct_np = ct[:, :, 100:200]
    ct_np = ct
    ptv = T.data["masks"][0, ..., 0]
    # ptv = ct[:, :, 100:200]

    ct_np = downsample_ct_by_2(ct_np)  # Assumes it returns a numpy array
    ptv = downsample_ct_by_2(ptv)  # Assumes it returns a numpy array

    # ct_np = np.transpose(ct_np, (2, 1, 0))  # shape: (Z, Y, X)
    X, Y, Z = ct_np.shape

    # ct_np = np.expand_dims(ct_np, axis=0)  # (1, Z, Y, X)
    ct_torch = torch.tensor(ct_np, dtype=torch.float32)

    if is_hu:
        # Convert normalized CT back to HU
        HU_MIN = -1000
        HU_MAX = 3000
        ct_torch = ((ct_torch + 1) / 2) * (HU_MAX - HU_MIN) + HU_MIN
    # else:
    #     ct_torch = (ct_torch - ct_torch.min()) / (
    #         ct_torch.max() - ct_torch.min() + 1e-8
    #     )

    return ct_torch, ptv, X, Y, Z


# if __name__ == "__main__":
#     batch_size = 1
#     number_of_cps = 90
#     num_leafs = 60
#     # voxel_sizes = (1.0, 1.0, 1.0)  # mm
#     voxel_sizes = tuple([0.75 / 60 * num_leafs for i in range(3)])  # mm
#     # voxel_sizes = tuple([0.1 for i in range(3)])  # mm
#     dx, dy, dz = voxel_sizes
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     ct_data, ptv, W, D, H = prepare_real(is_hu=False)

#     config = ModelConfig(
#         ct_array_shape=(W, D, H),
#         number_of_leaf_pairs=num_leafs,
#         number_of_cps=number_of_cps,
#         field_size=(num_leafs * 1.0, num_leafs * 1.0),
#         resolution=voxel_sizes,
#         tpr_20_10=0.72,
#     )

#     ct_data = ct_data.unsqueeze(0).expand(batch_size, -1, -1, -1)

#     # iso_x = (W // 2) * dx
#     # iso_y = (D // 2) * dy
#     # iso_z = (H // 2) * dz

#     # beam_angles = np.tile(
#     #     np.linspace(0, 360, number_of_cps, endpoint=False), (batch_size, 1)
#     # )

#     # masks = compute_leaf_bounds(ptv, beam_angles, num_leafs, leaf_width=1)
#     # print(masks[0, :, :])

#     # print()
#     # print()
#     # print()

#     # masks = compute_leaf_bounds(ptv, beam_angles, num_leafs, leaf_width=3)
#     # print(masks[0, :, :])

#     # a = 0
