"""
Quick-look debug plots for a dose calculation pipeline.

Two entry points are exported:

- :func:`plot_beam_debug` — save a per-beam panel showing the CT, fluence map,
  radiological-depth slice (when available) and the resulting dose.
- :func:`plot_total_dose_debug` — save a single panel with axial / sagittal /
  coronal slices of an accumulated dose through the isocenter.

Both functions are intentionally minimal: they only depend on numpy, matplotlib
and the tensors that any engine already produces, so they can be called from
anywhere in the pipeline without pulling in heavier plotting machinery.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib

# Use a non-interactive backend by default — debug plots are written to disk.
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(t: torch.Tensor | None) -> np.ndarray | None:
    if t is None:
        return None
    return t.detach().to(torch.float32).cpu().numpy()


def _pick_iso_voxel(shape, iso_center, spacing):
    """Return (ih, id, iw) voxel indices of the isocenter, clamped to the grid."""
    H, D, W = shape
    rx, ry, rz = spacing
    X, Y, Z = iso_center
    ih = int(max(0, min(H - 1, X / rx)))
    id_ = int(max(0, min(D - 1, Y / ry)))
    iw = int(max(0, min(W - 1, Z / rz)))
    return ih, id_, iw


def plot_beam_debug(
    out_path: str | Path,
    *,
    beam_index: int,
    gantry_angle_rad: float,
    mu: float | None,
    ct: torch.Tensor,
    dose: torch.Tensor,
    iso_center: tuple[float, float, float],
    dose_grid_spacing: tuple[float, float, float],
    fluence_map: torch.Tensor | None = None,
    rad_depth_bev: torch.Tensor | None = None,
    title: str | None = None,
) -> None:
    """Save a quick diagnostic figure for a single beam's contribution.

    Panels (rows × cols = 2 × 3):

    =========================================  ====================================
    CT axial slice at iso (+ dose overlay)     CT coronal at iso (+ dose overlay)
    CT sagittal at iso (+ dose overlay)        Fluence map (if provided)
    Rad-depth BEV mid slice (if provided)      Dose central-axis profile
    =========================================  ====================================

    Every tensor is assumed to live on any device and to have floating dtype;
    tensors are detached and moved to CPU automatically.
    """
    ct_np = _to_numpy(ct).squeeze()
    dose_np = _to_numpy(dose).squeeze()
    fluence_np = _to_numpy(fluence_map)
    rad_np = _to_numpy(rad_depth_bev)

    if ct_np.ndim != 3:
        raise ValueError(f"Expected 3-D CT volume after squeeze, got shape {ct_np.shape}")
    if dose_np.ndim != 3:
        raise ValueError(f"Expected 3-D dose volume after squeeze, got shape {dose_np.shape}")

    H, D, W = ct_np.shape
    ih, id_, iw = _pick_iso_voxel(ct_np.shape, iso_center, dose_grid_spacing)

    dose_max = float(np.nanmax(dose_np)) if np.isfinite(dose_np).any() else 0.0
    dose_max = dose_max if dose_max > 0 else 1.0

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # --- Row 1: anatomical slices with dose overlay
    def _overlay(ax, ct_slice, dose_slice, label):
        ax.imshow(ct_slice, cmap="gray", aspect="auto")
        ax.imshow(
            np.ma.masked_less(dose_slice, dose_max * 0.01),
            cmap="turbo", alpha=0.45, aspect="auto",
            vmin=0, vmax=dose_max,
        )
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])

    _overlay(axes[0, 0], ct_np[ih, :, :], dose_np[ih, :, :], f"Axial @ H={ih}")
    _overlay(axes[0, 1], ct_np[:, id_, :], dose_np[:, id_, :], f"Coronal @ D={id_}")
    _overlay(axes[0, 2], ct_np[:, :, iw], dose_np[:, :, iw], f"Sagittal @ W={iw}")

    # --- Row 2: fluence / rad-depth / central-axis dose profile
    if fluence_np is not None:
        axes[1, 0].imshow(fluence_np.squeeze(), cmap="viridis", aspect="auto")
        axes[1, 0].set_title("Fluence map")
    else:
        axes[1, 0].set_axis_off()
        axes[1, 0].set_title("Fluence map (n/a)")

    if rad_np is not None:
        # rad_depth_bev: [BG, D, H, W] or [D, H, W]
        if rad_np.ndim == 4:
            rad_np = rad_np[0]
        axes[1, 1].imshow(rad_np[:, rad_np.shape[1] // 2, :], cmap="magma", aspect="auto")
        axes[1, 1].set_title("Rad-depth BEV (mid H slice)")
    else:
        axes[1, 1].set_axis_off()
        axes[1, 1].set_title("Rad-depth (n/a)")

    # Dose along CT D axis at iso (probe down the patient), independent of
    # gantry angle — a quick sanity check of dose magnitude/attenuation.
    ry = dose_grid_spacing[1]
    axes[1, 2].plot(np.arange(D) * ry, dose_np[ih, :, iw], label="Through iso in CT")
    axes[1, 2].set_xlabel("CT depth [mm]")
    axes[1, 2].set_ylabel("Dose")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_title("Dose profile through iso")
    axes[1, 2].legend(fontsize=8)

    gantry_deg = float(gantry_angle_rad) * 180.0 / np.pi
    header = f"Beam {beam_index}  |  gantry={gantry_deg:.1f}°"
    if mu is not None:
        header += f"  |  MU={mu:.3g}"
    if title:
        header = f"{title}\n{header}"
    fig.suptitle(header, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_total_dose_debug(
    out_path: str | Path,
    *,
    ct: torch.Tensor,
    dose: torch.Tensor,
    iso_center: tuple[float, float, float],
    dose_grid_spacing: tuple[float, float, float],
    title: str | None = None,
) -> None:
    """Save a 1×3 axial / coronal / sagittal slice overlay through iso."""
    ct_np = _to_numpy(ct).squeeze()
    dose_np = _to_numpy(dose).squeeze()
    ih, id_, iw = _pick_iso_voxel(ct_np.shape, iso_center, dose_grid_spacing)
    dose_max = float(np.nanmax(dose_np)) if np.isfinite(dose_np).any() else 0.0
    dose_max = dose_max if dose_max > 0 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (ct_slice, dose_slice, label) in zip(
        axes,
        [
            (ct_np[ih, :, :], dose_np[ih, :, :], f"Axial @ H={ih}"),
            (ct_np[:, id_, :], dose_np[:, id_, :], f"Coronal @ D={id_}"),
            (ct_np[:, :, iw], dose_np[:, :, iw], f"Sagittal @ W={iw}"),
        ],
    ):
        ax.imshow(ct_slice, cmap="gray", aspect="auto")
        ax.imshow(
            np.ma.masked_less(dose_slice, dose_max * 0.01),
            cmap="turbo", alpha=0.5, aspect="auto",
            vmin=0, vmax=dose_max,
        )
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
