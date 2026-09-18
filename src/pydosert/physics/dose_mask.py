"""Where ion dose is scored.

Internal air (trachea, bowel gas, sinuses) carries dose; the air around the
patient does not. Both sit at ~0.0012 g/cm^3, so the split is topological, not a
density threshold.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "DEFAULT_BODY_DENSITY_THRESHOLD_G_CM3",
    "patient_dose_mask",
]

#: Mass density (g/cm^3) below which a voxel is treated as air.
DEFAULT_BODY_DENSITY_THRESHOLD_G_CM3 = 0.03



def patient_dose_mask(
    mass_density_g_cm3: torch.Tensor,
    density_threshold_g_cm3: float = DEFAULT_BODY_DENSITY_THRESHOLD_G_CM3,
) -> torch.Tensor:
    """Everything except the air open to the outside world.

    External air is the *largest* sub-threshold component touching the border --
    largest, because an internal cavity can touch it too (a trachea cut by the
    first slice). Cavities that vent to the outside stay connected to it, so the
    per-slice body contour is unioned in as well.

    Falls back to the plain threshold when that component holds less than half
    the sub-threshold voxels, i.e. the volume is not a patient surrounded by air.

    Args:
        mass_density_g_cm3: Mass density volume, ``[H, D, W]`` or ``[1, H, D, W]``.
        density_threshold_g_cm3: Density in g/cm^3 below which a voxel is air.

    Returns:
        A boolean mask shaped like the input, on its device.
    """
    from scipy import ndimage

    volume, restore = _as_3d(mass_density_g_cm3)
    threshold = float(density_threshold_g_cm3)

    sub = (volume <= threshold).detach().cpu().numpy()
    if not sub.any():
        return restore(torch.ones_like(volume, dtype=torch.bool))

    labels, _ = ndimage.label(sub)
    border = np.zeros_like(sub, dtype=bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    border[:, :, 0] = border[:, :, -1] = True
    border_labels = labels[border & sub]
    if border_labels.size == 0:
        return restore(torch.ones_like(volume, dtype=torch.bool))

    ids, counts = np.unique(border_labels, return_counts=True)
    external = labels == ids[int(np.argmax(counts))]
    if external.sum() < 0.5 * sub.sum():
        return restore(volume > threshold)

    body = _body_contour(volume, threshold)
    return restore(torch.from_numpy(~external | body).to(device=volume.device))


def _body_contour(
    mass_density_g_cm3: torch.Tensor,
    density_threshold_g_cm3: float,
    min_component_voxels: int = 5000,
) -> np.ndarray:
    """Patient exterior, as a per-axial-slice hole fill of the above-threshold region.

    Per-slice, not 3D: a 3D fill cannot close a lumen open at a z face, while
    in-plane the same lumen is enclosed by tissue. Components smaller than
    ``min_component_voxels`` are dropped as noise; a limb detached by the FOV
    survives as its own component.
    """
    from scipy import ndimage

    solid = (mass_density_g_cm3 > float(density_threshold_g_cm3)).detach().cpu().numpy()
    filled = np.stack([ndimage.binary_fill_holes(s) for s in solid])
    labels, n = ndimage.label(filled)
    if n == 0:
        return filled
    sizes = ndimage.sum(filled, labels, range(1, n + 1))
    keep = [i + 1 for i, size in enumerate(sizes) if size >= min_component_voxels]
    return np.isin(labels, keep)


def _as_3d(volume: torch.Tensor):
    """Accept ``[H, D, W]`` or ``[1, H, D, W]``; return the 3-D view and a restorer."""
    if volume.ndim == 3:
        return volume, lambda mask: mask
    if volume.ndim == 4 and volume.shape[0] == 1:
        return volume[0], lambda mask: mask.unsqueeze(0)
    raise ValueError(
        f"mass density must be [H, D, W] or [1, H, D, W], got {tuple(volume.shape)}"
    )
