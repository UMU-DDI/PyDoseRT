"""
DepthInterpolatedDoseLayer — per-voxel linear interpolation of pre-convolved
fluence volumes at a handful of reference radiological depths.

Given ``N`` fluence volumes that were produced by convolving a single fluence
with pencil-beam kernels computed at ``N`` fixed reference radiological depths,
plus a per-voxel radiological-depth volume from
:class:`VolumetricRadiologicalDepthLayer`, this layer linearly interpolates a
per-voxel dose estimate: the voxel's own radiological depth decides where it
lies between adjacent reference depths, and the two bracketing convolved
fluences are blended with that weight.

This is the companion layer to ``VolumetricRadiologicalDepthLayer`` in the
finite-size pencil-beam engine with 3D density correction: instead of one
kernel per BEV depth slice evaluated at a single central-axis depth, we keep a
small fixed set of kernels and let the per-voxel depth pick the right mix.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DepthInterpolatedDoseLayer(nn.Module):
    """Linear-interpolate pre-convolved fluence volumes using per-voxel depth.

    Attributes:
        reference_depths (Tensor buffer): Monotonically increasing reference
            radiological depths in mm, shape ``[N]``. Voxel depths outside this
            range are clamped to the closest endpoint.
    """

    def __init__(
        self,
        reference_depths: torch.Tensor | list[float],
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.dtype = dtype
        self.verbose = verbose

        if not isinstance(reference_depths, torch.Tensor):
            reference_depths = torch.tensor(reference_depths)
        reference_depths = reference_depths.to(device=device, dtype=dtype)
        if reference_depths.ndim != 1 or reference_depths.numel() < 2:
            raise ValueError(
                "reference_depths must be a 1-D tensor with at least 2 entries, "
                f"got shape {tuple(reference_depths.shape)}"
            )
        if not torch.all(reference_depths[1:] > reference_depths[:-1]):
            raise ValueError("reference_depths must be strictly increasing.")

        self.register_buffer("reference_depths", reference_depths)

    def forward(
        self,
        convolved_fluences: torch.Tensor,
        rad_depth: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate pre-convolved fluences at each voxel's radiological depth.

        Args:
            convolved_fluences: ``[BG, N, D, H, W]`` — fluence volume convolved
                with the kernel at each reference depth.
            rad_depth: ``[BG, D, H, W]`` — per-voxel radiological depth in mm
                (typically produced by ``VolumetricRadiologicalDepthLayer``).

        Returns:
            ``[BG, D, H, W, 1]`` interpolated dose volume. The trailing
            singleton channel matches the BEV fluence-volume layout expected by
            the rest of the pipeline (MU scaling, rotation, etc.).
        """
        if convolved_fluences.ndim != 5:
            raise ValueError(
                f"convolved_fluences must be 5-D [BG, N, D, H, W], got shape "
                f"{tuple(convolved_fluences.shape)}"
            )
        BG, N, D, H, W = convolved_fluences.shape
        if self.reference_depths.numel() != N:
            raise ValueError(
                f"reference_depths has {self.reference_depths.numel()} entries "
                f"but convolved_fluences has {N} depth channels."
            )
        if rad_depth.shape != (BG, D, H, W):
            raise ValueError(
                f"rad_depth shape mismatch: expected {(BG, D, H, W)}, got "
                f"{tuple(rad_depth.shape)}"
            )

        ref = self.reference_depths.to(convolved_fluences.dtype)

        # Clamp voxel depths to [ref[0], ref[-1]] so we never extrapolate.
        d = rad_depth.to(convolved_fluences.dtype).clamp(min=ref[0], max=ref[-1])

        # searchsorted with right=True returns the first index where ref > d,
        # so idx-1 is the lower bracket for every voxel. Clamp to a valid pair.
        idx = torch.searchsorted(ref, d, right=True).clamp(1, N - 1)
        idx_lo = (idx - 1).unsqueeze(1)  # [BG, 1, D, H, W]
        idx_hi = idx.unsqueeze(1)

        d_lo = ref[idx_lo.squeeze(1)]
        d_hi = ref[idx_hi.squeeze(1)]
        # Denominator is strictly positive because ref is strictly increasing,
        # but guard with a tiny epsilon for fp16 safety.
        t = (d - d_lo) / (d_hi - d_lo).clamp(min=1e-6)

        val_lo = torch.gather(convolved_fluences, 1, idx_lo).squeeze(1)
        val_hi = torch.gather(convolved_fluences, 1, idx_hi).squeeze(1)

        dose = val_lo + (val_hi - val_lo) * t  # [BG, D, H, W]
        return dose.unsqueeze(-1)
