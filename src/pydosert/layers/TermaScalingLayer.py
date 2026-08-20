"""Differentiable TERMA source scaling for the photon pencil-beam engines.

The correction follows Laakkonen, Fan and Harju (Med Phys. 2023): the water
fluence source is scaled locally as a function of effective field size and
smoothed relative density *before* pencil-beam convolution.  This module owns
no beam geometry; it only maps a fluence map and BEV density to a scale volume.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _odd_kernel(size_mm: float, spacing_mm: float) -> int:
    """Nearest positive odd number of voxels spanning ``size_mm``."""
    if spacing_mm <= 0:
        raise ValueError(f"voxel spacing must be positive, got {spacing_mm}")
    n = max(1, int(round(float(size_mm) / float(spacing_mm))))
    if n % 2 == 0:
        # Choose the closer odd integer. At an exact tie, prefer the larger
        # support so a requested 10 mm window is not systematically narrowed.
        lo, hi = max(1, n - 1), n + 1
        n = hi if abs(hi * spacing_mm - size_mm) <= abs(lo * spacing_mm - size_mm) else lo
    return n


def _logit(x: float) -> float:
    eps = 1e-6
    x = min(max(float(x), eps), 1.0 - eps)
    return math.log(x / (1.0 - x))


def _inverse_softplus(x: float) -> float:
    x = max(float(x), 1e-8)
    return x + math.log(-math.expm1(-x))


class TermaScalingLayer(nn.Module):
    """Compute the field-size-dependent TERMA scaling volume.

    ``spacing_mm`` follows the patient-grid convention ``(rH, rD, rW)``. The BEV density order is ``[B,G,D,H,W]``, hence the
    pooling kernel order is ``(rD,rH,rW)``.

    When ``learnable=True``, unconstrained raw parameters are transformed so
    ``0 < c1 < 1`` and ``c2 >= 0``. This permits global data calibration while
    retaining physically meaningful coefficients.
    """

    def __init__(
        self,
        c1: float,
        c2_per_mm: float,
        spacing_mm: tuple[float, float, float],
        *,
        fluence_pixel_size_mm: float = 1.0,
        smoothing_size_mm: float = 10.0,
        density_water_low: float = 0.95,
        density_water_high: float = 1.05,
        detach_field_size: bool = False,
        learnable: bool = False,
    ):
        super().__init__()
        if not 0.0 <= c1 <= 1.0:
            raise ValueError(f"c1 must lie in [0,1], got {c1}")
        if c2_per_mm < 0.0:
            raise ValueError(f"c2_per_mm must be non-negative, got {c2_per_mm}")
        if fluence_pixel_size_mm <= 0.0:
            raise ValueError("fluence_pixel_size_mm must be positive")
        if not density_water_low < density_water_high:
            raise ValueError("density_water_low must be below density_water_high")

        self.learnable = bool(learnable)
        if self.learnable:
            self.raw_c1 = nn.Parameter(torch.tensor(_logit(c1), dtype=torch.float32))
            self.raw_c2 = nn.Parameter(torch.tensor(_inverse_softplus(c2_per_mm), dtype=torch.float32))
        else:
            self.register_buffer("fixed_c1", torch.tensor(float(c1), dtype=torch.float32))
            self.register_buffer("fixed_c2", torch.tensor(float(c2_per_mm), dtype=torch.float32))

        r_h, r_d, r_w = (float(x) for x in spacing_mm)
        self.smoothing_kernel = (
            _odd_kernel(smoothing_size_mm, r_d),
            _odd_kernel(smoothing_size_mm, r_h),
            _odd_kernel(smoothing_size_mm, r_w),
        )
        self.fluence_pixel_size_mm = float(fluence_pixel_size_mm)
        self.density_water_low = float(density_water_low)
        self.density_water_high = float(density_water_high)
        self.detach_field_size = bool(detach_field_size)
        self.last_field_size_mm: torch.Tensor | None = None

    @property
    def c1(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_c1) if self.learnable else self.fixed_c1

    @property
    def c2_per_mm(self) -> torch.Tensor:
        return F.softplus(self.raw_c2) if self.learnable else self.fixed_c2

    def effective_field_size(self, fluence_maps: torch.Tensor) -> torch.Tensor:
        """Equivalent-square side length in mm, one value per fluence map."""
        if fluence_maps.ndim != 3:
            raise ValueError(
                f"fluence_maps must be [B*G,H,W], got shape {tuple(fluence_maps.shape)}")
        psi = fluence_maps.clamp_min(0.0)
        peak = psi.amax(dim=(-2, -1))
        integral = psi.sum(dim=(-2, -1))
        equivalent_pixels = integral / peak.clamp_min(torch.finfo(psi.dtype).eps)
        length = self.fluence_pixel_size_mm * torch.sqrt(equivalent_pixels.clamp_min(0.0))
        length = torch.where(peak > 0.0, length, torch.zeros_like(length))
        return length.detach() if self.detach_field_size else length

    def smooth_density(self, bev_density: torch.Tensor) -> torch.Tensor:
        if bev_density.ndim != 5:
            raise ValueError(
                f"bev_density must be [B,G,D,H,W], got shape {tuple(bev_density.shape)}")
        b, g, d, h, w = bev_density.shape
        x = bev_density.reshape(b * g, 1, d, h, w)
        kd, kh, kw = self.smoothing_kernel
        padded = F.pad(
            x,
            (kw // 2, kw // 2, kh // 2, kh // 2, kd // 2, kd // 2),
            mode="replicate",
        )
        average = F.avg_pool3d(padded, kernel_size=(kd, kh, kw), stride=1)
        outside_water = (x < self.density_water_low) | (x > self.density_water_high)
        return torch.where(outside_water, average, x).reshape(b, g, d, h, w)

    def forward(self, fluence_maps: torch.Tensor, bev_density: torch.Tensor) -> torch.Tensor:
        b, g = bev_density.shape[:2]
        if fluence_maps.shape[0] != b * g:
            raise ValueError(
                f"fluence/density batch mismatch: {fluence_maps.shape[0]} maps for B={b}, G={g}")

        length = self.effective_field_size(fluence_maps)
        self.last_field_size_mm = length.detach()
        rho = self.smooth_density(bev_density).reshape(b * g, *bev_density.shape[-3:])
        length = length.to(device=rho.device, dtype=rho.dtype).view(-1, 1, 1, 1)
        c1 = self.c1.to(device=rho.device, dtype=rho.dtype)
        c2 = self.c2_per_mm.to(device=rho.device, dtype=rho.dtype)

        low = 1.0 - (1.0 - rho) * (
            c1 * torch.exp(-c2 * (rho + 0.05) * length) + (1.0 - c1))
        high = 1.0 + (rho - 1.0) * (c1 * torch.exp(-c2 * rho * length))
        scale = torch.where(
            rho < self.density_water_low,
            low,
            torch.where(rho > self.density_water_high, high, torch.ones_like(rho)),
        )
        zero_fluence = (length == 0.0).expand_as(scale)
        scale = torch.where(zero_fluence, torch.ones_like(scale), scale)
        if not bool(torch.isfinite(scale).all()):
            raise FloatingPointError(
                "TERMA scaling produced non-finite values; check density and coefficients")
        return scale.unsqueeze(-1)

