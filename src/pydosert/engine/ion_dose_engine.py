"""Differentiable beam's-eye-view lattice pencil-beam engine for ion beams.

The engine computes the dose of a batch of ion beamlets (one gantry angle, one
energy, one spot each) on a *beam's-eye view* (BEV) lattice cropped around every
beamlet, and rotates the result back into the patient frame.

Pipeline
--------
1. **BEV resampling.** The patient density (stopping-power ratio) volume is
   resampled into each beamlet's BEV window with
   :func:`~pydosert.geometry.bev.build_bev_sampling_grid`, and integrated along
   depth into a per-ray water-equivalent depth (WEQ).
2. **Lattice pencil beam.** Each beamlet is split into ``n_sub_beams_per_dim**2``
   sub-beams on a quarter-FWHM grid. Every sub-beam looks its own WEQ column up
   in the kernel table, giving a per-sub-beam integrated depth dose and lateral
   sigma; the narrow core is laid down per sub-beam and the broad nuclear halo
   once per beamlet (see :meth:`IonDoseEngine.compute_layer_edep`).
3. **Correction hook.** The complete BEV payload is handed to
   ``bev_correction`` -- the injection point for a learned residual model -- which
   may return a corrected payload.
4. **Finalisation.** The corrected BEV energy deposition is rotated into the
   patient frame with :func:`~pydosert.geometry.bev.build_patient_sampling_grid`
   and converted from MeV to Gy.

Everything is plain differentiable torch: gradients flow to the beamlet weights,
positions, energies and sigmas, and to whatever the correction hook holds. The
engine never disables autograd itself -- wrap a call in :func:`torch.no_grad` if
you do not want a graph.

Known artifacts
---------------
Two approximations are deliberate, are not bugs, and are documented here because
a reader will otherwise find them by measuring:

1. **The distal pedestal.** Depth queries past the tabulated range use
   ``beyond_range="edge"``, so the kernel table's last tabulated IDD value is
   held constant for every depth beyond the proton range. The engine therefore
   lays down a small constant tail distal of the Bragg peak: 0.12 % of peak at
   41 MeV and 1.09 % at 114 MeV. That is not physical -- there is no dose past
   the range -- but it is the behaviour the commissioned corrections were fitted
   against, and it is pinned bit-for-bit by the numeric regression set.
   :class:`~pydosert.physics.kernels.ion_kernel_table.IonKernelTable` also
   offers ``beyond_range="zero"`` (the physical answer); switching the engine
   over is a deliberate re-commissioning, not a bug fix, so the default stays
   ``"edge"``.
2. **The result is dose to water.** :meth:`IonDoseEngine._convert_mev_to_gy`
   applies one MeV-to-Gy factor everywhere; the table's IDD is in MeV cm^2/g,
   already mass-normalised, so no density term is missing from that step.
   Dose to medium is a per-voxel multiply by the mass stopping-power ratio
   ``(S/rho)_med / (S/rho)_w``, which is ``spr / rho`` -- the engine is handed
   the SPR volume, so only the mass density is missing. The factor is within
   0.2 % for soft tissue but reaches -12.6 % in cortical bone (Geant4 DoseRAD
   material table, 150 MeV), so it is not a detail.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from pydosert.data.ion_beam import IonBeamletBatch
from pydosert.data.ion_machine import IonMachineConfig
from pydosert.geometry.bev import (
    BevCrop,
    build_bev_crop,
    build_bev_sampling_grid,
    build_patient_sampling_grid,
)
from pydosert.physics.dose_mask import (
    DEFAULT_BODY_DENSITY_THRESHOLD_G_CM3,
    patient_dose_mask,
)
from pydosert.physics.ion_scattering import fermi_eyges_excess
from pydosert.physics.kernels.ion_kernel_table import IonKernelTable

__all__ = [
    "DEFAULT_BODY_DENSITY_THRESHOLD_G_CM3",
    "BeamletDose",
    "IonDoseEngine",
    "patient_dose_mask",
]

#: MeV cm^2 g^-1 -> Gy mm^2. Converts an energy fluence (MeV per unit area)
#: into absorbed dose.
MEV_CM2_PER_G_TO_GY_MM2 = 1.6021766208e-08

#: FWHM of a Gaussian in units of its sigma.
_FWHM_PER_SIGMA = 2.354820045

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


# --------------------------------------------------------------------------- helpers


def _resolve_device(device: torch.device | str) -> torch.device:
    """Resolve a device spec to a concrete device, index included.

    ``torch.device("cuda")`` carries no index and compares unequal to the
    ``cuda:0`` a tensor reports, so a caller passing ``device="cuda"`` alongside a
    table already on the GPU would be told they disagree.
    """
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def _gauss_cell_1d(
    x_mm: torch.Tensor,
    sigma_mm: torch.Tensor,
    half_width_mm: float,
) -> torch.Tensor:
    """Integral of a unit-area 1-D Gaussian over one voxel.

    Args:
        x_mm: Cell-centre offset from the beam axis, mm.
        sigma_mm: Gaussian sigma, mm.
        half_width_mm: Half the cell width, mm.

    Returns:
        The fraction of the Gaussian falling inside the cell.
    """
    inv = _INV_SQRT2 / sigma_mm
    return 0.5 * (torch.erf((x_mm + half_width_mm) * inv) - torch.erf((x_mm - half_width_mm) * inv))


@dataclass(frozen=True)
class BeamletDose:
    """One beamlet's patient-frame dose, cropped to its non-zero support.

    Attributes:
        dose: ``[1, dz, dy, dx]`` dose in Gy.
        offset: ``(z, y, x)`` origin of :attr:`dose` in the patient dose grid,
            where ``(z, y, x)`` indexes ``(H, D, W)``.
        full_shape: ``(H, D, W)`` of the full patient dose grid.
    """

    dose: torch.Tensor
    offset: tuple[int, int, int]
    full_shape: tuple[int, int, int]


@dataclass(frozen=True)
class _BevLattice:
    """The BEV crop lattice shared by every beamlet of one call."""

    #: ``(H,)`` row indices of the crop, as floats on the engine device/dtype.
    h_coords: torch.Tensor
    #: ``(W,)`` column indices of the crop.
    w_coords: torch.Tensor
    #: Voxel spacing along the crop's ``H`` axis, mm.
    res_h: float
    #: Voxel spacing along the crop's ``W`` axis, mm.
    res_w: float
    #: Voxel spacing along the depth axis, mm.
    res_d: float

    @property
    def height(self) -> int:
        """Crop height in voxels."""
        return int(self.h_coords.shape[0])

    @property
    def width(self) -> int:
        """Crop width in voxels."""
        return int(self.w_coords.shape[0])


@dataclass(frozen=True)
class _BeamletContext:
    """Everything :meth:`IonDoseEngine.compute_layer_edep` needs for one beamlet.

    The reference implementation passed these as twenty positional arguments.
    """

    #: 0-d beamlet energy, MeV, on the engine device/dtype (may require grad).
    energy_mev: torch.Tensor
    #: The scalar value of :attr:`energy_mev`, to skip a device synchronisation.
    energy_value: float
    #: 0-d initial spot sigma along the crop ``W`` axis, mm.
    sigma_x_mm: torch.Tensor
    #: 0-d initial spot sigma along the crop ``H`` axis, mm.
    sigma_y_mm: torch.Tensor
    #: 0-d beamlet weight (particles).
    weight: torch.Tensor
    #: ``(D, H, W)`` water-equivalent depth of this beamlet's crop, mm.
    weq_bev: torch.Tensor
    #: 0-d radiological-depth offset of this beamlet, mm.
    depth_offset_mm: torch.Tensor
    #: 0-d beamlet centre along the crop ``H`` axis, in crop voxels.
    center_h: torch.Tensor
    #: 0-d beamlet centre along the crop ``W`` axis, in crop voxels.
    center_w: torch.Tensor
    #: ``(1, H, W)`` bool: crop cells that lie inside the dose grid.
    active: torch.Tensor


# --------------------------------------------------------------------------- engine


class IonDoseEngine(nn.Module):
    """BEV lattice pencil-beam dose engine for ion beamlets.

    Args:
        machine_config: Nozzle geometry; used only to turn an SSD into a
            radiological-depth offset.
        kernel_table: Commissioned pencil-beam base data.
        dose_grid_spacing: ``(rh, rd, rw)`` voxel spacing in mm, aligned with
            ``(H, D, W)``.
        dose_grid_shape: ``(H, D, W)`` dose-grid shape in voxels; ``D`` is the
            beam-depth axis at gantry angle 0.
        field_size: ``(field_h, field_w)`` size in voxels of the BEV window
            computed around each beamlet. Required, and must fit inside the dose
            grid -- otherwise the halo truncation would depend on the grid
            rather than on the request.
        lateral_model: ``"gauss_double"`` (narrow core plus nuclear halo, the
            commissioned model) or ``"gauss"`` (single Gaussian, no halo).
        heterogeneous_mcs: Add the Fermi-Eyges geometric lever-arm excess of
            :func:`~pydosert.physics.ion_scattering.fermi_eyges_excess` to the
            narrow-core sigma. Identically zero in water, so leaving it on costs
            only time on homogeneous phantoms.
        n_sub_beams_per_dim: Beamlet splitting: ``n**2`` sub-beams on a
            quarter-FWHM grid. This is an accuracy/speed knob, not a formality
            -- at ``n = 3`` the sub-beam sum under-samples the envelope and the
            core over-peaks by ~45 % against ``n = 9``.
        bev_correction: Optional module invoked on the BEV payload just before
            it is rotated into the patient frame; it receives the payload dict
            and the keyword ``engine=self`` and returns a payload. This is the
            documented injection point for a learned correction model. ``None``
            leaves the payload untouched.
        device: Device to compute on. Defaults to the kernel table's.
        dtype: Floating dtype to compute in. Defaults to the kernel table's.

    Raises:
        ValueError: If any geometry, the field size or ``lateral_model`` is
            invalid, or if the kernel table is not on the engine's
            device/dtype.
    """

    def __init__(
        self,
        machine_config: IonMachineConfig,
        kernel_table: IonKernelTable,
        dose_grid_spacing: Sequence[float],
        dose_grid_shape: Sequence[int],
        field_size: Sequence[int],
        *,
        lateral_model: str = "gauss_double",
        heterogeneous_mcs: bool = True,
        n_sub_beams_per_dim: int = 9,
        bev_correction: nn.Module | None = None,
        beamlet_chunk_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if len(dose_grid_shape) != 3:
            raise ValueError(f"dose_grid_shape must be (H, D, W), got {tuple(dose_grid_shape)}")
        if len(dose_grid_spacing) != 3:
            raise ValueError(f"dose_grid_spacing must be (rh, rd, rw), got {tuple(dose_grid_spacing)}")
        shape_h, shape_d, shape_w = (int(v) for v in dose_grid_shape)
        shape: tuple[int, int, int] = (shape_h, shape_d, shape_w)
        if shape_h < 1 or shape_d < 1 or shape_w < 1:
            raise ValueError(f"dose_grid_shape must be positive, got {shape}")
        res_h, res_d, res_w = (float(v) for v in dose_grid_spacing)
        spacing: tuple[float, float, float] = (res_h, res_d, res_w)
        if res_h <= 0.0 or res_d <= 0.0 or res_w <= 0.0:
            raise ValueError(f"dose_grid_spacing must be strictly positive, got {spacing}")
        if lateral_model not in {"gauss", "gauss_double"}:
            raise ValueError(
                f"lateral_model must be 'gauss' or 'gauss_double', got {lateral_model!r}"
            )
        if int(n_sub_beams_per_dim) < 1:
            raise ValueError(f"n_sub_beams_per_dim must be >= 1, got {n_sub_beams_per_dim}")

        self.machine_config = machine_config
        self.kernel_table = kernel_table
        self.dose_grid_shape = shape
        self.dose_grid_spacing = spacing
        self.device = kernel_table.device if device is None else _resolve_device(device)
        self.dtype = kernel_table.dtype if dtype is None else dtype
        self.lateral_model = lateral_model
        self.heterogeneous_mcs = bool(heterogeneous_mcs)
        self.n_sub_beams_per_dim = int(n_sub_beams_per_dim)
        self.beamlet_chunk_size = None if beamlet_chunk_size is None else max(1, int(beamlet_chunk_size))
        self.field_size = self._check_field_size(field_size)
        # The correction injection point. Registered as a plain submodule so its
        # parameters are part of the engine's state_dict and move with .to().
        self.bev_correction = bev_correction

        if kernel_table.device != self.device or kernel_table.dtype != self.dtype:
            raise ValueError(
                f"kernel_table is on {kernel_table.device}/{kernel_table.dtype} but the engine "
                f"computes on {self.device}/{self.dtype}; move it with IonKernelTable.to()"
            )

        self._grid_cache_key: tuple | None = None
        self._grid_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def _check_field_size(self, field_size: Sequence[int]) -> tuple[int, int]:
        """Validate ``field_size`` against the dose grid and narrow it to a pair.

        The reference engine clamped an oversized field into the grid silently.
        A field larger than the grid is a caller error -- it changes how much of
        the nuclear halo is kept -- so it raises here.

        Args:
            field_size: ``(field_h, field_w)`` in voxels.

        Returns:
            The validated ``(field_h, field_w)``.

        Raises:
            ValueError: If it is not a positive pair that fits inside the grid.
        """
        if len(field_size) != 2:
            raise ValueError(f"field_size must be (field_h, field_w), got {tuple(field_size)}")
        field_h, field_w = (int(v) for v in field_size)
        full_h, _full_d, full_w = self.dose_grid_shape
        if field_h < 1 or field_w < 1:
            raise ValueError(f"field_size must be positive, got {(field_h, field_w)}")
        if field_h > full_h or field_w > full_w:
            raise ValueError(
                f"field_size {(field_h, field_w)} does not fit inside the dose grid "
                f"{(full_h, full_w)}; shrink the field or enlarge the grid"
            )
        return field_h, field_w

    # ----------------------------------------------------------- geometry

    def _sampling_grids(self, beamlets: IonBeamletBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``(patient -> BEV, BEV -> patient)`` sampling grids of this batch.

        Both are pure functions of the grid geometry, the gantry angles and the
        isocentres, so they are cached on exactly those; a batch with different
        angles rebuilds them.
        """
        angles = beamlets.gantry_angle_rad.to(device=self.device, dtype=self.dtype)
        iso_centers = beamlets.iso_center_mm.to(device=self.device, dtype=self.dtype)
        if angles.requires_grad or iso_centers.requires_grad:
            # A cached grid would carry a stale autograd graph into the next call.
            return self._build_sampling_grids(angles, iso_centers)
        key = (
            self.dose_grid_shape,
            self.dose_grid_spacing,
            str(self.device),
            self.dtype,
            tuple(angles.detach().flatten().tolist()),
            tuple(iso_centers.detach().flatten().tolist()),
        )
        if self._grid_cache_key != key or self._grid_cache is None:
            self._grid_cache = self._build_sampling_grids(angles, iso_centers)
            self._grid_cache_key = key
        return self._grid_cache

    def _build_sampling_grids(
        self,
        angles: torch.Tensor,
        iso_centers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the two sampling grids of one beam geometry, uncached."""
        return (
            build_bev_sampling_grid(
                self.dose_grid_shape,
                self.dose_grid_spacing,
                angles,
                iso_centers,
                device=self.device,
                dtype=self.dtype,
            ),
            build_patient_sampling_grid(
                self.dose_grid_shape,
                self.dose_grid_spacing,
                angles,
                iso_centers,
                device=self.device,
                dtype=self.dtype,
            ),
        )

    def _build_crops(self, center_h_vox: torch.Tensor, center_w_vox: torch.Tensor) -> list[BevCrop]:
        """One :class:`BevCrop` per beamlet, centred on its spot."""
        full_h, _full_d, full_w = self.dose_grid_shape
        field_h, field_w = self.field_size
        return [
            build_bev_crop(
                center_h=float(center_h_vox[g].detach()),
                center_w=float(center_w_vox[g].detach()),
                size_h=field_h,
                size_w=field_w,
                full_shape_hw=(full_h, full_w),
            )
            for g in range(int(center_h_vox.shape[0]))
        ]

    @staticmethod
    def _sample(volume: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """The one ``grid_sample`` configuration this engine uses, everywhere."""
        return F.grid_sample(
            volume,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    def _density_and_weq_bev(
        self,
        density_image: torch.Tensor,
        crops: list[BevCrop],
        bev_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resample the patient density into every beamlet's BEV crop.

        Args:
            density_image: ``[1, H, D, W]`` stopping-power-ratio volume.
            crops: One crop per beamlet; all must share ``shape_hw``.
            bev_grid: ``[G, D, W, 2]`` patient-to-BEV sampling grid.

        Returns:
            ``(density_bev, weq_bev)``, both ``[G, D, h, w]``. ``weq_bev`` is the
            water-equivalent depth at the *centre* of each depth step.
        """
        batch, _full_h, num_depths, full_w = density_image.shape
        num_beamlets = len(crops)
        if batch != 1:
            raise ValueError("the ion engine computes one patient at a time (B == 1)")
        out_h, out_w = crops[0].shape_hw
        density_bev = density_image.new_zeros((batch, num_beamlets, num_depths, out_h, out_w))

        for g_idx, crop in enumerate(crops):
            if tuple(crop.shape_hw) != (out_h, out_w):
                raise ValueError("all BEV crops of one call must share shape_hw")
            if crop.is_empty:
                continue
            h_src, w_src = crop.h_src, crop.w_src
            src_h = h_src.stop - h_src.start

            density_src = density_image[:, h_src, :, :]
            density_flat = density_src.reshape(batch * src_h, 1, num_depths, full_w)
            grid_g = bev_grid[g_idx][:, w_src].to(
                device=density_image.device,
                dtype=density_image.dtype,
            )
            grid = grid_g.unsqueeze(0).expand(batch * src_h, -1, -1, -1)
            sampled = self._sample(density_flat, grid)
            sampled = sampled.reshape(batch, src_h, num_depths, w_src.stop - w_src.start)
            sampled = sampled.permute(0, 2, 1, 3).contiguous()
            density_bev[:, g_idx, :, crop.h_dst, crop.w_dst] = sampled

        segment_weq = density_bev.clamp_min(0.0) * self.dose_grid_spacing[1]
        boundary_end = torch.cumsum(segment_weq, dim=2)
        weq_bev = boundary_end - 0.5 * segment_weq
        return (
            density_bev.reshape(batch * num_beamlets, num_depths, out_h, out_w),
            weq_bev.reshape(batch * num_beamlets, num_depths, out_h, out_w),
        )

    def _resolve_depth_offset(
        self,
        beamlets: IonBeamletBatch,
        ssd_mm: torch.Tensor | float | None,
    ) -> torch.Tensor:
        """Per-beamlet radiological-depth offset in mm.

        The commissioned kernels were fitted with a fixed air gap, so a beam
        whose skin is closer to (or further from) the nozzle than that fit needs
        its depth scale shifted by the extra air::

            offset = 0.0011 * ((ssd + bams_to_iso) - sad - fit_air_offset)

        Args:
            beamlets: The batch, for its ``sad_mm``.
            ssd_mm: Source-to-skin distance, scalar or ``(G,)``. ``None`` means
                no offset at all (the depth scale starts at the grid entry).

        Returns:
            ``(G,)`` offset in mm; all zeros when ``ssd_mm`` is ``None``.
        """
        num_beamlets = len(beamlets)
        if ssd_mm is None:
            return torch.zeros(num_beamlets, device=self.device, dtype=self.dtype)
        ssd_tensor = torch.as_tensor(ssd_mm, device=self.device, dtype=self.dtype)
        if ssd_tensor.ndim == 0:
            ssd_tensor = ssd_tensor.expand(num_beamlets).clone()
        if ssd_tensor.shape != (num_beamlets,):
            raise ValueError(
                f"ssd_mm must be scalar or [{num_beamlets}], got {tuple(ssd_tensor.shape)}"
            )
        sad_tensor = beamlets.sad_mm.to(device=self.device, dtype=self.dtype)
        bams = float(self.machine_config.bams_to_iso_dist_mm)
        fit_air = float(self.machine_config.fit_air_offset_mm)
        nozzle_to_skin = (ssd_tensor + bams) - sad_tensor
        return 0.0011 * (nozzle_to_skin - fit_air)

    # ------------------------------------------------------- lateral kernel

    def _sub_beam_grid(
        self,
        sigma_x_mm: torch.Tensor,
        sigma_y_mm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sub-beam offsets, fluence weights and residual sigma of one beamlet.

        The sub-beams sit on a symmetric ``n x n`` grid of quarter-FWHM steps.
        The Gaussian-splitting variance identity says a superposition of
        sub-beams of width ``sigma_sub``, placed under a weight envelope of
        width ``sigma_env``, has width ``sigma_env^2 + sigma_sub^2``. So with
        ``sigma_sub = sigma_spot / sqrt(n)`` the envelope must be
        ``sqrt(sigma_spot^2 - sigma_sub^2)`` and **not** ``sigma_spot``, which
        would over-broaden the core by ``sqrt(1 + 1/n)`` (worst at low energy).

        Args:
            sigma_x_mm: 0-d initial spot sigma along the crop ``W`` axis.
            sigma_y_mm: 0-d initial spot sigma along the crop ``H`` axis.

        Returns:
            ``(offsets_yx_mm, weights, sub_sigma_xy)`` of shapes ``(S, 2)``,
            ``(S,)`` and ``(2,)``, with ``S = n**2``.
        """
        n_per_dim = self.n_sub_beams_per_dim
        device, dtype = self.device, self.dtype
        fwhm_x = _FWHM_PER_SIGMA * sigma_x_mm
        fwhm_y = _FWHM_PER_SIGMA * sigma_y_mm
        steps = torch.arange(n_per_dim, device=device, dtype=dtype) - (n_per_dim - 1) / 2.0
        offset_x = steps * (fwhm_x / 4.0)
        offset_y = steps * (fwhm_y / 4.0)
        offset_y2, offset_x2 = torch.meshgrid(offset_y, offset_x, indexing="ij")
        offsets_yx = torch.stack([offset_y2.reshape(-1), offset_x2.reshape(-1)], dim=-1)

        sub_sigma_x = (sigma_x_mm / math.sqrt(n_per_dim)).clamp_min(1e-3)
        sub_sigma_y = (sigma_y_mm / math.sqrt(n_per_dim)).clamp_min(1e-3)
        env_x_sq = (sigma_x_mm.square() - sub_sigma_x.square()).clamp_min(torch.finfo(dtype).eps)
        env_y_sq = (sigma_y_mm.square() - sub_sigma_y.square()).clamp_min(torch.finfo(dtype).eps)

        weights = torch.exp(
            -(offset_y2.reshape(-1) ** 2) / (2.0 * env_y_sq)
            - (offset_x2.reshape(-1) ** 2) / (2.0 * env_x_sq)
        )
        weights = weights / weights.sum().clamp_min(torch.finfo(dtype).eps)
        return offsets_yx, weights, torch.stack([sub_sigma_x, sub_sigma_y])

    def compute_layer_edep(self, ctx: _BeamletContext, lattice: _BevLattice) -> torch.Tensor:
        """Deposited energy of one beamlet on its BEV crop, in MeV.

        The beamlet is split into sub-beams (see :meth:`_sub_beam_grid`), each of
        which samples the WEQ column at its own lateral position -- that is what
        makes the result heterogeneity-aware, because neighbouring sub-beams see
        different tissue. Only the **narrow core** is resolved per sub-beam: it
        is sharp and heterogeneity-sensitive. The broad nuclear halo is wide and
        smooth, so it is laid down once per beamlet as a single Gaussian
        convolved with the full initial spot sigma; in homogeneous media that
        reduces exactly to the single-mode double Gaussian.

        Args:
            ctx: This beamlet's context.
            lattice: The crop lattice shared by the whole call.

        Returns:
            ``(D, H, W)`` deposited energy on the beamlet's crop, MeV.
        """
        eps = torch.finfo(self.dtype).eps
        table = self.kernel_table
        energy, energy_value = ctx.energy_mev, ctx.energy_value
        height, width = lattice.height, lattice.width
        res_h, res_w = lattice.res_h, lattice.res_w

        kernel_offset = table.kernel_offset(energy, energy_value_hint=energy_value)
        kernel_offset = torch.as_tensor(kernel_offset, device=self.device, dtype=self.dtype)

        offsets_yx, sub_weights, sub_sigma = self._sub_beam_grid(ctx.sigma_x_mm, ctx.sigma_y_mm)
        num_sub = offsets_yx.shape[0]

        # Sub-beam centres in crop voxel units.
        center_y = ctx.center_h + offsets_yx[:, 0] / res_h
        center_x = ctx.center_w + offsets_yx[:, 1] / res_w

        # Each sub-beam's own WEQ column, (S, D).
        weq_col = ctx.weq_bev
        num_depths = weq_col.shape[0]
        index_y = center_y.round().long().clamp(0, height - 1)
        index_x = center_x.round().long().clamp(0, width - 1)
        weq_s = weq_col[:, index_y, index_x].transpose(0, 1).contiguous()

        kernel_depth_s = (weq_s + ctx.depth_offset_mm - kernel_offset).clamp_min(0.0)
        ray_edep_s = table.edep(
            energy, kernel_depth_s, beyond_range="edge", energy_value_hint=energy_value
        ).clamp_min(0.0)

        # Lateral coordinates per sub-beam: (S, 1, W) and (S, H, 1).
        x_mm = (lattice.w_coords.view(1, 1, width) - center_x.view(num_sub, 1, 1)) * res_w
        y_mm = (lattice.h_coords.view(1, height, 1) - center_y.view(num_sub, 1, 1)) * res_h

        use_double = self.lateral_model == "gauss_double"
        active_b = ctx.active.view(1, 1, height, width)

        if use_double:
            sigma1_s, sigma2_s, halo_weight_s = table.double_gauss(
                energy, kernel_depth_s, beyond_range="edge", energy_value_hint=energy_value
            )
        else:
            sigma1_s = table.sigma(
                energy, kernel_depth_s, beyond_range="edge", energy_value_hint=energy_value
            ).clamp_min(0.0)
            halo_weight_s = torch.zeros_like(sigma1_s)

        if self.heterogeneous_mcs:
            excess = fermi_eyges_excess(kernel_depth_s, energy_value, lattice.res_d)
            sigma1_s = (sigma1_s.square() + excess).clamp_min(1e-12).sqrt()

        # ---- narrow core: per sub-beam, each depth plane normalised to sum 1
        half_x = 0.5 * res_w
        half_y = 0.5 * res_h
        core_x = (sigma1_s.square() + sub_sigma[0].square()).sqrt().clamp_min(1e-6)
        core_y = (sigma1_s.square() + sub_sigma[1].square()).sqrt().clamp_min(1e-6)
        gauss_x = _gauss_cell_1d(x_mm.view(num_sub, 1, 1, width), core_x.view(num_sub, num_depths, 1, 1), half_x)
        gauss_y = _gauss_cell_1d(y_mm.view(num_sub, 1, height, 1), core_y.view(num_sub, num_depths, 1, 1), half_y)
        narrow = gauss_x * gauss_y
        narrow = torch.where(active_b, narrow, torch.zeros_like(narrow))
        narrow = narrow / narrow.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
        core_amplitude = (
            sub_weights.view(num_sub, 1) * ray_edep_s * (1.0 - halo_weight_s)
        ).view(num_sub, num_depths, 1, 1)
        edep = (core_amplitude * narrow).sum(dim=0)

        # ---- broad halo: one IDD-conserving Gaussian over the whole crop
        if use_double:
            halo_amplitude = (sub_weights.view(num_sub, 1) * ray_edep_s * halo_weight_s).sum(dim=0)
            central = num_sub // 2  # the sub-beam at offset 0
            halo_sigma = sigma2_s[central]
            halo_x = (halo_sigma.square() + ctx.sigma_x_mm.square()).sqrt().clamp_min(1e-6)
            halo_y = (halo_sigma.square() + ctx.sigma_y_mm.square()).sqrt().clamp_min(1e-6)
            broad_x_mm = (lattice.w_coords.view(1, width) - ctx.center_w) * res_w
            broad_y_mm = (lattice.h_coords.view(height, 1) - ctx.center_h) * res_h
            broad_x = _gauss_cell_1d(broad_x_mm.view(1, 1, width), halo_x.view(num_depths, 1, 1), half_x)
            broad_y = _gauss_cell_1d(broad_y_mm.view(1, height, 1), halo_y.view(num_depths, 1, 1), half_y)
            broad = broad_x * broad_y
            broad = torch.where(ctx.active, broad, torch.zeros_like(broad))
            broad = broad / broad.sum(dim=(1, 2), keepdim=True).clamp_min(eps)
            edep = edep + halo_amplitude.view(num_depths, 1, 1) * broad

        return ctx.weight * edep

    # ---------------------------------------------------------- finalisation

    def _rotation_inputs(
        self,
        edep_bev: torch.Tensor,
        crop: BevCrop,
        g_idx: int,
        patient_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, slice] | None:
        """Prepare one beamlet for the rotation back into the patient frame.

        Reshapes its cropped BEV energy into ``grid_sample``'s ``(N, 1, D, W)``
        layout and re-normalises the full-grid sampling grid onto the crop's
        lateral window.

        Args:
            edep_bev: ``[1, G, D, h, w]`` BEV deposited energy.
            crop: That beamlet's crop.
            g_idx: Beamlet index.
            patient_grid: ``[G, D, W, 2]`` BEV-to-patient sampling grid.

        Returns:
            ``(volume, grid, h_src)`` ready for :meth:`_sample`, or ``None`` when
            the crop misses the dose grid entirely.
        """
        batch, _num_beamlets, num_depths, _crop_h, crop_w = edep_bev.shape
        full_w = crop.full_shape_hw[1]
        h_src, h_dst = crop.h_src, crop.h_dst
        if crop.h_is_empty:
            return None
        src_h = h_src.stop - h_src.start
        if h_dst.stop - h_dst.start != src_h:
            raise ValueError("BEV crop h_src/h_dst lengths do not match")

        volume = edep_bev[:, g_idx, :, h_dst, :]
        volume = volume.permute(0, 2, 1, 3).contiguous().reshape(batch * src_h, 1, num_depths, crop_w)

        grid_full = patient_grid[g_idx].to(device=edep_bev.device, dtype=edep_bev.dtype)
        full_x = ((grid_full[..., 0] + 1.0) * float(full_w) - 1.0) * 0.5
        crop_x = full_x - float(crop.target_w_start)
        crop_grid_x = (2.0 * (crop_x + 0.5) / float(crop_w)) - 1.0
        grid = torch.stack((crop_grid_x, grid_full[..., 1]), dim=-1)
        grid = grid.unsqueeze(0).expand(batch * src_h, -1, -1, -1)
        return volume, grid, h_src

    def _convert_mev_to_gy(self, deposited_energy_mev: torch.Tensor, dose_mask: torch.Tensor) -> torch.Tensor:
        """Convert deposited energy (MeV per transport step) into dose (Gy).

        The conversion is a single linear factor: the kernel table's integrated
        depth dose is an energy per unit *length*, so dividing the voxel volume
        by the depth step turns it into an energy per unit *area*.

        The same factor is applied at every voxel, whatever the local mass
        density. That is self-consistent -- the table's IDD is in MeV cm^2/g and
        is therefore already mass-normalised, so no density term is missing from
        *this* step -- but it does mean the result is **dose to water** and that
        no dose-to-medium conversion happens anywhere in the engine. See the
        module docstring.

        Args:
            deposited_energy_mev: Patient-frame deposited energy.
            dose_mask: Boolean mask, broadcastable to it, of voxels to score.

        Returns:
            Dose in Gy, zero wherever ``dose_mask`` is False.
        """
        dose_gy = deposited_energy_mev * (MEV_CM2_PER_G_TO_GY_MM2 / self._lateral_area_mm2())
        return torch.where(dose_mask, dose_gy, torch.zeros_like(dose_gy))

    def _lateral_area_mm2(self) -> float:
        """Voxel cross-section perpendicular to the beam, mm^2."""
        res_h, res_d, res_w = self.dose_grid_spacing
        voxel_volume_mm3 = float(res_h) * float(res_d) * float(res_w)
        return voxel_volume_mm3 / float(res_d)

    def _finalize_per_beamlet(
        self,
        edep_bev: torch.Tensor,
        dose_mask: torch.Tensor,
        crops: list[BevCrop],
        patient_grid: torch.Tensor,
    ) -> list[BeamletDose | None]:
        """Per-beamlet patient-frame dose, each cropped to its own support.

        Same physics as the summed finaliser -- the Gy conversion is linear, so
        the sum of these equals the summed volume -- just not accumulated. The
        results stay cropped so memory is the sum of the beamlet bounding boxes
        rather than ``G`` full grids.

        Returns:
            One :class:`BeamletDose` per beamlet, or ``None`` where the beamlet
            deposits nothing inside the grid.
        """
        _batch, num_beamlets, num_depths, _crop_h, _crop_w = edep_bev.shape
        full_h, full_w = crops[0].full_shape_hw
        out: list[BeamletDose | None] = []
        for g_idx in range(num_beamlets):
            prepared = self._rotation_inputs(edep_bev, crops[g_idx], g_idx, patient_grid)
            if prepared is None:
                out.append(None)
                continue
            volume, grid, h_src = prepared
            src_h = h_src.stop - h_src.start
            rotated = self._sample(volume, grid).reshape(1, src_h, num_depths, full_w)

            dose_slab = self._convert_mev_to_gy(rotated, dose_mask[:, h_src, :, :])
            nonzero = (dose_slab[0] > 0).nonzero()
            if nonzero.numel() == 0:
                out.append(None)
                continue
            mins = nonzero.min(dim=0).values
            maxs = nonzero.max(dim=0).values + 1
            z0, y0, x0 = int(mins[0]), int(mins[1]), int(mins[2])
            z1, y1, x1 = int(maxs[0]), int(maxs[1]), int(maxs[2])
            out.append(
                BeamletDose(
                    dose=dose_slab[:, z0:z1, y0:y1, x0:x1].contiguous(),
                    offset=(h_src.start + z0, y0, x0),
                    full_shape=(full_h, num_depths, full_w),
                )
            )
        return out

    def _finalize_patient_dose(
        self,
        edep_bev: torch.Tensor,
        dose_mask: torch.Tensor,
        crops: list[BevCrop],
        patient_grid: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        """Rotate every beamlet's BEV energy into the patient frame and sum it.

        Beamlets are rotated ``chunk_size`` at a time in one batched
        ``grid_sample``; the chunk size trades peak memory against launch
        overhead and does not change the result.
        """
        batch, num_beamlets, num_depths, crop_h, crop_w = edep_bev.shape
        full_h, full_w = crops[0].full_shape_hw
        total_edep = edep_bev.new_zeros((batch, full_h, num_depths, full_w))
        chunk_size = max(1, int(chunk_size))

        for start in range(0, num_beamlets, chunk_size):
            end = min(start + chunk_size, num_beamlets)
            volumes: list[torch.Tensor] = []
            grids: list[torch.Tensor] = []
            chunk_meta: list[tuple[slice, int]] = []

            for g_idx in range(start, end):
                crop = crops[g_idx]
                if tuple(crop.shape_hw) != (crop_h, crop_w):
                    raise ValueError("all BEV crops of one call must match edep_bev's shape")
                prepared = self._rotation_inputs(edep_bev, crop, g_idx, patient_grid)
                if prepared is None:
                    continue
                volume, grid, h_src = prepared
                volumes.append(volume)
                grids.append(grid)
                chunk_meta.append((h_src, h_src.stop - h_src.start))

            if not volumes:
                continue

            rotated_batch = self._sample(torch.cat(volumes, dim=0), torch.cat(grids, dim=0))

            offset = 0
            for h_src, src_h in chunk_meta:
                count = batch * src_h
                rotated = rotated_batch[offset : offset + count].reshape(batch, src_h, num_depths, full_w)
                total_edep[:, h_src, :, :] += rotated
                offset += count

        return self._convert_mev_to_gy(total_edep, dose_mask)

    # ------------------------------------------------------------ entry point

    def compute_dose(
        self,
        beamlets: IonBeamletBatch,
        density_image: torch.Tensor,
        dose_mask: torch.Tensor,
        *,
        ssd_mm: torch.Tensor | float | None = None,
        finalize_chunk_size: int = 4,
        beamlet_chunk_size: int | None = None,
        return_per_beamlet: bool = False,
    ) -> torch.Tensor | list[BeamletDose | None]:
        """Compute the dose of a beamlet batch. See :meth:`forward`."""
        return self(
            beamlets,
            density_image,
            dose_mask,
            ssd_mm=ssd_mm,
            finalize_chunk_size=finalize_chunk_size,
            beamlet_chunk_size=beamlet_chunk_size,
            return_per_beamlet=return_per_beamlet,
        )

    def forward(
        self,
        beamlets: IonBeamletBatch,
        density_image: torch.Tensor,
        dose_mask: torch.Tensor,
        *,
        ssd_mm: torch.Tensor | float | None = None,
        finalize_chunk_size: int = 4,
        beamlet_chunk_size: int | None = None,
        return_per_beamlet: bool = False,
    ) -> torch.Tensor | list[BeamletDose | None]:
        """Compute the patient-frame dose of a batch of ion beamlets.

        A single pass allocates BEV volumes for all ``G`` beamlets at once, so
        peak memory grows linearly with the number of spots and a full IMPT plan
        will not fit. ``beamlet_chunk_size`` splits the batch into groups that
        are computed one at a time and summed. Each group is wrapped in
        :func:`torch.utils.checkpoint.checkpoint`, so the backward pass
        recomputes one group's BEV volumes at a time instead of holding all of
        them; gradients are unaffected and reach every beamlet field.

        The result does not depend on the chunk size -- dose is a plain sum over
        beamlets and the Gy conversion is linear -- only peak memory does.

        Args:
            beamlets: The ``G`` beamlets to compute, all on the engine's device
                and dtype.
            density_image: Stopping-power-ratio volume, ``[H, D, W]`` or
                ``[1, H, D, W]``, matching ``dose_grid_shape``. This is what the
                transport integrates; it is *not* a mass density.
            dose_mask: Boolean volume of the same shape marking where dose is
                scored. Use :func:`patient_dose_mask`.
            ssd_mm: Source-to-skin distance in mm, scalar or ``(G,)``. ``None``
                applies no radiological-depth offset.
            finalize_chunk_size: Beamlets rotated per batched ``grid_sample``.
                Peak memory of the rotation step only.
            beamlet_chunk_size: Beamlets whose BEV volumes are built at once.
                ``None`` uses the engine's default; ``None`` there too means one
                unchunked pass. This is the setting that bounds peak memory.
            return_per_beamlet: Return the per-beamlet cropped doses instead of
                the summed volume. Chunking still applies, and the returned list
                is in beamlet order; checkpointing does not, since the outputs
                are kept anyway.

        Returns:
            ``[1, H, D, W]`` dose in Gy, or -- with ``return_per_beamlet`` -- a
            list of ``G`` :class:`BeamletDose` (``None`` for a beamlet that
            deposits nothing inside the grid).

        Raises:
            TypeError: If ``beamlets`` is not an :class:`IonBeamletBatch`.
            ValueError: On a shape, device or dtype mismatch of any input.
        """
        if not isinstance(beamlets, IonBeamletBatch):
            raise TypeError(f"beamlets must be an IonBeamletBatch, got {type(beamlets).__name__}")
        if beamlets.device != self.device or beamlets.dtype != self.dtype:
            raise ValueError(
                f"beamlets are on {beamlets.device}/{beamlets.dtype} but the engine computes on "
                f"{self.device}/{self.dtype}; move them with IonBeamletBatch.to()"
            )

        chunk_size = self.beamlet_chunk_size if beamlet_chunk_size is None else beamlet_chunk_size
        if chunk_size is None or int(chunk_size) >= len(beamlets):
            return self._forward_batch(
                beamlets,
                density_image,
                dose_mask,
                ssd_mm=ssd_mm,
                finalize_chunk_size=finalize_chunk_size,
                return_per_beamlet=return_per_beamlet,
            )
        chunk_size = max(1, int(chunk_size))

        # Expand ssd to one value per beamlet up front so each chunk can take its
        # own slice; a scalar would otherwise be re-broadcast to the chunk length.
        ssd_per_beamlet = self._expand_ssd(beamlets, ssd_mm)

        if return_per_beamlet:
            per_beamlet: list[BeamletDose | None] = []
            for start, end, chunk in beamlets.chunks(chunk_size):
                per_beamlet.extend(
                    self._forward_batch(
                        chunk,
                        density_image,
                        dose_mask,
                        ssd_mm=None if ssd_per_beamlet is None else ssd_per_beamlet[start:end],
                        finalize_chunk_size=finalize_chunk_size,
                        return_per_beamlet=True,
                    )
                )
            return per_beamlet

        dose = None
        for start, end, chunk in beamlets.chunks(chunk_size):
            chunk_dose = checkpoint(
                self._forward_batch_positional,
                chunk,
                density_image,
                dose_mask,
                None if ssd_per_beamlet is None else ssd_per_beamlet[start:end],
                finalize_chunk_size,
                use_reentrant=False,
            )
            dose = chunk_dose if dose is None else dose + chunk_dose
        return dose

    def _forward_batch_positional(
        self,
        beamlets: IonBeamletBatch,
        density_image: torch.Tensor,
        dose_mask: torch.Tensor,
        ssd_mm: torch.Tensor | None,
        finalize_chunk_size: int,
    ) -> torch.Tensor:
        """Positional-only :meth:`_forward_batch`, for ``checkpoint``.

        ``torch.utils.checkpoint`` forwards positional arguments only, so the
        keyword-only signature of :meth:`_forward_batch` cannot be checkpointed
        directly.
        """
        return self._forward_batch(
            beamlets,
            density_image,
            dose_mask,
            ssd_mm=ssd_mm,
            finalize_chunk_size=finalize_chunk_size,
            return_per_beamlet=False,
        )

    def _expand_ssd(
        self,
        beamlets: IonBeamletBatch,
        ssd_mm: torch.Tensor | float | None,
    ) -> torch.Tensor | None:
        """Broadcast ``ssd_mm`` to one value per beamlet so chunks can slice it.

        Args:
            beamlets: The full batch, for its length.
            ssd_mm: Source-to-skin distance, scalar or ``(G,)``, or ``None``.

        Returns:
            A ``(G,)`` tensor, or ``None`` when no offset was requested.

        Raises:
            ValueError: If ``ssd_mm`` is neither scalar nor ``(G,)``.
        """
        if ssd_mm is None:
            return None
        num_beamlets = len(beamlets)
        ssd_tensor = torch.as_tensor(ssd_mm, device=self.device, dtype=self.dtype)
        if ssd_tensor.ndim == 0:
            return ssd_tensor.expand(num_beamlets).clone()
        if ssd_tensor.shape != (num_beamlets,):
            raise ValueError(
                f"ssd_mm must be scalar or [{num_beamlets}], got {tuple(ssd_tensor.shape)}"
            )
        return ssd_tensor

    def _forward_batch(
        self,
        beamlets: IonBeamletBatch,
        density_image: torch.Tensor,
        dose_mask: torch.Tensor,
        *,
        ssd_mm: torch.Tensor | float | None = None,
        finalize_chunk_size: int = 4,
        return_per_beamlet: bool = False,
    ) -> torch.Tensor | list[BeamletDose | None]:
        """Compute the dose of one whole beamlet batch in a single BEV pass.

        This is the unchunked core: it materialises ``[1, G, depths, h, w]`` BEV
        volumes for every beamlet at once, so its peak memory scales with ``G``.
        :meth:`forward` calls it directly, or once per chunk.

        Args:
            beamlets: The ``G`` beamlets to compute, all on the engine's device
                and dtype.
            density_image: Stopping-power-ratio volume, ``[H, D, W]`` or
                ``[1, H, D, W]``, matching ``dose_grid_shape``. This is what the
                transport integrates; it is *not* a mass density.
            dose_mask: Boolean volume of the same shape marking where dose is
                scored. Use :func:`patient_dose_mask`.
            ssd_mm: Source-to-skin distance in mm, scalar or ``(G,)``. ``None``
                applies no radiological-depth offset.
            finalize_chunk_size: Beamlets rotated per batched ``grid_sample``.
                Peak memory only; the result is identical for any value.
            return_per_beamlet: Return the per-beamlet cropped doses instead of
                the summed volume, from the same single BEV pass.

        Returns:
            ``[1, H, D, W]`` dose in Gy, or -- with ``return_per_beamlet`` -- a
            list of ``G`` :class:`BeamletDose` (``None`` for a beamlet that
            deposits nothing inside the grid).

        Raises:
            ValueError: On a shape, device or dtype mismatch of any input.
        """
        if not isinstance(beamlets, IonBeamletBatch):
            raise TypeError(f"beamlets must be an IonBeamletBatch, got {type(beamlets).__name__}")
        if beamlets.device != self.device or beamlets.dtype != self.dtype:
            raise ValueError(
                f"beamlets are on {beamlets.device}/{beamlets.dtype} but the engine computes on "
                f"{self.device}/{self.dtype}; move them with IonBeamletBatch.to()"
            )
        num_beamlets = len(beamlets)

        density_image = self._as_volume(density_image, "density_image").to(
            device=self.device, dtype=self.dtype
        )
        dose_mask = self._as_volume(dose_mask, "dose_mask").to(device=self.device, dtype=torch.bool)

        depth_offset = self._resolve_depth_offset(beamlets, ssd_mm)
        bev_grid, patient_grid = self._sampling_grids(beamlets)

        res_h, res_d, res_w = self.dose_grid_spacing
        iso_centers = beamlets.iso_center_mm.to(device=self.device, dtype=self.dtype)
        position_mm = beamlets.position_mm.to(device=self.device, dtype=self.dtype)
        center_h_vox = iso_centers[:, 0] / res_h + position_mm[:, 1] / res_h
        center_w_vox = iso_centers[:, 2] / res_w + position_mm[:, 0] / res_w
        crops = self._build_crops(center_h_vox, center_w_vox)
        crop_centers_hw = torch.stack(
            (
                center_h_vox
                - torch.as_tensor(
                    [c.target_h_start for c in crops], device=self.device, dtype=self.dtype
                ),
                center_w_vox
                - torch.as_tensor(
                    [c.target_w_start for c in crops], device=self.device, dtype=self.dtype
                ),
            ),
            dim=1,
        )

        density_bev_flat, weq_bev_flat = self._density_and_weq_bev(density_image, crops, bev_grid)
        num_depths = density_bev_flat.shape[1]
        height, width = crops[0].shape_hw
        density_bev = density_bev_flat.view(1, num_beamlets, num_depths, height, width)
        weq_bev = weq_bev_flat.view(1, num_beamlets, num_depths, height, width)

        # Crop cells that lie inside the dose grid. Everything else is off the
        # edge of the patient volume and must not be normalised over.
        valid_lateral = torch.zeros((num_beamlets, height, width), device=self.device, dtype=torch.bool)
        for g_idx, crop in enumerate(crops):
            if not crop.is_empty:
                valid_lateral[g_idx, crop.h_dst, crop.w_dst] = True

        lattice = _BevLattice(
            h_coords=torch.arange(height, device=self.device, dtype=self.dtype),
            w_coords=torch.arange(width, device=self.device, dtype=self.dtype),
            res_h=res_h,
            res_w=res_w,
            res_d=res_d,
        )

        edep_layers = []
        weq_depths = []
        for g_idx in range(num_beamlets):
            energy = beamlets.energy_mev[g_idx]
            ctx = _BeamletContext(
                energy_mev=energy,
                energy_value=float(energy.detach()),
                sigma_x_mm=beamlets.sigma_mm[g_idx, 0],
                sigma_y_mm=beamlets.sigma_mm[g_idx, 1],
                weight=beamlets.weight[g_idx],
                weq_bev=weq_bev[0, g_idx],
                depth_offset_mm=depth_offset[g_idx],
                center_h=crop_centers_hw[g_idx, 0],
                center_w=crop_centers_hw[g_idx, 1],
                active=valid_lateral[g_idx].unsqueeze(0),
            )
            edep_layers.append(self.compute_layer_edep(ctx, lattice))
            weq_depths.append(self._center_ray_weq(weq_bev, g_idx, ctx, height, width))

        edep_5d = torch.stack(edep_layers, dim=0).view(1, num_beamlets, num_depths, height, width)
        weq_depths_flat = torch.stack(weq_depths, dim=1).view(num_beamlets, num_depths)

        edep_to_gy = edep_5d.new_tensor(MEV_CM2_PER_G_TO_GY_MM2 / self._lateral_area_mm2())
        payload = {
            "edep_bev": edep_5d,
            "density_bev": density_bev,
            "weq_bev": weq_bev,
            "weq_depths": weq_depths_flat,
            "density_image": density_image,
            "resolved_offset": depth_offset,
            "beamlets": beamlets,
            "edep_to_gy": edep_to_gy,
            "bev_crop": crops,
            "crop_centers_hw": crop_centers_hw,
        }
        if self.bev_correction is not None:
            payload = self.bev_correction(payload, engine=self)
        corrected_edep = payload["edep_bev"]

        if return_per_beamlet:
            return self._finalize_per_beamlet(corrected_edep, dose_mask, crops, patient_grid)
        return self._finalize_patient_dose(
            corrected_edep, dose_mask, crops, patient_grid, chunk_size=finalize_chunk_size
        )

    @staticmethod
    def _center_ray_weq(
        weq_bev: torch.Tensor,
        g_idx: int,
        ctx: _BeamletContext,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """``(1, D)`` WEQ depth along the beamlet's central ray.

        Bilinear in the lateral plane, because the beamlet centre generally
        falls between crop voxels. Reported to the correction hook, never used
        by the transport (which samples per sub-beam).
        """
        h = torch.clamp(ctx.center_h, 0.0, float(height - 1))
        w = torch.clamp(ctx.center_w, 0.0, float(width - 1))
        h0 = int(torch.floor(h).item())
        w0 = int(torch.floor(w).item())
        h1 = min(h0 + 1, height - 1)
        w1 = min(w0 + 1, width - 1)
        dh = h - h0
        dw = w - w0
        v00 = weq_bev[:, g_idx, :, h0, w0]
        v01 = weq_bev[:, g_idx, :, h0, w1]
        v10 = weq_bev[:, g_idx, :, h1, w0]
        v11 = weq_bev[:, g_idx, :, h1, w1]
        return (
            v00 * (1.0 - dh) * (1.0 - dw)
            + v01 * (1.0 - dh) * dw
            + v10 * dh * (1.0 - dw)
            + v11 * dh * dw
        )

    def _as_volume(self, volume: torch.Tensor, name: str) -> torch.Tensor:
        """Validate a ``[H, D, W]`` / ``[1, H, D, W]`` input and return it 4-D."""
        if not isinstance(volume, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(volume).__name__}")
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)
        if volume.dim() != 4 or volume.shape[0] != 1:
            raise ValueError(
                f"{name} must be [H, D, W] or [1, H, D, W], got {list(volume.shape)}"
            )
        if tuple(volume.shape[1:]) != self.dose_grid_shape:
            raise ValueError(
                f"{name} has shape {list(volume.shape[1:])} but the dose grid is "
                f"{list(self.dose_grid_shape)}"
            )
        return volume

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"IonDoseEngine(grid={self.dose_grid_shape}, spacing={self.dose_grid_spacing}, "
            f"field={self.field_size}, lateral_model={self.lateral_model!r}, "
            f"heterogeneous_mcs={self.heterogeneous_mcs}, n={self.n_sub_beams_per_dim}, "
            f"device={self.device}, dtype={self.dtype})"
        )
