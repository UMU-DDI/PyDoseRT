"""Beam's-eye-view (BEV) sampling grids and BEV crops for the ion dose engine.

The ion engine works in a *beam's-eye view*: a per-beam frame whose depth axis
runs along the central ray, whose origin sits at the point where that ray enters
the dose grid, and whose lateral axis is measured from the isocentre. Two
:func:`torch.nn.functional.grid_sample` grids move volumes between that frame and
the patient frame, and both are pure functions of the geometry (grid shape, voxel
spacing, gantry angles, isocentres) -- they carry no learnable state and never
need a :class:`torch.nn.Module`.

Both grids act on the ``(D, W)`` plane only; the ``H`` axis is a batch axis for
``grid_sample`` because a gantry rotation about the patient's long axis never
mixes ``H`` into anything.

Each function is named after the frame it produces:

* :func:`build_bev_sampling_grid` -- output is BEV, input sampled is the patient
  volume. The angle negation ``grid_sample`` needs is internal; callers pass the
  physical gantry angles.
* :func:`build_patient_sampling_grid` -- output is the patient frame, input
  sampled is a BEV volume.

Conventions
-----------
* ``grid_shape`` is ``(H, D, W)`` voxels; ``D`` is the beam-depth axis at gantry
  angle 0.
* ``spacing_mm`` is ``(rh, rd, rw)`` mm, aligned with ``(H, D, W)``.
* ``iso_center_mm`` is ``(h_mm, d_mm, w_mm)`` from the grid origin, either one
  shared centre ``(3,)`` or one per beam ``(G, 3)``.
* Angles are in radians and increase in the direction that carries ``+d`` toward
  ``-w``.
* Returned grids are ``align_corners=False`` normalised coordinates, last axis
  ordered ``(x, y)`` = ``(W, D)`` as ``grid_sample`` expects.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

__all__ = [
    "BevCrop",
    "build_bev_crop",
    "build_bev_sampling_grid",
    "build_patient_sampling_grid",
    "crop_slices",
    "normalize_iso_centers",
]

#: Distance in mm the virtual source is pushed back along the central ray before
#: intersecting the dose grid. Large enough to sit outside any clinical grid, so
#: the entry point is the true grid entry and not a point inside the volume.
_SOURCE_BACKOFF_MM = 1_000_000.0


# --------------------------------------------------------------------------- geometry helpers


def normalize_iso_centers(
    iso_center_mm: torch.Tensor | Sequence[float] | Sequence[Sequence[float]],
    num_beams: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast ``iso_center_mm`` to a ``(G, 3)`` tensor of mm coordinates.

    Args:
        iso_center_mm: One shared centre of shape ``(3,)`` or a per-beam
            ``(num_beams, 3)``, ordered ``(h, d, w)`` in mm.
        num_beams: The number of beams ``G`` to broadcast to.
        device: Device of the returned tensor.
        dtype: Floating dtype of the returned tensor.

    Returns:
        ``(G, 3)`` isocentres in mm.

    Raises:
        ValueError: If ``iso_center_mm`` is ``None`` or has any other shape. An
            isocentre of the wrong shape is a caller bug, not something to
            silently default to the volume centre.
    """
    if iso_center_mm is None:
        raise ValueError("iso_center_mm is required; entry-origin BEV grids have no volume-centre fallback")
    iso_tensor = torch.as_tensor(iso_center_mm, device=device, dtype=dtype)
    if iso_tensor.shape == (3,):
        return iso_tensor.unsqueeze(0).expand(num_beams, -1)
    if iso_tensor.shape == (num_beams, 3):
        return iso_tensor
    raise ValueError(
        f"iso_center_mm must be shape (3,) or ({num_beams}, 3), got {tuple(iso_tensor.shape)}"
    )


def _as_angles(
    angles_rad: torch.Tensor | Sequence[float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    angles = torch.as_tensor(angles_rad, device=device, dtype=dtype)
    if angles.ndim != 1:
        raise ValueError(f"gantry angles must be a 1-D sequence of radians, got shape {tuple(angles.shape)}")
    if angles.numel() == 0:
        raise ValueError("at least one gantry angle is required")
    return angles


def _check_geometry(
    grid_shape: Sequence[int],
    spacing_mm: Sequence[float],
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """Validate and narrow the grid geometry to fixed-length tuples."""
    if len(grid_shape) != 3:
        raise ValueError(f"grid_shape must be (H, D, W), got {tuple(grid_shape)}")
    if len(spacing_mm) != 3:
        raise ValueError(f"spacing_mm must be (rh, rd, rw) in mm, got {tuple(spacing_mm)}")
    height, depth, width = (int(v) for v in grid_shape)
    if height < 1 or depth < 1 or width < 1:
        raise ValueError(f"grid_shape must be positive, got {(height, depth, width)}")
    res_h, res_d, res_w = (float(v) for v in spacing_mm)
    if res_h <= 0.0 or res_d <= 0.0 or res_w <= 0.0:
        raise ValueError(f"spacing_mm must be strictly positive, got {(res_h, res_d, res_w)}")
    return (height, depth, width), (res_h, res_d, res_w)


def _beam_axis_lat_units(angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit vectors of the central ray (``axis``) and of the BEV lateral axis (``lat``).

    Both are ``(G, 3)`` in ``(h, d, w)`` order; the ``h`` component is always zero
    because a gantry rotation leaves ``H`` untouched.
    """
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    axis_units = torch.stack(
        (
            torch.zeros_like(cos_angles),
            cos_angles,
            -sin_angles,
        ),
        dim=1,
    )
    lat_units = torch.stack(
        (
            torch.zeros_like(cos_angles),
            sin_angles,
            cos_angles,
        ),
        dim=1,
    )
    return axis_units, lat_units


def _central_ray_entry_mm(
    grid_shape: tuple[int, int, int],
    axis_units: torch.Tensor,
    iso_centers: torch.Tensor,
    spacing_mm: tuple[float, float, float],
) -> torch.Tensor:
    """Where each central ray enters the dose grid, in mm ``(h, d, w)``.

    A slab test against the grid bounding box from a source pushed
    ``_SOURCE_BACKOFF_MM`` back along the ray. ``t_entry`` is clamped at zero, so
    an isocentre already outside the grid on the entry side still yields the
    source-side extreme rather than a point behind the source.
    """
    h, d, w = grid_shape
    rh, rd, rw = spacing_mm
    device, dtype = axis_units.device, axis_units.dtype

    bounds_min = torch.zeros((3,), device=device, dtype=dtype)
    bounds_max = torch.tensor(
        ((h - 1) * rh, (d - 1) * rd, (w - 1) * rw),
        device=device,
        dtype=dtype,
    )
    source = iso_centers - axis_units * torch.as_tensor(_SOURCE_BACKOFF_MM, device=device, dtype=dtype)
    eps = torch.finfo(dtype).eps
    safe_axis = torch.where(axis_units.abs() < eps, torch.full_like(axis_units, eps), axis_units)
    inv_axis = 1.0 / safe_axis
    t0 = (bounds_min - source) * inv_axis
    t1 = (bounds_max - source) * inv_axis
    zero = torch.zeros((axis_units.shape[0],), device=device, dtype=dtype)
    t_entry = torch.maximum(torch.minimum(t0, t1).amax(dim=1), zero)
    return source + axis_units * t_entry[:, None]


# --------------------------------------------------------------------------- sampling grids


def build_bev_sampling_grid(
    grid_shape: Sequence[int],
    spacing_mm: Sequence[float],
    gantry_angles_rad: torch.Tensor | Sequence[float],
    iso_center_mm: torch.Tensor | Sequence[float] | Sequence[Sequence[float]],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Grid that resamples a **patient**-frame ``(D, W)`` plane **into BEV**.

    For every BEV cell ``(d_bev, w_bev)`` the grid holds the normalised patient
    coordinate to read: walk ``d_bev`` steps of ``rd`` mm along the central ray
    from its grid-entry point, then ``w_bev - iso_w/rw`` steps of ``rw`` mm along
    the lateral axis.

    ``gantry_angles_rad`` are the physical gantry angles; the negation the
    coordinate map needs happens internally.

    Args:
        grid_shape: ``(H, D, W)`` dose-grid shape in voxels.
        spacing_mm: ``(rh, rd, rw)`` voxel spacing in mm.
        gantry_angles_rad: ``G`` gantry angles in radians.
        iso_center_mm: ``(3,)`` shared or ``(G, 3)`` per-beam isocentre in mm.
        device: Device for the returned grid.
        dtype: Floating dtype for the returned grid.

    Returns:
        ``[G, D, W, 2]`` normalised sampling grid, last axis ``(x, y)``.

    Raises:
        ValueError: If the grid shape, the spacing, the angles or the isocentre
            has the wrong shape or a non-positive entry.
    """
    (_H, D, W), (rh, rd, rw) = _check_geometry(grid_shape, spacing_mm)
    device = torch.device("cpu") if device is None else torch.device(device)

    # The BEV->patient map is the mirror of the patient->BEV one: negate.
    angles = -_as_angles(gantry_angles_rad, device, dtype)
    G = int(angles.numel())
    iso_centers = normalize_iso_centers(iso_center_mm, G, device, dtype)

    # _beam_axis_lat_units() evaluated at the *un*-negated angle, reached
    # through the negated one -- hence the sign pattern.
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    axis_units = torch.stack((torch.zeros_like(cos_a), cos_a, sin_a), dim=1)
    lat_units = torch.stack((torch.zeros_like(cos_a), -sin_a, cos_a), dim=1)

    entry = _central_ray_entry_mm((_H, D, W), axis_units, iso_centers, (rh, rd, rw))

    d_coords = torch.arange(D, device=device, dtype=dtype) * rd
    w_offsets = (torch.arange(W, device=device, dtype=dtype) - (iso_centers[:, 2:3] / rw)) * rw
    d_grid = d_coords.view(1, D, 1)
    w_grid = w_offsets.view(G, 1, W)
    points = (
        entry[:, None, None, :]
        + axis_units[:, None, None, :] * d_grid[..., None]
        + lat_units[:, None, None, :] * w_grid[..., None]
    )
    sample_y = points[..., 1] / rd
    sample_x = points[..., 2] / rw
    grid_x = (2.0 * (sample_x + 0.5) / W) - 1.0
    grid_y = (2.0 * (sample_y + 0.5) / D) - 1.0
    return torch.stack((grid_x, grid_y), dim=-1)


def build_patient_sampling_grid(
    grid_shape: Sequence[int],
    spacing_mm: Sequence[float],
    gantry_angles_rad: torch.Tensor | Sequence[float],
    iso_center_mm: torch.Tensor | Sequence[float] | Sequence[Sequence[float]],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Grid that resamples a **BEV** ``(D, W)`` plane back **into the patient frame**.

    For every patient cell ``(d, w)`` the grid holds the normalised BEV
    coordinate to read: the depth of that cell along the central ray measured
    from the ray's grid-entry point, and its lateral offset measured from the
    isocentre.

    The returned grid spans the **full** grid width. A consumer that holds a
    laterally cropped BEV volume must re-normalise the ``x`` channel onto the
    crop (see :class:`BevCrop`); the ``y`` channel is unaffected because the
    depth axis is never cropped.

    Args:
        grid_shape: ``(H, D, W)`` dose-grid shape in voxels.
        spacing_mm: ``(rh, rd, rw)`` voxel spacing in mm.
        gantry_angles_rad: ``G`` gantry angles in radians.
        iso_center_mm: ``(3,)`` shared or ``(G, 3)`` per-beam isocentre in mm.
        device: Device for the returned grid.
        dtype: Floating dtype for the returned grid.

    Returns:
        ``[G, D, W, 2]`` normalised sampling grid, last axis ``(x, y)``.

    Raises:
        ValueError: If the grid shape, the spacing, the angles or the isocentre
            has the wrong shape or a non-positive entry.
    """
    (_H, D, W), (rh, rd, rw) = _check_geometry(grid_shape, spacing_mm)
    device = torch.device("cpu") if device is None else torch.device(device)

    angles = _as_angles(gantry_angles_rad, device, dtype)
    G = int(angles.numel())
    iso_centers = normalize_iso_centers(iso_center_mm, G, device, dtype)

    axis_units, lat_units = _beam_axis_lat_units(angles)
    entry = _central_ray_entry_mm((_H, D, W), axis_units, iso_centers, (rh, rd, rw))

    d_coords = torch.arange(D, device=device, dtype=dtype) * rd
    w_coords = torch.arange(W, device=device, dtype=dtype) * rw
    d_grid, w_grid = torch.meshgrid(d_coords, w_coords, indexing="ij")
    points_dw = torch.stack((d_grid, w_grid), dim=-1).unsqueeze(0)  # [1, D, W, 2]

    entry_dw = entry[:, [1, 2]].view(G, 1, 1, 2)
    axis_dw = axis_units[:, [1, 2]].view(G, 1, 1, 2)
    lat_dw = lat_units[:, [1, 2]].view(G, 1, 1, 2)
    delta = points_dw - entry_dw
    depth_mm = (delta * axis_dw).sum(dim=-1)
    lateral_mm = (delta * lat_dw).sum(dim=-1)

    d_bev = depth_mm / rd
    w_bev = lateral_mm / rw + iso_centers[:, 2].view(G, 1, 1) / rw
    grid_x = (2.0 * (w_bev + 0.5) / W) - 1.0
    grid_y = (2.0 * (d_bev + 0.5) / D) - 1.0
    return torch.stack((grid_x, grid_y), dim=-1)


# --------------------------------------------------------------------------- BEV crops


def crop_slices(center: float, full_size: int, crop_size: int) -> tuple[slice, slice, slice]:
    """Slice a ``crop_size`` window centred on ``center`` out of ``[0, full_size)``.

    All three arguments are in voxels; ``center`` may be fractional.

    Returns ``(src, dst, target)``:

    * ``src`` -- where to read in the full axis, clipped to ``[0, full_size)``.
    * ``dst`` -- where those samples land inside the ``crop_size`` window.
    * ``target`` -- the unclipped window, whose ``.start`` may be negative and
      whose ``.stop`` may exceed ``full_size``. It is the crop's origin in
      full-axis coordinates.

    When the window lies entirely outside the axis, ``src`` (and ``dst``) come
    back empty -- ``src.stop <= src.start`` -- rather than raising; callers skip
    such beams.

    .. warning::
       For a window entirely below zero, ``src`` is ``slice(0, negative)``, and
       Python reads a negative stop as an offset from the end -- slicing with it
       yields a *large, non-empty* result. Test emptiness with ``stop <= start``
       (or :attr:`BevCrop.is_empty`) **before** using the slice.

    The centre uses Python's banker's rounding (``round(-11.5) == -12``).
    Half-integer centres are ordinary here -- a spot on a voxel boundary -- so
    the tie rule decides which voxels a beam covers; ``int(x + 0.5)`` or
    :func:`torch.round` would shift them.
    """
    crop_size = int(crop_size)
    center_i = int(round(float(center)))
    target_lo = center_i - crop_size // 2
    target_hi = target_lo + crop_size
    src_lo = max(target_lo, 0)
    src_hi = min(target_hi, int(full_size))
    dst_lo = src_lo - target_lo
    dst_hi = dst_lo + max(src_hi - src_lo, 0)
    return slice(src_lo, src_hi), slice(dst_lo, dst_hi), slice(target_lo, target_hi)


@dataclass(frozen=True)
class BevCrop:
    """One beam's lateral window onto the dose grid, in voxels.

    The engine computes BEV dose on a ``(field_h, field_w)`` window instead of
    the full ``(H, W)`` lateral extent. This carries everything needed to move a
    volume between the two: which full-grid voxels the window covers (``*_src``),
    where they sit inside the window (``*_dst``), and where the window sits in
    full-grid coordinates including the part that hangs off the edge
    (``*_target`` / ``target_*_start``).

    ``h`` indexes the patient ``H`` axis and ``w`` the patient ``W`` axis. The
    depth axis ``D`` is never cropped.
    """

    #: ``(size_h, size_w)`` of the window, after clamping to the grid.
    shape_hw: tuple[int, int]
    #: ``(H, W)`` of the full dose grid.
    full_shape_hw: tuple[int, int]
    #: Full-grid rows covered by the window, clipped to the grid.
    h_src: slice
    #: Where ``h_src`` lands inside the window.
    h_dst: slice
    #: The unclipped window in full-grid rows; ``.start`` may be negative.
    h_target: slice
    #: Full-grid columns covered by the window, clipped to the grid.
    w_src: slice
    #: Where ``w_src`` lands inside the window.
    w_dst: slice
    #: The unclipped window in full-grid columns; ``.start`` may be negative.
    w_target: slice
    #: ``h_target.start``, i.e. the window origin along ``H``.
    target_h_start: int
    #: ``w_target.start``, i.e. the window origin along ``W``.
    target_w_start: int

    @property
    def h_is_empty(self) -> bool:
        """True when the window does not overlap the grid along ``H``."""
        return self.h_src.stop <= self.h_src.start

    @property
    def w_is_empty(self) -> bool:
        """True when the window does not overlap the grid along ``W``."""
        return self.w_src.stop <= self.w_src.start

    @property
    def is_empty(self) -> bool:
        """True when the window does not overlap the grid at all."""
        return self.h_is_empty or self.w_is_empty


def build_bev_crop(
    center_h: float,
    center_w: float,
    size_h: int,
    size_w: int,
    full_shape_hw: Sequence[int],
) -> BevCrop:
    """Build the :class:`BevCrop` centred on ``(center_h, center_w)`` voxels.

    ``size_h`` / ``size_w`` are clamped into ``[1, full]``: a field larger than
    the grid becomes the whole grid rather than raising.

    Args:
        center_h: Window centre along ``H``, in voxels (may be fractional).
        center_w: Window centre along ``W``, in voxels (may be fractional).
        size_h: Requested window height in voxels.
        size_w: Requested window width in voxels.
        full_shape_hw: ``(H, W)`` of the full dose grid, in voxels.

    Returns:
        The crop; :attr:`BevCrop.is_empty` is True when it misses the grid.

    Raises:
        ValueError: If ``full_shape_hw`` is not a positive ``(H, W)`` pair.
    """
    if len(full_shape_hw) != 2:
        raise ValueError(f"full_shape_hw must be (H, W), got {tuple(full_shape_hw)}")
    full_h, full_w = (int(v) for v in full_shape_hw)
    if full_h < 1 or full_w < 1:
        raise ValueError(f"full_shape_hw must be positive, got {(full_h, full_w)}")

    size_h = max(1, min(int(size_h), full_h))
    size_w = max(1, min(int(size_w), full_w))
    h_src, h_dst, h_target = crop_slices(center_h, full_h, size_h)
    w_src, w_dst, w_target = crop_slices(center_w, full_w, size_w)
    return BevCrop(
        shape_hw=(size_h, size_w),
        full_shape_hw=(full_h, full_w),
        h_src=h_src,
        h_dst=h_dst,
        h_target=h_target,
        w_src=w_src,
        w_dst=w_dst,
        w_target=w_target,
        target_h_start=int(h_target.start),
        target_w_start=int(w_target.start),
    )
