"""Multilattice photon pencil beam.

The shipped :class:`~pydosert.engine.multislab_engine.MultislabEngine` traces a
SINGLE central-axis ray, so every voxel in a depth plane is convolved with the
kernel belonging to that one ray's radiological depth, and the lateral variation
is recovered afterwards by a scalar ``exp(-mu_eff * (d - d_cax))`` factor.  That
breaks down whenever the beam crosses laterally heterogeneous anatomy.

Multilattice keeps the same per-plane kernels but uses an ``L x L`` lattice of
rays instead of one.  The beam's-eye-view fluence is partitioned into tiles of
approximately EQUAL FLUENCE, each tile gets the kernel set belonging to a ray
through its own fluence-weighted centroid, and only that tile's fluence is
convolved with it.  The residual ``exp()`` factor is then referenced to the
tile's own ray rather than to the central axis, so it corrects a much smaller
difference.

An ``L = 1`` lattice is a single ray, but through the fluence-weighted CENTROID
of the aperture rather than through the isocentre, so it does not reproduce the
central-ray calculation bit for bit -- measured 2% apart on a centred field and
much more on a strongly off-axis one, where the centroid is the more
representative ray of the two.  Lattice size trades accuracy against ``L**2``
convolutions.

Units: ``divergent_radiological_depth`` returns density x CM, while
``PencilBeamKernelLayer`` expects the ``RadiologicalDepthLayer`` scale, which is
density x MM.  :func:`ray_depth_profile` returns cm and the caller multiplies by
``DEPTH_CM_TO_KERNEL_UNITS`` -- keeping the conversion in one named place.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pydosert.engine.multislab_engine import (
    MultislabEngine,
    divergent_radiological_depth,
)

DEPTH_CM_TO_KERNEL_UNITS = 10.0


def equal_fluence_edges(profile: torch.Tensor, parts: int) -> list[int]:
    """Bin edges splitting a non-negative 1-D profile into ~equal-sum parts.

    Equal fluence rather than equal width: a tile that carries almost no fluence
    contributes almost nothing, so spending a ray on it is wasted, while a
    narrow, intense part of the aperture deserves its own ray.
    """
    n = int(profile.numel())
    if parts <= 1 or float(profile.sum()) <= 0.0:
        return [0, n]
    cumulative = profile.double().cumsum(0)
    total = cumulative[-1]
    interior = []
    for i in range(1, parts):
        edge = int(torch.searchsorted(cumulative, total * i / parts).item()) + 1
        interior.append(min(max(edge, 1), n - 1))
    return [0, *sorted(set(interior)), n]


def _iso_indices(spacing, iso_center):
    r_h, r_d, r_w = (float(x) for x in spacing)
    return (float(iso_center[0]) / r_h,
            float(iso_center[1]) / r_d,
            float(iso_center[2]) / r_w), (r_h, r_d, r_w)


def _divergence_scale(depth_planes, sad_mm, iso_d, r_d, device, dtype):
    """Lateral magnification (z / SAD) of each depth plane, relative to isocentre."""
    z = torch.arange(depth_planes, device=device, dtype=dtype)
    return ((float(sad_mm) + (z - iso_d) * r_d) / float(sad_mm)).clamp_min(1e-3)


def ray_depth_profile(dense_depth: torch.Tensor, center_h: torch.Tensor,
                      center_w: torch.Tensor, sad_mm: float, spacing,
                      iso_center) -> torch.Tensor:
    """Radiological depth along one DIVERGENT ray, in density x cm.

    Args:
        dense_depth: per-voxel radiological depth ``[D, H, W]``.
        center_h, center_w: ray position in voxels, expressed at the isocentre
            plane; the ray fans out from there by the divergence scale.
    Returns:
        ``[D]`` depth profile sampled along the ray.
    """
    d, h, w = dense_depth.shape
    (iso_h, iso_d, iso_w), (_, r_d, _) = _iso_indices(spacing, iso_center)
    scale = _divergence_scale(d, sad_mm, iso_d, r_d, dense_depth.device, dense_depth.dtype)
    ray_h = iso_h + (center_h - iso_h) * scale
    ray_w = iso_w + (center_w - iso_w) * scale
    grid = torch.stack((2.0 * (ray_w + 0.5) / w - 1.0,
                        2.0 * (ray_h + 0.5) / h - 1.0,
                        2.0 * (torch.arange(d, device=dense_depth.device,
                                            dtype=dense_depth.dtype) + 0.5) / d - 1.0),
                       dim=-1).view(1, d, 1, 1, 3)
    return F.grid_sample(dense_depth.view(1, 1, d, h, w), grid, mode="bilinear",
                         padding_mode="border", align_corners=False)[0, 0, :, 0, 0]


def backprojected_tile_mask(shape, h_bounds, w_bounds, sad_mm, spacing,
                            iso_center, device, dtype) -> torch.Tensor:
    """Which voxels belong to a tile, at every depth plane.

    Tile bounds are defined once at the isocentre plane; a voxel belongs to the
    tile if its position mapped BACK to isocentre falls inside them.  That makes
    the tiles diverge with the beam, so they stay aligned with the fluence they
    were cut from.
    """
    d, h, w = shape
    (iso_h, iso_d, iso_w), (_, r_d, _) = _iso_indices(spacing, iso_center)
    scale = _divergence_scale(d, sad_mm, iso_d, r_d, device, dtype).view(d, 1, 1)
    h_at_iso = iso_h + (torch.arange(h, device=device, dtype=dtype).view(1, h, 1) - iso_h) / scale
    w_at_iso = iso_w + (torch.arange(w, device=device, dtype=dtype).view(1, 1, w) - iso_w) / scale
    return ((h_at_iso >= h_bounds[0]) & (h_at_iso < h_bounds[1])
            & (w_at_iso >= w_bounds[0]) & (w_at_iso < w_bounds[1]))


def lattice_tiles(fluence_bev: torch.Tensor, lattice_size: int, spacing,
                  iso_center) -> list[tuple]:
    """Cut one beam's fluence into ``L x L`` equal-fluence tiles.

    Returns one ``(h_bounds, w_bounds, centre_h, centre_w)`` per non-empty tile,
    with the centre being the fluence-weighted centroid at the isocentre plane.
    """
    d, h, w = fluence_bev.shape
    (_, iso_d, _), _ = _iso_indices(spacing, iso_center)
    plane = fluence_bev[int(min(max(iso_d, 0), d - 1))].clamp_min(0.0)
    h_edges = equal_fluence_edges(plane.sum(1), lattice_size)
    w_edges = equal_fluence_edges(plane.sum(0), lattice_size)
    h_idx = torch.arange(h, device=plane.device, dtype=plane.dtype).view(h, 1)
    w_idx = torch.arange(w, device=plane.device, dtype=plane.dtype).view(1, w)
    tiles = []
    for h0, h1 in zip(h_edges[:-1], h_edges[1:]):
        for w0, w1 in zip(w_edges[:-1], w_edges[1:]):
            block = plane[h0:h1, w0:w1]
            weight = block.sum()
            if float(weight) <= 0.0:
                continue
            tiles.append(((h0, h1), (w0, w1),
                          (block * h_idx[h0:h1]).sum() / weight,
                          (block * w_idx[:, w0:w1]).sum() / weight))
    return tiles


def multilattice_dose(fluence_volume: torch.Tensor, bev_density: torch.Tensor,
                      kernel_layer, conv_layer, sad_mm: float, spacing,
                      iso_center, lattice_size: int, mu_eff: float,
                      cf_clamp: tuple = (0.3, 3.0),
                      dense_depth: torch.Tensor | None = None,
                      source_scale: torch.Tensor | None = None,
                      tile_chunk: int = 4) -> torch.Tensor:
    """Pencil-beam dose from an ``L x L`` lattice of rays.

    Tiles are processed in CHUNKS through the grouped convolution rather than one
    at a time.  ``PencilBeamKernelLayer`` and ``BeamWiseConvolutionalLayer`` both
    take a leading batch dimension, so a chunk of tiles costs one kernel build
    and one convolution instead of one each per tile -- measured, the per-tile
    kernel build was a third of the whole loop.  ``tile_chunk`` bounds the extra
    memory, since each tile in flight holds its own masked fluence volume.

    Args:
        fluence_volume: BEV fluence ``[B*G, D, H, W, 1]``.
        bev_density: BEV density ``[B, G, D, H, W]``.
        kernel_layer: ``PencilBeamKernelLayer``; takes ``[N, D, 1]`` depths.
        conv_layer: ``BeamWiseConvolutionalLayer``.
        dense_depth: per-voxel depth ``[B, G, D, H, W]`` in density x cm,
            recomputed when not supplied.
        source_scale: optional ``[B*G, D, H, W]`` multiplier applied to the
            fluence BEFORE convolution -- this is where TERMA scaling enters, and
            it must be applied to the source rather than to the dose, because it
            models how much energy is released at the interaction site.
    Returns:
        ``[B, G, D, H, W]`` dose, unscaled (no mean energy, no MU).
    """
    b, g, d, h, w = bev_density.shape
    # Depths, tile geometry and kernels are PHYSICS -- nothing here is learnable,
    # and PencilBeamModel.get_pencil_beam builds its kernels with
    # ``torch.exp(..., out=K_numer)``, which autograd refuses if the input
    # requires grad. The stock engine detaches for the same reason; only the
    # convolution of the fluence stays in the graph.
    with torch.no_grad():
        if dense_depth is None:
            dense_depth = divergent_radiological_depth(
                bev_density, sad_mm, spacing, iso_center)
        dense_depth = dense_depth.detach()
    flat_depth = dense_depth.reshape(b * g, d, h, w)
    flat_fluence = fluence_volume.reshape(b * g, d, h, w)

    # Tiles are per beam: each has its own aperture, so its own lattice.
    jobs = [(i, tile)
            for i in range(b * g)
            for tile in lattice_tiles(flat_fluence[i], lattice_size, spacing, iso_center)]
    if not jobs:
        return torch.zeros_like(flat_fluence).reshape(b, g, d, h, w)
    # Accumulate per beam OUT OF PLACE. Writing into a preallocated tensor with
    # ``total[i] = ...`` is an in-place index_put_, which autograd rejects once
    # the fluence carries grad ("functions with out=... arguments don't support
    # automatic differentiation") -- and the engine does carry grad in training.
    per_beam_dose: list = [None] * (b * g)

    for start in range(0, len(jobs), max(1, int(tile_chunk))):
        chunk = jobs[start:start + max(1, int(tile_chunk))]
        with torch.no_grad():
            depths = flat_depth.new_zeros((len(chunk), d))
            masks = flat_fluence.new_zeros((len(chunk), d, h, w))
            for k, (i, ((h0, h1), (w0, w1), centre_h, centre_w)) in enumerate(chunk):
                depths[k] = ray_depth_profile(flat_depth[i], centre_h, centre_w,
                                              sad_mm, spacing, iso_center)
                masks[k] = backprojected_tile_mask(
                    (d, h, w), (h0, h1), (w0, w1), sad_mm, spacing, iso_center,
                    flat_fluence.device, flat_fluence.dtype)
            kernels = kernel_layer(
                (depths * DEPTH_CM_TO_KERNEL_UNITS).view(len(chunk), d, 1)).detach()
        # Only this part carries gradient: the fluence is what the rest of the
        # engine (and any correction model downstream) differentiates through.
        source = torch.stack([flat_fluence[i] for i, _t in chunk], dim=0) * masks
        if source_scale is not None:
            source = source * torch.stack([source_scale[i] for i, _t in chunk], dim=0)
        tile_dose = conv_layer(source.unsqueeze(-1), kernels).squeeze(-1)
        for k, (i, _tile) in enumerate(chunk):
            # Residual heterogeneity, relative to THIS tile's ray rather than the
            # central axis -- the difference it has to correct is far smaller.
            residual = torch.exp(-float(mu_eff)
                                 * (flat_depth[i] - depths[k].view(d, 1, 1))).clamp(*cf_clamp)
            contribution = tile_dose[k] * residual
            per_beam_dose[i] = (contribution if per_beam_dose[i] is None
                                else per_beam_dose[i] + contribution)
        del depths, source, kernels, tile_dose
    zero = torch.zeros_like(flat_fluence[0])
    total = torch.stack([x if x is not None else zero for x in per_beam_dose], dim=0)
    return total.reshape(b, g, d, h, w)


class MultilatticeEngine(MultislabEngine):
    """MultislabEngine with an ``L x L`` lattice of rays instead of one.

    ``lattice_size=1`` still differs from the parent engine: its one ray passes
    through the fluence centroid, not the isocentre.
    """

    def __init__(self, *args, lattice_size: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        if int(lattice_size) < 1:
            raise ValueError(f"lattice_size must be >= 1, got {lattice_size}")
        self.lattice_size = int(lattice_size)
