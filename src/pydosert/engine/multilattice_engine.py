"""Multilattice photon dose engine.

:class:`~pydosert.engine.dose_engine.DoseEngine` traces a SINGLE central-axis ray:
every voxel of a depth plane is convolved with the kernel belonging to that one
ray's radiological depth, so laterally heterogeneous anatomy (a beam clipping
lung, say) gets the wrong depth-dose everywhere off axis.

Multilattice keeps the same pencil-beam kernels but uses an ``L x L`` lattice of
rays.  The beam's-eye-view fluence is partitioned into tiles of approximately
EQUAL FLUENCE, each tile gets the kernel set belonging to a ray through its own
fluence-weighted centroid, and only that tile's fluence is convolved with it.
A residual factor then corrects each voxel to its own radiological depth,
referenced to the tile's ray rather than to the central axis, so the difference
it has to absorb is far smaller. By default that factor is the depth dose of the
tile's own field, ``D(d_voxel; R) / D(d_ray; R)``, taken from the same pencil-beam
model that builds the kernels (see :func:`field_depth_dose`). It has no free
parameter, follows the beam quality (TPR 20/10), is correct through buildup, and
adapts to the lattice: smaller tiles fall closer to the primary attenuation rate.

An ``L = 1`` lattice is still not the central-ray calculation: its single ray
passes through the fluence-weighted CENTROID of the aperture rather than through
the isocentre, which is the more representative ray on an off-axis field.
Lattice size trades accuracy against ``L**2`` convolutions.

Units: :func:`divergent_radiological_depth` returns density x CM, while
``PencilBeamKernelLayer`` expects the ``RadiologicalDepthLayer`` scale, which is
density x MM.  :func:`ray_depth_profile` returns cm and the caller multiplies by
``DEPTH_CM_TO_KERNEL_UNITS`` -- keeping the conversion in one named place.
"""
from __future__ import annotations

import math
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn

from pydosert.data import Beam, BeamSequence
from pydosert.engine.photon_base_engine import PhotonBaseEngine
from pydosert.geometry.rotations import build_rotation_grids, rotate_2d_images
from pydosert.layers.BeamRotationLayer import BeamRotationLayer
from pydosert.layers.BeamWiseConvolutionalLayer import BeamWiseConvolutionalLayer
from pydosert.layers.FluenceMapLayer import FluenceMapLayer
from pydosert.layers.FluenceVolumeLayer import FluenceVolumeLayer
from pydosert.layers.PencilBeamKernelLayer import PencilBeamKernelLayer

DEPTH_CM_TO_KERNEL_UNITS = 10.0


def divergent_radiological_depth(bev_density: torch.Tensor, sad_mm: float,
                                 spacing: tuple, iso_center: tuple,
                                 supersample: int = 1,
                                 beam_batch: int = 2) -> torch.Tensor:
    """Per-voxel radiological depth along DIVERGENT rays from the point source.

    The source sits ``sad_mm`` upstream of isocentre, so rays fan out with a
    divergence factor ``z / SAD`` at a plane distance ``z``. Rather than a
    parallel cumsum, we (1) resample the density so each ray becomes a straight
    column (un-diverge by the per-depth lateral scale), (2) cumsum density along
    the ray, (3) resample back to voxels (re-diverge). Two 3D grid_samples plus
    one cumsum -- cheap, and exact for a point source.

    Args:
        bev_density: [B, G, D, H, W] density in beam's-eye-view (beam axis = D).
        sad_mm: source-to-isocentre distance (mm).
        spacing: (rH, rD, rW) voxel spacing (mm).
        iso_center: (X, Y, Z) isocentre in mm (X=H, Y=D, Z=W).
        supersample: lateral ray-grid oversampling factor.
        beam_batch: beams resampled at once. The two grid_samples need a
            ``[N, D, H, W, 3]`` sampling grid, so the transient is proportional to
            this rather than to the beam count; it does not change the result.

    Returns:
        [B, G, D, H, W] radiological depth (density x cm).
    """
    B, G, D, H, W = bev_density.shape
    if beam_batch > 0 and G > beam_batch:
        return torch.cat([
            divergent_radiological_depth(bev_density[:, i:i + beam_batch], sad_mm, spacing,
                                         iso_center, supersample, beam_batch=0)
            for i in range(0, G, beam_batch)], dim=1)
    if supersample > 1:
        k = supersample
        up = F.interpolate(bev_density.reshape(B * G, 1, D, H, W),
                           scale_factor=(1, k, k), mode="trilinear", align_corners=False)
        rH, rD, rW = spacing
        depth = divergent_radiological_depth(up.reshape(B, G, D, H * k, W * k), sad_mm,
                                             (rH / k, rD, rW / k), iso_center, supersample=1)
        return F.interpolate(depth.reshape(B * G, 1, D, H * k, W * k), size=(D, H, W),
                             mode="trilinear", align_corners=False).reshape(B, G, D, H, W)
    dev, dt = bev_density.device, bev_density.dtype
    rH, rD, rW = spacing
    iso_d = iso_center[1] / rD
    iso_h = iso_center[0] / rH
    iso_w = iso_center[2] / rW
    dgrid = torch.arange(D, device=dev, dtype=dt)
    z = sad_mm + (dgrid - iso_d) * rD                          # distance source->plane [D]
    scale = (z / sad_mm).clamp_min(1e-3)                       # divergence factor [D]
    Z, Yh, Xw = torch.meshgrid(dgrid, torch.arange(H, device=dev, dtype=dt),
                               torch.arange(W, device=dev, dtype=dt), indexing="ij")

    def _grid(fac):                                            # fac: [D] lateral scale
        f = fac.view(D, 1, 1)
        h_in = iso_h + (Yh - iso_h) * f
        w_in = iso_w + (Xw - iso_w) * f
        gx = 2 * (w_in + 0.5) / W - 1
        gy = 2 * (h_in + 0.5) / H - 1
        gz = 2 * (Z + 0.5) / D - 1
        return torch.stack([gx, gy, gz], dim=-1).unsqueeze(0).expand(B * G, -1, -1, -1, -1)

    x = bev_density.reshape(B * G, 1, D, H, W)
    dens_par = F.grid_sample(x, _grid(scale), mode="bilinear", padding_mode="zeros",
                             align_corners=False)[:, 0]        # un-diverged [BG,D,H,W]
    step_cm = rD / 10.0
    d_rad_par = torch.cumsum(dens_par, dim=1) * step_cm - 0.5 * dens_par * step_cm
    d_rad = F.grid_sample(d_rad_par.unsqueeze(1), _grid(1.0 / scale), mode="bilinear",
                          padding_mode="border", align_corners=False)[:, 0]   # re-diverged
    return d_rad.reshape(B, G, D, H, W)


def equal_fluence_edges(profile: torch.Tensor, parts: int) -> list[int]:
    """Bin edges splitting a non-negative 1-D profile into ~equal-sum parts.

    Equal fluence rather than equal width: a tile carrying almost no fluence
    contributes almost nothing, so spending a ray on it is wasted, while a
    narrow, intense part of the aperture deserves its own ray.
    """
    n = int(profile.numel())
    if parts <= 1:
        return [0, n]
    cumulative = profile.double().cumsum(0)
    total = cumulative[-1]
    if float(total) <= 0.0:
        return [0, n]
    # One batched searchsorted and one device sync for all interior edges: the
    # per-edge .item() this replaces cost more than the arithmetic.
    quantiles = total * torch.arange(1, parts, device=profile.device,
                                     dtype=cumulative.dtype) / parts
    edges = (torch.searchsorted(cumulative, quantiles) + 1).clamp_(1, n - 1)
    return [0, *sorted(set(edges.tolist())), n]


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
                            iso_center, device, dtype,
                            origin: tuple[int, int] = (0, 0)) -> torch.Tensor:
    """Which voxels belong to a tile, at every depth plane.

    Tile bounds are defined once at the isocentre plane; a voxel belongs to the
    tile if its position mapped BACK to isocentre falls inside them. That makes
    the tiles diverge with the beam, so they stay aligned with the fluence they
    were cut from.

    ``shape`` may describe a lateral CROP of the volume, in which case ``origin``
    gives the crop's (h, w) start in voxels. The back-projection is depth
    dependent, so the crop has to enter as an index offset -- shifting
    ``iso_center`` instead would only be correct at one depth.
    """
    d, h, w = shape
    (iso_h, iso_d, iso_w), (_, r_d, _) = _iso_indices(spacing, iso_center)
    scale = _divergence_scale(d, sad_mm, iso_d, r_d, device, dtype).view(d, 1, 1)
    h_index = torch.arange(h, device=device, dtype=dtype).view(1, h, 1) + float(origin[0])
    w_index = torch.arange(w, device=device, dtype=dtype).view(1, 1, w) + float(origin[1])
    h_at_iso = iso_h + (h_index - iso_h) / scale
    w_at_iso = iso_w + (w_index - iso_w) / scale
    return ((h_at_iso >= h_bounds[0]) & (h_at_iso < h_bounds[1])
            & (w_at_iso >= w_bounds[0]) & (w_at_iso < w_bounds[1]))


def lattice_tiles(fluence_bev: torch.Tensor, lattice_size: int, spacing,
                  iso_center) -> list[tuple]:
    """Cut one beam's fluence into ``L x L`` equal-fluence tiles.

    Returns one ``(h_bounds, w_bounds, centre_h, centre_w, radius_cm)`` per
    non-empty tile: the centre is the fluence-weighted centroid at the isocentre
    plane, and ``radius_cm`` the radius of the circular field with the tile's
    equivalent area, ``sqrt(A_eq / pi)`` with ``A_eq = sum(psi) / max(psi)`` pixels.
    Fluence-weighted rather than the bounding box, so an aperture that only partly
    fills its tile gets the smaller field it really is.
    """
    d, h, w = fluence_bev.shape
    (_, iso_d, _), _ = _iso_indices(spacing, iso_center)
    plane = fluence_bev[int(min(max(iso_d, 0), d - 1))].clamp_min(0.0)
    h_edges = equal_fluence_edges(plane.sum(1), lattice_size)
    w_edges = equal_fluence_edges(plane.sum(0), lattice_size)

    # Tile weights and fluence-weighted centroids for the whole lattice in three
    # reductions, with a single device sync for the emptiness test. Doing this per
    # tile costs O(L^2) syncs per beam, which dominated the forward at L >= 3.
    h_idx = torch.arange(h, device=plane.device, dtype=plane.dtype).view(h, 1)
    w_idx = torch.arange(w, device=plane.device, dtype=plane.dtype).view(1, w)
    def _block_sums(x):
        """Sum x over the lattice blocks -> [len(h_edges)-1, len(w_edges)-1]."""
        rows = torch.stack([x[a:b].sum(0) for a, b in pairwise(h_edges)], dim=0)
        return torch.stack([rows[:, a:b].sum(1) for a, b in pairwise(w_edges)], dim=1)

    def _block_max(x):
        rows = torch.stack([x[a:b].amax(0) for a, b in pairwise(h_edges)], dim=0)
        return torch.stack([rows[:, a:b].amax(1) for a, b in pairwise(w_edges)], dim=1)

    weights = _block_sums(plane)
    centre_h = _block_sums(plane * h_idx)
    centre_w = _block_sums(plane * w_idx)
    nonempty = (weights > 0.0).tolist()                 # the one sync
    safe = weights.clamp_min(torch.finfo(plane.dtype).eps)
    centre_h = centre_h / safe
    centre_w = centre_w / safe
    (r_h, _, r_w) = (float(x) for x in spacing)
    pixel_area_cm2 = (r_h / 10.0) * (r_w / 10.0)
    equivalent_pixels = weights / _block_max(plane).clamp_min(torch.finfo(plane.dtype).eps)
    radius_cm = torch.sqrt(equivalent_pixels * pixel_area_cm2 / math.pi)

    tiles = []
    for r, (h0, h1) in enumerate(pairwise(h_edges)):
        for c, (w0, w1) in enumerate(pairwise(w_edges)):
            if not nonempty[r][c]:
                continue
            tiles.append(((h0, h1), (w0, w1), centre_h[r, c], centre_w[r, c],
                          radius_cm[r, c]))
    return tiles




def field_depth_dose(pencil_beam_model, depth_cm: torch.Tensor,
                     radius_cm: torch.Tensor) -> torch.Tensor:
    """Central-axis depth dose of a circular field, from the pencil-beam model.

    The Nyholm kernel ``K(d, r) = A e^{-ar}/r + B e^{-br}/r`` integrated over a
    field of radius ``R``::

        D(d; R) = A/a (1 - e^{-aR}) + B/b (1 - e^{-bR})        (up to 2 pi)

    Its depth dependence is what the multilattice needs to carry a tile's dose from
    the depth of the tile's ray to a voxel's own depth. It depends on the field
    size: small fields fall at the primary rate, broad fields slower because the
    scatter term builds up with depth. ``R -> 0`` is the pencil-beam limit and
    ``R -> inf`` the broad field ``A/a + B/b``; a tile sits between the two.

    Args:
        pencil_beam_model: a ``PencilBeamModel`` (``PencilBeamKernelLayer.pbm``).
        depth_cm: radiological depth in density x cm, any shape.
        radius_cm: field radius in cm, broadcastable against ``depth_cm``.

    Returns:
        Dose of the same shape as ``depth_cm``, in the model's (unnormalised) units.
    """
    d = depth_cm.float()
    r = radius_cm.float()
    a, b = pencil_beam_model.depth_a(d), pencil_beam_model.depth_b(d)
    return (pencil_beam_model.depth_A_per_a(d) * (1.0 - torch.exp(-a * r))
            + pencil_beam_model.depth_B_per_b(d) * (1.0 - torch.exp(-b * r)))


def tile_crop_bounds(shape, h_bounds, w_bounds, sad_mm, spacing, iso_center,
                     halo: tuple[int, int]) -> tuple[int, int, int, int]:
    """Lateral window holding everything a tile can contribute to.

    The tile bounds are given at the isocentre plane and diverge with depth, so the
    window is the union of the tile over all depth planes, grown by the kernel
    half-width. Because the masked source is exactly zero outside the tile, the
    convolution restricted to this window equals the full-volume convolution there
    and is exactly zero outside it -- cropping is not an approximation.
    """
    d, h, w = shape
    (iso_h, iso_d, iso_w), (_, r_d, _) = _iso_indices(spacing, iso_center)
    z_first = (float(sad_mm) + (0.0 - iso_d) * r_d) / float(sad_mm)
    z_last = (float(sad_mm) + (d - 1 - iso_d) * r_d) / float(sad_mm)
    s_lo, s_hi = max(min(z_first, z_last), 1e-3), max(z_first, z_last)

    def _span(lo, hi, iso, n, pad):
        edges = [iso + (lo - iso) * s for s in (s_lo, s_hi)]
        edges += [iso + (hi - iso) * s for s in (s_lo, s_hi)]
        return (max(0, math.floor(min(edges)) - pad),
                min(n, math.ceil(max(edges)) + pad))

    h0, h1 = _span(h_bounds[0], h_bounds[1], iso_h, h, halo[0])
    w0, w1 = _span(w_bounds[0], w_bounds[1], iso_w, w, halo[1])
    return h0, max(h1, h0 + 1), w0, max(w1, w0 + 1)


def multilattice_dose(fluence_volume: torch.Tensor, bev_density: torch.Tensor,
                      kernel_layer, conv_layer, sad_mm: float, spacing,
                      iso_center, lattice_size: int, mu_eff: float | None = None,
                      cf_clamp: tuple = (0.3, 3.0),
                      dense_depth: torch.Tensor | None = None,
                      source_scale: torch.Tensor | None = None,
                      tile_chunk: int = 4) -> torch.Tensor:
    """Pencil-beam dose from an ``L x L`` lattice of rays.

    Tiles are processed in CHUNKS through the grouped convolution rather than one
    at a time. ``PencilBeamKernelLayer`` and ``BeamWiseConvolutionalLayer`` both
    take a leading batch dimension, so a chunk of tiles costs one kernel build and
    one convolution instead of one each per tile -- measured, the per-tile kernel
    build was a third of the whole loop. ``tile_chunk`` bounds the extra memory,
    since each tile in flight holds its own masked fluence volume.

    Args:
        fluence_volume: BEV fluence ``[B*G, D, H, W, 1]``.
        bev_density: BEV density ``[B, G, D, H, W]``.
        kernel_layer: ``PencilBeamKernelLayer``; takes ``[N, D, 1]`` depths.
        conv_layer: ``BeamWiseConvolutionalLayer``.
        sad_mm: source-to-isocentre distance (mm).
        spacing: (rH, rD, rW) voxel spacing (mm).
        iso_center: (X, Y, Z) isocentre in mm.
        lattice_size: ``L``; the lattice has up to ``L x L`` tiles per beam.
        mu_eff: how voxels whose radiological depth differs from their tile's ray are
            corrected. ``None`` (default) uses the depth dose of the tile's own field,
            ``D(d_voxel; R) / D(d_ray; R)`` with :func:`field_depth_dose` and ``R`` the
            tile's equivalent radius -- no free parameter. A float applies a constant
            attenuation ``exp(-mu_eff * delta)`` per cm of water instead.
        cf_clamp: (min, max) clamp on that correction factor, for stability.
        dense_depth: per-voxel depth ``[B, G, D, H, W]`` in density x cm,
            recomputed when not supplied.
        source_scale: optional ``[B*G, D, H, W]`` multiplier applied to the fluence
            BEFORE convolution -- this is where TERMA scaling enters, and it must be
            applied to the source rather than to the dose, because it models how much
            energy is released at the interaction site.
        tile_chunk: number of tiles convolved per grouped convolution.

    Returns:
        ``[B, G, D, H, W]`` dose, unscaled (no mean energy, no MU).
    """
    b, g = bev_density.shape[:2]
    d, h, w = (dense_depth.shape[-3:] if dense_depth is not None
               else bev_density.shape[-3:])
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

    # Tiles are per beam: each has its own aperture, so its own lattice. The tiling
    # is geometry read off the fluence, not a differentiable function of it.
    with torch.no_grad():
        tiles_per_beam = [lattice_tiles(flat_fluence[i].detach(), lattice_size,
                                        spacing, iso_center)
                          for i in range(b * g)]
    if not any(tiles_per_beam):
        return torch.zeros_like(flat_fluence).reshape(b, g, d, h, w)

    halo = (kernel_layer.pbm.kernel_size_h // 2, kernel_layer.pbm.kernel_size_w // 2)
    # Accumulate per beam OUT OF PLACE. Writing into a preallocated tensor with
    # ``total[i] = ...`` is an in-place index_put_, which autograd rejects once
    # the fluence carries grad ("functions with out=... arguments don't support
    # automatic differentiation") -- and the engine does carry grad in training.
    per_beam_dose: list = [None] * (b * g)

    chunk_size = max(1, int(tile_chunk))
    for i, tiles in enumerate(tiles_per_beam):
        # Chunked WITHIN one beam so every tile in a chunk shares the same depth
        # volume: the ray profiles then come from one batched grid_sample instead
        # of one call per tile.
        for start in range(0, len(tiles), chunk_size):
            chunk = tiles[start:start + chunk_size]
            with torch.no_grad():
                crops = [tile_crop_bounds((d, h, w), t[0], t[1], sad_mm, spacing,
                                          iso_center, halo) for t in chunk]
                # One window size for the chunk so the tiles convolve as one batch;
                # each window is shifted to stay inside the volume.
                ch = max(c[1] - c[0] for c in crops)
                cw = max(c[3] - c[2] for c in crops)
                starts = [(min(c[0], h - ch), min(c[2], w - cw)) for c in crops]

                depths = torch.stack([
                    ray_depth_profile(flat_depth[i], t[2], t[3], sad_mm, spacing, iso_center)
                    for t in chunk], dim=0)                                  # [n, d]
                masks = torch.stack([
                    backprojected_tile_mask((d, ch, cw), t[0], t[1], sad_mm, spacing,
                                            iso_center, flat_fluence.device,
                                            flat_fluence.dtype, origin=(hs, ws))
                    for t, (hs, ws) in zip(chunk, starts)], dim=0)           # [n, d, ch, cw]
                kernels = kernel_layer(
                    (depths * DEPTH_CM_TO_KERNEL_UNITS).view(len(chunk), d, 1)).detach()

            # Only this part carries gradient: the fluence is what the rest of the
            # engine (and any correction model downstream) differentiates through.
            source = torch.stack([flat_fluence[i, :, hs:hs + ch, ws:ws + cw]
                                  for hs, ws in starts], dim=0) * masks
            if source_scale is not None:
                source = source * torch.stack([source_scale[i, :, hs:hs + ch, ws:ws + cw]
                                               for hs, ws in starts], dim=0)
            tile_dose = conv_layer(source.unsqueeze(-1), kernels).squeeze(-1)

            for k, (hs, ws) in enumerate(starts):
                # Residual heterogeneity, relative to THIS tile's ray rather than the
                # central axis -- the difference it has to correct is far smaller.
                voxel_depth = flat_depth[i, :, hs:hs + ch, ws:ws + cw]
                ray_depth = depths[k].view(d, 1, 1)
                if mu_eff is None:
                    radius = chunk[k][4]
                    ray_dose = field_depth_dose(kernel_layer.pbm, ray_depth, radius)
                    residual = (field_depth_dose(kernel_layer.pbm, voxel_depth, radius)
                                / ray_dose.clamp_min(torch.finfo(ray_dose.dtype).tiny))
                else:
                    residual = torch.exp(-float(mu_eff) * (voxel_depth - ray_depth))
                residual = residual.clamp(*cf_clamp).to(tile_dose.dtype)
                contribution = F.pad(tile_dose[k] * residual,
                                     (ws, w - ws - cw, hs, h - hs - ch))
                per_beam_dose[i] = (contribution if per_beam_dose[i] is None
                                    else per_beam_dose[i] + contribution)
            del depths, masks, source, kernels, tile_dose

    zero = torch.zeros_like(flat_fluence[0])
    total = torch.stack([x if x is not None else zero for x in per_beam_dose], dim=0)
    return total.reshape(b, g, d, h, w)


class MultilatticeEngine(PhotonBaseEngine):
    """Pencil-beam dose engine using an ``L x L`` lattice of rays per beam.

    A sibling of :class:`~pydosert.engine.dose_engine.DoseEngine`: same layers,
    same fluence model, same kernels, but the kernel depth is taken from a lattice
    of rays through the aperture instead of a single central-axis ray, and the
    residual per-voxel depth difference is corrected with the kernel model's own
    depth dependence (or a constant attenuation, if ``mu_eff`` is given).

    Usage mirrors DoseEngine::

        engine = MultilatticeEngine(machine_config, kernel_size, spacing, shape,
                                    lattice_size=3)
        dose = engine.compute_dose(beam_sequence, density_image)
    """

    def __init__(self, *args, lattice_size: int = 3, mu_eff: float | None = None,
                 cf_clamp: tuple[float, float] = (0.3, 3.0), tile_chunk: int = 4,
                 ray_supersample: int = 1, source_scale_layer: nn.Module | None = None,
                 **kwargs):
        """
        Args:
            lattice_size: ``L``; up to ``L x L`` equal-fluence tiles per beam.
            mu_eff: residual depth correction within a tile. ``None`` (default)
                uses the depth dose of each tile's own field from the pencil-beam
                model -- no free parameter, follows TPR 20/10; a float applies a
                constant attenuation per cm of water instead.
            cf_clamp: (min, max) clamp on that correction factor.
            tile_chunk: tiles convolved per grouped convolution (memory knob).
            ray_supersample: lateral oversampling of the divergent depth grid.
            source_scale_layer: optional module called as
                ``layer(fluence_maps, bev_density)`` returning a ``[B*G, D, H, W, 1]``
                multiplier applied to the fluence before convolution -- e.g.
                :class:`~pydosert.layers.TermaScalingLayer`. None disables it.

        Remaining arguments are those of :class:`PhotonBaseEngine`.
        """
        if int(lattice_size) < 1:
            raise ValueError(f"lattice_size must be >= 1, got {lattice_size}")
        self.lattice_size = int(lattice_size)
        self.mu_eff = mu_eff
        self.cf_clamp = cf_clamp
        self.tile_chunk = int(tile_chunk)
        self.ray_supersample = int(ray_supersample)
        # Assigned only after super(): nn.Module refuses submodule assignment before its
        # own __init__, and a class-level default would shadow the registered submodule.
        # super().__init__ may already run a forward pass (auto_calibrate), so
        # _forward_core reads this with getattr. That calibration is in water, where the
        # TERMA scaling is identity, so it is unaffected by the layer being attached after.
        super().__init__(*args, **kwargs)
        self.source_scale_layer = source_scale_layer

    def _initialize_layers(self, new_beam_data: BeamSequence | Beam, overwrite: bool = False) -> None:
        """Build or refresh the pipeline layers from a beam template.

        The same layers as DoseEngine minus the RadiologicalDepthLayer (the lattice
        traces its own divergent rays), plus the inverse rotation grid that maps the
        patient density into beam's-eye-view.
        """
        if new_beam_data is None:
            return

        initialize_fluence_map_layer = not hasattr(self, 'fluence_map_layer')
        initialize_fluence_volume_layer = not hasattr(self, 'fluence_volume_layer')
        initialize_beam_wise_conv_layer = not hasattr(self, 'beam_wise_conv_layer')
        initialize_pencil_beam_kernel_layer = not hasattr(self, 'pencil_beam_kernel_layer')
        initialize_rotation_layer = not hasattr(self, 'rotation_layer')

        if isinstance(new_beam_data, Beam):
            number_of_beams = 1
            gantry_angles = torch.tensor([new_beam_data.gantry_angle]).to(self.dtype).to(self.device)
            collimator_angles = torch.tensor([new_beam_data.collimator_angle]).to(self.dtype).to(self.device)
        elif isinstance(new_beam_data, BeamSequence):
            number_of_beams = len(new_beam_data)
            gantry_angles = new_beam_data.gantry_angles
            collimator_angles = new_beam_data.collimator_angles.to(self.dtype).to(self.device)

        if self.dtype is None:
            self.dtype = new_beam_data.dtype
        if self.device is None:
            self.device = new_beam_data.device

        if (self.number_of_beams is None or self.number_of_beams != number_of_beams
                or self.gantry_angles is None or (self.gantry_angles != gantry_angles).any()
                or self.collimator_angles is None or (self.collimator_angles != collimator_angles).any()):
            initialize_rotation_layer = True
        self.number_of_beams = number_of_beams
        self.gantry_angles = gantry_angles
        self.collimator_angles = collimator_angles

        if self.field_size is None or (self.field_size != new_beam_data.field_size):
            initialize_fluence_map_layer = True
            initialize_fluence_volume_layer = True
        self.field_size = new_beam_data.field_size

        self.SID = new_beam_data.sid
        if self.iso_center is None or (self.iso_center != new_beam_data.iso_center):
            initialize_fluence_volume_layer = True
            initialize_rotation_layer = True
        self.iso_center = new_beam_data.iso_center

        if (self.dtype is None or self.device is None or self.dose_grid_shape is None
                or self.dose_grid_spacing is None or self.number_of_beams is None):
            return

        if initialize_fluence_map_layer:
            self.fluence_map_layer = FluenceMapLayer(
                self.machine_config, device=self.device, dtype=self.dtype,
                field_size=self.field_size, verbose=self.verbose)

        if initialize_fluence_volume_layer:
            self.fluence_volume_layer = FluenceVolumeLayer(
                self.machine_config, device=self.device, dtype=self.dtype,
                resolution=self.dose_grid_spacing, ct_array_shape=self.dose_grid_shape,
                sid=self.SID, iso_center=self.iso_center, field_size=self.field_size,
                verbose=self.verbose)

        if initialize_pencil_beam_kernel_layer:
            self.pencil_beam_kernel_layer = PencilBeamKernelLayer(
                self.machine_config, device=self.device, dtype=self.dtype,
                resolution=self.dose_grid_spacing, kernel_size=self.kernel_size,
                verbose=self.verbose)

        if initialize_beam_wise_conv_layer:
            self.beam_wise_conv_layer = BeamWiseConvolutionalLayer(
                self.device, self.dtype, verbose=self.verbose)

        if initialize_rotation_layer:
            self.rotation_layer = BeamRotationLayer(
                self.machine_config, device=self.device, dtype=self.dtype,
                ct_array_shape=self.dose_grid_shape, gantry_angles=self.gantry_angles,
                iso_center=self.iso_center, resolution=self.dose_grid_spacing,
                verbose=self.verbose)

        self.inv_rot_grid = self._inverse_rotation_grid(self.gantry_angles)
        self.layers_initialized = True

    def _inverse_rotation_grid(self, gantry_angles: torch.Tensor) -> torch.Tensor:
        """Grid mapping the patient volume into beam's-eye-view (inverse of BeamRotationLayer)."""
        H, D, W = self.dose_grid_shape
        return build_rotation_grids(
            (1, gantry_angles.shape[0], D, H, W), -gantry_angles,
            self.device, self.dtype, iso_center=self.iso_center,
            resolution=self.dose_grid_spacing,
        )

    def _density_to_bev(self, density_image: torch.Tensor, inv_grid: torch.Tensor,
                        B: int, G: int) -> torch.Tensor:
        """Resample patient density [B, H, D, W] into per-beam BEV [B, G, D, H, W]."""
        H, D, W = density_image.shape[1], density_image.shape[2], density_image.shape[3]
        dp = density_image.unsqueeze(1).expand(B, G, H, D, W).reshape(B * G * H, 1, D, W)
        grid = inv_grid.repeat(B, 1, H, 1, 1, 1).reshape(B * G * H, D, W, 2).to(dp.dtype)
        rot = F.grid_sample(dp, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return rot.reshape(B, G, H, D, W).permute(0, 1, 3, 2, 4)      # [B, G, D, H, W]

    def _full_geometry(self) -> tuple[nn.Module, torch.Tensor]:
        """Geometry context for the full beam set: rotation layer and inverse grid."""
        return (self.rotation_layer, self.inv_rot_grid)

    def _build_chunk_geometry(self, chunk_size: int) -> list[tuple[int, int, tuple]]:
        """Per-chunk (rotation layer, inverse rotation grid); cached by the base class."""
        chunks = []
        for start in range(0, self.number_of_beams, chunk_size):
            end = min(start + chunk_size, self.number_of_beams)
            gantry_angles = self.gantry_angles[start:end]
            rotation_layer = BeamRotationLayer(
                self.machine_config, device=self.device, dtype=self.dtype,
                ct_array_shape=self.dose_grid_shape, gantry_angles=gantry_angles,
                iso_center=self.iso_center, resolution=self.dose_grid_spacing,
                verbose=self.verbose)
            chunks.append((start, end, (rotation_layer, self._inverse_rotation_grid(gantry_angles))))
        return chunks

    def _forward_core(self, leaf_positions, mus, jaw_positions, density_image,
                      geometry, collimator_angles, number_of_beams,
                      return_intermediates: bool = False, fluence_maps=None):
        """Run the multilattice pipeline for a (possibly partial) set of beams.

        Returns a dose tensor [B, D, H, W] summed over the given beams; with
        return_intermediates, a tuple (bev_density, fluence_maps, fluence_volumes, dose).
        """
        rotation_layer, inv_rot_grid = geometry
        with torch.amp.autocast(self.device.type, dtype=self.dtype):
            if density_image.dim() == 3:
                density_image = density_image.unsqueeze(0)
            G = number_of_beams

            if fluence_maps is not None:
                if fluence_maps.dim() == 4:
                    B = fluence_maps.shape[0]
                    batched_fluence_maps = fluence_maps.reshape(
                        B * G, fluence_maps.shape[2], fluence_maps.shape[3])
                else:
                    B = fluence_maps.shape[0] // G
                    batched_fluence_maps = fluence_maps
            else:
                batched_fluence_maps = self.fluence_map_layer(leaf_positions, jaw_positions)
                B = leaf_positions.shape[0]

            if (collimator_angles != 0.0).any():
                batched_fluence_maps = rotate_2d_images(
                    batched_fluence_maps, collimator_angles,
                    device=self.device, dtype=self.dtype)

            batched_fluence_volumes = self.fluence_volume_layer(batched_fluence_maps)

            with torch.no_grad():
                bev_density = self._density_to_bev(density_image, inv_rot_grid, B, G)
                dense_depth = divergent_radiological_depth(
                    bev_density, self.SID, self.dose_grid_spacing, self.iso_center,
                    supersample=self.ray_supersample)

            source_scale = None
            scale_layer = getattr(self, "source_scale_layer", None)
            if scale_layer is not None:
                source_scale = scale_layer(batched_fluence_maps, bev_density).squeeze(-1)

            # multilattice_dose only reads bev_density for its shape once dense_depth
            # is supplied, so the second full BEV volume can go here rather than being
            # held for the whole tile loop.
            bev_shape = bev_density.shape
            if not return_intermediates:
                del bev_density
                bev_density = torch.empty(bev_shape[:2] + (0, 0, 0), device=dense_depth.device,
                                          dtype=dense_depth.dtype)

            dose = multilattice_dose(
                batched_fluence_volumes, bev_density,
                self.pencil_beam_kernel_layer, self.beam_wise_conv_layer,
                self.SID, self.dose_grid_spacing, self.iso_center,
                self.lattice_size, self.mu_eff, cf_clamp=self.cf_clamp,
                dense_depth=dense_depth, source_scale=source_scale,
                tile_chunk=self.tile_chunk)

            dose = dose * self.machine_config.mean_photon_energy_MeV
            if mus is not None:
                dose = dose * mus[:, :, None, None, None]

            dose = rotation_layer(dose)
            dose = dose.sum(dim=1).to(self.dtype)

        if return_intermediates:
            return bev_density, batched_fluence_maps, batched_fluence_volumes, dose
        return dose
