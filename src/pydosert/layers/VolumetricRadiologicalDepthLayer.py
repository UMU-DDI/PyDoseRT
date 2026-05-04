"""
VolumetricRadiologicalDepthLayer — per-voxel radiological depth via 3D ray tracing.

Unlike :class:`pydosert.layers.RadiologicalDepthLayer.RadiologicalDepthLayer`,
which extracts a *single* radiological-depth profile along the central axis of
each beam, this layer resamples the full CT density volume into beam's-eye-view
(BEV) coordinates and accumulates density along the beam direction at every
voxel. The result is a BEV-shaped radiological-depth volume that captures
lateral as well as longitudinal inhomogeneities.

This per-voxel 3D depth map is the geometry input required for finite-size
pencil-beam (FSPB) algorithms with 3D density correction (Gu et al. 2011) and
for convolution/superposition-style kernel interpolation.

Rays are *divergent* and share the geometry used by
:class:`pydosert.layers.FluenceVolumeLayer.FluenceVolumeLayer`: every BEV
column ``(h, w)`` corresponds to a ray that starts at the source (at distance
SID from iso along the central axis) and fans out as it travels through the
patient. Each BEV voxel ``(h, d, w)`` is therefore mapped to the CT-frame
position the source-iso-detector ray actually passes through at depth ``d``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pydosert.data import MachineConfig


class VolumetricRadiologicalDepthLayer(nn.Module):
    """Compute a per-voxel radiological-depth volume in BEV coordinates.

    The CT density is resampled along divergent rays from the source through
    every BEV column and integrated along the beam axis. The output is aligned
    with the BEV fluence volume produced by
    :class:`pydosert.layers.FluenceVolumeLayer.FluenceVolumeLayer`, sharing the
    same SID-based divergence model so that fluence and rad-depth see the same
    physical points at every BEV voxel.

    Attributes:
        ct_array_shape (tuple): Shape of the CT volume as ``(H, D, W)``.
        resolution (tuple): Voxel spacing in mm ``(rx, ry, rz)``.
        iso_center (tuple): Isocenter in physical mm ``(X, Y, Z)``.
        SID (float): Source-to-isocenter distance in mm.
        gantry_angles (Tensor): Gantry angles in radians, one per beam.
        ct_to_bev_grid (Tensor buffer): Pre-computed sampling grid that maps
            BEV voxel coordinates back to CT voxel coordinates along divergent
            rays. Shape ``[G, D, H, W, 3]`` in ``F.grid_sample`` ``(x, y, z)``
            order, where ``x↔W_ct``, ``y↔D_ct``, ``z↔H_ct``.
        step_lengths (Tensor buffer): Physical path length traversed per BEV
            depth step for every ray, accounting for the off-axis stretch of
            divergent rays. Shape ``[G, H, W]``.
    """

    def __init__(
        self,
        machine_config: MachineConfig,
        resolution: tuple[float, float, float],
        ct_array_shape: tuple[int, int, int],
        gantry_angles: torch.Tensor,
        iso_center: tuple[float, float, float],
        sid: float = 1000.0,
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
        self.machine_config = machine_config
        self.verbose = verbose

        self.ct_array_shape = ct_array_shape
        self.resolution = resolution
        self.iso_center = iso_center
        self.SID = float(sid)
        self.gantry_angles = gantry_angles.to(device=device, dtype=dtype)

        ct_to_bev_grid, step_lengths = self._build_divergent_grid()
        self.register_buffer("ct_to_bev_grid", ct_to_bev_grid)
        self.register_buffer("step_lengths", step_lengths)

    def _build_divergent_grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the [G, D, H, W, 3] sampling grid for divergent rays.

        For every BEV voxel ``(h, d, w)``:

        1. Compute the corresponding physical point relative to the iso-center
           in the BEV frame (gantry=0). The ray that passes through the BEV
           column ``(h, w)`` at iso has lateral position ``(h_iso, w_iso)``
           there; at any other depth it is scaled by ``D_phys / SID`` where
           ``D_phys = SID + (d * res_d - iso_d)`` is the source-to-voxel
           distance along the central axis.
        2. Rotate the ``(d, w)`` components by ``-gantry_angle`` to bring them
           into the CT frame (the ``h`` axis is unaffected by gantry rotation).
        3. Translate back into CT voxel coordinates and normalise to
           ``[-1, 1]`` for ``F.grid_sample``.

        Also pre-compute the per-ray physical step length, which is the
        Euclidean distance between two adjacent BEV-depth samples along the
        ray. Off-axis rays are slightly longer than the central axis step
        ``res_d``; this matters for the cumulative integral over density.
        """
        H, D, W = self.ct_array_shape
        rx, ry, rz = self.resolution
        iso_h, iso_d, iso_w = self.iso_center
        G = self.gantry_angles.shape[0]
        device = self.device
        dtype = self.dtype

        # Physical coordinates of every BEV voxel relative to iso (gantry=0).
        h_phys = (torch.arange(H, device=device, dtype=dtype) * rx) - iso_h  # [H]
        d_phys = (torch.arange(D, device=device, dtype=dtype) * ry) - iso_d  # [D]
        w_phys = (torch.arange(W, device=device, dtype=dtype) * rz) - iso_w  # [W]

        # Source-to-voxel distance along the central beam axis: SID at iso.
        D_phys = self.SID + d_phys  # [D]
        scale = D_phys / self.SID   # [D] divergence factor

        # Broadcast to [D, H, W] BEV positions in the gantry=0 frame, before
        # rotation. h and w lateral positions diverge with depth; d is the
        # central-axis coord and is not scaled (it's the depth itself).
        H_bev = h_phys.view(1, H, 1) * scale.view(D, 1, 1)        # [D, H, W=1]
        W_bev = w_phys.view(1, 1, W) * scale.view(D, 1, 1)        # [D, 1, W]
        D_bev = d_phys.view(D, 1, 1).expand(D, H, W)              # [D, H, W]
        H_bev = H_bev.expand(D, H, W)
        W_bev = W_bev.expand(D, H, W)

        # Rotate (d, w) by -gantry_angle around iso. h is unaffected.
        ang = -self.gantry_angles  # CT->BEV is the inverse of BEV->CT
        cos_a = torch.cos(ang).view(G, 1, 1, 1)
        sin_a = torch.sin(ang).view(G, 1, 1, 1)

        D_bev_b = D_bev.unsqueeze(0)  # [1, D, H, W]
        W_bev_b = W_bev.unsqueeze(0)  # [1, D, H, W]
        H_bev_b = H_bev.unsqueeze(0)  # [1, D, H, W]

        # Rotation by -theta: applying the inverse of BEV<-CT (which uses +theta).
        D_ct = D_bev_b * cos_a + W_bev_b * sin_a
        W_ct = -D_bev_b * sin_a + W_bev_b * cos_a
        H_ct = H_bev_b.expand(G, D, H, W)

        # Translate back to absolute CT physical coords, then to voxel coords.
        h_vox = (H_ct + iso_h) / rx
        d_vox = (D_ct + iso_d) / ry
        w_vox = (W_ct + iso_w) / rz

        # Normalise to [-1, 1] for align_corners=False:
        # norm = (2 * (vox + 0.5) / size) - 1.
        h_norm = (2.0 * (h_vox + 0.5) / H) - 1.0
        d_norm = (2.0 * (d_vox + 0.5) / D) - 1.0
        w_norm = (2.0 * (w_vox + 0.5) / W) - 1.0

        # F.grid_sample 3D expects (x, y, z) where x↔W_in, y↔H_in, z↔D_in.
        # We will feed CT as [N, 1, H, D, W] (so D_in=H, H_in=D, W_in=W) ⇒
        # grid last-dim order is (W_norm, D_norm, H_norm).
        grid = torch.stack([w_norm, d_norm, h_norm], dim=-1)  # [G, D, H, W, 3]

        # Per-ray step length between adjacent BEV depth samples. Off-axis
        # rays cover slightly more physical distance per BEV-depth step than
        # the central axis. Using the iso-plane offsets (h_iso, w_iso) the
        # exact path-length factor is sqrt(1 + (h/SID)^2 + (w/SID)^2).
        step_factor = torch.sqrt(
            1.0
            + (h_phys.view(1, H, 1) / self.SID) ** 2
            + (w_phys.view(1, 1, W) / self.SID) ** 2
        )  # [1, H, W]
        step_lengths = ry * step_factor.expand(G, H, W).contiguous()  # [G, H, W]

        return grid.contiguous(), step_lengths

    def forward(self, ct_stack: torch.Tensor) -> torch.Tensor:
        """Compute BEV-aligned radiological depth for every voxel.

        Args:
            ct_stack: CT density volume of shape ``[B, H, D, W]``.

        Returns:
            Radiological depth volume of shape ``[B*G, D, H, W]`` — same
            spatial layout as the BEV fluence volume (minus the trailing
            channel dimension).
        """
        B, H, D, W = ct_stack.shape
        G = self.gantry_angles.shape[0]

        # Run a 3D grid_sample once per batch element. CT is laid out as
        # [B, H, D, W]; reshape to [B, 1, H, D, W] so the grid dim order
        # (W_norm, D_norm, H_norm) matches.
        ct = ct_stack.unsqueeze(1).to(self.dtype)  # [B, 1, H, D, W]

        # Expand grid to the batch dim and flatten (B, G) so we run a single
        # grid_sample call.
        grid = self.ct_to_bev_grid.unsqueeze(0).expand(B, G, D, H, W, 3)
        grid = grid.reshape(B * G, D, H, W, 3).to(self.dtype)
        ct = ct.unsqueeze(1).expand(B, G, 1, H, D, W).reshape(B * G, 1, H, D, W)

        density_bev = F.grid_sample(
            ct, grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )  # [B*G, 1, D, H, W]
        density_bev = density_bev.squeeze(1)  # [B*G, D, H, W]

        # Per-ray cumulative integral along BEV depth, with the divergent-ray
        # step length. step_lengths is [G, H, W], broadcast over B and D.
        step = self.step_lengths.unsqueeze(1)            # [G, 1, H, W]
        step = step.unsqueeze(0).expand(B, G, 1, H, W)   # [B, G, 1, H, W]
        step = step.reshape(B * G, 1, H, W)              # [B*G, 1, H, W]
        step = step.to(density_bev.dtype)

        contributions = density_bev * step
        cumsum = torch.cumsum(contributions, dim=1)
        # Shift to voxel-centre samples (subtract half of the local step).
        rad_depth = cumsum - 0.5 * contributions

        return rad_depth
