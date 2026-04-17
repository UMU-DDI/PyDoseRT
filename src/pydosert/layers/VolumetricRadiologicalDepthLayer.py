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

The integration is performed with parallel rays inside BEV (each (h, w) column
is summed along the BEV depth axis). This is the common FSPB approximation and
is consistent with how :class:`BeamRotationLayer` rotates the accumulated dose
back to CT coordinates: both layers treat rotation as a purely planar operation
in the depth-width plane.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pydosert.data import MachineConfig
from pydosert.geometry.rotations import build_rotation_grids


class VolumetricRadiologicalDepthLayer(nn.Module):
    """Compute a per-voxel radiological-depth volume in BEV coordinates.

    The CT density is rotated into BEV (the gantry frame in which the beam
    travels along the depth axis) and cumulatively integrated along that axis.
    The output is aligned with the BEV fluence volume produced by
    :class:`pydosert.layers.FluenceVolumeLayer.FluenceVolumeLayer`.

    Attributes:
        ct_array_shape (tuple): Shape of the CT volume as ``(H, D, W)``.
        resolution (tuple): Voxel spacing in mm ``(rx, ry, rz)``.
        iso_center (tuple): Isocenter in physical mm ``(X, Y, Z)``.
        gantry_angles (Tensor): Gantry angles in radians, one per beam.
        ct_to_bev_grid (Tensor buffer): Pre-computed sampling grid that maps
            CT voxel coordinates into the BEV frame. Shape
            ``[1, G, 1, D, W, 2]``.
    """

    def __init__(
        self,
        machine_config: MachineConfig,
        resolution: tuple[float, float, float],
        ct_array_shape: tuple[int, int, int],
        gantry_angles: torch.Tensor,
        iso_center: tuple[float, float, float],
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
        self.gantry_angles = gantry_angles.to(device=device, dtype=dtype)

        H, D, W = ct_array_shape
        # CT -> BEV is the inverse rotation of BEV -> CT used by
        # BeamRotationLayer; negating the gantry angles flips the rotation sign
        # (the translation terms for iso-center-centred rotation in
        # build_rotation_grids are symmetric under angle negation).
        ct_to_bev_grid = build_rotation_grids(
            (1, self.gantry_angles.shape[0], D, H, W),
            -self.gantry_angles,
            device=self.device,
            dtype=self.dtype,
            iso_center=iso_center,
            resolution=resolution,
        )  # [1, G, 1, D, W, 2]

        self.register_buffer("ct_to_bev_grid", ct_to_bev_grid)

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

        # Expand CT over the beam dimension and flatten to run grid_sample once
        # per (B, G, H) slice. Each H slice is rotated independently in the
        # D-W plane, matching BeamRotationLayer's convention.
        ct_expanded = ct_stack.unsqueeze(1).expand(B, G, H, D, W)
        ct_flat = ct_expanded.reshape(B * G * H, 1, D, W).to(self.dtype)

        grid = (
            self.ct_to_bev_grid
            .expand(B, G, H, D, W, 2)
            .reshape(B * G * H, D, W, 2)
            .to(self.dtype)
        )

        density_bev = F.grid_sample(
            ct_flat, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # [B*G*H, 1, D, W]
        density_bev = density_bev.view(B, G, H, D, W)

        # Rearrange into BEV fluence-volume layout [B*G, D, H, W].
        density_bev = density_bev.permute(0, 1, 3, 2, 4).reshape(B * G, D, H, W)

        step = float(self.resolution[1])
        # Voxel-centre integration: cumulative sum of density along depth gives
        # the density integrated up to the voxel's exit face; subtracting half
        # the local voxel contribution shifts the sample to the voxel centre.
        cumsum = torch.cumsum(density_bev, dim=1) * step
        rad_depth = cumsum - density_bev * step * 0.5

        return rad_depth
