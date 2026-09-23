"""Per-pencil radiological depth in the beam's-eye view.

``RadiologicalDepthLayer`` traces ONE ray per gantry angle -- the central axis --
and every voxel of a depth plane is given that ray's depth. That is exact for a
flat surface at normal incidence and wrong everywhere else: where the skin is
oblique or curved, a voxel just under its own skin is handed the central axis's
post-build-up depth, and where the central axis has not yet entered the patient
the whole plane is given a depth of zero.

This layer gives every pencil (BEV voxel column) its own depth. The patient
density is resampled into each beam's BEV frame with the exact inverse of
``BeamRotationLayer`` -- the same grid construction at ``-theta`` -- so the
density lands on the very grid the fluence and the BEV dose live on. The depth
is then the midpoint-rule cumulative sum along the beam axis, identical in form
and units to ``RadiologicalDepthLayer`` (density x mm), so the central-axis
pencil reproduces that layer's result.

Rays are parallel to the central axis. At VMAT prostate field sizes the lateral
entry-point error this leaves is ``x * dz / SAD`` -- about 3 mm at 3 cm off-axis
and 10 cm depth -- which is below the 2 mm grid's resolution where it matters,
at the skin.
"""

import torch
import torch.nn.functional as F
from torch import nn

from pydosert.geometry.rotations import build_rotation_grids


class PencilDepthLayer(nn.Module):
    """Radiological depth of every BEV pencil, for a fixed set of gantry angles.

    Args:
        ct_array_shape: Patient grid shape ``(H, D, W)``.
        iso_center: Isocentre in mm, ``(h, d, w)``.
        resolution: Voxel spacing in mm, ``(res_H, res_D, res_W)``.
        gantry_angles: ``[G]`` gantry angles in radians.
        device: Torch device.
        dtype: Dtype the depth is stored in; it is computed in float32.
    """

    def __init__(self, ct_array_shape, iso_center, resolution, gantry_angles,
                 device=None, dtype=torch.float32) -> None:
        super().__init__()
        self.ct_array_shape = tuple(int(x) for x in ct_array_shape)
        self.resolution = tuple(float(x) for x in resolution)
        self.device = device
        self.dtype = dtype
        H, D, W = self.ct_array_shape
        angles = gantry_angles.to(device=device, dtype=torch.float32)
        # The inverse of BeamRotationLayer: its grid maps patient -> BEV sampling
        # positions at +theta; at -theta it maps BEV -> patient positions, which
        # is what resampling the patient INTO the BEV frame needs.
        self.inv_grid = build_rotation_grids((1, angles.shape[0], D, H, W), -angles,
                                             device, torch.float32, iso_center=iso_center,
                                             resolution=resolution)      # [1, G, 1, D, W, 2]
        self.step_mm = self.resolution[1]

    def _bev_one(self, slices: torch.Tensor, g: int, B: int, H: int, D: int, W: int) -> torch.Tensor:
        """Patient density slices ``[B*H, 1, D, W]`` resampled into beam ``g``'s BEV, ``[B, D, H, W]``."""
        grid = self.inv_grid[0, g, 0].to(slices.dtype)                      # [D, W, 2]
        grid = grid.unsqueeze(0).expand(B * H, D, W, 2)
        rot = F.grid_sample(slices, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=False)                            # [B*H, 1, D, W]
        return rot.view(B, H, D, W).permute(0, 2, 1, 3)                     # [B, D, H, W]

    def bev_density(self, density: torch.Tensor) -> torch.Tensor:
        """Resample a patient density ``[B, H, D, W]`` into ``[B, G, D, H, W]`` BEV."""
        B, H, D, W = density.shape
        G = self.inv_grid.shape[1]
        out = torch.empty((B, G, D, H, W), device=density.device, dtype=density.dtype)
        slices = density.reshape(B * H, 1, D, W)
        for g in range(G):
            out[:, g] = self._bev_one(slices, g, B, H, D, W)
        return out

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        """Per-pencil radiological depth ``[B*G, D, H, W]`` in density x mm.

        Beams are resampled and summed one at a time, in float32, and stored in
        the layer's dtype, so the transient is one beam's volume rather than a
        few copies of all of them.

        Args:
            density: Patient relative density ``[B, H, D, W]``.

        Returns:
            Depth at each BEV voxel centre, by the same midpoint rule as
            ``RadiologicalDepthLayer``: ``(cumsum(rho) - rho / 2) * step``.
        """
        with torch.no_grad():
            B, H, D, W = density.shape
            G = self.inv_grid.shape[1]
            depth = torch.empty((B, G, D, H, W), device=density.device, dtype=self.dtype)
            slices = density.float().reshape(B * H, 1, D, W)
            for g in range(G):
                rho = self._bev_one(slices, g, B, H, D, W)
                d = torch.cumsum(rho, dim=1)
                d.sub_(rho, alpha=0.5).mul_(self.step_mm)
                depth[:, g] = d
                del rho, d
            return depth.reshape(B * G, D, H, W)
