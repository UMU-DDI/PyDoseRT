"""Collapsed-cone convolution/superposition dose engine (differentiable, torch).

Replaces the pencil-beam kernel convolution with genuine 3-D scatter transport:

  1. TERMA / total energy released:  E(r) = rho(r) * Psi(r),  where the primary
     energy fluence Psi(r) = fluence_volume(r) * exp(-mu_att * d_rad(r)) is the
     divergent, inverse-square, attenuated primary (d_rad = divergent radiological
     depth, so lung's reduced attenuation is exact).
  2. Collapsed-cone superposition: the released energy is transported along a set of
     cone directions (48 polar cones x n_azimuth), each with the MC cumulative-cone
     kernel A*exp(-a*s) + B*exp(-b*s). The distance s is the RADIOLOGICAL distance
     along the ray (cumulative density), so heterogeneity AND lateral electron
     disequilibrium (the lung over-prediction) emerge from the transport itself.

The scatter tail (small b) is transported with a numerically stable cumulative-sum
recurrence; the short-range primary (large a, ~4 mm) with a local causal conv.

Everything is grid_sample / cumsum / conv -> fully differentiable. Reuses the
MultislabEngine machinery (fluence pipeline, divergent radiological depth, BEV
density, rotation-to-patient).
"""
import math
import torch
import torch.nn.functional as F

from pydosert.engine.multislab_engine import MultislabEngine, divergent_radiological_depth
from pydosert.physics.kernels.collapsed_cone_kernel import CollapsedConeKernel


def _orthonormal_basis(d):
    """Given a unit 3-vector d (first axis), return a 3x3 rotation whose FIRST column
    is d and the other two complete a right-handed orthonormal basis."""
    d = d / (d.norm() + 1e-9)
    ref = torch.tensor([0.0, 0.0, 1.0], device=d.device, dtype=d.dtype)
    if abs(float(d[2])) > 0.9:
        ref = torch.tensor([0.0, 1.0, 0.0], device=d.device, dtype=d.dtype)
    e1 = torch.cross(d, ref, dim=0); e1 = e1 / (e1.norm() + 1e-9)
    e2 = torch.cross(d, e1, dim=0)
    return torch.stack([d, e1, e2], dim=1)   # columns [d, e1, e2]


class CollapsedConeEngine(MultislabEngine):
    def __init__(self, *args, mu_att: float = 0.06, n_azimuth: int = 4,
                 primary_taps: int = 6, het_gamma: float = 1.0, gain: float = 33.56,
                 prune_frac: float = 0.005, **kwargs):
        # strip MultislabEngine-only knobs we don't use
        kwargs.pop("lateral_scatter", None)
        super().__init__(*args, **kwargs)
        self.gain = gain                # output scale so dose lands on GT x1e5 (fit on all 75 patients)
        self.prune_frac = prune_frac    # drop cones carrying < this fraction of total kernel energy
        self.mu_att = mu_att            # primary attenuation for TERMA (/cm water)
        self.n_azimuth = n_azimuth
        self.primary_taps = primary_taps
        self.het_gamma = het_gamma      # kernel-range density-scaling exponent (rho^gamma); 1=full radiological
        self.kernel = CollapsedConeKernel(device=self.device, dtype=torch.float32)
        self._dir_cache = {}            # (D,H,W) -> (thetas, fwd_grids, inv_grids)

    # ---- cone directions + affine grids (built once per volume shape) --------
    def _cone_setup(self, D, H, W, device):
        key = (D, H, W)
        if key in self._dir_cache:
            return self._dir_cache[key]
        ang = self.kernel.angles_deg.to(device)              # [48] polar, deg
        # per-cone integrated energy A/a + B/b; drop negligible (backscatter) cones for speed
        Aa, aa, Bb, bb = self.kernel.A, self.kernel.a, self.kernel.B, self.kernel.b
        econe = (Aa / aa + Bb / bb)
        keep_mask = (econe >= self.prune_frac * float(econe.sum())).tolist()
        thetas, dirs = [], []
        for ci, th in enumerate(ang):
            if not keep_mask[ci]:
                continue
            thr = float(th) * math.pi / 180.0
            naz = 1 if (float(th) < 1e-3 or float(th) > 179.999) else self.n_azimuth
            for j in range(naz):
                phi = 2 * math.pi * j / naz
                # direction in (D, H, W): D = beam axis (cos theta)
                d = torch.tensor([math.cos(thr), math.sin(thr) * math.cos(phi),
                                  math.sin(thr) * math.sin(phi)], device=device, dtype=torch.float32)
                thetas.append(float(th)); dirs.append(d)
        # build affine grids (affine_grid uses (x,y,z)=(W,H,D) order)
        fwd, inv = [], []
        for d in dirs:
            R = _orthonormal_basis(d)                        # cols [d,e1,e2], (D,H,W) rows
            # reorder rows/cols D,H,W -> W,H,D for affine_grid
            P = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=device, dtype=torch.float32)
            Rg = P @ R @ P.t()
            fwd.append(Rg); inv.append(Rg.t())
        thetas = torch.tensor(thetas, device=device)
        A, a, B, b = self.kernel.interp_at(thetas)
        # per-direction azimuthal weight so a full polar cone sums to h_theta
        counts = {}
        for th in thetas.tolist(): counts[th] = counts.get(th, 0) + 1
        wt = torch.tensor([1.0 / counts[float(t)] for t in thetas.tolist()], device=device)
        packed = (thetas, torch.stack(fwd), torch.stack(inv),
                  A * wt, a, B * wt, b)
        self._dir_cache[key] = packed
        return packed

    def _rotate(self, vol, Rg):
        """vol [N,1,D,H,W]; Rg [3,3] affine rotation (W,H,D order). Returns rotated vol."""
        N = vol.shape[0]
        theta = torch.zeros(N, 3, 4, device=vol.device, dtype=vol.dtype)
        theta[:, :, :3] = Rg.to(vol.dtype).unsqueeze(0)
        grid = F.affine_grid(theta, vol.shape, align_corners=False)
        return F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def _ccc_dose(self, energy, bev_density):
        """energy, bev_density: [B,G,D,H,W]. Collapsed-cone superposition -> dose [B,G,D,H,W]."""
        B, G, D, H, W = energy.shape
        dev = energy.device
        dz_cm = self.dose_grid_spacing[1] / 10.0             # D-axis step (cm)
        thetas, fwd, inv, A, a, Bc, b = self._cone_setup(D, H, W, dev)
        E = energy.reshape(B * G, 1, D, H, W).float()
        rho = bev_density.reshape(B * G, 1, D, H, W).float()
        # short-range primary taps (water steps along the cone axis)
        kdz = torch.arange(self.primary_taps, device=dev, dtype=torch.float32) * dz_cm
        dose = torch.zeros_like(E)
        for i in range(fwd.shape[0]):
            Er = self._rotate(E, fwd[i])[:, 0]               # [N,D,H,W]
            rr = self._rotate(rho, fwd[i])[:, 0].clamp(min=0)
            ds = rr * dz_cm                                   # released-energy weight (T*rho*dz)
            s = torch.cumsum(rr.clamp(min=1e-3) ** self.het_gamma * dz_cm, dim=1)  # kernel radiological distance
            # scatter (long range, small b): stable cumsum recurrence
            bb = float(b[i]); ebs = torch.exp((bb * s).clamp(max=30.0))
            # D_sc[z] = B * exp(-b s[z]) * sum_{j<=z} E[j] ds[j] exp(b s[j])  (b small -> stable)
            Dsc = float(Bc[i]) * torch.exp(-bb * s) * torch.cumsum(Er * ds * ebs, dim=1)
            # primary (large a, radiological range): exact cumsum recurrence in float64
            # (a*s <~ 150 so exp(a*s) is finite in double), captures the full lung electron range
            src = Er * ds                                     # T * ds_rad  [N,D,H,W]
            ai = float(a[i]); Ai = float(A[i])
            s64 = s.double(); eas = torch.exp(ai * s64)
            Dpr = (Ai * torch.exp(-ai * s64) * torch.cumsum(src.double() * eas, dim=1)).to(Er.dtype)
            Dr = (Dpr + Dsc).unsqueeze(1)
            dose = dose + self._rotate(Dr, inv[i])
        return dose.reshape(B, G, D, H, W)

    def _forward_core(self, leaf_positions, mus, jaw_positions, density_image,
                      geometry, collimator_angles, number_of_beams,
                      return_intermediates: bool = False, fluence_maps=None):
        rad_depth_layer, rotation_layer, inv_rot_grid = geometry
        with torch.amp.autocast(self.device.type, dtype=self.dtype):
            if density_image.dim() == 3:
                density_image = density_image.unsqueeze(0)
            G = number_of_beams
            if fluence_maps is not None:
                fm = fluence_maps.reshape(-1, fluence_maps.shape[-2], fluence_maps.shape[-1])
                B = fm.shape[0] // G
            else:
                fm = self.fluence_map_layer(leaf_positions, jaw_positions)
                B = leaf_positions.shape[0]
            from pydosert.geometry.rotations import rotate_2d_images
            if (collimator_angles != 0.0).any():
                fm = rotate_2d_images(fm, collimator_angles, device=self.device, dtype=self.dtype)
            # primary energy fluence volume [B*G, D, H, W] (divergence + inv-square)
            psi = self.fluence_volume_layer(fm).squeeze(-1)
            D, H, W = self.dose_grid_shape[1], self.dose_grid_shape[0], self.dose_grid_shape[2]
            psi = psi.view(B, G, D, H, W)
            if mus is not None:
                psi = psi * mus[:, :, None, None, None]
            with torch.no_grad():
                bev_density = self._density_to_bev(density_image, inv_rot_grid, B, G)
                d_rad = divergent_radiological_depth(bev_density, self.SID, self.dose_grid_spacing,
                                                     self.iso_center, supersample=self.ray_supersample)
            # TERMA T = (mu/rho) * attenuated primary fluence (energy/mass, ~density-independent).
            # The single density factor enters via the radiological step ds = rho*dz inside the
            # transport (source per radiological length = T*ds), so dose comes out per-mass directly.
            # TERMA T = (mu/rho)*attenuated primary fluence (energy/mass). Transport uses the
            # source T*ds_rad along each cone -> dose comes out per-mass directly (the single rho
            # via ds_rad makes uniform-medium dose density-independent; no /rho at deposition).
            terma = psi * torch.exp(-self.mu_att * d_rad)
            dose = self._ccc_dose(terma, bev_density)
            dose = dose * self.gain
            dose = rotation_layer(dose)
            return dose.sum(dim=1).to(self.dtype)
