"""Multislab (equivalent-path-length) photon dose engine.

A sibling of :class:`DoseEngine` that adds a per-voxel radiological-depth
heterogeneity correction on top of the pencil-beam dose. The pencil beam scales
dose only along the CENTRAL-AXIS radiological depth (one ray per beam), so
off-axis lung/tissue gets the wrong depth-dose (the source of the thorax
over-shoot). Here we compute the radiological depth PER VOXEL — the CT density
rotated into beam's-eye-view and integrated along the beam axis, i.e. dense
parallel rays — and rescale each voxel by the ratio of its own primary
attenuation to the central-axis attenuation:

    dose_het(x) = dose_pb(x) * exp( -mu_eff * ( d_rad(x) - d_rad_central_axis ) )

This is the O'Connor / equivalent-TAR multislab correction. It is far cheaper
than a collapsed cone and fixes the depth-direction heterogeneity (the dominant
error in lung); it does NOT model lateral scatter broadening.

Because it only adds a post-multiply to DoseEngine's output, it inherits the
entire pencil-beam pipeline and the PhotonBaseEngine scaffolding unchanged;
``mu_eff`` is the single calibratable knob.
"""
import torch
from torch import nn
import torch.nn.functional as F

from pydosert.data import Beam, BeamSequence
from pydosert.engine.dose_engine import DoseEngine
from pydosert.layers.RadiologicalDepthLayer import RadiologicalDepthLayer
from pydosert.layers.BeamRotationLayer import BeamRotationLayer
from pydosert.geometry.rotations import build_rotation_grids, rotate_2d_images


def divergent_radiological_depth(bev_density: torch.Tensor, sad_mm: float,
                                 spacing: tuple, iso_center: tuple,
                                 supersample: int = 1) -> torch.Tensor:
    """Per-voxel radiological depth along DIVERGENT rays from the point source.

    We know the source is at ``sad_mm`` upstream of isocentre, so rays fan out with
    a divergence factor ``z/SAD`` at depth-plane distance ``z``. Rather than a
    parallel cumsum, we (1) resample the density so each ray becomes a straight
    column (un-diverge by the per-depth lateral scale), (2) cumsum density along the
    ray, (3) resample the result back to voxels (re-diverge). Two 3D grid_samples +
    one cumsum — cheap, and exact for a point source.

    Args:
        bev_density: [B, G, D, H, W] density in beam's-eye-view (beam axis = D).
        sad_mm: source-to-isocentre distance (mm).
        spacing: (rH, rD, rW) voxel spacing (mm).
        iso_center: (X, Y, Z) isocentre in mm (X=H, Y=D, Z=W).
    Returns:
        [B, G, D, H, W] radiological depth (g/cm^2, density x cm).
    """
    B, G, D, H, W = bev_density.shape
    if supersample > 1:
        # finer lateral ray grid: upsample density, compute depth, downsample.
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
                               torch.arange(W, device=dev, dtype=dt), indexing="ij")  # [D,H,W]

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


def _gaussian_blur_lateral(x: torch.Tensor, sigma_vox: float) -> torch.Tensor:
    """Separable 2D Gaussian blur over the lateral (H, W) axes of [B, G, D, H, W]."""
    if sigma_vox < 0.3:
        return x
    k = int(2 * round(3 * sigma_vox) + 1)
    c = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-0.5 * (c / sigma_vox) ** 2)
    g = g / g.sum()
    B, G, D, H, W = x.shape
    xf = x.reshape(B * G * D, 1, H, W)
    xf = F.conv2d(xf, g.view(1, 1, k, 1), padding=(k // 2, 0))
    xf = F.conv2d(xf, g.view(1, 1, 1, k), padding=(0, k // 2))
    return xf.reshape(B, G, D, H, W)


def lateral_scatter_correction(dose_bev: torch.Tensor, bev_density: torch.Tensor,
                               lat_sigma_mm: float, res_lat_mm: float,
                               lat_cap_mm: float = None) -> torch.Tensor:
    """Density-scaled lateral scatter for a BEV dose [B, G, D, H, W]: spread the dose in
    low-density regions with an extra Gaussian ~ sigma*(1/rho - 1) (electron range grows
    as 1/density), optionally capped. Dose-conserving. Shared by MultislabEngine and the
    training CorrectedDoseEngine so both apply identical physics."""
    out = dose_bev
    for rho, lo, hi in ((0.30, 0.0, 0.45), (0.60, 0.45, 0.75), (0.85, 0.75, 0.92)):
        m = ((bev_density >= lo) & (bev_density < hi)).to(dose_bev.dtype)
        d_bin = dose_bev * m
        extra_sigma_mm = lat_sigma_mm * (1.0 / rho - 1.0)
        if lat_cap_mm is not None:
            extra_sigma_mm = min(extra_sigma_mm, lat_cap_mm)
        out = out - d_bin + _gaussian_blur_lateral(d_bin, extra_sigma_mm / res_lat_mm)
    return out


class MultislabEngine(DoseEngine):
    """Pencil-beam engine + per-voxel radiological-depth heterogeneity correction.

    Usage mirrors DoseEngine:
        engine = MultislabEngine(machine_config, kernel_size, spacing, shape, mu_eff=0.05)
        dose = engine.compute_dose(beam_sequence, density_image)
    """

    def __init__(self, *args, mu_eff: float = 0.05, cf_clamp: tuple = (0.3, 3.0),
                 ray_supersample: int = 1, lateral_scatter: bool = False,
                 lat_sigma_mm: float = 3.0, lat_cap_mm: float = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu_eff = mu_eff                 # effective linear attenuation [/cm water]
        self.cf_clamp = cf_clamp             # clamp the correction factor for stability
        self.ray_supersample = ray_supersample   # lateral ray-grid oversampling factor
        self.lateral_scatter = lateral_scatter   # density-scaled lateral scatter broadening
        self.lat_sigma_mm = lat_sigma_mm     # base lateral scatter sigma in water (mm)
        self.lat_cap_mm = lat_cap_mm         # cap on the extra lateral sigma (mm); None = uncapped

    def _lateral_scatter(self, dose_bev: torch.Tensor, bev_density: torch.Tensor) -> torch.Tensor:
        return lateral_scatter_correction(dose_bev, bev_density, self.lat_sigma_mm,
                                          self.dose_grid_spacing[0], self.lat_cap_mm)

    # --- density -> beam's-eye-view (inverse of BeamRotationLayer) ---
    def _inverse_rotation_grid(self, gantry_angles: torch.Tensor) -> torch.Tensor:
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

    def _delta_rad(self, bev_density):
        """mu-independent depth term (d_rad - d_rad_cax), [B, G, D, H, W]. The
        correction factor is exp(-mu_eff * delta), so sweeping mu_eff only needs this
        computed once (the expensive part is the divergent radiological depth)."""
        H, D, W = self.dose_grid_shape
        # per-voxel radiological depth along DIVERGENT rays from the point source
        d_rad = divergent_radiological_depth(bev_density, self.SID, self.dose_grid_spacing,
                                             self.iso_center, supersample=self.ray_supersample)
        # central-axis reference at the isocentre's lateral position
        iso_h = int(min(max(self.iso_center[0] / self.dose_grid_spacing[0], 0), H - 1))
        iso_w = int(min(max(self.iso_center[2] / self.dose_grid_spacing[2], 0), W - 1))
        d_rad_cax = d_rad[:, :, :, iso_h:iso_h + 1, iso_w:iso_w + 1]            # [B,G,D,1,1]
        return d_rad - d_rad_cax

    def _finalize(self, base, bev_density, delta, rotation_layer, mu):
        """Apply the mu_eff correction (+ optional lateral scatter), rotate BEV->patient
        and sum beams. Split out of _forward_core so a mu_eff sweep can reuse `base`/`delta`."""
        cf = torch.exp(-mu * delta).clamp(self.cf_clamp[0], self.cf_clamp[1])
        dose = base * cf
        if self.lateral_scatter:
            dose = self._lateral_scatter(dose, bev_density)
        dose = rotation_layer(dose)
        return dose.sum(dim=1).to(self.dtype)

    # --- geometry: DoseEngine's (rad-depth, rotation) + the inverse-rotation grid ---
    def _initialize_layers(self, new_beam_data, overwrite: bool = False) -> None:
        super()._initialize_layers(new_beam_data, overwrite)
        if self.layers_initialized and self.gantry_angles is not None:
            self.inv_rot_grid = self._inverse_rotation_grid(self.gantry_angles)

    def _full_geometry(self):
        return (self.rad_depth_layer, self.rotation_layer, self.inv_rot_grid)

    def _build_chunk_geometry(self, chunk_size: int):
        chunks = []
        for start in range(0, self.number_of_beams, chunk_size):
            end = min(start + chunk_size, self.number_of_beams)
            gantry_angles = self.gantry_angles[start:end]
            rad_depth_layer = RadiologicalDepthLayer(
                self.machine_config, device=self.device, dtype=self.dtype,
                resolution=self.dose_grid_spacing, ct_array_shape=self.dose_grid_shape,
                gantry_angles=gantry_angles, iso_center=self.iso_center, verbose=self.verbose)
            rotation_layer = BeamRotationLayer(
                self.machine_config, device=self.device, dtype=self.dtype,
                ct_array_shape=self.dose_grid_shape, gantry_angles=gantry_angles,
                iso_center=self.iso_center, resolution=self.dose_grid_spacing, verbose=self.verbose)
            inv_grid = self._inverse_rotation_grid(gantry_angles)
            chunks.append((start, end, (rad_depth_layer, rotation_layer, inv_grid)))
        return chunks

    def _base_forward(self, leaf_positions, mus, jaw_positions, density_image,
                      geometry, collimator_angles, number_of_beams, fluence_maps=None):
        """Everything up to (but excluding) the mu_eff correction: the base pencil-beam
        dose in BEV [B,G,D,H,W], the BEV density, and the mu-independent depth term."""
        rad_depth_layer, rotation_layer, inv_rot_grid = geometry
        if density_image.dim() == 3:
            density_image = density_image.unsqueeze(0)
        with torch.no_grad():
            batched_radiological_depths = rad_depth_layer(density_image).detach()
            batched_kernels = self.pencil_beam_kernel_layer(batched_radiological_depths).detach()
        del batched_radiological_depths
        H, D, W = self.dose_grid_shape
        G = number_of_beams

        if fluence_maps is not None:
            if fluence_maps.dim() == 4:
                B = fluence_maps.shape[0]
                batched_fluence_maps = fluence_maps.reshape(B * G, fluence_maps.shape[2], fluence_maps.shape[3])
            else:
                B = fluence_maps.shape[0] // G
                batched_fluence_maps = fluence_maps
        else:
            batched_fluence_maps = self.fluence_map_layer(leaf_positions, jaw_positions)
            B = leaf_positions.shape[0]

        if (collimator_angles != 0.0).any():
            batched_fluence_maps = rotate_2d_images(
                batched_fluence_maps, collimator_angles, device=self.device, dtype=self.dtype)

        batched_fluence_volumes = self.fluence_volume_layer(batched_fluence_maps)
        base = self.beam_wise_conv_layer(batched_fluence_volumes, batched_kernels)
        base.mul_(self.machine_config.mean_photon_energy_MeV)
        del batched_fluence_volumes, batched_fluence_maps, batched_kernels

        D_, H_, W_, _ = base.shape[1:]
        base = base.view(B, G, D_, H_, W_)
        if mus is not None:
            base.mul_(mus[:, :, None, None, None])

        with torch.no_grad():
            bev_density = self._density_to_bev(density_image, inv_rot_grid, B, G)
            delta = self._delta_rad(bev_density)
        return base, bev_density, delta, rotation_layer

    def _forward_core(self, leaf_positions, mus, jaw_positions, density_image,
                      geometry, collimator_angles, number_of_beams,
                      return_intermediates: bool = False, fluence_maps=None):
        with torch.amp.autocast(self.device.type, dtype=self.dtype):
            base, bev_density, delta, rotation_layer = self._base_forward(
                leaf_positions, mus, jaw_positions, density_image, geometry,
                collimator_angles, number_of_beams, fluence_maps)
            return self._finalize(base, bev_density, delta, rotation_layer, self.mu_eff)

    def compute_dose_multi_mu(self, beam_input, density_image, mu_list, overwrite: bool = False):
        """Dose for several mu_eff values at once, sharing the base pencil-beam dose and
        the (expensive) divergent radiological depth. Returns [len(mu_list), 1, D, H, W].
        Intended for calibration sweeps — orders of magnitude cheaper than one forward
        per mu_eff."""
        self._initialize_layers(beam_input, overwrite)
        ct = density_image
        if ct is not None and ct.dim() == 3:
            ct = ct.unsqueeze(0)
        if isinstance(beam_input, Beam):
            leaf_positions = beam_input.leaf_positions.unsqueeze(0).unsqueeze(0)
            mus = beam_input.mu.unsqueeze(0).unsqueeze(0)
            jaw_positions = beam_input.jaw_positions.unsqueeze(0).unsqueeze(0)
        else:
            leaf_positions = beam_input.leaf_positions.unsqueeze(0)
            mus = beam_input.mus.unsqueeze(0)
            jaw_positions = beam_input.jaw_positions.unsqueeze(0)
        geometry = self._full_geometry()
        with torch.amp.autocast(self.device.type, dtype=self.dtype), torch.no_grad():
            base, bev_density, delta, rotation_layer = self._base_forward(
                leaf_positions, mus, jaw_positions, ct, geometry,
                self.collimator_angles, self.number_of_beams)
            out = [self._finalize(base, bev_density, delta, rotation_layer, float(mu))
                   for mu in mu_list]
        return torch.stack(out, dim=0)
