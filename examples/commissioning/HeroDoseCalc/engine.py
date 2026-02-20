"""
Core dose calculation logic: Raytracing, Projection, and Calibration.
"""
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage
import time
import copy

from .hardware import DEVICE, MemoryManager
from .data import ControlPoint, MachineConfig, Phantom
from .nyholm import NyholmBeamModel
from .fluence import FluenceGenerator

class BodyMaskGenerator:
    @staticmethod
    def generate(density_tensor: torch.Tensor, res_mm: float, dilation_mm: float = 5.0) -> torch.Tensor:
        dens_np = density_tensor.squeeze().cpu().numpy()
        mask = dens_np > 0.02
        labeled, n_components = scipy.ndimage.label(mask)
        if n_components > 0:
            sizes = scipy.ndimage.sum(mask, labeled, range(n_components + 1))
            largest_label = np.argmax(sizes[1:]) + 1
            mask = (labeled == largest_label)
        for z in range(mask.shape[0]):
            mask[z, :, :] = scipy.ndimage.binary_fill_holes(mask[z, :, :])
        iter_dilation = int(dilation_mm / res_mm)
        if iter_dilation > 0:
            mask = scipy.ndimage.binary_dilation(mask, iterations=iter_dilation)
        return torch.from_numpy(mask).to(density_tensor.device)

class DoseCalibrator:
    """
    Calibrates the engine by running a full simulation in a virtual water tank.
    Ensures the water surface is positioned exactly 'depth_cm' above the isocenter.
    """
    @staticmethod
    def calibrate(base_config: MachineConfig, 
                  beam_model: NyholmBeamModel, 
                  ref_mu: Optional[float] = None, 
                  ref_dose_gy: Optional[float] = None, 
                  depth_cm: float = 10.0,
                  sparsity: int = 2,
                  raytrace_step_mm: float = 3.0) -> MachineConfig:
        
        print(f"\n🧪 Full Engine Calibration ({base_config.energy})...")
        
        # 1. Resolve Reference Values
        target_mu = ref_mu if ref_mu is not None else getattr(base_config, 'reference_mu', 100.0)
        target_dose = ref_dose_gy if ref_dose_gy is not None else getattr(base_config, 'reference_dose_gy', 1.0)
        print(f"   Target: {target_dose} Gy for {target_mu} MU at 10x10cm field, Depth {depth_cm}cm")

        # 2. Create Virtual Water Tank
        # Dimensions: 40cm wide, 40cm deep (plenty for scatter)
        res = beam_model.res
        size_mm = 400.0
        n_pix = int(np.ceil(size_mm / res))
        if n_pix % 2 == 0: n_pix += 1
        
        phantom = Phantom((n_pix*res, n_pix*res, n_pix*res), res, device=beam_model.device)
        phantom.data[:] = 1.0 # Water density
        
        # --- GEOMETRY SETUP ---
        # We define Isocenter at (0, 0, 0).
        # We want the Isocenter to be at 'depth_cm' inside the water.
        # Therefore, the Surface Y must be at -depth_mm.
        
        depth_mm = depth_cm * 10.0
        surface_y = -depth_mm
        
        # Center X/Z laterally.
        half_size = (n_pix * res) / 2.0
        origin_x = -half_size
        origin_z = -half_size
        
        # Place Origin (Top-Left-Front corner of the box)
        # Origin Y is the surface level. The box extends +400mm from there.
        phantom.origin = torch.tensor([origin_x, surface_y, origin_z], 
                                      dtype=torch.float32, device=beam_model.device)
        
        print(f"   Phantom Surface Y: {surface_y:.1f} mm")
        print(f"   Phantom Bottom Y:  {surface_y + size_mm:.1f} mm")
        print(f"   Isocenter Y:       0.0 mm")

        # 3. Setup Physics & Engine
        temp_config = copy.deepcopy(base_config)
        temp_config.gy_per_mu = 1.0 
        
        gen = FluenceGenerator(res, temp_config, mlc_config=temp_config.mlc, device=beam_model.device)
        engine = DoseEngine(phantom, beam_model)
        
        # 4. Create Reference Beam
        # SAD Setup: Source=-1000, Iso=0.
        cp = ControlPoint.create_manual(
            gantry=0.0, 
            field_size_mm=(100.0, 100.0), 
            iso=(0.0, 0.0, 0.0), 
            mu=target_mu,
            sad=1000.0
        )
        
        # 5. Run Calculation
        print("   Running full raytracing simulation...")
        dose_grid = engine.calculate([cp], gen, batch_size=1, 
                                     sparsity=sparsity, 
                                     raytrace_step_mm=raytrace_step_mm)
        
        # 6. Sample Dose at Isocenter
        # We need to find the voxel index corresponding to (0, 0, 0)
        # origin + (idx + 0.5)*res = 0  =>  idx = (-origin - 0.5*res) / res
        
        # X and Z are centered, so middle index is correct
        idx_xz = n_pix // 2
        
        # Y Index:
        # 0 = surface_y + (idx_y + 0.5) * res
        # -surface_y = (idx_y + 0.5) * res
        # depth_mm = (idx_y + 0.5) * res
        # idx_y = (depth_mm / res) - 0.5
        idx_y = int(round((depth_mm / res) - 0.5))
        
        # Safety Check
        if not (0 <= idx_y < n_pix):
            raise ValueError(f"Calibration depth {depth_cm}cm is outside the phantom volume.")

        # Sample (Z, Y, X) order? 
        # Engine output is (NX, NY, NZ). Origin maps to (X, Y, Z).
        # So we access [idx_x, idx_y, idx_z]
        raw_output = dose_grid[idx_xz, idx_y, idx_xz]
        
        if raw_output == 0:
            raise ValueError("Calibration calculated 0 dose. Check beam/phantom alignment.")

        # 7. Calculate k
        k = target_dose / raw_output
        
        print(f"   - Raw Output at Iso: {raw_output:.4e} (Uncalibrated)")
        print(f"   - New Calibration Factor k: {k:.4e} Gy/MU_signal")
        
        base_config.gy_per_mu = k
        return base_config

class DoseEngine:
    def __init__(self, phantom: Phantom, beam_model: NyholmBeamModel):
        self.phantom = phantom
        self.physics = beam_model
        self.device = phantom.device
        self._raytrace_cache = {}
        self._grid_cache = {}

    def clear_raytrace_cache(self) -> None:
        self._raytrace_cache.clear()
        self._grid_cache.clear()

    @staticmethod
    def map_phys_to_index(phys_vals, grid_1d):
        indices = torch.bucketize(phys_vals, grid_1d)
        indices = torch.clamp(indices, 1, len(grid_1d) - 1)
        lower, upper = grid_1d[indices - 1], grid_1d[indices]
        return (indices - 1).float() + (phys_vals - lower) / (upper - lower + 1e-8)

    def _raytrace_chunked(self, grid_coords_flat, source_pos_batch, density_5d, chunk_size=500000, step_mm=3.0, debug_viz=False):
        n_voxels, batch_size = grid_coords_flat.shape[0], source_pos_batch.shape[0]
        target_dtype = density_5d.dtype 
        rad_depths = torch.zeros((batch_size, n_voxels), dtype=torch.float32, device=self.device)
        
        full_dims = (torch.tensor(self.phantom.shape, device=self.device) * self.phantom.res).to(dtype=target_dtype)
        origin = self.phantom.origin.view(1, 1, 1, 3).to(dtype=target_dtype)
        n_steps = int(torch.norm(full_dims.float()).item() / step_mm)
        t = torch.linspace(0, 1, n_steps, dtype=target_dtype, device=self.device).view(1, 1, -1, 1)

        has_plotted_debug = False

        for i in range(0, n_voxels, chunk_size):
            end = min(i + chunk_size, n_voxels)
            vox = grid_coords_flat[i:end]
            
            rays_d = vox.unsqueeze(0) - source_pos_batch.unsqueeze(1)
            dists = torch.norm(rays_d, dim=-1, keepdim=True)
            
            pts = source_pos_batch.view(batch_size, 1, 1, 3) + t * rays_d.unsqueeze(2)
            pts_norm = 2.0 * ((pts - origin) / full_dims.view(1, 1, 1, 3)) - 1.0
            
            # --- 🔍 ROBUST RAYTRACE BIOPSY ---
            if debug_viz and not has_plotted_debug:
                dists_to_iso = torch.norm(vox, dim=1)
                min_val, min_idx = torch.min(dists_to_iso, dim=0)
                
                if min_val.item() < 1.5:
                    import matplotlib.pyplot as plt
                    has_plotted_debug = True 
                    
                    ray_idx = min_idx.item()
                    target_pos = vox[ray_idx].cpu().numpy()
                    print(f"\n--- 🎯 FOUND ISOCENTER VOXEL ---")
                    print(f"   Chunk: {i} | Voxel Index: {ray_idx}")
                    print(f"   Position: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}] mm")
                    
                    trace_pts = pts_norm[0, ray_idx].detach().cpu().numpy()
                    vol = density_5d[0, 0].detach().cpu().numpy() 
                    D, H, W = vol.shape
                    
                    plt.figure(figsize=(12, 6))
                    
                    iso_z_norm = trace_pts[-1, 2] 
                    slice_idx = int((iso_z_norm + 1) / 2 * D)
                    slice_idx = np.clip(slice_idx, 0, D-1)
                    
                    plt.subplot(1, 2, 1)
                    plt.title(f"Axial Slice (Z-Index={slice_idx})\nTarget: Isocenter")
                    plt.imshow(vol[slice_idx, :, :], cmap='gray', origin='lower', extent=[-1, 1, -1, 1])
                    plt.plot(trace_pts[:, 0], trace_pts[:, 1], 'c-', linewidth=1)
                    plt.scatter(trace_pts[0, 0], trace_pts[0, 1], c='g', s=80, label='Source')
                    plt.scatter(trace_pts[-1, 0], trace_pts[-1, 1], c='r', s=80, label='Iso')
                    plt.legend()
                    
                    iso_y_norm = trace_pts[-1, 1]
                    slice_idx_y = int((iso_y_norm + 1) / 2 * H)
                    slice_idx_y = np.clip(slice_idx_y, 0, H-1)
                    
                    plt.subplot(1, 2, 2)
                    plt.title(f"Coronal Slice (Y-Index={slice_idx_y})\nCheck Vertical Alignment")
                    plt.imshow(vol[:, slice_idx_y, :], cmap='gray', origin='lower', extent=[-1, 1, -1, 1])
                    plt.plot(trace_pts[:, 0], trace_pts[:, 2], 'c-', linewidth=1)
                    plt.scatter(trace_pts[0, 0], trace_pts[0, 2], c='g', s=80)
                    plt.scatter(trace_pts[-1, 0], trace_pts[-1, 2], c='r', s=80)
                    plt.tight_layout()
                    plt.show()

            sample_grid = pts_norm.view(batch_size, -1, n_steps, 1, 3) 
            dens = F.grid_sample(density_5d.expand(batch_size, -1, -1, -1, -1), sample_grid, align_corners=True, padding_mode='zeros')
            dt = dists.view(batch_size, -1, 1) / n_steps
            chunk_res = torch.sum(dens.squeeze(1).squeeze(-1) * dt, dim=-1)
            
            rad_depths[:, i:end] = chunk_res.to(torch.float32)
            
        return rad_depths

    def _get_calculation_bounds(self, beams: List[ControlPoint], margin_mm: float):
        min_z, max_z = float('inf'), float('-inf')
        for b in beams:
            h = abs(b.jaw_y_mm[1] - b.jaw_y_mm[0])
            ext = (h / 2.0) * 1.5 
            min_z = min(min_z, b.isocenter_mm[2] - ext)
            max_z = max(max_z, b.isocenter_mm[2] + ext)
        origin_z = self.phantom.origin[2].item()
        res = self.phantom.res
        s = int((min_z - margin_mm - origin_z) / res)
        e = int((max_z + margin_mm - origin_z) / res)
        return max(0, s), min(self.phantom.shape[2], e)

    def _raytrace_cache_key(
        self,
        *,
        beams: List[ControlPoint],
        z_start: int,
        z_end: int,
        n_active_voxels: int,
        sparsity: int,
        raytrace_step_mm: float,
    ):
        # Cache key assumes phantom density/mask is unchanged for the engine instance.
        phantom_key = (
            tuple(int(v) for v in self.phantom.shape),
            float(self.phantom.res),
            tuple(float(v) for v in self.phantom.origin.cpu().tolist()),
            int(z_start),
            int(z_end),
            int(n_active_voxels),
            int(sparsity),
            float(raytrace_step_mm),
        )
        beam_key = tuple(
            (
                float(b.gantry_angle_deg),
                float(b.couch_angle_deg),
                float(b.collimator_angle_deg),
                float(b.source_distance_mm),
                float(b.isocenter_mm[0]),
                float(b.isocenter_mm[1]),
                float(b.isocenter_mm[2]),
            )
            for b in beams
        )
        return (phantom_key, beam_key)

    def _grid_cache_key(
        self,
        *,
        z_start: int,
        z_end: int,
        sparsity: int,
    ):
        return (
            tuple(int(v) for v in self.phantom.shape),
            float(self.phantom.res),
            tuple(float(v) for v in self.phantom.origin.cpu().tolist()),
            int(z_start),
            int(z_end),
            int(sparsity),
        )

    @staticmethod
    def _inverse_square_correction(rel_vec: torch.Tensor, sad: torch.Tensor) -> torch.Tensor:
        dist = torch.norm(rel_vec, dim=-1).clamp(min=1.0)
        return (sad.view(-1, 1) / dist) ** 2

    def calculate(self, beams: List[ControlPoint], 
                  fluence_generator: FluenceGenerator, 
                  batch_size: int = 5, 
                  crop_margin_mm: float = 50.0,
                  raytrace_step_mm: float = 3.0,
                  sparsity: int = 2,
                  use_fp16: bool = False,
                  profile: bool = False,
                  reuse_raytrace: bool = False,
                  disable_z_crop: bool = False) -> np.ndarray:
        
        # --- 🔧 DEBUG FLAGS ---
        DEBUG_DISABLE_INVSQ  = False 
        DEBUG_FIXED_DEPTH    = False 
        DEBUG_FLAT_FLUENCE   = False 
        DEBUG_SHOW_PLOTS     = False 
        DEBUG_CHECK_INTERP   = True # <--- NEW FLAG to check sparsity accuracy
        # ----------------------

        nx, ny, nz = self.phantom.shape
        res = self.phantom.res
        
        # 1. Z-Crop
        if disable_z_crop:
            z_start, z_end = 0, nz
        else:
            z_start, z_end = self._get_calculation_bounds(beams, crop_margin_mm)
        z_count = z_end - z_start
        if z_count <= 0: return np.zeros((nx, ny, nz), dtype=np.float32)
        print(f"✂️  Cropping Z: [{z_start}:{z_end}] ({z_count} slices)")
        
        # 2. Body Mask + Grids (cached when reusing raytrace)
        full_density_tensor = None
        grid_cache_key = None
        cached_grid = None
        if reuse_raytrace:
            grid_cache_key = self._grid_cache_key(z_start=z_start, z_end=z_end, sparsity=sparsity)
            cached_grid = self._grid_cache.get(grid_cache_key)

        if cached_grid is None:
            full_density_tensor = self.phantom.get_density_tensor()
            density_subset = full_density_tensor[:, :, z_start:z_end, :, :]
            body_mask = BodyMaskGenerator.generate(density_subset, res, dilation_mm=15.0)
            active_mask = body_mask.permute(2, 1, 0)
            n_active_voxels = active_mask.sum().item()
            print(f"🎭 Body Masking: {n_active_voxels} voxels")

            # 3. Grids
            ox, oy, oz = self.phantom.origin
            x = ox + (torch.arange(nx, dtype=torch.float32, device=self.device) + 0.5) * res
            y = oy + (torch.arange(ny, dtype=torch.float32, device=self.device) + 0.5) * res
            z = oz + (torch.arange(z_start, z_end, dtype=torch.float32, device=self.device) + 0.5) * res
            gx, gy, gz = torch.meshgrid(x, y, z, indexing='ij')
            
            grid_flat = torch.stack([gx[active_mask], gy[active_mask], gz[active_mask]], dim=1)
            
            # --- SPARSE GRID SETUP ---
            coarse_grid_flat = None
            coarse_min = None
            coarse_ext = None
            cnx = cny = cnz = 0
            if sparsity > 1:
                cnx = int(np.ceil(nx / sparsity))
                cny = int(np.ceil(ny / sparsity))
                cnz = int(np.ceil(z_count / sparsity))
                
                cx = ox + (torch.linspace(0, nx-1, cnx, device=self.device) + 0.5) * res
                cy = oy + (torch.linspace(0, ny-1, cny, device=self.device) + 0.5) * res
                cz = oz + (z_start + torch.linspace(0, z_count-1, cnz, device=self.device) + 0.5) * res
                
                cgx, cgy, cgz = torch.meshgrid(cx, cy, cz, indexing='ij')
                coarse_grid_flat = torch.stack([cgx.flatten(), cgy.flatten(), cgz.flatten()], dim=1)
                
                # --- FIX: DEFINE BOUNDS BASED ON COARSE GRID ---
                coarse_min = torch.tensor([cx.min(), cy.min(), cz.min()], device=self.device)
                coarse_max = torch.tensor([cx.max(), cy.max(), cz.max()], device=self.device)
                coarse_ext = coarse_max - coarse_min

            if reuse_raytrace and grid_cache_key is not None:
                self._grid_cache[grid_cache_key] = (
                    active_mask,
                    grid_flat,
                    n_active_voxels,
                    coarse_grid_flat,
                    coarse_min,
                    coarse_ext,
                    cnx,
                    cny,
                    cnz,
                )
        else:
            (
                active_mask,
                grid_flat,
                n_active_voxels,
                coarse_grid_flat,
                coarse_min,
                coarse_ext,
                cnx,
                cny,
                cnz,
            ) = cached_grid
        
        subset_dose = torch.zeros(n_active_voxels, dtype=torch.float32, device=self.device)
        dtype_calc = torch.float16 if (use_fp16 and self.device.type == 'cuda') else torch.float32
        
        if dtype_calc == torch.float16:
            grid_flat = grid_flat.half()
            if full_density_tensor is not None:
                full_density_tensor = full_density_tensor.half()
            if coarse_grid_flat is not None and self.device.type == 'cuda':
                coarse_grid_flat = coarse_grid_flat.half()

        if not MemoryManager.check_vram((n_active_voxels, 1, 1), batch_size, len(self.physics.slab_depths)):
            batch_size = max(1, batch_size // 2)

        num_batches = int(np.ceil(len(beams) / batch_size))

        for i in range(num_batches):
            batch = beams[i*batch_size : (i+1)*batch_size]
            B = len(batch)
            
            gantry = torch.tensor([cp.gantry_angle_deg for cp in batch], dtype=dtype_calc, device=self.device)
            couch  = torch.tensor([cp.couch_angle_deg for cp in batch], dtype=dtype_calc, device=self.device)
            coll   = torch.tensor([cp.collimator_angle_deg for cp in batch], dtype=dtype_calc, device=self.device)
            
            sad = torch.tensor([cp.source_distance_mm for cp in batch], dtype=dtype_calc, device=self.device).view(-1, 1)
            iso_np = np.stack([cp.isocenter_mm for cp in batch]) 
            iso = torch.tensor(iso_np, dtype=dtype_calc, device=self.device)
            
            rg, rc, rcol = torch.deg2rad(gantry), torch.deg2rad(couch), torch.deg2rad(coll)
            cg, sg = torch.cos(rg), torch.sin(rg)
            cc, sc = torch.cos(rc), torch.sin(rc)
            ccol, scol = torch.cos(rcol), torch.sin(rcol)
            z_zeros, o = torch.zeros_like(rg), torch.ones_like(rg)
            
            R_g = torch.stack([torch.stack([cg, -sg, z_zeros], 1), torch.stack([sg, cg, z_zeros], 1), torch.stack([z_zeros, z_zeros, o], 1)], 1)
            R_c = torch.stack([torch.stack([cc, z_zeros, sc], 1), torch.stack([z_zeros, o, z_zeros], 1), torch.stack([-sc, z_zeros, cc], 1)], 1)
            R_beam = torch.bmm(R_c, R_g)
            
            z_src = torch.zeros_like(sad)
            src_rel_vec = torch.cat([z_src, -sad, z_src], dim=1).unsqueeze(2) 
            sources = iso + (R_beam @ src_rel_vec).squeeze(2)
            
            vec_axis = (R_beam @ torch.tensor([0.,1.,0.], dtype=dtype_calc, device=self.device).view(1,3,1)).squeeze(2)
            u_base = (R_beam @ torch.tensor([1.,0.,0.], dtype=dtype_calc, device=self.device).view(1,3,1)).squeeze(2)
            v_base = (R_beam @ torch.tensor([0.,0.,1.], dtype=dtype_calc, device=self.device).view(1,3,1)).squeeze(2)
            
            vec_u = u_base * ccol.view(B,1) - v_base * scol.view(B,1)
            vec_v = u_base * scol.view(B,1) + v_base * ccol.view(B,1)
            
            if i == 0 and abs(gantry[0].item() - 90.0) < 1.0:
                print("\n--- 📐 VECTOR CHECK (Gantry 90) ---")
                print("Expected: Axis=[1,0,0], U=[0,1,0], V=[0,0,1]")
                v_ax = vec_axis[0].float().cpu().numpy()
                v_u  = vec_u[0].float().cpu().numpy()
                v_v  = vec_v[0].float().cpu().numpy()
                print(f"  Axis: [{v_ax[0]:.2f}, {v_ax[1]:.2f}, {v_ax[2]:.2f}]")
                print(f"  U:    [{v_u[0]:.2f}, {v_u[1]:.2f}, {v_u[2]:.2f}]")
                print(f"  V:    [{v_v[0]:.2f}, {v_v[1]:.2f}, {v_v[2]:.2f}]")
                print("------------------------------------\n")

            map_res = max(nx, ny, nz)
            fluence_4d = fluence_generator.generate_batch(batch, (map_res, map_res))
            if use_fp16: fluence_4d = fluence_4d.half()
            
            kernels = self.physics.kernel_weights.to(dtype=dtype_calc)
            pad = kernels.shape[-1] // 2
            slabs_4d = F.conv2d(fluence_4d, kernels, padding=pad) 
            
            slabs = slabs_4d.unsqueeze(1)
            
            if DEBUG_FLAT_FLUENCE: slabs = torch.ones_like(slabs)

            # --- RAYTRACING BLOCK ---
            d_rad = None
            if reuse_raytrace and not DEBUG_FIXED_DEPTH:
                cache_key = self._raytrace_cache_key(
                    beams=batch,
                    z_start=z_start,
                    z_end=z_end,
                    n_active_voxels=n_active_voxels,
                    sparsity=sparsity,
                    raytrace_step_mm=raytrace_step_mm,
                )
                cached = self._raytrace_cache.get(cache_key)
                if cached is not None:
                    print("Raytrace cache hit")
                    d_rad = cached.to(device=self.device, dtype=dtype_calc)
                else:
                    print("Raytrace cache miss")

            if d_rad is None and DEBUG_FIXED_DEPTH:
                d_rad = torch.full((B, n_active_voxels), 100.0, dtype=dtype_calc, device=self.device)
            elif d_rad is None and sparsity > 1:
                if full_density_tensor is None:
                    full_density_tensor = self.phantom.get_density_tensor()
                    if dtype_calc == torch.float16:
                        full_density_tensor = full_density_tensor.half()
                print(f"📉 Sparse Raytracing: {sparsity}x factor")
                # 1. Trace Coarse
                d_rad_coarse = self._raytrace_chunked(
                    coarse_grid_flat, sources, full_density_tensor, 
                    chunk_size=100000, step_mm=raytrace_step_mm, debug_viz=(i==0 and DEBUG_SHOW_PLOTS)
                )
                
                # Reshape Volume: (B, 1, Depth=X, Height=Y, Width=Z)
                d_rad_vol = d_rad_coarse.view(B, 1, cnx, cny, cnz).to(dtype=dtype_calc)
                
                # 2. Normalize Fine Points using COARSE Bounds
                c_min_t = coarse_min.to(dtype=dtype_calc)
                c_ext_t = coarse_ext.to(dtype=dtype_calc)

                # Get Normalized (X, Y, Z) coordinates
                norm_raw = 2.0 * ((grid_flat.to(dtype_calc) - c_min_t) / c_ext_t) - 1.0
                
                # --- FIX: SWAP COORDINATES TO (Z, Y, X) ---
                # grid_sample expects (x, y, z) -> (Width, Height, Depth)
                # Our volume is (X, Y, Z). So X is Depth, Y is Height, Z is Width.
                # We must pass (Z, Y, X) to grid_sample.
                norm_x = norm_raw[:, 0]
                norm_y = norm_raw[:, 1]
                norm_z = norm_raw[:, 2]
                
                norm_fine = torch.stack([norm_z, norm_y, norm_x], dim=-1)
                norm_fine = norm_fine.view(1, 1, 1, n_active_voxels, 3)
                
                d_rad = F.grid_sample(d_rad_vol, norm_fine.expand(B, -1, -1, -1, -1), mode='bilinear', align_corners=True, padding_mode='border').view(B, n_active_voxels)
                
                # --- 🔍 DEBUG: INTERPOLATION CHECK ---
                if i == 0 and DEBUG_CHECK_INTERP:
                    mid_idx = n_active_voxels // 2
                    val_interp = d_rad[0, mid_idx].item()
                    
                    one_vox = grid_flat[mid_idx:mid_idx+1]
                    val_exact_t = self._raytrace_chunked(one_vox, sources[0:1], full_density_tensor, chunk_size=1, step_mm=raytrace_step_mm)
                    val_exact = val_exact_t.item()
                    
                    err = abs(val_interp - val_exact)
                    print(f"\n--- 📉 SPARSITY CHECK ---")
                    print(f"   Voxel Index: {mid_idx}")
                    print(f"   Interpolated: {val_interp:.2f} mm")
                    print(f"   Exact Trace:  {val_exact:.2f} mm")
                    print(f"   Error:        {err:.2f} mm")
                    if err > 5.0:
                        print("   ❌ ERROR: Significant Mismatch! Coordinates likely swapped.")
                    else:
                        print("   ✅ MATCH: Interpolation is accurate.")
                    print("-------------------------\n")

            elif d_rad is None:
                if full_density_tensor is None:
                    full_density_tensor = self.phantom.get_density_tensor()
                    if dtype_calc == torch.float16:
                        full_density_tensor = full_density_tensor.half()
                # FIX: grid_coords_flat -> grid_flat
                d_rad = self._raytrace_chunked(grid_flat, sources, full_density_tensor, chunk_size=100000, step_mm=raytrace_step_mm)

            if reuse_raytrace and not DEBUG_FIXED_DEPTH and d_rad is not None:
                if len(self._raytrace_cache) > 8:
                    self._raytrace_cache.clear()
                if cache_key is not None:
                    self._raytrace_cache[cache_key] = d_rad.detach().to(dtype=torch.float32, device="cpu")
            
            # --- PROJECTION ---
            rel = grid_flat.unsqueeze(0) - sources.unsqueeze(1)
            dist_axis = (rel * vec_axis.unsqueeze(1)).sum(-1).clamp(min=1.0)
            mag = sad.view(B, 1) / dist_axis
            
            if DEBUG_DISABLE_INVSQ: inv_sq = torch.ones_like(mag)
            else: inv_sq = self._inverse_square_correction(rel, sad)
            
            u_p = (rel * vec_u.unsqueeze(1)).sum(-1) * mag
            v_p = (rel * vec_v.unsqueeze(1)).sum(-1) * mag
            
            d_indices = self.map_phys_to_index(d_rad.float(), self.physics.slab_depths)
            d_norm = 2.0 * (d_indices / (len(self.physics.slab_depths)-1)) - 1.0
            
            dim_mm_eff = (map_res - 1) * res 
            u_norm = 2.0 * (u_p / dim_mm_eff)
            v_norm = 2.0 * (v_p / dim_mm_eff)
            
            if i == 0 and DEBUG_SHOW_PLOTS:
                import matplotlib.pyplot as plt
                f_map = fluence_4d[0, 0].float().cpu().numpy()
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.title(f"Fluence (Angle {batch[0].gantry_angle_deg})")
                plt.imshow(f_map, origin='upper', cmap='viridis')
                plt.colorbar()
                plt.show()

            grid = torch.stack([u_norm, v_norm, d_norm.to(dtype_calc)], dim=-1)
            grid = grid.view(B, 1, 1, n_active_voxels, 3)
            
            dose_samp = F.grid_sample(slabs, grid, align_corners=True, padding_mode='zeros')
            dose = dose_samp.view(B, n_active_voxels)
            dose *= inv_sq
            
            subset_dose += dose.float().sum(0)
            
            del slabs, d_rad, grid, dose, fluence_4d, slabs_4d
            torch.cuda.empty_cache()

        print("💾 Reassembling full volume...")
        dose_block = torch.zeros((nx, ny, z_count), dtype=torch.float32, device=self.device)
        dose_block[active_mask] = subset_dose
        full_dose = torch.zeros((nx, ny, nz), dtype=torch.float32, device=self.device)
        full_dose[:, :, z_start:z_end] = dose_block
        
        return full_dose.cpu().numpy()
