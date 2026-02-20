"""
Fluence generation utilities for jaws/MLC apertures.
"""
from typing import List, Tuple, Optional, Union
import numpy as np
import torch
import math
import torch.nn.functional as F

from .data import ControlPoint, MachineConfig, MLCConfig
from .hardware import DEVICE


class FluenceGenerator:
    """Generates 2D fluence maps in the collimator frame (unrotated)."""

    def __init__(self, phantom_res: float, config: MachineConfig, mlc_config: Optional[MLCConfig] = None, device=DEVICE):
        self.res = phantom_res
        self.config = config
        self.mlc = mlc_config
        self.device = device
        
        # Prepare physics tensors on device
        self._prepare_kernels()
        self._prepare_luts()

    def _create_gaussian_kernel(self, sigma_mm: Union[float, Tuple[float, float]], size_mm=None):
        if isinstance(sigma_mm, (float, int)):
            sig_x = sig_y = float(sigma_mm)
        else:
            sig_x, sig_y = sigma_mm

        sig_px_x = sig_x / self.res
        sig_px_y = sig_y / self.res
        max_sigma_px = max(sig_px_x, sig_px_y)
        
        if size_mm:
            size_px = int(np.ceil(size_mm / self.res))
        else:
            size_px = int(np.ceil(4 * max_sigma_px)) * 2 + 1
            
        if size_px % 2 == 0:
            size_px += 1
            
        k_half = size_px // 2
        coords = torch.arange(-k_half, k_half + 1, dtype=torch.float32, device=self.device)
        u, v = torch.meshgrid(coords, coords, indexing="ij")
        
        # BUGFIX: meshgrid (indexing='ij') gives u->rows (Y) and v->cols (X), so sigmas must match axes
        exponent = -((u**2) / (2 * sig_px_y**2) + (v**2) / (2 * sig_px_x**2))
        kernel = torch.exp(exponent)
        return (kernel / kernel.sum()).view(1, 1, size_px, size_px)

    def _prepare_kernels(self):
        self.penumbra_kernel = self._create_gaussian_kernel(self.config.geometric_penumbra_mm)
        self.scatter_kernel = self._create_gaussian_kernel(self.config.head_scatter_sigma_mm, size_mm=200.0)

    def _prepare_luts(self):
        """Convert list-based lookup tables to tensors for GPU interpolation."""
        # 1. Radial Profile (GPU Tensor)
        prof = np.array(self.config.profile_curve)
        self.prof_x = torch.tensor(prof[:, 0], dtype=torch.float32, device=self.device)
        self.prof_y = torch.tensor(prof[:, 1], dtype=torch.float32, device=self.device)
        
        # 2. Output Factors (CPU Arrays - calculated per beam, not per pixel)
        sc = np.array(self.config.output_factor_curve)
        self.sc_x_np = sc[:, 0]
        self.sc_y_np = sc[:, 1]

    def _interpolate_profile_gpu(self, r_mm_tensor: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of radial profile on GPU."""
        # Find indices: x[i-1] <= val < x[i]
        indices = torch.bucketize(r_mm_tensor, self.prof_x)
        # Clamp to range [1, len-1]
        indices = torch.clamp(indices, 1, len(self.prof_x) - 1)
        
        x0 = self.prof_x[indices - 1]
        x1 = self.prof_x[indices]
        y0 = self.prof_y[indices - 1]
        y1 = self.prof_y[indices]
        
        # Alpha (0..1)
        # Add epsilon to avoid div/0 if x0==x1 (shouldn't happen in valid LUT)
        alpha = (r_mm_tensor - x0) / (x1 - x0 + 1e-8)
        
        # Interpolate
        return y0 + alpha * (y1 - y0)

    def _get_output_factor(self, s_sq_mm: float) -> float:
        # """Linear interpolation of Sc on CPU."""
        """Linear interpolation of commissioning OF residual correction on CPU."""  #FIXOF
        return float(np.interp(s_sq_mm, self.sc_x_np, self.sc_y_np))


    def _draw_aperture(self, cp: ControlPoint, nx: int, nz: int):
        # 1. SETUP COORDINATE GRID
        # u -> X-direction (Leaf Motion / Width)
        # v -> Y-direction (Leaf Rows / Height)
        u = (torch.arange(nx, device=self.device) - nx/2 + 0.5) * self.res
        v = (torch.arange(nz, device=self.device) - nz/2 + 0.5) * self.res
        
        # --- FIX: Grid shape is now (Height, Width) i.e., (nz, nx) ---
        # We swap the input order to meshgrid to get (Rows, Cols)
        V, U = torch.meshgrid(v, u, indexing='ij')
        
        # 2. JAW MASK
        x1, x2 = cp.jaw_x_mm
        y1, y2 = cp.jaw_y_mm
        
        jx_min, jx_max = min(x1, x2), max(x1, x2)
        jy_min, jy_max = min(y1, y2), max(y1, y2)
        
        vox_half = self.res / 2.0
        
        # X-direction overlap (Using U grid)
        overlap_x = torch.clamp(torch.min(torch.tensor(jx_max), U + vox_half) - 
                                torch.max(torch.tensor(jx_min), U - vox_half), 
                                min=0)
        transmission_x = overlap_x / self.res
        
        # Y-direction overlap (Using V grid)
        overlap_y = torch.clamp(torch.min(torch.tensor(jy_max), V + vox_half) - 
                                torch.max(torch.tensor(jy_min), V - vox_half), 
                                min=0)
        transmission_y = overlap_y / self.res

        mask_jaws = transmission_x * transmission_y
        
        # 3. MLC MASK (With DLG)
        mask_mlc = torch.ones_like(mask_jaws)
        mlc_cfg = getattr(self.config, 'mlc', getattr(self, 'mlc', None))

        if cp.mlc_positions_mm is not None and mlc_cfg is not None:
            boundaries = torch.tensor(mlc_cfg.leaf_boundaries, device=self.device)
            leaves = torch.tensor(cp.mlc_positions_mm, device=self.device)
            
            # V is now (nz, nx), so V_contig works perfectly for row lookup
            V_contig = V.contiguous()
            leaf_indices = torch.bucketize(V_contig, boundaries) - 1
            
            valid_rows = (leaf_indices >= 0) & (leaf_indices < len(leaves))
            safe_indices = torch.clamp(leaf_indices, 0, len(leaves)-1)
            
            # Apply DLG
            dlg_offset = mlc_cfg.dosimetric_leaf_gap_mm / 2.0
            bank_a = leaves[safe_indices, 0] - dlg_offset
            bank_b = leaves[safe_indices, 1] + dlg_offset
            
            # Open Region
            # is_open = (U > bank_a) & (U < bank_b) & valid_rows
            # mask_mlc = torch.where(is_open, 1.0, mlc_cfg.transmission)
            vox_half = self.res / 2.0  #FIXOF
            overlap = torch.clamp(torch.minimum(bank_b, U + vox_half) - torch.maximum(bank_a, U - vox_half), min=0.0)  #FIXOF
            open_frac = torch.clamp(overlap / self.res, 0.0, 1.0)  #FIXOF
            row_mask = mlc_cfg.transmission + open_frac * (1.0 - mlc_cfg.transmission)  #FIXOF
            mask_mlc = torch.where(valid_rows, row_mask, 1.0)  #FIXOF
            
        return mask_jaws * mask_mlc

    def generate_batch(self, control_points: List[ControlPoint], grid_shape_px: Tuple[int, int]):
        nx, nz = grid_shape_px
        
        # --- FIX: Generate (Height, Width) Grid ---
        u = (torch.arange(nx, dtype=torch.float32, device=self.device) - nx / 2 + 0.5) * self.res
        v = (torch.arange(nz, dtype=torch.float32, device=self.device) - nz / 2 + 0.5) * self.res
        
        # Note the swap: (v, u) -> (V, U) with shapes (nz, nx)
        V, U = torch.meshgrid(v, u, indexing="ij")
        
        # Radial Distance (Shape is now nz, nx)
        r_mm = torch.sqrt(U**2 + V**2)
        profile_map = self._interpolate_profile_gpu(r_mm)
        
        maps = []

        for cp in control_points:
            # --- FIX: View shape must match (1, 1, nz, nx) ---
            f_in = self._draw_aperture(cp, nx, nz).view(1, 1, nz, nx)

            pad_p = self.penumbra_kernel.shape[2] // 2
            pad_s = self.scatter_kernel.shape[2] // 2
            
            # Convolutions work fine on (H, W) images
            primary = F.conv2d(f_in, self.penumbra_kernel, padding=pad_p)
            primary *= profile_map
            scatter = F.conv2d(f_in, self.scatter_kernel, padding=pad_s)

            w = self.config.head_scatter_magnitude
            total = ((1.0 - w) * primary + w * scatter).squeeze()

            # Apply Output Factor (Sc)
            jaw_w_mm = abs(cp.jaw_x_mm[1] - cp.jaw_x_mm[0])  #FIXOF
            jaw_h_mm = abs(cp.jaw_y_mm[1] - cp.jaw_y_mm[0])  #FIXOF

            amp = float(self.config.head_scatter_magnitude)  #FIXOF
            sx_iso, sy_iso = self.config.head_scatter_sigma_mm  #FIXOF
            if amp > 0.0 and float(sx_iso) > 0.0 and float(sy_iso) > 0.0:  #FIXOF
                t10_x = math.erf(100.0 / (2.0 * math.sqrt(2.0) * float(sx_iso)))  #FIXOF
                t10_y = math.erf(100.0 / (2.0 * math.sqrt(2.0) * float(sy_iso)))  #FIXOF
                norm = 1.0 + amp * t10_x * t10_y  #FIXOF
                tx = math.erf(jaw_w_mm / (2.0 * math.sqrt(2.0) * float(sx_iso)))  #FIXOF
                ty = math.erf(jaw_h_mm / (2.0 * math.sqrt(2.0) * float(sy_iso)))  #FIXOF
                sc_occ = (1.0 + amp * tx * ty) / norm  #FIXOF
            else:
                sc_occ = 1.0  #FIXOF

            s_eq_mm = (2.0 * jaw_w_mm * jaw_h_mm) / (jaw_w_mm + jaw_h_mm + 1e-6)  #FIXOF
            of_residual = self._get_output_factor(s_eq_mm)  #FIXOF

            maps.append(total * sc_occ * of_residual * self.config.gy_per_mu * cp.monitor_units)  #FIXOF

        # Output shape: (Batch, 1, nz, nx)
        return torch.stack(maps).unsqueeze(1)
