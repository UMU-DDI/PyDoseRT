import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple



import numpy as np
import torch
from scipy.interpolate import interp1d
from typing import List, Tuple, Optional

def compute_profile_ratios(
    measured_profile: np.ndarray,
    modelled_profile: np.ndarray,
    x_scale_mm: np.ndarray,
    sample_points_mm: List[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ratios of measured/modelled profiles at specific radial distances.
    
    Args:
        measured_profile: 1D array of measured intensity values
        modelled_profile: 1D array of modelled intensity values
        x_scale_mm: 1D array of x-positions in mm corresponding to profiles
        sample_points_mm: List of radial distances (mm) where ratios should be sampled
        
    Returns:
        Tuple of (sample_points_mm as array, corresponding ratio values)
    """
    # Compute the ratio
    ratio = measured_profile / (modelled_profile + 1e-10)  # Small epsilon to avoid division by zero
    
    # Create interpolation function
    interpolator = interp1d(
        x_scale_mm, 
        ratio, 
        kind='cubic',
        bounds_error=False,
        fill_value='extrapolate'
    )
    
    # Sample at requested points
    sample_points = np.array(sample_points_mm)
    sampled_ratios = interpolator(sample_points)
    
    return sample_points, sampled_ratios

def precompute_head_scatter_kernel(sigma_cm, resolution_cm, kernel_half_width=5):
    """
    Create a 1D Gaussian kernel for convolution.
    
    Parameters:
    -----------
    sigma_cm : float
        Standard deviation in cm
    resolution_cm : float
        Pixel resolution in cm
    kernel_half_width : float
        Number of sigmas to extend the kernel on each side
        
    Returns:
    --------
    kernel : numpy array
        Normalized 1D Gaussian kernel
    """
    # Convert sigma to pixels
    sigma_pixels = sigma_cm / resolution_cm
    
    # Kernel extent in pixels (e.g., 5 sigmas on each side)
    n_pixels = int(np.ceil(kernel_half_width * sigma_pixels))
    
    # Create coordinate array
    x = np.arange(-n_pixels, n_pixels + 1)
    
    # Gaussian kernel
    kernel = np.exp(-0.5 * (x / sigma_pixels)**2)
    
    # Normalize so sum = 1
    kernel = kernel / np.sum(kernel)
    
    return torch.from_numpy(kernel)

def get_output_factor(field_size_mlc_mm, field_size_jaw_mm, output_factors):

    x = torch.Tensor(output_factors[0]).to(field_size_mlc_mm.device).to(field_size_mlc_mm.dtype)
    y = torch.Tensor(output_factors[1]).to(field_size_mlc_mm.device).to(field_size_mlc_mm.dtype)

    # Get insertion indices
    idx_mlc = torch.searchsorted(x, field_size_mlc_mm, right=False)

    # Clamp to valid range (so we can interpolate/extrapolate)
    idx1_mlc = torch.clamp(idx_mlc, 1, len(x) - 1)
    idx0_mlc = idx1_mlc - 1

    x0 = x[idx0_mlc]
    x1 = x[idx1_mlc]
    y0 = y[idx0_mlc]
    y1 = y[idx1_mlc]

    # Linear interpolation
    t = (field_size_mlc_mm - x0) / (x1 - x0)
    OF_mlc = y0 + t * (y1 - y0)


    # Get insertion indices
    idx_mlc = torch.searchsorted(x, field_size_jaw_mm, right=False)

    # Clamp to valid range (so we can interpolate/extrapolate)
    idx1_mlc = torch.clamp(idx_mlc, 1, len(x) - 1)
    idx0_mlc = idx1_mlc - 1

    x0 = x[idx0_mlc]
    x1 = x[idx1_mlc]
    y0 = y[idx0_mlc]
    y1 = y[idx1_mlc]

    # Linear interpolation
    t = (field_size_jaw_mm - x0) / (x1 - x0)
    OF_jaw = y0 + t * (y1 - y0)
    return (OF_mlc + OF_jaw) / 2

def create_radial_correction_map(
    sample_distances_mm: np.ndarray,
    sample_ratios: np.ndarray,
    image_shape: Tuple[int, int],
    pixel_size_mm: float,
    center: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """
    Create a 2D radial correction map from 1D sampled ratios.
    
    Args:
        sample_distances_mm: 1D array of radial distances in mm
        sample_ratios: 1D array of ratio values at corresponding distances
        image_shape: Tuple of (height, width) for output image
        pixel_size_mm: Physical size of each pixel in mm
        center: Optional tuple of (y_center, x_center) in pixels. 
                If None, uses image center.
        
    Returns:
        torch.Tensor of shape image_shape with radially interpolated ratios
    """
    height, width = image_shape
    
    # Determine center
    if center is None:
        cy, cx = height / 2.0, width / 2.0
    else:
        cy, cx = center
    
    # Create coordinate grids
    y = torch.arange(height, dtype=torch.float32)
    x = torch.arange(width, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # Compute radial distance from center in pixels
    radial_distance_pixels = torch.sqrt((yy - cy)**2 + (xx - cx)**2)
    
    # Convert to mm
    radial_distance_mm = radial_distance_pixels * pixel_size_mm
    
    # Interpolate ratios at each pixel's radial distance
    # Convert sample points to torch for interpolation
    sample_distances_torch = torch.tensor(
        sample_distances_mm, dtype=torch.float32
    )
    sample_ratios_torch = torch.tensor(
        sample_ratios, dtype=torch.float32
    )
    
    # Flatten the radial distance map for interpolation
    radial_flat = radial_distance_mm.flatten()
    
    # Perform 1D interpolation using torch
    correction_flat = torch_interp1d(
        sample_distances_torch,
        sample_ratios_torch,
        radial_flat
    )
    
    # Reshape back to image shape
    correction_map = correction_flat.reshape(image_shape)
    
    return correction_map


def torch_interp1d(
    x: torch.Tensor,
    y: torch.Tensor,
    x_new: torch.Tensor
) -> torch.Tensor:
    """
    1D linear interpolation in PyTorch (similar to numpy.interp).
    
    Args:
        x: 1D tensor of x-coordinates (must be sorted)
        y: 1D tensor of y-coordinates
        x_new: 1D tensor of new x-coordinates to interpolate at
        
    Returns:
        Interpolated y values at x_new positions
    """
    # Ensure x is sorted
    if not torch.all(x[1:] >= x[:-1]):
        raise ValueError("x coordinates must be sorted")
    
    # Find indices for interpolation
    indices = torch.searchsorted(x, x_new, right=False)
    indices = torch.clamp(indices, 1, len(x) - 1)
    
    # Get surrounding points
    x0 = x[indices - 1]
    x1 = x[indices]
    y0 = y[indices - 1]
    y1 = y[indices]
    
    # Linear interpolation
    slope = (y1 - y0) / (x1 - x0 + 1e-10)
    y_new = y0 + slope * (x_new - x0)
    
    # Handle extrapolation (use edge values)
    y_new = torch.where(x_new < x[0], y[0], y_new)
    y_new = torch.where(x_new > x[-1], y[-1], y_new)
    
    return y_new



class LearnableFluenceKernel(nn.Module):
    """
    Learnable 2D convolution to model all fluence-space effects:
    - Source penumbra
    - MLC scatter  
    - Head scatter
    - T&G effect
    - Any calibration errors
    """
    def __init__(self, kernel_size=15):
        super().__init__()
        
        # Initialize as delta function (no smoothing)
        kernel = torch.zeros(kernel_size, kernel_size)
        kernel[kernel_size//2, kernel_size//2] = 1.0
        
        # Make it learnable
        self.kernel = nn.Parameter(kernel.unsqueeze(0).unsqueeze(0))  # [1, 1, K, K]
        
        # Optional: Learn a global scaling factor too
        self.scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, fluence_map):
        """
        Apply learned kernel to fluence map.
        
        Args:
            fluence_map: [B*G, H, W] fluence map
        
        Returns:
            corrected_fluence: [B*G, H, W]
        """
        # Normalize kernel to sum to 1 (preserve total fluence)
        kernel_normalized = (self.kernel / (self.kernel.sum() + 1e-8)).to(fluence_map.device)
        
        # Apply convolution
        fluence_4d = fluence_map.unsqueeze(1)  # [B*G, 1, H, W]
        pad = self.kernel.shape[-1] // 2
        fluence_padded = F.pad(fluence_4d, (pad, pad, pad, pad), mode='replicate')
        fluence_corrected = F.conv2d(fluence_padded, kernel_normalized)
        
        # Apply learnable scaling
        fluence_corrected = fluence_corrected * self.scale
        
        return fluence_corrected.squeeze(1)  # [B*G, H, W]
    



# ============================================================================
# Precomputation functions for efficient forward passes
# ============================================================================

def precompute_source_penumbra_kernel(desired_penumbra_fwhm_mm: float,
                               device: torch.device,
                               dtype: torch.dtype) -> torch.Tensor:
    """
    Precompute a 1D Gaussian kernel that produces a desired penumbra width (FWHM)
    in millimeters at the isocenter plane.

    Args:
        desired_penumbra_fwhm_mm: Penumbra width (FWHM) in mm, typically 2–4 mm.
        device: Device to create kernel on.
        dtype: Torch dtype.

    Returns:
        kernel: [1, 1, 1, K] separable 1D convolution kernel.
    """

    # Convert physical FWHM to Gaussian sigma in pixel units
    sigma_pixels = desired_penumbra_fwhm_mm  / 2.355

    # Kernel size: 6σ rule of thumb (covers >99% of Gaussian energy)
    kernel_size = int(6 * sigma_pixels) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Coordinates centered at zero
    x = torch.linspace(-(kernel_size//2), kernel_size//2,
                       kernel_size, device=device, dtype=dtype)

    kernel_1d = torch.exp(-(x**2) / (2 * sigma_pixels**2))
    kernel_1d /= kernel_1d.sum()

    return kernel_1d.view(1, 1, 1, kernel_size)


def precompute_directional_source_penumbra_kernels(
    penumbra_fwhm_mlc_mm: float,
    penumbra_fwhm_jaw_mm: float,
    device: torch.device,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute two separate 1D Gaussian kernels for MLC and JAW directions
    with different penumbra widths (FWHM) in millimeters.

    Physical basis: The penumbra width can differ between MLC and JAW directions
    due to different geometric factors, leaf design, and collimator characteristics.

    Args:
        penumbra_fwhm_mlc_mm: Penumbra width (FWHM) in MLC direction (horizontal/width) in mm.
        penumbra_fwhm_jaw_mm: Penumbra width (FWHM) in JAW direction (vertical/height) in mm.
        device: Device to create kernels on.
        dtype: Torch dtype.

    Returns:
        kernel_mlc: [1, 1, 1, K_mlc] 1D convolution kernel for MLC direction (horizontal)
        kernel_jaw: [1, 1, K_jaw, 1] 1D convolution kernel for JAW direction (vertical)
    """
    # MLC direction kernel (horizontal)
    sigma_mlc_pixels = penumbra_fwhm_mlc_mm / 2.355
    kernel_size_mlc = int(6 * sigma_mlc_pixels) + 1
    if kernel_size_mlc % 2 == 0:
        kernel_size_mlc += 1

    x_mlc = torch.linspace(-(kernel_size_mlc//2), kernel_size_mlc//2,
                           kernel_size_mlc, device=device, dtype=dtype)
    kernel_mlc_1d = torch.exp(-(x_mlc**2) / (2 * sigma_mlc_pixels**2))
    kernel_mlc_1d /= kernel_mlc_1d.sum()
    kernel_mlc = kernel_mlc_1d.view(1, 1, 1, kernel_size_mlc)

    # JAW direction kernel (vertical)
    sigma_jaw_pixels = penumbra_fwhm_jaw_mm / 2.355
    kernel_size_jaw = int(6 * sigma_jaw_pixels) + 1
    if kernel_size_jaw % 2 == 0:
        kernel_size_jaw += 1

    x_jaw = torch.linspace(-(kernel_size_jaw//2), kernel_size_jaw//2,
                           kernel_size_jaw, device=device, dtype=dtype)
    kernel_jaw_1d = torch.exp(-(x_jaw**2) / (2 * sigma_jaw_pixels**2))
    kernel_jaw_1d /= kernel_jaw_1d.sum()
    kernel_jaw = kernel_jaw_1d.view(1, 1, kernel_size_jaw, 1)

    return kernel_mlc, kernel_jaw



# ============================================================================
# Fast application functions using precomputed kernels/masks
# ============================================================================


def apply_directional_precomputed_kernel(
    fluence: torch.Tensor,
    kernel_mlc: torch.Tensor,
    kernel_jaw: torch.Tensor,
    padding_mode: str = 'replicate'
) -> torch.Tensor:
    """
    Apply directional precomputed 1D convolution kernels using separable convolution
    with different sigma values for MLC and JAW directions.

    This allows modeling different penumbra widths in the two orthogonal directions,
    which is physically accurate since MLC and JAW geometries differ.

    Args:
        fluence: [B, 1, H, W] fluence map
        kernel_mlc: [1, 1, 1, K_mlc] 1D convolution kernel for MLC direction (horizontal/width)
        kernel_jaw: [1, 1, K_jaw, 1] 1D convolution kernel for JAW direction (vertical/height)
        padding_mode: Padding mode for convolution

    Returns:
        fluence_convolved: [B, 1, H, W] convolved fluence map
    """
    # Apply MLC direction convolution (horizontal, along width dimension)
    kernel_mlc_size = kernel_mlc.shape[-1]
    pad_mlc = kernel_mlc_size // 2
    fluence_padded_mlc = F.pad(fluence, (pad_mlc, pad_mlc, 0, 0), mode=padding_mode)
    fluence_mlc = F.conv2d(fluence_padded_mlc, kernel_mlc)

    # Apply JAW direction convolution (vertical, along height dimension)
    # kernel_jaw is already in the correct shape [1, 1, K_jaw, 1]
    kernel_jaw_size = kernel_jaw.shape[-2]
    pad_jaw = kernel_jaw_size // 2
    fluence_padded_jaw = F.pad(fluence_mlc, (0, 0, pad_jaw, pad_jaw), mode=padding_mode)
    fluence_convolved = F.conv2d(fluence_padded_jaw, kernel_jaw)

    return fluence_convolved


def estimate_field_size_1d(fluence_1d: torch.Tensor, pixel_size_mm: float = 1.0, threshold: float = 0.5) -> torch.Tensor:
    """
    Estimate the effective field size from a 1D fluence profile.

    Uses the width at threshold (default 50%) to determine field size.

    Args:
        fluence_1d: [B, W] 1D fluence profile
        pixel_size_mm: Pixel size in mm
        threshold: Threshold for field edge detection (fraction of max)

    Returns:
        field_size_cm: [B] effective field size in cm
    """
    B, W = fluence_1d.shape

    # Normalize each profile
    max_val = fluence_1d.max(dim=1, keepdim=True)[0] + 1e-10
    normalized = fluence_1d / max_val

    # Find width above threshold
    above_threshold = (normalized > threshold).float()
    width_pixels = above_threshold.sum(dim=1)

    return width_pixels * pixel_size_mm

def make_interpolator(point_dict):
    # Sort keys and values into tensors
    xs = torch.tensor([vals[0] for vals in point_dict], dtype=torch.float32)
    ys = torch.tensor([vals[1] for vals in point_dict], dtype=torch.float32)

    def interpolate(x):
        """
        x: tensor of any shape
        returns: tensor of same shape with interpolated values
        """

        # Ensure xs, ys are on the same device as x
        _xs = xs.to(x.device)
        _ys = ys.to(x.device)

        # searchsorted gives index of the right bin
        idx = torch.searchsorted(_xs, x)

        # Clamp to valid interpolation range
        idx = torch.clamp(idx, 1, len(_xs) - 1)

        x0 = _xs[idx - 1]
        x1 = _xs[idx]
        y0 = _ys[idx - 1]
        y1 = _ys[idx]

        # Linear interpolation
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)
    
    return interpolate


def apply_head_scatter_kernels(fluence_map, kernel_x, kernel_y, scatter_amplitude, scatter_fraction):
    """
    Pure PyTorch implementation using conv2d (faster on GPU).
    
    Parameters:
    -----------
    fluence_map : torch.Tensor
        2D fluence map [H, W] or [1, 1, H, W]
    kernel_x, kernel_y : numpy.ndarray or torch.Tensor
        1D convolution kernels
    scatter_fraction : float
        Scatter fraction S
        
    Returns:
    --------
    fluence_total : torch.Tensor
        Total fluence with scatter
    """
    device = fluence_map.device
    dtype = fluence_map.dtype
    
    # Ensure input is 4D [N, C, H, W]
    if fluence_map.ndim == 2:
        fluence_map = fluence_map.unsqueeze(0).unsqueeze(0)
    elif fluence_map.ndim == 3:
        fluence_map = fluence_map.unsqueeze(0)
    
    # Convert kernels to torch tensors
    if not isinstance(kernel_x, torch.Tensor):
        kernel_x = torch.from_numpy(kernel_x).to(device=device, dtype=dtype)
    if not isinstance(kernel_y, torch.Tensor):
        kernel_y = torch.from_numpy(kernel_y).to(device=device, dtype=dtype)
    
    # Create 2D separable kernel by outer product
    # Method 1: Convolve sequentially (more memory efficient)
    
    # Reshape kernel_x for horizontal convolution [out_ch, in_ch, height, width]
    kernel_x_2d = kernel_x.view(1, 1, 1, -1)
    
    # Reshape kernel_y for vertical convolution
    kernel_y_2d = kernel_y.view(1, 1, -1, 1)
    
    # Apply convolution in X direction
    padding_x = kernel_x_2d.shape[3] // 2
    fluence_conv = torch.nn.functional.conv2d(
        fluence_map, kernel_x_2d, padding=(0, padding_x)
    )
    
    # Apply convolution in Y direction
    padding_y = kernel_y_2d.shape[2] // 2
    fluence_conv = torch.nn.functional.conv2d(
        fluence_conv, kernel_y_2d, padding=(padding_y, 0)
    )
    fluence_conv = fluence_conv / torch.clamp(fluence_conv.max(), min=1e-8)
    
    # Combine primary and scattered
    fluence_total = scatter_amplitude * fluence_conv / scatter_fraction
    
    return fluence_total