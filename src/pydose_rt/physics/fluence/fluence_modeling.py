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
    
def apply_source_penumbra(fluence, source_size_mm=3.0, pixel_size_mm=1.0):
    """
    Apply geometric penumbra from finite source size.
    
    Physical basis: 
    - Linac bremsstrahlung target is ~2-5mm diameter
    - Creates penumbra at MLC plane (typically ~100cm from source)
    - Penumbra width ≈ source_size * (SDD/SAD) where SDD is distance to MLC
    
    Args:
        fluence: [B, 1, H, W] fluence map
        source_size_mm: Effective source diameter (typical: 2-5mm)
        pixel_size_mm: Pixel size in fluence map
    """
    # Calculate Gaussian sigma from source FWHM
    # FWHM = 2.355 * sigma
    sigma_pixels = (source_size_mm / pixel_size_mm) / 2.355
    
    # Create Gaussian kernel
    kernel_size = int(6 * sigma_pixels) + 1  # 6-sigma coverage
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    x = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size, device=fluence.device)
    kernel_1d = torch.exp(-x**2 / (2 * sigma_pixels**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    
    # Convolve with replicate padding (better for edge behavior)
    pad = kernel_size // 2
    fluence_padded = F.pad(fluence, (pad, pad, pad, pad), mode='replicate')
    fluence_with_penumbra = F.conv2d(fluence_padded, kernel_2d)
    
    return fluence_with_penumbra




def apply_head_scatter(fluence, scatter_amplitude=0.035, scatter_range_mm=150.0, pixel_size_mm=1.0):
    """
    Apply head scatter (phantom scatter) - long-range scatter from linac head.

    Physical basis:
    - Photons scatter in linac head (flattening filter, collimators, air)
    - Creates long-range dose contribution outside primary field
    - Depends on field size (larger fields → more scatter)
    - Decays slowly with distance (exponential)
    - Typically 2-4% of primary dose at field edge, decaying to ~1% at 10cm out

    Args:
        fluence: [B, 1, H, W] fluence map
        scatter_amplitude: Relative scatter contribution (unitless, typical: 0.03-0.05)
        scatter_range_mm: Characteristic decay distance (mm, typical: 100-200)
        pixel_size_mm: Pixel size in fluence map

    Returns:
        fluence_with_head_scatter: [B, 1, H, W] fluence map with head scatter added
    """
    # Calculate Gaussian sigma from scatter range
    # Head scatter has longer range than MLC scatter
    sigma_pixels = (scatter_range_mm / pixel_size_mm) / 2.0


    # Create large Gaussian kernel for head scatter
    kernel_size = int(6 * sigma_pixels) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1


    # Limit kernel size but allow larger than MLC scatter
    kernel_size = min(kernel_size, 401)  # Max ~200mm range at 1mm/pixel


    x = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size, device=fluence.device)
    kernel_1d = torch.exp(-x**2 / (2 * sigma_pixels**2))
    kernel_1d = kernel_1d / kernel_1d.sum()


    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)


    # Convolve entire fluence map
    pad = kernel_size // 2
    fluence_padded = F.pad(fluence, (pad, pad, pad, pad), mode='constant', value=0)
    scatter_contribution = F.conv2d(fluence_padded, kernel_2d)


    # Calculate field size scaling factor
    # Larger fields produce more scatter (proportional to open area)
    open_area = torch.sum(fluence, dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
    total_area = fluence.shape[2] * fluence.shape[3]
    field_size_factor = open_area / total_area


    # Add head scatter everywhere (not just blocked regions)
    # Scale by field size
    fluence_with_head_scatter = fluence + scatter_amplitude * scatter_contribution * field_size_factor


    return fluence_with_head_scatter




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


def precompute_head_scatter_kernel(scatter_range_mm: float, pixel_size_mm: float,
                                   device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Precompute the head scatter convolution kernel as 1D (for separable convolution).

    Args:
        scatter_range_mm: Characteristic decay distance (mm, typical: 100-200)
        pixel_size_mm: Pixel size in fluence map
        device: Device to create kernel on
        dtype: Data type for kernel

    Returns:
        kernel: [1, 1, 1, K] 1D convolution kernel (for separable convolution)
    """
    sigma_pixels = (scatter_range_mm / pixel_size_mm) / 2.0

    kernel_size = int(6 * sigma_pixels) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel_size = min(kernel_size, 401)

    x = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size, device=device, dtype=dtype)
    kernel_1d = torch.exp(-x**2 / (2 * sigma_pixels**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Return 1D kernel for separable convolution
    kernel_1d = kernel_1d.view(1, 1, 1, kernel_size)

    return kernel_1d


def precompute_directional_head_scatter_kernels(
    scatter_sigma_mlc_mm: float,
    scatter_sigma_jaw_mm: float,
    scatter_amplitude_mlc: float,
    scatter_amplitude_jaw: float,
    pixel_size_mm: float,
    device: torch.device,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Precompute two separate 1D Gaussian kernels for head scatter in MLC and JAW directions
    with different sigma values and amplitudes.

    Returns normalized kernels along with their amplitudes for independent application.
    Each direction contributes independently to the total scatter.

    Physical basis: Head scatter comes from photons scattering in the linac head
    (flattening filter, collimators, air). The scatter profile can differ between
    MLC and JAW directions due to different head geometries.

    Args:
        scatter_sigma_mlc_mm: Gaussian sigma for MLC direction scatter in mm
        scatter_sigma_jaw_mm: Gaussian sigma for JAW direction scatter in mm
        scatter_amplitude_mlc: Scatter amplitude as fraction of dose (e.g., 0.04 = 4%)
        scatter_amplitude_jaw: Scatter amplitude as fraction of dose (e.g., 0.06 = 6%)
        pixel_size_mm: Pixel size in fluence map
        device: Device to create kernels on
        dtype: Data type for kernel

    Returns:
        kernel_mlc: [1, 1, 1, K_mlc] Normalized 1D convolution kernel for MLC direction
        kernel_jaw: [1, 1, K_jaw, 1] Normalized 1D convolution kernel for JAW direction
        amplitude_mlc: MLC scatter amplitude (returned for application phase)
        amplitude_jaw: JAW scatter amplitude (returned for application phase)
    """
    # MLC direction kernel (horizontal)
    sigma_mlc_pixels = scatter_sigma_mlc_mm / pixel_size_mm
    kernel_size_mlc = int(6 * sigma_mlc_pixels) + 1
    if kernel_size_mlc % 2 == 0:
        kernel_size_mlc += 1
    kernel_size_mlc = min(kernel_size_mlc, 601)  # Allow large kernels for head scatter

    x_mlc = torch.linspace(-(kernel_size_mlc//2), kernel_size_mlc//2,
                           kernel_size_mlc, device=device, dtype=dtype)
    kernel_mlc_1d = torch.exp(-(x_mlc**2) / (2 * sigma_mlc_pixels**2))
    kernel_mlc_1d = kernel_mlc_1d / kernel_mlc_1d.sum()
    kernel_mlc = kernel_mlc_1d.view(1, 1, 1, kernel_size_mlc)

    # JAW direction kernel (vertical)
    sigma_jaw_pixels = scatter_sigma_jaw_mm / pixel_size_mm
    kernel_size_jaw = int(6 * sigma_jaw_pixels) + 1
    if kernel_size_jaw % 2 == 0:
        kernel_size_jaw += 1
    kernel_size_jaw = min(kernel_size_jaw, 601)  # Allow large kernels for head scatter

    x_jaw = torch.linspace(-(kernel_size_jaw//2), kernel_size_jaw//2,
                           kernel_size_jaw, device=device, dtype=dtype)
    kernel_jaw_1d = torch.exp(-(x_jaw**2) / (2 * sigma_jaw_pixels**2))
    kernel_jaw_1d = kernel_jaw_1d / kernel_jaw_1d.sum()
    kernel_jaw = kernel_jaw_1d.view(1, 1, kernel_size_jaw, 1)

    return kernel_mlc, kernel_jaw, scatter_amplitude_mlc, scatter_amplitude_jaw


# ============================================================================
# Fast application functions using precomputed kernels/masks
# ============================================================================

def apply_precomputed_kernel(fluence: torch.Tensor, kernel: torch.Tensor,
                            padding_mode: str = 'replicate') -> torch.Tensor:
    """
    Apply a precomputed 1D convolution kernel to fluence map using separable convolution.

    This is much faster than 2D convolution for Gaussian kernels:
    - 2D convolution: O(K^2) operations per pixel
    - Separable (2x1D): O(2K) operations per pixel
    - For K=401, that's ~200x fewer operations!

    Args:
        fluence: [B, 1, H, W] fluence map
        kernel: [1, 1, 1, K] 1D convolution kernel
        padding_mode: Padding mode for convolution

    Returns:
        fluence_convolved: [B, 1, H, W] convolved fluence map
    """
    # Kernel is [1, 1, 1, K] for horizontal conv
    kernel_size = kernel.shape[-1]
    pad = kernel_size // 2

    # Apply horizontal convolution (along width dimension)
    fluence_padded_h = F.pad(fluence, (pad, pad, 0, 0), mode=padding_mode)
    fluence_h = F.conv2d(fluence_padded_h, kernel)

    # Apply vertical convolution (along height dimension)
    # Transpose kernel from [1, 1, 1, K] to [1, 1, K, 1]
    kernel_v = kernel.transpose(-2, -1)
    fluence_padded_v = F.pad(fluence_h, (0, 0, pad, pad), mode=padding_mode)
    fluence_convolved = F.conv2d(fluence_padded_v, kernel_v)

    return fluence_convolved


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

def apply_precomputed_head_scatter(fluence: torch.Tensor, kernel: torch.Tensor,
                                   scatter_amplitude: float) -> torch.Tensor:
    """
    Apply head scatter using precomputed kernel.

    Args:
        fluence: [B, 1, H, W] fluence map
        kernel: [1, 1, K, K] precomputed scatter kernel
        scatter_amplitude: Relative scatter contribution

    Returns:
        fluence_with_head_scatter: [B, 1, H, W] fluence map with head scatter
    """
    scatter_contribution = apply_precomputed_kernel(fluence, kernel, padding_mode='constant')

    # Calculate field size scaling factor
    open_area = torch.sum(fluence, dim=(2, 3), keepdim=True)
    total_area = fluence.shape[2] * fluence.shape[3]
    field_size_factor = open_area / total_area

    fluence_with_head_scatter = fluence + scatter_amplitude * scatter_contribution * field_size_factor
    return fluence_with_head_scatter


def apply_directional_head_scatter(
    fluence: torch.Tensor,
    kernel_mlc: torch.Tensor,
    kernel_jaw: torch.Tensor,
    amplitude_mlc: float,
    amplitude_jaw: float,
    padding_mode: str = 'constant'
) -> torch.Tensor:
    """
    Apply directional head scatter using normalized kernels with independent amplitude scaling.

    Each direction (MLC and JAW) contributes independently to the total scatter. This avoids
    the amplitude multiplication issue that occurs with pre-scaled separable convolution.

    Args:
        fluence: [B, 1, H, W] fluence map
        kernel_mlc: [1, 1, 1, K_mlc] Normalized 1D convolution kernel for MLC direction
        kernel_jaw: [1, 1, K_jaw, 1] Normalized 1D convolution kernel for JAW direction
        amplitude_mlc: Scatter amplitude for MLC direction (e.g., 0.04 = 4%)
        amplitude_jaw: Scatter amplitude for JAW direction (e.g., 0.06 = 6%)
        padding_mode: Padding mode for convolution (default: 'constant')

    Returns:
        fluence_with_head_scatter: [B, 1, H, W] fluence map with head scatter added
    """
    # Apply MLC direction scatter (1D convolution along width dimension only)
    kernel_mlc_size = kernel_mlc.shape[-1]
    pad_mlc = kernel_mlc_size // 2
    fluence_padded_mlc = F.pad(fluence, (pad_mlc, pad_mlc, 0, 0), mode=padding_mode)
    scatter_mlc = F.conv2d(fluence_padded_mlc, kernel_mlc)

    # Apply JAW direction scatter (1D convolution along height dimension only)
    kernel_jaw_size = kernel_jaw.shape[-2]
    pad_jaw = kernel_jaw_size // 2
    fluence_padded_jaw = F.pad(fluence, (0, 0, pad_jaw, pad_jaw), mode=padding_mode)
    scatter_jaw = F.conv2d(fluence_padded_jaw, kernel_jaw)

    # Add both scatter contributions independently with their respective amplitudes
    fluence_with_head_scatter = fluence# + amplitude_mlc * scatter_mlc + amplitude_jaw * scatter_jaw

    return fluence_with_head_scatter


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