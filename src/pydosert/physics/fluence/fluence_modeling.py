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
        measured_profile (np.ndarray): Measured intensity values, shape [P].
        modelled_profile (np.ndarray): Modelled intensity values, shape [P].
        x_scale_mm (np.ndarray): X-positions in mm for each profile point, shape [P].
        sample_points_mm (List[float]): Radial distances (mm) where ratios are sampled, length S.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (sample points [S], sampled ratio values [S]).
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
    Create a normalized 1D Gaussian kernel for convolution.

    Args:
        sigma_cm (float): Standard deviation in cm.
        resolution_cm (float): Pixel resolution in cm.
        kernel_half_width (float): Number of sigmas to extend the kernel on each side. Default 5.

    Returns:
        torch.Tensor: Normalized 1D Gaussian kernel, shape [K] where
            K = 2 * ceil(kernel_half_width * sigma_cm / resolution_cm) + 1.
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
    """
    Compute the output factor by averaging MLC- and JAW-defined field-size lookups.

    Linearly interpolates a tabulated output-factor curve at the MLC and JAW
    field sizes and returns their mean.

    Args:
        field_size_mlc_mm (torch.Tensor): MLC-defined field size in mm, arbitrary shape [...].
        field_size_jaw_mm (torch.Tensor): JAW-defined field size in mm, same shape as field_size_mlc_mm [...].
        output_factors (Sequence): Length-2 sequence where output_factors[0] are the
            tabulated field sizes (mm, ascending, length L) and output_factors[1] the
            corresponding output factors (length L).

    Returns:
        torch.Tensor: Averaged output factors, same shape as field_size_mlc_mm [...].
    """
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
        sample_distances_mm (np.ndarray): Radial distances in mm, shape [S].
        sample_ratios (np.ndarray): Ratio values at corresponding distances, shape [S].
        image_shape (Tuple[int, int]): Output image size (height, width) = (H, W).
        pixel_size_mm (float): Physical size of each pixel in mm.
        center (Optional[Tuple[float, float]]): (y_center, x_center) in pixels.
            If None, uses image center.

    Returns:
        torch.Tensor: Radially interpolated correction map, shape [H, W].
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
        x (torch.Tensor): X-coordinates, must be sorted ascending, shape [L].
        y (torch.Tensor): Y-coordinates, shape [L].
        x_new (torch.Tensor): X-coordinates to interpolate at, arbitrary shape [...].

    Returns:
        torch.Tensor: Interpolated y values at x_new, same shape as x_new [...].
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
        """
        Initialize the learnable fluence kernel.

        Args:
            kernel_size (int): Side length K of the square convolution kernel. Default 15.
                The kernel is initialized as a delta (identity) of shape [1, 1, K, K].
        """
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
        Apply the learned (normalized) kernel and scale to a fluence map.

        Args:
            fluence_map (torch.Tensor): Fluence map, shape [B*G, H, W].

        Returns:
            torch.Tensor: Corrected fluence map, shape [B*G, H, W].
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
        desired_penumbra_fwhm_mm (float): Penumbra width (FWHM) in mm, typically 2-4 mm.
        device (torch.device): Device to create kernel on.
        dtype (torch.dtype): Torch dtype.

    Returns:
        torch.Tensor: Separable 1D convolution kernel, shape [1, 1, 1, K] where
            K = 6 * round(fwhm / 2.355) + 1 (forced odd).
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
        penumbra_fwhm_mlc_mm (float): Penumbra width (FWHM) in MLC direction (horizontal/width) in mm.
        penumbra_fwhm_jaw_mm (float): Penumbra width (FWHM) in JAW direction (vertical/height) in mm.
        device (torch.device): Device to create kernels on.
        dtype (torch.dtype): Torch dtype.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            kernel_mlc, shape [1, 1, 1, K_mlc], 1D kernel for the MLC direction (horizontal);
            kernel_jaw, shape [1, 1, K_jaw, 1], 1D kernel for the JAW direction (vertical).
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
        fluence (torch.Tensor): Fluence map, shape [B, 1, H, W].
        kernel_mlc (torch.Tensor): 1D convolution kernel for MLC direction (horizontal/width), shape [1, 1, 1, K_mlc].
        kernel_jaw (torch.Tensor): 1D convolution kernel for JAW direction (vertical/height), shape [1, 1, K_jaw, 1].
        padding_mode (str): Padding mode for convolution. Default 'replicate'.

    Returns:
        torch.Tensor: Convolved fluence map, shape [B, 1, H, W].
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
        fluence_1d (torch.Tensor): 1D fluence profile, shape [B, W].
        pixel_size_mm (float): Pixel size in mm. Default 1.0.
        threshold (float): Threshold for field edge detection (fraction of max). Default 0.5.

    Returns:
        torch.Tensor: Effective field size in mm (pixel_size_mm units), shape [B].
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
    """
    Build a 1D linear interpolation function from tabulated (x, y) pairs.

    Args:
        point_dict (Sequence): Iterable of length-L of (x, y) pairs; x values
            must be ascending. Pair element [0] is x, element [1] is y.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: A function mapping an input
            tensor of arbitrary shape to interpolated values of the same shape.
    """
    # Sort keys and values into tensors
    xs = torch.tensor([vals[0] for vals in point_dict], dtype=torch.float32)
    ys = torch.tensor([vals[1] for vals in point_dict], dtype=torch.float32)

    def interpolate(x):
        """
        Linearly interpolate the tabulated curve at the given points.

        Args:
            x (torch.Tensor): Query points, arbitrary shape [...].

        Returns:
            torch.Tensor: Interpolated values, same shape as x [...].
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


def apply_head_scatter_kernels(fluence_map, kernel_x, kernel_y):
    """
    Apply 2D separable head-scatter convolution to the fluence map.

    Convolves the aperture with a wide Gaussian kernel (separable, X then Y)
    to produce the scatter fluence component. The caller is responsible for
    the weighted combination ``total = (1 - w) * primary + w * scatter``.

    Args:
        fluence_map (torch.Tensor): Aperture / primary fluence map, shape
            [N, C, H, W], [N, H, W], or [H, W]; lower-rank inputs are promoted
            to 4D internally.
        kernel_x (torch.Tensor or np.ndarray): 1D Gaussian kernel for the
            horizontal (MLC leaf-motion / width) direction, shape [Kx].
        kernel_y (torch.Tensor or np.ndarray): 1D Gaussian kernel for the
            vertical (jaw / inline / height) direction, shape [Ky].

    Returns:
        torch.Tensor: Blurred scatter map, shape [N, C, H, W] (always 4D
            regardless of input rank).
    """
    device = fluence_map.device
    dtype = fluence_map.dtype

    # Ensure input is 4D [N, C, H, W]
    original_ndim = fluence_map.ndim
    if fluence_map.ndim == 2:
        fluence_map = fluence_map.unsqueeze(0).unsqueeze(0)
    elif fluence_map.ndim == 3:
        fluence_map = fluence_map.unsqueeze(0)

    # Convert kernels to torch tensors on the correct device/dtype
    if not isinstance(kernel_x, torch.Tensor):
        kernel_x = torch.from_numpy(kernel_x)
    kernel_x = kernel_x.to(device=device, dtype=dtype)

    if not isinstance(kernel_y, torch.Tensor):
        kernel_y = torch.from_numpy(kernel_y)
    kernel_y = kernel_y.to(device=device, dtype=dtype)

    # Separable convolution: horizontal then vertical
    kernel_x_2d = kernel_x.view(1, 1, 1, -1)
    padding_x = kernel_x_2d.shape[3] // 2
    fluence_conv = F.conv2d(fluence_map, kernel_x_2d, padding=(0, padding_x))

    kernel_y_2d = kernel_y.view(1, 1, -1, 1)
    padding_y = kernel_y_2d.shape[2] // 2
    fluence_conv = F.conv2d(fluence_conv, kernel_y_2d, padding=(padding_y, 0))

    return fluence_conv


def compute_sc_output_factor(
    jaw_w_mm: torch.Tensor,
    jaw_h_mm: torch.Tensor,
    sc_amplitude: float,
    sigma_x_mm: float,
    sigma_y_mm: float,
    ref_field_mm: float = 100.0,
) -> torch.Tensor:
    """
    Compute the collimator scatter factor Sc analytically.

    Sc is modelled as the integral of a 2-D Gaussian extended source over the
    jaw opening, normalised to a 10 x 10 cm reference field:

    .. math::

        S_c(W, H) = \\frac{1 + A \\cdot \\operatorname{erf}\\!
            \\left(\\frac{W}{2\\sqrt{2}\\sigma_x}\\right)
            \\operatorname{erf}\\!\\left(\\frac{H}{2\\sqrt{2}\\sigma_y}\\right)}
            {S_{c,\\mathrm{ref}}}

    Args:
        jaw_w_mm (torch.Tensor): Jaw opening in the crossline (X) direction in mm, shape [B].
        jaw_h_mm (torch.Tensor): Jaw opening in the inline (Y) direction in mm, shape [B].
        sc_amplitude (float): Head-scatter amplitude A (typically 0.03 - 0.15).
        sigma_x_mm (float): Effective source sigma at isocentre in the X direction, in mm.
        sigma_y_mm (float): Effective source sigma at isocentre in the Y direction, in mm.
        ref_field_mm (float): Side length of the reference square field in mm (default 100 mm = 10 cm).

    Returns:
        torch.Tensor: Sc values, shape [B], normalised so Sc = 1 for the reference field.
    """
    sqrt2 = 2.0 ** 0.5

    # Reference field normalisation (scalar, computed once)
    t_ref_x = float(torch.erf(torch.tensor(ref_field_mm / (2.0 * sqrt2 * sigma_x_mm))))
    t_ref_y = float(torch.erf(torch.tensor(ref_field_mm / (2.0 * sqrt2 * sigma_y_mm))))
    sc_ref = 1.0 + sc_amplitude * t_ref_x * t_ref_y

    # Actual field
    t_x = torch.erf(jaw_w_mm / (2.0 * sqrt2 * sigma_x_mm))
    t_y = torch.erf(jaw_h_mm / (2.0 * sqrt2 * sigma_y_mm))
    sc = (1.0 + sc_amplitude * t_x * t_y) / sc_ref

    return sc