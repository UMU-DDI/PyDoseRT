import torch
import torch.nn.functional as F
from torch import nn

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


def apply_mlc_scatter(fluence, scatter_amplitude=0.02, scatter_range_mm=30.0, pixel_size_mm=1.0):
    """
    Apply MLC scatter tail that decays with distance from field edges.
    Physical basis:
    - Photons scatter in/around MLC leaves and patient
    - Creates a dose tail in blocked regions near field edges
    - Decays exponentially/Gaussian with distance from field edge
    Args:
        fluence: [B, 1, H, W] fluence map (after transmission applied)
        scatter_amplitude: Relative scatter contribution at field edge (unitless)
        scatter_range_mm: Characteristic decay distance (mm)
        pixel_size_mm: Pixel size in fluence map
    Returns:
        fluence_with_scatter: [B, 1, H, W] fluence map with scatter tail added
    """
    # Calculate Gaussian sigma from scatter range
    # Use range as ~2*sigma (covers ~95% of scatter)
    sigma_pixels = (scatter_range_mm / pixel_size_mm) / 2.0

    # Create Gaussian kernel for scatter
    kernel_size = int(6 * sigma_pixels) + 1  # 6-sigma coverage
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Limit kernel size to prevent excessive computation
    kernel_size = min(kernel_size, 201)  # Max ~100mm range at 1mm/pixel

    x = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size, device=fluence.device)
    kernel_1d = torch.exp(-x**2 / (2 * sigma_pixels**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    # Convolve the open field with scatter kernel
    # This spreads fluence from open regions into blocked regions
    pad = kernel_size // 2
    fluence_padded = F.pad(fluence, (pad, pad, pad, pad), mode='replicate')
    scatter_contribution = F.conv2d(fluence_padded, kernel_2d)

    # Only add scatter to blocked regions (not open regions)
    # Use (1 - fluence) as mask: ~0 in open (no addition), ~1 in blocked (full scatter)
    # This ensures open field stays at 100%, while blocked regions get scatter tail
    fluence_with_scatter = fluence + scatter_amplitude * scatter_contribution * (1 - fluence)


    return fluence_with_scatter




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




def apply_tongue_and_groove(fluence, leaf_boundaries_mm, field_size_mm,
                            tg_reduction=0.08, tg_width_mm=1.0, pixel_size_mm=1.0):
    """
    Apply tongue-and-groove effect at MLC leaf boundaries.

    Physical basis:
    - Leaf sides have tongue-and-groove interlocking design
    - Creates 5-10% reduction in fluence at leaf boundaries
    - Width typically 1-2mm
    - Reduces interleaf leakage but creates dead zones

    Args:
        fluence: [B, 1, H, W] fluence map
        leaf_boundaries_mm: List of leaf boundary positions in mm (H coordinates)
        field_size_mm: Total field size in H direction (mm)
        tg_reduction: Fractional reduction at leaf boundary (unitless, typical: 0.05-0.10)
        tg_width_mm: Width of tongue-and-groove region (mm, typical: 1-2)
        pixel_size_mm: Pixel size in fluence map

    Returns:
        fluence_with_tg: [B, 1, H, W] fluence map with T&G effect applied
    """
    B, _, H, W = fluence.shape
    device = fluence.device


    # Create T&G mask
    tg_mask = torch.ones((1, 1, H, 1), device=device, dtype=fluence.dtype)


    # Convert leaf boundaries to pixel coordinates
    pixel_per_mm = 1.0 / pixel_size_mm
    field_center_pixel = H / 2.0


    # For each leaf boundary, apply reduction
    for boundary_mm in leaf_boundaries_mm:
        # Convert mm to pixel index (centered)
        boundary_pixel = field_center_pixel + boundary_mm * pixel_per_mm


        # Create Gaussian reduction centered at boundary
        h_coords = torch.arange(H, device=device, dtype=fluence.dtype)
        dist_from_boundary = torch.abs(h_coords - boundary_pixel)


        # Gaussian profile with width tg_width_mm
        sigma_pixels = (tg_width_mm * pixel_per_mm) / 2.355  # FWHM to sigma
        reduction_profile = tg_reduction * torch.exp(-dist_from_boundary**2 / (2 * sigma_pixels**2))
        reduction_profile = reduction_profile.view(1, 1, H, 1)


        tg_mask = tg_mask - reduction_profile


    # Clamp mask to [0, 1]
    tg_mask = torch.clamp(tg_mask, min=0.0, max=1.0)


    # Apply T&G mask to fluence
    fluence_with_tg = fluence * tg_mask


    return fluence_with_tg