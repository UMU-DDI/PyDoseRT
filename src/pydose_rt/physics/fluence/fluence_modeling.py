import torch
import torch.nn.functional as F

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
    blocked_region_mask = 1.0 - fluence
    fluence_with_scatter = fluence + scatter_amplitude * scatter_contribution * blocked_region_mask

    return fluence_with_scatter