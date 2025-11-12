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