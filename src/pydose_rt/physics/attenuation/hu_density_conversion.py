import torch

def convert_HU_to_density(hu_tensor, lut_table):
    """
    Interpolates HU values to densities using a lookup table (LUT).

    Args:
        hu_tensor (torch.Tensor): Tensor of HU values [B, M, N] (can be any shape).
    Returns:
        torch.Tensor: Tensor of the same shape as hu_tensor.
    """
    if not torch.is_tensor(lut_table):
        lut_table = torch.tensor(
            lut_table, dtype=torch.float32, device=hu_tensor.device
        )

    x = lut_table[:, 0].contiguous()  # HU values
    y = lut_table[:, 1].contiguous()  # Densities

    # Clamp hu_tensor to bounds of LUT to avoid out-of-range interpolation
    hu_tensor_clamped = hu_tensor.clamp(min=x.min().item(), max=x.max().item())

    # Perform 1D linear interpolation
    indices = torch.searchsorted(x, hu_tensor_clamped, right=True)
    indices = indices.clamp(min=1, max=len(x) - 1)

    x0 = x[indices - 1]
    x1 = x[indices]
    y0 = y[indices - 1]
    y1 = y[indices]

    # Linear interpolation formula
    slope = (y1 - y0) / (x1 - x0)
    interpolated = y0 + slope * (hu_tensor_clamped - x0)

    return interpolated
