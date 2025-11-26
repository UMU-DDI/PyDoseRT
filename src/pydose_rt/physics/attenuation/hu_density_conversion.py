import torch

def convert_HU_to_density(
        hu_tensor: torch.Tensor, 
        lut_table: torch.Tensor = torch.Tensor(
            [
                [-1000, 0.0],  # TODO: Added for safety
                [-992, 0.00109],
                [-960, 0.00109],
                [-500, 0.5],
                [-75, 0.95],
                [42, 1.04],
                [85, 1.08],
                [490, 1.29],
                [890, 1.52],
                [1240, 1.72],
                [1670, 1.95],
                [2155, 2.15],
                [2640, 2.34],
                [2832, 2.46],
                [2840, 6.6],
            ]
        )):
    """
    Interpolates HU values to densities using a lookup table (LUT).

    Args:
        hu_tensor (torch.Tensor): Tensor of HU values [B, M, N] (can be any shape).
    Returns:
        torch.Tensor: Tensor of the same shape as hu_tensor.
    """
    lut_table = lut_table.to(hu_tensor.dtype).to(hu_tensor.device)

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
