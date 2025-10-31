import argparse
import torch


def get_args():
    # Set up argument parser for running code from terminal
    parser = argparse.ArgumentParser(description="Welcome.")
    parser.add_argument("--reg_center", default=1e-4)
    parser.add_argument("--reg_width", default=1e-4)
    parser.add_argument("--reg_leaf_rate", default=1e-4)
    parser.add_argument("--reg_mu_rate", default=1e-3)
    parser.add_argument("--lr", default=0.00005)
    parser.add_argument("--alpha", default=1.0)
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--epochs", default=500)
    parser.add_argument("--num_filters", default=8)
    parser.add_argument("--is_debug", default=1, type=int)
    parser.add_argument("--is_comet", default=1, type=int)
    parser.add_argument("--kernel_size", default=15)
    parser.add_argument("--latent_dim", default=2048)
    parser.add_argument("--number_of_cps", default=3)
    parser.add_argument("--num_leafs", default=80)
    parser.add_argument("--downsampling_factor", default=(1, 1, 1), nargs=3, type=int)
    parser.add_argument("--num_fluence_plots", default=9, type=int)
    parser.add_argument("--gpu", default=0)
    args = parser.parse_args()

    return args


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
