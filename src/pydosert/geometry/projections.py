import torch
import torch.nn.functional as F

def soft_max(a, b, sharpness=10.0):
    """Smooth approximation of max(a, b) using LogSumExp with broadcasting support."""
    a_scaled = a * sharpness
    b_scaled = b * sharpness
    # Manual LogSumExp: log(exp(a) + exp(b)) with numerical stability
    max_val = torch.maximum(a_scaled, b_scaled)
    return (max_val + torch.log(torch.exp(a_scaled - max_val) + torch.exp(b_scaled - max_val))) / sharpness
 
def soft_min(a, b, sharpness=10.0):
    """Smooth approximation of min(a, b) using LogSumExp with broadcasting support."""
    # min(a, b) = -max(-a, -b)
    return -soft_max(-a, -b, sharpness)

def fractional_box_overlap(d, left, right, min_value=0.0, max_value=1.0) -> torch.Tensor:
    """
    Compute fractional overlap with geometric max/min for accurate overlap computation.
    Args:
        d: Bin center positions
        left: Left edge positions
        right: Right edge positions
    """
    half_w = 0.5
    bin_start = d - half_w
    bin_end   = d + half_w
 
    overlap_start_hard = torch.maximum(left - half_w, bin_start)
    overlap_end_hard = torch.minimum(right + half_w, bin_end)
    hard = torch.clamp(overlap_end_hard - overlap_start_hard, min=min_value, max=max_value)

    return hard

def resample_fluence_map(values: torch.Tensor, leaf_widths: torch.Tensor, field_size: int, dtype: type) -> torch.Tensor:
    """
    Resamples the fluence map based on leaf geometry and output bins. calculates
    the fluence values for each output bin by considering the overlapping leaf positions.
    Now one bin equals one pixel in the output fluence map.

    Args:
        values (torch.Tensor): Input fluence values of shape [B*G, W, N, 1].

    Returns:
        torch.Tensor: Resampled fluence map of shape [B*G, W, H, 1].
    """
    B, W, N, _ = values.shape
    H = field_size
    total_length = sum(leaf_widths)

    # leaf_widths
    leaf_widths = torch.tensor(
        leaf_widths, device=values.device, dtype=dtype
    )

    # Compute start and end positions for each leaf along axis perpendicular to leaf movement
    start_positions = torch.cumsum(
        torch.cat(
            [
                torch.tensor([0.0], device=values.device, dtype=dtype),
                leaf_widths[:-1],
            ]
        ),
        dim=0,
    )
    end_positions = start_positions + leaf_widths.clone().detach().to(values.device).to(dtype)

    # divide field in bin stripes parallel to leaf movement
    output_bin_edges = torch.linspace(
        0.0, total_length, H + 1, device=values.device, dtype=dtype
    )

    # Store start and end position of each bin
    output_bin_starts = output_bin_edges[:-1]
    output_bin_ends = output_bin_edges[1:]

    # Prepare for overlap calculation (Store leaf data in column vectors and bin data in row vectors)
    start_i = start_positions.view(N, 1)
    end_i = end_positions.view(N, 1)
    start_j = output_bin_starts.view(1, H)
    end_j = output_bin_ends.view(1, H)

    # Compute overlap between leaf and bin positions (Subtract later start with earlier end)
    overlap_start = torch.max(start_i, start_j)
    overlap_end = torch.min(end_i, end_j)
    overlap = (
        (overlap_end - overlap_start).clamp(min=0.0).to(dtype=dtype)
    )

    # For each bin and depth slice sum up the open area of overlapping leaf pairs in that depth and bin
    overlap_exp = overlap.view(1, 1, N, H)  # [1, 1, N, H]
    weighted = values * overlap_exp
    total_weighted = weighted.sum(dim=2)  # [B, W, H]

    # Isn't total_overlap just the bin width?
    # total_overlap = overlap.sum(dim=0)  # [H]
    total_overlap = overlap.sum(dim=0)  # [M]
    total_overlap = total_overlap.view(1, 1, H)

    result = total_weighted / (total_overlap + 1e-8)
    result = result.unsqueeze(-1)  # [B, W, H, 1]

    return result.to(dtype)
