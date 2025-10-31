import torch
from typing import Tuple


def bilinear_rescaling_matrix(
    img_size: Tuple[float, float],
    center: Tuple[float, float],
    scale_factor: float,
    device: str
) -> torch.Tensor:
    """
    Generate a sparse transformation matrix for bilinear rescaling of an image.

    Parameters:
    -----------
    img_size : Tuple[float, float]
        Size of the image as (height, width).

    center : Tuple[float, float]
        Center point (y, x) around which the scaling should occur.

    scale_factor : float
        The scaling factor to apply. Values >1 zoom in, values <1 zoom out.

    Returns:
    --------
    torch.Tensor
        A sparse matrix of shape (H*W, H*W) representing the bilinear rescaling transformation.
        When applied to a flattened image, this matrix performs the scaling transformation.
    """
    H, W = img_size

    # Create target grid
    y_tgt, x_tgt = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )

    # Back project to source coordinates
    y_src = (y_tgt - center[0]) / scale_factor + center[0]
    x_src = (x_tgt - center[1]) / scale_factor + center[1]

    # Compute integer neighbors
    y0 = torch.floor(y_src).to(torch.int64)
    x0 = torch.floor(x_src).to(torch.int64)
    y1 = y0 + 1
    x1 = x0 + 1

    dy = (y_src - y0.float()).clamp(0, 1)
    dx = (x_src - x0.float()).clamp(0, 1)

    weights = [
        (1 - dy) * (1 - dx),  # top-left
        (1 - dy) * dx,        # top-right
        dy * (1 - dx),        # bottom-left
        dy * dx               # bottom-right
    ]
    neighbor_offsets = [
        (y0, x0),
        (y0, x1),
        (y1, x0),
        (y1, x1)
    ]

    # Prepare the target (flattened) indices
    target_indices = (y_tgt * W + x_tgt).to(torch.int64).flatten()

    all_source_indices = []
    all_target_indices = []
    all_values = []

    for (y_n, x_n), w in zip(neighbor_offsets, weights):
        # Mask out-of-bounds neighbors
        valid_mask = (y_n >= 0) & (y_n < H) & (x_n >= 0) & (x_n < W)
        source_indices = (y_n * W + x_n).flatten()[valid_mask.flatten()]
        weights_flat = w.flatten()[valid_mask.flatten()]
        target_valid = target_indices[valid_mask.flatten()]

        all_source_indices.append(source_indices)
        all_target_indices.append(target_valid)
        all_values.append(weights_flat)

    source_indices = torch.cat(all_source_indices)
    target_indices = torch.cat(all_target_indices)
    values = torch.cat(all_values)

    indices = torch.stack([target_indices, source_indices], dim=0)
    sparse_matrix = torch.sparse_coo_tensor(indices, values, (int(H * W), int(H * W)))

    return sparse_matrix.coalesce()


def apply_transformation_matrix(
        image: torch.Tensor,
        matrix: torch.Tensor
) -> torch.Tensor:
    """
    Apply a sparse transformation matrix to a (C, H, W) image or batch of grayscale images.

    Parameters:
    -----------
    image : torch.Tensor
        Tensor of shape (C, H, W), where C is the number of channels or batch size.

    matrix : torch.Tensor
        Sparse transformation matrix of shape (H*W, H*W) produced by bilinear_rescaling.

    Returns:
    --------
    torch.Tensor
        Transformed image tensor of shape (C, H, W).
    """
    if image.ndim != 3:
        raise ValueError("Input image must have shape (C, H, W)")

    C, H, W = image.shape

    # Apply the matrix to each channel using sparse matrix multiplication
    transformed_flat = image.reshape(C, H*W) @ matrix.T

    return transformed_flat.reshape(C, H, W)