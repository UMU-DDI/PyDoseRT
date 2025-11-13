import torch
import math
import torch.nn.functional as F

def get_radiological_depth_indices(input_shape, angles_rad, dtype):
    """
    Generate sampling coordinates for radiological depth calculation using ray tracing.

    For each angle, creates a ray through the volume center with uniform voxel spacing
    in the rotated coordinate frame. Returns exactly D points per ray.

    Args:
        input_shape: (H, D, W) - shape of CT volume in voxels
        angles_rad: list/tensor of rotation angles in radians
        dtype: torch dtype for output

    Returns:
        indices: [1, G, D, 3] - floating point coordinates (x, y, z) for sampling
                 where x∈[0,W-1], y∈[0,D-1], z∈[0,H-1]
                 Each ray has exactly D points
    """
    H, D, W = input_shape

    # Center of the volume in voxel coordinates
    center_x = (W - 1) / 2.0
    center_y = (D - 1) / 2.0
    center_z = (H - 1) / 2.0

    # Create a line of D points along the Y axis (depth direction)
    # This is the reference line at angle=0
    y_line = torch.linspace(0, D - 1, D, dtype=dtype)
    x_line = torch.full_like(y_line, center_x)  # X stays at center

    indices_list = []

    for angle in angles_rad:
        theta = float(angle)

        # Rotation matrix
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # Shift to origin for rotation
        x_shifted = x_line - center_x
        y_shifted = y_line - center_y

        # Apply rotation
        x_rotated = x_shifted * cos_theta - y_shifted * sin_theta
        y_rotated = x_shifted * sin_theta + y_shifted * cos_theta

        # Shift back to center
        x_coords = x_rotated + center_x
        y_coords = y_rotated + center_y
        z_coords = torch.full_like(x_coords, center_z)

        # Stack into [D, 3] with order [x, y, z] = [W, D, H]
        ray_coords = torch.stack([x_coords, y_coords, z_coords], dim=-1)

        indices_list.append(ray_coords)

    stacked_indices = torch.stack(indices_list, dim=0)  # [G, D, 3]

    return stacked_indices.unsqueeze(0)  # [1, G, D, 3]

def build_rotation_grids(input_shape, angles_rad, device, dtype):
    """
    Build rotation grids for rotating D×W images by given angles.
    
    Args:
        input_shape: (B, G, D, H, W) 
        angles_rad: Tensor of G rotation angles in radians
        device: torch device
        dtype: torch dtype
    
    Returns:
        grid2d: [B*G*H, D, W, 2] sampling grid for grid_sample
    """
    B, G, D, H, W = input_shape
    a = angles_rad.to(device=device, dtype=dtype)
    # a -= math.pi

    cos_a = torch.cos(a)
    sin_a = torch.sin(a)
    mats = torch.zeros((G, 2, 3), device=device, dtype=dtype)
    mats[:, 0, 0] = cos_a
    mats[:, 0, 1] = sin_a
    mats[:, 1, 0] = -sin_a
    mats[:, 1, 1] = cos_a

    # Generate rotation grids for each angle
    grid2d = F.affine_grid(mats, size=(G, 1, D, W), align_corners=False)  # [G, 1, W, D, 2]
    # grid2d = grid2d[..., [1, 0]]
    
    # Expand for batch and height dimensions
    grid2d = grid2d.unsqueeze(1).unsqueeze(0)              # [1, G, 1, D, W, 2]
    grid2d = grid2d.repeat(B, 1, H, 1, 1, 1)               # [B, G, H, D, W, 2]
    grid2d = grid2d.reshape(B*G*H, D, W, 2)                # [B*G*H, D, W, 2]

    return grid2d