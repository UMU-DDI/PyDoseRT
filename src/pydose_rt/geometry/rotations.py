import torch
import math
import torch.nn.functional as F

def get_radiological_depth_indices(input_shape, angles_rad, dtype, iso_center=None, resolution=None):
    """
    Generate sampling coordinates for radiological depth calculation using ray tracing.

    For each angle, creates a ray through the isocenter (or volume center if not specified)
    with uniform voxel spacing in the rotated coordinate frame. Returns exactly D points per ray.

    Args:
        input_shape: (H, D, W) - shape of CT volume in voxels
        angles_rad: list/tensor of rotation angles in radians
        dtype: torch dtype for output
        iso_center: (X, Y, Z) - isocenter in physical coordinates (mm), where X=height, Y=depth, Z=width
        resolution: (rx, ry, rz) - voxel spacing in mm, where rx=res_height, ry=res_depth, rz=res_width

    Returns:
        indices: [1, G, D, 3] - floating point coordinates (x, y, z) for sampling
                 where x∈[0,W-1], y∈[0,D-1], z∈[0,H-1]
                 Each ray has exactly D points
    """
    H, D, W = input_shape

    # Calculate center in voxel coordinates
    if iso_center is not None and resolution is not None:
        # Convert physical isocenter to voxel coordinates
        # iso_center = (X, Y, Z) where X=height, Y=depth, Z=width (physical mm)
        # resolution = (rx, ry, rz) where rx=res_height, ry=res_depth, rz=res_width (mm/voxel)
        X, Y, Z = iso_center
        rx, ry, rz = resolution

        center_z = (X - rx / 2.0) / rx  # height dimension (z in voxel coords)
        center_y = (Y - ry / 2.0) / ry  # depth dimension (y in voxel coords)
        center_x = (Z - rz / 2.0) / rz  # width dimension (x in voxel coords)
    else:
        # Default to volume center if isocenter not specified
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

def rotate_2d_images(images, angles_rad, device, dtype):
    """
    Rotate 2D images by given angles using affine transformation.
    Args:
        images: [B*G, H, W] - batch of 2D images
        angles_rad: [G] - rotation angles in radians (one per control point)
        device: torch device
        dtype: torch dtype
    Returns:
        rotated_images: [B*G, H, W] - rotated images
    """
    BG, H, W = images.shape

    # Convert angles to tensor if needed
    if not isinstance(angles_rad, torch.Tensor):
        angles_rad = torch.tensor(angles_rad, device=device, dtype=dtype)
    else:
        angles_rad = angles_rad.to(device=device, dtype=dtype)

    # Flatten angles to [1, G] if needed
    if angles_rad.dim() == 2:
        angles_rad = angles_rad.view(-1)  # [G]
    G = angles_rad.shape[0]
    B = BG // G

    # Expand angles for batch dimension: [B*G]
    angles_expanded = angles_rad.unsqueeze(0).repeat(B, 1).view(BG)  # [B*G]

    cos_a = torch.cos(angles_expanded)
    sin_a = torch.sin(angles_expanded)

    # Create affine transformation matrices for rotation
    # Note: negative angle for counter-clockwise rotation in image space
    mats = torch.zeros((BG, 2, 3), device=device, dtype=dtype)
    mats[:, 0, 0] = cos_a
    mats[:, 0, 1] = sin_a
    mats[:, 1, 0] = -sin_a
    mats[:, 1, 1] = cos_a

    # Generate rotation grids
    grid = F.affine_grid(mats, size=(BG, 1, H, W), align_corners=False)  # [BG, H, W, 2]

    # Rotate images
    images_4d = images.unsqueeze(1)  # [BG, 1, H, W]
    rotated = F.grid_sample(images_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    rotated = rotated.squeeze(1)  # [BG, H, W]

    return rotated

def build_rotation_grids(input_shape, angles_rad, device, dtype, iso_center=None, resolution=None):
    """
    Build rotation grids for rotating D×W images by given angles around a specified point.

    Args:
        input_shape: (B, G, D, H, W)
        angles_rad: Tensor of G rotation angles in radians
        device: torch device
        dtype: torch dtype
        iso_center: (X, Y, Z) - isocenter in physical coordinates (mm), where X=height, Y=depth, Z=width
        resolution: (rx, ry, rz) - voxel spacing in mm, where rx=res_height, ry=res_depth, rz=res_width

    Returns:
        grid2d: [B*G*H, D, W, 2] sampling grid for grid_sample
    """
    B, G, D, H, W = input_shape
    a = angles_rad.to(device=device, dtype=dtype)

    cos_a = torch.cos(a)
    sin_a = torch.sin(a)
    mats = torch.zeros((G, 2, 3), device=device, dtype=dtype)
    mats[:, 0, 0] = cos_a
    mats[:, 0, 1] = sin_a
    mats[:, 1, 0] = -sin_a
    mats[:, 1, 1] = cos_a

    # If isocenter is specified, adjust rotation to be around that point
    if iso_center is not None and resolution is not None:
        # Convert physical isocenter to voxel coordinates
        # Rotation is in the D-W plane, so we need the Y (depth) and Z (width) components
        X, Y, Z = iso_center
        rx, ry, rz = resolution

        center_y = (Y - ry / 2.0) / ry  # depth dimension (corresponds to D)
        center_x = (Z - rz / 2.0) / rz  # width dimension (corresponds to W)

        # Convert voxel coordinates to normalized coordinates [-1, 1] (for align_corners=False)
        # norm = (2 * (voxel + 0.5) / size) - 1
        norm_cy = (2.0 * (center_y + 0.5) / D) - 1.0
        norm_cx = (2.0 * (center_x + 0.5) / W) - 1.0

        # Adjust translation to rotate around the isocenter instead of the center
        # Translation formula: t = center * (1 - cos) - other_coord * sin (for rotation in 2D)
        # For rotation around (norm_cx, norm_cy):
        # tx (corresponding to W/x): norm_cx * (1 - cos(θ)) - norm_cy * sin(θ)
        # ty (corresponding to D/y): norm_cy * (1 - cos(θ)) + norm_cx * sin(θ)
        mats[:, 0, 2] = norm_cx * (1.0 - cos_a) - norm_cy * sin_a
        mats[:, 1, 2] = norm_cy * (1.0 - cos_a) + norm_cx * sin_a

    # Generate rotation grids for each angle
    grid2d = F.affine_grid(mats, size=(G, 1, D, W), align_corners=False)  # [G, 1, D, W, 2]

    # Expand for batch and height dimensions
    grid2d = grid2d.unsqueeze(1).unsqueeze(0)              # [1, G, 1, D, W, 2]

    return grid2d