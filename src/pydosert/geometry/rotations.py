import torch
import math
import torch.nn.functional as F

def get_radiological_depth_indices(input_shape, angles_rad, dtype, iso_center=None, resolution=None):
    """
    Generate sampling coordinates for radiological depth calculation using ray tracing.

    For each angle, creates a ray through the isocenter (or volume center if not specified)
    with uniform voxel spacing in the rotated coordinate frame. Returns exactly D points per ray.

    Args:
        input_shape (tuple): (H, D, W) - shape of CT volume in voxels.
        angles_rad (Sequence[float] | torch.Tensor): G rotation angles in radians,
            iterated over to produce one ray per angle.
        dtype (type): torch dtype for the output coordinates.
        iso_center (Optional[tuple]): (X, Y, Z) isocenter in physical coordinates (mm),
            where X=height, Y=depth, Z=width. Defaults to the volume centre when None.
        resolution (Optional[tuple]): (rx, ry, rz) voxel spacing in mm, where rx=res_height,
            ry=res_depth, rz=res_width. Only used together with iso_center.

    Returns:
        torch.Tensor: Floating-point sampling coordinates of shape [1, G, D, 3], last
            axis ordered (x, y, z) with x in [0, W-1], y in [0, D-1], z in [0, H-1].
            Each ray has exactly D points.
    """
    H, D, W = input_shape

    # Calculate center in voxel coordinates
    if iso_center is not None and resolution is not None:
        # Convert physical isocenter to voxel coordinates
        # iso_center = (X, Y, Z) where X=height, Y=depth, Z=width (physical mm)
        # resolution = (rx, ry, rz) where rx=res_height, ry=res_depth, rz=res_width (mm/voxel)
        X, Y, Z = iso_center
        rx, ry, rz = resolution

        center_z = X / rx + 0.5  # height dimension (z in voxel coords)
        center_y = Y / ry + 0.5  # depth dimension (y in voxel coords)
        center_x = Z / rz + 0.5  # width dimension (x in voxel coords)
    else:
        # Default to volume center if isocenter not specified
        center_x = W / 2.0
        center_y = D / 2.0
        center_z = H / 2.0

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
        images (torch.Tensor): Batch of 2D images, shape [B*G, H, W].
        angles_rad (Sequence[float] | torch.Tensor): G rotation angles in radians
            (one per control point), shape [G] or [B, G]; a 2D tensor is flattened
            and the per-beam angles are repeated B times to match B*G images.
        device: torch device for the computation.
        dtype: torch dtype for the computation.

    Returns:
        torch.Tensor: Rotated images of shape [B*G, H, W].
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

    # Create affine transformation matrices for rotation.
    # Aspect-correct off-diagonal terms so non-square images get a pure
    # voxel-space rotation through affine_grid's normalized coordinates.
    aspect_HW = H / W
    mats = torch.zeros((BG, 2, 3), device=device, dtype=dtype)
    mats[:, 0, 0] = cos_a
    mats[:, 0, 1] = sin_a * aspect_HW
    mats[:, 1, 0] = -sin_a / aspect_HW
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
    Build rotation grids for rotating D×W slices by given angles around a specified point.

    Args:
        input_shape: (B, G, D, H, W) - shape of the dose volume to rotate. Only G,
            D, W are used to build the grid; B and H are broadcast by the caller.
        angles_rad (torch.Tensor): G rotation angles in radians, shape [G].
        device: torch device for the output grid.
        dtype: torch dtype for the output grid.
        iso_center: (X, Y, Z) - isocenter in physical coordinates (mm), where X=height, Y=depth, Z=width. Rotation is centred at the volume centre when None.
        resolution: (rx, ry, rz) - voxel spacing in mm, where rx=res_height, ry=res_depth, rz=res_width. Only used together with iso_center.

    Returns:
        grid2d (torch.Tensor): Sampling grid for grid_sample of shape
            [1, G, 1, D, W, 2], one rotation field per beam. The caller expands
            the leading and H dimensions and reshapes to [B*G*H, D, W, 2].
    """
    B, G, D, H, W = input_shape
    a = angles_rad.to(device=device, dtype=dtype)

    cos_a = torch.cos(a)
    sin_a = torch.sin(a)
    # affine_grid works in normalized coords where each axis spans [-1, 1], so
    # a 1-voxel step is 2/D along D and 2/W along W. For a pure rotation in
    # voxel space we have to scale the off-diagonal terms by the D/W aspect
    # ratio; otherwise non-square images get sheared instead of rotated.
    aspect_DW = D / W
    mats = torch.zeros((G, 2, 3), device=device, dtype=dtype)
    mats[:, 0, 0] = cos_a
    mats[:, 0, 1] = sin_a * aspect_DW
    mats[:, 1, 0] = -sin_a / aspect_DW
    mats[:, 1, 1] = cos_a

    # If isocenter is specified, adjust rotation to be around that point
    if iso_center is not None and resolution is not None:
        # Convert physical isocenter to voxel coordinates
        # Rotation is in the D-W plane, so we need the Y (depth) and Z (width) components
        X, Y, Z = iso_center
        rx, ry, rz = resolution

        center_y = Y / ry  # depth dimension (corresponds to D)
        center_x = Z / rz  # width dimension (corresponds to W)

        # Convert voxel coordinates to normalized coordinates [-1, 1] (for align_corners=False)
        # norm = (2 * (voxel + 0.5) / size) - 1
        norm_cy = (2.0 * (center_y + 0.5) / D) - 1.0
        norm_cx = (2.0 * (center_x + 0.5) / W) - 1.0

        # Translation that rotates around (norm_cx, norm_cy) using the
        # aspect-corrected rotation matrix above.
        mats[:, 0, 2] = norm_cx * (1.0 - cos_a) - norm_cy * sin_a * aspect_DW
        mats[:, 1, 2] = norm_cy * (1.0 - cos_a) + norm_cx * sin_a / aspect_DW

    # Generate rotation grids for each angle
    grid2d = F.affine_grid(mats, size=(G, 1, D, W), align_corners=False)  # [G, 1, D, W, 2]

    # Expand for batch and height dimensions
    grid2d = grid2d.unsqueeze(1).unsqueeze(0)              # [1, G, 1, D, W, 2]

    return grid2d