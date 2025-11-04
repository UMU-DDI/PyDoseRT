import torch
import math
import torch.nn.functional as F

def get_radiological_depth_indices(input_shape, angles_rad, dtype):
    H, D, W = input_shape
    y = torch.linspace(0, D - 1, D)
    x = torch.linspace(0, W - 1, W)
    # Storing the two seperately enables more efficient torch operations
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")  # shape [W, D]

    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, W, D, 2]

    indices_list = []

    for angle in angles_rad:
        theta = angle
        rot_matrix = torch.tensor(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ]
        )

        # Centered grid for rotation
        center_y = (D - 1) / 2.0
        center_x = (W - 1) / 2.0
        shifted = grid[0] - torch.tensor([center_x, center_y])
        rotated = torch.matmul(shifted, rot_matrix.T) + torch.tensor(
            [center_x, center_y]
        )

        mid_x = W // 2
        line_points = rotated[mid_x, :]  # shape [D, 2]

        z_index = H // 2
        z_col = torch.full(
            (line_points.shape[0], 1), z_index, dtype=dtype
        )
        indices = torch.cat([line_points, z_col], dim=-1)  # [D, 3]

        indices_list.append(indices.int())

    stacked_indices = torch.stack(indices_list, dim=0)

    G, P, _ = stacked_indices.shape
    stacked_indices = stacked_indices.view(1, G, P, 3)
    

    return stacked_indices

def build_rotation_grids(input_shape, angles_rad, device, dtype):
    B, G, D, H, W = input_shape
    cos_a = torch.cos(angles_rad)
    sin_a = torch.sin(angles_rad)
    mats = torch.zeros((G, 2, 3), device=device, dtype=dtype)
    mats[:, 0, 0] = cos_a
    mats[:, 0, 1] = -sin_a
    mats[:, 1, 0] = sin_a
    mats[:, 1, 1] = cos_a

    # One grid per angle
    grid2d = F.affine_grid(mats, size=(G, 1, D, W), align_corners=False)  # [G, 1, D, W, 2]
    
    # Repeat each grid H times, and B times
    grid2d = grid2d.unsqueeze(1).unsqueeze(0)              # [1, G, 1, D, W, 2]
    grid2d = grid2d.repeat(B, 1, H, 1, 1, 1)               # [B, G, H, D, W, 2]
    grid2d = grid2d.reshape(B*G*H, D, W, 2)                # [B*G*H, D, W, 2]

    return grid2d