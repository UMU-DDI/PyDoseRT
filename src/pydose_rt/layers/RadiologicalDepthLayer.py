"""
RadiologicalDepthLayer module for computing radiological depth profiles through CT volumes for radiotherapy.

This module provides the RadiologicalDepthLayer class, which calculates the cumulative radiological depth
along lines through a CT volume at specified gantry angles. It rotates and samples the CT volume,
converts Hounsfield Units (HU) to density, and integrates the density along the beam path for each angle.

Typical usage example::

    from ..MachineConfig import MachineConfig
    import torch
    config = MachineConfig(...)
    layer = RadiologicalDepthLayer(config)
    ct_stack = torch.tensor(...)
    depth_profiles = layer(ct_stack)

Classes:
    RadiologicalDepthLayer: Torch layer for computing radiological depth profiles through CT volumes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pydose_rt.data import MachineConfig, TreatmentConfig
from pydose_rt.physics.attenuation.hu_density_conversion import convert_HU_to_density
from pydose_rt.geometry.rotations import get_radiological_depth_indices


class RadiologicalDepthLayer(nn.Module):
    """
    Torch layer for computing radiological depth profiles through CT volumes at specified gantry angles.

    This layer rotates and samples the CT volume along lines corresponding to different gantry angles,
    converts HU to density, and integrates the density along the beam path to produce radiological
    depth profiles for dose calculation.

    Attributes:
        config (MachineConfig): Configuration object containing CT array shape, gantry angles, resolution, and lookup table for HU-to-density conversion.
        verbose (bool): Flag to enable verbose logging.
        device (torch.device): Device on which computations are performed (CPU or CUDA).
        stacked_indices (torch.Tensor): Precomputed indices for sampling CT volume along rotated lines for each gantry angle.

    """

    def __init__(self, machine_config: MachineConfig, treatment_config: TreatmentConfig, 
        dtype, 
        device, verbose: bool = False):
        """
        Initializes the RadiologicalDepthLayer and precomputes sampling indices for each gantry angle.

        Args:
            config (MachineConfig): Configuration object with CT array shape, gantry angles, resolution, and lookup table.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.
        """
        super(RadiologicalDepthLayer, self).__init__()

        self.device=device
        self.dtype=dtype
        self.machine_config = machine_config
        self.treatment_config = treatment_config
        self.verbose = verbose

        # Determine if we should use full-sized CT for depth extraction
        self.downsample_depths = self.treatment_config.downsampling_factor != (1, 1, 1)

        # Store the target (downsampled) CT shape
        self.ct_array_shape = tuple([int(x / y) for x, y in zip(machine_config.ct_array_shape,  treatment_config.downsampling_factor)])
        self.target_ct_shape = self.ct_array_shape
        self.resolution = tuple([x * y for x, y in zip(machine_config.resolution,  treatment_config.downsampling_factor)])

        # If using full CT, compute indices for full-sized CT
        self.full_ct_shape = self.machine_config.ct_array_shape
        stacked_indices = get_radiological_depth_indices(
            self.full_ct_shape, self.treatment_config.gantry_angles, self.dtype
        ).to(self.device)

        # Final shape: [1, G, P, 3]
        self.register_buffer("stacked_indices", stacked_indices)


    def forward(self, ct_stack: torch.Tensor) -> torch.Tensor:
        """
        Computes radiological depth profiles through the CT volume for each gantry angle.

        Args:
            ct_stack (torch.Tensor): CT volume tensor of shape [B, H, D, W].
                                    Can be full-sized or downsampled based on initialization.

        Returns:
            torch.Tensor: Radiological depth profiles of shape [B*G, P_target, 1], where
            P_target is the number of points in the downsampled depth profile.
        """
        with torch.no_grad():
            B, H, D, W = ct_stack.shape
            _, G, P, _ = self.stacked_indices.shape

            # Sample CT volume using trilinear interpolation at floating-point coordinates
            # stacked_indices: [1, G, P, 3] with order [x, y, z] = [W, D, H]

            # Expand for batch dimension: [B, G, P, 3]
            coords = self.stacked_indices.expand(B, G, P, 3)

            # Extract coordinates
            x_coords = coords[..., 0]  # W dimension [B, G, P]
            y_coords = coords[..., 1]  # D dimension [B, G, P]
            z_coords = coords[..., 2]  # H dimension [B, G, P]

            # Perform trilinear interpolation manually
            # Clamp coordinates to valid range
            x_coords = torch.clamp(x_coords, 0, W - 1)
            y_coords = torch.clamp(y_coords, 0, D - 1)
            z_coords = torch.clamp(z_coords, 0, H - 1)

            # Get integer parts (floor) and fractional parts
            x0 = torch.floor(x_coords).long()
            y0 = torch.floor(y_coords).long()
            z0 = torch.floor(z_coords).long()

            x1 = torch.clamp(x0 + 1, 0, W - 1)
            y1 = torch.clamp(y0 + 1, 0, D - 1)
            z1 = torch.clamp(z0 + 1, 0, H - 1)

            xd = x_coords - x0.float()
            yd = y_coords - y0.float()
            zd = z_coords - z0.float()

            # Sample at 8 corners for trilinear interpolation
            # ct_stack shape: [B, H, D, W]
            # Need to expand batch indices
            b_idx = torch.arange(B, device=self.device).view(B, 1, 1).expand(B, G, P)

            c000 = ct_stack[b_idx, z0, y0, x0]
            c001 = ct_stack[b_idx, z0, y0, x1]
            c010 = ct_stack[b_idx, z0, y1, x0]
            c011 = ct_stack[b_idx, z0, y1, x1]
            c100 = ct_stack[b_idx, z1, y0, x0]
            c101 = ct_stack[b_idx, z1, y0, x1]
            c110 = ct_stack[b_idx, z1, y1, x0]
            c111 = ct_stack[b_idx, z1, y1, x1]

            # Trilinear interpolation
            c00 = c000 * (1 - xd) + c001 * xd
            c01 = c010 * (1 - xd) + c011 * xd
            c10 = c100 * (1 - xd) + c101 * xd
            c11 = c110 * (1 - xd) + c111 * xd

            c0 = c00 * (1 - yd) + c01 * yd
            c1 = c10 * (1 - yd) + c11 * yd

            gathered = c0 * (1 - zd) + c1 * zd  # [B, G, P]

            # Convert HU to density using lookup table
            density = convert_HU_to_density(
                gathered, self.treatment_config.lookup_table
            )  # shape: [B, G, P]

            # Calculate physical step size per angle (accounts for anisotropic voxels)
            if P > 1:
                # Compute for all rays: [G, P, 3]
                all_rays = self.stacked_indices[0, :, :, :]  # [G, P, 3]
                diff = all_rays[:, 1:, :] - all_rays[:, :-1, :]  # [G, P-1, 3]

                # Convert voxel differences to physical distances
                res_tensor = torch.tensor(
                    [self.resolution[0], self.resolution[1], self.resolution[2]],
                    device=self.device, dtype=self.dtype
                )
                physical_diff = diff * res_tensor

                # Calculate euclidean distance for each step: [G, P-1]
                step_distances = torch.sqrt((physical_diff ** 2).sum(dim=-1))
                # Average step size per angle: [G]
                step_sizes = step_distances.mean(dim=-1).view(1, G, 1)  # [1, G, 1]
            else:
                step_sizes = torch.tensor(
                    self.resolution[1],
                    device=self.device,
                    dtype=self.dtype
                ).view(1, 1, 1)

            # Integrate density along each line (cumulative sum) and scale by step size
            # Each angle gets its own physically correct step size
            # This accumulates radiological depth from source (entrance) toward patient interior
            cumsum = torch.cumsum(density, dim=-1) * step_sizes  # shape: [B, G, P]

            # If we extracted from full-sized CT, downsample the radiological depths
            if self.downsample_depths:
                # Reshape for interpolation: [B, G, P] -> [B*G, 1, P]
                cumsum = cumsum.view(B * G, 1, P)

                # Calculate target size based on downsampling factor
                downsample_factor = max(self.treatment_config.downsampling_factor)
                P_target = P // downsample_factor

                # Downsample using linear interpolation
                cumsum = F.interpolate(
                    cumsum, size=P_target, mode='linear', align_corners=False
                )

                # Reshape to [B*G, P_target, 1]
                cumsum = cumsum.view(B * G, P_target, 1)
            else:
                # No downsampling needed, just reshape to [B*G, P, 1]
                cumsum = cumsum.view(B * G, P, 1)

            return cumsum