"""
FluenceVolumeLayer module for projecting 2D fluence maps into 3D dose volumes in radiotherapy.

This module provides the FluenceVolumeLayer class, which takes a 2D fluence map and projects it through a CT volume,
applying geometric and profile corrections to generate a 3D volume suitable for dose calculation. It precomputes sampling grids
and profile corrections for efficient forward passes and accurate modeling of the dose distribution.

Typical usage example::

    from pydose_rt.data import MachineConfig
    import torch
    machine_config = MachineConfig(...)
    layer = FluenceVolumeLayer(
        machine_config, device, dtype, sid,
        resolution, ct_array_shape, iso_center, field_size
    )

Classes:
    FluenceVolumeLayer: Torch layer for projecting 2D fluence maps into 3D dose volumes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pydose_rt.data import MachineConfig


class FluenceVolumeLayer(nn.Module):
    """
    FluenceVolumeLayer for projecting 2D fluence maps into 3D dose volumes.

    This layer takes a 2D fluence map and projects it through the CT volume, applying geometric and profile corrections
    to generate a 3D volume suitable for dose calculation. It precomputes sampling grids and profile corrections for efficient forward passes.

    Attributes:
        machine_config (MachineConfig): Configuration object containing machine parameters.
        device (torch.device): Device on which computations are performed (CPU or CUDA).
        dtype (type): Data type for tensors.
        verbose (bool): Flag to enable verbose logging.
        SID (float): Source-to-isocenter distance.
        resolution (tuple): Voxel spacing in mm.
        profile_corrections (torch.Tensor): Precomputed profile corrections for each depth.
        sampling_grids (torch.Tensor): Precomputed ray sampling grids for mapping MLC plane to CT volume.
    """

    def __init__(self, machine_config: MachineConfig, 
                 resolution: tuple[float, float, float],
                 ct_array_shape: tuple[float, float, float],
                 sid: float = 1000.0,
                 iso_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 field_size: tuple[int, int] = (400, 400),
                 device: torch.device | str | None = None,
                 dtype: torch.dtype = torch.float32,
                 verbose: bool = False) -> 'FluenceVolumeLayer':
        """
        Initializes the FluenceVolumeLayer and precomputes profile corrections and sampling grids.

        Args:
            machine_config (MachineConfig): Configuration object with machine parameters.
            resolution (tuple[float, float, float]): Voxel spacing in mm (dH, dD, dW).
            ct_array_shape (tuple[int, int, int]): Shape of the CT array as (H, D, W).
            sid (float): Source-to-isocenter distance (mm).
            iso_center (tuple[float, float, float]): Isocenter position in mm (H, D, W).
            field_size (tuple[int, int]): Field size (height, width) in pixels of the fluence map.
            device (torch.device | str | None): Device for computation (CPU or CUDA).
            dtype (torch.dtype): Data type for tensors.
            verbose (bool): If True, prints some debug info.
        """
        super().__init__()

        # Handle device default / normalization
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.dtype = dtype
        self.machine_config = machine_config
        self.verbose = verbose

        # Configuration & constants
        self.SID = sid
        self.resolution = resolution
        self.ct_array_shape = ct_array_shape
        self.iso_center = iso_center
        self.field_size = field_size

        H, D, W = self.ct_array_shape
        self.D = D

        # -----------------------------
        # Precompute depths [D]
        # -----------------------------
        # depth coordinate is along the second axis (D)
        depths = (
            self.SID - iso_center[1]
            + torch.arange(D, dtype=self.dtype, device=self.device) * self.resolution[1]
        )  # [D], in mm

        # -----------------------------
        # Spatial coordinates [W, H]
        # -----------------------------
        # H: "vertical" axis, W: "horizontal" axis in CT indexing (H, D, W)
        hs = (
            torch.arange(H, dtype=self.dtype, device=self.device) + 0.5
        ) * self.resolution[0] - iso_center[0]  # [H]
        ws = (
            torch.arange(W, dtype=self.dtype, device=self.device) + 0.5
        ) * self.resolution[2] - iso_center[2]  # [W]

        # Meshgrid in (W, H) order to match later indexing self.sampling_grids[d][w, h, :]
        WT, HT = torch.meshgrid(ws, hs, indexing="ij")  # [W, H] each

        # -----------------------------
        # Normalize to fluence map coords using field size
        # -----------------------------
        H_field, W_field = field_size
        WT_max = W_field / 2.0
        HT_max = H_field / 2.0

        WT = WT / WT_max  # [W, H]
        HT = HT / HT_max  # [W, H]

        # Base grid at SID=1 (before depth scaling)
        base_grid = torch.stack((WT, HT), dim=-1)  # [W, H, 2]

        # -----------------------------
        # Vectorized depth scaling
        # -----------------------------
        # Scale per depth slice: SID / depth
        scales = self.SID / depths  # [D]

        # Inverse square distance corrections per depth
        profile_corrections = scales**2  # [D]

        # Broadcast scales onto base_grid:
        # scales: [D, 1, 1, 1], base_grid: [1, W, H, 2] -> [D, W, H, 2]
        sampling_grids = base_grid.unsqueeze(0) * scales.view(D, 1, 1, 1)

        # -----------------------------
        # Register as buffers (not parameters)
        # -----------------------------
        self.register_buffer("profile_corrections", profile_corrections)
        self.register_buffer("sampling_grids", sampling_grids)

        if self.verbose:
            total_elems = sampling_grids.numel()
            mem_mb = total_elems * torch.finfo(self.dtype).bits / 8 / 1024**2
            print(
                f"[FluenceVolumeLayer] Precomputed sampling_grids with shape {sampling_grids.shape}, "
                f"~{mem_mb:.1f} MB on {self.device}"
            )

    def forward(
        self, fluence_map: torch.Tensor, bbox: tuple[int, int, int, int] = (None, None, None, None)
    ) -> torch.Tensor:
        """
        Projects the 2D fluence map into the 3D CT volume, applying geometric and profile corrections.

        Args:
            fluence_map (torch.Tensor): Input fluence map of shape [B*G,1,H_field,W_field].
            bbox (h_min_idx, h_max_idx, w_min_idx, w_max_idx) (int): Crop indices for output volume.

        Returns:
            torch.Tensor: 3D volume grid of shape [B*G, D, cropped_W, cropped_H, 1] representing the projected fluence.
        """
        B = fluence_map.shape[0]
        fluence_map = fluence_map.unsqueeze(1)
        H, D, W = self.ct_array_shape
        h_min_idx, h_max_idx, w_min_idx, w_max_idx = bbox
        h_min_idx = 0 if h_min_idx is None else h_min_idx
        h_max_idx = H - 1 if h_max_idx is None else h_max_idx
        w_min_idx = 0 if w_min_idx is None else w_min_idx
        w_max_idx = W - 1 if w_max_idx is None else w_max_idx
        
        vol_slices = []
        open_volumes = torch.sum(fluence_map, [1, 2, 3], keepdims=True)
        for d in range(self.D):
            # Get the precomputed sampling grid of the slice, crop to region
            grid = (
                self.sampling_grids[d][
                    w_min_idx : w_max_idx + 1, h_min_idx : h_max_idx + 1, :
                ]
                .unsqueeze(0)
                .repeat(B, 1, 1, 1)
            ).to(fluence_map.dtype)
            # Use the grid to sample the 2D fluence map into the slice
            sampled = F.grid_sample(
                fluence_map,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.permute(0, 2, 3, 1)  # [B*G,cropped_W,cropped_H,1]
            # Apply correction
            corr = self.profile_corrections[d].unsqueeze(0).unsqueeze(-1)
            # corr = (open_volumes / torch.sum(sampled, (1, 2, 3), keepdims=True)).to(self.dtype)
            vol_slices.append(sampled * corr)
        volume_grid = torch.stack(vol_slices, dim=1)  # [B*G,D,cropped_W,cropped_H,1]
        del sampled, fluence_map, grid, corr, vol_slices, open_volumes

        volume_grid = volume_grid.permute(0, 1, 3, 2, 4)
        return volume_grid
