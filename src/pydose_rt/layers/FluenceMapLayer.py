"""
FluenceMapLayer module for generating and resampling fluence maps from leaf positions in radiotherapy.

This module provides the FluenceMapLayer class, which computes the fluence map based on the positions and widths
of multi-leaf collimator (MLC) leaves and jaws. The fluence map is resampled to match the output bin configuration of the
treatment machine, enabling accurate dose modeling and further processing.

Typical usage example::

    from pydose_rt.data import MachineConfig
    import torch
    machine_config = MachineConfig(...)
    layer = FluenceMapLayer(machine_config, device, dtype, resolution, field_size)
    leaf_positions = torch.tensor(...)
    jaw_positions = torch.tensor(...)
    fluence_map = layer(leaf_positions, jaw_positions)

Classes:
    FluenceMapLayer: Torch layer for calculating and resampling fluence maps from leaf positions.
"""

import torch
import torch.nn as nn
from pydose_rt.data import MachineConfig
from pydose_rt.geometry.projections import fractional_box_overlap, resample_fluence_map
from pydose_rt.physics.fluence.fluence_modeling import (
    precompute_source_penumbra_kernel,
    precompute_mlc_scatter_kernel,
    precompute_head_scatter_kernel,
    precompute_tongue_and_groove_mask,
    apply_precomputed_kernel,
    apply_precomputed_mlc_scatter,
    apply_precomputed_head_scatter,
    apply_precomputed_tongue_and_groove
)



class FluenceMapLayer(nn.Module):
    """
    FluenceMapLayer for generating and resampling fluence maps from leaf positions.

    This layer computes the fluence map based on leaf and jaw positions, resampling the map
    according to the configuration of the treatment machine. It handles the geometric mapping
    and overlap calculations required for accurate dose modeling.

    Attributes:
        config (MachineConfig): Configuration object containing field size, leaf sizes, and number of leafs.
        verbose (bool): Flag to enable verbose logging.
        device (torch.device): Device on which computations are performed (CPU or CUDA).
    """

    def __init__(
        self,
        machine_config: MachineConfig,
        device: torch.device,
        dtype: type,
        resolution: tuple[float, float, float],
        field_size: tuple[float, float],
        verbose: bool = False,
        training_sharpness: float = 10.0,
    ):
        """
        Initializes the FluenceMapLayer.

        Args:            
            machine_config (MachineConfig): Configuration object with machine parameters.
            device (torch.device): Device on which computations are performed.
            dtype (type): Data type for tensors.
            resolution (tuple[float, float, float]): Voxel spacing in mm.
            field_size (tuple[float, float]): Field size (width, height) in pixels.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.
            training_sharpness (float, optional): Sharpness parameter for smooth gradients during training. Defaults to 10.0.
            eval_sharpness (float, optional): Sharpness parameter for sharp edges during evaluation. Defaults to 1000.0.
        """
        super().__init__()

        self.device=device
        self.dtype=dtype
        self.machine_config = machine_config
        self.resolution = resolution
        self.verbose = verbose
        self.training_sharpness = training_sharpness
        self.field_size = field_size

        if self.machine_config.leaf_widths is None:
            self.leaf_widths = torch.ones((self.machine_config.number_of_leaf_pairs, ), dtype=self.dtype) * self.field_size[1] / self.machine_config.number_of_leaf_pairs
        else:
            self.leaf_widths = self.machine_config.leaf_widths

        # Precompute depth indices
        W = self.field_size[1]
        N = machine_config.number_of_leaf_pairs
        centers = (torch.arange(W, dtype=self.dtype) + 0.5) - (W / 2)  # [H]
        depth_indices = centers.view(W, 1).repeat(1, N)  # [H, N]
        self.register_buffer("depth_indices", depth_indices.unsqueeze(0).to(self.dtype))  # [1, H, N]

        H = self.field_size[0]
        centers = (torch.arange(H, dtype=self.dtype) + 0.5) - (H / 2)  # [W]
        jaw_indices = centers.view(1, H).repeat(1, 1)
        self.register_buffer("jaw_indices", jaw_indices.unsqueeze(0).to(self.dtype))  # [1, W, N]

        # ============================================================================
        # Precompute physics augmentation kernels/masks for efficient forward pass
        # ============================================================================

        # Precompute source penumbra kernel (always applied)
        source_penumbra_kernel = precompute_source_penumbra_kernel(
            source_size_mm=self.machine_config.source_size_mm,
            pixel_size_mm=1.0,
            device=self.device,
            dtype=self.dtype
        )
        self.register_buffer("source_penumbra_kernel", source_penumbra_kernel)

        # Precompute MLC scatter kernel if amplitude > 0
        if self.machine_config.mlc_scatter_amplitude > 0:
            mlc_scatter_kernel = precompute_mlc_scatter_kernel(
                scatter_range_mm=self.machine_config.mlc_scatter_range_mm,
                pixel_size_mm=1.0,
                device=self.device,
                dtype=self.dtype
            )
            self.register_buffer("mlc_scatter_kernel", mlc_scatter_kernel)
        else:
            self.mlc_scatter_kernel = None

        # Precompute head scatter kernel if amplitude > 0
        if self.machine_config.head_scatter_amplitude > 0:
            head_scatter_kernel = precompute_head_scatter_kernel(
                scatter_range_mm=self.machine_config.head_scatter_range_mm,
                pixel_size_mm=1.0,
                device=self.device,
                dtype=self.dtype
            )
            self.register_buffer("head_scatter_kernel", head_scatter_kernel)
        else:
            self.head_scatter_kernel = None

        # Precompute tongue-and-groove mask if reduction > 0
        if self.machine_config.tongue_groove_reduction > 0:
            # Calculate leaf boundary positions
            leaf_boundaries_mm = []
            if self.machine_config.leaf_widths is not None:
                cumulative_pos = -self.field_size[0] / 2.0
                for width in self.machine_config.leaf_widths[:-1]:
                    cumulative_pos += width
                    leaf_boundaries_mm.append(cumulative_pos)
            else:
                # Uniform leaf widths
                n_leaves = self.machine_config.number_of_leaf_pairs
                leaf_width = self.field_size[0] / n_leaves
                for i in range(1, n_leaves):
                    boundary_pos = -self.field_size[0] / 2.0 + i * leaf_width
                    leaf_boundaries_mm.append(boundary_pos)

            tg_mask = precompute_tongue_and_groove_mask(
                leaf_boundaries_mm=leaf_boundaries_mm,
                field_size_mm=self.field_size[0],
                tg_reduction=self.machine_config.tongue_groove_reduction,
                tg_width_mm=self.machine_config.tongue_groove_width_mm,
                pixel_size_mm=1.0,
                H=self.field_size[0],
                device=self.device,
                dtype=self.dtype
            )
            self.register_buffer("tg_mask", tg_mask)
        else:
            self.tg_mask = None


    def forward(
        self, leaf_positions: torch.Tensor, 
        jaw_positions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Computes the fluence map from leaf and jaw positions.

        Args:
            leaf_positions (torch.Tensor): Tensor of leaf positions of shape [B, G, N, 2].
            jaw_positions (torch.Tensor): Tensor of jaw positions of shape [B, G, 2].

        Returns:
            torch.Tensor: Fluence map tensor of shape [B*G, H, W].
        """
        B, G, N, _ = leaf_positions.shape  # [B, G, N, 2]
        leaf_positions = leaf_positions.reshape(
            B * G, N, 2
        )  # [B*G, N, 2]

        left_positions = leaf_positions[..., 0]   # [B*G, N]
        right_positions = leaf_positions[..., 1]   # [B*G, N]

        W = self.field_size[1]

        left_positions = left_positions.unsqueeze(1).repeat(1, W, 1)  # [B*G, H, N]
        right_positions = right_positions.unsqueeze(1).repeat(1, W, 1)  # [B*G, H, N]

        d = self.depth_indices
        if d.device != leaf_positions.device:
            d = d.to(leaf_positions.device)  # [1, H, N]

        # Use training-dependent sharpness: smooth gradients during training, sharp during eval
        sharpness = self.training_sharpness if self.training else None

        # ---------- new box (no sigmoids) ----------
        mask = fractional_box_overlap(d, left_positions, right_positions, sharpness)
        # -------------------------------------------

        # Reshape
        mask = mask.view(B, G, W, N)
        mask = mask.view(B * G, W, N, 1)

        mask = resample_fluence_map(mask, self.leaf_widths, self.field_size[0], self.dtype)  # [B*G, H, M, 1]

        if jaw_positions is not None:
            jaw_positions = jaw_positions.reshape(B * G, 2)  # [B*G, 2]
            bottom_positions = jaw_positions[:, 0].unsqueeze(1)  # [B*G]
            top_positions = jaw_positions[:, 1].unsqueeze(1)  # [B*G]
            H = self.field_size[0]
            bottom_positions = bottom_positions.unsqueeze(2).repeat(1, 1, H)  # [B*G, H]
            top_positions = top_positions.unsqueeze(2).repeat(1, 1, H)

            j = self.jaw_indices
            if j.device != leaf_positions.device:
                j = j.to(leaf_positions.device)  # [1, H, N]
            jaw_mask = fractional_box_overlap(j, bottom_positions, top_positions, sharpness)

            jaw_mask = jaw_mask.view(B, G, H, 1)
            jaw_mask = jaw_mask.view(B * G, 1, H, 1)
            jaw_mask = jaw_mask.repeat(1, W, 1, 1)

            mask *= jaw_mask

        fluence_map = mask.permute(0, 3, 2, 1)

        # ============================================================================
        # Apply precomputed physics augmentation effects
        # ============================================================================

        # Apply tongue-and-groove effect using precomputed mask
        if self.tg_mask is not None:
            fluence_map = apply_precomputed_tongue_and_groove(
                fluence_map,
                tg_mask=self.tg_mask
            ).to(self.dtype)

        # Apply source penumbra using precomputed kernel
        fluence_map = apply_precomputed_kernel(
            fluence_map,
            kernel=self.source_penumbra_kernel,
            padding_mode='replicate'
        ).to(self.dtype)

        # Apply MLC scatter using precomputed kernel
        if self.mlc_scatter_kernel is not None:
            fluence_map = apply_precomputed_mlc_scatter(
                fluence_map,
                kernel=self.mlc_scatter_kernel,
                scatter_amplitude=self.machine_config.mlc_scatter_amplitude
            ).to(self.dtype)

        # Apply head scatter using precomputed kernel
        if self.head_scatter_kernel is not None:
            fluence_map = apply_precomputed_head_scatter(
                fluence_map,
                kernel=self.head_scatter_kernel,
                scatter_amplitude=self.machine_config.head_scatter_amplitude
            ).to(self.dtype)

        fluence_map = fluence_map[:, 0, :, :]  # [B*G, H, W]

        return fluence_map