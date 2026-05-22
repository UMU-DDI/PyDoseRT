"""
FluenceMapLayer module for generating and resampling fluence maps from leaf positions in radiotherapy.

This module provides the FluenceMapLayer class, which computes the fluence map based on the positions and widths
of multi-leaf collimator (MLC) leaves and jaws. The fluence map is resampled to match the output bin configuration of the
treatment machine, enabling accurate dose modeling and further processing.

Typical usage example::

    from pydosert.data import MachineConfig
    import torch
    machine_config = MachineConfig(...)
    layer = FluenceMapLayer(machine_config, device, dtype, field_size)
    leaf_positions = torch.tensor(...)
    jaw_positions = torch.tensor(...)
    fluence_map = layer(leaf_positions, jaw_positions)

Classes:
    FluenceMapLayer: Torch layer for calculating and resampling fluence maps from leaf positions.
"""

import torch
import torch.nn as nn
from pydosert.data import MachineConfig
from pydosert.geometry.projections import fractional_box_overlap, resample_fluence_map
from pydosert.physics.fluence.fluence_modeling import (
    create_radial_correction_map,
    precompute_head_scatter_kernel,
    apply_head_scatter_kernels,
    get_output_factor,
    compute_sc_output_factor,
    apply_directional_precomputed_kernel,
    estimate_field_size_1d,
    precompute_directional_source_penumbra_kernels,
)



class FluenceMapLayer(nn.Module):
    """
    FluenceMapLayer for generating and resampling fluence maps from leaf positions.

    This layer computes the fluence map based on leaf and jaw positions, resampling the map
    according to the configuration of the treatment machine. It handles the geometric mapping
    and overlap calculations required for accurate dose modeling.

    Attributes:
        machine_config (MachineConfig): Configuration object containing field size, leaf sizes, and number of leafs.
        verbose (bool): Flag to enable verbose logging.
        device (torch.device): Device on which computations are performed (CPU or CUDA).
        dtype (torch.dtype): Data type for tensors.
        field_size (tuple[int, int]): Field size (H, W) in pixels.
        pixel_size_mm (float): Physical size of one fluence-map pixel in mm.
        training_sharpness (float): Sharpness parameter for smooth gradients during training.
        leaf_widths (torch.Tensor): Per-leaf-pair widths, shape [N].
        depth_indices (torch.Tensor): Pixel-center positions in mm along W per leaf, shape [1, W, N].
        jaw_indices (torch.Tensor): Pixel-center positions in mm along H, shape [1, 1, H].
    """

    def __init__(
        self,
        machine_config: MachineConfig,
        field_size: tuple[int, int] = (400, 400),
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        verbose: bool = False,
        training_sharpness: float = 10.0,
        pixel_size_mm: float = 1.0,
    ) -> 'FluenceMapLayer':
        """
        Initializes the FluenceMapLayer.

        Args:
            machine_config (MachineConfig): Configuration object with machine parameters.
            field_size (tuple[int, int]): Field size (H, W) in pixels.
            device (torch.device | str | None, optional): Device on which computations are performed. Defaults to CUDA if available, else CPU.
            dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.
            training_sharpness (float, optional): Sharpness parameter for smooth gradients during training. Defaults to 10.0.
            pixel_size_mm (float, optional): Physical size of one fluence-map pixel in mm; scales the precomputed depth/jaw indices so leaf/jaw positions (mm) are directly comparable. Defaults to 1.0.
        """
        super().__init__()

        # Handle device default
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device=device
        self.dtype=dtype
        self.machine_config = machine_config
        self.verbose = verbose
        self.training_sharpness = training_sharpness
        self.field_size = field_size
        self.pixel_size_mm = pixel_size_mm
        self.training = False

        if self.machine_config.leaf_widths is None:
            self.leaf_widths = torch.ones((self.machine_config.number_of_leaf_pairs, ), dtype=self.dtype) * self.field_size[1] / self.machine_config.number_of_leaf_pairs
        else:
            self.leaf_widths = self.machine_config.leaf_widths

        # Precompute depth/jaw indices in mm (not pixel units).
        # Scaling by pixel_size_mm ensures that leaf/jaw positions (in mm) are
        # directly comparable to these indices in fractional_box_overlap.
        W = self.field_size[1]
        N = machine_config.number_of_leaf_pairs
        centers_mm = ((torch.arange(W, dtype=self.dtype) + 0.5) - W / 2) * pixel_size_mm
        depth_indices = centers_mm.view(W, 1).repeat(1, N)
        self.register_buffer("depth_indices", depth_indices.unsqueeze(0).to(self.dtype))  # [1, W, N]

        H = self.field_size[0]
        centers_mm = ((torch.arange(H, dtype=self.dtype) + 0.5) - H / 2) * pixel_size_mm
        jaw_indices = centers_mm.view(1, H)
        self.register_buffer("jaw_indices", jaw_indices.unsqueeze(0).to(self.dtype))  # [1, 1, H]

        # ============================================================================
        # Precompute physics augmentation kernels/masks for efficient forward pass
        # ============================================================================

        # Precompute source penumbra kernels
        # Use directional kernels if both MLC and JAW FWHM are specified
        self.use_penumbra = False
        if hasattr(self.machine_config, 'penumbra_fwhm'):
            if self.machine_config.penumbra_fwhm is not None:
                if len(self.machine_config.penumbra_fwhm) == 1:
                    penumbra_mlc = self.machine_config.penumbra_fwhm[0]
                    penumbra_jaw = self.machine_config.penumbra_fwhm[0]
                elif len(self.machine_config.penumbra_fwhm) == 2:
                    penumbra_mlc = self.machine_config.penumbra_fwhm[0]
                    penumbra_jaw = self.machine_config.penumbra_fwhm[1]
                else:
                    raise Exception("Penumbra parameters must not contain more than two elements.")
                
                kernel_mlc, kernel_jaw = precompute_directional_source_penumbra_kernels(
                    penumbra_fwhm_mlc_mm=penumbra_mlc,
                    penumbra_fwhm_jaw_mm=penumbra_jaw,
                    device=self.device,
                    dtype=self.dtype
                )
                self.register_buffer("source_penumbra_kernel_mlc", kernel_mlc)
                self.register_buffer("source_penumbra_kernel_jaw", kernel_jaw)
                self.use_penumbra = True

        # Check if head scatter parameters are configured
        # Head scatter is now applied using physics-based Sc(field_size) model
        self.use_head_scatter = False
        if (hasattr(self.machine_config, 'head_scatter_amplitude') and 
            hasattr(self.machine_config, 'head_scatter_sigma')):
            if self.machine_config.head_scatter_amplitude is not None:
                if len(self.machine_config.head_scatter_amplitude) == 1:
                    head_scatter_amplitude_mlc = self.machine_config.head_scatter_amplitude[0]
                    head_scatter_amplitude_jaw = self.machine_config.head_scatter_amplitude[0]
                    head_scatter_sigma_mlc = self.machine_config.head_scatter_sigma[0]
                    head_scatter_sigma_jaw = self.machine_config.head_scatter_sigma[0]
                elif len(self.machine_config.head_scatter_amplitude) == 2:
                    head_scatter_amplitude_mlc = self.machine_config.head_scatter_amplitude[0]
                    head_scatter_amplitude_jaw = self.machine_config.head_scatter_amplitude[1]
                    head_scatter_sigma_mlc = self.machine_config.head_scatter_sigma[0]
                    head_scatter_sigma_jaw = self.machine_config.head_scatter_sigma[1]
                kernel_mlc = precompute_head_scatter_kernel(head_scatter_sigma_mlc, resolution_cm=0.1)
                kernel_jaw = precompute_head_scatter_kernel(head_scatter_sigma_jaw, resolution_cm=0.1)
                self.register_buffer("head_scatter_kernel_mlc", kernel_mlc.to(self.dtype).to(self.device))
                self.register_buffer("head_scatter_kernel_jaw", kernel_jaw.to(self.dtype).to(self.device))
                self.head_scatter_amplitude_mlc = head_scatter_amplitude_mlc
                self.head_scatter_amplitude_jaw = head_scatter_amplitude_jaw
                self.use_head_scatter = True

        self.use_output_factor = False
        if hasattr(self.machine_config, 'output_factors') and (self.machine_config.output_factors is not None):
            self.output_factors = self.machine_config.output_factors
            self.use_output_factor = True

        # Precompute off-axis profile correction
        if hasattr(self.machine_config, 'profile_corrections') and (self.machine_config.profile_corrections is not None):
            profile_correction_map = create_radial_correction_map(
                self.machine_config.profile_corrections[0],
                self.machine_config.profile_corrections[1],
                self.field_size,
                self.pixel_size_mm
            ).unsqueeze(0).unsqueeze(0).to(self.device).to(self.dtype).detach()
            self.register_buffer("profile_correction_map", profile_correction_map)
            self.use_profile_correction = True
        else:
            self.use_profile_correction = False

        self.mlc_transmission = self.machine_config.mlc_transmission

        # DLG: half of the gap is added to each bank edge
        self.dlg_half_mm = (self.machine_config.dlg_mm or 0.0) / 2.0

        # Physics-based Sc model (erf integral of Gaussian source over jaw)
        self.use_sc_model = False
        if (hasattr(self.machine_config, 'sc_source_sigma_mm') and
                self.machine_config.sc_source_sigma_mm is not None):
            sigmas = self.machine_config.sc_source_sigma_mm
            self.sc_sigma_x_mm = float(sigmas[0])
            self.sc_sigma_y_mm = float(sigmas[1]) if len(sigmas) > 1 else float(sigmas[0])
            # sc_amplitude reuses the head-scatter amplitude (same physics parameter)
            self.sc_amplitude = (
                float(self.machine_config.head_scatter_amplitude[0])
                if (self.machine_config.head_scatter_amplitude is not None)
                else 0.0
            )
            self.use_sc_model = True


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
        leaf_positions = leaf_positions.reshape(B * G, N, 2)  # [B*G, N, 2]

        left_positions = leaf_positions[..., 0]   # [B*G, N]
        right_positions = leaf_positions[..., 1]   # [B*G, N]

        # ── DLG: widen each MLC bank by half the dosimetric leaf gap ────────────
        if self.dlg_half_mm > 0.0:
            left_positions = left_positions - self.dlg_half_mm
            right_positions = right_positions + self.dlg_half_mm

        W = self.field_size[1]

        left_positions = left_positions.unsqueeze(1).repeat(1, W, 1)   # [B*G, W, N]
        right_positions = right_positions.unsqueeze(1).repeat(1, W, 1)  # [B*G, W, N]

        d = self.depth_indices
        if d.device != leaf_positions.device:
            d = d.to(leaf_positions.device)  # [1, W, N]

        # ── MLC aperture mask ────────────────────────────────────────────────────
        mask = fractional_box_overlap(d, left_positions, right_positions,
                                      min_value=self.mlc_transmission,
                                      pixel_size=self.pixel_size_mm)

        mask = mask.view(B, G, W, N)
        mask = mask.view(B * G, W, N, 1)
        mask = resample_fluence_map(mask, self.leaf_widths, self.field_size[0], self.dtype)  # [B*G, W, H, 1]

        # ── Jaw aperture mask ────────────────────────────────────────────────────
        if jaw_positions is not None:
            jaw_positions = jaw_positions.reshape(B * G, 2)  # [B*G, 2]
            bottom_positions = jaw_positions[:, 0].unsqueeze(1)  # [B*G, 1]
            top_positions    = jaw_positions[:, 1].unsqueeze(1)  # [B*G, 1]
            H = self.field_size[0]
            bottom_positions = bottom_positions.unsqueeze(2).repeat(1, 1, H)
            top_positions    = top_positions.unsqueeze(2).repeat(1, 1, H)

            j = self.jaw_indices
            if j.device != leaf_positions.device:
                j = j.to(leaf_positions.device)
            jaw_mask = fractional_box_overlap(j, bottom_positions, top_positions,
                                              pixel_size=self.pixel_size_mm)
            jaw_mask = jaw_mask.view(B, G, H, 1)
            jaw_mask = jaw_mask.view(B * G, 1, H, 1)
            jaw_mask = jaw_mask.repeat(1, W, 1, 1)

            mask *= jaw_mask

        # aperture: [B*G, W, H, 1] → [B*G, 1, H, W]
        aperture = mask.permute(0, 3, 2, 1)

        # ── Estimate jaw field sizes for OF / Sc (from the jaw mask alone) ──────
        need_field_size = self.use_output_factor or self.use_sc_model
        if need_field_size:
            if jaw_positions is not None:
                # jaw_mask is [B*G, W, H, 1]; average over the orthogonal axis
                field_size_x_mm = estimate_field_size_1d(
                    jaw_mask.permute(0, 2, 1, 3).mean(dim=1).squeeze(2), self.pixel_size_mm
                )
                field_size_y_mm = estimate_field_size_1d(
                    jaw_mask.mean(dim=2).squeeze(2), self.pixel_size_mm
                )
            else:
                # No jaws: use the MLC aperture projected profiles
                field_size_x_mm = estimate_field_size_1d(
                    aperture.mean(dim=2).squeeze(1), self.pixel_size_mm
                )
                field_size_y_mm = estimate_field_size_1d(
                    aperture.mean(dim=3).squeeze(1), self.pixel_size_mm
                )

        # ════════════════════════════════════════════════════════════════════════
        # Physics pipeline
        #
        #  1. penumbra  conv (source blur, directional 1-D Gaussians)
        #  2. profile correction  (radial off-axis factor, applied to primary)
        #  3. head-scatter blend  (1 - w)*primary + w*scatter
        #     scatter is convolved from the *original aperture* with a wide kernel
        #  4. output factor  (Sc erf model and/or empirical LUT)
        # ════════════════════════════════════════════════════════════════════════

        # Step 1 – source penumbra (Gaussian blur in MLC and jaw directions)
        if self.use_penumbra:
            primary = apply_directional_precomputed_kernel(
                aperture,
                kernel_mlc=self.source_penumbra_kernel_mlc,
                kernel_jaw=self.source_penumbra_kernel_jaw,
                padding_mode='replicate'
            ).to(self.dtype)
        else:
            primary = aperture

        # Step 2 – radial off-axis profile correction (applied to primary only)
        if self.use_profile_correction:
            primary = primary * self.profile_correction_map

        # Step 3 – head-scatter blend
        #   scatter is computed from the *raw aperture* with a wide Gaussian
        if self.use_head_scatter:
            scatter = apply_head_scatter_kernels(
                aperture,
                self.head_scatter_kernel_mlc,
                self.head_scatter_kernel_jaw,
            ).to(self.dtype)
            w = self.head_scatter_amplitude_mlc
            fluence_map = (1.0 - w) * primary + w * scatter
        else:
            fluence_map = primary

        # Step 4 – output factor
        #   (a) Physics-based Sc erf model
        if self.use_sc_model:
            sc = compute_sc_output_factor(
                jaw_w_mm=field_size_x_mm,
                jaw_h_mm=field_size_y_mm,
                sc_amplitude=self.sc_amplitude,
                sigma_x_mm=self.sc_sigma_x_mm,
                sigma_y_mm=self.sc_sigma_y_mm,
            )
            fluence_map = sc[:, None, None, None] * fluence_map

        #   (b) Empirical OF LUT (can be used as residual correction on top of Sc,
        #       or as the sole OF model when sc_source_sigma_mm is not configured)
        if self.use_output_factor:
            OF = get_output_factor(field_size_x_mm, field_size_y_mm, self.output_factors)
            fluence_map = OF[:, None, None, None] * fluence_map

        fluence_map = fluence_map[:, 0, :, :]  # [B*G, H, W]

        return fluence_map