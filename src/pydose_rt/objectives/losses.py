import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydose_rt.utils.utils import get_model_input, create_bound_weight_matrix
import math

def scale_loss(loss, weight):
    return loss * weight



def constraint_loss(
    dose_pred,
    lower_bound_gy,
    higher_bound_gy,
    masks,
    region_weights=None,
    number_regions=1,
):
    """
    Computes the constraint loss for a predicted dose distribution.
    Assumes dose_pred and bounds are [B, 1, D, H, W] or [B, D, H, W] (broadcastable).
    """
    penalty_lower = F.relu(lower_bound_gy - dose_pred) ** 2
    penalty_upper = F.relu(dose_pred - higher_bound_gy) ** 2

    if region_weights is not None:
        penalty_lower = penalty_lower * region_weights
        penalty_upper = penalty_upper * region_weights

    loss_lower_bound_gy = 0.0
    loss_higher_bound_gy = 0.0
    for mask in masks.values():
        loss_lower_bound_gy += (penalty_lower * mask).mean()
        loss_higher_bound_gy += (penalty_upper * mask).mean()

    return loss_lower_bound_gy, loss_higher_bound_gy


# ======================================================================================
# l2 loss
# ======================================================================================
def compute_l2_loss(dose_pred, masks, region_weights=None, number_regions=1):
    """
    Computes the L2 loss for a set of regions, encouraging the predicted dose to be near 0.
    This is intended for regions where a low dose is desired (e.g. OARs).
    dose_pred: [B, 1, D, H, W]
    masks: dict of [B, 1, D, H, W]
    """
    loss_list = []
    for region, mask in masks.items():
        if (region.startswith("PTV") or region.startswith("CTV")):
            continue
        region_loss = torch.mean((dose_pred * mask) ** 2)
        if region_weights is not None:
            # region_weights can be a dict or tensor
            if isinstance(region_weights, dict):
                weight = (mask * region_weights[region]).max()
            else:
                weight = (mask * region_weights).max()
            region_loss *= weight
        loss_list.append(region_loss)
    if len(loss_list) == 0:
        return torch.tensor(0.0, device=dose_pred.device)
    total_loss = torch.stack(loss_list).sum()
    return total_loss * number_regions

# ======================================================================================
# mu loss
# ======================================================================================
def mus_loss(mus, config):
    def mu_rate_reg(mus, reg_mus):
        diffs = mus[:, 1:] - mus[:, :-1]
        violation = torch.clamp(diffs - reg_mus, min=0.0)
        penalty = torch.mean(violation**2)
        return penalty

    dose_rate = (
        (config.maximum_dose_rate / 60)
        * (config.gantry_diff_deg / max(config.minimum_gantry_angle_speed, 1e-3))
    )
    mu_rate_loss = mu_rate_reg(mus, dose_rate)
    
    mu_complexity_loss = torch.mean(torch.abs(mus - mus.mean()))
    return mu_rate_loss, mu_complexity_loss


# ======================================================================================
# leaf loss
# ======================================================================================
def leafs_loss(leafs, config):
    def leaf_speed_reg(leafs, leaf_rate, huge_penalty=1):
        left_positions = leafs[:, 0, :, :] - (leafs[:, 1, :, :] / 2)
        right_positions = leafs[:, 0, :, :] + (leafs[:, 1, :, :] / 2)

        left_diffs = torch.abs(left_positions[:, 1:, :] - left_positions[:, :-1, :])
        right_diffs = torch.abs(right_positions[:, 1:, :] - right_positions[:, :-1, :])

        left_violations = torch.sqrt(torch.clamp(left_diffs - leaf_rate, min=0))
        right_violations = torch.sqrt(torch.clamp(right_diffs - leaf_rate, min=0))

        left_reg = torch.mean(huge_penalty * left_violations**2)
        right_reg = torch.mean(huge_penalty * right_violations**2)

        loss = (left_reg + right_reg) / 2
        return loss

    leaf_rate_in_pixels = (
        config.maximum_leaf_speed / config.resolution[1]
    ) / config.field_size[1]
    leaf_rate = (
        leaf_rate_in_pixels
        * (config.gantry_diff_deg / max(config.minimum_gantry_angle_speed, 1e-3))
    )
    leaf_reg_loss = leaf_speed_reg(leafs, leaf_rate)
    # leaf_complexity_loss = torch.mean(torch.abs(leafs[:, 0, :, :] - leafs[:, 0, :, :].mean(1, keepdims=True))) + torch.mean(torch.abs(leafs[:, 1, :, :] - leafs[:, 1, :, :].mean(1, keepdims=True)))
    leaf_complexity_loss = torch.mean(torch.abs(leafs[:, 0, :, :] - 0.5))**2 + torch.mean(torch.abs(leafs[:, 1, :, :] - 0.0)**2)
    return leaf_reg_loss, leaf_complexity_loss

# ======================================================================================
# jaws loss
# ======================================================================================
def jaws_loss(jaws, config):
    def jaw_speed_reg(jaws, jaw_rate, huge_penalty=1):
        lower_positions = jaws[:, 0, :] - (jaws[:, 1, :] / 2)
        upper_positions = jaws[:, 0, :] + (jaws[:, 1, :] / 2)

        lower_diffs = torch.abs(lower_positions[:, 1:] - lower_positions[:, :-1])
        upper_diffs = torch.abs(upper_positions[:, 1:] - upper_positions[:, :-1])

        lower_violations = torch.sqrt(torch.clamp(lower_diffs - jaw_rate, min=0))
        upper_violations = torch.sqrt(torch.clamp(upper_diffs - jaw_rate, min=0))

        lower_reg = torch.mean(huge_penalty * lower_violations**2)
        upper_reg = torch.mean(huge_penalty * upper_violations**2)

        loss = (lower_reg + upper_reg) / 2
        return loss

    jaw_rate_in_pixels = (
        config.maximum_jaw_speed / config.resolution[1]
    ) / config.field_size[1]
    jaw_rate = (
        jaw_rate_in_pixels
        * (config.gantry_diff_deg / max(config.minimum_gantry_angle_speed, 1e-3))
    )
    jaws_reg_loss = jaw_speed_reg(jaws, jaw_rate)
    # jaws_complexity_loss = torch.mean(torch.abs(jaws[:, 0, :] - jaws[:, 0, :].mean(1, keepdims=True))) + torch.mean(torch.abs(jaws[:, 1, :] - jaws[:, 1, :].mean(1, keepdims=True)))
    jaws_complexity_loss = torch.mean(torch.abs(jaws[:, 0, :] - 0.5)) + torch.mean(torch.abs(jaws[:, 1, :] - 0.0))
    return jaws_reg_loss, jaws_complexity_loss


# ======================================================================================
# total loss
# ======================================================================================
def dose_loss(x, dose_pred, constraints, masks, region_weights=None, loss_weights=0):
    # masks: [B, 7, D, H, W]
    masks_dict = dict()
    for idx, const in enumerate(constraints.structures):
        masks_dict[const.name] = masks[:, idx : idx + 1, ...]

    loss_lower_bound_gy, loss_higher_bound_gy = constraint_loss(
        dose_pred,
        lower_bound_gy=x[:, 1:2, ...],
        higher_bound_gy=x[:, 2:3, ...],
        masks=masks_dict,
        region_weights=region_weights,
        number_regions=len(masks_dict),
    )

    # loss_lower_bound_target, loss_higher_bound_target = DVHLoss(
    #     constraints,
    #     k=50,
    #     masks=masks_dict,
    #     region_weights=region_weights,
    # ).get(None, dose_pred)

    l2_loss_oars_and_background = compute_l2_loss(
        dose_pred, masks_dict, region_weights,number_regions=len(masks_dict)
    )

    loss_lower_bound_target = torch.tensor(0.0, device=loss_lower_bound_gy.device)
    loss_higher_bound_target = torch.tensor(0.0, device=loss_lower_bound_gy.device)
    # l2_loss_oars_and_background = torch.tensor(0.0, device=loss_lower_bound_gy.device)

    return (
        loss_lower_bound_gy,
        loss_higher_bound_gy,
        loss_lower_bound_target,
        loss_higher_bound_target,
        l2_loss_oars_and_background,
    )

def create_sphere_mask(center, radius, shape=(64, 64, 64)):
    """
    Create a spherical binary mask given a center and radius.
    Returns [1, 1, D, H, W] for PyTorch convention.
    """
    grid = np.indices(shape).transpose(1, 2, 3, 0)  # shape: [H, W, D, 3]
    dist = np.sqrt(np.sum((grid - np.array(center)) ** 2, axis=-1))
    mask = (dist <= radius).astype(np.float32)
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    return mask


def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(
            min_lr / optimizer.defaults["lr"],
            0.5 * (1.0 + math.cos(math.pi * progress)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def compute_loss(patient, treatment, machine_config, dose_pred, dose_true, pred_mus, leafs, pred_jaws, weights, masks, _masks):

    region_weights = torch.from_numpy(create_bound_weight_matrix(patient.structures, treatment.weights))
    region_weights = region_weights.to(treatment.device)

    x = get_model_input(patient, treatment)
    x = torch.from_numpy(x)
    x = x.expand(1, -1, -1, -1, -1)
    (
        loss_lower_bound_gy,
        loss_higher_bound_gy,
        loss_lower_bound_target,
        loss_higher_bound_target,
        l2_loss_oars_and_background,
    ) = dose_loss(x, dose_pred, treatment, masks, region_weights, None)
    mu_rate_loss, mu_complexity_loss = mus_loss(pred_mus, machine_config)
    leaf_reg_loss, leaf_complexity_loss = leafs_loss(leafs, machine_config)
    jaw_opening_loss, jaw_complexity_loss = jaws_loss(pred_jaws, machine_config)
    all_losses = [
        scale_loss(loss_lower_bound_gy, weights["loss_lower_bound_gy"]),
        scale_loss(loss_higher_bound_gy, weights["loss_higher_bound_gy"]),
        scale_loss(loss_lower_bound_target, weights["loss_lower_bound_target"]),
        scale_loss(loss_higher_bound_target, weights["loss_higher_bound_target"]),
        scale_loss(l2_loss_oars_and_background, weights["l2_loss_oars_and_background"]),
        scale_loss(mu_rate_loss, weights["mu_rate_loss"]),
        scale_loss(mu_complexity_loss, weights["mu_complexity_loss"]),
        scale_loss(leaf_reg_loss, weights["leaf_reg_loss"]),
        scale_loss(leaf_complexity_loss, weights["leaf_complexity_loss"]),
        scale_loss(jaw_opening_loss, weights["jaw_opening_loss"]),
        scale_loss(jaw_complexity_loss, weights["jaw_complexity_loss"]),
    ]
    return all_losses

def compute_mae_loss(patient, treatment, machine_config, dose_pred, dose_true, beam_sequence, weights):
    losses = []
    for name, mask in patient.structures.items():
        losses.append(treatment.weights[name] * torch.mean(torch.abs(dose_true - dose_pred)[0, mask]))

    jaw_loss = torch.mean((torch.abs(beam_sequence.leaf_positions[1:, ...] - beam_sequence.leaf_positions[:-1, ...]))**2)
    bank_loss = leaf_range_loss(beam_sequence.leaf_positions, beam_sequence.field_size[0], machine_config.maximum_leaf_tip_overlap)
    losses.append(scale_loss(jaw_loss, weights["leaf_complexity_loss"]))
    losses.append(scale_loss(bank_loss, weights["leaf_reg_loss"]))

    return losses

def leaf_range_loss(leafs, field_size=400, threshold_mm=150.0):
    """
    Penalize leaf tip differences (max - min) that exceed threshold.

    Args:
        leafs: [B, 2, CP, num_leafs] - leaf positions (normalized 0-1)
        config: machine config with field_size
        threshold_mm: maximum allowed range in mm (default 150.0)
    """
    # Convert threshold from mm to normalized units
    threshold_normalized = threshold_mm / field_size

    # Compute range (max - min) for each leaf bank
    bank0_range = leafs[..., 0].max() - leafs[..., 0].min()
    bank1_range = leafs[..., 1].max() - leafs[..., 1].min()

    # Penalize when range exceeds threshold
    # Using ReLU so we only penalize violations, and squaring for smooth gradients
    bank0_violation = torch.nn.LeakyReLU(negative_slope=0.01)(bank0_range - threshold_normalized) ** 2
    bank1_violation = torch.nn.LeakyReLU(negative_slope=0.01)(bank1_range - threshold_normalized) ** 2

    return bank0_violation + bank1_violation


# ======================================================================================
# DVH Percentile Loss - Top-k volume targeting
# ======================================================================================

def dvh_percentile_loss(
    dose_pred: torch.Tensor,
    structure_mask: torch.Tensor,
    target_dose: float,
    volume_percent: float,
    constraint_type: str = "at_least",
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Loss that targets a specific percentile of the DVH curve (Dx%).

    Selects the top-k% of voxels by dose and pushes them toward a target.
    This is more precise than whole-structure losses for DVH optimization.

    Args:
        dose_pred: Predicted dose [B, 1, D, H, W] or [B, D, H, W]
        structure_mask: Binary mask [B, 1, D, H, W] or [B, D, H, W]
        target_dose: Target dose in Gy
        volume_percent: Volume percentage (0-100).
                       95.0 means "dose at 95% of volume" (D95%)
                       0.0 means "maximum dose" (Dmax)
        constraint_type: 'at_least' (for targets) or 'at_most' (for OARs)
        temperature: Softness of top-k selection (lower = harder selection)

    Returns:
        Scalar loss tensor

    Example:
        # Push D95% of PTV to be >= prescription dose
        loss = dvh_percentile_loss(dose, ptv_mask, 42.7, 95.0, "at_least")

        # Push Dmax of bladder to be <= tolerance dose
        loss = dvh_percentile_loss(dose, bladder_mask, 45.0, 0.0, "at_most")

        # Push D2% of PTV to be <= hot spot limit
        loss = dvh_percentile_loss(dose, ptv_mask, 45.7, 2.0, "at_most")
    """
    # Ensure inputs have channel dimension
    if dose_pred.ndim == 4:
        dose_pred = dose_pred.unsqueeze(1)
    if structure_mask.ndim == 4:
        structure_mask = structure_mask.unsqueeze(1)

    # Extract doses within structure
    structure_doses = dose_pred * structure_mask

    # Flatten spatial dimensions
    B, C = structure_doses.shape[:2]
    structure_doses_flat = structure_doses.view(B, C, -1)  # [B, C, N]
    structure_mask_flat = structure_mask.view(B, C, -1)  # [B, C, N]

    # Count voxels in structure
    n_voxels = structure_mask_flat.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, C, 1]

    # Calculate how many voxels to select
    # For D95%, select top 5% (100-95)
    # For Dmax, select top 0.1% (to be differentiable)
    percentile_to_select = max(100.0 - volume_percent, 0.1)
    k_voxels = torch.ceil(n_voxels * percentile_to_select / 100.0).long()  # [B, C, 1]

    # Soft top-k selection using temperature-scaled softmax
    # Higher doses get higher weights
    weights = torch.softmax(structure_doses_flat / temperature, dim=-1)  # [B, C, N]
    weights = weights * structure_mask_flat  # Only in structure
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # Renormalize

    # Weighted dose at this percentile
    dose_at_percentile = (structure_doses_flat * weights).sum(dim=-1)  # [B, C]

    # Target dose as tensor
    if not torch.is_tensor(target_dose):
        target_dose = torch.tensor(target_dose, dtype=dose_pred.dtype, device=dose_pred.device)

    # Compute loss based on constraint type
    if constraint_type == "at_least":
        # For targets: penalize if dose is below target
        # D95% >= target means we want dose_at_95% to be high
        loss = F.relu(target_dose - dose_at_percentile) ** 2
    else:  # at_most
        # For OARs: penalize if dose exceeds target
        # Dmax <= target means we want maximum dose to be low
        loss = F.relu(dose_at_percentile - target_dose) ** 2

    return loss.mean()


def dvh_volume_at_dose_loss(
    dose_pred: torch.Tensor,
    structure_mask: torch.Tensor,
    dose_threshold: float,
    target_volume_percent: float,
    constraint_type: str = "at_most",
    temperature: float = 10.0
) -> torch.Tensor:
    """
    Loss for volume-at-dose constraints (Vx Gy).

    Computes the volume receiving at least a certain dose and compares to target.

    Args:
        dose_pred: Predicted dose [B, 1, D, H, W] or [B, D, H, W]
        structure_mask: Binary mask [B, 1, D, H, W] or [B, D, H, W]
        dose_threshold: Dose threshold in Gy
        target_volume_percent: Target volume percentage (0-100)
        constraint_type: 'at_most' (typical for OARs) or 'at_least' (rare)
        temperature: Softness of threshold (higher = softer sigmoid)

    Returns:
        Scalar loss tensor

    Example:
        # V40Gy <= 15% for bladder
        loss = dvh_volume_at_dose_loss(dose, bladder_mask, 40.0, 15.0, "at_most")

        # V95% of prescription >= 99% for PTV
        loss = dvh_volume_at_dose_loss(dose, ptv_mask, 40.6, 99.0, "at_least")
    """
    # Ensure inputs have channel dimension
    if dose_pred.ndim == 4:
        dose_pred = dose_pred.unsqueeze(1)
    if structure_mask.ndim == 4:
        structure_mask = structure_mask.unsqueeze(1)

    # Convert dose threshold to tensor
    if not torch.is_tensor(dose_threshold):
        dose_threshold = torch.tensor(dose_threshold, dtype=dose_pred.dtype, device=dose_pred.device)

    # Soft indicator: voxels above threshold
    soft_indicator = torch.sigmoid(temperature * (dose_pred - dose_threshold))

    # Volume fraction above threshold
    numerator = (soft_indicator * structure_mask).sum(dim=(2, 3, 4))  # [B, C]
    denominator = structure_mask.sum(dim=(2, 3, 4)).clamp(min=1)  # [B, C]
    volume_fraction = (numerator / denominator) * 100.0  # Convert to percentage

    # Target volume as tensor
    if not torch.is_tensor(target_volume_percent):
        target_volume_percent = torch.tensor(
            target_volume_percent, dtype=dose_pred.dtype, device=dose_pred.device
        )

    # Compute loss based on constraint type
    if constraint_type == "at_most":
        # V40Gy <= 15% means we want volume_fraction to be low
        loss = F.relu(volume_fraction - target_volume_percent) ** 2
    else:  # at_least
        # V42.7Gy >= 99% means we want volume_fraction to be high
        loss = F.relu(target_volume_percent - volume_fraction) ** 2

    return loss.mean()
