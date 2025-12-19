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
        (config.maximum_dose_rate)
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

def compute_dvh_loss(patient, optimization, machine_config, dose_pred, dose_true, beam_sequence, weights):
    dose_pred = dose_pred * 7
    raw_losses = []
    # PTV_Prostata_gol_4270

    raw_losses.append(scale_loss(torch.mean(torch.abs(dose_pred[patient.structures["PTVT_42.7"]] - 42.7)), optimization.structures["PTVT_42.7"]["weight"]))
    raw_losses.append(scale_loss(torch.mean(torch.abs(dose_pred[patient.structures["CTVT"]] - 42.7)), optimization.structures["CTVT"]["weight"]))

    for struct_name in ['PenileBulb', 'Prostate', 'FemoralHead_L', 'FemoralHead_R', 'Bladder', 'Rectum', 'SeminalVesicles', 'External']:
        raw_losses.append(scale_loss(torch.mean(torch.abs(dose_pred[patient.structures[struct_name]])), optimization.structures[struct_name]["weight"]))

    # raw_losses.append(scale_loss(dvh_percentile_objective(dose_pred, patient.structures["FemoralHead_L"], 20), optimization.structures["FemoralHead_L"]["weight"])) # 
    # raw_losses.append(scale_loss(dvh_percentile_objective(dose_pred, patient.structures["FemoralHead_R"], 20), optimization.structures["FemoralHead_R"]["weight"])) # D_
    # raw_losses.append(scale_loss(dvh_percentile_objective(dose_pred, patient.structures["Bladder"], 40), optimization.structures["Bladder"]["weight"]))
    # raw_losses.append(scale_loss(dvh_volume_objective(dose_pred, patient.structures["Bladder"], 21.0), optimization.structures["Bladder"]["weight"]))
    # raw_losses.append(scale_loss(dvh_percentile_objective(dose_pred, patient.structures["Rectum"], 40), optimization.structures["Rectum"]["weight"]))
    # raw_losses.append(scale_loss(dvh_volume_objective(dose_pred, patient.structures["Rectum"], 21.0), optimization.structures["Rectum"]["weight"]))
    

    # raw_losses.append(scale_loss(torch.mean((torch.abs(beam_sequence.leaf_positions[1:, ...] - beam_sequence.leaf_positions[:-1, ...]))), weights["leaf_complexity_loss"]))
    # raw_losses.append(scale_loss(leaf_range_loss(beam_sequence.leaf_positions, beam_sequence.field_size[0], machine_config.maximum_leaf_tip_overlap), weights["leaf_reg_loss"]))
    # raw_losses.append(scale_loss(torch.mean((torch.abs(beam_sequence.mus[1:, ...] - beam_sequence.mus[:-1, ...]))), weights["mu_complexity_loss"]))
    # raw_losses.append(scale_loss(torch.mean((torch.abs(beam_sequence.jaw_positions[1:, ...] - beam_sequence.jaw_positions[:-1, ...]))), weights["jaw_complexity_loss"]))

    return raw_losses

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
import math
import torch
import torch.nn.functional as F


def dvh_percentile_objective(
    dose_pred: torch.Tensor,
    structure_mask: torch.Tensor,
    volume_percent: float,
    p_norm: float = 1.0,
) -> torch.Tensor:
    """
    Pure objective for D_p% (dose at p% volume).

    Returns:
        D_p^p_norm  (scalar tensor)

    You control direction with a sign outside:
        - to push D_p UP:   use   loss += -w * dvh_percentile_objective(...)
        - to push D_p DOWN: use   loss += +w * dvh_percentile_objective(...)

    Args:
        dose_pred:       [B, 1, D, H, W] or [B, D, H, W] or [D, H, W]
        structure_mask:  same shape (0/1 or bool)
        volume_percent:  p in [0,100], e.g. 99 -> D99%, 0 -> D0% (≈ Dmax)
        p_norm:          exponent on D_p (1.0 = linear, 2.0 = quadratic, etc.)
    """
    # Ensure channel dimension
    if dose_pred.ndim == 3:
        dose_pred = dose_pred.unsqueeze(0)       # [1, D, H, W]
    if dose_pred.ndim == 4:
        dose_pred = dose_pred.unsqueeze(1)       # [B, 1, D, H, W]
    if structure_mask.ndim == 3:
        structure_mask = structure_mask.unsqueeze(0)
    if structure_mask.ndim == 4:
        structure_mask = structure_mask.unsqueeze(1)

    # Extract structure voxels
    structure_doses = dose_pred[structure_mask > 0]  # [N_voxels]
    if structure_doses.numel() == 0:
        return torch.tensor(0.0, device=dose_pred.device, dtype=dose_pred.dtype)

    # Sort descending
    sorted_doses, _ = torch.sort(structure_doses, descending=True)
    N = sorted_doses.numel()

    # Compute index for D_p%
    p = float(volume_percent) / 100.0
    p = max(0.0, min(1.0, p))
    idx = int(math.ceil(p * N)) - 1
    if idx < 0:
        idx = 0
    if idx >= N:
        idx = N - 1

    D_p = sorted_doses[idx]  # scalar
    return D_p ** p_norm
def dvh_volume_objective(
    dose_pred: torch.Tensor,
    structure_mask: torch.Tensor,
    dose_threshold: float,
    temperature: float = 10.0,
    p_norm: float = 1.0,
) -> torch.Tensor:
    """
    Pure objective for V_x (volume receiving at least x Gy).

    Returns:
        V_x^p_norm  where V_x is in %, averaged over batch.

    Again, direction is controlled outside:
        - to push V_x DOWN: loss += +w * dvh_volume_objective(...)
        - to push V_x UP:   loss += -w * dvh_volume_objective(...)

    Args:
        dose_pred:       [B, 1, D, H, W] or [B, D, H, W] or [D, H, W]
        structure_mask:  same shape
        dose_threshold:  x in V_x Gy (float, Gy)
        temperature:     sigmoid slope (softness around dose_threshold)
        p_norm:          exponent on V_x (%)
    """
    # Ensure channel dimension
    if dose_pred.ndim == 3:
        dose_pred = dose_pred.unsqueeze(0)
    if dose_pred.ndim == 4:
        dose_pred = dose_pred.unsqueeze(1)
    if structure_mask.ndim == 3:
        structure_mask = structure_mask.unsqueeze(0)
    if structure_mask.ndim == 4:
        structure_mask = structure_mask.unsqueeze(1)

    # Threshold as tensor
    if not torch.is_tensor(dose_threshold):
        dose_threshold = torch.tensor(
            dose_threshold, dtype=dose_pred.dtype, device=dose_pred.device
        )

    # Soft indicator for dose >= threshold
    soft_indicator = torch.sigmoid(temperature * (dose_pred - dose_threshold))

    # Volume fraction in %
    numerator = (soft_indicator * structure_mask).sum(dim=(2, 3, 4))  # [B, 1]
    denominator = structure_mask.sum(dim=(2, 3, 4)).clamp(min=1)      # [B, 1]
    volume_fraction = (numerator / denominator) * 100.0               # [%]

    V_x = volume_fraction.mean()     # scalar (across batch)
    return V_x ** p_norm

import math
import torch
import torch.nn.functional as F


def dvh_percentile_loss_with_threshold(
    dose_pred: torch.Tensor,
    structure_mask: torch.Tensor,
    volume_percent: float,
    alpha: float,            # +1: push Dp down,  -1: push Dp up
    bound_value: float,      # Gy: threshold for Dp (e.g. 29.9 for Dmax<=29.9)
    p_norm: float = 1.0,     # exponent on Dp, e.g. 1.0 or 2.0
    slope: float = 0.1,      # 0 < slope <= 1: strength after constraint is satisfied
) -> torch.Tensor:
    """
    Hybrid DVH percentile objective:

        D_p% := dose at given volume_percent (e.g. 99 -> D99, 0 -> Dmax)
        base_objective = D_p^p_norm

    Direction:
        alpha > 0  => minimizing loss pushes D_p DOWN (toward 0 Gy)
        alpha < 0  => minimizing loss pushes D_p UP   (toward large dose)

    Threshold logic with bound_value (in Gy):
        - If alpha > 0 (push down), we interpret bound_value as an *upper bound*:
              D_p >  bound_value  => full strength (scale = 1)
              D_p <= bound_value  => weakened strength (scale = slope)
        - If alpha < 0 (push up), we interpret bound_value as a *lower bound*:
              D_p <  bound_value  => full strength (scale = 1)
              D_p >= bound_value  => weakened strength (scale = slope)

    No hinge / no sign flip at the threshold:
        The objective always pushes in the same direction given by alpha,
        but its *weight* changes once the constraint is satisfied.

    Returns:
        Scalar loss tensor.
    """
    # Ensure channel dimension
    if dose_pred.ndim == 3:
        dose_pred = dose_pred.unsqueeze(0)       # [1, D, H, W]
    if dose_pred.ndim == 4:
        dose_pred = dose_pred.unsqueeze(1)       # [B, 1, D, H, W]
    if structure_mask.ndim == 3:
        structure_mask = structure_mask.unsqueeze(0)
    if structure_mask.ndim == 4:
        structure_mask = structure_mask.unsqueeze(1)

    # Extract structure voxels
    structure_doses = dose_pred[structure_mask > 0]
    if structure_doses.numel() == 0:
        return torch.tensor(0.0, device=dose_pred.device, dtype=dose_pred.dtype)

    # Sort descending to get D_p%
    sorted_doses, _ = torch.sort(structure_doses, descending=True)
    N = sorted_doses.numel()

    p = float(volume_percent) / 100.0
    p = max(0.0, min(1.0, p))
    idx = int(math.ceil(p * N)) - 1
    if idx < 0:
        idx = 0
    if idx >= N:
        idx = N - 1

    D_p = sorted_doses[idx]  # scalar (Gy)

    # Base objective: D_p^p_norm (always ≥ 0)
    base = D_p ** p_norm

    # Determine scale depending on whether constraint is violated or satisfied
    if not torch.is_tensor(bound_value):
        bound_value = torch.tensor(
            bound_value, dtype=dose_pred.dtype, device=dose_pred.device
        )

    if alpha > 0:
        # Push DOWN; bound_value is an upper bound.
        # If D_p > bound_value: still violating => full strength.
        # If D_p <= bound_value: satisfied => scaled by slope.
        if D_p > bound_value:
            scale = 1.0
        else:
            scale = slope
    elif alpha < 0:
        # Push UP; bound_value is a lower bound.
        # If D_p < bound_value: violating => full strength.
        # If D_p >= bound_value: satisfied => scaled by slope.
        if D_p < bound_value:
            scale = 1.0
        else:
            scale = slope
    else:
        # alpha == 0 => no contribution
        return torch.tensor(0.0, device=dose_pred.device, dtype=dose_pred.dtype)

    # Final loss
    loss = alpha * scale * base
    return loss

def dvh_volume_loss_with_threshold(
    dose_pred: torch.Tensor,
    structure_mask: torch.Tensor,
    dose_threshold: float,        # Gy: x in V_x Gy
    alpha: float,                 # +1: push Vx down,  -1: push Vx up
    volume_bound_percent: float,  # %: threshold on Vx (e.g. 15 for V40Gy<=15%)
    temperature: float = 10.0,    # sigmoid slope on dose
    p_norm: float = 1.0,          # exponent on Vx
    slope: float = 0.1,           # scale factor once constraint satisfied
) -> torch.Tensor:
    """
    Hybrid DVH volume-at-dose objective:

        V_x := volume fraction (%) of structure receiving >= dose_threshold

        base_objective = V_x^p_norm

    Direction:
        alpha > 0  => minimizing loss pushes V_x DOWN  (less volume ≥ x Gy)
        alpha < 0  => minimizing loss pushes V_x UP    (more volume ≥ x Gy)

    Threshold logic with volume_bound_percent (in %):
        - If alpha > 0 (push down), interpret volume_bound_percent as *upper bound*:
              V_x >  bound  => full strength
              V_x <= bound  => scaled by slope
        - If alpha < 0 (push up), interpret volume_bound_percent as *lower bound*:
              V_x <  bound  => full strength
              V_x >= bound  => scaled by slope

    Always pushes in same direction (no hinge), weight just changes after threshold.

    Returns:
        Scalar loss tensor.
    """
    # Ensure channel dimension
    if dose_pred.ndim == 3:
        dose_pred = dose_pred.unsqueeze(0)
    if dose_pred.ndim == 4:
        dose_pred = dose_pred.unsqueeze(1)
    if structure_mask.ndim == 3:
        structure_mask = structure_mask.unsqueeze(0)
    if structure_mask.ndim == 4:
        structure_mask = structure_mask.unsqueeze(1)

    # Threshold as tensor
    if not torch.is_tensor(dose_threshold):
        dose_threshold = torch.tensor(
            dose_threshold, dtype=dose_pred.dtype, device=dose_pred.device
        )

    # Soft indicator: dose >= dose_threshold
    soft_indicator = torch.sigmoid(temperature * (dose_pred - dose_threshold))

    # Volume fraction above threshold (%)
    numerator = (soft_indicator * structure_mask).sum(dim=(2, 3, 4))  # [B, 1]
    denominator = structure_mask.sum(dim=(2, 3, 4)).clamp(min=1)      # [B, 1]
    volume_fraction = (numerator / denominator) * 100.0               # [%]

    # Average over batch/channels
    V_x = volume_fraction.mean()   # scalar %

    # Base objective
    base = V_x ** p_norm

    if not torch.is_tensor(volume_bound_percent):
        volume_bound_percent = torch.tensor(
            volume_bound_percent, dtype=dose_pred.dtype, device=dose_pred.device
        )

    if alpha > 0:
        # Push DOWN; bound is upper bound.
        if V_x > volume_bound_percent:
            scale = 1.0
        else:
            scale = slope
    elif alpha < 0:
        # Push UP; bound is lower bound.
        if V_x < volume_bound_percent:
            scale = 1.0
        else:
            scale = slope
    else:
        return torch.tensor(0.0, device=dose_pred.device, dtype=dose_pred.dtype)

    loss = alpha * scale * base
    return loss

def dvh_Dp_loss(
    dose_pred,
    mask,
    p,                   # percentile (0–100)
    target,
    direction,           # "at_least" or "at_most"
    violation_weight=10,
    slack_weight=0.1
):
    """
    Smooth hinge loss on D_p%.
    Extremely stable. Always converges.
    """
    # Extract structure
    if dose_pred.ndim == 3: dose_pred = dose_pred.unsqueeze(0).unsqueeze(0)
    if dose_pred.ndim == 4: dose_pred = dose_pred.unsqueeze(1)
    if mask.ndim == 3: mask = mask.unsqueeze(0).unsqueeze(0)
    if mask.ndim == 4: mask = mask.unsqueeze(1)

    vals = dose_pred[mask > 0]

    if vals.numel() == 0:
        return torch.tensor(0.0, device=dose_pred.device)

    # Compute Dp
    vals_sorted, _ = torch.sort(vals)
    k = int((p/100.0) * (vals_sorted.numel() - 1))
    Dp = vals_sorted[k]

    # Smooth hinge
    if direction == "at_least":
        diff = target - Dp
        violation = F.relu(diff)          # Dp < target
        slack     = F.relu(-diff)         # Dp > target
    else: # at_most
        diff = Dp - target
        violation = F.relu(diff)          # Dp > target
        slack     = F.relu(-diff)         # Dp < target

    return violation_weight * violation**2 + slack_weight * slack**2
def dvh_Vx_loss(
    dose_pred,
    mask,
    x,                     # dose threshold (Gy)
    target_volume_percent,
    direction,             # "at_least" or "at_most"
    temperature=10.0,
    violation_weight=10,
    slack_weight=0.1
):
    """
    Smooth hinge loss on V_x%.
    Very stable, differentiable, converges well.
    """
    # Ensure shape
    if dose_pred.ndim == 3: dose_pred = dose_pred.unsqueeze(0).unsqueeze(0)
    if dose_pred.ndim == 4: dose_pred = dose_pred.unsqueeze(1)
    if mask.ndim == 3: mask = mask.unsqueeze(0).unsqueeze(0)
    if mask.ndim == 4: mask = mask.unsqueeze(1)

    # Soft indicator dose >= x
    soft_indicator = torch.sigmoid(temperature * (dose_pred - x))

    num = (soft_indicator * mask).sum()
    den = mask.sum().clamp(min=1)
    Vx = (num / den) * 100.0  # percent

    diff = Vx - target_volume_percent

    if direction == "at_most":
        violation = F.relu(diff)           # too much volume
        slack     = F.relu(-diff)
    else: # at_least
        violation = F.relu(-diff)          # too little volume
        slack     = F.relu(diff)

    return violation_weight * violation**2 + slack_weight * slack**2
