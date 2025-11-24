import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydose_rt.utils.utils import get_model_input, create_bound_weight_matrix
import math

def scale_loss(loss, weight):
    return loss * weight

class TrainableLossWeightsNormalized(nn.Module):
    def __init__(self, num_losses=2, sum_value=100):
        """
        Implements trainable loss weights that are normalized to sum to 1.
        Args:
            num_losses (int): Number of loss components.
        """
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(num_losses))
        self.sum_value = sum_value

    def forward(self, losses):
        """
        Combines the individual loss components with normalized trainable weights.
        Args:
            losses: A list or tensor of loss terms [L1, L2, ..., Ln].
        Returns:
            A scalar tensor representing the weighted sum of losses.
        """
        losses = torch.stack(losses)
        raw_weights = torch.exp(-2.0 * self.log_sigma)
        sum_weights = raw_weights.sum() + 1e-8
        norm_weights = raw_weights / sum_weights * self.sum_value
        total_loss = (norm_weights * losses).sum()
        return total_loss


class SumLoss(nn.Module):
    """
    Sums all loss components in a list or tensor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, losses):
        """
        Args:
            losses: A list or tensor of loss terms [L1, L2, ..., Ln].
        Returns:
            A scalar tensor representing the sum of losses.
        """
        losses = torch.stack(losses)
        return losses.sum()


# ======================================================================================
# Auxiliary Loss Function
# ======================================================================================
def auxiliary_loss(dose_pred, dose_bypass):
    """
    Auxiliary loss that encourages the bypass branch output to match the static dose.
    For example, use Mean Squared Error between dose_static and dose_bypass.
    """
    return torch.mean((dose_pred - dose_bypass) ** 2)



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
# dvh loss
# ======================================================================================


class DVHLoss:
    def __init__(self, constraints, k=1000, masks=None, region_weights=None):
        self.lower_bound_gy = constraints["lower_bound_gy"]
        self.higher_bound_gy = constraints["higher_bound_gy"]
        self.lower_bound_target_percent = constraints["lower_bound_target_percent"]
        self.higher_bound_target_percent = constraints["higher_bound_target_percent"]
        self.k = k
        self.masks = masks
        self.region_weights = region_weights

    def soft_fraction_above(self, dose, mask, threshold):
        if not torch.is_tensor(threshold):
            threshold = torch.tensor(threshold, dtype=dose.dtype, device=dose.device)
        else:
            threshold = threshold.to(dose.device, dtype=dose.dtype)
        soft_indicator = torch.sigmoid(self.k * (dose - threshold))
        numerator = (soft_indicator * mask).sum(dim=(2, 3, 4))
        denominator = mask.sum(dim=(2, 3, 4)) + 1e-8
        fraction = numerator / denominator  # shape: [B, 1]
        return fraction.squeeze(1)  # shape: [B]

    def get(self, y_true, y_pred, factor=100):
        if self.masks is None:
            raise ValueError("Masks dictionary must be provided.")

        loss_lower_bound_target = []
        loss_higher_bound_target = []
        weight_list = []

        for region, mask in self.masks.items():
            if self.region_weights is not None:
                if isinstance(self.region_weights, dict):
                    weight = (mask * self.region_weights[region]).max()
                else:
                    weight = (mask * self.region_weights).max()
            else:
                weight = 1.0
            weight_list.append(weight)

            lb_target = self.lower_bound_target_percent.get(region, 0) / factor
            lb_gy = self.lower_bound_gy.get(region, None)
            if lb_gy is None:
                raise ValueError(f"Lower bound Gy not provided for region {region}")
            frac_lb = self.soft_fraction_above(y_pred, mask, lb_gy)
            # Ensure lb_target is a tensor on the same device as frac_lb
            if not torch.is_tensor(lb_target):
                lb_target = torch.tensor(
                    lb_target, dtype=frac_lb.dtype, device=frac_lb.device
                )
            else:
                lb_target = lb_target.to(frac_lb.device, dtype=frac_lb.dtype)
            loss_lb = (F.relu(lb_target - frac_lb)) ** 2
            loss_lb = loss_lb * weight
            loss_lower_bound_target.append(loss_lb.mean())

            hb_target = self.higher_bound_target_percent.get(region, 1) / factor
            hb_gy = self.higher_bound_gy.get(region, None)
            if hb_gy is None:
                raise ValueError(f"Higher bound Gy not provided for region {region}")
            frac_hb = self.soft_fraction_above(y_pred, mask, hb_gy)
            # Ensure hb_target is a tensor on the same device as frac_hb
            if not torch.is_tensor(hb_target):
                hb_target = torch.tensor(
                    hb_target, dtype=frac_hb.dtype, device=frac_hb.device
                )
            else:
                hb_target = hb_target.to(frac_hb.device, dtype=frac_hb.dtype)
            # SAT: I replaced this original line:
            # loss_hb = (F.relu(frac_hb - (1 - hb_target))) ** 2
            loss_hb = F.relu(frac_hb - hb_target)**2
            loss_hb = loss_hb * weight
            loss_higher_bound_target.append(loss_hb.mean())

        loss_lower_bound_target = torch.stack(loss_lower_bound_target).sum()
        loss_higher_bound_target = torch.stack(loss_higher_bound_target).sum()

        weight_sum = sum(weight_list)
        if weight_sum == 0:
            weight_sum = 1.0
        loss_lower_bound_target = loss_lower_bound_target / weight_sum
        loss_higher_bound_target = loss_higher_bound_target / weight_sum
        return loss_lower_bound_target, loss_higher_bound_target


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
    ) / config.field_size_in_pixels[1]
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
    ) / config.field_size_in_pixels[1]
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
    relevant_masks = [
        patient.structures["CTV"], 
        patient.structures["PTVT_42.7"], 
        patient.structures["External"]
    ]
    for mask in relevant_masks:
        losses.append(torch.mean(torch.abs((dose_true - dose_pred)[0, mask > 0])**2))

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