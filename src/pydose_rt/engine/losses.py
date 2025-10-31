import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydose_rt.engine.config import config as PARAMS
from pydose_rt import ModelConfig
from pydose_rt.engine.utils.test_utils import TestSetup


# ======================================================================================
# Trainable weights for loss terms
# ======================================================================================
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


def _constraint_loss(
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
    for mask_name in masks:
        if mask_name != "PTV":
            continue

        mask = masks[mask_name]
        loss_lower_bound_gy += (penalty_lower * mask).mean()
        loss_higher_bound_gy += (penalty_upper * mask).mean()


    return loss_lower_bound_gy, loss_higher_bound_gy


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
        if region == "PTV":
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
    def mlc_opening_reg(leafs, min_opening, huge_penalty=1):
        # leafs: [B, C, H, W]
        violation = torch.clamp(leafs[:, 1, :, :] - min_opening, min=0)
        penalty = torch.mean(huge_penalty * violation**2)
        loss = penalty
        return loss

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
    masks_dict = {
        "PTV": masks[:, 0:1, ...],
        "ROI1": masks[:, 1:2, ...],
        "ROI2": masks[:, 2:3, ...],
        "ROI3": masks[:, 3:4, ...],
        "ROI4": masks[:, 4:5, ...],
        "ROI5": masks[:, 5:6, ...],
        "ROI6": masks[:, 6:7, ...],
    }

    loss_lower_bound_gy, loss_higher_bound_gy = constraint_loss(
        dose_pred,
        lower_bound_gy=x[:, 1:2, ...],
        higher_bound_gy=x[:, 2:3, ...],
        masks=masks_dict,
        region_weights=region_weights,
        number_regions=len(masks_dict),
    )

    loss_lower_bound_target, loss_higher_bound_target = DVHLoss(
        constraints,
        k=50,
        masks=masks_dict,
        region_weights=region_weights,
    ).get(None, dose_pred)

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

def result_validation(config, pred_dose, pred_mlc, pred_jaws, pred_mus, x, dose_pred, constraints, masks, region_weights=None, loss_weights=0):
    results = {}

    # Start with values in the predictions
    if (pred_dose.min() < 0):
        results["check_min_dose_pass"] = 0
    else:
        results["check_min_dose_pass"] = 1

    if (pred_mlc.min() < 0) or (pred_mlc.max() > 1):
        results["check_mlc_bounds_pass"] = 0
    else:
        results["check_mlc_bounds_pass"] = 1

    if (pred_jaws.min() < 0) or (pred_jaws.max() > 1):
        results["check_jaws_bounds_pass"] = 0
    else:
        results["check_jaws_bounds_pass"] = 1

    if (pred_mus.min() < 0):
        results["check_mus_bounds_pass"] = 0
    else:
        results["check_mus_bounds_pass"] = 1

    

    return results

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


if __name__ == "__main__":
    # # Example test with PyTorch channel-first convention
    # masks = {
    #     "PTV": torch.ones(size=(1, 1, 1, 1, 1)),
    #     # "ROI1": torch.ones(size=(1, 1, 1, 1, 1)),
    #     # "ROI2": torch.ones(size=(1, 1, 1, 1, 1)),
    #     # "ROI3": torch.ones(size=(1, 1, 1, 1, 1)),
    #     # "ROI4": torch.ones(size=(1, 1, 1, 1, 1)),
    #     # "ROI5": torch.ones(size=(1, 1, 1, 1, 1)),
    #     # "ROI6": torch.ones(size=(1, 1, 1, 1, 1)),
    # }

    # y_pred = torch.ones(size=(1, 1, 1, 1, 1)) * 0.6
    # lower_bound_gy = torch.ones(size=(1, 1, 1, 1, 1)) * 0.74
    # higher_bound_gy = torch.ones(size=(1, 1, 1, 1, 1)) * 0.83
    # region_weights = torch.ones(size=(1, 1, 1, 1, 1)) * 1000
    # y_true = None

    # loss_value = DVHLoss(PARAMS.constraints, k=1000, masks=masks).get(y_true, y_pred)

    # print(
    #     "Computed DVH Loss:",
    #     loss_value[0].detach().cpu().numpy(),
    #     loss_value[1].detach().cpu().numpy(),
    # )

    # cl = constraint_loss(
    #     y_pred, lower_bound_gy, higher_bound_gy, region_weights=region_weights
    # )

    # print(
    #     "constraint_loss:", cl[0].detach().cpu().numpy(), cl[1].detach().cpu().numpy()
    # )

    T = TestSetup()
    T.create_real()
    data = T.data
    config = T.config
    mus = T.mus
    leafs = T.leafs
    constraints = PARAMS.constraints

    dose = data["dose"]
    x = data["x"].transpose(0, 4, 1, 2, 3)
    masks = data["masks"].transpose(0, 4, 1, 2, 3)
    region_weights = x[:, -1]
    dose_bypass = data["dose_bypass"]

    dose = torch.tensor(dose)
    x = torch.tensor(x, dtype=dose.dtype, device=dose.device)
    masks = torch.tensor(masks, dtype=dose.dtype, device=dose.device)
    region_weights = torch.tensor(region_weights, dtype=dose.dtype, device=dose.device)
    dose_bypass = torch.tensor(dose_bypass, dtype=dose.dtype, device=dose.device)
    mus = torch.tensor(mus, dtype=dose.dtype, device=dose.device)
    leafs = torch.tensor(leafs, dtype=dose.dtype, device=dose.device)

    a = 2

    (
        loss_lower_bound_gy,
        loss_higher_bound_gy,
        loss_lower_bound_target,
        loss_higher_bound_target,
        l2_loss_oars_and_background,
    ) = dose_loss(
        x,
        dose,
        constraints,
        masks,
        region_weights,
        None,
    )
    aux_loss = auxiliary_loss(dose, dose_bypass)

    print(f"loss_lower_bound_gy: {loss_lower_bound_gy}")
    print(f"loss_higher_bound_gy: {loss_higher_bound_gy}")
    print(f"loss_lower_bound_target: {loss_lower_bound_target}")
    print(f"loss_higher_bound_target: {loss_higher_bound_target}")
    print(f"l2_loss_oars_and_background: {l2_loss_oars_and_background}")
    print(f"aux_loss: {aux_loss}")

    mu_rate_loss, mu_complexity_loss = mus_loss(mus, config)
    leaf_opening_loss, leaf_rate_loss = leafs_loss(leafs, config)
    print(f"mu_rate_loss: {mu_rate_loss}")
    print(f"mu_complexity_loss: {mu_complexity_loss}")
    print(f"leaf_opening_loss: {leaf_opening_loss}")
    print(f"leaf_rate_loss: {leaf_rate_loss}")


"""
PyTorch:
    loss_lower_bound_gy: 2.146205588360317e-06
    loss_higher_bound_gy: 2.0532525013550185e-05
    loss_lower_bound_target: 0.0082307830452919
    loss_higher_bound_target: 0.004896602127701044
    l2_loss_oars_and_background: 0.05023684725165367
    aux_loss: 0.00012230045103933662
    mu_rate_loss: 1.0163290653508739e-07
    mu_complexity_loss: 0.0028868757653981447
    leaf_opening_loss: 0.0
    leaf_rate_loss: 0.30721014738082886


Tensorflow:
    loss_lower_bound_gy: 2.1462053609866416e-06
    loss_higher_bound_gy: 2.0532526832539588e-05
    loss_lower_bound_target: 0.0082307830452919
    loss_higher_bound_target: 0.004896602127701044
    l2_loss_oars_and_background: 0.05023684725165367
    aux_loss: 0.00012230045103933662
    mu_rate_loss: 1.0163288521880531e-07
    mu_complexity_loss: 0.002886875532567501
    leaf_opening_loss: 0.0
    leaf_rate_loss: 0.3072100281715393
"""
