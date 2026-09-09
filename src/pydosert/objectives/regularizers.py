import torch

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

