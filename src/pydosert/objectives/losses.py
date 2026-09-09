import torch

def upper_penalty(x, c, weight=1.0):
    """
    One-sided squared penalty on values above ``c``: ``weight * mean(relu(x - c)^2)``.

    With ``c = 0`` this reduces to ``weight * mean(x^2)`` (e.g. the peri-PTV ring).

    Args:
        x (torch.Tensor): Voxel doses.
        c (float): Threshold; only doses above it are penalised.
        weight (float): Scaling factor applied to the penalty.

    Returns:
        torch.Tensor: Scalar penalty.
    """
    return weight * torch.relu(x - c).pow(2).mean()


def lower_penalty(x, c, weight=1.0):
    """
    One-sided squared penalty on values below ``c``: ``weight * mean(relu(c - x)^2)``.

    Args:
        x (torch.Tensor): Voxel doses.
        c (float): Threshold; only doses below it are penalised.
        weight (float): Scaling factor applied to the penalty.

    Returns:
        torch.Tensor: Scalar penalty.
    """
    return weight * torch.relu(c - x).pow(2).mean()


def mean_upper_penalty(x, c, weight=1.0):
    """
    Squared penalty on the mean exceeding ``c``: ``weight * relu(mean(x) - c)^2``.

    Args:
        x (torch.Tensor): Voxel doses.
        c (float): Threshold on the mean dose.
        weight (float): Scaling factor applied to the penalty.

    Returns:
        torch.Tensor: Scalar penalty.
    """
    return weight * torch.relu(x.mean() - c).pow(2)


def squared_penalty(x, c, weight=1.0):
    """
    Two-sided squared deviation from ``c``: ``weight * mean((x - c)^2)``.

    Equivalent to ``lower_penalty(x, c) + upper_penalty(x, c)``.

    Args:
        x (torch.Tensor): Voxel doses.
        c (float): Target value.
        weight (float): Scaling factor applied to the penalty.

    Returns:
        torch.Tensor: Scalar penalty.
    """
    return weight * (x - c).pow(2).mean()


def geud(x, a):
    """
    Generalized equivalent uniform dose (gEUD) over a set of voxel doses:
    ``gEUD = mean(x^a)^(1/a)``.

    ``a = 1`` gives the mean dose, ``a -> +inf`` the max dose (serial organ) and
    ``a -> -inf`` the min dose (target). This is a reduction, not a loss: compose
    it with the squared-hinge penalties above to build EUD objectives, since those
    act on a scalar. For example ``upper_penalty(geud(x, 2.5), c)`` is a max-EUD
    constraint and ``squared_penalty(geud(x, 1.0), c)`` a target mean dose
    (``mean_upper_penalty`` equals ``upper_penalty(geud(x, 1), c)``).

    Args:
        x (torch.Tensor): Voxel doses.
        a (float): gEUD exponent.

    Returns:
        torch.Tensor: Scalar gEUD value.
    """
    x = x.clamp_min(0.0)
    if x.numel() == 0:
        return x.new_zeros(())
    a = float(a)
    if abs(a - 1.0) < 1e-6:
        return x.mean()
    m = x.max().clamp_min(1e-6)          # scale by max so x^a stays fp16-stable
    return m * (x / m).pow(a).mean().pow(1.0 / a)
