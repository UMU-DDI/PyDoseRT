"""Collapsed-cone dose kernel — physical coefficients from a Monte-Carlo 6 MV water
kernel (Ahnesjo cumulative-cone parameterisation).

Per polar cone angle theta (from the primary/beam direction), the cumulative-cone
dose kernel along the cone axis at radiological distance s (cm) is:

    h_theta(s) = A(theta) * exp(-a(theta) * s) + B(theta) * exp(-b(theta) * s)

A/a is the primary (short-range) component; B/b the scatter (long-range) tail.
"""
import torch

from pydosert.physics.kernels import collapsed_cone_kernel_data as _data


class CollapsedConeKernel:
    """Wraps the baked 6 MV cumulative-cone coefficients (48 polar cones)."""

    def __init__(self, device=None, dtype=torch.float32):
        self.beam_quality = _data.BEAM_QUALITY
        self.device = device
        self.dtype = dtype
        t = lambda x: torch.tensor(x, device=device, dtype=dtype)
        self.angles_deg = t(_data.ANGLES_DEG)      # [48] polar angle from beam axis
        self.A = t(_data.A)                          # [48] primary amplitude
        self.a = t(_data.a)                          # [48] primary rate (/cm)
        self.B = t(_data.B)                          # [48] scatter amplitude
        self.b = t(_data.b)                          # [48] scatter rate (/cm)
        self.n_polar = len(_data.ANGLES_DEG)

    def interp_at(self, theta_deg: torch.Tensor):
        """Linear-interpolate (A, a, B, b) at arbitrary polar angles [deg]."""
        ang = self.angles_deg
        idx = torch.clamp(torch.searchsorted(ang, theta_deg), 1, self.n_polar - 1)
        lo, hi = idx - 1, idx
        w = ((theta_deg - ang[lo]) / (ang[hi] - ang[lo])).clamp(0, 1)
        lerp = lambda v: v[lo] * (1 - w) + v[hi] * w
        return lerp(self.A), lerp(self.a), lerp(self.B), lerp(self.b)

    def to(self, device=None, dtype=None):
        return CollapsedConeKernel(device=device or self.device, dtype=dtype or self.dtype)
