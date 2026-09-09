"""Heterogeneity-aware multiple-Coulomb-scattering (MCS) for ion pencil beams.

A pencil-beam kernel table stores one lateral sigma per (energy, depth), measured
in **water**. Two rays at the same water-equivalent depth (WEQ) can have
travelled very different *geometric* distances -- a ray through lung much further
than one through muscle -- and both the scattering and the lever arm to the
scoring point follow the geometric path. The tabulated water sigma is therefore
too narrow behind low-density material and slightly too wide behind dense
material.

:func:`fermi_eyges_excess` is that correction and nothing else: the Fermi-Eyges
lever-arm variance *excess* to be added to the squared transport sigma,

.. math::
    E(z_d) = \\sum_{i \\le d} \\left[(z_d - z_i)^2 - (w_d - w_i)^2\\right]\\,
             \\Delta\\Theta^2_i

with :math:`z` the geometric depth and :math:`w` the WEQ depth. The two terms
cancel exactly when :math:`z \\equiv w`, i.e. **the excess is identically zero in
water** -- a homogeneous water phantom is unaffected by construction, which is
the property the engine's regression set pins.

The per-step angular variance :math:`\\Delta\\Theta^2_i` is Kanematsu's
differential-Highland scattering power (Kanematsu, NIM B 266 (2008); Fuchs et
al., Med. Phys. 39 (2012)), closed form in the local residual range and WEQ:

.. math::
    T_{dH} = f_{dH}(\\ell)\\,\\frac{E_s^2}{X_0}\\left(\\frac{z}{pv}\\right)^2,
    \\qquad E_s = 15\\,\\mathrm{MeV}

    f_{dH}(\\ell) \\approx 0.970\\,(1 + \\ln\\ell/20.7)(1 + \\ln\\ell/22.7),
    \\qquad \\ell = \\mathrm{WEQ} / X_0^{H_2O}

    (pv/\\mathrm{MeV})^2 = (R / 4.67\\times10^{-4}\\,\\mathrm{cm})^{1.08},
    \\qquad R = R_0 - \\mathrm{WEQ}
"""

from __future__ import annotations

import torch

__all__ = ["WATER_MASS_RADIATION_LENGTH_G_CM2", "fermi_eyges_excess"]

#: Mass radiation length of water, g/cm^2. Sets the scattering rate per unit
#: water-equivalent path; also the unit in which the radiative path length
#: ``ell`` of the differential-Highland form factor is measured.
WATER_MASS_RADIATION_LENGTH_G_CM2 = 36.08

#: Highland scattering constant E_s, squared, in MeV^2.
_HIGHLAND_ES_SQ_MEV2 = 225.0

#: Proton rest mass, MeV.
_PROTON_MASS_MEV = 938.272

#: Bragg-Kleeman coefficient of the water range-energy relation, cm/MeV^1.77.
_BRAGG_KLEEMAN_ALPHA_CM = 4.67e-4

#: Residual range, cm, below which scattering is frozen: the closed-form
#: scattering power diverges as the range goes to zero, and the last millimetre
#: of the track carries no dose worth broadening.
_MIN_RESIDUAL_RANGE_CM = 0.1


def fermi_eyges_excess(
    weq_depth_mm: torch.Tensor,
    energy_mev: float,
    depth_step_mm: float,
    *,
    mass_radiation_length_g_cm2: float = WATER_MASS_RADIATION_LENGTH_G_CM2,
) -> torch.Tensor:
    """Lever-arm variance excess to add to ``sigma_transport**2``, in mm^2.

    Args:
        weq_depth_mm: ``(S, D)`` water-equivalent depth in mm along ``D`` rays'
            worth of transport steps, one row per sub-beam. Must be
            monotonically non-decreasing along ``D`` (it is a cumulative path
            length); a decrease is clamped to a zero step.
        energy_mev: Beamlet kinetic energy at the phantom surface, MeV.
        depth_step_mm: Geometric length of one step along ``D``, mm. Together
            with the row index this defines the geometric depth ``z``.
        mass_radiation_length_g_cm2: Mass radiation length ``rho * X0`` for the
            scattering rate. Defaults to water, which is what the engine uses:
            the correction then depends only on the geometry/WEQ mismatch, not
            on a material table.

    Returns:
        ``(S, D)`` variance excess in mm^2, ``>= 0`` wherever the geometric path
        exceeds the water-equivalent one, and **identically zero** where the two
        agree (homogeneous water).
    """
    if weq_depth_mm.ndim != 2:
        raise ValueError(f"weq_depth_mm must be (S, D), got {tuple(weq_depth_mm.shape)}")

    _S, num_steps = weq_depth_mm.shape
    weq_mm = weq_depth_mm
    device, dtype = weq_mm.device, weq_mm.dtype

    # Residual range from the Bragg-Kleeman relation, in water.
    pv_surface = energy_mev * (energy_mev + 2.0 * _PROTON_MASS_MEV) / (energy_mev + _PROTON_MASS_MEV)
    range_cm = _BRAGG_KLEEMAN_ALPHA_CM * (pv_surface * pv_surface) ** (1.0 / 1.08)
    weq_cm = weq_mm / 10.0
    residual_cm = range_cm - weq_cm
    # Floored so that pv stays finite as the residual range goes to zero.
    pv = (residual_cm.clamp_min(_MIN_RESIDUAL_RANGE_CM) / _BRAGG_KLEEMAN_ALPHA_CM) ** 0.54

    # Differential-Highland form factor of the radiative path length.
    ell = (weq_cm / mass_radiation_length_g_cm2).clamp_min(1e-6)
    log_ell = torch.log(ell)
    form_factor = (0.970 * (1.0 + log_ell / 20.7) * (1.0 + log_ell / 22.7)).clamp_min(0.0)

    # WEQ increment per step, mm. Step 0 measures from the surface.
    weq_step_mm = torch.zeros_like(weq_mm)
    weq_step_mm[:, 0] = weq_mm[:, 0]
    weq_step_mm[:, 1:] = weq_mm[:, 1:] - weq_mm[:, :-1]
    weq_step_mm = weq_step_mm.clamp_min(0.0)

    in_range = (residual_cm > _MIN_RESIDUAL_RANGE_CM).to(dtype)
    delta_theta_sq = (
        in_range
        * form_factor
        * (_HIGHLAND_ES_SQ_MEV2 / (pv * pv))
        * (weq_step_mm / 10.0 / mass_radiation_length_g_cm2)
    )

    # E(z_d) expanded into running sums so the double sum costs one pass.
    z_mm = torch.arange(num_steps, device=device, dtype=dtype) * depth_step_mm
    sum_theta = delta_theta_sq.cumsum(1)
    sum_theta_z = (delta_theta_sq * z_mm).cumsum(1)
    sum_theta_w = (delta_theta_sq * weq_mm).cumsum(1)
    sum_theta_z2 = (delta_theta_sq * z_mm * z_mm).cumsum(1)
    sum_theta_w2 = (delta_theta_sq * weq_mm * weq_mm).cumsum(1)
    return (
        (z_mm * z_mm - weq_mm * weq_mm) * sum_theta
        - 2.0 * z_mm * sum_theta_z
        + 2.0 * weq_mm * sum_theta_w
        + sum_theta_z2
        - sum_theta_w2
    )
