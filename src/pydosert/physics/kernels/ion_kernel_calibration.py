"""Learnable residuals over an :class:`IonKernelTable` -- differentiable kernel-table calibration.

The commissioned kernel table is a *fit* to Monte Carlo, and the fit is never
perfect: the depth dose and the lateral halo of a proton pencil beam carry a
few per cent of residual error against a water-phantom MC reference. Because
:class:`~pydosert.engine.ion_dose_engine.IonDoseEngine` is differentiable
end-to-end (it disables autograd nowhere), that error can be removed by
gradient descent on the table itself: run the engine, compare with MC,
backpropagate into the depth curves.

This module is the *parameterisation* of that calibration. It holds one set of
learnable residuals per selected energy row and produces a calibrated
:class:`IonKernelTable` from them; the fitting recipe (MC loading, the loss, the
optimiser) lives in ``commissioning/calibrate_ion_kernel_table.py``.

What the parameterisation guarantees
------------------------------------
* **Exact identity at initialisation.** With the residuals at zero the
  calibrated table is *bit-identical* to the input table. Hence residuals that
  are additive in value space: ``softplus(inv_softplus(x))`` is not exactly
  ``x``, so re-deriving the base curve would perturb every sample by an ulp
  before the first optimiser step.
* **Physical curves for any parameter value.** No negative IDD or sigma, no
  broad Gaussian narrower than the narrow one, no halo weight outside
  ``[0, 1]`` -- via ``softplus`` for the positive curves, ``sigmoid`` for the
  weight, ``sigma2 = sigma1 + softplus(gap)`` for the ordering, plus a final
  clamp on each output.
* **Padding is never written.** A residual is masked to its row's ``n_valid``
  samples, so the ``+inf``/zero/edge padding comes through untouched.
* **Unselected rows pass through** bit-identically.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .ion_kernel_table import IonKernelTable

#: Positive curves are floored here before ``inv_softplus``: ``log(expm1(0))`` is
#: ``-inf``. Matches the historical calibration driver.
SOFTPLUS_FLOOR = 1e-6

#: The halo weight is clamped away from the poles of ``logit`` by this much.
#: Matches the historical calibration driver.
LOGIT_MARGIN = 1e-4


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`torch.nn.functional.softplus`, floored at :data:`SOFTPLUS_FLOOR`.

    Non-positive entries are floored rather than rejected: a commissioned curve
    legitimately contains exact zeros (the transport sigma at zero depth, the
    IDD past the range).
    """
    return torch.log(torch.expm1(y.clamp_min(SOFTPLUS_FLOOR)))


def inv_sigmoid(p: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`torch.sigmoid`, with the poles clamped by :data:`LOGIT_MARGIN`."""
    return torch.logit(p.clamp(LOGIT_MARGIN, 1.0 - LOGIT_MARGIN))


class IonKernelCalibration(nn.Module):
    """Learnable residuals over selected energy rows of an :class:`IonKernelTable`.

    Five curves are calibrated per selected energy: the integrated depth dose
    ``idd``, the single-Gaussian transport ``sigma``, and the double-Gaussian
    ``sigma1`` / ``sigma2`` / ``weight``. Each is parameterised so that the
    optimiser works in an unconstrained space while the curve it produces stays
    physical:

    ==============  =======================================================================
    curve           calibrated value
    ==============  =======================================================================
    ``idd``         ``clamp_min(idd0 + softplus(r0 + d) - softplus(r0), 0)``
    ``sigma``       ``clamp_min(sigma0 + softplus(r0 + d) - softplus(r0), 0)``
    ``sigma1``      ``clamp_min(sigma1_0 + softplus(r0 + d) - softplus(r0), 0)``
    ``sigma2``      ``max(sigma1 + gap, sigma1)`` with ``gap`` built like the curves above
    ``weight``      ``clamp(weight0 + sigmoid(r0 + d) - sigmoid(r0), 0, 1)``
    ==============  =======================================================================

    where ``r0`` is the inverse transform of the base curve (a constant buffer)
    and ``d`` the learnable residual, masked to the row's valid samples. At
    ``d = 0`` every transform difference is exactly zero and every clamp is
    inert on commissioned data, so :meth:`apply` returns the input table
    bit-for-bit.

    ``sigma2 = sigma1 + gap`` with ``gap >= 0`` keeps the broad component from
    falling below the narrow one whatever the optimiser does; ``sigma2_fix``
    carries the one-ulp term that ``sigma1 + (sigma2 - sigma1)`` loses, and the
    outer ``max`` restores the ordering the correction could otherwise break.

    The base curves are snapshotted at construction; mutating the source table
    afterwards does not update them.

    Args:
        kernel_table: The table to calibrate. It is never mutated.
            Available afterwards as :attr:`kernel_table`.
        energies_mev: Tabulated energies (MeV) to calibrate; ``None`` selects
            every row. Energies are resolved with
            :meth:`~pydosert.physics.kernels.ion_kernel_table.IonKernelTable.row_index`,
            so an untabulated energy raises rather than snapping to a neighbour.

    Raises:
        ValueError: If ``energies_mev`` is empty, names the same row twice, or
            if the table has ``sigma2 < sigma1`` anywhere (an unphysical table
            that the ordering constraint would silently repair).
        EnergyNotInTableError: If an energy is not tabulated.
    """

    def __init__(self, kernel_table: IonKernelTable, energies_mev: Sequence[float] | None = None) -> None:
        super().__init__()

        if energies_mev is None:
            rows = list(range(kernel_table.num_energies))
        else:
            energies = [float(e) for e in energies_mev]
            if not energies:
                raise ValueError("energies_mev is empty; pass None to calibrate every row")
            rows = [kernel_table.row_index(e) for e in energies]
            if len(set(rows)) != len(rows):
                raise ValueError(f"energies_mev selects the same table row twice: {energies}")

        device, dtype = kernel_table.device, kernel_table.dtype
        row_index = torch.tensor(rows, device=device, dtype=torch.long)
        n_valid = kernel_table.n_valid.index_select(0, row_index)
        d_max = kernel_table.depth_mm.shape[1]
        positions = torch.arange(d_max, device=device).view(1, d_max)
        valid_mask = positions < n_valid.view(-1, 1)

        def base(name: str) -> torch.Tensor:
            return getattr(kernel_table, name).index_select(0, row_index).detach().clone()

        idd_base = base("idd")
        sigma_base = base("sigma_mm")
        sigma1_base = base("sigma1_mm")
        sigma2_base = base("sigma2_mm")
        weight_base = base("weight")

        if bool((sigma2_base < sigma1_base).any()):
            raise ValueError(
                "the table has sigma2 < sigma1 on a selected row; the calibration builds "
                "sigma2 as sigma1 + a non-negative gap and would silently repair that, "
                "which would hide a broken commissioning table"
            )

        gap_base = sigma2_base - sigma1_base
        # `sigma1 + (sigma2 - sigma1)` misses `sigma2` by an ulp on a few thousand
        # samples of a real table. The correction is exact (Sterbenz), so adding
        # it back reproduces sigma2 bit-for-bit at initialisation.
        sigma2_fix = sigma2_base - (sigma1_base + gap_base)

        self.kernel_table = kernel_table
        self.register_buffer("row_index", row_index)
        self.register_buffer("n_valid", n_valid)
        self.register_buffer("valid_mask", valid_mask)
        self.register_buffer("idd_base", idd_base)
        self.register_buffer("sigma_base", sigma_base)
        self.register_buffer("sigma1_base", sigma1_base)
        self.register_buffer("gap_base", gap_base)
        self.register_buffer("sigma2_fix", sigma2_fix)
        self.register_buffer("weight_base", weight_base)
        self.register_buffer("idd_raw", inv_softplus(idd_base))
        self.register_buffer("sigma_raw", inv_softplus(sigma_base))
        self.register_buffer("sigma1_raw", inv_softplus(sigma1_base))
        # The floor only sets the *scale* of the residual; the base gap itself is
        # exact, so a hair-thin gap is reproduced rather than widened.
        self.register_buffer("gap_raw", inv_softplus(gap_base.clamp_min(1e-3)))
        self.register_buffer("weight_raw", inv_sigmoid(weight_base))

        shape = (len(rows), d_max)
        self.idd_residual = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))
        self.sigma_residual = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))
        self.sigma1_residual = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))
        self.gap_residual = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))
        self.weight_residual = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))

    # ------------------------------------------------------------- properties

    @property
    def num_rows(self) -> int:
        """Number of calibrated energy rows ``R``."""
        return int(self.row_index.numel())

    @property
    def energies_mev(self) -> list[float]:
        """The calibrated energies in MeV, in the order the parameters are stored."""
        return [float(self.kernel_table.energy_mev[i]) for i in self.row_index.tolist()]

    @property
    def device(self) -> torch.device:
        """Device the residuals live on."""
        return self.idd_residual.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating dtype of the residuals."""
        return self.idd_residual.dtype

    # --------------------------------------------------------------- internals

    def _masked(self, residual: torch.Tensor) -> torch.Tensor:
        """Zero a residual outside its row's valid samples, keeping the gradient there zero."""
        return torch.where(self.valid_mask, residual, torch.zeros((), device=residual.device, dtype=residual.dtype))

    def _positive(self, base: torch.Tensor, raw: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """A softplus-shaped residual added to a non-negative base curve.

        Exactly ``base`` at ``residual == 0``: the two ``softplus`` calls cancel
        bit-for-bit. The clamp is a rail for extreme parameter values.
        """
        shifted = F.softplus(raw + self._masked(residual)) - F.softplus(raw)
        return (base + shifted).clamp_min(0.0)

    # -------------------------------------------------------------- the curves

    def calibrated_rows(self) -> dict[str, torch.Tensor]:
        """The calibrated curves of the selected rows.

        Returns:
            ``{"idd", "sigma_mm", "sigma1_mm", "sigma2_mm", "weight"}`` -- the
            attribute names of the curves on :class:`IonKernelTable` -- each
            ``(R, Dmax)`` and differentiable with respect to the residuals.
        """
        idd = self._positive(self.idd_base, self.idd_raw, self.idd_residual)
        sigma = self._positive(self.sigma_base, self.sigma_raw, self.sigma_residual)
        sigma1 = self._positive(self.sigma1_base, self.sigma1_raw, self.sigma1_residual)
        gap = self._positive(self.gap_base, self.gap_raw, self.gap_residual)
        # `max(..., sigma1)` is the ordering guarantee: only the ulp-sized
        # `sigma2_fix` could push the sum below sigma1, and it cannot survive it.
        sigma2 = torch.maximum(sigma1 + gap + self.sigma2_fix, sigma1)
        weight_shift = torch.sigmoid(self.weight_raw + self._masked(self.weight_residual)) - torch.sigmoid(
            self.weight_raw
        )
        weight = (self.weight_base + weight_shift).clamp(0.0, 1.0)
        return {
            "idd": idd,
            "sigma_mm": sigma,
            "sigma1_mm": sigma1,
            "sigma2_mm": sigma2,
            "weight": weight,
        }

    def apply(self) -> IonKernelTable:
        """Return the calibrated table.

        Deliberately shadows :meth:`torch.nn.Module.apply`; ``.to()``, ``.cuda()``
        and friends go through ``_apply`` and are unaffected.

        The input table is not mutated: calibrated rows are written into
        out-of-place copies, so unselected rows stay bit-identical and the
        result carries gradients back to the residuals.

        Returns:
            A new :class:`IonKernelTable` with the same geometry, energies and
            metadata as the input, and calibrated ``idd`` / ``sigma`` /
            ``sigma1`` / ``sigma2`` / ``weight`` rows.
        """
        curves = self.calibrated_rows()
        rows = self.row_index
        table = self.kernel_table
        return replace(
            table,
            idd=table.idd.index_copy(0, rows, curves["idd"]),
            sigma_mm=table.sigma_mm.index_copy(0, rows, curves["sigma_mm"]),
            sigma1_mm=table.sigma1_mm.index_copy(0, rows, curves["sigma1_mm"]),
            sigma2_mm=table.sigma2_mm.index_copy(0, rows, curves["sigma2_mm"]),
            weight=table.weight.index_copy(0, rows, curves["weight"]),
        )

    def forward(self) -> IonKernelTable:
        """Alias of :meth:`apply`, so the calibration composes as a normal module."""
        return self.apply()

    # ------------------------------------------------------------ the prior

    def smoothness_penalty(self, curves: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Second-difference roughness of the calibrated curves.

        The depth curves are smooth; without this prior a per-sample residual
        happily fits MC noise. Mean squared second difference over each row's
        valid samples, summed over the five curves and averaged over the rows.

        Args:
            curves: The output of :meth:`calibrated_rows`, if it has already
                been computed. Recomputed when ``None``.

        Returns:
            A 0-d tensor, differentiable with respect to the residuals.
        """
        curves = self.calibrated_rows() if curves is None else curves
        total = torch.zeros((), device=self.device, dtype=self.dtype)
        for row in range(self.num_rows):
            n = int(self.n_valid[row])
            for curve in curves.values():
                values = curve[row, :n]
                total = total + (values[2:] - 2.0 * values[1:-1] + values[:-2]).pow(2).mean()
        return total / float(self.num_rows)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        energies = self.energies_mev
        listed = ", ".join(f"{e:.4f}" for e in energies[:4]) + (", ..." if len(energies) > 4 else "")
        return (
            f"IonKernelCalibration(rows={self.num_rows}, Dmax={self.idd_residual.shape[1]}, "
            f"energies=[{listed}] MeV, device={self.device}, dtype={self.dtype})"
        )
