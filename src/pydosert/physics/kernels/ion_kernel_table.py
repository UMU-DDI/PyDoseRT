"""Padded-rectangular proton pencil-beam kernel table.

The table is the commissioned base data of an ion machine: for every tabulated
energy it carries the integrated depth dose (``idd``), the single-Gaussian
transport sigma, the double-Gaussian (``sigma1``/``sigma2``/``weight``)
parameters, the kernel depth offset and the initial-focus (spot size vs. source
distance) curve.

It is loaded from an ``.npz`` produced by
``commissioning/conversion/convert_proton_mat_to_npz.py``; the ``.mat`` parsing
(and therefore scipy) lives entirely in that converter. Rows are stored
padded-rectangular rather than resampled onto a common depth grid, so the values
round-trip bit-exactly from the commissioning data.

Design rules:

* All lookups are O(1) or O(log E) -- never a linear scan over the energy table.
* No caches. Every accessor is a pure function of the stored tensors, which is
  what makes learnable per-row residuals possible later.
* Failures are loud. An energy that is not in the table raises; a depth past the
  tabulated range raises for the lateral parameters (a sigma extrapolated 360 mm
  past the range is not a physical answer) and returns zero for the IDD (which
  *is* the physical answer: there is no dose past the range).

Naming: ``idd`` is the *stored* curve of a row; :meth:`IonKernelTable.edep` and
:meth:`IonKernelTable.edep_curve` sample it at arbitrary depths. Same quantity,
named for what the engine does with it (deposited energy per unit path length).

.. warning::
   ``beyond_range="edge"`` holds ``idd[-1]`` past the tabulated range instead of
   returning zero. That is what
   :class:`~pydosert.engine.ion_dose_engine.IonDoseEngine` asks for, and it
   leaves a constant pedestal distal of the Bragg peak -- 0.12 % of peak at
   41 MeV, 1.09 % at 114 MeV. It is preserved deliberately, for numerical
   compatibility with commissioned corrections fitted against that behaviour;
   ``beyond_range="zero"`` is the physical answer and is the default of
   :meth:`edep`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch

FORMAT_VERSION = 1

#: Arrays every table archive must contain.
_REQUIRED_KEYS = (
    "energy_mev",
    "n_valid",
    "depth_mm",
    "idd",
    "sigma",
    "sigma1",
    "sigma2",
    "weight",
    "offset_mm",
    "focus_dist_mm",
    "focus_sigma_mm",
    "sad_mm",
    "bams_to_iso_mm",
)

#: How an accessor reacts to depths past the tabulated range of a row.
#:
#: ``"raise"``  -- raise :class:`DepthOutOfRangeError` (default for the lateral
#:                 parameters, which have no meaningful continuation).
#: ``"zero"``   -- return zero past the range (default for the IDD).
#: ``"edge"``   -- hold the last tabulated value; what the engine uses, and the
#:                 source of the distal pedestal described in the module
#:                 docstring.
BeyondRange = Literal["raise", "zero", "edge"]


class EnergyNotInTableError(ValueError):
    """Raised when a requested energy is not one of the tabulated energies."""


class DepthOutOfRangeError(ValueError):
    """Raised when a requested depth lies past the tabulated range of a row."""


def _interp1d(
    x: torch.Tensor,
    y: torch.Tensor,
    x_new: torch.Tensor,
    left: torch.Tensor | float | None = None,
    right: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Linear 1-D interpolation with edge-value extrapolation.

    Vectorised over an arbitrary shape of ``x_new`` (the engine passes ``(S, D)``
    sub-beam depth blocks). ``x`` must be strictly increasing.

    Args:
        x: Sample positions, shape ``(N,)``.
        y: Sample values, shape ``(N,)``.
        x_new: Query positions, any shape.
        left: Value returned below ``x[0]`` (default ``y[0]``).
        right: Value returned above ``x[-1]`` (default ``y[-1]``).

    Returns:
        Interpolated values with the shape of ``x_new``.
    """
    if x.numel() != y.numel():
        raise ValueError("x and y must have the same length")
    if x.numel() == 1:
        return torch.zeros_like(x_new, dtype=y.dtype, device=y.device) + y[0]

    x_new_flat = x_new.reshape(-1)
    indices = torch.searchsorted(x, x_new_flat, right=False).clamp(1, x.numel() - 1)

    x0 = x[indices - 1]
    x1 = x[indices]
    y0 = y[indices - 1]
    y1 = y[indices]
    slope = (y1 - y0) / (x1 - x0).clamp_min(torch.finfo(y.dtype).eps)
    y_new = y0 + slope * (x_new_flat - x0)

    if left is None:
        left = y[0]
    if right is None:
        right = y[-1]

    left_tensor = torch.as_tensor(left, device=y.device, dtype=y.dtype)
    right_tensor = torch.as_tensor(right, device=y.device, dtype=y.dtype)
    y_new = torch.where(x_new_flat < x[0], left_tensor, y_new)
    y_new = torch.where(x_new_flat > x[-1], right_tensor, y_new)
    return y_new.view_as(x_new)


@dataclass(frozen=True, eq=False)
class IonKernelTable:
    """Commissioned ion pencil-beam base data as padded-rectangular torch tensors.

    All curve tensors live on a single ``device``/``dtype`` (moved once, at load
    time). Row ``e`` uses only its first ``n_valid[e]`` samples; the padding is
    ``+inf`` for ``depth_mm``, ``0`` for ``idd`` and the row's edge value for the
    lateral parameters, so a padded row is never mistaken for data.

    ``energy_mev`` is deliberately kept at float64 whatever ``dtype`` is: it is a
    lookup key, not a curve, and rounding it to float32 would collapse the exact
    match that :meth:`row_index` performs.

    Attributes:
        energy_mev: ``(E,)`` float64 tabulated energies, strictly increasing (MeV).
        n_valid: ``(E,)`` int64 number of tabulated depth samples per energy.
        depth_mm: ``(E, Dmax)`` depth grid per energy (mm), padded with ``+inf``.
        idd: ``(E, Dmax)`` integrated depth dose (matRad ``Z``), padded with 0.
        sigma_mm: ``(E, Dmax)`` single-Gaussian transport sigma (mm).
        sigma1_mm: ``(E, Dmax)`` narrow double-Gaussian sigma (mm).
        sigma2_mm: ``(E, Dmax)`` broad double-Gaussian sigma (mm).
        weight: ``(E, Dmax)`` broad-component weight in ``[0, 1]``.
        offset_mm: ``(E,)`` kernel depth offset (mm).
        focus_dist_mm: ``(E, F)`` initial-focus source distances (mm), increasing.
        focus_sigma_mm: ``(E, F)`` initial spot sigma at those distances (mm).
        sad_mm: Source-to-axis distance (mm).
        bams_to_iso_dist_mm: Beam-application-monitor-system (nozzle exit) to
            isocentre distance (mm). Same quantity as
            :attr:`~pydosert.data.ion_machine.IonMachineConfig.bams_to_iso_dist_mm`.
        allow_energy_interpolation: Default for the accessors' ``allow_energy_interpolation``
            argument. ``False`` (the default) makes an untabulated energy an error.
        source_path: Where the table was loaded from, for error messages.
    """

    energy_mev: torch.Tensor
    n_valid: torch.Tensor
    depth_mm: torch.Tensor
    idd: torch.Tensor
    sigma_mm: torch.Tensor
    sigma1_mm: torch.Tensor
    sigma2_mm: torch.Tensor
    weight: torch.Tensor
    offset_mm: torch.Tensor
    focus_dist_mm: torch.Tensor
    focus_sigma_mm: torch.Tensor
    sad_mm: float
    bams_to_iso_dist_mm: float
    allow_energy_interpolation: bool = False
    source_path: Path | None = None

    # Host-side mirrors derived in __post_init__, never passed to the
    # constructor, so that row lookup never forces a device synchronisation.
    _energies_np: np.ndarray = field(init=False, repr=False, compare=False)
    _n_valid_np: np.ndarray = field(init=False, repr=False, compare=False)
    _energy_index: dict[float, int] = field(init=False, repr=False, compare=False)

    # ------------------------------------------------------------------ setup

    def __post_init__(self) -> None:
        n_energies = int(self.energy_mev.shape[0])
        if self.energy_mev.ndim != 1 or n_energies == 0:
            raise ValueError("energy_mev must be a non-empty 1-D tensor")
        energies = self.energy_mev.detach().to(device="cpu", dtype=torch.float64).numpy()
        if np.any(np.diff(energies) <= 0.0):
            raise ValueError("energy_mev must be strictly increasing")

        d_max = int(self.depth_mm.shape[1])
        for name in ("depth_mm", "idd", "sigma_mm", "sigma1_mm", "sigma2_mm", "weight"):
            tensor = getattr(self, name)
            if tuple(tensor.shape) != (n_energies, d_max):
                raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {(n_energies, d_max)}")
            if tensor.device != self.device or tensor.dtype != self.dtype:
                raise ValueError(f"{name} must be on {self.device} with dtype {self.dtype}")
        for name in ("offset_mm",):
            if tuple(getattr(self, name).shape) != (n_energies,):
                raise ValueError(f"{name} has shape {tuple(getattr(self, name).shape)}, expected {(n_energies,)}")
        if self.focus_dist_mm.shape != self.focus_sigma_mm.shape:
            raise ValueError("focus_dist_mm and focus_sigma_mm must have the same shape")
        if self.focus_dist_mm.shape[0] != n_energies:
            raise ValueError("focus_dist_mm must have one row per energy")

        n_valid_np = self.n_valid.detach().to(device="cpu", dtype=torch.int64).numpy()
        if n_valid_np.shape != (n_energies,):
            raise ValueError("n_valid must have one entry per energy")
        if n_valid_np.min() < 2 or n_valid_np.max() > d_max:
            raise ValueError(f"n_valid must lie in [2, {d_max}], got [{n_valid_np.min()}, {n_valid_np.max()}]")

        # ``frozen=True`` forbids plain assignment.
        object.__setattr__(self, "_energies_np", energies)
        object.__setattr__(self, "_n_valid_np", n_valid_np)
        object.__setattr__(self, "_energy_index", {round(float(e), 6): i for i, e in enumerate(energies)})

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        allow_energy_interpolation: bool = False,
    ) -> "IonKernelTable":
        """Load a kernel table from an ``.npz`` written by the commissioning converter.

        Args:
            path: Path to the ``.npz`` archive.
            device: Device to hold the curves on (default: CPU).
            dtype: Floating dtype of the curves. The archive stores float64; pass
                ``torch.float64`` to keep full precision.
            allow_energy_interpolation: Default for the accessors' argument of the
                same name. Leave ``False`` unless you deliberately want untabulated
                energies to be interpolated.

        Returns:
            The loaded :class:`IonKernelTable`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the archive is missing arrays or has an unknown format version.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Ion kernel table not found: {path}")
        if not dtype.is_floating_point:
            raise ValueError(f"dtype must be a floating point type, got {dtype}")
        device = torch.device("cpu") if device is None else torch.device(device)

        with np.load(path) as archive:
            missing = [key for key in _REQUIRED_KEYS if key not in archive]
            if missing:
                raise ValueError(f"{path} is not an ion kernel table; missing arrays: {missing}")
            version = int(archive["format_version"]) if "format_version" in archive else 0
            if version != FORMAT_VERSION:
                raise ValueError(
                    f"{path} has kernel-table format version {version}, expected {FORMAT_VERSION}; "
                    "re-run commissioning/conversion/convert_proton_mat_to_npz.py"
                )
            arrays = {key: archive[key] for key in _REQUIRED_KEYS}

        def curve(key: str) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(arrays[key])).to(device=device, dtype=dtype)

        return cls(
            energy_mev=torch.from_numpy(np.ascontiguousarray(arrays["energy_mev"])).to(
                device=device, dtype=torch.float64
            ),
            n_valid=torch.from_numpy(np.ascontiguousarray(arrays["n_valid"])).to(device=device, dtype=torch.int64),
            depth_mm=curve("depth_mm"),
            idd=curve("idd"),
            sigma_mm=curve("sigma"),
            sigma1_mm=curve("sigma1"),
            sigma2_mm=curve("sigma2"),
            weight=curve("weight"),
            offset_mm=curve("offset_mm"),
            focus_dist_mm=curve("focus_dist_mm"),
            focus_sigma_mm=curve("focus_sigma_mm"),
            sad_mm=float(arrays["sad_mm"]),
            bams_to_iso_dist_mm=float(arrays["bams_to_iso_mm"]),
            allow_energy_interpolation=allow_energy_interpolation,
            source_path=path,
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "IonKernelTable":
        """Return a copy of the table on another device and/or floating dtype.

        Args:
            device: Target device (unchanged if ``None``).
            dtype: Target floating dtype of the curves (unchanged if ``None``).

        Returns:
            A new :class:`IonKernelTable`; the original is untouched.
        """
        device = self.device if device is None else torch.device(device)
        dtype = self.dtype if dtype is None else dtype
        if not dtype.is_floating_point:
            raise ValueError(f"dtype must be a floating point type, got {dtype}")
        moved = {
            name: getattr(self, name).to(device=device, dtype=dtype)
            for name in (
                "depth_mm",
                "idd",
                "sigma_mm",
                "sigma1_mm",
                "sigma2_mm",
                "weight",
                "offset_mm",
                "focus_dist_mm",
                "focus_sigma_mm",
            )
        }
        return replace(
            self,
            energy_mev=self.energy_mev.to(device=device),
            n_valid=self.n_valid.to(device=device),
            **moved,
        )

    # ------------------------------------------------------------- properties

    @property
    def device(self) -> torch.device:
        """Device the curve tensors live on."""
        return self.depth_mm.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating dtype of the curve tensors."""
        return self.depth_mm.dtype

    @property
    def num_energies(self) -> int:
        """Number of tabulated energies ``E``."""
        return int(self.energy_mev.shape[0])

    @property
    def available_energies(self) -> list[float]:
        """The tabulated energies in MeV, ascending."""
        return [float(e) for e in self._energies_np]

    @property
    def has_double_gauss(self) -> bool:
        """Always ``True``: the table type carries every curve, or conversion fails."""
        return True

    @property
    def has_initial_focus(self) -> bool:
        """Always ``True``: the table type carries the focus curve, or conversion fails."""
        return True

    # ----------------------------------------------------------- row indexing

    def row_index(self, energy_mev: float | torch.Tensor) -> int:
        """Return the table row of ``energy_mev``. O(1) on a hit, O(log E) otherwise.

        Args:
            energy_mev: The energy to look up (MeV).

        Returns:
            The row index.

        Raises:
            EnergyNotInTableError: If the energy is not tabulated. The message names
                the two neighbouring tabulated energies.
        """
        energy = float(energy_mev)
        index = self._energy_index.get(round(energy, 6))
        if index is not None:
            return index

        energies = self._energies_np
        position = int(np.searchsorted(energies, energy))
        for candidate in (position - 1, position):
            if 0 <= candidate < energies.size and bool(np.isclose(energy, energies[candidate])):
                return candidate

        below = f"{energies[position - 1]:.6f}" if position > 0 else "none (below the table)"
        above = f"{energies[position]:.6f}" if position < energies.size else "none (above the table)"
        raise EnergyNotInTableError(
            f"Energy {energy:.6f} MeV is not in the kernel table"
            + (f" loaded from {self.source_path}" if self.source_path is not None else "")
            + f"; nearest tabulated energies are {below} MeV and {above} MeV. "
            "Pass allow_energy_interpolation=True to interpolate between the neighbouring rows."
        )

    def n_samples(self, energy_mev: float | torch.Tensor) -> int:
        """Number of tabulated depth samples for the row of ``energy_mev``."""
        return int(self._n_valid_np[self.row_index(energy_mev)])

    def depth_max_mm(self, energy_mev: float | torch.Tensor) -> float:
        """Deepest tabulated depth (mm) for the row of ``energy_mev``."""
        index = self.row_index(energy_mev)
        return float(self.depth_mm[index, self._n_valid_np[index] - 1])

    # -------------------------------------------------------------- row views

    def _row(self, index: int, name: str) -> torch.Tensor:
        """Return the valid part of one stored curve row (a view, no copy)."""
        return getattr(self, name)[index, : self._n_valid_np[index]]

    def row_curves(self, energy_mev: float | torch.Tensor) -> dict[str, torch.Tensor]:
        """Return every tabulated curve of one row, trimmed to its valid samples.

        Args:
            energy_mev: A tabulated energy (MeV).

        Returns:
            ``{"depth_mm", "idd", "sigma_mm", "sigma1_mm", "sigma2_mm", "weight"}``
            -- the attribute names of the stored curves -- as 1-D views of length
            ``n_valid``.
        """
        index = self.row_index(energy_mev)
        return {
            name: self._row(index, name)
            for name in ("depth_mm", "idd", "sigma_mm", "sigma1_mm", "sigma2_mm", "weight")
        }

    # -------------------------------------------------------------- internals

    def _resolve_energy(
        self,
        energy_mev: float | torch.Tensor,
        energy_value_hint: float | None,
        allow_energy_interpolation: bool | None,
    ) -> tuple[torch.Tensor, float, int, int]:
        """Resolve an energy argument to (tensor, scalar value, lo row, hi row).

        ``lo == hi`` means the energy is tabulated. ``lo != hi`` only happens when
        interpolation is explicitly allowed.
        """
        energy = torch.as_tensor(energy_mev, device=self.device, dtype=self.dtype)
        if energy_value_hint is None:
            energy_value_hint = float(energy.detach().cpu())
        if allow_energy_interpolation is None:
            allow_energy_interpolation = self.allow_energy_interpolation

        try:
            index = self.row_index(energy_value_hint)
        except EnergyNotInTableError:
            if not allow_energy_interpolation:
                raise
            energies = self._energies_np
            if energy_value_hint < energies[0] or energy_value_hint > energies[-1]:
                raise EnergyNotInTableError(
                    f"Energy {energy_value_hint:.6f} MeV is outside the tabulated range "
                    f"[{energies[0]:.6f}, {energies[-1]:.6f}] MeV; interpolation cannot extrapolate."
                ) from None
            hi = int(np.searchsorted(energies, energy_value_hint))
            return energy, energy_value_hint, hi - 1, hi
        return energy, energy_value_hint, index, index

    def _energy_fraction(self, energy: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        """Linear position of ``energy`` between rows ``lo`` and ``hi``."""
        lo_energy = self.energy_mev[lo].to(self.dtype)
        hi_energy = self.energy_mev[hi].to(self.dtype)
        return (energy - lo_energy) / (hi_energy - lo_energy).clamp_min(torch.finfo(self.dtype).eps)

    def _common_depth_grid(self, lo_depth: torch.Tensor, hi_depth: torch.Tensor) -> torch.Tensor:
        """Uniform grid spanning both rows at their finest tabulated spacing."""
        max_depth = float(max(lo_depth[-1].item(), hi_depth[-1].item()))
        diffs_lo = (lo_depth[1:] - lo_depth[:-1]).abs()
        diffs_hi = (hi_depth[1:] - hi_depth[:-1]).abs()
        pos = torch.cat([diffs_lo[diffs_lo > 0.0], diffs_hi[diffs_hi > 0.0]])
        step = float(pos.min().item()) if pos.numel() > 0 else 1.0
        n = int(np.ceil(max_depth / step)) + 1
        return torch.arange(n, device=lo_depth.device, dtype=lo_depth.dtype) * step

    def _peak_shift_positions(
        self,
        energy: torch.Tensor,
        lo: int,
        hi: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared setup of the range-shift energy interpolation.

        Returns the common depth grid, the low row's positions rescaled so that its
        Bragg peak lands on the interpolated peak depth, the energy fraction and the
        interpolated / low peak depths.
        """
        lo_depth = self._row(lo, "depth_mm")
        hi_depth = self._row(hi, "depth_mm")
        lo_idd = self._row(lo, "idd")
        hi_idd = self._row(hi, "idd")

        depth = self._common_depth_grid(lo_depth, hi_depth)
        frac = self._energy_fraction(energy, lo, hi)

        lo_peak_idx = torch.argmax(lo_idd)
        hi_peak_idx = torch.argmax(hi_idd)
        lo_peak_depth = lo_depth[lo_peak_idx]
        hi_peak_depth = hi_depth[hi_peak_idx]
        interp_peak_depth = lo_peak_depth + frac * (hi_peak_depth - lo_peak_depth)
        safe_peak_depth = interp_peak_depth.clamp_min(torch.finfo(self.dtype).eps)
        scaled_positions = (lo_peak_depth / safe_peak_depth) * depth
        return depth, scaled_positions, frac, interp_peak_depth, lo_peak_depth

    def _beyond_range(
        self,
        values: torch.Tensor,
        depth: torch.Tensor,
        depth_max: torch.Tensor,
        mode: BeyondRange,
        what: str,
        energy_value: float,
    ) -> torch.Tensor:
        """Apply the out-of-range policy to freshly interpolated ``values``."""
        if mode == "edge":
            return values
        past = depth > depth_max
        if mode == "zero":
            return torch.where(past, torch.zeros_like(values), values)
        if mode == "raise":
            if bool(past.any()):
                worst = float(depth[past].max())
                raise DepthOutOfRangeError(
                    f"{what} requested at depth {worst:.3f} mm, past the tabulated range "
                    f"{float(depth_max):.3f} mm of the {energy_value:.6f} MeV row. "
                    "Clamp the depth (see depth_max_mm) or pass beyond_range='zero'/'edge' "
                    "if a continuation past the range is really what you want."
                )
            return values
        raise ValueError(f"Unknown beyond_range mode {mode!r}; expected 'raise', 'zero' or 'edge'")

    # -------------------------------------------------------------- accessors

    def edep_curve(
        self,
        energy_mev: float | torch.Tensor,
        *,
        energy_value_hint: float | None = None,
        allow_energy_interpolation: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(depth, idd)`` curve of one energy.

        Args:
            energy_mev: Energy in MeV, as a float or a (possibly differentiable) scalar tensor.
            energy_value_hint: The scalar value of ``energy_mev``, to skip a
                device synchronisation when a CUDA tensor is passed.
            allow_energy_interpolation: Override the table default. When enabled and
                the energy is untabulated, the neighbouring rows are combined with the
                range-shift/peak-shift interpolation.

        Returns:
            ``(depth_mm, idd)``, both 1-D and of equal length.
        """
        energy, value, lo, hi = self._resolve_energy(energy_mev, energy_value_hint, allow_energy_interpolation)
        if lo == hi:
            return self._row(lo, "depth_mm"), self._row(lo, "idd")

        depth, scaled_positions, frac, interp_peak_depth, _ = self._peak_shift_positions(energy, lo, hi)
        lo_depth = self._row(lo, "depth_mm")
        lo_idd = self._row(lo, "idd")
        hi_idd = self._row(hi, "idd")
        lo_peak = lo_idd[torch.argmax(lo_idd)].clamp_min(torch.finfo(self.dtype).eps)
        hi_peak = hi_idd[torch.argmax(hi_idd)]
        interp_peak = lo_peak + frac * (hi_peak - lo_peak)
        scaled = _interp1d(lo_depth, lo_idd, scaled_positions) * (interp_peak / lo_peak)
        valid = (interp_peak_depth > 0.0) & (lo_peak > 0.0)
        scaled = torch.where(valid, scaled, torch.zeros_like(scaled))
        return depth, scaled.clamp_min(0.0)

    def edep(
        self,
        energy_mev: float | torch.Tensor,
        depth_mm: torch.Tensor | float,
        *,
        beyond_range: BeyondRange = "zero",
        energy_value_hint: float | None = None,
        allow_energy_interpolation: bool | None = None,
    ) -> torch.Tensor:
        """Integrated depth dose at arbitrary depths.

        Args:
            energy_mev: Energy in MeV.
            depth_mm: Water-equivalent depths, any shape (the engine passes ``(S, D)``).
            beyond_range: Policy past the tabulated range; ``"zero"`` by default,
                which is the physical answer -- there is no dose past the range.
            energy_value_hint: See :meth:`edep_curve`.
            allow_energy_interpolation: See :meth:`edep_curve`.

        Returns:
            IDD values shaped like ``depth_mm``.
        """
        energy, value, _, _ = self._resolve_energy(energy_mev, energy_value_hint, allow_energy_interpolation)
        depth = torch.as_tensor(depth_mm, device=self.device, dtype=self.dtype)
        depth_curve, edep_curve = self.edep_curve(
            energy,
            energy_value_hint=value,
            allow_energy_interpolation=allow_energy_interpolation,
        )
        values = _interp1d(depth_curve, edep_curve, depth)
        return self._beyond_range(values, depth, depth_curve[-1], beyond_range, "IDD", value)

    def sigma_curve(
        self,
        energy_mev: float | torch.Tensor,
        *,
        energy_value_hint: float | None = None,
        allow_energy_interpolation: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(depth, sigma)`` single-Gaussian transport curve of one energy."""
        energy, value, lo, hi = self._resolve_energy(energy_mev, energy_value_hint, allow_energy_interpolation)
        if lo == hi:
            return self._row(lo, "depth_mm"), self._row(lo, "sigma_mm")

        depth, scaled_positions, frac, _, _ = self._peak_shift_positions(energy, lo, hi)
        lo_value = _interp1d(self._row(lo, "depth_mm"), self._row(lo, "sigma_mm"), scaled_positions)
        hi_value = _interp1d(self._row(hi, "depth_mm"), self._row(hi, "sigma_mm"), depth)
        return depth, (lo_value + frac * (hi_value - lo_value)).clamp_min(0.0)

    def sigma(
        self,
        energy_mev: float | torch.Tensor,
        depth_mm: torch.Tensor | float,
        *,
        beyond_range: BeyondRange = "raise",
        energy_value_hint: float | None = None,
        allow_energy_interpolation: bool | None = None,
    ) -> torch.Tensor:
        """Single-Gaussian transport sigma (mm) at arbitrary depths.

        Args:
            energy_mev: Energy in MeV.
            depth_mm: Water-equivalent depths, any shape.
            beyond_range: Policy past the tabulated range. ``"raise"`` by default:
                a sigma held constant hundreds of mm past the range is not an answer.
            energy_value_hint: See :meth:`edep_curve`.
            allow_energy_interpolation: See :meth:`edep_curve`.

        Returns:
            Sigma values shaped like ``depth_mm``.
        """
        energy, value, _, _ = self._resolve_energy(energy_mev, energy_value_hint, allow_energy_interpolation)
        depth = torch.as_tensor(depth_mm, device=self.device, dtype=self.dtype)
        depth_curve, sigma_curve = self.sigma_curve(
            energy,
            energy_value_hint=value,
            allow_energy_interpolation=allow_energy_interpolation,
        )
        values = _interp1d(depth_curve, sigma_curve, depth).clamp_min(0.0)
        return self._beyond_range(values, depth, depth_curve[-1], beyond_range, "sigma", value)

    def double_gauss_curves(
        self,
        energy_mev: float | torch.Tensor,
        *,
        energy_value_hint: float | None = None,
        allow_energy_interpolation: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the ``(depth, sigma1, sigma2, weight)`` double-Gaussian curves."""
        energy, value, lo, hi = self._resolve_energy(energy_mev, energy_value_hint, allow_energy_interpolation)
        if lo == hi:
            return (
                self._row(lo, "depth_mm"),
                self._row(lo, "sigma1_mm"),
                self._row(lo, "sigma2_mm"),
                self._row(lo, "weight"),
            )

        depth, scaled_positions, frac, _, _ = self._peak_shift_positions(energy, lo, hi)
        lo_depth = self._row(lo, "depth_mm")
        hi_depth = self._row(hi, "depth_mm")

        def blend(name: str) -> torch.Tensor:
            lo_value = _interp1d(lo_depth, self._row(lo, name), scaled_positions)
            hi_value = _interp1d(hi_depth, self._row(hi, name), depth)
            return lo_value + frac * (hi_value - lo_value)

        return (
            depth,
            blend("sigma1_mm").clamp_min(0.0),
            blend("sigma2_mm").clamp_min(0.0),
            blend("weight").clamp(0.0, 1.0),
        )

    def double_gauss(
        self,
        energy_mev: float | torch.Tensor,
        depth_mm: torch.Tensor | float,
        *,
        beyond_range: BeyondRange = "raise",
        energy_value_hint: float | None = None,
        allow_energy_interpolation: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Double-Gaussian lateral parameters at arbitrary depths.

        Args:
            energy_mev: Energy in MeV.
            depth_mm: Water-equivalent depths, any shape (the engine passes ``(S, D)``).
            beyond_range: Policy past the tabulated range; ``"raise"`` by default.
            energy_value_hint: See :meth:`edep_curve`.
            allow_energy_interpolation: See :meth:`edep_curve`.

        Returns:
            ``(sigma1, sigma2, weight)``, each shaped like ``depth_mm``.
        """
        energy, value, _, _ = self._resolve_energy(energy_mev, energy_value_hint, allow_energy_interpolation)
        depth = torch.as_tensor(depth_mm, device=self.device, dtype=self.dtype)
        depth_curve, s1_curve, s2_curve, w_curve = self.double_gauss_curves(
            energy,
            energy_value_hint=value,
            allow_energy_interpolation=allow_energy_interpolation,
        )
        depth_max = depth_curve[-1]
        sigma1 = _interp1d(depth_curve, s1_curve, depth).clamp_min(0.0)
        sigma2 = _interp1d(depth_curve, s2_curve, depth).clamp_min(0.0)
        weight = _interp1d(depth_curve, w_curve, depth).clamp(0.0, 1.0)
        sigma1 = self._beyond_range(sigma1, depth, depth_max, beyond_range, "sigma1", value)
        sigma2 = self._beyond_range(sigma2, depth, depth_max, beyond_range, "sigma2", value)
        weight = self._beyond_range(weight, depth, depth_max, beyond_range, "weight", value)
        return sigma1, sigma2, weight

    def kernel_offset(
        self,
        energy_mev: float | torch.Tensor,
        *,
        energy_value_hint: float | None = None,
        allow_energy_interpolation: bool | None = None,
    ) -> torch.Tensor:
        """Kernel depth offset (mm) of one energy, as a 0-d tensor.

        Called once per beamlet, so this is the accessor that used to dominate the
        per-beamlet cost (a linear scan over 114 energies, ~500 us at high energy).

        Args:
            energy_mev: Energy in MeV.
            energy_value_hint: See :meth:`edep_curve`.
            allow_energy_interpolation: See :meth:`edep_curve`.

        Returns:
            The offset as a 0-d tensor on the table's device/dtype.
        """
        energy, _, lo, hi = self._resolve_energy(energy_mev, energy_value_hint, allow_energy_interpolation)
        if lo == hi:
            return self.offset_mm[lo]
        frac = self._energy_fraction(energy, lo, hi)
        return self.offset_mm[lo] + frac * (self.offset_mm[hi] - self.offset_mm[lo])

    def initial_sigma(
        self,
        energy_mev: float | torch.Tensor,
        source_to_surface_mm: torch.Tensor | float,
        *,
        energy_value_hint: float | None = None,
        allow_energy_interpolation: bool | None = None,
    ) -> torch.Tensor:
        """Initial spot sigma (mm) from the ``initFocus`` table.

        The focus curve is interpolated in source-to-surface distance and held at
        its edge values outside the tabulated distances, matching matRad.

        Args:
            energy_mev: Energy in MeV.
            source_to_surface_mm: Source-to-surface distance(s), any shape.
            energy_value_hint: See :meth:`edep_curve`.
            allow_energy_interpolation: See :meth:`edep_curve`.

        Returns:
            Sigma values shaped like ``source_to_surface_mm``.
        """
        energy, _, lo, hi = self._resolve_energy(energy_mev, energy_value_hint, allow_energy_interpolation)
        distance = torch.as_tensor(source_to_surface_mm, device=self.device, dtype=self.dtype)
        lo_value = _interp1d(self.focus_dist_mm[lo], self.focus_sigma_mm[lo], distance)
        if lo == hi:
            return lo_value.clamp_min(0.0)
        hi_value = _interp1d(self.focus_dist_mm[hi], self.focus_sigma_mm[hi], distance)
        frac = self._energy_fraction(energy, lo, hi)
        return (lo_value + frac * (hi_value - lo_value)).clamp_min(0.0)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"IonKernelTable(E={self.num_energies}, Dmax={self.depth_mm.shape[1]}, "
            f"energies=[{self._energies_np[0]:.3f}..{self._energies_np[-1]:.3f}] MeV, "
            f"device={self.device}, dtype={self.dtype})"
        )
