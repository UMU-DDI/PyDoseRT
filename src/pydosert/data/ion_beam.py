"""Flat beamlet batch for the ion dose engine.

An ion engine call computes dose for a *batch of beamlets*: independent pencil
beams, each with its own gantry angle, energy, spot position, spot size and
weight. :class:`IonBeamletBatch` is exactly that -- one flat leading axis ``G``
and nothing else. A spot-scanning plan is expanded into such a batch by the
caller; there is no layer/spot indirection here.

Conventions
-----------
* ``iso_center_mm`` is ``(h_mm, d_mm, w_mm)`` from the dose-grid origin, matching
  the ``(H, D, W)`` grid axis order used by :mod:`pydosert.geometry.bev`.
* ``position_mm[:, 0]`` displaces the beamlet along the patient **W** axis and
  ``position_mm[:, 1]`` along the patient **H** axis (at gantry 0).
* ``sigma_mm[:, 0]`` is the sigma of the same axis as ``position_mm[:, 0]``
  (``sigma_x``, the W/x direction) and ``sigma_mm[:, 1]`` is ``sigma_y`` (H/y).
* Angles are radians, distances mm, energies MeV.

Design rules
------------
* **Validate, never coerce.** A wrong shape, a mixed device, a NaN or a negative
  weight raises; nothing is broadcast, padded or cast on the caller's behalf.
* **No machine geometry defaults.** ``sad_mm`` must be given. The BEV field size
  is not here at all -- it is an engine setting.
* **Gradients are opt-in.** Constructing a batch never calls
  ``requires_grad_``; use :meth:`with_requires_grad`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch

__all__ = ["IonBeamletBatch"]

#: A per-beamlet scalar quantity as the caller may spell it.
_Scalars = torch.Tensor | Sequence[float]
#: A per-beamlet pair (or triple) as the caller may spell it.
_Vectors = torch.Tensor | Sequence[Sequence[float]]


def _as_tensor(
    value: _Scalars | _Vectors,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Materialise a constructor argument as a float tensor on ``device``/``dtype``."""
    return torch.as_tensor(value, dtype=dtype).to(device=device)


@dataclass(frozen=True, eq=False)
class IonBeamletBatch:
    """A flat batch of ``G`` independent ion beamlets.

    Every field is a tensor whose leading axis is the beamlet axis. All float
    fields share one device and one floating-point dtype; that pair is the
    batch's :attr:`device` / :attr:`dtype`.

    Attributes:
        gantry_angle_rad: ``(G,)`` gantry angle in **radians** (the constructor
            takes degrees).
        iso_center_mm: ``(G, 3)`` isocentre in mm from the dose-grid origin,
            ordered ``(height, depth, width)`` = ``(h, d, w)``. Per-beamlet, so
            a batch may mix isocentres.
        sad_mm: ``(G,)`` source-to-axis distance in mm. Required, no default.
        position_mm: ``(G, 2)`` beamlet position in the beam frame, mm from the
            central axis, ordered ``(x, y)``: **x displaces the beamlet along
            the patient W axis, y along the patient H axis**. Note this is
            *not* the ``(h, d, w)`` order of :attr:`iso_center_mm`.
        energy_mev: ``(G,)`` beamlet kinetic energy in MeV, strictly positive.
        sigma_mm: ``(G, 2)`` initial spot sigma in mm, ordered
            ``(sigma_x, sigma_y)`` -- the same axes as :attr:`position_mm`.
            Non-negative; a round spot is given explicitly as two equal values.
        weight: ``(G,)`` beamlet weight (particles / MU). Non-negative. Dose is
            exactly linear in it.
    """

    gantry_angle_rad: torch.Tensor
    iso_center_mm: torch.Tensor
    sad_mm: torch.Tensor
    position_mm: torch.Tensor
    energy_mev: torch.Tensor
    sigma_mm: torch.Tensor
    weight: torch.Tensor

    #: ``(field name, trailing shape)`` for every tensor field, in declaration order.
    _FIELD_SHAPES = (
        ("gantry_angle_rad", ()),
        ("iso_center_mm", (3,)),
        ("sad_mm", ()),
        ("position_mm", (2,)),
        ("energy_mev", ()),
        ("sigma_mm", (2,)),
        ("weight", ()),
    )

    # ------------------------------------------------------------------ setup

    def __post_init__(self) -> None:
        for name, _ in self._FIELD_SHAPES:
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
            if not value.is_floating_point():
                raise ValueError(
                    f"{name} must have a floating-point dtype, got {value.dtype}. "
                    "Every IonBeamletBatch field is a physical quantity, not an index."
                )

        num_beamlets = int(self.gantry_angle_rad.shape[0]) if self.gantry_angle_rad.ndim >= 1 else -1
        if self.gantry_angle_rad.ndim != 1:
            raise ValueError(
                f"gantry_angle_rad must be [G], got {tuple(self.gantry_angle_rad.shape)}"
            )
        if num_beamlets == 0:
            raise ValueError("IonBeamletBatch requires at least one beamlet")

        for name, trailing in self._FIELD_SHAPES:
            value = getattr(self, name)
            expected = (num_beamlets,) + trailing
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must be {list(expected)}, got {list(value.shape)}")

        devices = {getattr(self, name).device for name, _ in self._FIELD_SHAPES}
        if len(devices) != 1:
            raise ValueError(f"IonBeamletBatch tensors must share one device, got {devices}")
        dtypes = {getattr(self, name).dtype for name, _ in self._FIELD_SHAPES}
        if len(dtypes) != 1:
            raise ValueError(f"IonBeamletBatch tensors must share one dtype, got {dtypes}")

        for name, _ in self._FIELD_SHAPES:
            value = getattr(self, name)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains non-finite values (NaN or inf)")

        if bool((self.weight < 0).any()):
            raise ValueError("weight must be non-negative")
        if bool((self.sigma_mm < 0).any()):
            raise ValueError("sigma_mm must be non-negative")
        if bool((self.energy_mev <= 0).any()):
            raise ValueError("energy_mev must be strictly positive")
        if bool((self.sad_mm <= 0).any()):
            raise ValueError("sad_mm must be strictly positive")

    # ------------------------------------------------------------ constructor

    @classmethod
    def create(
        cls,
        *,
        gantry_angle_deg: _Scalars,
        position_mm: _Vectors,
        energy_mev: _Scalars,
        sigma_mm: _Scalars | _Vectors,
        weight: _Scalars,
        iso_center_mm: _Scalars | _Vectors,
        sad_mm: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
        sigma_requires_grad: bool = False,
    ) -> "IonBeamletBatch":
        """Build a batch from per-beamlet sequences, in the units people quote.

        The only conversion is degrees to radians. Scalars broadcast only where
        the quantity is shared by construction: one ``(3,)`` isocentre and one
        ``sad_mm``.

        Args:
            gantry_angle_deg: ``(G,)`` gantry angles in **degrees**.
            position_mm: ``(G, 2)`` beamlet positions, ``(x, y)``; see
                :attr:`position_mm`.
            energy_mev: ``(G,)`` energies in MeV.
            sigma_mm: ``(G, 2)`` spot sigmas ``(sigma_x, sigma_y)``, or ``(G,)``
                for an isotropic spot.
            weight: ``(G,)`` beamlet weights.
            iso_center_mm: ``(3,)`` shared or ``(G, 3)`` per-beamlet isocentre,
                ordered ``(h, d, w)`` mm.
            sad_mm: Source-to-axis distance in mm. No default.
            device: Device for every tensor.
            dtype: Floating-point dtype for every tensor.
            requires_grad: Attach gradients to ``position_mm``, ``weight`` and
                ``energy_mev``. Off by default.
            sigma_requires_grad: Attach gradients to ``sigma_mm`` as well.

        Returns:
            The validated batch.

        Raises:
            ValueError: If ``dtype`` is not floating point, or any argument has
                the wrong shape (validated in :meth:`__post_init__`).
        """
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError(f"dtype must be a floating-point torch.dtype, got {dtype!r}")
        device = torch.device(device)

        # Converted in float64 and cast afterwards; multiplying in float32
        # instead can differ by an ULP.
        angles_deg = torch.as_tensor(gantry_angle_deg, dtype=torch.float64)
        if angles_deg.ndim != 1:
            raise ValueError(f"gantry_angle_deg must be [G], got {list(angles_deg.shape)}")
        num_beamlets = int(angles_deg.shape[0])
        angles_rad = (angles_deg * (math.pi / 180.0)).to(device=device, dtype=dtype)

        positions = _as_tensor(position_mm, device, dtype)
        energies = _as_tensor(energy_mev, device, dtype)
        weights = _as_tensor(weight, device, dtype)

        sigmas = _as_tensor(sigma_mm, device, dtype)
        if sigmas.shape == (num_beamlets,):
            sigmas = torch.stack((sigmas, sigmas), dim=-1)

        iso = _as_tensor(iso_center_mm, device, dtype)
        if iso.shape == (3,):
            iso = iso.unsqueeze(0).expand(num_beamlets, 3).clone()

        sad_value = float(sad_mm)
        sads = torch.full((num_beamlets,), sad_value, device=device, dtype=dtype)

        batch = cls(
            gantry_angle_rad=angles_rad.contiguous(),
            iso_center_mm=iso.contiguous(),
            sad_mm=sads,
            position_mm=positions.contiguous(),
            energy_mev=energies.contiguous(),
            sigma_mm=sigmas.contiguous(),
            weight=weights.contiguous(),
        )
        if requires_grad or sigma_requires_grad:
            batch = batch.with_requires_grad(
                position=requires_grad,
                weight=requires_grad,
                energy=requires_grad,
                sigma=sigma_requires_grad,
            )
        return batch

    # ------------------------------------------------------------ ergonomics

    def __len__(self) -> int:
        """Number of beamlets ``G``."""
        return int(self.gantry_angle_rad.shape[0])

    @property
    def num_beamlets(self) -> int:
        """Number of beamlets ``G``."""
        return len(self)

    @property
    def device(self) -> torch.device:
        """The single device every field lives on."""
        return self.gantry_angle_rad.device

    @property
    def dtype(self) -> torch.dtype:
        """The single floating-point dtype every field uses."""
        return self.gantry_angle_rad.dtype

    @property
    def gantry_angle_deg(self) -> torch.Tensor:
        """``(G,)`` gantry angles converted to degrees."""
        return self.gantry_angle_rad * (180.0 / math.pi)

    @property
    def sigma_x_mm(self) -> torch.Tensor:
        """``(G,)`` spot sigma along the x / patient-W axis."""
        return self.sigma_mm[:, 0]

    @property
    def sigma_y_mm(self) -> torch.Tensor:
        """``(G,)`` spot sigma along the y / patient-H axis."""
        return self.sigma_mm[:, 1]

    @property
    def requires_grad(self) -> bool:
        """True when any field is part of an autograd graph."""
        return any(getattr(self, name).requires_grad for name, _ in self._FIELD_SHAPES)

    def _map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "IonBeamletBatch":
        return replace(self, **{name: fn(getattr(self, name)) for name, _ in self._FIELD_SHAPES})

    def to(self, target: torch.device | str | torch.dtype) -> "IonBeamletBatch":
        """Move or cast every field.

        Args:
            target: A device, a device string, or a floating-point dtype.

        Returns:
            A new batch. The original is untouched.

        Raises:
            ValueError: If ``target`` is a non-floating dtype.
        """
        if isinstance(target, torch.dtype) and not target.is_floating_point:
            raise ValueError(f"cannot cast an IonBeamletBatch to the non-float dtype {target}")
        return self._map(lambda t: t.to(target))

    def detach(self) -> "IonBeamletBatch":
        """Return the same values, detached from any autograd graph."""
        return self._map(lambda t: t.detach())

    def clone(self) -> "IonBeamletBatch":
        """Return a deep copy of every tensor."""
        return self._map(lambda t: t.clone())

    def with_requires_grad(
        self,
        *,
        position: bool = False,
        weight: bool = False,
        energy: bool = False,
        sigma: bool = False,
        iso_center: bool = False,
        gantry_angle: bool = False,
        sad: bool = False,
    ) -> "IonBeamletBatch":
        """Opt in to gradients, per field.

        Each selected field becomes a fresh autograd leaf (detached clone with
        ``requires_grad_(True)``); the rest pass through unchanged. Returns a
        new batch, leaving the original's gradient state alone.
        """
        selected = {
            "position_mm": position,
            "weight": weight,
            "energy_mev": energy,
            "sigma_mm": sigma,
            "iso_center_mm": iso_center,
            "gantry_angle_rad": gantry_angle,
            "sad_mm": sad,
        }
        updates = {
            name: getattr(self, name).detach().clone().requires_grad_(True)
            for name, wanted in selected.items()
            if wanted
        }
        return replace(self, **updates) if updates else self

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        energies = self.energy_mev.detach().to(device="cpu", dtype=torch.float64)
        return (
            f"IonBeamletBatch(G={len(self)}, device={self.device}, dtype={self.dtype}, "
            f"energy_mev=[{float(energies.min()):.4g}, {float(energies.max()):.4g}], "
            f"requires_grad={self.requires_grad})"
        )
