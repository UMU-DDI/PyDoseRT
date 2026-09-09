"""Tests for :class:`pydosert.data.ion_beam.IonBeamletBatch`.

Self-contained: no data files, no golden oracle, no engine. Every validation
rule gets its own ``pytest.raises``, because the point of the type is that it
raises instead of coercing.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from pydosert.data.ion_beam import IonBeamletBatch  # noqa: E402

# --------------------------------------------------------------------------- helpers


def make_batch(**overrides) -> IonBeamletBatch:
    """A valid two-beamlet batch; ``overrides`` replace ``create`` kwargs."""
    kwargs = dict(
        gantry_angle_deg=[0.0, 45.0],
        position_mm=[[0.0, 0.0], [6.0, -4.0]],
        energy_mev=[120.0, 199.5],
        sigma_mm=[[4.6, 4.6], [4.0, 4.0]],
        weight=[1.0e7, 5.0e6],
        iso_center_mm=[47.0, 150.0, 47.0],
        sad_mm=1000.0,
    )
    kwargs.update(overrides)
    return IonBeamletBatch.create(**kwargs)


def raw_fields(num_beamlets: int = 2, dtype=torch.float32, device="cpu") -> dict:
    """Field kwargs for the raw dataclass constructor (no ``create`` sugar)."""
    opts = {"dtype": dtype, "device": device}
    return {
        "gantry_angle_rad": torch.zeros(num_beamlets, **opts),
        "iso_center_mm": torch.zeros(num_beamlets, 3, **opts),
        "sad_mm": torch.full((num_beamlets,), 1000.0, **opts),
        "position_mm": torch.zeros(num_beamlets, 2, **opts),
        "energy_mev": torch.full((num_beamlets,), 120.0, **opts),
        "sigma_mm": torch.full((num_beamlets, 2), 4.6, **opts),
        "weight": torch.ones(num_beamlets, **opts),
    }


ALL_FIELDS = (
    "gantry_angle_rad",
    "iso_center_mm",
    "sad_mm",
    "position_mm",
    "energy_mev",
    "sigma_mm",
    "weight",
)


# --------------------------------------------------------------------------- construction


def test_create_shapes_and_values():
    batch = make_batch()
    assert len(batch) == 2
    assert batch.num_beamlets == 2
    assert batch.gantry_angle_rad.shape == (2,)
    assert batch.iso_center_mm.shape == (2, 3)
    assert batch.sad_mm.shape == (2,)
    assert batch.position_mm.shape == (2, 2)
    assert batch.energy_mev.shape == (2,)
    assert batch.sigma_mm.shape == (2, 2)
    assert batch.weight.shape == (2,)
    assert batch.device == torch.device("cpu")
    assert batch.dtype == torch.float32


def test_create_converts_degrees_to_radians_like_math_radians():
    angles = [0.0, 45.0, 137.0, 270.0]
    batch = make_batch(
        gantry_angle_deg=angles,
        position_mm=[[0.0, 0.0]] * 4,
        energy_mev=[120.0] * 4,
        sigma_mm=[4.6] * 4,
        weight=[1.0] * 4,
    )
    expected = torch.tensor([math.radians(a) for a in angles], dtype=torch.float64).to(torch.float32)
    assert torch.equal(batch.gantry_angle_rad, expected)
    assert torch.allclose(batch.gantry_angle_deg, torch.tensor(angles), atol=1e-4)


def test_create_broadcasts_shared_iso_center_per_beamlet():
    batch = make_batch(iso_center_mm=[47.0, 150.0, 47.0])
    assert torch.equal(batch.iso_center_mm[0], batch.iso_center_mm[1])
    assert torch.equal(batch.iso_center_mm[0], torch.tensor([47.0, 150.0, 47.0]))


def test_create_accepts_per_beamlet_iso_center():
    iso = [[47.0, 150.0, 47.0], [31.0, 118.0, 63.0]]
    batch = make_batch(iso_center_mm=iso)
    assert torch.equal(batch.iso_center_mm, torch.tensor(iso))


def test_create_duplicates_isotropic_sigma_into_both_components():
    batch = make_batch(sigma_mm=[4.6, 4.0])
    assert torch.equal(batch.sigma_mm, torch.tensor([[4.6, 4.6], [4.0, 4.0]]))
    assert torch.equal(batch.sigma_x_mm, torch.tensor([4.6, 4.0]))
    assert torch.equal(batch.sigma_y_mm, torch.tensor([4.6, 4.0]))


def test_create_position_axis_order_is_x_then_y():
    """``position_mm[:, 0]`` is the W/x displacement, ``[:, 1]`` the H/y one.

    Pinning the *asymmetric* value here is what makes a silent transpose a test
    failure rather than a coincidence.
    """
    batch = make_batch(position_mm=[[0.0, 0.0], [6.0, -4.0]])
    assert float(batch.position_mm[1, 0]) == 6.0
    assert float(batch.position_mm[1, 1]) == -4.0


def test_create_requires_explicit_sad():
    with pytest.raises(TypeError):
        IonBeamletBatch.create(
            gantry_angle_deg=[0.0],
            position_mm=[[0.0, 0.0]],
            energy_mev=[120.0],
            sigma_mm=[4.6],
            weight=[1.0],
            iso_center_mm=[0.0, 0.0, 0.0],
        )


def test_batch_has_no_field_size():
    """``field_size`` is an engine setting; carrying it here made it disagree."""
    assert not hasattr(make_batch(), "field_size")
    with pytest.raises(TypeError):
        make_batch(field_size=(401, 401))


def test_create_rejects_non_float_dtype():
    with pytest.raises(ValueError, match="floating-point"):
        make_batch(dtype=torch.int64)


def test_create_rejects_scalar_gantry_angle():
    with pytest.raises(ValueError, match=r"gantry_angle_deg must be \[G\]"):
        make_batch(gantry_angle_deg=0.0)


def test_create_float64():
    batch = make_batch(dtype=torch.float64)
    assert batch.dtype == torch.float64
    for name in ALL_FIELDS:
        assert getattr(batch, name).dtype == torch.float64


# --------------------------------------------------------------------------- validation


def test_rejects_non_tensor_field():
    fields = raw_fields()
    fields["weight"] = [1.0, 1.0]
    with pytest.raises(TypeError, match="weight must be a torch.Tensor"):
        IonBeamletBatch(**fields)


@pytest.mark.parametrize("name", ALL_FIELDS)
def test_rejects_integer_dtype_per_field(name):
    fields = raw_fields()
    fields[name] = fields[name].to(torch.int64)
    with pytest.raises(ValueError, match="floating-point"):
        IonBeamletBatch(**fields)


def test_rejects_zero_beamlets():
    with pytest.raises(ValueError, match="at least one beamlet"):
        IonBeamletBatch(**raw_fields(num_beamlets=0))


def test_rejects_non_1d_gantry_angle():
    fields = raw_fields()
    fields["gantry_angle_rad"] = torch.zeros(2, 1)
    with pytest.raises(ValueError, match=r"gantry_angle_rad must be \[G\]"):
        IonBeamletBatch(**fields)


def test_rejects_inconsistent_beamlet_count():
    fields = raw_fields(num_beamlets=2)
    fields["weight"] = torch.ones(3)
    with pytest.raises(ValueError, match=r"weight must be \[2\], got \[3\]"):
        IonBeamletBatch(**fields)


@pytest.mark.parametrize(
    "name, bad_shape",
    [
        ("iso_center_mm", (2, 2)),
        ("position_mm", (2, 3)),
        ("sigma_mm", (2,)),
        ("sad_mm", (2, 1)),
        ("energy_mev", (2, 1)),
    ],
)
def test_rejects_wrong_trailing_shape(name, bad_shape):
    fields = raw_fields()
    fields[name] = torch.full(bad_shape, 1.0)
    with pytest.raises(ValueError, match=f"{name} must be"):
        IonBeamletBatch(**fields)


def test_rejects_mixed_dtypes():
    fields = raw_fields()
    fields["weight"] = fields["weight"].to(torch.float64)
    with pytest.raises(ValueError, match="must share one dtype"):
        IonBeamletBatch(**fields)


def test_rejects_mixed_devices():
    if not torch.cuda.is_available():
        pytest.skip("needs a second device")
    fields = raw_fields()
    fields["weight"] = fields["weight"].cuda()
    with pytest.raises(ValueError, match="must share one device"):
        IonBeamletBatch(**fields)


@pytest.mark.parametrize("name", ALL_FIELDS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_values(name, bad):
    fields = raw_fields()
    tensor = fields[name].clone()
    tensor.view(-1)[0] = bad
    fields[name] = tensor
    with pytest.raises(ValueError, match=f"{name} contains non-finite"):
        IonBeamletBatch(**fields)


def test_rejects_negative_weight():
    fields = raw_fields()
    fields["weight"] = torch.tensor([1.0, -1e-6])
    with pytest.raises(ValueError, match="weight must be non-negative"):
        IonBeamletBatch(**fields)


def test_accepts_zero_weight():
    fields = raw_fields()
    fields["weight"] = torch.tensor([1.0, 0.0])
    assert float(IonBeamletBatch(**fields).weight[1]) == 0.0


def test_rejects_negative_sigma():
    fields = raw_fields()
    fields["sigma_mm"] = torch.tensor([[4.6, 4.6], [4.0, -0.1]])
    with pytest.raises(ValueError, match="sigma_mm must be non-negative"):
        IonBeamletBatch(**fields)


@pytest.mark.parametrize("bad_energy", [0.0, -120.0])
def test_rejects_non_positive_energy(bad_energy):
    fields = raw_fields()
    fields["energy_mev"] = torch.tensor([120.0, bad_energy])
    with pytest.raises(ValueError, match="energy_mev must be strictly positive"):
        IonBeamletBatch(**fields)


@pytest.mark.parametrize("bad_sad", [0.0, -1000.0])
def test_rejects_non_positive_sad(bad_sad):
    fields = raw_fields()
    fields["sad_mm"] = torch.tensor([1000.0, bad_sad])
    with pytest.raises(ValueError, match="sad_mm must be strictly positive"):
        IonBeamletBatch(**fields)


def test_validation_never_coerces():
    """A merely-convertible input is rejected, not fixed up."""
    fields = raw_fields()
    fields["position_mm"] = torch.zeros(2, 2, 1)  # a squeeze away from valid
    with pytest.raises(ValueError):
        IonBeamletBatch(**fields)


# --------------------------------------------------------------------------- ergonomics


def test_to_dtype_moves_every_field():
    batch = make_batch()
    moved = batch.to(torch.float64)
    assert moved.dtype == torch.float64
    for name in ALL_FIELDS:
        assert getattr(moved, name).dtype == torch.float64
    assert batch.dtype == torch.float32, "to() must not mutate the original"
    assert torch.equal(moved.position_mm.float(), batch.position_mm)


def test_to_device_string():
    batch = make_batch()
    moved = batch.to("cpu")
    assert moved.device == torch.device("cpu")
    assert torch.equal(moved.energy_mev, batch.energy_mev)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_to_cuda_and_back():
    batch = make_batch()
    on_gpu = batch.to("cuda")
    assert on_gpu.device.type == "cuda"
    for name in ALL_FIELDS:
        assert getattr(on_gpu, name).device.type == "cuda"
    assert torch.equal(on_gpu.to("cpu").weight, batch.weight)


def test_to_rejects_integer_dtype():
    with pytest.raises(ValueError, match="non-float dtype"):
        make_batch().to(torch.int32)


def test_clone_is_independent():
    batch = make_batch()
    copy = batch.clone()
    assert torch.equal(copy.weight, batch.weight)
    copy.weight[0] = 42.0
    assert float(batch.weight[0]) == 1.0e7


def test_len_matches_beamlet_count():
    assert len(make_batch(
        gantry_angle_deg=[0.0, 45.0, 90.0],
        position_mm=[[0.0, 0.0]] * 3,
        energy_mev=[120.0] * 3,
        sigma_mm=[4.6] * 3,
        weight=[1.0] * 3,
    )) == 3


# --------------------------------------------------------------------------- gradients


def test_gradients_are_off_by_default():
    batch = make_batch()
    assert batch.requires_grad is False
    for name in ALL_FIELDS:
        assert getattr(batch, name).requires_grad is False


def test_create_requires_grad_opt_in():
    batch = make_batch(requires_grad=True)
    assert batch.requires_grad is True
    assert batch.position_mm.requires_grad
    assert batch.weight.requires_grad
    assert batch.energy_mev.requires_grad
    assert not batch.sigma_mm.requires_grad
    assert not batch.iso_center_mm.requires_grad


def test_create_sigma_requires_grad_opt_in():
    batch = make_batch(sigma_requires_grad=True)
    assert batch.sigma_mm.requires_grad
    assert not batch.position_mm.requires_grad


def test_with_requires_grad_is_selective_and_non_mutating():
    batch = make_batch()
    tracked = batch.with_requires_grad(weight=True)
    assert tracked.weight.requires_grad
    assert not tracked.position_mm.requires_grad
    assert not batch.weight.requires_grad, "the original must keep its grad state"


def test_with_requires_grad_no_selection_returns_self():
    batch = make_batch()
    assert batch.with_requires_grad() is batch


def test_gradient_flows_to_the_leaf():
    batch = make_batch().with_requires_grad(weight=True, position=True)
    (batch.weight.sum() * 2.0 + batch.position_mm.square().sum()).backward()
    assert torch.equal(batch.weight.grad, torch.full((2,), 2.0))
    assert torch.equal(batch.position_mm.grad, 2.0 * batch.position_mm.detach())


def test_detach_drops_the_graph_and_keeps_values():
    batch = make_batch().with_requires_grad(weight=True)
    detached = batch.detach()
    assert detached.requires_grad is False
    assert torch.equal(detached.weight, batch.weight.detach())


def test_moving_a_tracked_batch_keeps_it_differentiable():
    batch = make_batch().with_requires_grad(weight=True)
    moved = batch.to(torch.float64)
    moved.weight.sum().backward()
    assert batch.weight.grad is not None
