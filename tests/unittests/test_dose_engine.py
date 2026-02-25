import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import torch
from pydose_rt import DoseEngine
from pydose_rt.data import BeamSequence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dose_engine(
    default_machine_config,
    default_ct_array_shape,
    default_resolution,
    default_beam_sequence,
    default_kernel_size,
    default_device,
    default_dtype,
):
    return DoseEngine(
        machine_config=default_machine_config,
        kernel_size=default_kernel_size,
        dose_grid_spacing=default_resolution,
        dose_grid_shape=default_ct_array_shape,
        beam_template=default_beam_sequence,
        device=default_device,
        dtype=default_dtype,
    )


@pytest.fixture
def default_ct_image(default_ct_array_shape, default_device, default_dtype):
    return torch.zeros((1, *default_ct_array_shape), device=default_device, dtype=default_dtype)


@pytest.fixture
def multi_beam_sequence(
    default_machine_config,
    default_field_size,
    default_iso_center,
    default_sid,
    default_device,
    default_dtype,
):
    return BeamSequence.create(
        gantry_angles_deg=[0.0, 90.0, 180.0, 270.0],
        number_of_leaf_pairs=default_machine_config.number_of_leaf_pairs,
        field_size=default_field_size,
        iso_center=default_iso_center,
        sid=default_sid,
        device=default_device,
        dtype=default_dtype,
    )


@pytest.fixture
def multi_beam_dose_engine(
    default_machine_config,
    default_ct_array_shape,
    default_resolution,
    multi_beam_sequence,
    default_kernel_size,
    default_device,
    default_dtype,
):
    return DoseEngine(
        machine_config=default_machine_config,
        kernel_size=default_kernel_size,
        dose_grid_spacing=default_resolution,
        dose_grid_shape=default_ct_array_shape,
        beam_template=multi_beam_sequence,
        device=default_device,
        dtype=default_dtype,
    )


# ---------------------------------------------------------------------------
# Output shape tests
# ---------------------------------------------------------------------------

def test_forward_fluence_maps_4d_output_shape(
    dose_engine, default_field_size, default_device, default_dtype, default_ct_image
):
    """forward() with 4D [B, G, H, W] fluence_maps produces correct dose shape."""
    B, G = 1, dose_engine.number_of_beams
    H, W = default_field_size
    fluence_maps = torch.ones(B, G, H, W, device=default_device, dtype=default_dtype)
    mus = torch.ones(B, G, device=default_device, dtype=default_dtype)

    dose = dose_engine.forward(
        leaf_positions=None,
        mus=mus,
        jaw_positions=None,
        density_image=default_ct_image,
        fluence_maps=fluence_maps,
    )

    assert dose.shape == (B, *dose_engine.dose_grid_shape)


def test_forward_fluence_maps_3d_output_shape(
    dose_engine, default_field_size, default_device, default_dtype, default_ct_image
):
    """forward() with 3D [B*G, H, W] fluence_maps produces correct dose shape."""
    B, G = 1, dose_engine.number_of_beams
    H, W = default_field_size
    fluence_maps = torch.ones(B * G, H, W, device=default_device, dtype=default_dtype)
    mus = torch.ones(B, G, device=default_device, dtype=default_dtype)

    dose = dose_engine.forward(
        leaf_positions=None,
        mus=mus,
        jaw_positions=None,
        density_image=default_ct_image,
        fluence_maps=fluence_maps,
    )

    assert dose.shape == (B, *dose_engine.dose_grid_shape)


def test_compute_dose_fluence_maps_4d_output_shape(
    dose_engine, default_beam_sequence, default_field_size, default_device, default_dtype, default_ct_image
):
    """compute_dose() with [1, G, H, W] fluence_maps produces correct dose shape."""
    G = len(default_beam_sequence)
    H, W = default_field_size
    fluence_maps = torch.ones(1, G, H, W, device=default_device, dtype=default_dtype)

    dose = dose_engine.compute_dose(
        beam_input=default_beam_sequence,
        density_image=default_ct_image,
        fluence_maps=fluence_maps,
    )

    assert dose.shape == (1, *dose_engine.dose_grid_shape)


def test_compute_dose_fluence_maps_3d_autounsqueeze(
    dose_engine, default_beam_sequence, default_field_size, default_device, default_dtype, default_ct_image
):
    """compute_dose() auto-unsqueezes [G, H, W] fluence_maps to [1, G, H, W]."""
    G = len(default_beam_sequence)
    H, W = default_field_size
    fluence_maps = torch.ones(G, H, W, device=default_device, dtype=default_dtype)

    dose = dose_engine.compute_dose(
        beam_input=default_beam_sequence,
        density_image=default_ct_image,
        fluence_maps=fluence_maps,
    )

    assert dose.shape == (1, *dose_engine.dose_grid_shape)


def test_forward_fluence_maps_multiple_beams_output_shape(
    multi_beam_dose_engine, default_field_size, default_device, default_dtype, default_ct_image
):
    """forward() with multi-beam engine and fluence_maps produces correct dose shape."""
    B = 1
    G = multi_beam_dose_engine.number_of_beams  # 4 beams
    H, W = default_field_size
    fluence_maps = torch.ones(B, G, H, W, device=default_device, dtype=default_dtype)
    mus = torch.ones(B, G, device=default_device, dtype=default_dtype)

    dose = multi_beam_dose_engine.forward(
        leaf_positions=None,
        mus=mus,
        jaw_positions=None,
        density_image=default_ct_image,
        fluence_maps=fluence_maps,
    )

    assert dose.shape == (B, *multi_beam_dose_engine.dose_grid_shape)


# ---------------------------------------------------------------------------
# mus=None tests  (fluence optimisation path — no aperture geometry needed)
# ---------------------------------------------------------------------------

def test_forward_fluence_maps_without_mus_output_shape(
    dose_engine, default_field_size, default_device, default_dtype, default_ct_image
):
    """forward() with fluence_maps and mus=None should run without error and produce correct shape."""
    B, G = 1, dose_engine.number_of_beams
    H, W = default_field_size
    fluence_maps = torch.ones(B, G, H, W, device=default_device, dtype=default_dtype)

    dose = dose_engine.forward(
        leaf_positions=None,
        mus=None,
        jaw_positions=None,
        density_image=default_ct_image,
        fluence_maps=fluence_maps,
    )

    assert dose.shape == (B, *dose_engine.dose_grid_shape)


def test_forward_fluence_maps_mus_scales_dose(
    dose_engine, default_field_size, default_device, default_dtype, default_ct_image
):
    """Providing mus should scale the dose proportionally compared to mus=None."""
    B, G = 1, dose_engine.number_of_beams
    H, W = default_field_size
    fluence_maps = torch.ones(B, G, H, W, device=default_device, dtype=default_dtype)

    dose_no_mus = dose_engine.forward(
        leaf_positions=None,
        mus=None,
        jaw_positions=None,
        density_image=default_ct_image,
        fluence_maps=fluence_maps,
    )

    scale = 3.0
    mus = torch.full((B, G), scale, device=default_device, dtype=default_dtype)
    dose_with_mus = dose_engine.forward(
        leaf_positions=None,
        mus=mus,
        jaw_positions=None,
        density_image=default_ct_image,
        fluence_maps=fluence_maps,
    )

    assert torch.allclose(dose_with_mus, dose_no_mus * scale, atol=1e-5), (
        "dose with mus=scale should equal scale * dose without mus"
    )


def test_forward_fluence_maps_gradient_flow_without_mus(
    dose_engine, default_field_size, default_device, default_dtype
):
    """Gradients should propagate back through fluence_maps even when mus=None."""
    B, G = 1, dose_engine.number_of_beams
    H, W = default_field_size
    fluence_maps = torch.ones(B, G, H, W, device=default_device, dtype=default_dtype, requires_grad=True)
    # Use uniform non-zero density so pencil-beam kernels are non-trivial
    water_ct = torch.ones((B, *dose_engine.dose_grid_shape), device=default_device, dtype=default_dtype)

    dose = dose_engine.forward(
        leaf_positions=None,
        mus=None,
        jaw_positions=None,
        density_image=water_ct,
        fluence_maps=fluence_maps,
    )
    dose.sum().backward()

    assert fluence_maps.grad is not None, "Gradients should reach fluence_maps even without mus"
    assert torch.any(fluence_maps.grad != 0), "At least some fluence_maps gradients should be non-zero"


# ---------------------------------------------------------------------------
# Equivalence test
# ---------------------------------------------------------------------------

def test_fluence_maps_equivalent_to_aperture_path(
    dose_engine, default_beam_sequence, default_ct_image
):
    """Dose from pre-computed fluence_maps must equal dose from aperture parameters."""
    leaf_positions = default_beam_sequence.leaf_positions.unsqueeze(0)
    mus = default_beam_sequence.mus.unsqueeze(0)
    jaw_positions = default_beam_sequence.jaw_positions.unsqueeze(0)

    # Standard aperture-based run; capture the intermediate fluence maps [B*G, H, W]
    _, batched_fluence_maps, _, dose_aperture = dose_engine.forward(
        leaf_positions=leaf_positions,
        mus=mus,
        jaw_positions=jaw_positions,
        density_image=default_ct_image,
        return_intermediates=True,
    )

    # Re-run with the extracted maps and the same mus — both paths now scale by mus
    dose_from_maps = dose_engine.forward(
        leaf_positions=None,
        mus=mus,
        jaw_positions=None,
        density_image=default_ct_image,
        fluence_maps=batched_fluence_maps.detach(),
    )

    assert torch.allclose(dose_aperture, dose_from_maps, atol=1e-5), (
        "Dose from fluence_maps path should match dose from aperture path"
    )


# ---------------------------------------------------------------------------
# Gradient flow test (with mus)
# ---------------------------------------------------------------------------

def test_forward_fluence_maps_gradient_flow(
    dose_engine, default_field_size, default_device, default_dtype
):
    """Gradients should propagate back through fluence_maps into the loss."""
    B, G = 1, dose_engine.number_of_beams
    H, W = default_field_size
    fluence_maps = torch.ones(B, G, H, W, device=default_device, dtype=default_dtype, requires_grad=True)
    mus = torch.ones(B, G, device=default_device, dtype=default_dtype)
    # Use uniform non-zero density so pencil-beam kernels are non-trivial
    water_ct = torch.ones((B, *dose_engine.dose_grid_shape), device=default_device, dtype=default_dtype)

    dose = dose_engine.forward(
        leaf_positions=None,
        mus=mus,
        jaw_positions=None,
        density_image=water_ct,
        fluence_maps=fluence_maps,
    )
    dose.sum().backward()

    assert fluence_maps.grad is not None, "Gradients should reach fluence_maps"
    assert torch.any(fluence_maps.grad != 0), "At least some fluence_maps gradients should be non-zero"


# ---------------------------------------------------------------------------
# Validation / error tests
# ---------------------------------------------------------------------------

def test_forward_fluence_maps_wrong_spatial_dims_raises(
    dose_engine, default_device, default_dtype, default_ct_image
):
    """forward() with fluence_maps whose spatial dims don't match field_size should raise."""
    B, G = 1, dose_engine.number_of_beams
    bad_maps = torch.ones(B, G, 100, 100, device=default_device, dtype=default_dtype)

    with pytest.raises(AssertionError):
        dose_engine.forward(
            leaf_positions=None,
            mus=None,
            jaw_positions=None,
            density_image=default_ct_image,
            fluence_maps=bad_maps,
        )


def test_forward_fluence_maps_wrong_ndim_raises(
    dose_engine, default_field_size, default_device, default_dtype, default_ct_image
):
    """forward() with a 2D fluence_maps tensor should raise ValueError."""
    bad_maps = torch.ones(*default_field_size, device=default_device, dtype=default_dtype)  # 2D

    with pytest.raises(ValueError):
        dose_engine.forward(
            leaf_positions=None,
            mus=None,
            jaw_positions=None,
            density_image=default_ct_image,
            fluence_maps=bad_maps,
        )
