import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import torch

from pydosert import HeterogeneityDoseEngine, DoseEngine, BaseDoseEngine
from pydosert.data import BeamSequence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reference_depths_mm():
    return (0.0, 10.0, 20.0, 40.0, 50.0, 100.0)


@pytest.fixture
def het_engine(
    default_machine_config,
    default_ct_array_shape,
    default_resolution,
    default_beam_sequence,
    default_kernel_size,
    default_device,
    default_dtype,
    reference_depths_mm,
):
    return HeterogeneityDoseEngine(
        machine_config=default_machine_config,
        kernel_size=default_kernel_size,
        dose_grid_spacing=default_resolution,
        dose_grid_shape=default_ct_array_shape,
        beam_template=default_beam_sequence,
        reference_depths_mm=reference_depths_mm,
        device=default_device,
        dtype=default_dtype,
    )


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
def multi_beam_het_engine(
    default_machine_config,
    default_ct_array_shape,
    default_resolution,
    multi_beam_sequence,
    default_kernel_size,
    default_device,
    default_dtype,
    reference_depths_mm,
):
    return HeterogeneityDoseEngine(
        machine_config=default_machine_config,
        kernel_size=default_kernel_size,
        dose_grid_spacing=default_resolution,
        dose_grid_shape=default_ct_array_shape,
        beam_template=multi_beam_sequence,
        reference_depths_mm=reference_depths_mm,
        device=default_device,
        dtype=default_dtype,
    )


@pytest.fixture
def default_ct_image(default_ct_array_shape, default_device, default_dtype):
    return torch.zeros((1, *default_ct_array_shape), device=default_device, dtype=default_dtype)


@pytest.fixture
def water_ct_image(default_ct_array_shape, default_device, default_dtype):
    return torch.ones((1, *default_ct_array_shape), device=default_device, dtype=default_dtype)


# ---------------------------------------------------------------------------
# Basic sanity
# ---------------------------------------------------------------------------

def test_engine_inherits_base(het_engine):
    assert isinstance(het_engine, BaseDoseEngine)


def test_reference_depths_validation(
    default_machine_config, default_ct_array_shape, default_resolution,
    default_beam_sequence, default_kernel_size, default_device, default_dtype,
):
    """A single reference depth is rejected — interpolation needs at least two."""
    with pytest.raises(ValueError):
        HeterogeneityDoseEngine(
            machine_config=default_machine_config,
            kernel_size=default_kernel_size,
            dose_grid_spacing=default_resolution,
            dose_grid_shape=default_ct_array_shape,
            beam_template=default_beam_sequence,
            reference_depths_mm=(50.0,),
            device=default_device,
            dtype=default_dtype,
        )


def test_forward_output_shape_with_fluence_maps(
    het_engine, default_field_size, default_device, default_dtype, default_ct_image
):
    B, G = 1, het_engine.number_of_beams
    H, W = default_field_size
    fluence_maps = torch.ones(B, G, H, W, device=default_device, dtype=default_dtype)
    mus = torch.ones(B, G, device=default_device, dtype=default_dtype)

    dose = het_engine.forward(
        leaf_positions=None,
        mus=mus,
        jaw_positions=None,
        density_image=default_ct_image,
        fluence_maps=fluence_maps,
    )
    assert dose.shape == (B, *het_engine.dose_grid_shape)


def test_forward_output_shape_multi_beam(
    multi_beam_het_engine, default_field_size, default_device, default_dtype, default_ct_image,
):
    B = 1
    G = multi_beam_het_engine.number_of_beams
    H, W = default_field_size
    fluence_maps = torch.ones(B, G, H, W, device=default_device, dtype=default_dtype)
    mus = torch.ones(B, G, device=default_device, dtype=default_dtype)

    dose = multi_beam_het_engine.forward(
        leaf_positions=None,
        mus=mus,
        jaw_positions=None,
        density_image=default_ct_image,
        fluence_maps=fluence_maps,
    )
    assert dose.shape == (B, *multi_beam_het_engine.dose_grid_shape)


def test_compute_dose_auto_unsqueeze(
    het_engine, default_beam_sequence, default_field_size, default_device, default_dtype, default_ct_image,
):
    G = len(default_beam_sequence)
    H, W = default_field_size
    fluence_maps = torch.ones(G, H, W, device=default_device, dtype=default_dtype)

    dose = het_engine.compute_dose(
        beam_input=default_beam_sequence,
        density_image=default_ct_image,
        fluence_maps=fluence_maps,
    )
    assert dose.shape == (1, *het_engine.dose_grid_shape)


# ---------------------------------------------------------------------------
# Semantics
# ---------------------------------------------------------------------------

def test_zero_density_gives_zero_dose(
    het_engine, default_field_size, default_device, default_dtype, default_ct_image,
):
    """With vacuum everywhere, all rad depths are at the low reference (0 mm),
    where the pencil-beam model delivers zero dose (below the depth threshold)."""
    B, G = 1, het_engine.number_of_beams
    H, W = default_field_size
    fluence_maps = torch.ones(B, G, H, W, device=default_device, dtype=default_dtype)
    mus = torch.ones(B, G, device=default_device, dtype=default_dtype)

    dose = het_engine.forward(
        leaf_positions=None,
        mus=mus,
        jaw_positions=None,
        density_image=default_ct_image,
        fluence_maps=fluence_maps,
    )
    assert torch.all(dose == 0), "Vacuum CT must give zero dose everywhere."


def test_mus_scales_dose(
    het_engine, default_field_size, default_device, default_dtype, water_ct_image,
):
    B, G = 1, het_engine.number_of_beams
    H, W = default_field_size
    fluence_maps = torch.ones(B, G, H, W, device=default_device, dtype=default_dtype)

    dose_no_mus = het_engine.forward(
        leaf_positions=None,
        mus=None,
        jaw_positions=None,
        density_image=water_ct_image,
        fluence_maps=fluence_maps,
    )
    scale = 3.0
    mus = torch.full((B, G), scale, device=default_device, dtype=default_dtype)
    dose_with_mus = het_engine.forward(
        leaf_positions=None,
        mus=mus,
        jaw_positions=None,
        density_image=water_ct_image,
        fluence_maps=fluence_maps,
    )
    assert torch.allclose(dose_with_mus, dose_no_mus * scale, atol=1e-4)


def test_gradient_flow_through_fluence_maps(
    het_engine, default_field_size, default_device, default_dtype,
):
    B, G = 1, het_engine.number_of_beams
    H, W = default_field_size
    fluence_maps = torch.ones(B, G, H, W, device=default_device, dtype=default_dtype, requires_grad=True)
    mus = torch.ones(B, G, device=default_device, dtype=default_dtype)
    water_ct = torch.ones((B, *het_engine.dose_grid_shape), device=default_device, dtype=default_dtype)

    dose = het_engine.forward(
        leaf_positions=None,
        mus=mus,
        jaw_positions=None,
        density_image=water_ct,
        fluence_maps=fluence_maps,
    )
    dose.sum().backward()

    assert fluence_maps.grad is not None
    assert torch.any(fluence_maps.grad != 0)


# ---------------------------------------------------------------------------
# Cross-engine behavioural comparison
# ---------------------------------------------------------------------------

def test_similar_order_of_magnitude_to_baseline_in_water(
    default_machine_config, default_ct_array_shape, default_resolution,
    default_beam_sequence, default_kernel_size, default_device, default_dtype,
    reference_depths_mm,
):
    """In a homogeneous water phantom the two engines should agree within ~1e-1
    in dose magnitude — not identical (they use different kernel-sampling
    strategies), but close enough that a bug in the new engine would be obvious."""
    base = DoseEngine(
        machine_config=default_machine_config,
        kernel_size=default_kernel_size,
        dose_grid_spacing=default_resolution,
        dose_grid_shape=default_ct_array_shape,
        beam_template=default_beam_sequence,
        device=default_device,
        dtype=default_dtype,
    )
    vol = HeterogeneityDoseEngine(
        machine_config=default_machine_config,
        kernel_size=default_kernel_size,
        dose_grid_spacing=default_resolution,
        dose_grid_shape=default_ct_array_shape,
        beam_template=default_beam_sequence,
        reference_depths_mm=reference_depths_mm,
        device=default_device,
        dtype=default_dtype,
    )

    B, G = 1, base.number_of_beams
    H, W = base.field_size
    fluence_maps = torch.ones(B, G, H, W, device=default_device, dtype=default_dtype)
    mus = torch.ones(B, G, device=default_device, dtype=default_dtype)
    water_ct = torch.ones((B, *default_ct_array_shape), device=default_device, dtype=default_dtype)

    d_base = base.forward(None, mus, None, water_ct, fluence_maps=fluence_maps)
    d_vol = vol.forward(None, mus, None, water_ct, fluence_maps=fluence_maps)

    # Both engines must produce some dose; the new one should be within an
    # order of magnitude of the baseline in a homogeneous phantom.
    assert d_base.abs().sum() > 0
    assert d_vol.abs().sum() > 0
    ratio = d_vol.abs().sum() / d_base.abs().sum()
    assert 0.1 < float(ratio) < 10.0, f"Dose magnitudes differ too much: ratio={ratio}"


def test_lateral_inhomogeneity_changes_dose(
    het_engine, default_field_size, default_device, default_dtype, default_ct_array_shape,
):
    """The whole point of 3D density correction: a lateral inhomogeneity should
    change the dose somewhere in the volume."""
    B, G = 1, het_engine.number_of_beams
    H, W = default_field_size
    fluence_maps = torch.ones(B, G, H, W, device=default_device, dtype=default_dtype)
    mus = torch.ones(B, G, device=default_device, dtype=default_dtype)

    water_ct = torch.ones((B, *default_ct_array_shape), device=default_device, dtype=default_dtype)
    inhomogeneous_ct = water_ct.clone()
    # Insert a low-density air channel on one lateral side; the baseline central-
    # axis-only engine would not "see" this, but a 3D engine must.
    inhomogeneous_ct[:, :, :, : default_ct_array_shape[2] // 4] = 0.0

    dose_water = het_engine.forward(None, mus, None, water_ct, fluence_maps=fluence_maps)
    dose_inh = het_engine.forward(None, mus, None, inhomogeneous_ct, fluence_maps=fluence_maps)

    assert not torch.allclose(dose_water, dose_inh, atol=1e-4), (
        "3D density correction should respond to a lateral inhomogeneity."
    )


def test_return_intermediates(
    het_engine, default_field_size, default_device, default_dtype, default_ct_image,
    reference_depths_mm,
):
    B, G = 1, het_engine.number_of_beams
    H, W = default_field_size
    fluence_maps = torch.ones(B, G, H, W, device=default_device, dtype=default_dtype)
    mus = torch.ones(B, G, device=default_device, dtype=default_dtype)

    rad_depth, fm, conv, dose = het_engine.forward(
        leaf_positions=None,
        mus=mus,
        jaw_positions=None,
        density_image=default_ct_image,
        fluence_maps=fluence_maps,
        return_intermediates=True,
    )
    BG = B * G
    D, H_, W_ = het_engine.dose_grid_shape
    assert rad_depth.shape == (BG, D, H_, W_)
    assert conv.shape == (BG, len(reference_depths_mm), D, H_, W_)
    assert dose.shape == (B, D, H_, W_)
