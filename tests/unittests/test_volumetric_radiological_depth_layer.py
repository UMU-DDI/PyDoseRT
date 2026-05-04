import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
import math
import pytest
import torch

from pydosert.layers import VolumetricRadiologicalDepthLayer


@pytest.fixture
def layer_factory(default_machine_config, default_resolution, default_ct_array_shape,
                  default_iso_center, default_device, default_dtype):
    """Build a VolumetricRadiologicalDepthLayer for a supplied list of gantry angles."""
    def _make(gantry_angles):
        if not isinstance(gantry_angles, torch.Tensor):
            gantry_angles = torch.tensor(gantry_angles, device=default_device, dtype=default_dtype)
        return VolumetricRadiologicalDepthLayer(
            machine_config=default_machine_config,
            resolution=default_resolution,
            ct_array_shape=default_ct_array_shape,
            gantry_angles=gantry_angles,
            iso_center=default_iso_center,
            device=default_device,
            dtype=default_dtype,
        )
    return _make


def test_output_shape_matches_bev_layout(layer_factory, default_ct_array_shape,
                                         default_device, default_dtype):
    H, D, W = default_ct_array_shape
    gantry_angles = torch.tensor([0.0], device=default_device, dtype=default_dtype)
    layer = layer_factory(gantry_angles)

    B = 2
    ct = torch.ones((B, H, D, W), device=default_device, dtype=default_dtype)
    out = layer(ct)

    assert out.shape == (B * gantry_angles.shape[0], D, H, W), \
        f"Expected BEV layout [B*G, D, H, W], got {out.shape}"


def test_zero_density_gives_zero_depth(layer_factory, default_ct_array_shape,
                                       default_device, default_dtype):
    H, D, W = default_ct_array_shape
    gantry_angles = torch.tensor([0.0, math.pi / 2], device=default_device, dtype=default_dtype)
    layer = layer_factory(gantry_angles)

    ct = torch.zeros((1, H, D, W), device=default_device, dtype=default_dtype)
    rad = layer(ct)
    assert torch.all(rad == 0), "Vacuum CT must give zero radiological depth everywhere."


def test_uniform_water_linear_growth_along_depth(layer_factory, default_ct_array_shape,
                                                 default_resolution, default_iso_center,
                                                 default_sid, default_device, default_dtype):
    """With density=1 everywhere and gantry=0, rad depth should grow linearly along D.

    For voxel d the integrated density at voxel centre is ``(d + 0.5) * step``,
    where ``step`` is the physical path length per BEV-depth step. Because the
    layer integrates along divergent rays, off-axis columns have a step that
    is slightly longer than ``res_d`` (by the SAD/SID-style stretch factor).
    """
    H, D, W = default_ct_array_shape
    res_h, res_d, res_w = default_resolution
    iso_h, _, iso_w = default_iso_center
    gantry_angles = torch.tensor([0.0], device=default_device, dtype=default_dtype)
    layer = layer_factory(gantry_angles)

    ct = torch.ones((1, H, D, W), device=default_device, dtype=default_dtype)
    rad = layer(ct)  # [1, D, H, W]
    # Path length along the divergent ray that passes through the BEV column
    # at (H // 2, W // 2): sqrt(1 + (h/SID)^2 + (w/SID)^2) * res_d.
    h_iso = (H // 2) * res_h - iso_h
    w_iso = (W // 2) * res_w - iso_w
    step = res_d * (1.0 + (h_iso / default_sid) ** 2 + (w_iso / default_sid) ** 2) ** 0.5
    expected = (torch.arange(D, device=default_device, dtype=default_dtype) + 0.5) * step
    # Interior column; away from borders that can suffer from grid_sample edge effects.
    column = rad[0, :, H // 2, W // 2]
    assert torch.allclose(column, expected, atol=1e-2), \
        f"Expected linear growth {expected.tolist()[:5]}..., got {column.tolist()[:5]}..."


def test_rotation_changes_depth_profile(layer_factory, default_ct_array_shape,
                                        default_device, default_dtype):
    """At gantry=90 the BEV depth axis aligns with CT width; a W-slab becomes a D-slab."""
    H, D, W = default_ct_array_shape
    gantry_angles = torch.tensor([0.0, math.pi / 2], device=default_device, dtype=default_dtype)
    layer = layer_factory(gantry_angles)

    # Slab of density in the first half along the CT depth direction, with air
    # margins on the H and W sides to mirror a realistic clinical CT volume
    # (and avoid border-padding artifacts at the BEV edges).
    ct = torch.zeros((1, H, D, W), device=default_device, dtype=default_dtype)
    ct[:, H // 4 : 3 * H // 4, : D // 2, W // 4 : 3 * W // 4] = 1.0

    rad = layer(ct)  # [2, D, H, W]
    # At gantry=0 the slab sits in the shallow half of the BEV depth axis.
    col0 = rad[0, :, H // 2, W // 2]
    assert col0[0] < col0[-1], "Gantry=0 rad depth should increase with BEV depth."

    # At gantry=90 the CT depth slab becomes a lateral slab in BEV; a column
    # that sits outside the slab laterally sees less density than the central
    # gantry=0 column.
    col90_edge = rad[1, :, H // 2, W - 1]
    assert torch.all(col90_edge == 0) or col90_edge.sum() < col0.sum(), (
        "At gantry=90 far edges should have much less (or zero) radiological depth than on-axis gantry=0."
    )


def test_no_grad_output(layer_factory, default_ct_array_shape, default_device, default_dtype):
    """Radiological depth has no gradient dependency expected on fluence —
    but it should still flow gradients through the CT input when requires_grad=True.
    (Rad-depth is typically consumed under torch.no_grad, so this is mainly a
    sanity check that the layer itself is differentiable.)"""
    H, D, W = default_ct_array_shape
    gantry_angles = torch.tensor([0.0], device=default_device, dtype=default_dtype)
    layer = layer_factory(gantry_angles)

    ct = torch.ones((1, H, D, W), device=default_device, dtype=default_dtype, requires_grad=True)
    rad = layer(ct)
    rad.sum().backward()
    assert ct.grad is not None
    assert torch.any(ct.grad != 0)
