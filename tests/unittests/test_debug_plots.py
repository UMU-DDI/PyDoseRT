import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import torch

from pydosert import DoseEngine, HeterogeneityDoseEngine
from pydosert.utils.debug_plots import plot_beam_debug, plot_total_dose_debug


@pytest.fixture
def dose_engine(
    default_machine_config, default_ct_array_shape, default_resolution,
    default_beam_sequence, default_kernel_size, default_device, default_dtype,
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
def het_engine(
    default_machine_config, default_ct_array_shape, default_resolution,
    default_beam_sequence, default_kernel_size, default_device, default_dtype,
):
    return HeterogeneityDoseEngine(
        machine_config=default_machine_config,
        kernel_size=default_kernel_size,
        dose_grid_spacing=default_resolution,
        dose_grid_shape=default_ct_array_shape,
        beam_template=default_beam_sequence,
        device=default_device,
        dtype=default_dtype,
    )


def test_plot_beam_debug_writes_png(tmp_path, default_ct_array_shape, default_device, default_dtype,
                                     default_iso_center, default_resolution, default_field_size):
    H, D, W = default_ct_array_shape
    ct = torch.ones((1, H, D, W), device=default_device, dtype=default_dtype)
    dose = torch.ones((1, H, D, W), device=default_device, dtype=default_dtype)
    fluence = torch.ones(default_field_size, device=default_device, dtype=default_dtype)
    rad_depth = torch.linspace(0, 100, H * D * W, device=default_device, dtype=default_dtype)
    rad_depth = rad_depth.reshape(1, D, H, W)

    out = tmp_path / "beam.png"
    plot_beam_debug(
        out_path=out,
        beam_index=0,
        gantry_angle_rad=0.0,
        mu=1.0,
        ct=ct,
        dose=dose,
        iso_center=default_iso_center,
        dose_grid_spacing=default_resolution,
        fluence_map=fluence,
        rad_depth_bev=rad_depth,
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_total_dose_debug_writes_png(tmp_path, default_ct_array_shape, default_device, default_dtype,
                                          default_iso_center, default_resolution):
    H, D, W = default_ct_array_shape
    ct = torch.ones((1, H, D, W), device=default_device, dtype=default_dtype)
    dose = torch.rand((1, H, D, W), device=default_device, dtype=default_dtype)
    out = tmp_path / "total.png"
    plot_total_dose_debug(
        out_path=out,
        ct=ct,
        dose=dose,
        iso_center=default_iso_center,
        dose_grid_spacing=default_resolution,
    )
    assert out.exists() and out.stat().st_size > 0


def test_compute_dose_sequential_debug_dir_baseline(
    dose_engine, default_beam_sequence, default_ct_array_shape, default_device, default_dtype, tmp_path,
):
    ct = torch.ones((1, *default_ct_array_shape), device=default_device, dtype=default_dtype)
    dose_engine.compute_dose_sequential(
        default_beam_sequence, density_image=ct, debug_dir=str(tmp_path),
    )
    # At least one per-beam PNG and the total PNG should exist.
    beam_pngs = list(tmp_path.glob("beam_*.png"))
    assert len(beam_pngs) == len(default_beam_sequence)
    assert (tmp_path / "total.png").exists()


def test_compute_dose_sequential_debug_dir_heterogeneity(
    het_engine, default_beam_sequence, default_ct_array_shape, default_device, default_dtype, tmp_path,
):
    ct = torch.ones((1, *default_ct_array_shape), device=default_device, dtype=default_dtype)
    het_engine.compute_dose_sequential(
        default_beam_sequence, density_image=ct, debug_dir=str(tmp_path),
    )
    beam_pngs = list(tmp_path.glob("beam_*.png"))
    assert len(beam_pngs) == len(default_beam_sequence)
    assert (tmp_path / "total.png").exists()
