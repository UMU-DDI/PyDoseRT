import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from skimage import measure
from matplotlib.colors import ListedColormap
import os
from scipy import ndimage
import cv2
from pydosert.data.beam import BeamSequence
from pydosert.engine.dose_engine import DoseEngine
from pydosert.data import Patient, OptimizationConfig
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D

def overlay_mask_outline(mask_slice, color="red", linewidth=1, sigma=2.0):
    # Smooth the binary mask to produce clean contour boundaries
    smoothed = gaussian_filter(mask_slice.astype(float), sigma=sigma)

    for contour in measure.find_contours(smoothed, 0.5):
        plt.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth, linestyle=(0, (1, 2)))

def print_paper_plot(
    experiment,
    treatment: object,
    patient: object,
    dose_pred: torch.Tensor,
    out_path=None,
    *,
    dose_alpha=0.6,
    isodose_percent_levels=(20, 40, 60, 80, 90, 95, 100, 105, 107, 110),
    cmap_dose="turbo",
    sagittal_z_index_range=(33, 70),
):
    """Publication-style optimized figure: axial, sagittal, and DVH."""

    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 25,
        "axes.labelsize": 20,
        "legend.fontsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })

    if dose_pred.ndim == 4:
        dose_pred = dose_pred[0]

    ct = patient._ct_tensor.cpu().detach().numpy()
    dose_ref = patient.number_of_fractions * patient.dose.cpu().detach().numpy()
    dose_calc = patient.number_of_fractions * dose_pred.cpu().detach().numpy()

    prescribed = getattr(treatment, "prescription_gy", None)
    if prescribed is None:
        dose_max = float(max(np.max(dose_ref), np.max(dose_calc)))
    else:
        dose_max = float(prescribed)

    patient_structures = {
        name: mask.cpu().detach().numpy()
        for name, mask in patient.structures.items()
    }

    def _resolve_mask(struct_name: str):
        if struct_name in patient_structures:
            return patient_structures[struct_name]
        low = struct_name.lower()
        for key in patient_structures:
            key_low = key.lower()
            if low in key_low or key_low in low:
                return patient_structures[key]
        return None

    def _iter_plot_structures(skip_body: bool = False):
        for struct_name, struct_cfg in treatment.structures.items():
            low = struct_name.lower()
            if skip_body and ("body" in low or "external" in low):
                continue
            mask = _resolve_mask(struct_name)
            if mask is None:
                continue
            yield struct_name, struct_cfg, mask

    def _structure_color(struct_name: str, struct_cfg: dict) -> str:
        low = struct_name.lower()
        if "ptv" in low:
            return "#d7191c"  # red
        if "bladder" in low:
            return "#1b9e77"  # green
        if "rectum" in low:
            return "#8c510a"  # brown
        if "femoralhead_l" in low or ("femoral" in low and "_l" in low):
            return "#6a3d9a"  # dark purple
        if "femoralhead_r" in low or ("femoral" in low and "_r" in low):
            return "#b57edc"  # light purple
        return struct_cfg.get("color", "white")

    reference_mask = None
    for key, mask in patient_structures.items():
        if "ptv" in key.lower():
            reference_mask = mask
            break
    if reference_mask is None:
        reference_mask = next(iter(patient_structures.values()))

    body_mask = None
    for key, mask in patient_structures.items():
        low = key.lower()
        if "body" in low or "external" in low:
            body_mask = mask
            break
    if body_mask is None:
        body_mask = reference_mask

    com = np.array(ndimage.center_of_mass(reference_mask), dtype=np.int32)
    axial_z = int(np.clip(com[0], 0, ct.shape[0] - 1))
    sagittal_x = int(np.clip(com[2], 0, ct.shape[2] - 1))

    body_slice = body_mask[axial_z] > 0
    ys, xs = np.where(body_slice)
    if ys.size > 0 and xs.size > 0:
        y0 = max(int(ys.min()) - 8, 0)
        y1 = min(int(ys.max()) + 9, ct.shape[1])
        x0 = max(int(xs.min()) - 8, 0)
        x1 = min(int(xs.max()) + 9, ct.shape[2])
        trim_y = int(0.06 * (y1 - y0))
        trim_x = int(0.03 * (x1 - x0))
        y0 = min(max(y0 + trim_y, 0), ct.shape[1] - 2)
        y1 = max(min(y1 - trim_y, ct.shape[1]), y0 + 2)
        x0 = min(max(x0 + trim_x, 0), ct.shape[2] - 2)
        x1 = max(min(x1 - trim_x, ct.shape[2]), x0 + 2)
    else:
        y0 = max(int(com[1]) - 64, 0)
        y1 = min(int(com[1]) + 64, ct.shape[1])
        x0 = max(int(com[2]) - 64, 0)
        x1 = min(int(com[2]) + 64, ct.shape[2])

    z_low, z_high = sagittal_z_index_range
    z0 = max(0, int(z_low))
    # +1 to treat range as inclusive, e.g. (20, 70) -> slices [20..70]
    z1 = min(ct.shape[0], int(z_high) + 1)
    if z1 <= z0 + 1:
        z0 = max(int(com[0]) - 40, 0)
        z1 = min(int(com[0]) + 40, ct.shape[0])

    ct_axial = ct[axial_z, y0:y1, x0:x1]
    dose_axial = dose_calc[axial_z, y0:y1, x0:x1]
    ct_sag = np.flipud(ct[z0:z1, y0:y1, sagittal_x])
    dose_sag = np.flipud(dose_calc[z0:z1, y0:y1, sagittal_x])

    boundaries_pct = (0,) + tuple(isodose_percent_levels)
    boundaries_abs = [b / 100.0 * dose_max for b in boundaries_pct]
    abs_max = float(max(np.max(dose_ref), np.max(dose_calc)))
    band_labels = [f"{int(boundaries_pct[i])}%" for i in range(len(boundaries_pct) - 1)]
    if abs_max > boundaries_abs[-1]:
        boundaries_abs.append(abs_max)
        band_labels.append(f">{isodose_percent_levels[-1]}%")

    n_colors = len(boundaries_abs) - 1
    alphas = np.linspace(0.15, 0.95, n_colors)
    base_cmap = plt.get_cmap(cmap_dose)
    rgb_colors = base_cmap(np.linspace(0, 1, n_colors))[:, :3]
    rgba_colors = [(r, g, b, a) for (r, g, b), a in zip(rgb_colors, alphas)]
    cmap_disc = ListedColormap(rgba_colors)

    legend_indices = [i for i, label in enumerate(band_labels) if label not in {"0%", "20%"}]
    isodose_handles = [
        Line2D([0], [0], color=rgba_colors[i][:3], linewidth=4, label=band_labels[i])
        for i in legend_indices
    ]

    fig = plt.figure(figsize=(23.5, 9.2))
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.05, 0.14, 2.15],
        height_ratios=[1, 1],
        wspace=0.18,
        hspace=0.20,
    )

    ax_axial = fig.add_subplot(gs[0, 0])
    ax_sag = fig.add_subplot(gs[1, 0])
    ax_leg = fig.add_subplot(gs[:, 1])
    ax_dvh = fig.add_subplot(gs[:, 2])
    ax_leg.axis("off")

    ax_axial.imshow(ct_axial, cmap="gray", interpolation="none", aspect="equal")
    ax_axial.contourf(dose_axial, levels=boundaries_abs, cmap=cmap_disc, antialiased=True)
    ax_axial.contour(dose_axial, levels=boundaries_abs, linewidths=0.7, colors="white", alpha=0.9)
    for struct_name, struct_cfg, roi in _iter_plot_structures(skip_body=True):
        plt.sca(ax_axial)
        overlay_mask_outline(
            roi[axial_z, y0:y1, x0:x1],
            color=_structure_color(struct_name, struct_cfg),
            linewidth=2.0,
        )
    ax_axial.set_title("PyDoseRT Optimized - axial")
    x_ticks_axial = np.linspace(0, ct_axial.shape[1] - 1, 6, dtype=int)
    y_ticks_axial = np.linspace(0, ct_axial.shape[0] - 1, 5, dtype=int)
    ax_axial.set_xticks(x_ticks_axial)
    ax_axial.set_yticks(y_ticks_axial)
    ax_axial.set_xticklabels((x0 + x_ticks_axial).astype(int))
    ax_axial.set_yticklabels((y0 + y_ticks_axial).astype(int))
    ax_axial.set_xlabel("x index")
    ax_axial.set_ylabel("y index")
    ax_axial.tick_params(axis="both", labelsize=17)

    ax_sag.imshow(ct_sag, cmap="gray", interpolation="none", aspect="equal")
    ax_sag.contourf(dose_sag, levels=boundaries_abs, cmap=cmap_disc, antialiased=True)
    ax_sag.contour(dose_sag, levels=boundaries_abs, linewidths=0.7, colors="white", alpha=0.9)
    for struct_name, struct_cfg, roi in _iter_plot_structures(skip_body=True):
        plt.sca(ax_sag)
        overlay_mask_outline(
            np.flipud(roi[z0:z1, y0:y1, sagittal_x]),
            color=_structure_color(struct_name, struct_cfg),
            linewidth=2.0,
        )
    ax_sag.set_title("PyDoseRT Optimized - sagittal")
    x_ticks_sag = np.linspace(0, ct_sag.shape[1] - 1, 6, dtype=int)
    y_ticks_sag = np.linspace(0, ct_sag.shape[0] - 1, 5, dtype=int)
    ax_sag.set_xticks(x_ticks_sag)
    ax_sag.set_yticks(y_ticks_sag)
    ax_sag.set_xticklabels((y0 + x_ticks_sag).astype(int))
    ax_sag.set_yticklabels((z1 - 1 - y_ticks_sag).astype(int))
    ax_sag.set_xlabel("y index")
    ax_sag.set_ylabel("z index")
    ax_sag.tick_params(axis="both", labelsize=17)

    ax_leg.legend(
        handles=isodose_handles,
        title="Isodose levels",
        loc="center left",
        bbox_to_anchor=(-1, 0.5),
        ncol=1,
        frameon=False,
        title_fontsize=18,
    )

    dvh_upper = max(dose_max, abs_max)
    for struct_name, struct_cfg, roi in _iter_plot_structures(skip_body=False):
        dvh_color = _structure_color(struct_name, struct_cfg)

        dose_values = dose_calc[roi > 0.0]
        if dose_values.size == 0:
            continue
        bins = np.linspace(0, dvh_upper, 1000)
        hist, bin_edges = np.histogram(dose_values, bins=bins, density=False)
        cumulative_hist = np.cumsum(hist[::-1])[::-1]
        cumulative_hist_normalized = cumulative_hist / cumulative_hist.max()
        ax_dvh.plot(
            bin_edges[:-1],
            cumulative_hist_normalized,
            linestyle="solid",
            label=struct_name,
            color=dvh_color,
            linewidth=2.0,
        )

    ax_dvh.set_xlabel("Dose (Gy)")
    ax_dvh.set_ylabel("Volume Fraction")
    ax_dvh.set_title("Dose Volume Histogram (DVH)")
    ax_dvh.set_xlim(0.0, dvh_upper * 1.03 if dvh_upper > 0 else 1.0)
    ax_dvh.set_ylim(0.0, 1.05)
    ax_dvh.grid(True, linestyle=":", linewidth=0.7)
    ax_dvh.legend(loc="lower left", frameon=False)

    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.1, top=0.93, wspace=0.18, hspace=0.20)

    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        if experiment is not None:
            experiment.log_figure(out_path, overwrite=True)
        plt.close(fig)

def print_comparison_plot(
    treatment: object,
    patient: object,
    dose_pred: torch.Tensor,
    out_path=None,
    isodose_percent_levels=(20, 40, 60, 80, 90, 95, 100, 105, 107, 110),
    cmap_dose="turbo",
    profile_xlim=(25, 150),
):
    """Publication-oriented comparison plot (TPS vs PyDoseRT)."""

    plt.rcParams.update({
        "font.size": 17,
        "axes.titlesize": 21,
        "axes.labelsize": 17,
        "legend.fontsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    })

    ct = patient._ct_tensor.cpu().detach().numpy()
    dose_ref = patient.number_of_fractions * patient.dose.cpu().detach().numpy()
    dose_calc = patient.number_of_fractions * dose_pred.cpu().detach().numpy()

    prescribed = getattr(treatment, "prescription_gy", None)
    if prescribed is None:
        dose_max = float(max(np.max(dose_ref), np.max(dose_calc)))
    else:
        dose_max = float(prescribed)

    patient_structures = {
        name: mask.cpu().detach().numpy()
        for name, mask in patient.structures.items()
    }

    def _resolve_mask(struct_name: str):
        if struct_name in patient_structures:
            return patient_structures[struct_name]
        low = struct_name.lower()
        for key in patient_structures:
            key_low = key.lower()
            if low in key_low or key_low in low:
                return patient_structures[key]
        return None

    def _pick_reference_mask():
        for key, mask in patient_structures.items():
            if "ptv" in key.lower():
                return mask
        return next(iter(patient_structures.values()))

    def _pick_body_mask(fallback_mask):
        for key, mask in patient_structures.items():
            key_low = key.lower()
            if "body" in key_low or "external" in key_low:
                return mask
        return fallback_mask

    ref_mask = _pick_reference_mask()
    com = np.array(ndimage.center_of_mass(ref_mask), dtype=np.int32)
    axial_z = int(np.clip(com[0], 0, ct.shape[0] - 1))

    body_mask = _pick_body_mask(ref_mask)
    body_slice = body_mask[axial_z] > 0
    ys, xs = np.where(body_slice)
    if ys.size > 0 and xs.size > 0:
        y0 = max(int(ys.min()) - 8, 0)
        y1 = min(int(ys.max()) + 9, ct.shape[1])
        x0 = max(int(xs.min()) - 8, 0)
        x1 = min(int(xs.max()) + 9, ct.shape[2])
        # Slight extra zoom to reduce air above/below the body.
        trim_y = int(0.06 * (y1 - y0))
        trim_x = int(0.03 * (x1 - x0))
        y0 = min(max(y0 + trim_y, 0), ct.shape[1] - 2)
        y1 = max(min(y1 - trim_y, ct.shape[1]), y0 + 2)
        x0 = min(max(x0 + trim_x, 0), ct.shape[2] - 2)
        x1 = max(min(x1 - trim_x, ct.shape[2]), x0 + 2)
    else:
        y0 = max(int(com[1]) - 64, 0)
        y1 = min(int(com[1]) + 64, ct.shape[1])
        x0 = max(int(com[2]) - 64, 0)
        x1 = min(int(com[2]) + 64, ct.shape[2])

    y_slice = int(np.clip(com[1], y0, y1 - 1))
    x_slice = int(np.clip(com[2], x0, x1 - 1))

    boundaries_pct = (0,) + tuple(isodose_percent_levels)
    boundaries_abs = [b / 100.0 * dose_max for b in boundaries_pct]
    abs_max = float(max(np.max(dose_ref), np.max(dose_calc)))
    band_labels = []
    for i in range(len(boundaries_pct) - 1):
        band_labels.append(f"{int(boundaries_pct[i])}%")
    if abs_max > boundaries_abs[-1]:
        boundaries_abs.append(abs_max)
        band_labels.append(f">{isodose_percent_levels[-1]}%")

    n_colors = len(boundaries_abs) - 1
    alphas = np.linspace(0.15, 0.95, n_colors)
    base_cmap = plt.get_cmap(cmap_dose)
    rgb_colors = base_cmap(np.linspace(0, 1, n_colors))[:, :3]
    rgba_colors = [(r, g, b, a) for (r, g, b), a in zip(rgb_colors, alphas)]
    cmap_disc = ListedColormap(rgba_colors)

    # Keep lower bands in the plot, but remove 0% and 20% from legend to reduce clutter.
    legend_indices = [i for i, label in enumerate(band_labels) if label not in {"0%", "20%"}]
    isodose_handles = [
        Line2D([0], [0], color=rgba_colors[i][:3], linewidth=4, label=band_labels[i])
        for i in legend_indices
    ]

    outline_items = []
    for struct_name, struct_cfg in treatment.structures.items():
        low = struct_name.lower()
        if "body" in low or "external" in low:
            continue
        resolved = _resolve_mask(struct_name)
        if resolved is not None:
            outline_items.append((resolved, struct_cfg.get("color", "white")))

    if not outline_items:
        outline_items = [(ref_mask, "white")]

    fig = plt.figure(figsize=(20, 10.2))
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.0, 1.0],
        height_ratios=[1, 1],
        wspace=0.38,
        hspace=0.22,
    )
    ax_ref = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[1, 0])

    ct_axial = ct[axial_z, y0:y1, x0:x1]
    ref_axial = dose_ref[axial_z, y0:y1, x0:x1]
    pred_axial = dose_calc[axial_z, y0:y1, x0:x1]
    line_y = y_slice - y0
    line_x = x_slice - x0

    ref_slice_guide_color = "#ff006e"
    calc_slice_guide_color = "#11a5ff"

    def _draw_axial(ax, dose_axial, title, show_x_ticks, guide_color):
        ax.imshow(ct_axial, cmap="gray", interpolation="none", aspect="equal")
        ax.contourf(dose_axial, levels=boundaries_abs, cmap=cmap_disc, antialiased=True)
        ax.contour(dose_axial, levels=boundaries_abs, linewidths=0.7, colors="white", alpha=0.9)
        for struct_mask, color in outline_items:
            plt.sca(ax)
            overlay_mask_outline(
                struct_mask[axial_z, y0:y1, x0:x1],
                color=color,
                linewidth=2.0,
            )
        ax.axhline(line_y, color=guide_color, linestyle="--", linewidth=2.0)
        ax.axvline(line_x, color=guide_color, linestyle="--", linewidth=2.0)
        ax.set_title(title, pad=10)
        y_ticks = np.linspace(0, ct_axial.shape[0] - 1, 5, dtype=int)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels((y0 + y_ticks).astype(int))
        ax.set_ylabel("y index")
        if show_x_ticks:
            x_ticks = np.linspace(0, ct_axial.shape[1] - 1, 6, dtype=int)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels((x0 + x_ticks).astype(int))
            ax.set_xlabel("x index")
        else:
            ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=14)
        if show_x_ticks:
            ax.tick_params(axis="x", labelsize=14)

    _draw_axial(
        ax_ref,
        ref_axial,
        "Reference TPS dose - axial",
        show_x_ticks=False,
        guide_color=ref_slice_guide_color,
    )
    _draw_axial(
        ax_pred,
        pred_axial,
        "PyDoseRT dose - axial",
        show_x_ticks=True,
        guide_color=calc_slice_guide_color,
    )

    # One shared vertical legend between axial and profile columns.
    fig.legend(
        handles=isodose_handles,
        title="Isodose levels",
        loc="center",
        bbox_to_anchor=(0.502, 0.56),
        ncol=1,
        frameon=False,
        title_fontsize=14,
    )

    # Profiles panel
    ax_lat = fig.add_subplot(gs[0, 1])
    ax_ap = fig.add_subplot(gs[1, 1], sharex=ax_lat)
    ax_lat_diff = ax_lat.twinx()
    ax_ap_diff = ax_ap.twinx()

    lateral_ref = dose_ref[axial_z, y_slice, :]
    lateral_pred = dose_calc[axial_z, y_slice, :]
    ap_ref = dose_ref[axial_z, :, x_slice]
    ap_pred = dose_calc[axial_z, :, x_slice]

    profile_len = min(lateral_ref.shape[0], ap_ref.shape[0])
    x = np.arange(profile_len)
    lateral_ref = lateral_ref[:profile_len]
    lateral_pred = lateral_pred[:profile_len]
    ap_ref = ap_ref[:profile_len]
    ap_pred = ap_pred[:profile_len]
    lateral_diff = np.abs(lateral_pred - lateral_ref)
    ap_diff = np.abs(ap_pred - ap_ref)

    ref_color = "#ff006e"
    pred_color = "#11a5ff"
    diff_color = "#ffbe0b"
    diff_style = (0, (6, 2, 1.2, 2))

    line_ref, = ax_lat.plot(x, lateral_ref, linestyle="--", color=ref_color, linewidth=2.3, label="Reference")
    line_pred, = ax_lat.plot(x, lateral_pred, linestyle="-", color=pred_color, linewidth=2.5, label="PyDoseRT")
    ax_lat_diff.fill_between(x, 0.0, lateral_diff, color=diff_color, alpha=0.20, zorder=1)
    line_diff, = ax_lat_diff.plot(
        x,
        lateral_diff,
        linestyle=diff_style,
        color=diff_color,
        linewidth=2.2,
        marker="o",
        markersize=2.8,
        markevery=8,
        label="Dose diff (Gy)",
        zorder=3,
    )

    ax_ap.plot(x, ap_ref, linestyle="--", color=ref_color, linewidth=2.3)
    ax_ap.plot(x, ap_pred, linestyle="-", color=pred_color, linewidth=2.5)
    ax_ap_diff.fill_between(x, 0.0, ap_diff, color=diff_color, alpha=0.20, zorder=1)
    ax_ap_diff.plot(
        x,
        ap_diff,
        linestyle=diff_style,
        color=diff_color,
        linewidth=2.2,
        marker="o",
        markersize=2.8,
        markevery=8,
        zorder=3,
    )

    ax_lat.set_title("Lateral profile")
    ax_ap.set_title("Anterior-posterior profile")
    ax_lat.set_ylabel("Dose (Gy)")
    ax_ap.set_ylabel("Dose (Gy)")
    ax_lat_diff.set_ylabel("Dose diff (Gy)")
    ax_ap.set_xlabel("Profile index (voxel)")
    ax_ap_diff.set_ylabel("Dose diff (Gy)")

    x_start = max(0, min(int(profile_xlim[0]), profile_len - 2))
    x_end = max(x_start + 1, min(int(profile_xlim[1]), profile_len - 1))
    ax_lat.set_xlim(x_start, x_end)
    ax_lat.set_ylim(bottom=0.0)
    ax_ap.set_ylim(bottom=0.0)

    ax_lat.grid(True, linestyle=":", linewidth=0.7)
    ax_ap.grid(True, linestyle=":", linewidth=0.7)

    diff_window = slice(x_start, x_end + 1)
    max_diff = max(float(np.max(lateral_diff[diff_window])), float(np.max(ap_diff[diff_window])))
    diff_ylim = max(0.1, 1.1 * max_diff)
    ax_lat_diff.set_ylim(0.0, diff_ylim)
    ax_ap_diff.set_ylim(0.0, diff_ylim)

    legend_handles = [
        line_ref,
        line_pred,
        Line2D(
            [0], [0],
            color=diff_color,
            linewidth=2.2,
            linestyle=diff_style,
            marker="o",
            markersize=4,
            label="Dose diff (Gy)",
        ),
    ]
    ax_lat.legend(handles=legend_handles, loc="upper right", ncol=1, frameon=False)

    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.08, top=0.97, wspace=0.38, hspace=0.22)

    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def print_results(
    experiment,
    treatment: OptimizationConfig,
    patient: Patient,
    beam_sequence: BeamSequence,
    dose_pred,
    title,
    plot_ct=True,
    preset="varian_10MV",
    out_path=None
):
    dose_max = patient.number_of_fractions * max(patient.dose.max(), dose_pred.max()).item()
    def _hide_ticks(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(bottom=False, left=False)

    def _imshow_fullwidth(ax, img, *, cmap='gray', vmin=None, vmax=None, alpha=1.0):
        """
        Show any array so it fills the axes horizontally and uses a fixed panel height.
        Keeping data coordinates unchanged ensures overlays (contours) stay aligned.
        """
        ax.imshow(
            img,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation='none',
            aspect='auto',   # <-- critical: fills the axes regardless of array shape
            alpha=alpha
        )
        _hide_ticks(ax)

    # Scales for gradients
    pred_mlc = beam_sequence.leaf_positions.unsqueeze(0)
    pred_mus = beam_sequence.mus.unsqueeze(0)
    pred_jaws = beam_sequence.jaw_positions.unsqueeze(0)

    # Visual parameters
    
    alpha = 0.30  # overlay transparency for gradients

    # Figure + GridSpec: one column, all rows share the same height
    # Adjust nrows if you add/remove panels. Here: 5 (machine) + 4 (dose) + 1 (DVH) = 10
    nrows = 10
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(nrows, 
                           1, 
                           figure=fig, 
                           hspace=0.45,
                           height_ratios=[1,1,1,1,1,1,1,1,1,4.0])
    
    if plot_ct:
        dose_alpha = 0.8
    else:
        dose_alpha = 1.0

    fig.suptitle(
        title,
        y=0.995
    )

    # --- 1) Jaws (centers)
    ax = fig.add_subplot(gs[0])
    ax.set_title('Jaws (lower)')
    _imshow_fullwidth(
        ax,
        np.transpose(pred_jaws.cpu().detach().numpy()[0, :, 0:1]),
        cmap='gray', vmin=-200.0, vmax=200.0
    )

    # --- 2) Jaws (widths)
    ax = fig.add_subplot(gs[1])
    ax.set_title('Jaws (higher)')
    _imshow_fullwidth(
        ax,
        np.transpose(pred_jaws.cpu().detach().numpy()[0, :, 1:2]),
        cmap='gray', vmin=-200.0, vmax=200.0
    )

    # --- 3) MLCs (centers)
    ax = fig.add_subplot(gs[2])
    ax.set_title('MLCs (left)')
    _imshow_fullwidth(
        ax,
        np.transpose(pred_mlc.cpu().detach().numpy()[0, :, :, 0]),
        cmap='gray', vmin=-200.0, vmax=200.0
    )

    # --- 4) MLCs (widths)
    ax = fig.add_subplot(gs[3])
    ax.set_title('MLCs (right)')
    _imshow_fullwidth(
        ax,
        np.transpose(pred_mlc.cpu().detach().numpy()[0, :, :, 1]),
        cmap='gray', vmin=-200.0, vmax=200.0
    )

    # --- 5) MUs
    ax = fig.add_subplot(gs[4])
    ax.set_title('MUs')
    _imshow_fullwidth(
        ax,
        pred_mus.cpu().detach().numpy(),
        cmap='gray', vmin=0.0, vmax=None
    )

    if (preset == "lund"):
        axial_z = 49
        axial_xstart = 64
        axial_xend = 192
        coronal_x = 128
        coronal_zstart = 16
        coronal_zend = 80
        coronal_ystart = 32
        coronal_yend = 224
    elif (preset == "varian_10MV"):
        axial_z = 84
        axial_xstart = 64
        axial_xend = 124
        coronal_x = 94
        coronal_zstart = 48
        coronal_zend = 124
        coronal_ystart = 64
        coronal_yend = 124
    elif (preset == "gold-atlas"):
        CoM = np.array(ndimage.measurements.center_of_mass(list(patient.structures.values())[0].cpu().detach().numpy()), dtype=np.int32)
        axial_z = CoM[0]
        axial_xstart = max(CoM[2] - 64, 0)
        axial_xend = CoM[2] + 64
        coronal_x = CoM[2]
        coronal_zstart = max(CoM[0] - 32, 0)
        coronal_zend = CoM[0] + 32
        coronal_ystart = max(CoM[1] - 64, 0)
        coronal_yend = CoM[1] + 64
    else:
        raise Exception("Preset missing")

    # If overlay_mask_outline expects already-sliced 2D arrays (as in your original code),
    # use these two helpers instead:
    def _dose_slice_axial(arr, z=44, x_start=0, x_end=256):
        return arr[z, x_start:x_end, :]

    def _dose_slice_coronal(arr, x=128, y_start=0, y_end=256, z_start=0, z_end=256):
        # coronal view, transpose to show (z, y) or (y, z) consistently
        # matching your original "np.transpose(...[0, 64:198, 128, :])"
        return np.flipud(arr[z_start:z_end, y_start:y_end,x])

    # --- 6) Dose distribution (pred, axial)
    ax = fig.add_subplot(gs[5])
    _imshow_fullwidth(ax, _dose_slice_axial(patient.number_of_fractions * dose_pred.cpu().detach().numpy(), z=axial_z, x_start=axial_xstart, x_end=axial_xend), cmap='jet', vmin=0.0, vmax=dose_max)
    _hide_ticks(ax)
    ax.set_title('Dose distribution (pred, axial)')
    for idx, color in enumerate([struct["color"] for struct_name, struct in treatment.structures.items()][:-1]):
        if len(patient.structures) <= idx:
            continue
        roi = list(patient.structures.values())[idx]
        overlay_mask_outline(roi.cpu().detach().numpy()[axial_z, axial_xstart:axial_xend, :], color=color)

    # --- 7) Dose distribution (pred, sagittal)
    ax = fig.add_subplot(gs[6])
    _imshow_fullwidth(ax, _dose_slice_coronal(patient.number_of_fractions * dose_pred.cpu().detach().numpy(), x=coronal_x, y_start=coronal_ystart, y_end=coronal_yend, z_start=coronal_zstart, z_end=coronal_zend), cmap='jet', vmin=0.0, vmax=dose_max)
    _hide_ticks(ax)
    ax.set_title('Dose distribution (pred, coronal)')
    for idx, color in enumerate([struct["color"] for struct_name, struct in treatment.structures.items()][:-1]):
        if len(patient.structures) <= idx:
            continue
        roi = list(patient.structures.values())[idx]
        overlay_mask_outline(np.flipud(roi.cpu().detach().numpy()[coronal_zstart:coronal_zend, coronal_ystart:coronal_yend, coronal_x]), color=color)

    # --- 8) Dose distribution (gt, axial)
    ax = fig.add_subplot(gs[7])
    if plot_ct:
        _imshow_fullwidth(ax, _dose_slice_axial(patient._ct_tensor.cpu().detach().numpy(), z=axial_z, x_start=axial_xstart, x_end=axial_xend), cmap='gray')
    _imshow_fullwidth(ax, _dose_slice_axial(patient.number_of_fractions * patient.dose.cpu().detach().numpy(), z=axial_z, x_start=axial_xstart, x_end=axial_xend), cmap='jet', vmin=0.0, vmax=dose_max, alpha=dose_alpha)
    _hide_ticks(ax)
    ax.set_title('Dose distribution (gt, axial)')
    for idx, color in enumerate([struct["color"] for struct_name, struct in treatment.structures.items()][:-1]):
        if len(patient.structures) <= idx:
            continue
        roi = list(patient.structures.values())[idx]
        overlay_mask_outline(roi.cpu().detach().numpy()[axial_z, axial_xstart:axial_xend, :], color=color)

    # --- 9) Dose distribution (gt, sagittal)
    ax = fig.add_subplot(gs[8])
    if plot_ct:
        _imshow_fullwidth(ax, _dose_slice_coronal(patient.number_of_fractions * patient.dose.cpu().detach().numpy(), x=coronal_x, y_start=coronal_ystart, y_end=coronal_yend, z_start=coronal_zstart, z_end=coronal_zend), cmap='gray')
    _imshow_fullwidth(ax, _dose_slice_coronal(patient.number_of_fractions * patient.dose.cpu().detach().numpy(), x=coronal_x, y_start=coronal_ystart, y_end=coronal_yend, z_start=coronal_zstart, z_end=coronal_zend), cmap='jet', vmin=0.0, vmax=dose_max, alpha=dose_alpha)
    _hide_ticks(ax)
    ax.set_title('Dose distribution (gt, coronal)')
    for idx, color in enumerate([struct["color"] for struct_name, struct in treatment.structures.items()][:-1]):
        roi = list(patient.structures.values())[idx]
        overlay_mask_outline(np.flipud(roi.cpu().detach().numpy()[coronal_zstart:coronal_zend, coronal_ystart:coronal_yend, coronal_x]), color=color)

    # --- 10) DVH (line plot; same panel height as others for uniformity)
    ax = fig.add_subplot(gs[9])
    for idx, (color, roi_name) in enumerate([(struct["color"], struct_name) for struct_name, struct in treatment.structures.items()]):
        if len(patient.structures) <= idx:
            continue
        roi = list(patient.structures.values())[idx]
        dose_values = patient.number_of_fractions * dose_pred[roi > 0.0].cpu().detach().numpy()
        if dose_values.size == 0:
            continue
        bins = np.linspace(0, dose_max, 1000)
        hist, bin_edges = np.histogram(dose_values, bins=bins, density=False)
        cumulative_hist = np.cumsum(hist[::-1])[::-1]
        cumulative_hist_normalized = np.divide(cumulative_hist, cumulative_hist.max())
        ax.plot(bin_edges[:-1], cumulative_hist_normalized, linestyle="solid", label=roi_name, color=color)

    for idx, color in enumerate([struct["color"] for struct_name, struct in treatment.structures.items()]):
        if len(patient.structures) <= idx:
            continue
        roi = list(patient.structures.values())[idx]
        dose_values = patient.number_of_fractions * patient.dose[roi > 0.0].cpu().detach().numpy()
        if dose_values.size == 0:
            continue
        bins = np.linspace(0, dose_max, 1000)
        hist, bin_edges = np.histogram(dose_values, bins=bins, density=False)
        cumulative_hist = np.cumsum(hist[::-1])[::-1]
        cumulative_hist_normalized = np.divide(cumulative_hist, cumulative_hist.max())
        ax.plot(bin_edges[:-1], cumulative_hist_normalized, linestyle="dashed", color=color)

    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Volume Fraction")
    ax.set_title("Dose Volume Histogram (DVH)")
    ax.grid(True)
    ax.legend(loc="lower left")

    # Layout & save
    fig.tight_layout(rect=[0, 0, 1, 0.97])  # keep space for the suptitle
    
    if (out_path is None):
        if (experiment is not None):
            save_path = "out/exp.png"
            plt.savefig(save_path, dpi=150)
            experiment.log_figure(save_path, overwrite=True)
        else:
            plt.show()
    else:
        plt.savefig(out_path)
        if (experiment is not None):
            experiment.log_figure(out_path, overwrite=True)
        plt.close()

def make_animation(experiment, 
                   patient_data: Patient, 
                   dose_layer: DoseEngine, 
                   beam_sequence: BeamSequence, 
                   dose_max=50.0,
                   out_path=None):
    """
    Modified version with tight square layout - two squares stacked vertically
    """
    density_image = (patient_data.density_image * patient_data.structures["External"]).unsqueeze(0)

    # Get the base colormap (jet)
    alpha_max = 1.0
    jet = plt.get_cmap('jet', 256)
    colors = jet(np.linspace(0, 1, 256))

    values = np.linspace(0, 1, 256)  # normalized 0..1
    alpha = np.clip(np.interp(values, [0, 1], [0.0, alpha_max]), 0, alpha_max)
    # this is equivalent to alpha = values, but more explicit

    colors[:, -1] = alpha
    jet_alpha = ListedColormap(colors)
    num_cps = len(beam_sequence)
    CoM = np.array(ndimage.measurements.center_of_mass(list(patient_data.structures.values())[0].cpu().detach().numpy()), dtype=np.int32)
    slice_idx = CoM[0]
    ct_data = patient_data._ct_tensor.cpu().detach().numpy()[slice_idx, :, :]
    dose_data = np.zeros(patient_data.density_image.shape[1:])
    beam_sequence = beam_sequence.to_delivery()
    # Create output directory if needed
    os.makedirs("out", exist_ok=True)
    iso_center_axial = dose_layer.iso_center_voxel[1:]
    
    # List to store frames
    frames = []
    
    # Loop through all control points
    for cp_idx in range(len(beam_sequence)):
        fig = plt.figure(figsize=(12, 9))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], hspace=0.15, wspace=0.05)
        ax_depth = fig.add_subplot(gs[0, :])  # Depth profile spans both columns
        ax1 = fig.add_subplot(gs[1, 0])  # Fluence map
        ax2 = fig.add_subplot(gs[1, 1])  # CT with dose overlay
        beam = beam_sequence[cp_idx]

        # Get dose and map for current control point
        with torch.no_grad():
            pred_depths, pred_map, _, pred_dose  = dose_layer.compute_dose(
                beam, 
                density_image=density_image,
                overwrite=True,
                return_intermediates=True
            )
        # pred_dose = torch.where(mask_external, pred_dose, torch.zeros_like(pred_dose))
        
        # Plot radiological depth profile
        central_profile = np.diff(pred_depths.cpu().detach().numpy()[0, :, 0])  # Adjust indexing as needed
        ax_depth.plot(central_profile, linewidth=2)
        ax_depth.set_ylim([0, 10.0])
        ax_depth.set_ylabel('Radiological Depth')
        ax_depth.set_title(f'Control Point {cp_idx + 1}/{num_cps} ({int(beam.gantry_angle_deg)} deg)', pad=5)
        ax_depth.grid(True, alpha=0.3)
        
        # Plot beam's eye view (fluence map) - make it square
        fluence_data = pred_map.cpu().detach().numpy()[0, :, :]
        w, h = fluence_data.shape
        im1 = ax1.imshow(fluence_data, interpolation='none', cmap='gray', vmin=0.0, vmax=1.0, aspect=h/w)
        ax1.set_title('Fluence Map', pad=5)
        ax1.axis('off')
        
        # Plot CT slice with dose overlay - already square
        pred_dose = pred_dose.cpu().detach().numpy()[0, slice_idx, :, :]
        dose_data += pred_dose
        
        ax2.imshow(ct_data, cmap='gray', vmin=-1000, vmax=1000, aspect='equal')
        ax2.imshow(dose_data, cmap=jet_alpha, vmin=0.0, vmax=dose_max, aspect='equal')
        ax2.plot(iso_center_axial[1], iso_center_axial[0], marker='o', color='red')
        overlay_mask_outline(pred_dose > 0.01 * pred_dose.max(), color='orange')
        
        # Add ROI contours
        for idx, struct_name in enumerate(patient_data.structures):
            if (struct_name == "FemoralHead_R"):
                continue
            roi = patient_data.structures[struct_name].cpu().detach().numpy()
            overlay_mask_outline(roi[slice_idx, :, :], 
                               color='white')
        
        ax2.set_title('Dose Overlay', pad=5)
        ax2.axis('off')
        ax2.set_aspect('equal', 'box')  # Force square aspect
        
        # Make the layout tight
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        # Save frame as image with tight bounding box
        frame_path = f"out/frame_{cp_idx:03d}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        
        # Read the saved image and add to frames list
        frame = cv2.imread(frame_path)
        if frame is not None:
            frames.append(frame)
        else:
            print(f"Failed to read frame image: {frame_path}")
        
        if os.path.exists(frame_path):
            os.remove(frame_path)
    print(f"The dose map produced a max of {dose_data.max()}")

    if frames:
        if (len(frames) != num_cps):
            print(f"Warning: Number of frames ({len(frames)}) does not match number of control points ({num_cps})")
        # Get dimensions from first frame
        height, width, layers = frames[0].shape
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10  # Frames per second (adjust as needed)
        if out_path is None:
            video_path = "out/animation.mp4"
        else:
            video_path = out_path
        
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Write all frames to video
        for frame in frames:
            video_writer.write(frame)
        
        # Release the video writer
        video_writer.release()
    else:
        print("Animation failed")

    if experiment is not None:
        experiment.log_video(video_path, overwrite=True)


def quick_plot(patient, dose_pred, title, show_ct: bool = False, out_path = None):
    dose_max = patient.number_of_fractions * max(patient.dose.max(), dose_pred.max()).item()
    dose_volume = patient.number_of_fractions * patient.dose.cpu().detach().numpy()
    ct_volume = patient._ct_tensor.cpu().detach().numpy()
    dose_pred = patient.number_of_fractions * dose_pred.cpu().detach().numpy()
    mae_max = 0.1 * dose_max
    alpha = 0.6 if show_ct else 1.0
    CoM = np.array(ndimage.measurements.center_of_mass(list(patient.structures.values())[0].cpu().detach().numpy()), dtype=np.int32)
    plt.figure()

    for axis in range(3):
        plot_idx = (axis * 3) + 1
        slice_idx = CoM[axis]
        plt.subplot(3, 3, plot_idx)
        if show_ct:
            plt.imshow(np.take(ct_volume, slice_idx, axis=axis), cmap='gray')
        plt.imshow(np.take(dose_volume, slice_idx, axis=axis), cmap='jet', vmax=dose_max, alpha=alpha)
        plt.axis('off')
        plt.colorbar()
        plt.subplot(3, 3, plot_idx + 1)
        plt.title(title)
        if show_ct:
            plt.imshow(np.take(ct_volume, slice_idx, axis=axis), cmap='gray')
        plt.imshow(np.take(dose_pred, slice_idx, axis=axis), cmap='jet', vmax=dose_max, alpha=alpha)
        plt.axis('off')
        plt.colorbar()
        plt.subplot(3, 3, plot_idx + 2)
        if show_ct:
            plt.imshow(np.take(ct_volume, slice_idx, axis=axis), cmap='gray')
        plt.imshow(np.take(dose_volume - dose_pred, slice_idx, axis=axis), cmap='coolwarm', vmin=-mae_max, vmax=mae_max, alpha=alpha)
        plt.axis('off')
        plt.colorbar()


    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)
        plt.close()
