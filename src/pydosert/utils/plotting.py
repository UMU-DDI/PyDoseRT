import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from skimage import measure
from matplotlib.colors import ListedColormap, PowerNorm
import os
from pathlib import Path
from scipy import ndimage
import cv2
from pydosert.data.beam import BeamSequence
from pydosert.engine.dose_engine import DoseEngine
from pydosert.data import Patient, OptimizationConfig
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D
from pydosert.geometry.rotations import rotate_2d_images

def overlay_mask_outline(mask_slice, color="red", linewidth=1, sigma=2.0):
    # Smooth the binary mask to produce clean contour boundaries
    smoothed = gaussian_filter(mask_slice.astype(float), sigma=sigma)

    for contour in measure.find_contours(smoothed, 0.5):
        plt.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth, linestyle=(0, (1, 2)))

def plot_overview(
    patient: Patient,
    dose_pred: torch.Tensor = None,
    treatment: OptimizationConfig = None,
    out_path=None,
    *,
    views=("axial", "sagittal"),
    isodose_percent_levels=(20, 40, 60, 80, 90, 95, 100, 105, 107, 110),
    cmap_dose="turbo",
):
    """Publication-style dose overview: stacked isodose panels and a DVH.

    Renders whichever doses are available, all in the same paper style:

    * only ``patient.dose``  -> a single reference-dose column
    * only ``dose_pred``     -> a single predicted-dose column
    * both                   -> reference and predicted columns plus an
                                error (prediction - reference) column

    Pass ``dose_pred=None`` to plot just the reference dose. ``views`` picks the
    orthogonal planes to stack (one row each) -- any one or two of ``"axial"``,
    ``"coronal"``, ``"sagittal"``; a single view yields larger panels.
    """

    if isinstance(views, str):
        views = (views,)
    valid_views = {"axial", "coronal", "sagittal"}
    if not 1 <= len(views) <= 2 or any(v not in valid_views for v in views):
        raise ValueError("views must be one or two of 'axial', 'coronal', 'sagittal'.")

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 19,
        "axes.labelsize": 16,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    if dose_pred is not None and dose_pred.ndim == 4:
        dose_pred = dose_pred[0]

    nf = patient.number_of_fractions
    ct = patient._ct_tensor.cpu().detach().numpy()
    gt = nf * patient.dose.cpu().detach().numpy() if patient.dose is not None else None
    pred = nf * dose_pred.cpu().detach().numpy() if dose_pred is not None else None
    if gt is None and pred is None:
        raise ValueError("plot_overview needs at least one of patient.dose or dose_pred.")

    present = [d for d in (gt, pred) if d is not None]
    prescribed = getattr(treatment, "prescription_gy", None)
    abs_max = float(max(np.max(d) for d in present))
    dose_max = float(prescribed) if prescribed else abs_max

    patient_structures = {
        name: mask.cpu().detach().numpy()
        for name, mask in patient.structures.items()
    }
    struct_cfgs = _structure_cfgs(patient, treatment)

    def _resolve_mask(struct_name):
        if struct_name in patient_structures:
            return patient_structures[struct_name]
        low = struct_name.lower()
        for key in patient_structures:
            key_low = key.lower()
            if low in key_low or key_low in low:
                return patient_structures[key]
        return None

    def _iter_plot_structures(skip_body=False):
        for struct_name, struct_cfg in struct_cfgs.items():
            low = struct_name.lower()
            if skip_body and ("body" in low or "external" in low):
                continue
            mask = _resolve_mask(struct_name)
            if mask is None:
                continue
            yield struct_name, struct_cfg, mask

    def _structure_color(struct_name, struct_cfg):
        low = struct_name.lower()
        if "ptv" in low:
            return "#d7191c"
        if "bladder" in low:
            return "#1b9e77"
        if "rectum" in low:
            return "#8c510a"
        if "femoralhead_l" in low or ("femoral" in low and "_l" in low):
            return "#6a3d9a"
        if "femoralhead_r" in low or ("femoral" in low and "_r" in low):
            return "#b57edc"
        return struct_cfg.get("color", "white")

    # --- Geometry: axial slice from the target, sagittal/z-extent from the dose.
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

    geom_dose = pred if pred is not None else gt
    com = np.array(ndimage.center_of_mass(reference_mask), dtype=np.int32)
    axial_z = int(np.clip(com[0], 0, ct.shape[0] - 1))
    sagittal_x = int(np.argmax(geom_dose.sum(axis=(0, 1))))

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

    dose_per_z = geom_dose.sum(axis=(1, 2))
    if dose_per_z.max() > 0:
        z_with_dose = np.where(dose_per_z > 0.01 * dose_per_z.max())[0]
        z0 = max(int(z_with_dose.min()) - 5, 0)
        z1 = min(int(z_with_dose.max()) + 6, ct.shape[0])
    else:
        z0 = max(int(com[0]) - 40, 0)
        z1 = min(int(com[0]) + 40, ct.shape[0])

    coronal_y = int(np.clip(np.argmax(geom_dose.sum(axis=(0, 2))), y0, y1 - 1))

    def _axial(vol):
        return vol[axial_z, y0:y1, x0:x1]

    def _coronal(vol):
        return np.flipud(vol[z0:z1, coronal_y, x0:x1])

    def _sagittal(vol):
        return np.flipud(vol[z0:z1, y0:y1, sagittal_x])

    selected_views = [
        {"axial": (_axial, "Axial"),
         "coronal": (_coronal, "Coronal"),
         "sagittal": (_sagittal, "Sagittal")}[v]
        for v in views
    ]
    n_rows = len(selected_views)

    # --- Discrete isodose colormap, shared by all dose columns.
    boundaries_pct = (0,) + tuple(isodose_percent_levels)
    boundaries_abs = [b / 100.0 * dose_max for b in boundaries_pct]
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

    # --- Columns to draw: one per available dose, plus an error column if both.
    image_cols = []
    if gt is not None:
        image_cols.append(("Reference TPS dose", gt, "dose"))
    if pred is not None:
        image_cols.append(("PyDoseRT dose", pred, "dose"))
    has_error = gt is not None and pred is not None
    err_levels = None
    if has_error:
        error = pred - gt
        err_abs = np.abs(error)
        err_vmax = float(np.percentile(err_abs[err_abs > 0], 99)) if np.any(err_abs > 0) else 1.0
        err_levels = np.linspace(-err_vmax, err_vmax, 21)
        image_cols.append(("Error (PyDoseRT - TPS)", error, "error"))

    n_img = len(image_cols)
    fig = plt.figure(figsize=(5.6 * n_img + 6.5, 4.6 * n_rows))
    # Columns: isodose legend | image columns | spacer | DVH.
    gs = gridspec.GridSpec(
        n_rows, n_img + 3, figure=fig,
        width_ratios=[0.5] + [1.0] * n_img + [0.35, 1.7],
        height_ratios=[1] * n_rows, wspace=0.12, hspace=0.16,
    )

    err_mappable = None
    err_axes = []
    for col, (col_title, data, kind) in enumerate(image_cols):
        for row, (view, view_name) in enumerate(selected_views):
            ax = fig.add_subplot(gs[row, col + 1])
            ax.imshow(view(ct), cmap="gray", interpolation="none", aspect="equal")
            if kind == "dose":
                ax.contourf(view(data), levels=boundaries_abs, cmap=cmap_disc, antialiased=True)
                ax.contour(view(data), levels=boundaries_abs, linewidths=0.7, colors="white", alpha=0.9)
            else:
                err_mappable = ax.contourf(view(data), levels=err_levels, cmap="coolwarm",
                                           antialiased=True, extend="both")
                err_axes.append(ax)
            for struct_name, struct_cfg, roi in _iter_plot_structures(skip_body=True):
                plt.sca(ax)
                overlay_mask_outline(view(roi), color=_structure_color(struct_name, struct_cfg),
                                     linewidth=2.0)
            if row == 0:
                ax.set_title(col_title)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(view_name)
            else:
                for spine in ax.spines.values():
                    spine.set_visible(False)

    # --- Isodose legend (far-left column).
    ax_leg = fig.add_subplot(gs[:, 0])
    ax_leg.axis("off")
    ax_leg.legend(handles=isodose_handles, title="Isodose levels", loc="center",
                  ncol=1, frameon=False, title_fontsize=14)

    # --- Error colorbar (only when both doses present); inset below the bottom
    #     error panel so it does not resize/misalign the image panels.
    if has_error and err_mappable is not None:
        cax = err_axes[-1].inset_axes([0.0, -0.12, 1.0, 0.04])
        cbar = fig.colorbar(err_mappable, cax=cax, orientation="horizontal")
        cbar.set_label("Prediction - reference (Gy)")

    # --- DVH: predicted solid, reference dashed (or a single solid set).
    ax_dvh = fig.add_subplot(gs[:, n_img + 2])
    dvh_upper = max(dose_max, abs_max)
    bins = np.linspace(0, dvh_upper, 1000)

    def _plot_dvh(dose_vol, linestyle, with_label):
        for struct_name, struct_cfg, roi in _iter_plot_structures(skip_body=False):
            vals = dose_vol[roi > 0.0]
            if vals.size == 0:
                continue
            hist, edges = np.histogram(vals, bins=bins, density=False)
            cum = np.cumsum(hist[::-1])[::-1]
            cum = cum / cum.max() if cum.max() > 0 else cum
            ax_dvh.plot(edges[:-1], cum, linestyle=linestyle,
                        label=struct_name if with_label else None,
                        color=_structure_color(struct_name, struct_cfg), linewidth=2.0)

    if has_error:
        _plot_dvh(pred, "solid", True)
        _plot_dvh(gt, "dashed", False)
    else:
        _plot_dvh(present[0], "solid", True)

    ax_dvh.set_xlabel("Dose (Gy)")
    ax_dvh.set_ylabel("Volume Fraction")
    ax_dvh.set_title("Dose Volume Histogram (DVH)")
    ax_dvh.set_xlim(0.0, dvh_upper * 1.03 if dvh_upper > 0 else 1.0)
    ax_dvh.set_ylim(0.0, 1.05)
    ax_dvh.grid(True, linestyle=":", linewidth=0.7)
    struct_legend = ax_dvh.legend(loc="lower left", frameon=False)
    if has_error:
        ax_dvh.add_artist(struct_legend)
        ax_dvh.legend(
            handles=[
                Line2D([0], [0], color="k", linestyle="solid", label="Prediction"),
                Line2D([0], [0], color="k", linestyle="dashed", label="Reference"),
            ],
            loc="upper right", frameon=False,
        )

    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

def plot_comparison(
    patient: Patient,
    dose_pred: torch.Tensor,
    treatment: OptimizationConfig = None,
    out_path=None,
    isodose_percent_levels=(20, 40, 60, 80, 90, 95, 100, 105, 107, 110),
    cmap_dose="turbo",
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
    for struct_name, struct_cfg in _structure_cfgs(patient, treatment).items():
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

    # Zoom the profile x-axis to where dose is present (union of all profiles).
    profile_stack = np.vstack([lateral_ref, lateral_pred, ap_ref, ap_pred])
    sig = np.where(profile_stack.max(axis=0) > 0.05 * profile_stack.max())[0]
    if sig.size > 0:
        x_start = max(0, int(sig.min()) - 5)
        x_end = min(profile_len - 1, int(sig.max()) + 5)
    else:
        x_start, x_end = 0, profile_len - 1
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

def plot_animation(patient: Patient,
                   dose_engine: DoseEngine,
                   beam_sequence: BeamSequence,
                   out_path=None,
                   dose_max=50.0):
    """
    Modified version with tight square layout - two squares stacked vertically
    """
    patient_data = patient
    dose_layer = dose_engine
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
        ax2.plot(iso_center_axial[1], iso_center_axial[0], marker='o', color='red', markersize=4)

        # Beam geometry for this control point: a diverging fan from the source
        # through the field edges, centred on the isocenter. Purely geometric
        # (gantry angle + SID + jaw width), not tied to the computed dose.
        theta = np.deg2rad(beam.gantry_angle_deg)
        sy = dose_layer.dose_grid_spacing[1]
        sx = dose_layer.dose_grid_spacing[2]
        sid = float(dose_layer.SID)
        iso_row, iso_col = iso_center_axial[0], iso_center_axial[1]
        u = np.array([np.sin(theta), -np.cos(theta)])       # iso -> source (x, y)
        perp = np.array([np.cos(theta), np.sin(theta)])     # in-plane field axis
        jl, ju = float(beam.jaw_positions[0]), float(beam.jaw_positions[1])
        if abs(ju - jl) < 1.0:                               # fall back to field size
            half = float(dose_layer.field_size[0]) / 2.0
            jl, ju = -half, half

        def _to_vox(mm_x, mm_y):
            return iso_col + mm_x / sx, iso_row + mm_y / sy

        src_x, src_y = _to_vox(sid * u[0], sid * u[1])
        far_pts = []
        for edge in (jl, ju):
            ex, ey = _to_vox(edge * perp[0], edge * perp[1])
            fx = src_x + (ex - src_x) * 1.8                 # extend past the isocenter
            fy = src_y + (ey - src_y) * 1.8
            far_pts.append((fx, fy))
            ax2.plot([src_x, fx], [src_y, fy], color='deepskyblue', linewidth=1.3, alpha=0.9)
        ax2.fill([src_x, far_pts[0][0], far_pts[1][0]],
                 [src_y, far_pts[0][1], far_pts[1][1]],
                 color='deepskyblue', alpha=0.12, linewidth=0)

        # Add ROI contours
        for idx, struct_name in enumerate(patient_data.structures):
            if (struct_name == "FemoralHead_R"):
                continue
            roi = patient_data.structures[struct_name].cpu().detach().numpy()
            overlay_mask_outline(roi[slice_idx, :, :],
                               color='white')

        # Keep the view on the CT (the source sits far outside the image).
        ax2.set_xlim(0, ct_data.shape[1])
        ax2.set_ylim(ct_data.shape[0], 0)
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


# ---------------------------------------------------------------------------
# Unified, plug-and-play plots
#
# The helpers below take PyDoseRT base classes directly (BeamSequence, Patient,
# OptimizationConfig) and keep no internal state, so they can be dropped into
# any workflow.  Monitor units are always rendered through ``_draw_mu_polar`` so
# every MU plot looks identical, and DVH curves always come from
# ``compute_dvh_curves`` so the maths is shared.
# ---------------------------------------------------------------------------

_ARC_COLORS = ["steelblue", "darkorange", "seagreen", "crimson"]


def _draw_mu_polar(ax_polar, mu_vals, angles_rad, arc_sizes=None,
                   title="Monitor Units"):
    """Draw an MU-vs-gantry-angle polar plot onto an existing polar axis.

    This is the single source of truth for MU rendering, shared by
    :func:`plot_mu_polar` and :func:`plot_fluence_and_mu`.  ``arc_sizes``
    (e.g. ``[179, 179]``) splits the control points into separate arcs, each
    drawn as its own MU curve; if ``None`` all points are treated as one arc.
    """
    def _one_arc(a, m, color, label=None):
        order = np.argsort(a)
        sa, sm = a[order], m[order]
        half = (sa[1] - sa[0]) / 2 if len(sa) > 1 else np.pi
        pa, pm = [], []
        for i in range(len(sa)):
            pa.extend([sa[i] - half, sa[i] + half])
            pm.extend([sm[i], sm[i]])
        pa.append(pa[0] + 2 * np.pi)
        pm.append(pm[0])
        ax_polar.fill(pa, pm, alpha=0.18, color=color)
        ax_polar.plot(pa, pm, color=color, linewidth=1.0, label=label)

    if arc_sizes and len(arc_sizes) > 1:
        s = 0
        for k, sz in enumerate(arc_sizes):
            _one_arc(angles_rad[s:s + sz], mu_vals[s:s + sz],
                     _ARC_COLORS[k % len(_ARC_COLORS)], label=f"arc {k + 1}")
            s += sz
        ax_polar.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.25, 1.1))
    else:
        _one_arc(angles_rad, mu_vals, _ARC_COLORS[0])

    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_title(title, pad=20, fontsize=13)


def _mu_and_angles(beam_sequence: BeamSequence):
    """Pull MU values and gantry angles (radians) out of a ``BeamSequence``."""
    mu_vals = beam_sequence.mus.cpu().detach().numpy()
    if beam_sequence.gantry_angles is not None:
        angles_rad = beam_sequence.gantry_angles.cpu().detach().numpy()
    else:
        angles_rad = np.linspace(0, 2 * np.pi, len(beam_sequence), endpoint=False)
    return mu_vals, angles_rad


def _save_or_show(fig, out_path, dpi=150):
    if out_path is None:
        plt.show()
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=dpi)
        plt.close(fig)


def _patient_arrays(patient: Patient, dose_pred: torch.Tensor):
    """Common numpy arrays shared by the dose plots.

    Returns ``(ct, dose_ref, dose_pred, struct_masks)`` where the doses are in
    absolute Gy (scaled by the number of fractions) and ``dose_ref`` is ``None``
    when the patient has no reference dose.
    """
    if dose_pred.ndim == 4:
        dose_pred = dose_pred[0]
    nf = patient.number_of_fractions
    ct = patient._ct_tensor.cpu().detach().numpy()
    dose_calc = nf * dose_pred.cpu().detach().numpy()
    dose_ref = nf * patient.dose.cpu().detach().numpy() if patient.dose is not None else None
    masks = {n: m.cpu().detach().numpy() for n, m in patient.structures.items()}
    return ct, dose_ref, dose_calc, masks


def _structure_cfgs(patient: Patient, treatment) -> dict:
    """``{name: {"color": ...}}`` plotting config for each structure.

    Uses the ``OptimizationConfig`` structures when given; otherwise falls back
    to the patient's structures, colouring them from the default matplotlib
    cycle so plots still work without a treatment.
    """
    if treatment is not None:
        return treatment.structures
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["white"])
    return {name: {"color": cycle[i % len(cycle)]}
            for i, name in enumerate(patient.structures)}


def _structure_colors(patient, treatment) -> dict:
    """``{structure_name: color}`` for the DVH-style plots."""
    return {n: cfg.get("color") for n, cfg in _structure_cfgs(patient, treatment).items()}


def _dose_vmax(treatment, *dose_arrays) -> float:
    """Dose colour-scale ceiling: the prescription if set, else the data max."""
    presc = getattr(treatment, "prescription_gy", None) if treatment is not None else None
    if presc:
        return float(presc)
    arrays = [a for a in dose_arrays if a is not None]
    return float(max(a.max() for a in arrays)) if arrays else 1.0


def plot_mu_polar(beam_sequence: BeamSequence, out_path=None, arc_sizes=None):
    """Polar Monitor-Unit plot for a ``BeamSequence`` (one curve per arc)."""
    mu_vals, angles_rad = _mu_and_angles(beam_sequence)
    fig = plt.figure(figsize=(6, 6))
    ax_polar = fig.add_subplot(111, projection="polar")
    _draw_mu_polar(ax_polar, mu_vals, angles_rad, arc_sizes=arc_sizes)
    plt.tight_layout()
    _save_or_show(fig, out_path)


def compute_fluence_maps(dose_engine, beam_sequence: BeamSequence, chunk_size=10):
    """Fluence maps ``[num_beams, H, W]`` for a ``BeamSequence``.

    Computed in beam-sized chunks and detached to the CPU, so large arcs do not
    blow up GPU memory the way a single ``compute_dose(..., return_intermediates=True)``
    pass would (that path is not chunked). The collimator (beam limiting device)
    rotation is applied per beam just as the engine does internally, so these
    match the fluence maps seen in the dose pipeline. Ready to hand to
    :func:`plot_fluence_and_mu`.
    """
    dose_engine._initialize_layers(beam_sequence, False)
    leaf = beam_sequence.leaf_positions.unsqueeze(0)  # [1, G, N, 2]
    jaw = beam_sequence.jaw_positions.unsqueeze(0)     # [1, G, 2]
    collimator_angles = dose_engine.collimator_angles
    maps = []
    with torch.no_grad():
        for s in range(0, len(beam_sequence), chunk_size):
            e = min(s + chunk_size, len(beam_sequence))
            fm = dose_engine.fluence_map_layer(leaf[:, s:e], jaw[:, s:e])
            if collimator_angles is not None and (collimator_angles[s:e] != 0.0).any():
                fm = rotate_2d_images(fm, collimator_angles[s:e],
                                      device=dose_engine.device, dtype=dose_engine.dtype)
            maps.append(fm.detach().cpu())
    return torch.cat(maps, dim=0)


def plot_fluence_and_mu(beam_sequence: BeamSequence, fluence_maps,
                        out_path=None, arc_sizes=None):
    """Fluence-map montage beside the polar MU plot.

    ``fluence_maps`` is a ``[num_beams, H, W]`` tensor or array, e.g. the second
    element returned by ``DoseEngine.compute_dose(..., return_intermediates=True)``.
    The MU panel is rendered with the same routine as :func:`plot_mu_polar`.
    """
    if isinstance(fluence_maps, torch.Tensor):
        fluence_maps = fluence_maps.cpu().detach().numpy()

    mu_vals, angles_rad = _mu_and_angles(beam_sequence)
    angles_deg = np.rad2deg(angles_rad)
    num_beams = fluence_maps.shape[0]

    fm_cols = 10
    fm_rows = int(np.ceil(num_beams / fm_cols))

    fig = plt.figure(figsize=(2 + 1.5 * fm_cols, max(8, 1.5 * fm_rows)))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 2])

    ax_polar = fig.add_subplot(gs[0], projection="polar")
    _draw_mu_polar(ax_polar, mu_vals, angles_rad, arc_sizes=arc_sizes)

    gs_inner = gs[1].subgridspec(fm_rows, fm_cols, wspace=0.05, hspace=0.3)
    vmax = float(fluence_maps.max())
    for i in range(num_beams):
        r, c = divmod(i, fm_cols)
        ax = fig.add_subplot(gs_inner[r, c])
        ax.imshow(fluence_maps[i], cmap="gray", vmin=0, vmax=vmax, aspect="equal")
        ax.set_title(f"{angles_deg[i]:.0f}°", fontsize=7)
        ax.axis("off")
    for i in range(num_beams, fm_rows * fm_cols):
        r, c = divmod(i, fm_cols)
        fig.add_subplot(gs_inner[r, c]).axis("off")

    plt.tight_layout()
    _save_or_show(fig, out_path)


def compute_dvh_curves(dose_3d, struct_masks, dose_vmax=50.0, n_bins=1000):
    """Cumulative DVH curves for each structure.

    Returns ``{struct_name: (dose_axis, volume_fraction)}`` where ``dose_axis``
    is in Gy and ``volume_fraction`` is the fraction of the structure's voxels
    receiving at least that dose.  Shared by every DVH plot.
    """
    bins = np.linspace(0, dose_vmax, n_bins)
    curves = {}
    for struct_name, mask_np in struct_masks.items():
        roi = mask_np > 0.0
        vals = dose_3d[roi]
        if vals.size == 0:
            continue
        hist, bin_edges = np.histogram(vals, bins=bins, density=False)
        cum = np.cumsum(hist[::-1])[::-1]
        cum_norm = cum / cum.max() if cum.max() > 0 else cum
        curves[struct_name] = (bin_edges[:-1], cum_norm)
    return curves


def plot_dvh(patient: Patient, dose_pred: torch.Tensor, treatment=None,
             out_path=None, dose_vmax=None):
    """Cumulative DVH: predicted dose (solid) vs reference TPS dose (dashed).

    Structure colours are taken from ``treatment.structures`` when an
    ``OptimizationConfig`` is supplied, otherwise matplotlib defaults are used.
    """
    _, dose_ref, dose_calc, struct_masks = _patient_arrays(patient, dose_pred)
    colors = _structure_colors(patient, treatment)
    if dose_vmax is None:
        dose_vmax = _dose_vmax(treatment, dose_calc, dose_ref)

    pred_curves = compute_dvh_curves(dose_calc, struct_masks, dose_vmax=dose_vmax)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (dose_axis, vol) in pred_curves.items():
        ax.plot(dose_axis, vol, linestyle="solid", label=name,
                color=colors.get(name))
    if dose_ref is not None:
        for name, (dose_axis, vol) in compute_dvh_curves(
                dose_ref, struct_masks, dose_vmax=dose_vmax).items():
            ax.plot(dose_axis, vol, linestyle="dashed", color=colors.get(name))

    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Volume Fraction")
    ax.set_title("Dose Volume Histogram (DVH)")
    ax.set_xlim(0.0, dose_vmax * 1.03 if dose_vmax > 0 else 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend(loc="lower left", frameon=False)
    plt.tight_layout()
    _save_or_show(fig, out_path)


def plot_profiles(patient: Patient, dose_pred: torch.Tensor = None, treatment=None,
                  out_path=None, *, depth_axis=1, n_depths=3):
    """Depth-dose (PDD) curve and lateral dose profiles, in the overview style.

    The depth-dose curve runs along ``depth_axis`` (0=z, 1=y, 2=x) through the
    maximum-dose point; lateral profiles are sampled at ``n_depths`` depths
    spanning the irradiated region. The prediction is drawn solid and the
    reference (``patient.dose``) dashed; pass ``dose_pred=None`` for the
    reference only.
    """
    plt.rcParams.update({
        "font.size": 16, "axes.titlesize": 19, "axes.labelsize": 16,
        "legend.fontsize": 12, "xtick.labelsize": 13, "ytick.labelsize": 13,
    })

    if dose_pred is not None and dose_pred.ndim == 4:
        dose_pred = dose_pred[0]
    nf = patient.number_of_fractions
    gt = nf * patient.dose.cpu().detach().numpy() if patient.dose is not None else None
    pred = nf * dose_pred.cpu().detach().numpy() if dose_pred is not None else None
    if gt is None and pred is None:
        raise ValueError("plot_profiles needs at least one of patient.dose or dose_pred.")
    geom = pred if pred is not None else gt

    res = patient.resolution
    remaining = [a for a in (0, 1, 2) if a != depth_axis]
    lateral_axis, fixed_axis = remaining[-1], remaining[0]
    center = np.unravel_index(int(np.argmax(geom)), geom.shape)

    def _pdd(vol):
        idx = [slice(None)] * 3
        idx[lateral_axis] = center[lateral_axis]
        idx[fixed_axis] = center[fixed_axis]
        return vol[tuple(idx)]

    def _lateral(vol, depth_idx):
        idx = [slice(None)] * 3
        idx[depth_axis] = depth_idx
        idx[fixed_axis] = center[fixed_axis]
        return vol[tuple(idx)]

    pdd_geom = _pdd(geom)
    sig = np.where(pdd_geom > 0.05 * pdd_geom.max())[0]
    lo, hi = (int(sig.min()), int(sig.max())) if sig.size else (0, len(pdd_geom) - 1)
    depth_idxs = np.unique(np.linspace(lo, hi, n_depths).astype(int))

    ref_color, pred_color = "#ff006e", "#11a5ff"
    fig, (ax_pdd, ax_lat) = plt.subplots(1, 2, figsize=(15, 6))

    depth_mm = np.arange(geom.shape[depth_axis]) * res[depth_axis]
    if pred is not None:
        ax_pdd.plot(depth_mm, _pdd(pred), color=pred_color, linewidth=2.3, label="PyDoseRT")
    if gt is not None:
        ax_pdd.plot(depth_mm, _pdd(gt), color=ref_color, linewidth=2.3,
                    linestyle="--", label="Reference")
    ax_pdd.set_xlabel("Depth (mm)")
    ax_pdd.set_ylabel("Dose (Gy)")
    ax_pdd.set_title("Depth-dose profile")
    ax_pdd.set_xlim(depth_mm[lo], depth_mm[hi])
    ax_pdd.set_ylim(bottom=0.0)
    ax_pdd.grid(True, linestyle=":", linewidth=0.7)
    ax_pdd.legend(frameon=False)

    lat_mm = (np.arange(geom.shape[lateral_axis]) - center[lateral_axis]) * res[lateral_axis]
    depth_colors = plt.get_cmap("turbo")(np.linspace(0.1, 0.9, len(depth_idxs)))
    for color, d in zip(depth_colors, depth_idxs):
        if pred is not None:
            ax_lat.plot(lat_mm, _lateral(pred, d), color=color, linewidth=2.0,
                        label=f"{depth_mm[d]:.0f} mm")
        if gt is not None:
            ax_lat.plot(lat_mm, _lateral(gt, d), color=color, linewidth=2.0, linestyle="--",
                        label=None if pred is not None else f"{depth_mm[d]:.0f} mm")
    lat_sig = np.where(_lateral(geom, depth_idxs[len(depth_idxs) // 2]) > 0.05 * pdd_geom.max())[0]
    if lat_sig.size:
        ax_lat.set_xlim(lat_mm[lat_sig.min()], lat_mm[lat_sig.max()])
    ax_lat.set_xlabel("Lateral offset (mm)")
    ax_lat.set_ylabel("Dose (Gy)")
    ax_lat.set_title("Lateral profiles")
    ax_lat.set_ylim(bottom=0.0)
    ax_lat.grid(True, linestyle=":", linewidth=0.7)
    ax_lat.legend(frameon=False, title="Depth")

    plt.tight_layout()
    _save_or_show(fig, out_path)


def plot_kernel(kernels, depths_mm=None, out_path=None, *, n_panels=12, cmap="turbo"):
    """Pencil-beam kernel montage with central lateral profiles.

    ``kernels`` is a ``[n_depths, K, K]`` array/tensor. The montage shows the
    kernel at ``n_panels`` depths (power-normalised so the low-dose spread is
    visible) and the side panel overlays the central lateral profile at each of
    those depths. ``depths_mm`` labels the panels (voxel index if omitted).
    """
    if isinstance(kernels, torch.Tensor):
        kernels = kernels.cpu().detach().numpy()
    n_depths = kernels.shape[0]
    depths_mm = np.arange(n_depths) if depths_mm is None else np.asarray(depths_mm)

    panel_idxs = np.unique(np.linspace(0, n_depths - 1, n_panels).astype(int))
    n_panels = len(panel_idxs)
    cols = 4
    rows = int(np.ceil(n_panels / cols))

    fig = plt.figure(figsize=(3.0 * cols + 6.0, 2.6 * rows + 1))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[cols * 0.75, 1.3], wspace=0.18)
    gs_m = gs[0].subgridspec(rows, cols, wspace=0.08, hspace=0.25)

    vmax = float(kernels.max())
    norm = PowerNorm(gamma=0.4, vmin=0.0, vmax=vmax)
    kc = kernels.shape[1] // 2
    for i, d in enumerate(panel_idxs):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs_m[r, c])
        ax.imshow(kernels[d], cmap=cmap, norm=norm, aspect="equal")
        ax.set_title(f"{depths_mm[d]:.0f} mm", fontsize=10)
        ax.axis("off")
    for i in range(n_panels, rows * cols):
        r, c = divmod(i, cols)
        fig.add_subplot(gs_m[r, c]).axis("off")

    ax_prof = fig.add_subplot(gs[1])
    lat = np.arange(kernels.shape[2]) - kc
    prof_colors = plt.get_cmap(cmap)(np.linspace(0.1, 0.9, n_panels))
    for color, d in zip(prof_colors, panel_idxs):
        ax_prof.plot(lat, kernels[d, kc, :], color=color, linewidth=1.6,
                     label=f"{depths_mm[d]:.0f} mm")
    ax_prof.set_yscale("log")
    ax_prof.set_ylim(vmax * 1e-4, vmax * 1.2)
    ax_prof.set_xlabel("Lateral offset (voxels)")
    ax_prof.set_ylabel("Kernel value")
    ax_prof.set_title("Central lateral profiles")
    ax_prof.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax_prof.legend(fontsize=8, frameon=False, title="Depth", ncol=2)

    _save_or_show(fig, out_path)
