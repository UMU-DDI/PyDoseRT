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
from pydose_rt.data.beam import BeamSequence
from pydose_rt.engine.dose_engine import DoseEngine
from pydose_rt.data import Patient, OptimizationConfig
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
):
    """Updated plotting routine for publication-ready figures.

    Minimal changes from the original function but with a few cleanups:
      - avoids deprecated ndimage.measurements
      - adds isodose contours (percent levels by default)
      - uses a cleaner colormap and a shared colorbar for dose panels
      - small style tweaks (font sizes, line widths) for publication

    Parameters
    ----------
    experiment, treatment, patient, dose_pred, out_path
        same meaning as in the original function
    dose_alpha : float
        alpha for overlaying dose prediction over the background CT
    isodose_percent_levels : sequence of int
        isodose percent levels to draw on the dose panels (e.g. [95,80,50,...])
    cmap_dose : str
        matplotlib colormap used for dose wash
    """

    # --- style tweaks for publication
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })

    # compute a consistent dose max across predicted and reference dose
    dose_max = float(max(patient.dose.max(), dose_pred.max()).item())

    def _hide_ticks(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(bottom=False, left=False)

    def _imshow_fullwidth(ax, img, *, cmap='gray', vmin=None, vmax=None, alpha=1.0):
        """
        Show any array so it fills the axes horizontally and uses a fixed panel height.
        Keeping data coordinates unchanged ensures overlays (contours) stay aligned.
        """
        im = ax.imshow(
            img,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation='none',
            aspect='auto',
            alpha=alpha,
        )
        _hide_ticks(ax)
        return im

    # Figure + GridSpec: two narrow image columns and a wider DVH column
    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 2], wspace=0.25)

    # compute center of mass safely (avoid deprecated measurements namespace)
    CoM = np.array(ndimage.center_of_mass(list(patient.structures.values())[0].cpu().detach().numpy()), dtype=np.int32)
    axial_z = CoM[0]
    axial_xstart = max(CoM[2] - 64, 0)
    axial_xend = CoM[2] + 64
    axial_ystart = max(CoM[1] - 64, 0)
    axial_yend = CoM[1] + 64
    coronal_x = CoM[2]
    coronal_zstart = max(CoM[0] - 40, 0)
    coronal_zend = CoM[0] + 40
    coronal_ystart = max(CoM[1] - 40, 0)
    coronal_yend = CoM[1] + 40

    def _dose_slice_axial(arr, z=44, y_start=0, y_end=256, x_start=0, x_end=256):
        return arr[z, y_start:y_end, x_start:x_end]

    def _dose_slice_coronal(arr, x=128, y_start=0, y_end=256, z_start=0, z_end=256):
        return np.flipud(arr[z_start:z_end, y_start:y_end, x])

    # draw axial panel (CT background + dose wash + isodose contours + structure outlines)
    ax_axial = fig.add_subplot(gs[0])
    ax_axial.set_aspect('equal')
    ct_axial = _dose_slice_axial(patient._ct_tensor.cpu().detach().numpy(), z=axial_z, y_start=axial_ystart, y_end=axial_yend, x_start=axial_xstart, x_end=axial_xend)
    dose_axial = _dose_slice_axial(dose_pred.cpu().detach().numpy(), z=axial_z, y_start=axial_ystart, y_end=axial_yend, x_start=axial_xstart, x_end=axial_xend)

    _imshow_fullwidth(ax_axial, ct_axial, cmap='gray')

    # ---- ROI outlines ----
    for idx, color in enumerate([struct["color"] for struct_name, struct in treatment.structures.items()][:-1]):
        if len(patient.structures) <= idx:
            continue
        roi = list(patient.structures.values())[idx]
        overlay_mask_outline(
            roi.cpu().detach().numpy()[axial_z, axial_ystart:axial_yend, axial_xstart:axial_xend],
            color=color,
            linewidth=2.0
        )

    # ---- Discrete isodose levels (0%,10%,20%,...,90% for example) ----
    boundaries_pct = (0,) + isodose_percent_levels                # e.g. (0,10,20,...)
    boundaries_abs = [b/100.0 * dose_max for b in boundaries_pct]

    # ---- Progressive alpha: 0.0 → 1.0 ----
    alphas = np.linspace(0.0, 1.0, len(boundaries_pct))

    # ---- Build a colormap with (r,g,b,alpha) per band ----
    n_colors = len(boundaries_pct) - 1
    base_cmap = plt.get_cmap(cmap_dose)
    rgb_colors = base_cmap(np.linspace(0, 1, n_colors))[:, :3]    # strip old alpha

    rgba_colors = [(r, g, b, a) for (r, g, b), a in zip(rgb_colors, alphas[1:])]
    cmap_disc = ListedColormap(rgba_colors)

    # ---- Isodose legend handles (percent-based) ----
    isodose_handles = [
        Line2D(
            [0], [0],
            color=rgba_colors[i][:3],   # RGB only (legend ignores alpha well)
            linewidth=3,
            label=f"{isodose_percent_levels[i]}%"
        )
        for i in range(len(isodose_percent_levels))
    ]

    # ---- Filled isodose bands (transparent → opaque) ----
    ax_axial.contourf(
        dose_axial,
        levels=boundaries_abs,
        cmap=cmap_disc,
        antialiased=True
    )

    # ---- Thin white outlines between bands ----
    ax_axial.contour(
        dose_axial,
        levels=boundaries_abs,
        linewidths=0.6,
        colors='white'
    )

    ax_axial.set_title('PyDoseRT Optimized — axial')
    ax_axial.legend(
        handles=isodose_handles,
        title="Isodose levels",
        loc="lower left",
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )
    # coronal / sagittal panel
    ax_cor = fig.add_subplot(gs[1])
    ax_cor.set_aspect('equal')
    ct_cor = _dose_slice_coronal(patient._ct_tensor.cpu().detach().numpy(), x=coronal_x, y_start=coronal_ystart, y_end=coronal_yend, z_start=coronal_zstart, z_end=coronal_zend)
    dose_cor = _dose_slice_coronal(dose_pred.cpu().detach().numpy(), x=coronal_x, y_start=coronal_ystart, y_end=coronal_yend, z_start=coronal_zstart, z_end=coronal_zend)

    _imshow_fullwidth(ax_cor, ct_cor, cmap='gray')

    # ---- ROI outlines ----
    for idx, color in enumerate([struct["color"] for struct_name, struct in treatment.structures.items()][:-1]):
        if len(patient.structures) <= idx:
            continue
        roi = list(patient.structures.values())[idx]
        overlay_mask_outline(
            np.flipud(roi.cpu().detach().numpy()[coronal_zstart:coronal_zend, coronal_ystart:coronal_yend, coronal_x]),
            color=color,
            linewidth=2.0
        )

    # ---- Discrete isodose levels (0%, 10%, ..., etc.) ----
    boundaries_pct = (0,) + isodose_percent_levels
    boundaries_abs = [b/100.0 * dose_max for b in boundaries_pct]

    # ---- Progressive alpha: 0.0 → 1.0 ----
    alphas = np.linspace(0.0, 1.0, len(boundaries_pct))

    # ---- Build RGBA colormap with band-wise alpha ----
    n_colors = len(boundaries_pct) - 1
    base_cmap = plt.get_cmap(cmap_dose)
    rgb_colors = base_cmap(np.linspace(0, 1, n_colors))[:, :3]  # drop existing alpha
    rgba_colors = [(r, g, b, a) for (r, g, b), a in zip(rgb_colors, alphas[1:])]
    cmap_disc = ListedColormap(rgba_colors)

    # ---- Filled isodose bands (transparent → opaque) ----
    ax_cor.contourf(
        dose_cor,
        levels=boundaries_abs,
        cmap=cmap_disc,
        antialiased=True
    )

    # ---- Thin white boundaries ----
    ax_cor.contour(
        dose_cor,
        levels=boundaries_abs,
        linewidths=0.6,
        colors='white'
    )

    ax_cor.set_title('PyDoseRT Optimized — sagittal')
    ax_cor.legend(
        handles=isodose_handles,
        title="Isodose levels",
        loc="lower left",
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )

    # DVH panel
    ax = fig.add_subplot(gs[2])
    for idx, (struct_name, struct) in enumerate(treatment.structures.items()):
        if len(patient.structures) <= idx:
            continue
        color = struct["color"]
        roi = list(patient.structures.values())[idx]
        dose_values = dose_pred[roi > 0.0].cpu().detach().numpy()
        if dose_values.size == 0:
            continue
        bins = np.linspace(0, dose_max, 1000)
        hist, bin_edges = np.histogram(dose_values, bins=bins, density=False)
        cumulative_hist = np.cumsum(hist[::-1])[::-1]
        cumulative_hist_normalized = cumulative_hist / cumulative_hist.max()
        ax.plot(bin_edges[:-1], cumulative_hist_normalized, linestyle='solid', label=struct_name, color=color, linewidth=1.25)

    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Volume Fraction")
    ax.set_title("Dose Volume Histogram (DVH)")
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.legend(loc="lower left", frameon=False)

    # Layout & save
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if out_path is None:
        if experiment is not None:
            save_path = "out/paper.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            experiment.log_figure(save_path, overwrite=True)
            plt.close(fig)
        else:
            plt.show()
    else:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
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
):
    """Updated plotting routine for publication-ready figures.

    Minimal changes from the original function but with a few cleanups:
      - avoids deprecated ndimage.measurements
      - adds isodose contours (percent levels by default)
      - uses a cleaner colormap and a shared colorbar for dose panels
      - small style tweaks (font sizes, line widths) for publication

    Parameters
    ----------
    experiment, treatment, patient, dose_pred, out_path
        same meaning as in the original function
    dose_alpha : float
        alpha for overlaying dose prediction over the background CT
    isodose_percent_levels : sequence of int
        isodose percent levels to draw on the dose panels (e.g. [95,80,50,...])
    cmap_dose : str
        matplotlib colormap used for dose wash
    """

    # --- style tweaks for publication
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })

    # compute a consistent dose max across predicted and reference dose
    dose_max = treatment.prescription_gy

    def _hide_ticks(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(bottom=False, left=False)

    def _imshow_fullwidth(ax, img, *, cmap='gray', vmin=None, vmax=None, alpha=1.0):
        """
        Show any array so it fills the axes horizontally and uses a fixed panel height.
        Keeping data coordinates unchanged ensures overlays (contours) stay aligned.
        """
        im = ax.imshow(
            img,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation='none',
            aspect='auto',
            alpha=alpha,
        )
        _hide_ticks(ax)
        return im

    # Figure + GridSpec: two narrow image columns and a wider DVH column
    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 2], wspace=0.25)

    # compute center of mass safely (avoid deprecated measurements namespace)
    CoM = np.array(ndimage.center_of_mass(list(patient.structures.values())[0].cpu().detach().numpy()), dtype=np.int32)
    axial_z = CoM[0]
    axial_xstart = max(CoM[2] - 64, 0)
    axial_xend = CoM[2] + 64
    axial_ystart = max(CoM[1] - 64, 0)
    axial_yend = CoM[1] + 64

    def _dose_slice_axial(arr, z=44, y_start=0, y_end=256, x_start=0, x_end=256):
        return arr[z, y_start:y_end, x_start:x_end]


    # draw axial panel (CT background + dose wash + isodose contours + structure outlines)
    ax_axial = fig.add_subplot(gs[0])
    ax_axial.set_aspect('equal')
    ct_axial = _dose_slice_axial(patient._ct_tensor.cpu().detach().numpy(), z=axial_z, y_start=axial_ystart, y_end=axial_yend, x_start=axial_xstart, x_end=axial_xend)
    dose_axial = _dose_slice_axial(patient.number_of_fractions * patient.dose.cpu().detach().numpy(), z=axial_z, y_start=axial_ystart, y_end=axial_yend, x_start=axial_xstart, x_end=axial_xend)

    _imshow_fullwidth(ax_axial, ct_axial, cmap='gray')

    # ---- ROI outlines ----
    for idx, color in enumerate([struct["color"] for struct_name, struct in treatment.structures.items()][:-1]):
        if len(patient.structures) <= idx:
            continue
        roi = list(patient.structures.values())[idx]
        overlay_mask_outline(
            roi.cpu().detach().numpy()[axial_z, axial_ystart:axial_yend, axial_xstart:axial_xend],
            color=color,
            linewidth=2.0
        )

    # ---- Discrete isodose levels (0%,10%,20%,...,90% for example) ----
    boundaries_pct = (0,) + isodose_percent_levels                # e.g. (0,10,20,...)
    boundaries_abs = [b/100.0 * dose_max for b in boundaries_pct]

    # ---- Progressive alpha: 0.0 → 1.0 ----
    alphas = np.linspace(0.0, 1.0, len(boundaries_pct))

    # ---- Build a colormap with (r,g,b,alpha) per band ----
    n_colors = len(boundaries_pct) - 1
    base_cmap = plt.get_cmap(cmap_dose)
    rgb_colors = base_cmap(np.linspace(0, 1, n_colors))[:, :3]    # strip old alpha

    rgba_colors = [(r, g, b, a) for (r, g, b), a in zip(rgb_colors, alphas[1:])]
    cmap_disc = ListedColormap(rgba_colors)

    # ---- Isodose legend handles (percent-based) ----
    isodose_handles = [
        Line2D(
            [0], [0],
            color=rgba_colors[i][:3],   # RGB only (legend ignores alpha well)
            linewidth=3,
            label=f"{isodose_percent_levels[i]}%"
        )
        for i in range(len(isodose_percent_levels))
    ]

    # ---- Filled isodose bands (transparent → opaque) ----
    ax_axial.contourf(
        dose_axial,
        levels=boundaries_abs,
        cmap=cmap_disc,
        antialiased=True
    )

    # ---- Thin white outlines between bands ----
    ax_axial.contour(
        dose_axial,
        levels=boundaries_abs,
        linewidths=0.6,
        colors='white'
    )

    ax_axial.set_title('Reference TPS dose — axial')
    leg = ax_axial.legend(
        handles=isodose_handles,
        title="Isodose levels",
        loc="lower left",
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )
    for text in leg.get_texts():
        text.set_color("white")
    # coronal / sagittal panel

    ax_axial = fig.add_subplot(gs[1])
    ax_axial.set_aspect('equal')
    ct_axial = _dose_slice_axial(patient._ct_tensor.cpu().detach().numpy(), z=axial_z, y_start=axial_ystart, y_end=axial_yend, x_start=axial_xstart, x_end=axial_xend)
    dose_axial = _dose_slice_axial(patient.number_of_fractions * dose_pred.cpu().detach().numpy(), z=axial_z, y_start=axial_ystart, y_end=axial_yend, x_start=axial_xstart, x_end=axial_xend)

    _imshow_fullwidth(ax_axial, ct_axial, cmap='gray')

    # ---- ROI outlines ----
    for idx, color in enumerate([struct["color"] for struct_name, struct in treatment.structures.items()][:-1]):
        if len(patient.structures) <= idx:
            continue
        roi = list(patient.structures.values())[idx]
        overlay_mask_outline(
            roi.cpu().detach().numpy()[axial_z, axial_ystart:axial_yend, axial_xstart:axial_xend],
            color=color,
            linewidth=2.0
        )

    # ---- Discrete isodose levels (0%,10%,20%,...,90% for example) ----
    boundaries_pct = (0,) + isodose_percent_levels                # e.g. (0,10,20,...)
    boundaries_abs = [b/100.0 * dose_max for b in boundaries_pct]

    # ---- Progressive alpha: 0.0 → 1.0 ----
    alphas = np.linspace(0.0, 1.0, len(boundaries_pct))

    # ---- Build a colormap with (r,g,b,alpha) per band ----
    n_colors = len(boundaries_pct) - 1
    base_cmap = plt.get_cmap(cmap_dose)
    rgb_colors = base_cmap(np.linspace(0, 1, n_colors))[:, :3]    # strip old alpha

    rgba_colors = [(r, g, b, a) for (r, g, b), a in zip(rgb_colors, alphas[1:])]
    cmap_disc = ListedColormap(rgba_colors)

    # ---- Isodose legend handles (percent-based) ----
    isodose_handles = [
        Line2D(
            [0], [0],
            color=rgba_colors[i][:3],   # RGB only (legend ignores alpha well)
            linewidth=3,
            label=f"{isodose_percent_levels[i]}%"
        )
        for i in range(len(isodose_percent_levels))
    ]

    # ---- Filled isodose bands (transparent → opaque) ----
    ax_axial.contourf(
        dose_axial,
        levels=boundaries_abs,
        cmap=cmap_disc,
        antialiased=True
    )

    # ---- Thin white outlines between bands ----
    ax_axial.contour(
        dose_axial,
        levels=boundaries_abs,
        linewidths=0.6,
        colors='white'
    )

    ax_axial.set_title('PyDoseRT result — axial')
    leg = ax_axial.legend(
        handles=isodose_handles,
        title="Isodose levels",
        loc="lower left",
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )
    for text in leg.get_texts():
        text.set_color("white")

    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=gs[0, 2],
        height_ratios=[1, 1],
        hspace=0.25
    )

    y_slice = axial_ystart + (axial_yend - axial_ystart) // 2
    x_slice = axial_xstart + (axial_xend - axial_xstart) // 2
    ax_right_top = fig.add_subplot(gs_right[0, 0])
    ax_right_top.plot(patient.number_of_fractions * dose_pred.cpu().detach().numpy()[axial_z, y_slice, :], linestyle='solid', color='orange', label="PyDoseRT")
    ax_right_top.plot(patient.number_of_fractions * patient.dose.cpu().detach().numpy()[axial_z, y_slice, :], linestyle='dashed', color='blue', label="Reference")
    ax_right_top.set_title("Lateral Dose Profile")
    ax_right_top.set_ylabel("Dose (Gy)")
    ax_right_top.grid(True, linestyle=':', linewidth=0.5)
    ax_right_top.legend(loc="upper left", frameon=False)
    # ax_right_top.plot(bin_edges[:-1], cumulative_hist_normalized, linestyle='solid', label=struct_name, color=color, linewidth=1.25)
    ax_right_bottom = fig.add_subplot(gs_right[1, 0])
    ax_right_bottom.plot(patient.number_of_fractions * dose_pred.cpu().detach().numpy()[axial_z, :, x_slice], linestyle='solid', color='orange', label="PyDoseRT")
    ax_right_bottom.plot(patient.number_of_fractions * patient.dose.cpu().detach().numpy()[axial_z, :, x_slice], linestyle='dashed', color='blue', label="Reference")
    ax_right_bottom.set_title("Anterior–Posterior Dose Profile")
    ax_right_bottom.set_ylabel("Dose (Gy)")
    ax_right_bottom.grid(True, linestyle=':', linewidth=0.5)
    ax_right_bottom.legend(loc="upper left", frameon=False)

    # Layout & save
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if out_path is not None:
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
    preset="umea",
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
    elif (preset == "umea"):
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