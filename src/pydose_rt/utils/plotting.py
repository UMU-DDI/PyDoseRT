import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from skimage import measure
from matplotlib.colors import ListedColormap
import os
import cv2
from pydose_rt.engine.dose_engine import DoseEngine
from pydose_rt.data import MachineConfig, TreatmentConfig, Patient


def overlay_mask_outline(mask_slice, color="red", linewidth=1):
    for contour in measure.find_contours(mask_slice, 0.5):
        plt.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)

def print_results(
    experiment,
    treatment,
    raw_losses,
    y_dose,
    pred_mlc,
    pred_mus,
    pred_jaws,
    pred_mlc_grads,
    pred_jaws_grads,
    pred_mus_grads,
    best_results,
    dose_pred,
    true_ct,
    masks,
    mae_loss,
    plot_ct=True,
    dose_max=10.0,
    preset="umea"
):
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
    scale_mlc  = float(np.max(np.abs(pred_mlc_grads)))  if np.any(pred_mlc_grads)  else 1.0
    scale_jaws = float(np.max(np.abs(pred_jaws_grads))) if np.any(pred_jaws_grads) else 1.0
    scale_mus  = float(np.max(np.abs(pred_mus_grads)))  if np.any(pred_mus_grads)  else 1.0

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
        f"MAE - {str(mae_loss)} Gy\n"
        f"Test #{len(best_results)}: {[str(np.round(v, 4)) for v in raw_losses]}",
        y=0.995
    )

    # --- 1) Jaws (centers)
    ax = fig.add_subplot(gs[0])
    ax.set_title('Jaws (lower)')
    _imshow_fullwidth(
        ax,
        pred_jaws.cpu().detach().numpy()[0, 0:1, :],
        cmap='gray', vmin=-200.0, vmax=200.0
    )
    if (pred_jaws_grads is not None):
        _imshow_fullwidth(
            ax,
            pred_jaws_grads[0, 0:1, :],
            cmap='coolwarm', vmin=-scale_jaws, vmax=scale_jaws, alpha=alpha
        )

    # --- 2) Jaws (widths)
    ax = fig.add_subplot(gs[1])
    ax.set_title('Jaws (higher)')
    _imshow_fullwidth(
        ax,
        pred_jaws.cpu().detach().numpy()[0, 1:2, :],
        cmap='gray', vmin=-200.0, vmax=200.0
    )
    if (pred_jaws_grads is not None):
        _imshow_fullwidth(
            ax,
            pred_jaws_grads[0, 1:2, :],
            cmap='coolwarm', vmin=-scale_jaws, vmax=scale_jaws, alpha=alpha
        )

    # --- 3) MLCs (centers)
    ax = fig.add_subplot(gs[2])
    ax.set_title('MLCs (left)')
    _imshow_fullwidth(
        ax,
        np.transpose(pred_mlc.cpu().detach().numpy()[0, 0, :, :]),
        cmap='gray', vmin=-200.0, vmax=200.0
    )
    if (pred_mlc_grads is not None):
        _imshow_fullwidth(
            ax,
            np.transpose(pred_mlc_grads[0, 0, :, :]),
            cmap='coolwarm', vmin=-scale_mlc, vmax=scale_mlc, alpha=alpha
        )

    # --- 4) MLCs (widths)
    ax = fig.add_subplot(gs[3])
    ax.set_title('MLCs (right)')
    _imshow_fullwidth(
        ax,
        np.transpose(pred_mlc.cpu().detach().numpy()[0, 1, :, :]),
        cmap='gray', vmin=-200.0, vmax=200.0
    )
    if (pred_mlc_grads is not None):
        _imshow_fullwidth(
            ax,
            np.transpose(pred_mlc_grads[0, 1, :, :]),
            cmap='coolwarm', vmin=-scale_mlc, vmax=scale_mlc, alpha=alpha
        )

    # --- 5) MUs
    ax = fig.add_subplot(gs[4])
    ax.set_title('MUs')
    _imshow_fullwidth(
        ax,
        pred_mus.cpu().detach().numpy(),
        cmap='gray', vmin=0.0, vmax=None
    )
    if (pred_mus_grads is not None):
        _imshow_fullwidth(
            ax,
            pred_mus_grads,
            cmap='coolwarm', vmin=-scale_mus, vmax=scale_mus, alpha=alpha
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
    else:
        raise Exception("Preset missing")

    # If overlay_mask_outline expects already-sliced 2D arrays (as in your original code),
    # use these two helpers instead:
    def _dose_slice_axial(arr, z=44, x_start=0, x_end=256):
        return arr[0, z, x_start:x_end, :]

    def _dose_slice_coronal(arr, x=128, y_start=0, y_end=256, z_start=0, z_end=256):
        # coronal view, transpose to show (z, y) or (y, z) consistently
        # matching your original "np.transpose(...[0, 64:198, 128, :])"
        return np.flipud(arr[0, z_start:z_end, y_start:y_end,x])

    # --- 6) Dose distribution (pred, axial)
    ax = fig.add_subplot(gs[5])
    _imshow_fullwidth(ax, _dose_slice_axial(dose_pred.cpu().detach().numpy(), z=axial_z, x_start=axial_xstart, x_end=axial_xend), cmap='jet', vmin=0.0, vmax=dose_max)
    _hide_ticks(ax)
    ax.set_title('Dose distribution (pred, axial)')
    for idx, color in enumerate([structure.color for structure in treatment.structures][:-1]):
        roi = masks[idx]
        overlay_mask_outline(roi.cpu().detach().numpy()[0, axial_z, axial_xstart:axial_xend, :], color=color)

    # --- 7) Dose distribution (pred, sagittal)
    ax = fig.add_subplot(gs[6])
    _imshow_fullwidth(ax, _dose_slice_coronal(dose_pred.cpu().detach().numpy(), x=coronal_x, y_start=coronal_ystart, y_end=coronal_yend, z_start=coronal_zstart, z_end=coronal_zend), cmap='jet', vmin=0.0, vmax=dose_max)
    _hide_ticks(ax)
    ax.set_title('Dose distribution (pred, coronal)')
    for idx, color in enumerate([structure.color for structure in treatment.structures][:-1]):
        roi = masks[idx]
        overlay_mask_outline(np.flipud(roi.cpu().detach().numpy()[0, coronal_zstart:coronal_zend, coronal_ystart:coronal_yend, coronal_x]), color=color)

    # --- 8) Dose distribution (gt, axial)
    ax = fig.add_subplot(gs[7])
    if plot_ct:
        _imshow_fullwidth(ax, _dose_slice_axial(true_ct.cpu().detach().numpy(), z=axial_z, x_start=axial_xstart, x_end=axial_xend), cmap='gray')
    _imshow_fullwidth(ax, _dose_slice_axial(y_dose.cpu().detach().numpy(), z=axial_z, x_start=axial_xstart, x_end=axial_xend), cmap='jet', vmin=0.0, vmax=dose_max, alpha=dose_alpha)
    _hide_ticks(ax)
    ax.set_title('Dose distribution (gt, axial)')
    for idx, color in enumerate([structure.color for structure in treatment.structures][:-1]):
        roi = masks[idx]
        overlay_mask_outline(roi.cpu().detach().numpy()[0, axial_z, axial_xstart:axial_xend, :], color=color)

    # --- 9) Dose distribution (gt, sagittal)
    ax = fig.add_subplot(gs[8])
    if plot_ct:
        _imshow_fullwidth(ax, _dose_slice_coronal(y_dose.cpu().detach().numpy(), x=coronal_x, y_start=coronal_ystart, y_end=coronal_yend, z_start=coronal_zstart, z_end=coronal_zend), cmap='gray')
    _imshow_fullwidth(ax, _dose_slice_coronal(y_dose.cpu().detach().numpy(), x=coronal_x, y_start=coronal_ystart, y_end=coronal_yend, z_start=coronal_zstart, z_end=coronal_zend), cmap='jet', vmin=0.0, vmax=dose_max, alpha=dose_alpha)
    _hide_ticks(ax)
    ax.set_title('Dose distribution (gt, coronal)')
    for idx, color in enumerate([structure.color for structure in treatment.structures][:-1]):
        roi = masks[idx]
        overlay_mask_outline(np.flipud(roi.cpu().detach().numpy()[0, coronal_zstart:coronal_zend, coronal_ystart:coronal_yend, coronal_x]), color=color)

    # --- 10) DVH (line plot; same panel height as others for uniformity)
    ax = fig.add_subplot(gs[9])
    for idx, (color, roi_name) in enumerate([(structure.color, structure.name) for structure in treatment.structures]):
        roi = masks[idx]
        dose_values = dose_pred[roi > 0.0].cpu().detach().numpy()
        if dose_values.size == 0:
            continue
        bins = np.linspace(0, dose_max, 1000)
        hist, bin_edges = np.histogram(dose_values, bins=bins, density=False)
        cumulative_hist = np.cumsum(hist[::-1])[::-1]
        cumulative_hist_normalized = np.divide(cumulative_hist, cumulative_hist.max())
        ax.plot(bin_edges[:-1], cumulative_hist_normalized, linestyle="solid", label=roi_name, color=color)

    for idx, color in enumerate([structure.color for structure in treatment.structures]):
        roi = masks[idx]
        dose_values = y_dose[roi > 0.0].cpu().detach().numpy()
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
    save_path = "out/figure.png"
    plt.savefig(save_path, dpi=150)
    if (experiment is not None):
        experiment.log_figure(save_path, overwrite=True)
    plt.close()

def make_animation(experiment, treatment: TreatmentConfig, machine_config: MachineConfig, patient_data: Patient, dose_layer: DoseEngine, pred_mlc, pred_mus, pred_jaws, dose_max=50.0):
    """
    Modified version with tight square layout - two squares stacked vertically
    """
    mask_external = torch.tensor(np.expand_dims(list(patient_data.structures.values())[-1], 0), dtype=treatment.dtype, device=treatment.device) > 0
    ct_volume = torch.tensor(np.expand_dims(patient_data.ct_array, 0), dtype=treatment.dtype, device=treatment.device)

    # Get the base colormap (jet)
    alpha_max = 1.0
    jet = plt.get_cmap('jet', 256)
    colors = jet(np.linspace(0, 1, 256))
    values = np.linspace(0, 1, 256)
    alpha = np.clip(np.interp(values, [0, 1], [0.0, alpha_max]), 0, alpha_max)
    colors[:, -1] = alpha
    jet_alpha = ListedColormap(colors)
    dose_layer.eval()
    num_cps = treatment.number_of_cps
    slice_idx = patient_data.ct_array.shape[0] // 2
    ct_data = ct_volume.cpu().detach().numpy()[0, slice_idx, :, :]
    dose_data = np.zeros(patient_data.ct_array.shape[1:])
    
    # Create output directory if needed
    os.makedirs("out", exist_ok=True)
    
    # List to store frames
    frames = []
    
    # Loop through all control points
    for cp_idx in range(num_cps):
        fig = plt.figure(figsize=(12, 9))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], hspace=0.15, wspace=0.05)
        ax_depth = fig.add_subplot(gs[0, :])  # Depth profile spans both columns
        ax1 = fig.add_subplot(gs[1, 0])  # Fluence map
        ax2 = fig.add_subplot(gs[1, 1])  # CT with dose overlay
        
        # Get dose and map for current control point
        with torch.no_grad():
            pred_dose, pred_map, pred_depths = dose_layer(
                pred_mlc, 
                pred_mus, 
                jaw_positions=pred_jaws, 
                ct_image=ct_volume,
                jaw_x=7.0,
                jaw_y=-8.5,
                single_cp=cp_idx
            )
        # pred_dose = torch.where(mask_external, pred_dose, torch.zeros_like(pred_dose))
        
        # Plot radiological depth profile
        central_profile = np.diff(pred_depths.cpu().detach().numpy()[0, :, 0])  # Adjust indexing as needed
        ax_depth.plot(central_profile, linewidth=2)
        ax_depth.set_ylim([0, 10.0])
        ax_depth.set_ylabel('Radiological Depth')
        ax_depth.set_title(f'Control Point {cp_idx + 1}/{num_cps}', pad=5)
        ax_depth.grid(True, alpha=0.3)
        
        # Plot beam's eye view (fluence map) - make it square
        fluence_data = pred_map.cpu().detach().numpy()[0, 0, :, :]
        w, h = fluence_data.shape
        im1 = ax1.imshow(fluence_data, interpolation='none', cmap='gray', vmin=0.0, vmax=1.0, aspect=h/w)
        ax1.set_title('Fluence Map', pad=5)
        ax1.axis('off')
        
        # Plot CT slice with dose overlay - already square
        pred_dose = pred_dose.cpu().detach().numpy()[0, slice_idx, :, :]
        dose_data += pred_dose
        
        ax2.imshow(ct_data, cmap='gray', vmin=-1000, vmax=1000, aspect='equal')
        ax2.imshow(dose_data, cmap=jet_alpha, vmin=0.0, vmax=dose_max, aspect='equal')
        overlay_mask_outline(pred_dose > 0.01 * pred_dose.max(), color='orange')
        
        # Add ROI contours
        for idx, struct_name in enumerate(patient_data.structures):
            if (struct_name == "FemoralHead_R"):
                continue
            roi = patient_data.structures[struct_name]
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
        
        # if os.path.exists(frame_path):
        #     os.remove(frame_path)


    if frames:
        if (len(frames) != num_cps):
            print(f"Warning: Number of frames ({len(frames)}) does not match number of control points ({num_cps})")
        # Get dimensions from first frame
        height, width, layers = frames[0].shape
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10  # Frames per second (adjust as needed)
        video_path = "out/animation.mp4"
        
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
    return
