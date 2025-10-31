from comet_ml import Experiment
import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colormaps
import os
import torch
import time
import math

from matplotlib import cm
from matplotlib.colors import ListedColormap
import torch
import torch.nn.functional as F
from pydose_rt import ModelConfig
from pydose_rt import DoseEngine
from pydose_rt.layers import ValidParametersLayer
from pydose_rt.engine.utils.plotting import *
from pydose_rt.utils.kernel import *
from pydose_rt.engine.utils.grad_monitor import GradMonitor
from torch.utils.data import DataLoader  # PyTorch DataLoader
from pydose_rt.engine.data_augment import DataGenerator
from pydose_rt.engine.config import config as PARAMS
import numpy as np
from skimage import measure
from pydose_rt.engine.losses import dose_loss, leafs_loss, mus_loss, jaws_loss, result_validation
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_initial_weights():
    min_int_range = -3
    max_int_range = 2
    weights = {
        "loss_lower_bound_gy": 1.0, # 10**np.random.randint(min_int_range, max_int_range),
        "loss_higher_bound_gy": 1.0, #10**np.random.randint(min_int_range, max_int_range),
        "loss_lower_bound_target": 0.0, # 10**np.random.randint(min_int_range, max_int_range),
        "loss_higher_bound_target": 0.0, # 10**np.random.randint(min_int_range, max_int_range),
        "l2_loss_oars_and_background": 10**np.random.randint(-3, 1),
        "mu_rate_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "mu_complexity_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "leaf_reg_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "leaf_complexity_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(-2, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "jaw_opening_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "jaw_complexity_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
    }
    
    return weights

def get_example_data(data_path="/media/bolo/Datasets/converted_lund/"):
    gen = DataGenerator(data_path, 
                        "training", 
                        None, 
                        None, 
                        shuffle=False, 
                        batch_size=1, 
                        downsampling_factor=(1,1,1), 
                        constraints=PARAMS.constraints_lund_probe,
                        )
    
    val_loader = DataLoader(
        gen,
        batch_size=gen.batch_size,  # Use same batch size or 1 for validation
        shuffle=gen.shuffle,
        num_workers=0,
        pin_memory=True,
    )

    for i, batch_data in enumerate(val_loader):
        x, y_dose, masks, region_weights, constraints_batch = batch_data
        break

    ct_volume = (1000.0 * x[:, 0, ...]).to(device)  # scale to HU

    config = ModelConfig(preset="lund-probe", number_of_cps=240, downsampling_factor=(1,4,4))
    valid_parameters_layer = ValidParametersLayer(config, leafs_centered=True)

    mask_target = masks[0, 0, ...].expand(1, -1, -1, -1).clone().detach().to(device) > 0
    mask_external = masks.sum(1).clone().detach().to(device) > 0
    mask_oar = torch.sum(masks[0, 1:-1, ...], 0).expand(1, -1, -1, -1).clone().detach().to(device) > 0
    dose_target = y_dose.expand(1, -1, -1, -1).clone().detach().to(device)
    masks_torch = []
    for i in range(masks.shape[1]):
        masks_torch.append(masks[0, i, ...].expand(1, -1, -1, -1))
    x = x.to(config.device)
    y_dose = y_dose.to(config.device)
    y_dose = torch.where(mask_external, y_dose, torch.zeros_like(y_dose))
    masks = masks.to(config.device)
    region_weights = region_weights.to(config.device)

    current_res = [np.inf]
    weights = get_initial_weights()
    latest = {"raw_losses": None, "loss_val": None, "dose_pred": None}

    pred_mlc_init = torch.ones((1, 2, config.number_of_cps, config.number_of_leaf_pairs), dtype=torch.float32, device=device)
    pred_mlc_init[:, 0, :, :] = 0.5
    pred_mlc_init[:, 1, :, :] = 0.0
    pred_mlc = pred_mlc_init.clone().detach().requires_grad_(True)
    pred_jaws_init = torch.zeros((1, 2, config.number_of_cps), dtype=torch.float32, device=device)
    pred_jaws_init[:, 0, :] = 0.5
    pred_jaws = pred_jaws_init.clone().detach().requires_grad_(True)
    pred_mus_init = (100.0 / config.number_of_cps) * torch.ones((1, config.number_of_cps), dtype=torch.float32, device=device)
    pred_mus = pred_mus_init.clone().detach().requires_grad_(True)
    return x, y_dose, masks, region_weights, constraints_batch, ct_volume, config, valid_parameters_layer, mask_target, mask_external, mask_oar, dose_target, current_res, weights, latest, pred_mlc, pred_jaws, pred_mus, masks_torch

def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(
            min_lr / optimizer.defaults["lr"],
            0.5 * (1.0 + math.cos(math.pi * progress)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def overlay_mask_outline(mask_slice, color="red", linewidth=1):
    for contour in measure.find_contours(mask_slice, 0.5):
        plt.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)

def scale_loss(loss, weight):
    return loss * weight

def print_results(
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
    plot_ct=True
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
    dose_max = 50.0
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
    ax.set_title('Jaws (centers)')
    _imshow_fullwidth(
        ax,
        pred_jaws.cpu().detach().numpy()[0, 0:1, :],
        cmap='gray', vmin=0.0, vmax=1.0
    )
    _imshow_fullwidth(
        ax,
        pred_jaws_grads[0, 0:1, :],
        cmap='coolwarm', vmin=-scale_jaws, vmax=scale_jaws, alpha=alpha
    )

    # --- 2) Jaws (widths)
    ax = fig.add_subplot(gs[1])
    ax.set_title('Jaws (widths)')
    _imshow_fullwidth(
        ax,
        pred_jaws.cpu().detach().numpy()[0, 1:2, :],
        cmap='gray', vmin=0.0, vmax=1.0
    )
    _imshow_fullwidth(
        ax,
        pred_jaws_grads[0, 1:2, :],
        cmap='coolwarm', vmin=-scale_jaws, vmax=scale_jaws, alpha=alpha
    )

    # --- 3) MLCs (centers)
    ax = fig.add_subplot(gs[2])
    ax.set_title('MLCs (centers)')
    _imshow_fullwidth(
        ax,
        np.transpose(pred_mlc.cpu().detach().numpy()[0, 0, :, :]),
        cmap='gray', vmin=0.0, vmax=1.0
    )
    _imshow_fullwidth(
        ax,
        np.transpose(pred_mlc_grads[0, 0, :, :]),
        cmap='coolwarm', vmin=-scale_mlc, vmax=scale_mlc, alpha=alpha
    )

    # --- 4) MLCs (widths)
    ax = fig.add_subplot(gs[3])
    ax.set_title('MLCs (widths)')
    _imshow_fullwidth(
        ax,
        np.transpose(pred_mlc.cpu().detach().numpy()[0, 1, :, :]),
        cmap='gray', vmin=0.0, vmax=1.0
    )
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
    _imshow_fullwidth(
        ax,
        pred_mus_grads,
        cmap='coolwarm', vmin=-scale_mus, vmax=scale_mus, alpha=alpha
    )

    axial_z = 49
    axial_xstart = 64
    axial_xend = 192
    coronal_x = 128
    coronal_ystart = 32
    coronal_yend = 224
    coronal_zstart = 24
    coronal_zend = 72
    # If overlay_mask_outline expects already-sliced 2D arrays (as in your original code),
    # use these two helpers instead:
    def _dose_slice_axial(arr, z=44, x_start=0, x_end=256):
        return arr[0, x_start:x_end, :, z]

    def _dose_slice_coronal(arr, x=128, y_start=0, y_end=256, z_start=0, z_end=256):
        # coronal view, transpose to show (z, y) or (y, z) consistently
        # matching your original "np.transpose(...[0, 64:198, 128, :])"
        return np.transpose(arr[0, x, y_start:y_end, z_start:z_end])

    # --- 6) Dose distribution (pred, axial)
    ax = fig.add_subplot(gs[5])
    _imshow_fullwidth(ax, _dose_slice_axial(dose_pred.cpu().detach().numpy(), z=axial_z, x_start=axial_xstart, x_end=axial_xend), cmap='jet', vmin=0.0, vmax=dose_max)
    _hide_ticks(ax)
    ax.set_title('Dose distribution (pred, axial)')
    for idx, (key, value) in enumerate(list(PARAMS.structure_names.items())[:-1]):
        roi = masks[idx]
        overlay_mask_outline(roi.cpu().numpy()[0, axial_xstart:axial_xend, :, axial_z], color=PARAMS.roi_colors[key])

    # --- 7) Dose distribution (pred, sagittal)
    ax = fig.add_subplot(gs[6])
    _imshow_fullwidth(ax, _dose_slice_coronal(dose_pred.cpu().detach().numpy(), x=coronal_x, y_start=coronal_ystart, y_end=coronal_yend, z_start=coronal_zstart, z_end=coronal_zend), cmap='jet', vmin=0.0, vmax=dose_max)
    _hide_ticks(ax)
    ax.set_title('Dose distribution (pred, coronal)')
    for idx, (key, value) in enumerate(list(PARAMS.structure_names.items())[:-1]):
        roi = masks[idx]
        overlay_mask_outline(np.transpose(roi.cpu().numpy()[0, coronal_x, coronal_ystart:coronal_yend, coronal_zstart:coronal_zend]), color=PARAMS.roi_colors[key])

    # --- 8) Dose distribution (gt, axial)
    ax = fig.add_subplot(gs[7])
    if plot_ct:
        _imshow_fullwidth(ax, _dose_slice_axial(true_ct.cpu().detach().numpy(), z=axial_z, x_start=axial_xstart, x_end=axial_xend), cmap='gray')
    _imshow_fullwidth(ax, _dose_slice_axial(y_dose.cpu().detach().numpy(), z=axial_z, x_start=axial_xstart, x_end=axial_xend), cmap='jet', vmin=0.0, vmax=dose_max, alpha=dose_alpha)
    _hide_ticks(ax)
    ax.set_title('Dose distribution (gt, axial)')
    for idx, (key, value) in enumerate(list(PARAMS.structure_names.items())[:-1]):
        roi = masks[idx]
        overlay_mask_outline(roi.cpu().numpy()[0, axial_xstart:axial_xend, :, axial_z], color=PARAMS.roi_colors[key])

    # --- 9) Dose distribution (gt, sagittal)
    ax = fig.add_subplot(gs[8])
    if plot_ct:
        _imshow_fullwidth(ax, _dose_slice_coronal(y_dose.cpu().detach().numpy(), x=coronal_x, y_start=coronal_ystart, y_end=coronal_yend, z_start=coronal_zstart, z_end=coronal_zend), cmap='gray')
    _imshow_fullwidth(ax, _dose_slice_coronal(y_dose.cpu().detach().numpy(), x=coronal_x, y_start=coronal_ystart, y_end=coronal_yend, z_start=coronal_zstart, z_end=coronal_zend), cmap='jet', vmin=0.0, vmax=dose_max, alpha=dose_alpha)
    _hide_ticks(ax)
    ax.set_title('Dose distribution (gt, coronal)')
    for idx, (key, value) in enumerate(list(PARAMS.structure_names.items())[:-1]):
        roi = masks[idx]
        overlay_mask_outline(np.transpose(roi.cpu().numpy()[0, coronal_x, coronal_ystart:coronal_yend, coronal_zstart:coronal_zend]), color=PARAMS.roi_colors[key])

    # --- 10) DVH (line plot; same panel height as others for uniformity)
    ax = fig.add_subplot(gs[9])
    for idx, (key, value) in enumerate(PARAMS.structure_names.items()):
        roi_name = value
        roi = masks[idx]
        dose_values = dose_pred[roi > 0.0].cpu().detach().numpy()
        if dose_values.size == 0:
            continue
        bins = np.linspace(0, dose_max, 1000)
        hist, bin_edges = np.histogram(dose_values, bins=bins, density=False)
        cumulative_hist = np.cumsum(hist[::-1])[::-1]
        cumulative_hist_normalized = np.divide(cumulative_hist, cumulative_hist.max())
        ax.plot(bin_edges[:-1], cumulative_hist_normalized, linestyle="solid", label=roi_name, color=PARAMS.roi_colors[key])

    for idx, (key, value) in enumerate(PARAMS.structure_names.items()):
        roi_name = value
        roi = masks[idx]
        dose_values = y_dose[roi > 0.0].cpu().detach().numpy()
        if dose_values.size == 0:
            continue
        bins = np.linspace(0, dose_max, 1000)
        hist, bin_edges = np.histogram(dose_values, bins=bins, density=False)
        cumulative_hist = np.cumsum(hist[::-1])[::-1]
        cumulative_hist_normalized = np.divide(cumulative_hist, cumulative_hist.max())
        ax.plot(bin_edges[:-1], cumulative_hist_normalized, linestyle="dashed", color=PARAMS.roi_colors[key])

    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Volume Fraction")
    ax.set_title("Dose Volume Histogram (DVH)")
    ax.grid(True)
    ax.legend(loc="lower left")

    # Layout & save
    fig.tight_layout(rect=[0, 0, 1, 0.97])  # keep space for the suptitle
    save_path = "out/figure.png"
    plt.savefig(save_path, dpi=150)
    experiment.log_figure(save_path, overwrite=True)
    plt.close()

def make_animation(dose_layer: DoseEngine, pred_mlc, pred_mus, pred_jaws, ct_volume, masks):
    """
    Modified version with tight square layout - two squares stacked vertically
    """
    slice_idx = 49
    dose_max = 50.0
    

    # Get the base colormap (jet)
    alpha_max = 0.6
    jet = plt.get_cmap('jet', 256)
    colors = jet(np.linspace(0, 1, 256))
    values = np.linspace(0, 100, 256)
    alpha = np.clip(np.interp(values, [0, dose_max], [0.0, alpha_max]), 0, alpha_max)
    colors[:, -1] = alpha
    jet_alpha = ListedColormap(colors)
    dose_layer.eval()
    num_cps = config.number_of_cps
    ct_data = ct_volume.cpu().detach().numpy()[0, :, :, slice_idx]
    dose_data = np.zeros((256, 256))
    
    # Create output directory if needed
    os.makedirs("out", exist_ok=True)
    
    # List to store frames
    frames = []
    
    # Loop through all control points
    for cp_idx in range(num_cps):
        # Create figure with tight layout
        # Using equal aspect ratio for both subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), 
                                        gridspec_kw={'hspace': 0.02})  # Minimal vertical spacing
        
        # Get dose and map for current control point
        with torch.no_grad():
            pred_dose, pred_map = dose_layer(
                pred_mlc, 
                pred_mus, 
                jaw_positions=pred_jaws, 
                ct_image=ct_volume, 
                single_cp=cp_idx
            )
        pred_dose = torch.where(mask_external, pred_dose, torch.zeros_like(pred_dose))
        
        # Plot beam's eye view (fluence map) - make it square
        fluence_data = np.transpose(pred_map.cpu().detach().numpy()[0, :, :, 0])
        w, h = fluence_data.shape
        im1 = ax1.imshow(fluence_data, interpolation='none', cmap='gray', vmin=0.0, vmax=1.0, aspect=h/w)
        ax1.set_title(f'Control Point {cp_idx + 1}/{num_cps}', pad=5)
        ax1.axis('off')
        
        # Plot CT slice with dose overlay - already square
        dose_data += pred_dose.cpu().detach().numpy()[0, :, :, slice_idx]
        
        ax2.imshow(ct_data, cmap='gray', vmin=-1000, vmax=1000, aspect='equal')
        ax2.imshow(dose_data, cmap=jet_alpha, vmin=0.0, vmax=dose_max, aspect='equal')
        
        # Add ROI contours
        for idx, (key, value) in enumerate(list(PARAMS.structure_names.items())[:-1]):
            roi = masks[idx]
            overlay_mask_outline(roi.cpu().numpy()[0, :, :, slice_idx], 
                               color=PARAMS.roi_colors[key])
        
        ax2.axis('off')
        ax2.set_aspect('equal', 'box')  # Force square aspect
        
        # Make the layout very tight
        plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.01, hspace=0.02)
        
        # Save frame as image with tight bounding box
        frame_path = f"out/frame_{cp_idx:03d}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches='tight', pad_inches=0.02)
        
        # Read the saved image and add to frames list
        frame = cv2.imread(frame_path)
        if frame is not None:
            frames.append(frame)
        
        plt.close(fig)
        if os.path.exists(frame_path):
            os.remove(frame_path)

    
    if frames:
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

    experiment.log_video(video_path, overwrite=True)
    return

def compute_claude_loss(dose_pred, dose_true, pred_mus, leafs, pred_jaws, weights, _masks):
    """
    Simplified loss function for sharp dose distributions.
    Uses high-order penalties and gradient-based edge sharpening.
    """
    # Unpack masks - assuming _masks contains [ptv, oars..., external]
    mask_ptv = _masks[0] > 0.5
    mask_oars = torch.stack(_masks[1:-1], dim=0).sum(0) > 0.5 if len(_masks) > 2 else torch.zeros_like(mask_ptv)
    mask_external = _masks[-1] > 0.5
    mask_ptv = mask_ptv.to(dose_pred.device)
    mask_oars = mask_oars.to(dose_pred.device)
    mask_external = mask_external.to(dose_pred.device)
    
    # Target dose (assuming 50 Gy prescription for PTV)
    target_dose = 42.7
    
    # 1. PTV Coverage Loss - Use high-order penalty for sharper convergence
    # This pushes dose strongly toward the target value
    ptv_dose_diff = (dose_pred - target_dose) * mask_ptv
    ptv_underdose = torch.relu(-ptv_dose_diff)  # Penalize dose < target
    ptv_overdose = torch.relu(ptv_dose_diff)    # Penalize dose > target
    
    # Use 4th power for very sharp penalty around target dose
    loss_ptv_coverage = torch.mean(ptv_underdose**4) * 10.0 + torch.mean(ptv_overdose**2) * 1.0
    
    # 2. OAR Sparing - Exponential penalty for high doses
    # This creates a sharp cutoff for OAR doses
    oar_doses = dose_pred * mask_oars
    oar_threshold = 30.0  # Threshold above which we heavily penalize
    excess_oar_dose = torch.relu(oar_doses - oar_threshold)
    loss_oar = torch.mean(torch.exp(excess_oar_dose / 10.0) - 1.0) * 0.1
    
    # 3. Edge Sharpness Loss - Maximize gradient magnitude at PTV boundary
    # This is the key for sharp edges
    # Compute spatial gradients
    dose_grad_x = dose_pred[:, :, 1:, :] - dose_pred[:, :, :-1, :]
    dose_grad_y = dose_pred[:, :, :, 1:] - dose_pred[:, :, :, :-1]
    dose_grad_z = dose_pred[:, 1:, :, :] - dose_pred[:, :-1, :, :]
    
    # Compute PTV boundary (dilated PTV minus eroded PTV)
    from torch.nn.functional import max_pool3d
    kernel_size = 3
    ptv_float = mask_ptv.float()
    ptv_dilated = max_pool3d(ptv_float.unsqueeze(0), kernel_size, stride=1, padding=1).squeeze(0)
    ptv_eroded = -max_pool3d(-ptv_float.unsqueeze(0), kernel_size, stride=1, padding=1).squeeze(0)
    ptv_boundary = (ptv_dilated - ptv_eroded) > 0.5
    
    # We want HIGH gradients at the boundary (negative loss encourages high gradients)
    grad_mag_x = torch.abs(dose_grad_x[:, :, :-1, :]) * ptv_boundary[:, :, 1:-1, :]
    grad_mag_y = torch.abs(dose_grad_y[:, :, :, :-1]) * ptv_boundary[:, :, :, 1:-1]
    grad_mag_z = torch.abs(dose_grad_z[:, :-1, :, :]) * ptv_boundary[:, 1:-1, :, :]
    
    # Negative because we want to maximize gradient magnitude
    loss_edge_sharpness = -(torch.mean(grad_mag_x) + torch.mean(grad_mag_y) + torch.mean(grad_mag_z)) * 0.01
    
    # 4. Dose Conformity - Penalize dose outside PTV
    outside_ptv = (~mask_ptv) & mask_external
    dose_outside = dose_pred * outside_ptv
    conformity_threshold = 25.0  # 50% of prescription
    excess_outside_dose = torch.relu(dose_outside - conformity_threshold)
    loss_conformity = torch.mean(excess_outside_dose**3)
    
    # 5. MU efficiency (keep MUs reasonable)
    mu_total = torch.sum(pred_mus)
    loss_mu_efficiency = mu_total * 0.0001
    
    # 6. Leaf smoothness (prevent jagged leaf patterns)
    leaf_diff_cp = leafs[:, :, 1:, :] - leafs[:, :, :-1, :]  # Between control points
    leaf_diff_pairs = leafs[:, :, :, 1:] - leafs[:, :, :, :-1]  # Between leaf pairs
    loss_leaf_smooth = (torch.mean(leaf_diff_cp**2) + torch.mean(leaf_diff_pairs**2)) * 0.001
    
    # Combine all losses
    all_losses = [
        loss_ptv_coverage * weights.get("loss_ptv_coverage", 1.0),
        loss_oar * weights.get("loss_oar", 1.0),
        loss_edge_sharpness * weights.get("loss_edge_sharpness", 1.0),
        loss_conformity * weights.get("loss_conformity", 1.0),
        loss_mu_efficiency * weights.get("loss_mu_efficiency", 1.0),
        loss_leaf_smooth * weights.get("loss_leaf_smooth", 1.0),
        torch.tensor(0.0).to(dose_pred.device),  # Placeholder for compatibility
        torch.tensor(0.0).to(dose_pred.device),
        torch.tensor(0.0).to(dose_pred.device),
        torch.tensor(0.0).to(dose_pred.device),
        torch.tensor(0.0).to(dose_pred.device),
    ]
    
    return all_losses

def compute_loss(dose_pred, dose_true, pred_mus, leafs, pred_jaws, weights, _masks):
    (
        loss_lower_bound_gy,
        loss_higher_bound_gy,
        loss_lower_bound_target,
        loss_higher_bound_target,
        l2_loss_oars_and_background,
    ) = dose_loss(x, dose_pred, constraints_batch, masks, region_weights, None)
    mu_rate_loss, mu_complexity_loss = mus_loss(pred_mus, config)
    leaf_reg_loss, leaf_complexity_loss = leafs_loss(leafs, config)
    jaw_opening_loss, jaw_complexity_loss = jaws_loss(pred_jaws, config)
    all_losses = [
        scale_loss(loss_lower_bound_gy, weights["loss_lower_bound_gy"]),
        scale_loss(loss_higher_bound_gy, weights["loss_higher_bound_gy"]),
        scale_loss(loss_lower_bound_target, weights["loss_lower_bound_target"]),
        scale_loss(loss_higher_bound_target, weights["loss_higher_bound_target"]),
        scale_loss(l2_loss_oars_and_background, weights["l2_loss_oars_and_background"]),
        scale_loss(mu_rate_loss, weights["mu_rate_loss"]),
        scale_loss(mu_complexity_loss, weights["mu_complexity_loss"]),
        scale_loss(leaf_reg_loss, weights["leaf_reg_loss"]),
        scale_loss(leaf_complexity_loss, weights["leaf_complexity_loss"]),
        scale_loss(jaw_opening_loss, weights["jaw_opening_loss"]),
        scale_loss(jaw_complexity_loss, weights["jaw_complexity_loss"]),
    ]
    return all_losses

def compute_mae_loss(dose_pred, dose_true, pred_mus, leafs, pred_jaws, weights, masks):
    losses = []
    for index, mask in enumerate(masks):
        if (index == 0):
            losses.append(torch.mean(torch.abs((dose_true - dose_pred)[mask > 0])**2))
        else:
            losses.append(torch.mean(torch.abs((dose_true - dose_pred)[mask > 0])**2))

    losses.append(100.0 * torch.mean(torch.abs(leafs[:, 0, :, :] - 0.5)) + torch.mean(torch.abs(leafs[:, 1, :, :] - 0.0)))
    return losses

print_stuff = 0
loss_plot = 1.0
best_results = []
n_tests = 200
patience_thr = 500
max_iter = 2000

oar_dose = 10.0

for test_i in range(n_tests):

    experiment = Experiment(
        api_key="ro9UfCMFS2O73enclmXbXfJJj", project_name="autoplan_static"
    )
    try:
        x, y_dose, masks, region_weights, constraints_batch, ct_volume, config, valid_parameters_layer, mask_target, mask_external, mask_oar, dose_target, current_res, weights, latest, pred_mlc, pred_jaws, pred_mus, masks_torch = get_example_data()

        patience = 0
        epoch = 0
        lr = 1e-2 # 4e-3
        kernel_size = 3
        lr_decay = 1e-6
        optimizer = torch.optim.AdamW([pred_mlc, pred_mus, pred_jaws], lr=lr, weight_decay=lr_decay)
        # optimizer = torch.optim.LBFGS([pred_mlc, pred_mus, pred_jaws], lr=lr, tolerance_grad=0.0, tolerance_change=0.0, history_size=10, line_search_fn='strong_wolfe')
        
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)

        dose_layer = DoseEngine(config, kernel_size, permute_ct=True, leafs_centered=True)
        dose_layer.train()

        experiment.log_parameters(
            {
                "lr_0": lr,
                "kernel_size": kernel_size,
                "lr_decay": lr_decay,
                "weights": weights,
                "physical_size": config.physical_size_ct,
                "roi_weights": constraints_batch["weight"]
            }, nested_support=True
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)

            # Forward
            dose_pred = dose_layer(pred_mlc, pred_mus, jaw_positions=pred_jaws, ct_image=ct_volume)
            # dose_pred = torch.where(mask_external, dose_pred, torch.zeros_like(dose_pred))

            # Compute loss
            raw_losses = compute_loss(dose_pred, y_dose, pred_mus, pred_mlc, pred_jaws, weights, masks_torch)
            loss = torch.stack(raw_losses).sum()
            
            # Backprop
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(pred_mlc, max_norm=1 / 40.0)
            # torch.nn.utils.clip_grad_norm_(pred_jaws, max_norm=0.0)
            # torch.nn.utils.clip_grad_norm_(pred_mus, max_norm=1.0)

            # stash anything you want to inspect/plot after step()
            latest["raw_losses"] = [v.detach().item() for v in raw_losses]
            latest["loss_val"]   = loss.detach().item()
            latest["dose_pred"]  = dose_pred.detach()

            return loss

        start_time = time.time()
        while patience < patience_thr:
            if (epoch > max_iter):
                break
            
            # --- the actual optimizer step ---
            loss = optimizer.step(closure)   # returns the last loss the closure returned
            # scheduler.step(loss)

            raw_losses = latest["raw_losses"]
            dose_pred = latest["dose_pred"]
            loss_val = latest["loss_val"]
            mae_loss = np.round(torch.mean(torch.abs((y_dose - dose_pred)[masks_torch[-1] > 0])).cpu().detach().numpy(), 4)
            
            
            patience += 1
            if (loss < current_res[0]):
                patience = 0
                current_res = [loss, weights, pred_mlc, pred_mus, pred_jaws, mae_loss]
                
            else:
                # print("Patience count:", patience)
                if ((patience >= patience_thr) | torch.isnan(dose_pred).any()):
                    best_results.append(current_res)
                    print("Best result for this test:", current_res)
                    break

            lr_now = lr # scheduler.get_last_lr()[0]
            experiment.log_metrics(
                {
                    "loss": loss.item(),
                    "dose_mae": mae_loss,
                    "lr": lr_now,
                },
                epoch=epoch,
            )

            epoch += 1

        print(f"Optimization finished in {int(time.time() - start_time)}s.")
        pred_mlc = current_res[2]
        pred_mus = current_res[3]
        pred_jaws = current_res[4]
        pred_mlc_grads = pred_mlc.grad.cpu().detach().numpy()
        pred_jaws_grads = pred_jaws.grad.cpu().detach().numpy()
        pred_mus_grads = pred_mus.grad.cpu().detach().numpy()
        pred_mlc_valid, pred_mus_valid, pred_jaws_valid = valid_parameters_layer(
            pred_mlc, pred_mus, pred_jaws
        )
        results = result_validation(config, dose_pred, pred_mlc_valid, pred_jaws_valid, pred_mus_valid, x, dose_pred, constraints_batch, masks, region_weights)
        experiment.log_metrics(
            {
                "results": results,
            },
            epoch=epoch,
        )

        print_results(raw_losses, y_dose, pred_mlc_valid, pred_mus_valid, pred_jaws_valid, pred_mlc_grads, pred_jaws_grads, pred_mus_grads, best_results, dose_pred, ct_volume, masks_torch, mae_loss)
        make_animation(dose_layer, pred_mlc, pred_mus, pred_jaws, ct_volume, masks_torch)
    except Exception as e:
        print("Exception during test:", e)
        
    experiment.end()
