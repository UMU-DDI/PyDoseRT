from comet_ml import Experiment
import sys
sys.path.append('../')
import numpy as np
import os
import torch
import time
import math
import torch
from pydose_rt.data import DoseConfig, PatientConfig, TreatmentConfig
from pydose_rt import DoseEngine
from pydose_rt.layers import ValidParametersLayer
from pydose_rt.utils.plotting import *
from pydose_rt.physics.kernels.pencil_beam_model import *
from pydose_rt.utils.grad_monitor import GradMonitor
import numpy as np
from pydose_rt.objectives.losses import dose_loss, leafs_loss, mus_loss, jaws_loss, result_validation, scale_loss
from pydose_rt.utils.utils import create_bound_weight_matrix, prune_patients, get_initial_weights, get_model_input
from pydose_rt.utils.plotting import print_results, make_animation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_example_data(data_path="/media/bolo/Datasets/converted_lund/"):
    patient_list = prune_patients([os.path.join(data_path, name) for name in os.listdir(data_path)])
    patient = PatientConfig.from_nifti(
        folder_path=patient_list[0]
    )
    config = DoseConfig(
        patient=patient,
        machine_preset="lund-probe", 
        treatment_preset="lund-probe",
        downsampling_factor=(1,2,2), 
    )
    config.treatment.randomize_weights()
    x = patient.ct_array
    y_dose = torch.from_numpy(patient.dose)
    masks = torch.from_numpy(np.stack([v for k,v in patient.structures.items()], 0))
    region_weights = torch.from_numpy(create_bound_weight_matrix(patient.structures, config.treatment.weights))
    x = get_model_input(config.patient, config.treatment)
    x = torch.from_numpy(x)
    x = x.expand(1, -1, -1, -1, -1)
    y_dose = y_dose.expand(1, -1, -1, -1)
    masks = masks.expand(1, -1, -1, -1, -1)

    ct_volume = (1000.0 * x[:, 0, ...]).to(device)  # scale to HU

    valid_parameters_layer = ValidParametersLayer(config.machine, leafs_centered=True)

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

    pred_mlc_init = torch.ones((1, 2, config.machine.number_of_cps, config.machine.number_of_leaf_pairs), dtype=torch.float32, device=device)
    pred_mlc_init[:, 0, :, :] = 0.5
    pred_mlc_init[:, 1, :, :] = 0.0
    pred_mlc = pred_mlc_init.clone().detach().requires_grad_(True)
    pred_jaws_init = torch.zeros((1, 2, config.machine.number_of_cps), dtype=torch.float32, device=device)
    pred_jaws_init[:, 0, :] = 0.5
    pred_jaws = pred_jaws_init.clone().detach().requires_grad_(True)
    pred_mus_init = (100.0 / config.machine.number_of_cps) * torch.ones((1, config.machine.number_of_cps), dtype=torch.float32, device=device)
    pred_mus = pred_mus_init.clone().detach().requires_grad_(True)
    return x, y_dose, masks, region_weights, config.treatment, ct_volume, config.machine, valid_parameters_layer, mask_target, mask_external, mask_oar, dose_target, current_res, weights, latest, pred_mlc, pred_jaws, pred_mus, masks_torch

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
    ) = dose_loss(x, dose_pred, treatment, masks, region_weights, None)
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
        x, y_dose, masks, region_weights, treatment, ct_volume, config, valid_parameters_layer, mask_target, mask_external, mask_oar, dose_target, current_res, weights, latest, pred_mlc, pred_jaws, pred_mus, masks_torch = get_example_data("/mimer/NOBACKUP/groups/naiss2023-6-64/attila/converted_lund/")

        patience = 0
        epoch = 0
        lr = 1e-2 # 4e-3
        kernel_size = 3
        lr_decay = 1e-6
        optimizer = torch.optim.AdamW([pred_mlc, pred_mus, pred_jaws], lr=lr, weight_decay=lr_decay)
        # optimizer = torch.optim.LBFGS([pred_mlc, pred_mus, pred_jaws], lr=lr, tolerance_grad=0.0, tolerance_change=0.0, history_size=10, line_search_fn='strong_wolfe')
        
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)

        dose_layer = DoseEngine(config, kernel_size, permute_ct=False, leafs_centered=True)
        dose_layer.train()

        experiment.log_parameters(
            {
                "lr_0": lr,
                "kernel_size": kernel_size,
                "lr_decay": lr_decay,
                "weights": weights,
                "physical_size": config.physical_size_ct,
                "roi_weights": treatment.weights
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
        results = result_validation(config, dose_pred, pred_mlc_valid, pred_jaws_valid, pred_mus_valid, x, dose_pred, treatment, masks, region_weights)
        experiment.log_metrics(
            {
                "results": results,
            },
            epoch=epoch,
        )

        print_results(experiment, treatment, raw_losses, y_dose, pred_mlc_valid, pred_mus_valid, pred_jaws_valid, pred_mlc_grads, pred_jaws_grads, pred_mus_grads, best_results, dose_pred, ct_volume, masks_torch, mae_loss)
        make_animation(experiment, treatment, dose_layer, config, mask_external, pred_mlc, pred_mus, pred_jaws, ct_volume, masks_torch)
    except Exception as e:
        print("Exception during test:", e)
        
    experiment.end()
