from comet_ml import Experiment
import sys
sys.path.append('../')
import numpy as np
import os
import torch
import time
import math
import torch
from pydose_rt.data import DoseConfig, PatientData, TreatmentConfig
from pydose_rt import DoseEngine
from pydose_rt.layers import ValidParametersLayer
from pydose_rt.utils.plotting import *
from pydose_rt.physics.kernels.pencil_beam_model import *
from pydose_rt.utils.grad_monitor import GradMonitor
import numpy as np
from pydose_rt.objectives.losses import dose_loss, leafs_loss, mus_loss, jaws_loss, scale_loss
from pydose_rt.objectives.metrics import result_validation
from pydose_rt.utils.utils import create_bound_weight_matrix, get_initial_weights, get_model_input
from pydose_rt.utils.plotting import print_results, make_animation
from dotenv import load_dotenv
load_dotenv()  # will look for .env in project root

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if (os.path.exists("/mimer/NOBACKUP/groups/naiss2023-6-64/attila/miqa/")):
    remote = True
else:
    remote = False


if remote:
    
    ct_folder = "/mimer/NOBACKUP/groups/naiss2023-6-64/attila/miqa/0e54d72a21/"
    rtplan_path = "/mimer/NOBACKUP/groups/naiss2023-6-64/attila/miqa/0e54d72a21_plans/1ARC/RP1.2.752.243.1.1.20251031145134399.7000.37887.dcm"
    rtdose_path = "/mimer/NOBACKUP/groups/naiss2023-6-64/attila/miqa/0e54d72a21_plans/1ARC/RD1.2.752.243.1.1.20251031145134399.8000.21005.dcm"
    dtype = torch.float32

    config = DoseConfig.from_dicom(
        ct_folder=ct_folder, 
        dose_path=rtdose_path,
        plan_path=rtplan_path,
        struct_names=["CTV", "PTVT_42.7", "FemoralHead_L", "FemoralHead_R", "Bladder", "External"],
        machine_preset="umea",
        treatment_preset="umea",
        downsampling_factor=(1, 2, 2),
        dtype=dtype,
        device=device
    )
    max_iter = 2000
    kernel_size = 15
else:
    ct_folder = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21/"
    rtplan_path = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21_plans/1ARC/RP1.2.752.243.1.1.20251031145134399.7000.37887.dcm"
    rtdose_path = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21_plans/1ARC/RD1.2.752.243.1.1.20251031145134399.8000.21005.dcm"
    dtype = torch.float16

    config = DoseConfig.from_dicom(
        ct_folder=ct_folder, 
        dose_path=rtdose_path,
        plan_path=rtplan_path,
        struct_names=["CTV", "PTVT_42.7", "FemoralHead_L", "FemoralHead_R", "Bladder", "External"],
        machine_preset="umea",
        treatment_preset="umea",
        downsampling_factor=(1, 4, 4),
        dtype=dtype,
        device=device
    )
    max_iter = 100
    kernel_size = 3

def get_example_data():
    current_res = [np.inf]
    weights = get_initial_weights()
    latest = {"raw_losses": None, "loss_val": None, "dose_pred": None}
    config.treatment.randomize_weights()

    x = config.patient.ct_array
    y_dose = torch.from_numpy(config.patient.dose)
    masks = torch.from_numpy(np.stack([v for k,v in config.patient.structures.items()], 0))
    region_weights = torch.from_numpy(create_bound_weight_matrix(config.patient.structures, config.treatment.weights))
    x = get_model_input(config.patient, config.treatment)
    x = torch.from_numpy(x)
    x = x.expand(1, -1, -1, -1, -1)
    y_dose = y_dose.expand(1, -1, -1, -1)
    masks = masks.expand(1, -1, -1, -1, -1)

    ct_volume = (1000.0 * x[:, 0, ...]).to(device).to(dtype)  # scale to HU


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


    pred_mlc_init = torch.ones((1, 2, config.machine.number_of_cps, config.machine.number_of_leaf_pairs), dtype=dtype, device=device)
    pred_mlc_init[:, 0, :, :] = -50.0
    pred_mlc_init[:, 1, :, :] = 50.0
    pred_mlc = pred_mlc_init.clone().detach().requires_grad_(True)
    pred_jaws_init = torch.from_numpy(config.patient.plan_jaws).to(device).to(dtype).clone().detach()
    # pred_jaws_init = torch.zeros((1, 2, config.machine.number_of_cps), dtype=torch.float32, device=device)
    # pred_jaws_init[:, 0, :] = 0.1
    # pred_jaws_init[:, 1, :] = 0.9
    pred_jaws = pred_jaws_init.clone().detach().requires_grad_(True)
    pred_mus_init = (17000.0 / config.machine.number_of_cps) * torch.ones((1, config.machine.number_of_cps), dtype=dtype, device=device)
    pred_mus = pred_mus_init.clone().detach().requires_grad_(True)
    return x, y_dose, masks, region_weights, config, ct_volume, mask_target, mask_external, mask_oar, dose_target, current_res, weights, latest, pred_mlc, pred_jaws, pred_mus, masks_torch

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

def compute_loss(dose_pred, dose_true, pred_mus, leafs, pred_jaws, weights, _masks):
    (
        loss_lower_bound_gy,
        loss_higher_bound_gy,
        loss_lower_bound_target,
        loss_higher_bound_target,
        l2_loss_oars_and_background,
    ) = dose_loss(x, dose_pred, treatment, masks, region_weights, None)
    mu_rate_loss, mu_complexity_loss = mus_loss(pred_mus, config.machine)
    leaf_reg_loss, leaf_complexity_loss = leafs_loss(leafs, config.machine)
    jaw_opening_loss, jaw_complexity_loss = jaws_loss(pred_jaws, config.machine)
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
    # losses.append(torch.mean(torch.abs((dose_true - dose_pred))**2))
    for index, mask in enumerate([masks[0], masks[1], masks[-1]]):
        losses.append(torch.mean(torch.abs((dose_true - dose_pred)[mask > 0])**2))

    # jaw_loss = torch.mean((torch.abs(leafs[:, :, 1:, :] - leafs[:, :, :-1, :]))**2)
    # bank_loss = leaf_range_loss(leafs, config.machine)
    # losses.append(scale_loss(jaw_loss, weights["leaf_complexity_loss"]))
    # losses.append(scale_loss(bank_loss, weights["leaf_reg_loss"]))

    return losses

def leaf_range_loss(leafs, config, threshold_mm=150.0):
    """
    Penalize leaf tip differences (max - min) that exceed threshold.
    
    Args:
        leafs: [B, 2, CP, num_leafs] - leaf positions (normalized 0-1)
        config: machine config with field_size
        threshold_mm: maximum allowed range in mm (default 150.0)
    """
    # Convert threshold from mm to normalized units
    threshold_normalized = threshold_mm / config.field_size[0]
    
    # Compute range (max - min) for each leaf bank
    bank0_range = leafs[:, 0, :, :].max() - leafs[:, 0, :, :].min()
    bank1_range = leafs[:, 1, :, :].max() - leafs[:, 1, :, :].min()
    
    # Penalize when range exceeds threshold
    # Using ReLU so we only penalize violations, and squaring for smooth gradients
    bank0_violation = torch.nn.LeakyReLU(negative_slope=0.01)(bank0_range - threshold_normalized) ** 2
    bank1_violation = torch.nn.LeakyReLU(negative_slope=0.01)(bank1_range - threshold_normalized) ** 2
    
    return bank0_violation + bank1_violation

print_stuff = 0
loss_plot = 1.0
best_results = []
n_tests = 200
patience_thr = 500

oar_dose = 10.0

for test_i in range(n_tests):

    experiment = Experiment(
        api_key=os.getenv("COMET_API"), project_name="autoplan_static"
    )
    try:
        x, y_dose, masks, region_weights, config, ct_volume, mask_target, mask_external, mask_oar, dose_target, current_res, weights, latest, pred_mlc, pred_jaws, pred_mus, masks_torch = get_example_data()
        treatment = config.treatment
        machine_config = config.machine

        patience = 0
        epoch = 0
        lr = 10**(np.random.uniform(-2, 2)) # 1e-1 # 4e-3
        lr_decay = 1e-4
        optimizer = torch.optim.AdamW([pred_mlc, pred_mus, pred_jaws], lr=lr, weight_decay=lr_decay)
        # optimizer = torch.optim.LBFGS([pred_mlc, pred_mus, pred_jaws], lr=lr, tolerance_grad=0.0, tolerance_change=0.0, history_size=10, line_search_fn='strong_wolfe')
        
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)

        dose_layer = DoseEngine(machine_config, kernel_size, permute_ct=False, leafs_centered=False, adjust_values=True)
        valid_parameters_layer = ValidParametersLayer(config.machine, leafs_centered=False, adjust_values=True)
        dose_layer.train()

        experiment.log_parameters(
            {
                "lr_0": lr,
                "kernel_size": kernel_size,
                "lr_decay": lr_decay,
                "weights": weights,
                "physical_size": machine_config.physical_size_ct,
                "roi_weights": treatment.weights
            }, nested_support=True
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)

            # Forward
            dose_pred = dose_layer(pred_mlc, pred_mus, jaw_positions=pred_jaws, ct_image=ct_volume)
            dose_pred = torch.where(mask_external, dose_pred, torch.zeros_like(dose_pred))

            # Compute loss
            raw_losses = compute_mae_loss(dose_pred, y_dose, pred_mus, pred_mlc, pred_jaws, weights, masks_torch)
            loss = torch.stack(raw_losses).sum()
            
            # Backprop
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(pred_mlc, max_norm=1 / 40.0)
            torch.nn.utils.clip_grad_norm_(pred_jaws, max_norm=0.0)
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
                    # "leaf_reg": raw_losses[-1] / weights["leaf_complexity_loss"],
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
        results = result_validation(config, dose_pred, pred_mlc_valid, pred_jaws_valid, pred_mus_valid)
        experiment.log_metrics(
            {
                "results": results,
            },
            epoch=epoch,
        )

        experiment.log_asset_data(pred_mlc_valid.cpu().detach().numpy(), "mlc_positions.npy")
        experiment.log_asset_data(pred_mus_valid.cpu().detach().numpy(), "mu_values.npy")

        print_results(experiment, treatment, raw_losses, y_dose, pred_mlc_valid, pred_mus_valid, pred_jaws_valid, pred_mlc_grads, pred_jaws_grads, pred_mus_grads, best_results, dose_pred, ct_volume, masks_torch, mae_loss)
        make_animation(experiment, config, dose_layer, pred_mlc, pred_mus, pred_jaws, dose_max=50.0)
    except Exception as e:
        print("Exception during test:", e)
        
    experiment.end()
