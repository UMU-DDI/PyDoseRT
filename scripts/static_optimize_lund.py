from comet_ml import Experiment
import sys
sys.path.append('../')
import numpy as np
import os
import torch
import time
import math
import torch
from pydose_rt.data import Patient, MachineConfig, loaders
from pydose_rt import DoseEngine
from pydose_rt.layers import BeamValidationLayer
from pydose_rt.utils.plotting import *
from pydose_rt.physics.kernels.pencil_beam_model import *
from pydose_rt.utils.grad_monitor import GradMonitor
import numpy as np
from pydose_rt.objectives.losses import compute_loss, compute_mae_loss
from pydose_rt.objectives.metrics import result_validation
from pydose_rt.utils.utils import create_bound_weight_matrix, get_initial_weights, prune_patients
from pydose_rt.utils.plotting import print_results, make_animation
from dotenv import load_dotenv
load_dotenv()  # will look for .env in project root

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if (os.path.exists("/mimer/NOBACKUP/groups/naiss2023-6-64/attila/miqa/")):
    remote = True
else:
    remote = False


if remote:
    
    data_path = "/mimer/NOBACKUP/groups/naiss2023-6-64/attila/converted_lund/"
    patient_list = prune_patients([os.path.join(data_path, name) for name in os.listdir(data_path)])
    patient = loaders.load_nifti(
        folder_path=patient_list[0]
    )
    optimization = OptimizationConfig(preset="src/pydose_rt/data/optimization_presets/lund-probe.json")

    kernel_size = 15
    device = device
    dtype = torch.float16

    machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/lund-probe.json", resolution=patient.voxel_spacing_mm, ct_array_shape=patient.ct_array.shape)
    max_iter = 3000
else:
    data_path = "/media/bolo/Datasets/converted_lund/"
    patient_list = prune_patients([os.path.join(data_path, name) for name in os.listdir(data_path)])
    patient = loaders.load_nifti(
        folder_path=patient_list[0]
    )
    optimization = OptimizationConfig(preset="src/pydose_rt/data/optimization_presets/lund-probe.json")

    kernel_size = 3
    device = device
    dtype = torch.float16

    machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/lund-probe.json", resolution=patient.voxel_spacing_mm, ct_array_shape=patient.ct_array.shape)
    max_iter = 100



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
        current_res = [np.inf]
        weights = get_initial_weights()
        latest = {"raw_losses": None, "loss_val": None, "dose_pred": None}
        optimization.randomize_weights()


        dose_layer = DoseEngine(machine_config, permute_ct=False, leafs_centered=False, adjust_values=True)
        valid_parameters_layer = BeamValidationLayer(machine_config, device, dtype, dose_layer.field_size, leafs_centered=False, adjust_values=True)
        dose_layer.train()
        pred_mlc, pred_jaws, pred_mus = dose_layer.get_open_parameters()

        patience = 0
        epoch = 0
        lr = 10**(np.random.uniform(-2, 0)) # 1e-1 # 4e-3
        lr_decay = 1e-4
        optimizer = torch.optim.AdamW([pred_mlc, pred_mus, pred_jaws], lr=lr, weight_decay=lr_decay)

        experiment.log_parameters(
            {
                "lr_0": lr,
                "kernel_size": kernel_size,
                "lr_decay": lr_decay,
                "weights": weights,
                "physical_size": machine_config.physical_size_ct,
                "roi_weights": optimization.weights
            }, nested_support=True
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)

            # Forward
            dose_pred = dose_layer(pred_mlc, pred_mus, jaw_positions=pred_jaws, ct_image=ct_volume)
            dose_pred = torch.where(mask_external, dose_pred, torch.zeros_like(dose_pred))

            # Compute loss
            raw_losses = compute_mae_loss(patient, treatment, machine_config, dose_pred, dose_target, pred_mus, pred_mlc, pred_jaws, weights, masks, masks_torch)
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

        print(f"Optimization finished in {int(time.time() - start_time)}s. It used the new fluence map!")
        pred_mlc = current_res[2]
        pred_mus = current_res[3]
        pred_jaws = current_res[4]
        pred_mlc_grads = pred_mlc.grad.cpu().detach().numpy()
        pred_jaws_grads = pred_jaws.grad.cpu().detach().numpy()
        pred_mus_grads = pred_mus.grad.cpu().detach().numpy()
        pred_mlc_valid, pred_mus_valid, pred_jaws_valid = valid_parameters_layer(
            pred_mlc, pred_mus, pred_jaws
        )
        results = result_validation(patient, machine_config, treatment, dose_pred, pred_mlc_valid, pred_jaws_valid, pred_mus_valid)
        experiment.log_metrics(
            {
                "results": results,
            },
            epoch=epoch,
        )

        experiment.log_asset_data(pred_mlc_valid.cpu().detach().numpy(), "mlc_positions.npy")
        experiment.log_asset_data(pred_mus_valid.cpu().detach().numpy(), "mu_values.npy")

        print_results(experiment, treatment, raw_losses, y_dose, pred_mlc_valid, pred_mus_valid, pred_jaws_valid, pred_mlc_grads, pred_jaws_grads, pred_mus_grads, best_results, dose_pred, ct_volume, masks_torch, mae_loss)
        make_animation(experiment, treatment, machine_config, patient, dose_layer, pred_mlc, pred_mus, pred_jaws, dose_max=50.0)
    except Exception as e:
        print("Exception during test:", e)
        
    experiment.end()
