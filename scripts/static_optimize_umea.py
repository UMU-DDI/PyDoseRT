from comet_ml import Experiment
import sys
sys.path.append('../')
import numpy as np
import os
import torch
import time
import math
import torch
from pydose_rt.data import Patient, TreatmentConfig, MachineConfig, loaders
from pydose_rt import DoseEngine
from pydose_rt.layers import ValidParametersLayer
from pydose_rt.utils.plotting import *
from pydose_rt.physics.kernels.pencil_beam_model import *
from pydose_rt.utils.grad_monitor import GradMonitor
import numpy as np
from pydose_rt.objectives.losses import compute_loss, compute_mae_loss
from pydose_rt.objectives.metrics import result_validation
from pydose_rt.utils.utils import get_initial_weights
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

    patient, treatment = loaders.load_dicom(
                ct_folder=ct_folder, 
                dose_path=rtdose_path, 
                plan_path=rtplan_path, 
                struct_names=["CTV", "PTVT_42.7", "FemoralHead_L", "FemoralHead_R", "Bladder", "External"],
                treatment_preset="src/pydose_rt/data/treatment_presets/umea.json"
                )

    treatment.kernel_size = 25
    treatment.device = device
    treatment.downsampling_factor = (1, 2, 2)
    treatment.dtype = dtype

    machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/umea.json", resolution=patient.voxel_spacing_mm, ct_array_shape=patient.ct_array.shape)
    max_iter = 500
else:
    ct_folder = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21/"
    rtplan_path = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21_plans/1ARC/RP1.2.752.243.1.1.20251031145134399.7000.37887.dcm"
    rtdose_path = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21_plans/1ARC/RD1.2.752.243.1.1.20251031145134399.8000.21005.dcm"
    dtype = torch.float32

    patient, treatment = loaders.load_dicom(
                ct_folder=ct_folder, 
                dose_path=rtdose_path, 
                plan_path=rtplan_path, 
                struct_names=["CTV", "PTVT_42.7", "FemoralHead_L", "FemoralHead_R", "Bladder", "External"],
                treatment_preset="src/pydose_rt/data/treatment_presets/umea.json"
                )

    treatment.kernel_size = 3
    treatment.device = device
    treatment.dtype = dtype
    treatment.downsampling_factor = (1, 4, 4)

    machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/umea.json", resolution=patient.voxel_spacing_mm, ct_array_shape=patient.ct_array.shape)
    max_iter = 10



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
        latest = {"raw_losses": None, "loss_val": None, "dose_pred": None, "pred_mlc": None, "pred_mus": None, "pred_jaws": None}
        treatment.randomize_weights()

        y_dose = torch.from_numpy(patient.dose)
        masks = torch.from_numpy(np.stack([v for k,v in patient.structures.items()], 0))
        y_dose = y_dose.expand(1, -1, -1, -1)
        masks = masks.expand(1, -1, -1, -1, -1)

        ct_volume = torch.from_numpy(np.expand_dims(patient.ct_array, 0)).to(device).to(dtype)  # scale to HU


        mask_target = masks[0, 0, ...].expand(1, -1, -1, -1).clone().detach().to(device) > 0
        mask_external = masks.sum(1).clone().detach().to(device) > 0
        mask_oar = torch.sum(masks[0, 1:-1, ...], 0).expand(1, -1, -1, -1).clone().detach().to(device) > 0
        dose_target = y_dose.expand(1, -1, -1, -1).clone().detach().to(device)
        masks_torch = []
        for i in range(masks.shape[1]):
            masks_torch.append(masks[0, i, ...].expand(1, -1, -1, -1))
        y_dose = y_dose.to(treatment.device)
        y_dose = torch.where(mask_external, y_dose, torch.zeros_like(y_dose))
        masks = masks.to(treatment.device)

        dose_layer = DoseEngine(machine_config, treatment, permute_ct=False, leafs_centered=False, adjust_values=True)
        valid_parameters_layer = ValidParametersLayer(machine_config, treatment, leafs_centered=False, adjust_values=True)
        dose_layer.train()
        # pred_mlc, pred_jaws, pred_mus = dose_layer.get_open_parameters()
        base_mlc = torch.from_numpy(treatment.plan_mlcs).to(treatment.device).to(treatment.dtype)
        base_jaws = torch.from_numpy(treatment.plan_jaws).to(treatment.device).to(treatment.dtype)
        base_mus = torch.from_numpy(treatment.plan_mus).to(treatment.device).to(treatment.dtype)
        # pred_mlc = torch.from_numpy(treatment.plan_mlcs).to(treatment.device).to(treatment.dtype)
        # pred_jaws = torch.from_numpy(treatment.plan_jaws).to(treatment.device).to(treatment.dtype)
        # pred_mus = torch.from_numpy(treatment.plan_mus).to(treatment.device).to(treatment.dtype)
        pred_scales = torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0]).to(treatment.device).to(treatment.dtype).requires_grad_(True)
        # pred_mus *= pred_scales[0]
        # pred_mlc[:, 0, ...] *= pred_scales[1]
        # pred_mlc[:, 0, ...] += pred_scales[2]
        # pred_mlc[:, 1, ...] *= pred_scales[3]
        # pred_mlc[:, 1, ...] += pred_scales[4]
        # pred_jaws[:, 0, ...] += pred_scales[5]
        # pred_jaws[:, 1, ...] += pred_scales[6]



        patience = 0
        epoch = 0
        lr = 10**np.random.uniform(-3, 0) # 0.1
        lr_decay = 1e-4
        optimizer = torch.optim.AdamW([pred_scales], lr=lr, weight_decay=lr_decay)

        experiment.log_parameters(
            {
                "lr_0": lr,
                "kernel_size": treatment.kernel_size,
                "lr_decay": lr_decay,
                "weights": weights,
                "physical_size": machine_config.physical_size_ct,
                "roi_weights": treatment.weights
            }, nested_support=True
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            
            pred_mus = base_mus * pred_scales[0]
 
            # Construct pred_mlc without in-place operations
            mlc_left = base_mlc[:, 0, ...] + pred_scales[1]
            mlc_right = base_mlc[:, 1, ...] + pred_scales[2]
            pred_mlc = torch.stack([mlc_left, mlc_right], dim=1)
 
            # Construct pred_jaws without in-place operations
            jaws_left = base_jaws[:, 0, ...] + pred_scales[3]
            jaws_right = base_jaws[:, 1, ...] + pred_scales[4]
            pred_jaws = torch.stack([jaws_left, jaws_right], dim=1)

            # Forward
            dose_pred = dose_layer(pred_mlc, pred_mus, jaw_positions=pred_jaws, ct_image=ct_volume)
            dose_pred = torch.where(mask_external, dose_pred, torch.zeros_like(dose_pred))

            # Compute loss
            raw_losses = compute_mae_loss(patient, treatment, machine_config, dose_pred, dose_target, pred_mus, pred_mlc, pred_jaws, weights, masks, masks_torch)
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
            latest["pred_mlc"]   = pred_mlc
            latest["pred_mus"]   = pred_mus
            latest["pred_jaws"]  = pred_jaws

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
            pred_mlc = latest["pred_mlc"]
            pred_mus = latest["pred_mus"]
            pred_jaws = latest["pred_jaws"]
            mae_loss = np.round(torch.mean(torch.abs((y_dose - dose_pred)[masks_torch[-1] > 0])).cpu().detach().numpy(), 4)
            
            
            patience += 1
            if (loss < current_res[0]):
                patience = 0
                current_res = [loss, weights , pred_mlc, pred_mus, pred_jaws, mae_loss]
                
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
        print(pred_scales)
        pred_mlc = current_res[2]
        pred_mus = current_res[3]
        pred_jaws = current_res[4]
        pred_mlc_grads = None # pred_mlc.grad.cpu().detach().numpy()
        pred_jaws_grads = None # pred_jaws.grad.cpu().detach().numpy()
        pred_mus_grads = None # pred_mus.grad.cpu().detach().numpy()
        pred_mlc_valid, pred_mus_valid, pred_jaws_valid = valid_parameters_layer(
            pred_mlc, pred_mus, pred_jaws
        )
        results = result_validation(patient, machine_config, treatment, dose_pred.cpu().detach().numpy(), pred_mlc_valid.cpu().detach().numpy(), pred_jaws_valid.cpu().detach().numpy(), pred_mus_valid.cpu().detach().numpy(), compute_gamma=True)
        experiment.log_metrics(
            {
                "results": results,
            },
            epoch=epoch,
        )

        experiment.log_asset_data(pred_mlc_valid.cpu().detach().numpy(), "mlc_positions.npy")
        experiment.log_asset_data(pred_mus_valid.cpu().detach().numpy(), "mu_values.npy")

        print_results(experiment, treatment, raw_losses, y_dose, pred_mlc_valid, pred_mus_valid, pred_jaws_valid, pred_mlc_grads, pred_jaws_grads, pred_mus_grads, best_results, dose_pred, ct_volume, masks_torch, mae_loss, dose_max=7.0)
        make_animation(experiment, treatment, machine_config, patient, dose_layer, pred_mlc, pred_mus, pred_jaws, dose_max=50.0)
    except Exception as e:
        print("Exception during test:", e)
        
    experiment.end()
