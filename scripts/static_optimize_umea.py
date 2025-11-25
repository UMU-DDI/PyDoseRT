from comet_ml import Experiment
import sys
sys.path.append('../')
import numpy as np
import os
import torch
import time
from pathlib import Path
import math
import torch
from pydose_rt.data import Patient, OptimizationConfig, MachineConfig, loaders, BeamSequence
from pydose_rt import DoseEngine
from pydose_rt.layers import BeamValidationLayer
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

patient_name = "P01"
if remote:
    base = Path(f"/mimer/NOBACKUP/groups/naiss2023-6-64/attila/GoldAtlasPlans/{patient_name}")

    ct_folder = base / "[CT] Deformed CT"
    rtplan_path = next((base / "[RP] CT").iterdir())
    rtdose_path = next((base / "[RD] CT Dose").iterdir())
    rtstruct_path = next((base / "[RS] RayStation").iterdir())

    patient, beam_sequence = loaders.load_dicom(
                ct_folder=ct_folder, 
                dose_path=rtdose_path, 
                plan_path=rtplan_path, 
                struct_path=rtstruct_path,
                struct_names=["CTVT", "PTVT_42.7", "FemoralHead_L", "FemoralHead_R", "Bladder", "Rectum", "External"]
                )
    beam_sequence = beam_sequence[0].clone()
    optimization = OptimizationConfig(preset="src/pydose_rt/data/optimization_presets/gold-atlas.json")

    kernel_size = 25
    device = device
    dtype = torch.float32
    downsampling_factor = (1, 2, 2)

    machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/umea_10MV.json", resolution=patient.voxel_spacing_mm, ct_array_shape=patient.ct_array.shape)
    max_iter = 1000
else:
    base = Path(f"/home/bolo/Documents/PyDoseRT/test_data/GoldAtlasPlans/{patient_name}")

    ct_folder = base / "[CT] Deformed CT"
    rtplan_path = next((base / "[RP] CT").iterdir())
    rtdose_path = next((base / "[RD] CT Dose").iterdir())
    rtstruct_path = next((base / "[RS] RayStation").iterdir())

    patient, beam_sequence = loaders.load_dicom(
                ct_folder=ct_folder, 
                dose_path=rtdose_path, 
                plan_path=rtplan_path, 
                struct_path=rtstruct_path,
                struct_names=["CTVT", "PTVT_42.7", "FemoralHead_L", "FemoralHead_R", "Bladder", "Rectum", "External"]
                )
    beam_sequence: BeamSequence = beam_sequence[0]
    beam_sequence = beam_sequence[::16].clone()

    optimization = OptimizationConfig(preset="src/pydose_rt/data/optimization_presets/gold-atlas.json")

    kernel_size = 3
    device = device
    dtype = torch.float32
    downsampling_factor = (1, 2, 2)


    machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/umea_10MV.json", resolution=patient.voxel_spacing_mm, ct_array_shape=patient.ct_array.shape)
    max_iter = 10


gantry_angles = beam_sequence.gantry_angles
number_of_leaf_pairs  = beam_sequence.num_leaf_pairs
field_size = beam_sequence.field_size
iso_center = beam_sequence.iso_center
beam_limiting_device_angles = beam_sequence.beam_limiting_device_angles
sid = beam_sequence.sid
open_field_size = 100.0

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
        current_res = { "loss": np.inf }
        weights = get_initial_weights()
        latest = {"raw_losses": None, "loss_val": None, "dose_pred": None, "pred_mlc": None, "pred_mus": None, "pred_jaws": None}
        optimization.randomize_weights()
        beam_sequence = BeamSequence.create(gantry_angles,
                                            number_of_leaf_pairs,
                                            field_size,
                                            iso_center,
                                            beam_limiting_device_angles,
                                            sid,
                                            open_field_size,
                                            device,
                                            dtype,
                                            True)
        beam_sequence.jaw_positions.requires_grad_(False)

        patient = patient.to(device).to(dtype)
        ct_volume = patient.get_masked_ct("External").unsqueeze(0)
        dose_target = patient.get_masked_dose("External").unsqueeze(0)
        
        engine = DoseEngine(
            ct_array_shape=patient.ct_array.shape, 
            resolution=patient.voxel_spacing_mm, 
            machine_config=machine_config,
            beam_input=beam_sequence.to_delivery(), 
            downsampling_factor=downsampling_factor,
            kernel_size=kernel_size, 
            dtype=dtype, 
            device=device
        )
        valid_parameters_layer = BeamValidationLayer(
            machine_config=machine_config, 
            device=device,
            dtype=dtype,
            field_size=beam_sequence.field_size
        )
        engine.train()
        
        patience = 0
        epoch = 0
        lr = 10**np.random.uniform(-3, 0) # 0.1
        lr_decay = 1e-4
        optimizer = torch.optim.AdamW(beam_sequence.parameters(), lr=lr, weight_decay=lr_decay)

        experiment.log_parameters(
            {
                "lr_0": lr,
                "kernel_size": engine.kernel_size,
                "lr_decay": lr_decay,
                "weights": weights,
                "physical_size": patient.physical_size,
                "roi_weights": optimization.weights
            }, nested_support=True
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            
            # Forward
            dose_pred = engine.compute_beam_sequnce(
                beam_sequence.to_delivery(),
                ct_image=ct_volume
            )
            dose_pred = torch.where(patient.structures["External"], dose_pred, torch.zeros_like(dose_pred))

            # Compute loss
            raw_losses = compute_mae_loss(patient, optimization, machine_config, dose_pred, dose_target, beam_sequence, weights)
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
            latest["beam_sequence"]   = beam_sequence

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
            beam_sequence = latest["beam_sequence"]
            mae_loss = np.round(torch.mean(torch.abs((dose_target - dose_pred)[0, patient.structures["External"]])).cpu().detach().numpy(), 4)
            
            patience += 1
            if (loss < current_res["loss"]):
                patience = 0
                current_res = {
                    "loss": loss, 
                    "weights": weights, 
                    "beam_sequence": beam_sequence.clone(),
                    "mae_loss": mae_loss
                }
                
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
        pred_mlc = current_res["beam_sequence"].leaf_positions
        pred_mus = current_res["beam_sequence"].mus
        pred_jaws = current_res["beam_sequence"].jaw_positions

        pred_mlc_grads = None
        pred_jaws_grads = None
        pred_mus_grads = None

        pred_mlc_valid, pred_jaws_valid, pred_mus_valid = valid_parameters_layer(
            pred_mlc, pred_mus, pred_jaws
        )
        beam_sequence.leaf_positions = pred_mlc_valid
        beam_sequence.mus = pred_mus_valid
        beam_sequence.jaw_positions = pred_jaws_valid

        results = result_validation(patient.to('cpu'), machine_config, beam_sequence.to('cpu'), dose_pred[0].to('cpu'), optimization, compute_gamma=False, compute_clinical_criteria=True)
        experiment.log_metrics(
            {
                "results": results,
            },
            epoch=epoch,
        )

        experiment.log_asset_data(beam_sequence.leaf_positions.cpu().detach().numpy(), "mlc_positions.npy")
        experiment.log_asset_data(beam_sequence.mus.cpu().detach().numpy(), "mu_values.npy")
        print_results(experiment, optimization, raw_losses, dose_target, beam_sequence, None, None, None, best_results, dose_pred, ct_volume, [mask.unsqueeze(0) for mask in list(patient.structures.values())], mae_loss, preset="gold-atlas", dose_max=7.0)
        make_animation(experiment, machine_config, patient, engine, beam_sequence, dose_max=7.0)
    except Exception as e:
        print("Exception during test:", e)
        
    experiment.end()
