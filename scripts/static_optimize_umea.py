import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
from comet_ml import Experiment
import sys
sys.path.append('../')
import numpy as np
import os
import torch
import time
from pathlib import Path
from pydose_rt.data import Patient, OptimizationConfig, MachineConfig, loaders, BeamSequence
from pydose_rt import DoseEngine
from pydose_rt.objectives.losses import compute_dvh_loss, scale_loss
from pydose_rt.objectives.metrics import dose_at_volume_percent
from pydose_rt.layers import BeamValidationLayer
from pydose_rt.utils.plotting import print_results, make_animation, print_paper_plot
from pydose_rt.objectives.metrics import result_validation
from pydose_rt.utils.utils import get_initial_weights
from dotenv import load_dotenv
import argparse
load_dotenv()  # will look for .env in project root

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if (os.path.exists("/mimer/NOBACKUP/groups/naiss2023-6-64/attila/miqa/")):
    remote = True
else:
    remote = False

# -----------------------------------------
# Parse command-line arguments
# -----------------------------------------
parser = argparse.ArgumentParser(description="Autoplan static optimization script")
parser.add_argument(
    "--patient_name",
    type=str,
    required=False,
    default="P01",
    help="Name of patient (e.g. P01)"
)
args = parser.parse_args()
patient_name = args.patient_name

if remote:
    base = Path(f"/mimer/NOBACKUP/groups/naiss2023-6-64/attila/GoldAtlasPlans/{patient_name}")

    ct_folder = base / "[CT] Deformed CT"
    rtplan_path = next((base / "[RP] CT").iterdir())
    rtdose_path = next((base / "[RD] CT Dose").iterdir())
    rtstruct_path = next((base / "[RS] RayStation").iterdir())

    patient, beam_sequence = loaders.load_dicom(
                ct_folder=ct_folder,
                dose_path=rtdose_path,
                plan_path=[ rtplan_path ],
                struct_path=rtstruct_path,
                struct_names=["CTVT", "PTV", "PenileBulb", "FemoralHead_L", "FemoralHead_R", "Bladder", "Rectum", "SeminalVesicles", "External"]
                )
    beam_sequence = beam_sequence[0]
    optimization = OptimizationConfig.from_json("src/pydose_rt/data/optimization_presets/gold-atlas.json")

    kernel_size = 55
    n_tests = 50
    device = device
    dtype = torch.float32

    max_iter = 150
else:
    base = Path(f"/home/bolo/Documents/PyDoseRT/test_data/GoldAtlasPlans/10X/{patient_name}")

    ct_folder = base / "[CT] Deformed CT"
    rtplan_path = next((base / "[RP] CT").iterdir())
    rtdose_path = next((base / "[RD] CT Dose").iterdir())
    rtstruct_path = next((base / "[RS] RayStation").iterdir())

    patient, beam_sequence = loaders.load_dicom(
                ct_folder=ct_folder, 
                dose_path=rtdose_path, 
                plan_path=rtplan_path, 
                struct_path=rtstruct_path,
                struct_names=["CTVT", "PTV", "PenileBulb", "FemoralHead_L", "FemoralHead_R", "Bladder", "Rectum", "SeminalVesicles", "External"]
                )
    beam_sequence: BeamSequence = beam_sequence[0]
    beam_sequence = beam_sequence.clone()[::16]

    optimization = OptimizationConfig.from_json("src/pydose_rt/data/optimization_presets/gold-atlas.json")

    kernel_size = 3
    n_tests = 1
    device = device
    dtype = torch.float32


    max_iter = 1

ptv_struct_name = [key for key in patient.structures.keys() if "PTV" in key][0]
machine_config = MachineConfig(
    preset="src/pydose_rt/data/machine_presets/umea_10MV.json",
    profile_corrections=None,
    output_factors=None,
    )

gantry_angles = torch.rad2deg(beam_sequence.gantry_angles)
number_of_leaf_pairs  = beam_sequence.num_leaf_pairs
field_size = beam_sequence.field_size
iso_center = beam_sequence.iso_center
collimator_angles = torch.rad2deg(beam_sequence.collimator_angles)
sid = beam_sequence.sid
open_field_size = 10.0  # np.random.uniform(10.0, 200.0)

print_stuff = 0
loss_plot = 1.0
best_results = []
patience_thr = max_iter
oar_dose = 10.0

for test_i in range(n_tests):
    experiment = Experiment(
        api_key=os.getenv("COMET_API"), project_name="autoplan_static"
    )
    try:
        current_res = { "loss": np.inf }
        weights = get_initial_weights()
        latest = {"raw_losses": None, "loss_val": None, "dose_pred": None, "pred_mlc": None, "pred_mus": None, "pred_jaws": None}
        optimization.structures['CTVT']["weight"] = 0.0
        optimization.structures['SeminalVesicles']["weight"] = 0.0
        optimization.structures[ptv_struct_name]["weight"] = 100.0
        optimization.structures['PenileBulb']["weight"] = 0.0
        optimization.structures['FemoralHead_L']["weight"] = 0.1
        optimization.structures['FemoralHead_R']["weight"] = 0.1
        optimization.structures['Bladder']["weight"] = np.random.choice([0.1, 1.0, 10.0])
        optimization.structures['Rectum']["weight"] = np.random.choice([0.1, 1.0, 10.0])
        optimization.structures['External']["weight"] = np.random.choice([0.1, 1.0, 10.0])


        beam_sequence = BeamSequence.create(
            gantry_angles_deg=gantry_angles,
            number_of_leaf_pairs=number_of_leaf_pairs,
            field_size=field_size,
            iso_center=iso_center,
            collimator_angles_deg=collimator_angles,
            sid=sid,
            open_field_size=open_field_size,
            device=device,
            dtype=dtype,
            requires_grad=True
            )
        # beam_sequence.jaw_positions.requires_grad_(False)

        patient = patient.to(device).to(dtype)
        ct_volume = patient.density_image.unsqueeze(0)
        dose_target = patient.dose.unsqueeze(0)
        
        engine = DoseEngine(
            machine_config=machine_config,
            dose_grid_spacing=patient.resolution,
            dose_grid_shape=patient.density_image.shape,
            beam_template=beam_sequence.to_delivery(), 
            kernel_size=kernel_size, 
            adjust_values=True,
            dtype=dtype, 
            device=device
        )
        engine.calibrate(machine_config.calibration_mu, beam_sequence.to_delivery())
        valid_parameters_layer = BeamValidationLayer(
            machine_config=machine_config, 
            device=device,
            dtype=dtype,
            field_size=beam_sequence.field_size
        )
        
        patience = 0
        epoch = 0
        lr = 1.0
        optimizer = torch.optim.AdamW(beam_sequence.parameters(),
                                      lr=lr,
                                      weight_decay=1e-4
                                      )
        # lr = np.random.choice([0.1, 0.05, 0.01, 0.005, 0.001])
        # optimizer = torch.optim.LBFGS(
        #     beam_sequence.parameters(), 
        #     lr=lr,
        #     history_size=10,  # Reduce history (default 100)
        #     line_search_fn='strong_wolfe'  # More conservative line search
        #     )

        experiment.log_parameters(
            {
                "patient_name": patient_name,
                "lr_0": lr,
                "kernel_size": engine.kernel_size,
                "weights": weights,
                "physical_size": patient.physical_size,
                "roi_weights": optimization.get_parameters("weight")
            }, nested_support=True
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            st = time.time()
            # Forward
            dose_pred = engine.compute_dose(
                beam_sequence.to_delivery(),
                density_image=ct_volume
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"{time.time() - st}s for forward pass")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            st = time.time()
            # Compute loss
            dose_pred_loss = dose_pred[0] * 7
            raw_losses = []
            # PTV_Prostata_gol_4270

            if ptv_struct_name in patient.structures.keys():
                raw_losses.append(scale_loss(torch.mean(torch.abs(dose_pred_loss[patient.structures[ptv_struct_name]] - 42.7)), optimization.structures[ptv_struct_name]["weight"]))

            for struct_name in ['PenileBulb', 'FemoralHead_L', 'FemoralHead_R', 'Bladder', 'Rectum', 'External']:
                if struct_name in patient.structures.keys():
                    raw_losses.append(scale_loss(torch.mean(torch.abs(dose_pred_loss[patient.structures[struct_name]])), optimization.structures[struct_name]["weight"]))

            raw_losses.append(0.001 * torch.mean(torch.abs(beam_sequence.leaf_positions)))
            loss = torch.stack(raw_losses).sum()
            
            # Backprop
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(beam_sequence.leaf_positions, max_norm=1 / 40.0)
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
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            st = time.time()
            # --- the actual optimizer step ---
            loss = optimizer.step(closure)   # returns the last loss the closure returned
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"{time.time() - st}s for forward+backward pass")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            st = time.time()
            # scheduler.step(loss)
            raw_losses = latest["raw_losses"]
            raw_loss_dict = {f"loss_{i+1}": v for i, v in enumerate(raw_losses)}
            dose_pred = latest["dose_pred"]
            loss_val = latest["loss_val"]
            beam_sequence = latest["beam_sequence"]
            mae_loss = np.round(torch.mean(torch.abs((dose_target - dose_pred))[dose_target > (0.1 * dose_target.max())]).cpu().detach().numpy(), 4)

            mask_oar = patient.structures["PenileBulb"].clone()
            for struct_name in ["FemoralHead_L", "FemoralHead_R", "Bladder", "Rectum", "SeminalVesicles"]:
                if struct_name in patient.structures.keys():
                    mask_oar += patient.structures[struct_name]

            patience += 1
            if (loss < current_res["loss"]):
                patience = 0
                current_res = {
                    "loss": loss, 
                    "weights": weights, 
                    "beam_sequence": beam_sequence.clone(),
                }
            else:
                # print("Patience count:", patience)
                if ((patience >= patience_thr) | torch.isnan(dose_pred).any()):
                    best_results.append(current_res)
                    print("Best result for this test:", current_res)
                    break

            proxy_pred = (7*dose_pred).cpu().detach().numpy() > optimization.prescription_gy
            proxy_dice = 2 * np.sum(proxy_pred * patient.structures["PTV"].cpu().detach().numpy()) / (np.sum(proxy_pred) + np.sum(patient.structures["PTV"].cpu().detach().numpy()))
            lr_now = lr # scheduler.get_last_lr()[0]
            experiment.log_metrics(
                {
                    "loss": loss.item(),
                    "dose_mae": mae_loss,
                    "lr": lr_now,
                    "mae_loss": mae_loss,
                    "proxy_dice": proxy_dice,
                    "Rectum_D_mean": 7*dose_pred[0, patient.structures["Rectum"]].mean().item(),
                    "Bladder_D_mean": 7*dose_pred[0, patient.structures["Bladder"]].mean().item(),
                    "PTV_D_95": 7*dose_at_volume_percent(dose_pred[0].cpu().detach().numpy(), patient.structures[ptv_struct_name].cpu().detach().numpy(), 95) if ptv_struct_name in patient.structures.keys() else 0.0,
                    "PTV_D_98": 7*dose_at_volume_percent(dose_pred[0].cpu().detach().numpy(), patient.structures[ptv_struct_name].cpu().detach().numpy(), 98) if ptv_struct_name in patient.structures.keys() else 0.0,
                    "Rectum_D_mean_ref": 7*patient.dose[patient.structures["Rectum"]].mean().item(),
                    "Bladder_D_mean_ref": 7*patient.dose[patient.structures["Bladder"]].mean().item(),
                    "PTV_D_95_ref": 7*dose_at_volume_percent(patient.dose.cpu().detach().numpy(), patient.structures[ptv_struct_name].cpu().detach().numpy(), 95) if ptv_struct_name in patient.structures.keys() else 0.0,
                    "PTV_D_98_ref": 7*dose_at_volume_percent(patient.dose.cpu().detach().numpy(), patient.structures[ptv_struct_name].cpu().detach().numpy(), 98) if ptv_struct_name in patient.structures.keys() else 0.0,
                    "mae_ptv": torch.mean(torch.abs(dose_pred[0, patient.structures[ptv_struct_name]] - 6.1)).item() if ptv_struct_name in patient.structures.keys() else 0.0,
                    "mae_ctv": torch.mean(torch.abs(dose_pred[0, patient.structures["CTVT"]] - 6.1)).item() if "CTVT" in patient.structures.keys() else 0.0,
                    "mae_penilebulb": torch.mean(torch.abs(dose_pred[0, patient.structures["PenileBulb"]])).item() if "PenileBulb" in patient.structures.keys() else 0.0,
                    "mae_femoralhead_l": torch.mean(torch.abs(dose_pred[0, patient.structures["FemoralHead_L"]])).item() if "FemoralHead_L" in patient.structures.keys() else 0.0,
                    "mae_femoralhead_r": torch.mean(torch.abs(dose_pred[0, patient.structures["FemoralHead_R"]])).item() if "FemoralHead_R" in patient.structures.keys() else 0.0,
                    "mae_bladder": torch.mean(torch.abs(dose_pred[0, patient.structures["Bladder"]])).item() if "Bladder" in patient.structures.keys() else 0.0,
                    "mae_rectum": torch.mean(torch.abs(dose_pred[0, patient.structures["Rectum"]])).item() if "Rectum" in patient.structures.keys() else 0.0,
                    "mae_seminalvesicles": torch.mean(torch.abs(dose_pred[0, patient.structures["SeminalVesicles"]])).item() if "SeminalVesicles" in patient.structures.keys() else 0.0,
                    "mae_oar" : torch.mean(torch.abs(dose_pred[0, mask_oar])).item(),
                    "mae_external": torch.mean(torch.abs(dose_pred[0, patient.structures["External"]])).item() if "External" in patient.structures.keys() else 0.0,
                    **raw_loss_dict,
                },
                epoch=epoch,
            )

            epoch += 1

        print(f"Optimization finished in {int(time.time() - start_time)}s.")
        beam_sequence = current_res["beam_sequence"]
        animation_sequence = beam_sequence.clone()
        pred_mlc = beam_sequence.leaf_positions
        pred_mus = beam_sequence.mus
        pred_jaws = beam_sequence.jaw_positions

        pred_mlc_valid, pred_jaws_valid, pred_mus_valid = valid_parameters_layer(
            pred_mlc, pred_mus, pred_jaws
        )
        beam_sequence.leaf_positions = pred_mlc_valid
        beam_sequence.mus = pred_mus_valid
        beam_sequence.jaw_positions = pred_jaws_valid

        results = result_validation(patient, machine_config, beam_sequence.to('cpu'), dose_pred[0].to('cpu'), optimization, compute_gamma=False, compute_clinical_criteria=True)
        experiment.log_metrics(
            {
                "results": results,
            },
            epoch=epoch,
        )
        
        title = f"MAE - {str(mae_loss)} Gy\nTest #{len([0])}: {[str(np.round(v, 4)) for v in [raw_losses]]}"
        experiment.log_asset_data(beam_sequence.leaf_positions.cpu().detach().numpy(), "mlc_positions.npy")
        experiment.log_asset_data(beam_sequence.mus.cpu().detach().numpy(), "mu_values.npy")
        experiment.log_asset_data(dose_pred[0].cpu().detach().numpy(), "dose.npy")
        print_results(experiment, optimization, patient, beam_sequence, dose_pred[0], title, plot_ct=True, preset="gold-atlas")
        print_paper_plot(experiment, optimization, patient, 7*dose_pred[0]) # dose_pred[0]
        make_animation(experiment, patient, engine, animation_sequence, dose_max=7.0)
    except Exception as e:
        print("Exception during test:", e)
        
    experiment.end()
