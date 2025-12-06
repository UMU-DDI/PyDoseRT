from comet_ml import Experiment
import sys
sys.path.append('../')
import numpy as np
import os
import torch
from torch.amp import autocast, GradScaler  # AMP for stable float16 training
import time
from pathlib import Path
from pydose_rt.data import Patient, OptimizationConfig, MachineConfig, loaders, BeamSequence
from pydose_rt import DoseEngine
from pydose_rt.objectives.losses import compute_dvh_loss
from pydose_rt.layers import BeamValidationLayer
from pydose_rt.utils.plotting import print_results, make_animation
from pydose_rt.objectives.metrics import result_validation
from pydose_rt.utils.utils import get_initial_weights
from dotenv import load_dotenv
import argparse
load_dotenv()  # will look for .env in project root

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
                struct_names=["CTVT", "PTVT_42.7", "PenileBulb", "Prostate", "FemoralHead_L", "FemoralHead_R", "Bladder", "Rectum", "SeminalVesicles", "External"]
                )
    beam_sequence = beam_sequence[0].clone()[::2]
    optimization = OptimizationConfig.from_json("src/pydose_rt/data/optimization_presets/gold-atlas.json")

    kernel_size = 5
    device = device
    # CRITICAL: Parameters must be float32 for AMP to work correctly
    # AMP will automatically use float16 for forward pass (saves memory on activations)
    # but keeps parameters and gradients in float32 (prevents NaN gradients)
    param_dtype = torch.float32  # Parameters in float32
    data_dtype = torch.float16   # Input data can be float16 for memory savings
    use_amp = True  # Use AMP to prevent NaN gradients

    max_iter = 1000
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
                struct_names=["CTVT", "PTVT_42.7", "PenileBulb", "Prostate", "FemoralHead_L", "FemoralHead_R", "Bladder", "Rectum", "SeminalVesicles", "External"]
                )
    beam_sequence: BeamSequence = beam_sequence[0]
    beam_sequence = beam_sequence[::16].clone()

    optimization = OptimizationConfig.from_json("src/pydose_rt/data/optimization_presets/gold-atlas.json")

    kernel_size = 3
    device = device
    param_dtype = torch.float32
    data_dtype = torch.float16
    use_amp = False  # Dev machine doesn't need AMP

    max_iter = 10

machine_config = MachineConfig(
    preset="src/pydose_rt/data/machine_presets/umea_10MV.json",            
    penumbra_fwhm=None,
    head_scatter_amplitude=None,
    head_scatter_sigma=None,
    profile_corrections=None,
    output_factors=None,
    )

gantry_angles = beam_sequence.gantry_angles
number_of_leaf_pairs  = beam_sequence.num_leaf_pairs
field_size = beam_sequence.field_size
iso_center = beam_sequence.iso_center
collimator_angles = beam_sequence.collimator_angles
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
        optimization.structures['CTVT']["weight"] = 100
        optimization.structures['PTVT_42.7']["weight"] = 100
        optimization.structures['PenileBulb']["weight"] = np.random.choice([0.0, 0.01, 0.1, 1.0, 10.0, 100.0])
        optimization.structures['Prostate']["weight"] = np.random.choice([0.0, 0.01, 0.1, 1.0, 10.0, 100.0])
        optimization.structures['FemoralHead_L']["weight"] = np.random.choice([0.0, 0.01, 0.1, 1.0, 10.0, 100.0])
        optimization.structures['FemoralHead_R']["weight"] = np.random.choice([0.0, 0.01, 0.1, 1.0, 10.0, 100.0])
        optimization.structures['Bladder']["weight"] = np.random.choice([0.0, 0.01, 0.1, 1.0, 10.0, 100.0])
        optimization.structures['Rectum']["weight"] = np.random.choice([0.0, 0.01, 0.1, 1.0, 10.0, 100.0])
        optimization.structures['SeminalVesicles']["weight"] = np.random.choice([0.0, 0.01, 0.1, 1.0, 10.0, 100.0])
        optimization.structures['External']["weight"] = np.random.choice([0.0, 0.01, 0.1, 1.0, 10.0, 100.0])


        beam_sequence = BeamSequence.create(
            gantry_angles=gantry_angles,
            number_of_leaf_pairs=number_of_leaf_pairs,
            field_size=field_size,
            iso_center=iso_center,
            collimator_angles=collimator_angles,
            sid=sid,
            open_field_size=open_field_size,
            device=device,
            dtype=data_dtype,  # Parameters in float32 for AMP stability
            requires_grad=True
            )
        beam_sequence.jaw_positions.requires_grad_(False)

        patient = patient.to(device).to(data_dtype)  # Input data can be float16
        ct_volume = patient.density_image.unsqueeze(0)
        dose_target = patient.dose.unsqueeze(0)

        # Debug: Print actual dtypes being used
        print("\n" + "="*60)
        print("DTYPE CHECK")
        print("="*60)
        print(f"remote: {remote}")
        print(f"param_dtype (parameters): {param_dtype}")
        print(f"data_dtype (input data): {data_dtype}")
        print(f"use_amp: {use_amp}")
        print(f"beam_sequence.leaf_positions.dtype: {beam_sequence.leaf_positions.dtype}")
        print(f"beam_sequence.mus.dtype: {beam_sequence.mus.dtype}")
        print(f"ct_volume.dtype: {ct_volume.dtype}")
        print(f"dose_target.dtype: {dose_target.dtype}")
        print("="*60 + "\n")
        
        engine = DoseEngine(
            machine_config=machine_config,
            dose_grid_spacing=patient._resolution,
            dose_grid_shape=patient.density_image.shape,
            beam_template=beam_sequence.to_delivery(),
            kernel_size=kernel_size,
            adjust_values=True,
            dtype=param_dtype,  # Engine parameters in float32
            device=device
        )
        valid_parameters_layer = BeamValidationLayer(
            machine_config=machine_config,
            device=device,
            dtype=param_dtype,  # Validation layer in float32
            adjust_values=True,
            field_size=beam_sequence.field_size
        )
        
        patience = 0
        epoch = 0
        lr = 10**np.random.uniform(-2, 1) # 0.1
        lr_decay = 1e-4
        optimizer = torch.optim.AdamW(beam_sequence.parameters(), lr=lr, weight_decay=lr_decay)

        # AMP: GradScaler prevents NaN gradients in float16 backward pass
        scaler = GradScaler(device.type, enabled=use_amp)

        experiment.log_parameters(
            {
                "patient_name": patient_name,
                "lr_0": lr,
                "kernel_size": engine.kernel_size,
                "lr_decay": lr_decay,
                "weights": weights,
                "physical_size": patient.physical_size,
                "roi_weights": optimization.get_parameters("weight"),
                "amp_enabled": use_amp,
                "param_dtype": str(param_dtype),
                "data_dtype": str(data_dtype)
            }, nested_support=True
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)

            # Check leaf positions BEFORE forward pass
            if torch.isnan(beam_sequence.leaf_positions).any():
                print(f"\n💀 FATAL: Leaf positions already NaN at start of epoch {epoch}!")
                print(f"   This happened during previous optimizer step.")
                raise RuntimeError("Leaf positions became NaN - stopping training")

            # AMP: Autocast runs forward pass in float16 to save memory on activations
            # Parameters stay in float32, gradients computed in float32 for stability
            # This is the correct way to use mixed precision training
            with autocast(device.type, enabled=use_amp, dtype=torch.float16):
                # Forward pass - will run in float16 if use_amp=True
                dose_pred = engine.compute_dose(
                    beam_sequence.to_delivery(),
                    density_image=ct_volume
                )

                # Check dose after forward pass
                if torch.isnan(dose_pred).any():
                    print(f"\n⚠️  NaN in dose_pred after forward pass at epoch {epoch}!")
                    print(f"   Issue is in forward pass, not backward pass")
                    raise RuntimeError("NaN in forward pass")

                # Compute loss - also in float16 inside autocast
                raw_losses = compute_dvh_loss(patient, optimization, machine_config, dose_pred[0], dose_target, beam_sequence, weights)
                loss = torch.stack(raw_losses).sum()

            # Check loss
            if torch.isnan(loss):
                print(f"\n⚠️  NaN in loss at epoch {epoch}!")
                print(f"   Raw losses: {[l.item() for l in raw_losses]}")
                raise RuntimeError("NaN in loss computation")

            # AMP: Scale loss before backward to prevent gradient underflow in float16
            # This is safe because parameters are float32, so gradients will be float32
            scaler.scale(loss).backward()

            # AMP: Unscale gradients back to normal range before clipping
            # This works because gradients are float32 (from float32 parameters)
            scaler.unscale_(optimizer)

            # Debug: Check for NaN in gradients IMMEDIATELY after backward
            if beam_sequence.leaf_positions.grad is not None:
                has_nan_grad = torch.isnan(beam_sequence.leaf_positions.grad).any()
                if has_nan_grad:
                    print(f"\n⚠️  NaN detected in leaf_positions GRADIENT at epoch {epoch}!")
                    print(f"   This is a BACKWARD PASS issue (server GPU likely doesn't support float16 properly)")
                    print(f"   Loss value: {loss.item():.6f} (valid)")
                    print(f"   Loss dtype: {loss.dtype}")
                    print(f"   Dose pred dtype: {dose_pred.dtype}")
                    print(f"   Leaf positions dtype: {beam_sequence.leaf_positions.dtype}")
                    print(f"   Leaf positions valid: {not torch.isnan(beam_sequence.leaf_positions).any()}")
                    print(f"   Gradient dtype: {beam_sequence.leaf_positions.grad.dtype}")

                    # Don't try to compute norm of NaN gradient
                    num_nan = torch.isnan(beam_sequence.leaf_positions.grad).sum().item()
                    total = beam_sequence.leaf_positions.grad.numel()
                    print(f"   NaN count: {num_nan}/{total} gradient elements")

                    # Save state for debugging
                    torch.save({
                        'epoch': epoch,
                        'loss': loss.item(),
                        'leaf_positions': beam_sequence.leaf_positions.detach().cpu(),
                        'gradient': beam_sequence.leaf_positions.grad.detach().cpu(),
                        'dose_pred': dose_pred.detach().cpu(),
                    }, f'/tmp/nan_debug_epoch_{epoch}.pt')
                    print(f"   Saved debug info to /tmp/nan_debug_epoch_{epoch}.pt")
                    print(f"\n   SOLUTION: Use torch.cuda.amp.autocast() or switch to float32")
                    raise RuntimeError("NaN gradients from backward pass - GPU incompatibility")

            # Relaxed gradient clipping for float16 stability (1/40=0.025 is too small for float16)
            torch.nn.utils.clip_grad_norm_(beam_sequence.leaf_positions, max_norm=1.0)
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
            loss = optimizer.step(closure)

            # AMP: Update scaler after optimizer step
            scaler.update()

            # scheduler.step(loss)
            raw_losses = latest["raw_losses"]
            raw_loss_dict = {f"loss_{i+1}": v for i, v in enumerate(raw_losses)}
            dose_pred = latest["dose_pred"]
            loss_val = latest["loss_val"]
            beam_sequence = latest["beam_sequence"]
            mae_loss = np.round(torch.mean(torch.abs((dose_target - dose_pred))).cpu().detach().numpy(), 4)
            
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
        make_animation(experiment, patient, engine, animation_sequence, dose_max=7.0)
    except Exception as e:
        print("Exception during test:", e)
        
    experiment.end()