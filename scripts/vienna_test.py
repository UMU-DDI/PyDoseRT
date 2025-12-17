import os
import time
from pydose_rt.data.beam import BeamSequence
from pathlib import Path
import pandas as pd
from pydose_rt.data import MachineConfig, Patient, OptimizationConfig, loaders
from pydose_rt.objectives.metrics import result_validation
from pydose_rt.utils.utils import find_patient_paths
import numpy as np
from pydose_rt import DoseEngine
from scipy.ndimage import binary_fill_holes, binary_erosion
from pydose_rt.utils.plotting import print_comparison_plot, print_results, make_animation, quick_plot
import torch

optimization = OptimizationConfig.from_json("src/pydose_rt/data/optimization_presets/vienna.json",)
machine_config = MachineConfig(
    preset="src/pydose_rt/data/machine_presets/vienna_10MV.json",
    profile_corrections=None,
    output_factors=None,
    head_scatter_amplitude=None,
    head_scatter_sigma=None,
    mlc_transmission=0.0
    )
all_results = []
kernel_size = 51
base_path = Path('/home/bolo/Documents/PyDoseRT/test_data/transfer_files/')
for patient_name in sorted(os.listdir(base_path)):
    try:
        patient_dir = base_path / patient_name
        ct_folder, rtplan_path, rtdose_path, rtstruct_path = find_patient_paths(patient_dir)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        if torch.cuda.is_available():
            torch.cuda.device(device)
            torch.randn(1, device=device)  # triggers context creation
            torch.cuda.synchronize()


        patient, beam_sequences = loaders.load_dicom(
                    ct_folder=ct_folder,
                    dose_path=rtdose_path,
                    plan_path=rtplan_path,
                    struct_path=rtstruct_path,
                    new_spacing=(3.0, 3.0, 3.0),
                    struct_names=["CTV", "PTV", "FemoralHead_L", "FemoralHead_R", "Bladder", "Rectum", "Body"],
                    use_delivery=True
                    )
        patient.dose = patient.dose * patient.structures["Body"]
        beam_sequence = beam_sequences[0]
        # beam_sequence = beam_sequence[0:2]

        ptv_struct_name = [key for key in patient.structures.keys() if "PTV" in key][0]
        patient = patient.to(device).to(dtype)
        dose_volume = patient.dose
        density_image = torch.where(patient.structures["Body"], patient.density_image, 0.0)
        # ct_volume = patient.get_masked_ct("Body")
        # dose_volume = patient.get_masked_dose("Body")

        doses = []
        beam_sequence = beam_sequence.to(device).to(dtype)
        dose_engine = DoseEngine(
            kernel_size=kernel_size,
            dose_grid_spacing=patient.resolution,
            machine_config=machine_config,
            dose_grid_shape=density_image.shape,
            beam_template=beam_sequence,
            # use_multislab=True,
            device=device,
            dtype=dtype
        )

        dose_engine.calibrate(
            calibration_mu=machine_config.calibration_mu,
            original_beam_template=beam_sequence
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        st = time.time()
        dose_pred = dose_engine.compute_dose_sequential(beam_sequence, density_image=density_image).detach()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"Dose computation in {time.time() - st} seconds.")
        # np.savez(f"out/dose_{patient_name}.npy", dose_pred.cpu().numpy())

        ext_mask = binary_erosion(binary_fill_holes(patient.structures["Body"].cpu().detach().numpy()), np.ones((3, 3, 3)), iterations=7)
        ext_mask *= (dose_volume > 0.1 * dose_volume.max()).cpu().detach().numpy()
        dose_pred = torch.where(patient.structures["Body"], dose_pred[0], 0.0)
        scale = dose_volume[patient.structures[ptv_struct_name] > 0].mean() / dose_pred[patient.structures[ptv_struct_name] > 0].mean()
        print(scale)
        # dose_pred = dose_pred * scale

        dose_max = patient.number_of_fractions * max(dose_volume.max(), dose_pred.max()).item()

        mae_map = torch.abs(dose_pred - dose_volume)
        mae_loss = np.mean(10.0*mae_map[ext_mask].cpu().detach().numpy())
        res_string = f"Patient {patient_name}: MAE {str(np.round(mae_loss, 4))}"
        print(res_string)

        leafs = beam_sequence.leaf_positions.unsqueeze(0)
        mus = beam_sequence.mus.unsqueeze(0)
        jaws = beam_sequence.jaw_positions.unsqueeze(0)
        res = result_validation(patient, machine_config, beam_sequence, dose_pred, optimization, compute_gamma=True, compute_clinical_criteria=True, global_normalisation=None, gamma_threshold_distance=3.0, gamma_threshold_dose=3.0)
        if "clinical_criteria" in res.keys():
            print(f"Passed {int(100*res['clinical_criteria']['passed_test'])}% of clinical criteria.")
        if "gamma_pass_rate" in res.keys():
            res_string += f" Gamma pass rate {str(np.round(res['gamma_pass_rate'], 2))}"

        print(res_string)
        quick_plot(patient, dose_pred, title=res_string, out_path=f"out/quick_{patient_name}.png")
        print_comparison_plot(optimization, patient, dose_pred, out_path=f"out/comparison_{patient_name}.png")

        row = {"patient_name": patient_name}
        row.update(res)        # Adds all scalar keys from res
        all_results.append(row)

        title = f"MAE - {str(mae_loss)} Gy\nTest #{len([0])}: {[str(np.round(v, 4)) for v in [mae_loss]]}"
        print_results(
            None, 
            optimization, 
            patient, 
            beam_sequence, 
            dose_pred, 
            title=title, 
            preset="gold-atlas",
            out_path=f"out/final_{patient_name}.png"
            )

        # make_animation(
        #     None, 
        #     patient, 
        #     dose_engine, 
        #     beam_sequence,
        #     dose_max=dose_max,
        #     out_path=f"out/video_{patient_name}.mp4"
        #     )
        
        del dose_engine, dose_pred, dose_volume, patient
    except Exception as e:
        print(e)
        
df = pd.DataFrame(all_results)
df.to_csv("out/vienna_results_summary.csv", index=False)
