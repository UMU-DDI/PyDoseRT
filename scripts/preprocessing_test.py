import os

from pydose_rt.data.optimization_config import OptimizationConfig
from pathlib import Path
import pandas as pd
from pydose_rt.data import MachineConfig, Patient, OptimizationConfig, loaders
from pydose_rt.objectives.metrics import result_validation
from pydose_rt.utils.utils import find_patient_paths
import numpy as np
from pydose_rt import DoseEngine
from pydose_rt.utils.plotting import print_results, make_animation, quick_plot
import torch

all_results = []
np_folder = "/home/bolo/Downloads/Josef_Testing/large/MrAlderson_test_loc_large_1/"
base_path = Path('/home/bolo/Downloads/Josef_Testing/Data/') # /home/bolo/Documents/PyDoseRT/test_data/GoldAtlasPlans/NODES/
for patient_name in sorted(os.listdir(base_path)):
    try:
        patient_dir = base_path / patient_name
        ct_folder, rtplan_path, rtdose_path, rtstruct_path = find_patient_paths(patient_dir)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        kernel_size = 151
        downsampling_factor = (1, 1, 1)

        patient, beam_sequences = loaders.load_dicom(
                    ct_folder=ct_folder, 
                    dose_path=rtdose_path, 
                    plan_path=rtplan_path, 
                    struct_path=rtstruct_path,
                    struct_names=["CTV", "PTV", "FemoralHead_L", "FemoralHead_R", "Bladder", "Rectum", "External"],
                    use_delivery=True
                    )
        preprocessed_dose = np.load(np_folder + "Dose.npy")
        preprocessed_beams = np.load(np_folder + "Beam.npy", allow_pickle=True)
        preprocessed_ct = np.load(np_folder + "CT.npy")
        preprocessed_structures = np.load(np_folder + "StructureSet.npy")
        optimization = OptimizationConfig.from_json("src/pydose_rt/data/optimization_presets/gold-atlas.json")

        ptv_struct_name = [key for key in patient.structures.keys() if "PTV" in key][0]
        machine_config = MachineConfig(
            preset="src/pydose_rt/data/machine_presets/umea_10MV.json",
            number_of_leaf_pairs=80,
            leaf_widths=None,
            head_scatter_amplitude=None,
            head_scatter_sigma=None,
            profile_corrections=None,
            mlc_transmission=0.0
            )
            
        patient = patient.to(device).to(dtype)
        dose_volume = patient.dose
        # ct_volume = patient.get_masked_ct("External")
        # dose_volume = patient.get_masked_dose("External")

        doses = []
        for beam_sequence in beam_sequences:
            beam_sequence = beam_sequence.to(device).to(dtype)
            dose_engine = DoseEngine(kernel_size=kernel_size,
                                     resolution=patient._resolution,
                                     machine_config=machine_config,
                                     image_template=patient.density_image,
                                     beam_template=beam_sequence,
                                     device=device,
                                     dtype=dtype
                                    )
            
            dose_engine.calibrate(calibration_mu=machine_config.calibration_mu,
                                  original_beam_template=beam_sequence)

            dose_pred = dose_engine.compute_dose_sequential(beam_sequence, ct_image=patient.density_image)
            doses.append(dose_pred.detach())
        dose_pred = sum(doses)[0]
        # dose_pred = torch.where(patient.structures["External"], dose_pred[0], 0.0)
        dose_pred = 1.14 * dose_pred# * dose_volume[patient.structures["PTV"] > 0].mean() / dose_pred[patient.structures["PTV"] > 0].mean()

        dose_max = max(dose_volume.max(), dose_pred.max()).item()

        mae_map = torch.abs(dose_pred - dose_volume)
        mae_loss = np.mean(torch.mean(mae_map).item())
        res_string = f"Patient {patient_name}:\tMAE {str(np.round(mae_loss, 4))}"
        print(res_string)

        # print(scale.item())
        # print(mae_loss)
        leafs = beam_sequence.leaf_positions.unsqueeze(0)
        mus = beam_sequence.mus.unsqueeze(0)
        jaws = beam_sequence.jaw_positions.unsqueeze(0)
        res = result_validation(patient, machine_config, beam_sequence, dose_pred, optimization, compute_gamma=True, compute_clinical_criteria=False, global_normalisation=None)

        # print(f"Passed {int(100*res['clinical_criteria']['passed_test'])}% of clinical criteria.")
        res_string += f" Gamma pass rate {str(np.round(res['gamma_pass_rate'], 2))}"

        print(res_string)
        quick_plot(patient, dose_pred, res_string, dose_max, f"out/quick_{patient_name}.png")

        row = {"patient_name": patient_name}
        row.update(res)        # Adds all scalar keys from res
        all_results.append(row)

        # print_results(None, optimization, [0.0], patient, beam_sequence, [], dose_pred, mae_loss, dose_max=dose_max, out_path=f"out/final_{patient_name}.png")

        # make_animation(None, 
        #                treatment, 
        #                machine_config, 
        #                patient, 
        #                dose_layer, 
        #                (leafs[:, :, :-1, :] + leafs[:, :, 1:, :]) / 2, 
        #                (mus[:, :-1] + mus[:, 1:]) / 2, 
        #                (jaws[:, :, :-1] + jaws[:, :, 1:]) / 2,
        #                dose_max
        #                )
        del dose_engine, dose_pred, dose_volume, patient, optimization
    except Exception as e:
        print(e)
        
df = pd.DataFrame(all_results)
print(df.to_string(index=False))
df.to_csv("out/results_summary.csv", index=False)
