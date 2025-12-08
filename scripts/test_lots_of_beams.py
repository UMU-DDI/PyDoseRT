import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from pydose_rt.data import MachineConfig, Patient, OptimizationConfig, loaders
from pydose_rt.objectives.metrics import result_validation
from pydose_rt.utils.utils import find_patient_paths
import numpy as np
from pydose_rt import DoseEngine
from pydose_rt.utils.plotting import print_results, make_animation, quick_plot
import torch

all_results = []
base_path = Path('/home/bolo/Documents/PyDoseRT/test_data/LotsOfBeams/')
for patient_name in sorted(os.listdir(base_path)):
    try:
        patient_dir = base_path / patient_name
        ct_folder, rtplan_paths, rtdose_paths, rtstruct_path = find_patient_paths(patient_dir)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        kernel_size = 201

        patient, beam_sequences = loaders.load_dicom(
                    ct_folder=ct_folder, 
                    dose_path=rtdose_paths, 
                    plan_path=rtplan_paths, 
                    struct_path=rtstruct_path,
                    struct_names=[],
                    use_delivery=False
                    )
        optimization = OptimizationConfig.from_json("src/pydose_rt/data/optimization_presets/gold-atlas.json")

        # ptv_struct_name = [key for key in patient.structures.keys() if "PTV" in key][0]
        machine_config = MachineConfig(
            preset="src/pydose_rt/data/machine_presets/umea_10MV.json",
            # head_scatter_amplitude=None,
            # head_scatter_sigma=None,
            # profile_corrections=None,
            # output_factors=None,
            )
            
        patient = patient.to(device).to(dtype)
        dose_volume = patient.dose
        density_image = patient.density_image

        doses = []
        for beam_sequence in beam_sequences:
            # beam_sequence.iso_center = (165.0, 100.0, 215.0)
            beam_sequence.iso_center = beam_sequence.iso_center
            beam_sequence = beam_sequence.to(device).to(dtype)
            dose_engine = DoseEngine(kernel_size=kernel_size,
                                     dose_grid_spacing=patient.resolution,
                                     machine_config=machine_config,
                                     dose_grid_shape=patient.density_image,
                                     beam_template=beam_sequence,
                                     device=device,
                                     dtype=dtype
                                    )
            
            dose_engine.calibrate(calibration_mu=machine_config.calibration_mu,
                                  original_beam_template=beam_sequence)

            dose_pred = dose_engine.compute_dose(beam_sequence, ct_image=density_image)
            doses.append(dose_pred.detach())
        # dose_pred = sum(doses)
        # dose_pred = torch.where(patient.structures["External"], dose_pred[0], 0.0)
        # dose_pred = dose_pred * dose_volume[patient.structures["PTV_56"] > 0].mean() / dose_pred[patient.structures["PTV_56"] > 0].mean()

        dose_max = max(dose_volume.max(), dose_pred.max()).item()

        mae_map = torch.abs(dose_pred - dose_volume)
        mae_loss = np.mean(torch.mean(mae_map).item())
        res_string = f"Patient {patient_name}: MAE {str(np.round(mae_loss, 4))}"
        print(res_string)

        # print(scale.item())
        # print(mae_loss)
        # res = result_validation(patient, machine_config, beam_sequence, dose_pred, optimization, compute_gamma=False, compute_clinical_criteria=False, global_normalisation=2.2)

        # print(f"Passed {int(100*res['clinical_criteria']['passed_test'])}% of clinical criteria.")
        # res_string += f" Gamma pass rate {str(np.round(res['gamma_pass_rate'], 2))}"

        patient_name += f"_beam_4"
        print(res_string)
        quick_plot(patient, dose_pred[0], title=res_string, show_ct=True, out_path=f"out/quick_{patient_name}.png")
        
        center_x, center_y, center_z = np.array(dose_volume.shape) // 2

        plt.figure()
        plt.subplot(311)
        plt.plot(dose_volume[:, center_y, center_z].cpu().detach().numpy(), linestyle='--', color='gray', label='RS')
        plt.plot(dose_pred[0, :, center_y, center_z].cpu().detach().numpy(), linestyle='-', color='orange', label='PDRT')

        plt.subplot(312)
        plt.plot(dose_volume[center_x, :, center_z].cpu().detach().numpy(), linestyle='--', color='gray', label='RS')
        plt.plot(dose_pred[0, center_x, :, center_z].cpu().detach().numpy(), linestyle='-', color='orange', label='PDRT')

        plt.subplot(313)
        plt.plot(dose_volume[center_x, center_y, :].cpu().detach().numpy(), linestyle='--', color='gray', label='RS')
        plt.plot(dose_pred[0, center_x, center_y, :].cpu().detach().numpy(), linestyle='-', color='orange', label='PDRT')
        
        plt.savefig(f"out/profiles_{patient_name}.png")
        plt.close()


        title = f"MAE - {str(mae_loss)} Gy\nTest #{len([0])}: {[str(np.round(v, 4)) for v in [mae_loss]]}"
        print_results(None, optimization, patient, beam_sequence, dose_pred, title=title, out_path=f"out/final_{patient_name}.png")

        # make_animation(None, 
        #                treatment, 
        #                patient, 
        #                dose_layer, 
        #               beam_sequence
        #                dose_max
        #                )
        del dose_engine, dose_pred, dose_volume, patient, optimization
    except Exception as e:
        print(e)