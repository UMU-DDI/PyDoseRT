from comet_ml import Experiment
from re import M
import sys
import os
import torch.nn.functional as F

from pydose_rt.data.optimization_config import OptimizationConfig
sys.path.append('../')
sys.path.append('../../')
import pydicom
import time
from pathlib import Path
import math
import nibabel as nib

from pydicom.data import get_testdata_file
from pydose_rt.data import MachineConfig, Patient, OptimizationConfig, loaders
# from pydose_rt.data import MachineConfig
from pydose_rt.objectives.metrics import result_validation
from pydose_rt.utils.utils import find_patient_paths
import numpy as np
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, rotate
from pydose_rt import DoseEngine
import SimpleITK as sitk
from pydose_rt.utils.plotting import print_results, make_animation, quick_plot
import torch



# Set paths
# patient_name = "0e54d72a21"
# ct_folder = f"/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/{patient_name}/"
# rtstruct_path = next((f for f in Path(ct_folder).iterdir() if "RS" in f.name.upper() or "RTSTRUCT" in f.name.upper()), None)
# rtplan_path = f"/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/{patient_name}_plans/1ARC/RP1.2.752.243.1.1.20251031145134399.7000.37887.dcm"
# rtdose_path = f"/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/{patient_name}_plans/1ARC/RD1.2.752.243.1.1.20251031145134399.8000.21005.dcm"
base_path = Path('/home/bolo/Documents/PyDoseRT/test_data/GoldAtlasPlans/NODES/')
for patient_name in sorted(os.listdir(base_path)):
    try:
        patient_dir = base_path / patient_name
        ct_folder, rtplan_path, rtdose_path, rtstruct_path = find_patient_paths(patient_dir)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16
        kernel_size = 51
        downsampling_factor = (1, 1, 1)

        patient, beam_sequences = loaders.load_dicom(
                    ct_folder=ct_folder, 
                    dose_path=rtdose_path, 
                    plan_path=rtplan_path, 
                    struct_path=rtstruct_path,
                    struct_names=["CTV", "PTV", "FemoralHead_L", "FemoralHead_R", "Bladder", "Rectum", "External"],
                    use_delivery=True
                    )
        optimization = OptimizationConfig(
            preset="src/pydose_rt/data/optimization_presets/umea.json"
        )

        ptv_struct_name = [key for key in patient.structures.keys() if "PTV" in key][0]
        machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/umea_10MV.json")
            
        patient = patient.to(device).to(dtype)
        dose_volume = patient.dose
        # ct_volume = patient.get_masked_ct("External")
        # dose_volume = patient.get_masked_dose("External")

        doses = []
        for beam_sequence in beam_sequences:
            beam_sequence = beam_sequence.to(device).to(dtype)
            # dose_engine = DoseEngine(patient.ct_array.shape, patient.resolution, machine_config, beam_sequence, kernel_size, device, dtype, downsampling_factor)
            dose_engine = DoseEngine(kernel_size=15,
                                     machine_config=machine_config,
                                     image_template=patient.density_image,
                                     beam_template=beam_sequence
                                    )
            # dose_engine.calibrate()

            dose_pred = dose_engine.compute_dose_sequential(beam_sequence, ct_image=patient.density_image)
            doses.append(dose_pred.detach())
        dose_pred = sum(doses)
        dose_pred = torch.where(patient.structures["External"], dose_pred[0], 0.0)

        dose_max = max(dose_volume.max(), dose_pred.max()).item()

        mae_map = torch.abs(dose_pred - dose_volume)
        mae_loss = np.mean(torch.mean(mae_map[patient.structures["External"]]).item())
        print(mae_loss)


        # print(scale.item())
        # print(mae_loss)
        leafs = beam_sequence.leaf_positions.unsqueeze(0)
        mus = beam_sequence.mus.unsqueeze(0)
        jaws = beam_sequence.jaw_positions.unsqueeze(0)
        res = result_validation(patient, machine_config, beam_sequence, dose_pred[0], optimization, compute_gamma=True, compute_clinical_criteria=False, global_normalisation=2.2)
        # print([c['passed'] for s in res["clinical_criteria"].values() for c in s['criteria']])
        print(f"Patient {patient_name}:\t{res['gamma_pass_rate']}\t{res['mean_gamma']}")

        quick_plot(patient, dose_pred, f"MAE {str(np.round(mae_loss, 4))} Gamma pass rate {str(np.round(res['gamma_pass_rate'], 2))}", dose_max, f"out/quick_{patient_name}.png")

        print_results(None, optimization, [0.0], patient, beam_sequence, None, None, None, [], dose_pred, mae_loss, dose_max=dose_max, out_path=f"out/final_{patient_name}.png")

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