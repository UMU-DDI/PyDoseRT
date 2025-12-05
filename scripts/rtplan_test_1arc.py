from comet_ml import Experiment
from re import M
import sys
import os
import torch.nn.functional as F
sys.path.append('../')
sys.path.append('../../')
import pydicom
import time
from pathlib import Path
import math
import nibabel as nib
import time

from pydicom.data import get_testdata_file
from pydose_rt.data import MachineConfig, Patient, OptimizationConfig, loaders
from pydose_rt.objectives.metrics import result_validation, validate_unit_dose
from pydose_rt.utils.utils import mae_optimal_scale
import numpy as np
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, rotate
from pydose_rt import DoseEngine
import SimpleITK as sitk
from pydose_rt.utils.plotting import print_results, make_animation, quick_plot
import torch

# Set paths
patient_name = "0e54d72a21"
ct_folder = f"/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/{patient_name}/"
rtstruct_path = next((f for f in Path(ct_folder).iterdir() if "RS" in f.name.upper() or "RTSTRUCT" in f.name.upper()), None)
rtplan_path = f"/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/{patient_name}_plans/1ARC/RP1.2.752.243.1.1.20251031145134399.7000.37887.dcm"
rtdose_path = f"/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/{patient_name}_plans/1ARC/RD1.2.752.243.1.1.20251031145134399.8000.21005.dcm"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

patient, beam_sequences = loaders.load_dicom(
            ct_folder=ct_folder, 
            dose_path=rtdose_path, 
            plan_path=rtplan_path, 
            struct_path=rtstruct_path,
            struct_names=["CTV", "PTV", "FemoralHead_L", "FemoralHead_R", "Bladder", "Rectum", "External"],
            use_delivery=True
            )

optimization = OptimizationConfig(
    preset="src/pydose_rt/data/optimization_presets/umea.json",
)

ptv_struct_name = [key for key in patient.structures.keys() if "PTV" in key][0]
machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/umea_10MV.json")
# ref_dose, calibration_factor = validate_unit_dose(machine_config, 110)
# if (np.abs(ref_dose - 1.0) > 0.001):
#     # print(f"Calibration failed. Adjusting calibration factor to: {calibration_factor}")
#     machine_config.mean_photon_energy_MeV = calibration_factor

doses = []
for beam_sequence in beam_sequences:
    dose_layer = DoseEngine(
        dose_grid_shape=patient.ct_array.shape,
        dose_grid_spacing=patient.resolution,
        machine_config=machine_config, 
        dtype=dtype, 
        device=device, 
        kernel_size=25,
        beam_input=beam_sequence
    )
    dose_layer.eval()
    patient = patient.to(dose_layer.device).to(dose_layer.dtype)
    ct_volume = patient.get_masked_ct("External").unsqueeze(0)
    dose_volume = patient.get_masked_dose("External").unsqueeze(0)
    beam_sequence = beam_sequence.to(dose_layer.device).to(dose_layer.dtype)

    start_time = time.time()
    dose_pred = dose_layer.compute_beam_sequence(beam_sequence, ct_volume)
    print(f"Dose computed in {time.time() - start_time}")
    doses.append(dose_pred.detach())
dose_pred = sum(doses)
dose_pred = torch.where(patient.structures["External"], dose_pred, 0.0)

# scale = mae_optimal_scale(dose_pred[0, ...], dose_volume, mask=masks["PTVT_42.7"] > 0)
scale = torch.mean(dose_volume[0, patient.structures[ptv_struct_name]]) / torch.mean(dose_pred[0, patient.structures[ptv_struct_name]])
# scale = 5.51 / np.quantile(dose_pred[0, masks["PTVT_42.7"] > 0], 0.01)
dose_pred = dose_pred * scale
dose_max = max(dose_volume.max(), dose_pred.max()).item()


# affine = np.eye(4)   # Identity matrix: simple default
# img = nib.Nifti1Image(dose_volume, affine)
# nib.save(img, "out/dose_true.nii.gz")   # or "output.nii"
# img = nib.Nifti1Image(dose_pred[0], affine)
# nib.save(img, "out/dose_pred.nii.gz")   # or "output.nii"

mae_map = torch.abs(dose_pred[0] - dose_volume[0])
mae_loss = np.mean(torch.mean(mae_map[patient.structures["External"]]).item())


res = result_validation(patient, machine_config, beam_sequence, dose_pred[0], optimization, compute_gamma=True, compute_clinical_criteria=False)
# print([c['passed'] for s in res["clinical_criteria"].values() for c in s['criteria']])
print(f"Patient {patient_name}:\t{res['gamma_pass_rate']}\t{res['mean_gamma']}")

quick_plot(dose_volume, dose_pred, ct_volume, f"MAE {str(np.round(mae_loss, 4))} Gamma pass rate {str(np.round(res['gamma_pass_rate'], 2))}", dose_max, f"out/quick_{patient_name}.png")

print_results(None, optimization, [0.0], dose_volume, beam_sequence, None, None, None, [], dose_pred, ct_volume, [mask.unsqueeze(0) for mask in list(patient.structures.values())], mae_loss, dose_max=dose_max, out_path=f"out/final_{patient_name}.png")

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