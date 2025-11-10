import sys
sys.path.append('../')
sys.path.append('../../')
import pydicom
from IPython.display import clear_output
import time
import math

from pydicom.data import get_testdata_file
from pydose_rt.data import MachineConfig, PatientData, DoseConfig
# from pydose_rt.data import MachineConfig
from pydose_rt.objectives.metrics import result_validation, validate_unit_dose
import numpy as np
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, rotate
from pydose_rt import DoseEngine
import SimpleITK as sitk
from pydose_rt.utils.plotting import print_results, make_animation
import torch

# Set paths
ct_folder = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21/"
rtplan_path = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21_plans/1ARC/RP1.2.752.243.1.1.20251031145134399.7000.37887.dcm"
# rtplan_path = "out/plan.dcm"
rtdose_path = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21_plans/1ARC/RD1.2.752.243.1.1.20251031145134399.8000.21005.dcm"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kernel_size = 55

config = DoseConfig.from_dicom(
    ct_folder=ct_folder, 
    dose_path=rtdose_path,
    plan_path=rtplan_path,
    struct_names=["External", "CTV", "FemoralHead_R", "FemoralHead_L", "Bladder", "PTVT_42.7"],
    machine_preset="umea",
    downsampling_factor=(1, 2, 2),
    dtype=torch.float32,
    device=device
)
# ref_dose, calibration_factor = validate_unit_dose(config, kernel_size, 130)
# if (np.abs(ref_dose - 1.0) > 0.001):
#     raise Exception(f"Calibration failed. please use calibration factor: {calibration_factor}")
    
ct_image = config.patient.ct_array
dose = config.patient.dose
masks = config.patient.structures
leafs = config.patient.plan_mlcs
mus = config.patient.plan_mus
jaws = config.patient.plan_jaws

dose_volume = dose
ct_volume = ct_image
external_mask = masks["External"]
ct_volume = np.where(external_mask, ct_volume, -1000.0)

ct_slices = np.array(np.expand_dims(ct_volume, 0))
# leafs[:, 0, :, :] = 0.3
# leafs[:, 1, :, :] = 0.1
# mus = np.ones_like(mus)
results = []

dose_layer = DoseEngine(config.machine, kernel_size, permute_ct=False, leafs_centered=False)
# jaws = np.zeros(config.machine.shape_jaws)
# jaws[:, 0, :] = 0.5
# jaws[:, 1, :] = 1.0

leafs = torch.tensor(np.array(leafs), dtype=config.dtype, device=device)
mus = torch.tensor(np.array(mus) / 10, dtype=config.dtype, device=device)
jaws = torch.tensor(np.array(jaws), dtype=config.dtype, device=device)

dose_pred = dose_layer(leafs, mus, jaws, ct_image=torch.tensor(ct_slices, dtype=config.dtype, device=device))
dose_pred = dose_pred.cpu().detach().numpy()

dose_pred = np.where(external_mask, dose_pred, 0.0)
# dose_pred = dose_pred * (np.quantile(dose_volume, 0.99) / np.quantile(dose_pred, 0.99))


vmax = 15
slice_idx = dose_volume.shape[0] // 2
plt.figure()
plt.subplot(131)
# plt.imshow(ct_volume[ct_shape[0] // 2, :, :], cmap='gray')
plt.imshow(dose_volume[slice_idx, :, :], cmap='jet')
plt.colorbar()
plt.subplot(132)
plt.title(f"MAE {np.mean(np.abs(dose_pred[0] - dose_volume))}")
# plt.imshow(ct_volume[ct_shape[0] // 2, :, :], cmap='gray')
plt.imshow(dose_pred[0, slice_idx, :, :], cmap='jet')
plt.colorbar()
plt.subplot(133)
# plt.imshow(ct_volume[ct_shape[0] // 2, :, :], cmap='gray')
plt.imshow(dose_volume[slice_idx, :, :] - dose_pred[0, slice_idx, :, :], cmap='coolwarm', vmin=-vmax, vmax=vmax, alpha=1.0)
plt.colorbar()
plt.show()

# result_validation(config, dose_pred, leafs, jaws, mus)
# make_animation(None, config, dose_layer, leafs, mus, jaws, dose_pred.max())