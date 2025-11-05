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
from pydose_rt.objectives.metrics import result_validation
import numpy as np
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, rotate
from pydose_rt import DoseEngine
import SimpleITK as sitk
import torch

# Set paths
ct_folder = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21/"
rtplan_path = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21_plans/1ARC/RP1.2.752.243.1.1.20251031145134399.7000.37887.dcm"
rtdose_path = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21_plans/1ARC/RD1.2.752.243.1.1.20251031145134399.8000.21005.dcm"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = DoseConfig.from_dicom(
    ct_folder=ct_folder, 
    dose_path=rtdose_path,
    plan_path=rtplan_path,
    struct_names=["External", "CTV", "FemoralHead_R", "FemoralHead_L", "Bladder", "PTVT_42.7"],
    machine_preset="umea",
    downsampling_factor=(1, 1, 1),
    dtype=torch.float16,
    device=device
)
ct_image = config.patient.ct_array
dose = config.patient.dose
masks = config.patient.structures
mlc_inputs = config.patient.plan_mlcs

dose_volume = dose
ct_volume = ct_image
external_mask = masks["External"]
ct_volume = np.where(external_mask, ct_volume, -1000.0)

ct_slices = np.array(np.expand_dims(ct_volume, 0))
leafs_1, mus_1 = mlc_inputs[0]
results = []

dose_layer_1 = DoseEngine(config.machine, 55, permute_ct=False, leafs_centered=True)
jaws_1 = np.zeros(config.machine.shape_jaws)
jaws_1[:, 0, :] = 0.5
jaws_1[:, 1, :] = 1.0
doses = []

dose_pred = dose_layer_1(torch.tensor(np.array(leafs_1), dtype=config.dtype, device=device), torch.tensor(np.array(mus_1), dtype=config.dtype, device=device), torch.tensor(np.array(jaws_1), dtype=config.dtype, device=device), ct_image=torch.tensor(ct_slices, dtype=config.dtype, device=device)).cpu().detach().numpy()

# dose_pred = dose_pred * np.max(dose_volume) / np.max(dose_pred)
dose_pred = dose_pred * (np.quantile(dose_volume, 0.999) / np.quantile(dose_pred, 0.999))

vmax = 15
slice_idx = dose_volume.shape[0] // 2
plt.figure()
plt.subplot(131)
# plt.imshow(ct_volume[ct_shape[0] // 2, :, :], cmap='gray')
plt.imshow(dose_volume[slice_idx, :, :], cmap='jet')
plt.colorbar()
plt.subplot(132)
# plt.imshow(ct_volume[ct_shape[0] // 2, :, :], cmap='gray')
plt.imshow(dose_pred[0, slice_idx, :, :], cmap='jet')
plt.colorbar()
plt.subplot(133)
# plt.imshow(ct_volume[ct_shape[0] // 2, :, :], cmap='gray')
plt.imshow(dose_volume[slice_idx, :, :] - dose_pred[0, slice_idx, :, :], cmap='coolwarm', vmin=-vmax, vmax=vmax, alpha=1.0)
plt.colorbar()
plt.show()
# result_validation(config, dose_pred, leafs_1, jaws_1, mus_1)
# print(f"{starting_angle_1}/{starting_angle_2}/{cw_1}/{cw_2}:\t{np.mean(diff)}") # Baseline is 0.143663