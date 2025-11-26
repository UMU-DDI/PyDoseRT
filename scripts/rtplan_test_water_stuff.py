from re import M
import sys
sys.path.append('../')
sys.path.append('../../')
import pydicom
from IPython.display import clear_output
import time
import math

from pydicom.data import get_testdata_file
from pydose_rt.data import MachineConfig, Patient, loaders
# from pydose_rt.data import MachineConfig
from pydose_rt.objectives.metrics import result_validation, validate_unit_dose
from pydose_rt.utils.utils import mae_optimal_scale
import numpy as np
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, rotate
from pydose_rt import DoseEngine
import SimpleITK as sitk
from pydose_rt.utils.plotting import print_results, make_animation
import torch

# Set paths
ct_folder = "/home/bolo/Documents/PyDoseRT/test_data/Fields_in_water_box/"
rtplan_path = "/home/bolo/Documents/PyDoseRT/test_data/Fields_in_water_box/RP1.2.752.243.1.1.20251119152249150.9000.36474.dcm"

doses = ["RD1.2.752.243.1.1.20251119152249151.1600.25202.dcm", 
         "RD1.2.752.243.1.1.20251119152249151.1700.70054.dcm", 
         "RD1.2.752.243.1.1.20251119152249151.1800.60620.dcm", 
         "RD1.2.752.243.1.1.20251119152249151.1900.42250.dcm"]
rtdose_path = "/home/bolo/Documents/PyDoseRT/test_data/Fields_in_water_box/" + doses[0]



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


patient, treatment = loaders.load_dicom(
            ct_folder=ct_folder, 
            dose_path=rtdose_path, 
            plan_path=rtplan_path, 
            struct_names=["External"],
            treatment_preset="src/pydose_rt/data/optimization_presets/umea.json"
            )

treatment.kernel_size = 75
treatment.device = device
treatment.dtype = torch.float16

machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/umea_6MV.json", resolution=patient.resolution, ct_array_shape=patient.ct_array.shape)
# ref_dose, calibration_factor = validate_unit_dose(machine_config, treatment, 110)
# if (np.abs(ref_dose - 1.0) > 0.001):
#     print(f"Calibration failed. Adjusting calibration factor to: {calibration_factor}")
#     machine_config.mean_photon_energy_MeV = calibration_factor
    
ct_image = patient.ct_array
dose = patient.dose
masks = patient.structures
leafs = treatment.plan_mlcs
mus = treatment.plan_mus
jaws = treatment.plan_jaws

dose_volume = dose
ct_volume = ct_image
external_mask = masks["External"]
ct_volume = np.where(external_mask, ct_volume, -1000.0)

ct_slices = np.array(np.expand_dims(ct_volume, 0))
results = []

dose_layer = DoseEngine(machine_config, treatment, permute_ct=False, leafs_centered=False, adjust_values=False)
dose_layer.eval()

leafs = torch.tensor(np.array(leafs), dtype=dose_layer.dtype, device=dose_layer.device)
mus = torch.tensor(np.array(mus), dtype=dose_layer.dtype, device=dose_layer.device)
jaws = torch.tensor(np.array(jaws), dtype=dose_layer.dtype, device=dose_layer.device)
   
dose_pred = dose_layer(
    leafs, 
    mus, 
    jaws, 
    ct_image=torch.tensor(ct_slices, dtype=dose_layer.dtype, device=device), 
    leaf_x=0.0, 
    leaf_y=0.0, 
    jaw_x=0.0, 
    jaw_y=0.0
)
dose_pred = dose_pred.cpu().detach().numpy()


dose_pred = np.where(external_mask, dose_pred, 0.0)
scale = mae_optimal_scale(dose_pred[0, ...], dose_volume, mask=masks["External"] > 0)
# scale = np.quantile(dose_volume[masks["CTV"] > 0], 0.9) / np.quantile(dose_pred[0, masks["CTV"] > 0], 0.9)
dose_pred = dose_pred * scale
dose_max = max(dose_volume.max(), dose_pred.max())


vmax = 0.04
mae_map = np.abs(dose_pred[0] - dose_volume)
mae_loss = np.mean(mae_map[masks["External"] > 0])

plt.figure()
slice_idx = dose_volume.shape[0] // 2
plt.subplot(331)
# plt.imshow(ct_volume[slice_idx, :, :], cmap='gray')
plt.imshow(dose_volume[slice_idx, :, :], cmap='jet', vmax=dose_max)
plt.axis('off')
plt.colorbar()
plt.subplot(332)
plt.title(f"({str(np.round(scale, 3))})MAE {mae_loss}")
# plt.imshow(ct_volume[slice_idx, :, :], cmap='gray')
plt.imshow(dose_pred[0, slice_idx, :, :], cmap='jet', vmax=dose_max)
plt.axis('off')
plt.colorbar()
plt.subplot(333)
plt.imshow(ct_volume[slice_idx, :, :], cmap='gray')
plt.imshow(dose_volume[slice_idx, :, :] - dose_pred[0, slice_idx, :, :], cmap='coolwarm', vmin=-vmax, vmax=vmax, alpha=0.6)
plt.axis('off')
plt.colorbar()

slice_idx = dose_volume.shape[1] // 2
plt.subplot(334)
# plt.imshow(ct_volume[slice_idx, :, :], cmap='gray')
plt.imshow(dose_volume[:, slice_idx, :], cmap='jet', vmax=dose_max)
plt.axis('off')
plt.colorbar()
plt.subplot(335)
# plt.title(f"MAE {mae_loss}")
# plt.imshow(ct_volume[slice_idx, :, :], cmap='gray')
plt.imshow(dose_pred[0, :, slice_idx, :], cmap='jet', vmax=dose_max)
plt.axis('off')
plt.colorbar()
plt.subplot(336)
plt.imshow(ct_volume[:, slice_idx, :], cmap='gray')
plt.imshow(dose_volume[:, slice_idx, :] - dose_pred[0, :, slice_idx, :], cmap='coolwarm', vmin=-vmax, vmax=vmax, alpha=0.6)
plt.axis('off')
plt.colorbar()

slice_idx = dose_volume.shape[2] // 2
plt.subplot(337)
# plt.imshow(ct_volume[slice_idx, :, :], cmap='gray')
plt.imshow(dose_volume[:, :, slice_idx], cmap='jet', vmax=dose_max)
plt.axis('off')
plt.colorbar()
plt.subplot(338)
# plt.title(f"MAE {mae_loss}")
# plt.imshow(ct_volume[slice_idx, :, :], cmap='gray')
plt.imshow(dose_pred[0, :, :, slice_idx], cmap='jet', vmax=dose_max)
plt.axis('off')
plt.colorbar()
plt.subplot(339)
plt.imshow(ct_volume[:, :, slice_idx], cmap='gray')
plt.imshow(dose_volume[:, :, slice_idx] - dose_pred[0, :, :, slice_idx], cmap='coolwarm', vmin=-vmax, vmax=vmax, alpha=0.6)
plt.axis('off')
plt.colorbar()

plt.show()

# print_results(None, treatment, [0.0], torch.from_numpy(np.expand_dims(dose_volume, 0)), leafs, mus, jaws, None, None, None, [], torch.from_numpy(dose_pred), torch.from_numpy(np.expand_dims(ct_volume, 0)), [torch.from_numpy(np.expand_dims(mask, 0)) for mask in list(masks.values())], mae_loss, dose_max=dose_max)

# res = result_validation(patient, machine_config, treatment, dose_pred, leafs, jaws, mus, compute_gamma=False)
# print(res)
# make_animation(None, treatment, machine_config, patient, dose_layer, leafs, mus, jaws, dose_pred.max())