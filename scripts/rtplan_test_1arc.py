from re import M
import sys
sys.path.append('../')
sys.path.append('../../')
import pydicom
from IPython.display import clear_output
import time
import math

from pydicom.data import get_testdata_file
from pydose_rt.data import MachineConfig, Patient, TreatmentConfig, loaders
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


def mae_optimal_scale(A: np.ndarray, P: np.ndarray, mask=None):
    """
    Finds scalar c that minimizes MAE(||c*A - P||_1).
    A, P : numpy arrays of same shape (3D or any shape)
    mask : optional boolean array (same shape) to include only specific voxels
    """
    if mask is not None:
        A = A[mask]
        P = P[mask]

    valid = A > 0  # ignore zero or negative A if intensities are positive
    A = A[valid]
    P = P[valid]

    ratios = P / A
    weights = np.abs(A)

    # Sort ratios by value
    idx = np.argsort(ratios)
    sorted_ratios = ratios[idx]
    sorted_weights = weights[idx]

    # Cumulative weight
    cumulative = np.cumsum(sorted_weights)
    cutoff = cumulative[-1] / 2.0

    # Weighted median = first ratio where cumulative weight >= half total
    median_idx = np.searchsorted(cumulative, cutoff)
    c = sorted_ratios[median_idx]
    return c

# Set paths
ct_folder = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21/"
rtplan_path = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21_plans/1ARC/RP1.2.752.243.1.1.20251031145134399.7000.37887.dcm"
# rtplan_path = "out/plan.dcm"
rtdose_path = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21_plans/1ARC/RD1.2.752.243.1.1.20251031145134399.8000.21005.dcm"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


patient, treatment = loaders.load_dicom(
            ct_folder=ct_folder, 
            dose_path=rtdose_path, 
            plan_path=rtplan_path, 
            struct_names=["CTV", "PTVT_42.7", "FemoralHead_L", "FemoralHead_R", "Bladder", "External"],
            treatment_preset="src/pydose_rt/data/treatment_presets/umea.json"
            )

treatment.kernel_size = 75
treatment.device = device
treatment.dtype = torch.float16

machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/umea.json", resolution=patient.voxel_spacing_mm, ct_array_shape=patient.ct_array.shape)
# ref_dose, calibration_factor = validate_unit_dose(config, kernel_size, 130)
# if (np.abs(ref_dose - 1.0) > 0.001):
#     raise Exception(f"Calibration failed. please use calibration factor: {calibration_factor}")
    
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

leafs = torch.tensor(np.array(leafs), dtype=dose_layer.dtype, device=dose_layer.device)
mus = torch.tensor(np.array(mus), dtype=dose_layer.dtype, device=dose_layer.device)
jaws = torch.tensor(np.array(jaws), dtype=dose_layer.dtype, device=dose_layer.device)

for leaf_x in [-1.0]:#np.linspace(-3, 3, 5, endpoint=True):
    for leaf_y in [0.0]:#np.linspace(0, 4, 5, endpoint=True):
        for jaw_x in np.linspace(-10.0, 10.0, 21, endpoint=True):#np.linspace(4.0, 6.0, 3, endpoint=True):
            for jaw_y in np.linspace(-10.0, 10.0, 21, endpoint=True): # np.linspace(-9, -5.0, 5, endpoint=True):    
                dose_pred = dose_layer(leafs, mus, jaws, ct_image=torch.tensor(ct_slices, dtype=dose_layer.dtype, device=device), leaf_x=leaf_x, leaf_y=leaf_y, jaw_x=jaw_x, jaw_y=jaw_y)
                dose_pred = dose_pred.cpu().detach().numpy()


                dose_pred = np.where(external_mask, dose_pred, 0.0)
                scale = mae_optimal_scale(dose_pred[0, ...], dose_volume, mask=masks["CTV"] > 0)
                # scale = np.quantile(dose_volume[masks["CTV"] > 0], 0.9) / np.quantile(dose_pred[0, masks["CTV"] > 0], 0.9)
                dose_pred = dose_pred * scale
                dose_max = max(dose_volume.max(), dose_pred.max())


                vmax = 10
                mae_map = np.abs(dose_pred[0] - dose_volume)
                mae_losses = [np.mean(mae_map[mask]) for mask in [masks["CTV"] > 0, masks["PTVT_42.7"] > 0, masks["Bladder"] > 0, masks["FemoralHead_L"] > 0, masks["FemoralHead_R"] > 0]]
                mae_loss = np.mean(mae_losses)

                print_results(None, treatment, [0.0], torch.from_numpy(np.expand_dims(dose_volume, 0)), leafs, mus, jaws, None, None, None, [], torch.from_numpy(dose_pred), torch.from_numpy(np.expand_dims(ct_volume, 0)), [torch.from_numpy(np.expand_dims(mask, 0)) for mask in list(masks.values())], mae_loss)
                print(f"Leafs offset: ({leaf_x}, {leaf_y}), Jaws offset: ({jaw_x}, {jaw_y}) => Scale: {scale:.4f}, MAE: {mae_losses}")
# plt.figure()

# slice_idx = dose_volume.shape[0] // 2 - 5
# plt.subplot(331)
# # plt.imshow(ct_volume[slice_idx, :, :], cmap='gray')
# plt.imshow(dose_volume[slice_idx, :, :], cmap='jet', vmax=dose_max)
# plt.axis('off')
# plt.colorbar()
# plt.subplot(332)
# plt.title(f"({str(np.round(scale, 3))})MAE {mae_loss}")
# # plt.imshow(ct_volume[slice_idx, :, :], cmap='gray')
# plt.imshow(dose_pred[0, slice_idx, :, :], cmap='jet', vmax=dose_max)
# plt.axis('off')
# plt.colorbar()
# plt.subplot(333)
# plt.imshow(ct_volume[slice_idx, :, :], cmap='gray')
# plt.imshow(dose_volume[slice_idx, :, :] - dose_pred[0, slice_idx, :, :], cmap='coolwarm', vmin=-vmax, vmax=vmax, alpha=0.6)
# plt.axis('off')
# plt.colorbar()

# slice_idx = dose_volume.shape[1] // 2 - 5
# plt.subplot(334)
# # plt.imshow(ct_volume[slice_idx, :, :], cmap='gray')
# plt.imshow(dose_volume[:, slice_idx, :], cmap='jet', vmax=dose_max)
# plt.axis('off')
# plt.colorbar()
# plt.subplot(335)
# # plt.title(f"MAE {mae_loss}")
# # plt.imshow(ct_volume[slice_idx, :, :], cmap='gray')
# plt.imshow(dose_pred[0, :, slice_idx, :], cmap='jet', vmax=dose_max)
# plt.axis('off')
# plt.colorbar()
# plt.subplot(336)
# plt.imshow(ct_volume[:, slice_idx, :], cmap='gray')
# plt.imshow(dose_volume[:, slice_idx, :] - dose_pred[0, :, slice_idx, :], cmap='coolwarm', vmin=-vmax, vmax=vmax, alpha=0.6)
# plt.axis('off')
# plt.colorbar()

# slice_idx = dose_volume.shape[2] // 2 - 5
# plt.subplot(337)
# # plt.imshow(ct_volume[slice_idx, :, :], cmap='gray')
# plt.imshow(dose_volume[:, :, slice_idx], cmap='jet', vmax=dose_max)
# plt.axis('off')
# plt.colorbar()
# plt.subplot(338)
# # plt.title(f"MAE {mae_loss}")
# # plt.imshow(ct_volume[slice_idx, :, :], cmap='gray')
# plt.imshow(dose_pred[0, :, :, slice_idx], cmap='jet', vmax=dose_max)
# plt.axis('off')
# plt.colorbar()
# plt.subplot(339)
# plt.imshow(ct_volume[:, :, slice_idx], cmap='gray')
# plt.imshow(dose_volume[:, :, slice_idx] - dose_pred[0, :, :, slice_idx], cmap='coolwarm', vmin=-vmax, vmax=vmax, alpha=0.6)
# plt.axis('off')
# plt.colorbar()

# plt.show()

# print_results(None, config.treatment, [0.0], torch.from_numpy(np.expand_dims(dose_volume, 0)), leafs, mus, jaws, None, None, None, [], torch.from_numpy(dose_pred), torch.from_numpy(np.expand_dims(ct_volume, 0)), [torch.from_numpy(np.expand_dims(mask, 0)) for mask in list(masks.values())], mae_loss)
# res = result_validation(patient, machine_config, treatment, dose_pred, leafs, jaws, mus, compute_gamma=True)
# print(res)
# make_animation(None, config, dose_layer, leafs, mus, jaws, dose_pred.max())