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

kernel_size = 51

config = DoseConfig.from_dicom(
    ct_folder=ct_folder, 
    dose_path=rtdose_path,
    plan_path=rtplan_path,
    struct_names=["CTV", "PTVT_42.7", "FemoralHead_L", "FemoralHead_R", "Bladder", "External"],
    machine_preset="umea",
        treatment_preset="umea",
    downsampling_factor=(1, 2, 2),
    dtype=torch.float16,
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
# ct_volume[:, :, :90] = -1000.0

ct_slices = np.array(np.expand_dims(ct_volume, 0))
# leafs[:, 0, :, :] = 0.3
# leafs[:, 1, :, :] = 0.1
# mus = np.ones_like(mus)
results = []

dose_layer = DoseEngine(config.machine, kernel_size, permute_ct=False, leafs_centered=False, adjust_values=False)
# jaws = np.zeros(config.machine.shape_jaws)
# jaws[:, 0, :] = 0.5
# jaws[:, 1, :] = 1.0

leafs = torch.tensor(np.array(leafs), dtype=config.dtype, device=device)
mus = torch.tensor(np.array(mus), dtype=config.dtype, device=device)
jaws = torch.tensor(np.array(jaws), dtype=config.dtype, device=device)

dose_pred = dose_layer(leafs, mus, jaws, ct_image=torch.tensor(ct_slices, dtype=config.dtype, device=device))
dose_pred = dose_pred.cpu().detach().numpy()


dose_pred = np.where(external_mask, dose_pred, 0.0)
dose_pred = dose_pred * mae_optimal_scale(dose_pred[0, ...], dose_volume)
# dose_pred = dose_pred * dose_volume[masks["CTV"] > 0].mean() / dose_pred[0, ...][masks["CTV"] > 0].mean()


vmax = 10
slice_idx = dose_volume.shape[0] // 2 - 5
mae_loss = np.mean(np.abs(dose_pred[0] - dose_volume))
plt.figure()
plt.subplot(131)
# plt.imshow(ct_volume[slice_idx, :, :], cmap='gray')
plt.imshow(dose_volume[slice_idx, 64:119, 64:128], cmap='jet')
plt.axis('off')
plt.colorbar()
plt.subplot(132)
# plt.title(f"MAE {mae_loss}")
# plt.imshow(ct_volume[slice_idx, :, :], cmap='gray')
plt.imshow(dose_pred[0, slice_idx, 64:119, 64:128], cmap='jet')
plt.axis('off')
plt.colorbar()
plt.subplot(133)
plt.imshow(ct_volume[slice_idx, 64:119, 64:128], cmap='gray')
plt.imshow(dose_volume[slice_idx, 64:119, 64:128] - dose_pred[0, slice_idx, 64:119, 64:128], cmap='coolwarm', vmin=-vmax, vmax=vmax, alpha=0.6)
plt.axis('off')
plt.colorbar()
plt.show()

print_results(None, config.treatment, [0.0], torch.from_numpy(np.expand_dims(dose_volume, 0)), leafs, mus, jaws, None, None, None, [], torch.from_numpy(dose_pred), torch.from_numpy(np.expand_dims(ct_volume, 0)), [torch.from_numpy(np.expand_dims(mask, 0)) for mask in list(masks.values())], mae_loss)
res = result_validation(config, dose_pred, leafs, jaws, mus, compute_gamma=True)
print(res)
# make_animation(None, config, dose_layer, leafs, mus, jaws, dose_pred.max())