from re import M
import sys
import torch.nn.functional as F
sys.path.append('../')
sys.path.append('../../')
import pydicom
import time
import math

from pydicom.data import get_testdata_file
from pydose_rt.data import MachineConfig, Patient, TreatmentConfig, loaders
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
ct_folder = "/mimer/NOBACKUP/groups/naiss2023-6-64/attila/miqa/0e54d72a21/"
rtplan_path = "/mimer/NOBACKUP/groups/naiss2023-6-64/attila/miqa/0e54d72a21_plans/1ARC/RP1.2.752.243.1.1.20251031145134399.7000.37887.dcm"
rtdose_path = "/mimer/NOBACKUP/groups/naiss2023-6-64/attila/miqa/0e54d72a21_plans/1ARC/RD1.2.752.243.1.1.20251031145134399.8000.21005.dcm"
# ct_folder = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21/"
# rtplan_path = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21_plans/1ARC/RP1.2.752.243.1.1.20251031145134399.7000.37887.dcm"
# rtdose_path = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21_plans/1ARC/RD1.2.752.243.1.1.20251031145134399.8000.21005.dcm"

# rtplan_path = "/home/bolo/Downloads/rs_doses/RS_Imported_in_Water/RP1.2.752.243.1.1.20251119095513498.5300.35324.dcm"
# rtdose_path = "/home/bolo/Downloads/rs_doses/RS_Imported_in_Water/RD1.2.752.243.1.1.20251119095513499.5600.75370.dcm"

# rtplan_path = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21_plans/1ARC/RP1.2.752.243.1.1.20251031145134399.7000.37887.dcm"
# rtdose_path = "/home/bolo/Downloads/rs_doses/RS_Old_in_Water/RD1.2.752.243.1.1.20251119095655132.6200.21611.dcm"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

patient, treatment = loaders.load_dicom(
            ct_folder=ct_folder, 
            dose_path=rtdose_path, 
            plan_path=rtplan_path, 
            struct_names=["CTV", "PTVT_42.7", "FemoralHead_L", "FemoralHead_R", "Bladder", "Rectum", "External"],
            treatment_preset="src/pydose_rt/data/treatment_presets/vienna.json"
            )

treatment.kernel_size = 55
treatment.downsampling_factor = (1, 2, 2)
treatment.device = device
treatment.dtype = torch.float32

machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/umea_10MV.json", resolution=patient.voxel_spacing_mm, ct_array_shape=patient.ct_array.shape)
# ref_dose, calibration_factor = validate_unit_dose(machine_config, treatment, 110)
# if (np.abs(ref_dose - 1.0) > 0.001):
#     print(f"Calibration failed. Adjusting calibration factor to: {calibration_factor}")
#     machine_config.mean_photon_energy_MeV = calibration_factor
    
ct_image = patient.ct_array
dose = patient.dose
masks = patient.structures
leafs = torch.from_numpy(np.array(treatment.plan_mlcs))
mus = torch.from_numpy(np.array(treatment.plan_mus))
jaws = torch.from_numpy(np.array(treatment.plan_jaws))

dose_volume = dose
ct_volume = ct_image
external_mask = masks["External"] > 0
ct_volume = np.where(external_mask, ct_volume, -1000.0)
# ct_volume = np.where(np.logical_not(external_mask), ct_volume, 0.0)

ct_slices = np.array(np.expand_dims(ct_volume, 0))
results = []

dose_layer = DoseEngine(machine_config, treatment, permute_ct=False, leafs_centered=False, adjust_values=False)

# Freeze EVERYTHING first
for param in dose_layer.parameters():
    param.requires_grad = False

# Then UNFREEZE only the learnable kernel
for param in dose_layer.fluence_map_layer.learnable_kernel.parameters():
    param.requires_grad = True

# Verify (should only show learnable_kernel parameters)
trainable_params = [name for name, p in dose_layer.named_parameters() if p.requires_grad]
print(f"Trainable parameters: {trainable_params}")
print(f"Total trainable params: {sum(p.numel() for p in dose_layer.parameters() if p.requires_grad)}")


leafs = leafs.to(dose_layer.dtype).to(dose_layer.device)
mus = mus.to(dose_layer.dtype).to(dose_layer.device)
jaws = jaws.to(dose_layer.dtype).to(dose_layer.device)

optimizer = torch.optim.Adam([
    {'params': dose_layer.fluence_map_layer.learnable_kernel.parameters(), 'lr': 1e-2}
])
dose_tensor = torch.from_numpy(dose_volume).unsqueeze(0).to(dose_layer.device)
ct_tensor = torch.tensor(ct_slices, dtype=dose_layer.dtype, device=device)
leafs = (leafs[:, :, :-1, :] + leafs[:, :, 1:, :]) / 2
mus = (mus[:, :-1] + mus[:, 1:]) / 2
jaws = (jaws[:, :, :-1] + jaws[:, :, 1:]) / 2

for epoch in range(100000):
    dose_pred = dose_layer(
        leafs,
        mus,
        jaws,
        ct_image=ct_tensor
    )
    loss = F.l1_loss(dose_pred, dose_tensor)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 1000 == 0:
        print(f"Computing results for epoch {epoch}...")
        res = result_validation(patient, machine_config, treatment, dose_pred.cpu().detach().numpy(), leafs.cpu().detach().numpy(), jaws.cpu().detach().numpy(), mus.cpu().detach().numpy(), compute_gamma=True, compute_clinical_criteria=False)
        print(f"MAE: {loss.item():.6f}") 
        print(f"{res['gamma_pass_rate']}\t{res['mean_gamma']}")
        print(dose_layer.fluence_map_layer.learnable_kernel.kernel)
        print(dose_layer.fluence_map_layer.learnable_kernel.scale)
        print("\n")
    del dose_pred, loss
# dose_pred = dose_pred.cpu().detach().numpy()


# dose_pred = np.where(external_mask, dose_pred, 0.0)
# # scale = mae_optimal_scale(dose_pred[0, ...], dose_volume, mask=masks["CTV"] > 0)
# scale = np.quantile(dose_volume[masks["CTV"] > 0], 0.9) / np.quantile(dose_pred[0, masks["CTV"] > 0], 0.9)
# # scale = 5.51 / np.quantile(dose_pred[0, masks["PTVT_42.7"] > 0], 0.01)
# dose_pred = dose_pred * scale
# dose_max = max(dose_volume.max(), dose_pred.max())

# mae_map = np.abs(dose_pred[0] - dose_volume)
# mae_losses = [np.mean(mae_map[mask]) for mask in [masks["CTV"] > 0, masks["PTVT_42.7"] > 0, masks["Bladder"] > 0, masks["FemoralHead_L"] > 0, masks["FemoralHead_R"] > 0]]
# mae_loss = np.mean(mae_losses)

# vmax = 1
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

# slice_idx = dose_volume.shape[2] // 2 + 5
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


# print(mae_loss)
# res = result_validation(patient, machine_config, treatment, dose_pred, leafs, jaws, mus, compute_gamma=True, compute_clinical_criteria=True)
# print([c['passed'] for s in res["clinical_criteria"].values() for c in s['criteria']])
# print(f"{res['gamma_pass_rate']}\t{res['mean_gamma']}")

# print_results(None, treatment, [0.0], torch.from_numpy(np.expand_dims(dose_volume, 0)), leafs, mus, jaws, None, None, None, [], torch.from_numpy(dose_pred), torch.from_numpy(np.expand_dims(ct_volume, 0)), [torch.from_numpy(np.expand_dims(mask, 0)) for mask in list(masks.values())], mae_loss, dose_max=dose_max)

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