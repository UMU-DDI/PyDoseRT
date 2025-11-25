from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
import numpy as np
import pytest
import os
import torch
from pydose_rt.data import MachineConfig, TreatmentConfig, loaders
from pydose_rt.objectives.metrics import validate_unit_dose
from pydose_rt import DoseEngine
import SimpleITK as sitk


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("kernel_size", [15, 25])
def test_real_rtplan(rtp_data_dir, rtp_dose_path, rtp_plan_path, dtype, kernel_size):
    if not rtp_data_dir.exists():
        pytest.skip(f"Missing case folder: {rtp_data_dir}")

    # Arrange
    expected = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patient, treatment = loaders.load_dicom(
                ct_folder=rtp_data_dir, 
                dose_path=rtp_dose_path, 
                plan_path=rtp_plan_path, 
                struct_names=["CTV", "PTVT_42.7", "FemoralHead_L", "FemoralHead_R", "Bladder", "External"],
                treatment_preset="src/pydose_rt/data/optimization_presets/umea.json"
                )

    treatment.kernel_size = kernel_size
    treatment.device = device
    treatment.dtype = dtype
    treatment.downsampling_factor = (1, 2, 2)

    machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/umea_10MV.json", resolution=patient.voxel_spacing_mm, ct_array_shape=patient.ct_array.shape)

    ref_dose, calibration_factor = validate_unit_dose(machine_config, treatment, 130)
    if (np.abs(ref_dose - 1.0) > 0.001):
        print(f"Calibration failed. Adjusting calibration factor to: {calibration_factor}")
        machine_config.mean_photon_energy_MeV = calibration_factor
        
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

    dose_layer = DoseEngine(machine_config, treatment, permute_ct=False, leafs_centered=False, adjust_values=False)

    leafs = torch.tensor((np.array(leafs)[:, :, :-1, :] + np.array(leafs)[:, :, 1:, :]) / 2, dtype=dose_layer.dtype, device=dose_layer.device)
    mus = torch.tensor((np.array(mus)[:, :-1] + np.array(mus)[:, 1:]) / 2, dtype=dose_layer.dtype, device=dose_layer.device)
    jaws = torch.tensor((np.array(jaws)[:, :, :-1] + np.array(jaws)[:, :, 1:]) / 2, dtype=dose_layer.dtype, device=dose_layer.device)
        
    dose_pred = dose_layer(leafs, mus, jaws, ct_image=torch.tensor(ct_slices, dtype=dose_layer.dtype, device=device))
    dose_pred = dose_pred.cpu().detach().numpy()


    dose_pred = np.where(external_mask, dose_pred, 0.0)
    scale = np.quantile(dose_volume[masks["CTV"] > 0], 0.9) / np.quantile(dose_pred[0, masks["CTV"] > 0], 0.9)
    dose_pred = dose_pred * scale
    mae_map = np.abs(dose_pred[0] - dose_volume)
    actual = np.mean(mae_map[masks["External"] > 0])

    assert expected >= actual, "The dose engine did not perform well enough for real plan."