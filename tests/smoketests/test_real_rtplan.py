from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
import numpy as np
import pytest
import os
import torch
from pydose_rt.data import MachineConfig, loaders
from pydose_rt import DoseEngine
import SimpleITK as sitk


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("kernel_size", [15, 25])
def test_real_rtplan(rtp_data_dir, rtp_struct_path, rtp_dose_path, rtp_plan_path, dtype, kernel_size):
    if not rtp_data_dir.exists():
        pytest.skip(f"Missing case folder: {rtp_data_dir}")

    # Arrange
    expected = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patient, beam_sequence = loaders.load_dicom(
                ct_folder=rtp_data_dir, 
                struct_path=rtp_struct_path,
                dose_path=rtp_dose_path, 
                plan_path=rtp_plan_path, 
                struct_names=["CTV", "PTVT_42.7", "FemoralHead_L", "FemoralHead_R", "Bladder", "External"],
                )
    beam_sequence = beam_sequence[0]
    beam_sequence = beam_sequence[::4]

    kernel_size = kernel_size
    device = device
    dtype = dtype
    downsampling_factor = (1, 2, 2)

    machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/umea_10MV.json")

    # ref_dose, calibration_factor = validate_unit_dose(machine_config, patient, 110, 1, downsampling_factor, device, dtype)
    # if (np.abs(ref_dose - 1.0) > 0.001):
    #     print(f"Calibration failed. Adjusting calibration factor to: {calibration_factor}")
    #     machine_config.mean_photon_energy_MeV = calibration_factor
        
    ct_image = patient.density_image
    dose = patient.dose
    masks = patient.structures

    dose_volume = dose
    ct_volume = ct_image
    external_mask = masks["External"]
    ct_volume = np.where(external_mask, ct_volume, -1000.0)

    ct_slices = np.array(np.expand_dims(ct_volume, 0))

    dose_layer = DoseEngine(machine_config=machine_config,
                            kernel_size=kernel_size,
                            image_template=patient.density_image, 
                            beam_template=beam_sequence,
                            downsampling_factor=downsampling_factor)

    dose_pred = dose_layer.compute_beam_sequence(beam_sequence, ct_image=torch.tensor(ct_slices, dtype=dose_layer.dtype, device=device))
    dose_pred = dose_pred.cpu().detach().numpy()


    dose_pred = np.where(external_mask, dose_pred, 0.0)
    scale = np.quantile(dose_volume[masks["CTV"] > 0], 0.9) / np.quantile(dose_pred[0, masks["CTV"] > 0], 0.9)
    dose_pred = dose_pred * scale
    mae_map = np.abs(dose_pred[0] - dose_volume[0].cpu().detach().numpy())
    actual = np.mean(mae_map[masks["External"] > 0])

    assert expected >= actual, "The dose engine did not perform well enough for real plan."