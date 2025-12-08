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

    patient = patient.to(device).to(dtype)
    beam_sequence = beam_sequence.to(device).to(dtype)

    machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/umea_10MV.json")
        
    ct_volume = patient.get_masked_ct("External").unsqueeze(0)
    dose_target = patient.get_masked_dose("External").cpu().detach().numpy()

    dose_layer = DoseEngine(machine_config=machine_config,
                            kernel_size=kernel_size,
                            dose_grid_spacing=patient.resolution,
                            dose_grid_shape=patient.density_image.shape, 
                            beam_template=beam_sequence)

    dose_pred = dose_layer.compute_dose_sequential(beam_sequence, density_image=ct_volume)
    dose_pred = torch.where(patient.structures["External"], dose_pred[0], 0.0)
    dose_pred = dose_pred.cpu().detach().numpy()

    actual = np.mean(np.abs(dose_target - dose_pred))

    assert expected >= actual, "The dose engine did not perform well enough for real plan."