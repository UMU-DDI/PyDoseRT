from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
import numpy as np
import pytest
import os
import torch
from pydose_rt.data import MachineConfig
from pydose_rt import DoseEngine
from pydose_rt.utils.data_loading import load_rtp_data
import SimpleITK as sitk

def test_real_rtplan(rtp_data_dir, rtp_dose_path, rtp_plan_path):
    # Arrange
    expected = 0.5
    if not rtp_data_dir.exists():
        pytest.skip(f"Missing case folder: {rtp_data_dir}")

    ct_image, doses, masks, mlc_inputs = load_rtp_data(rtp_data_dir, dose_path=[rtp_dose_path], plan_path=[rtp_plan_path], scaling=400)
    dose_volume = sitk.GetArrayFromImage(doses['dose_0'])
    ct_volume = sitk.GetArrayFromImage(ct_image)
    external_mask = sitk.GetArrayFromImage(masks["External"])
    ct_volume = np.where(external_mask, ct_volume, -1000.0)

    ct_spacing = [ct_image.GetSpacing()[0], ct_image.GetSpacing()[1], ct_image.GetSpacing()[2]]
    ct_shape = ct_volume.shape

    ct_slices = np.array(np.expand_dims(ct_volume, 0), dtype=np.float32)
    leafs_1, mus_1 = mlc_inputs[0]
    leafs_2, mus_2 = mlc_inputs[1]
    config_1 = MachineConfig(ct_array_shape=ct_shape, 
                        resolution=np.divide(ct_spacing, 10), 
                        downsampling_factor=(2, 2, 2), 
                        field_size=(50, 50), 
                        number_of_leaf_pairs=60,
                        tpr_20_10=0.72, 
                        number_of_cps=178, 
                        starting_angle=0.5,
                        )
    dose_layer_1 = DoseEngine(config_1, 55, permute_ct=False, leafs_centered=True)
    dose_1 = dose_layer_1(torch.tensor(np.array(leafs_1), dtype=torch.float32, device=config_1.device), torch.tensor(np.array(mus_1), dtype=torch.float32, device=config_1.device), ct_image=torch.tensor(ct_slices, dtype=torch.float32, device=config_1.device))

    config_2 = MachineConfig(ct_array_shape=ct_shape, 
                        resolution=np.divide(ct_spacing, 10), 
                        downsampling_factor=(2, 2, 2), 
                        field_size=(50, 50), 
                        number_of_leaf_pairs=60, 
                        tpr_20_10=0.72, 
                        number_of_cps=178, 
                        starting_angle=2.0,
                        )
    dose_layer_2 = DoseEngine(config_2, 55, permute_ct=False, leafs_centered=True)
    dose_2 = dose_layer_2(torch.tensor(np.array(leafs_2), dtype=torch.float32, device=config_2.device), torch.tensor(np.array(mus_2), dtype=torch.float32, device=config_2.device), ct_image=torch.tensor(ct_slices, dtype=torch.float32, device=config_2.device))
    doses = [dose_1, dose_2]
    dose_plot = torch.stack([dose_1, dose_2]).sum(dim=0).cpu().detach().numpy()[0, ...]
    dose_plot = dose_plot * (np.quantile(dose_volume, 0.999) / np.quantile(dose_plot, 0.999))
    ext_mask = sitk.GetArrayFromImage(masks["External"]) > 0
    actual = np.mean(ext_mask * np.abs(dose_volume - dose_plot))
    print(f"Mean absolute dose difference in external: {actual}")
    
    assert expected >= actual, "The dose engine did not perform well enough for real plan."