import sys
sys.path.append('../')
sys.path.append('../../')
import pydicom
from IPython.display import clear_output
import time
import math

from pydicom.data import get_testdata_file
from pydose_rt import ModelConfig
# from pydose_rt import ModelConfig
import numpy as np
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, rotate
from pydose_rt import DoseEngine
from pydose_rt.utils.data_loading import load_rtp_data
import SimpleITK as sitk
import torch

# Set paths
ct_folder = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21/"
rtplan_path = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21_plans/1ARC/RP1.2.752.243.1.1.20251031145134399.7000.37887.dcm"
rtdose_path = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21_plans/1ARC/RD1.2.752.243.1.1.20251031145134399.8000.21005.dcm"
# Load CT

ct_image, doses, masks, mlc_inputs = load_rtp_data(ct_folder, dose_path=[rtdose_path], plan_path=[rtplan_path], scaling=400)
dose_volume = sitk.GetArrayFromImage(doses['dose_0'])
ct_volume = sitk.GetArrayFromImage(ct_image)
external_mask = sitk.GetArrayFromImage(masks["External"])
ct_volume = np.where(external_mask, ct_volume, -1000.0)

ct_spacing = [ct_image.GetSpacing()[0], ct_image.GetSpacing()[1], ct_image.GetSpacing()[2]]
ct_shape = ct_volume.shape

# print(config)
# for cw_1 in [True, False]:
#     for cw_2 in [True, False]:
#         for starting_angle_1 in [0.0, 180.0]:
#             for starting_angle_2 in [0.0, 180.0]: #np.linspace(178.0, 182.0, 5, endpoint=True):
ct_slices = np.array(np.expand_dims(ct_volume, 0), dtype=np.float32)
leafs_1, mus_1 = mlc_inputs[0]
results = []
config_1 = ModelConfig(preset="umea",
                    ct_array_shape=ct_shape, 
                    resolution=ct_spacing,
                    downsampling_factor=(1, 2, 2), 
                    number_of_cps=178,
                    starting_angle=180.0,
                    clockwise=True,
                    )
dose_layer_1 = DoseEngine(config_1, 15, permute_ct=False, leafs_centered=True)
jaws_1 = np.zeros(config_1.shape_jaws)
jaws_1[:, 0, :] = 0.5
jaws_1[:, 1, :] = 1.0
dose_1 = dose_layer_1(torch.tensor(np.array(leafs_1), dtype=torch.float32, device='cuda'), torch.tensor(np.array(mus_1), dtype=torch.float32, device='cuda'), torch.tensor(np.array(jaws_1), dtype=torch.float32, device='cuda'), ct_image=torch.tensor(ct_slices, dtype=torch.float32, device='cuda')).cpu().detach().numpy()

dose_pred = dose_1
# dose_pred = dose_pred * np.max(dose_volume) / np.max(dose_pred)
dose_pred = dose_pred * (np.quantile(dose_volume, 0.999) / np.quantile(dose_pred, 0.999))
print(f"{np.mean(dose_pred[0][sitk.GetArrayFromImage(masks['PTVT_42.7']) > 0])}")
print(f"{np.mean(dose_volume[sitk.GetArrayFromImage(masks['PTVT_42.7']) > 0])}")
ext_mask = sitk.GetArrayFromImage(masks["External"]) > 0
diff = ext_mask * np.abs(dose_volume - dose_pred)**2
results.append(np.mean(diff))
print(np.mean(diff))
# print(f"{starting_angle_1}/{starting_angle_2}/{cw_1}/{cw_2}:\t{np.mean(diff)}") # Baseline is 0.143663