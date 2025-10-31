import os
import sys
import gc

import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread
import torch

from types import MappingProxyType

from pydose_rt import ModelConfig

# from pydose_rt.engine.data import DataGenerator
from pydose_rt.engine.data_augment import DataGenerator
import engine.utils.plot_utils as plot_utils
from pydose_rt.layers.FluenceMapLayer import FluenceMapLayer
from pydose_rt.layers.FluenceVolumeLayer import FluenceVolumeLayer
from pydose_rt import DoseEngine

from pydose_rt.engine.data_augment import DataGenerator  # Your converted PyTorch DataGenerator
from torch.utils.data import DataLoader  # PyTorch DataLoader
from pydose_rt.engine.simple_dose_model import *

batch_size = 1
number_of_leaf_pairs = 60
number_of_cps = 1


config = ModelConfig(
    preset="lund-probe",
    number_of_leaf_pairs=number_of_leaf_pairs,
    number_of_cps=number_of_cps,
)


print(config)
x_ct = 0.0 * np.expand_dims(np.ones(config.ct_array_shape), 0)
y_mlc = np.zeros((1, 2, config.number_of_cps, config.number_of_leaf_pairs))
y_mlc[:, 0, :, :] = 0.5
y_mlc[:, 1, :, :] = 1.0
mus = np.ones((1, config.number_of_cps), dtype=np.float32)

notebook_dir = os.getcwd()
parent_dir = os.path.dirname(notebook_dir)
# data_path = os.path.join(parent_dir, "database/AUTORPT/")
data_path = "/media/bolo/Datasets/converted_lund/"
# data_path = "database/AUTORPT/"
gen = DataGenerator(data_path, "plotting", True, 1)


def prepare_real(is_hu=False):
    gen_dataset = DataGenerator(
        data_path,
        "training",
        shuffle=True,  # Shuffle handled by DataLoader now
        batch_size=1,  # This is ignored by DataLoader, but kept for DataGenerator's internal setup
        downsampling_factor=(1, 1, 1),
        is_debug=0,
        verbose=1,
    )
    # Use DataLoader to create iterable batches
    train_loader = DataLoader(
        gen_dataset,
        batch_size=1,
        shuffle=False,  # Important for training, tells DataLoader to call dataset.on_epoch_end()
        num_workers=0,  # Adjust based on your CPU cores; 0 for main process
        pin_memory=True,  # For faster GPU data transfer
    )

    for i, batch_data in enumerate(train_loader):
        # Move batch data to the appropriate device
        x, y_dose, masks, region_weights, constraints_batch = batch_data
        if i == 0:
            break

    # ct_np = np.transpose(x[0, 0, :], (2, 0, 1))
    # ptv = np.transpose(masks[0, ..., 0], (2, 0, 1))

    ct_np = x[0, 0, :]
    ptv = masks[0, ..., 0]

    X, Y, Z = ct_np.shape
    ct_torch = torch.tensor(ct_np, dtype=torch.float32)
    return ct_torch, ptv, X, Y, Z


device = "cuda" if torch.cuda.is_available() else "cpu"

ct_data, ptv, W, D, H = prepare_real(is_hu=False)

config = ModelConfig(
    preset="umea",
    number_of_leaf_pairs=number_of_leaf_pairs,
    number_of_cps=number_of_cps,
)

ct_data = ct_data.unsqueeze(0).expand(batch_size, -1, -1, -1)

x_ct = np.repeat(ct_data, 2, 0)

# plt.imshow(x_ct[0, :, :, 80], cmap="gray")


print(x_ct.shape)
print(x_ct.max())
print(x_ct.min())


def compute_plot(x_ct, y_mlc, mus, dose_layer, epoch):
    dose = dose_layer(y_mlc, mus, jaw_positions=None, ct_image=x_ct)
    print(dose.shape)

    # by z
    slice_idx = x_ct.shape[3] // 2
    plt.imshow(x_ct[0, :, :, slice_idx].cpu(), cmap="gray")
    plt.imshow(dose[0, :, :, slice_idx].cpu(), cmap="jet", alpha=0.2)
    plt.colorbar()
    plt.show()

    # # y
    # slice_idx = x_ct.shape[2] // 2
    # plt.imshow(x_ct[0, :, slice_idx, :].cpu(), cmap="gray")
    # plt.imshow(dose[0, :, slice_idx, :].cpu(), cmap="jet", alpha=0.2)
    # plt.colorbar()
    # plt.show()

    # # x
    # slice_idx = x_ct.shape[1] // 2
    # plt.imshow(x_ct[0, slice_idx, :, :].cpu(), cmap="gray")
    # plt.imshow(dose[0, slice_idx, :, :].cpu(), cmap="jet", alpha=0.2)
    # plt.colorbar()
    # plt.show()

    return dose


def process(config):
    print()
    print()
    # print("config:", config)
    device = config.device

    fluence_map_layer = FluenceMapLayer(config)

    H, D, W = config.ct_array_shape
    h_min_idx = 0
    h_max_idx = H - 1
    w_min_idx = 0
    w_max_idx = W - 1
    fluence_volume_layer = FluenceVolumeLayer(config)

    y_mlc = np.zeros((2, 2, config.number_of_cps, config.number_of_leaf_pairs))

    center_leaf_start_idx = (60 // 2) - (10 // 2)
    center_leaf_end_idx = center_leaf_start_idx + 10
    y_mlc[:, 0, :, center_leaf_start_idx:center_leaf_end_idx] = 0.5
    y_mlc[:, 1, :, center_leaf_start_idx:center_leaf_end_idx] = 0.5

    y_mlc[:, 0, :, :] = 0.5
    y_mlc[:, 1, :, :] = 0.1

    fluence_map = fluence_map_layer(torch.tensor(y_mlc, device=device)).cpu()
    # plt.imshow(fluence_map[0, 0, :, :].cpu())
    # plt.colorbar()
    # plt.show()

    fluence_volume = fluence_volume_layer(
        torch.tensor(fluence_map, device=device),
        (h_min_idx, h_max_idx, w_min_idx, w_max_idx),
    ).cpu()
    print("fluence_volume shape:", np.shape(fluence_volume))

    slice_number = int(np.shape(fluence_volume)[3] / 2)
    slice_y_number = int(np.shape(fluence_volume)[2] / 2)
    # plt.imshow(fluence_volume[0, :, :, slice_number, 0], interpolation="none")
    # plt.colorbar()
    # plt.show()

    dose_layer = DoseEngine(config, 15, leafs_centered=True, permute_ct=True)
    save_path = os.path.join(parent_dir, "database/temp/")
    # y_mlc = np.zeros((2, 2, config.number_of_cps, config.number_of_leaf_pairs))
    mus = np.array(np.ones((2, config.number_of_cps)), dtype=np.float32) * 1
    mus = mus * (30.0 / 18.0 * 2.0) / config.number_of_cps * 100
    # mus = mus * (1.5 / 0.464)

    dose = compute_plot(
        torch.tensor(x_ct * 1000, device=device, dtype=torch.float32),
        torch.tensor(y_mlc, device=device, dtype=torch.float32),
        torch.tensor(mus, device=device, dtype=torch.float32),
        dose_layer,
        epoch=0,
    ).cpu()
    print("dose min and max:", dose.numpy().min(), dose.numpy().max())
    print()
    print()

    # del dose
    # gc.collect()
    # torch.cuda.empty_cache()


process(
    config=ModelConfig(
        preset="lund-probe",
        number_of_leaf_pairs=number_of_leaf_pairs,
        number_of_cps=1,
    )
)
process(
    config=ModelConfig(
        preset="lund-probe",
        number_of_leaf_pairs=number_of_leaf_pairs,
        number_of_cps=3,
    )
)
process(
    config=ModelConfig(
        preset="lund-probe",
        number_of_leaf_pairs=number_of_leaf_pairs,
        number_of_cps=5,
    )
)
process(
    config=ModelConfig(
        preset="lund-probe",
        number_of_leaf_pairs=number_of_leaf_pairs,
        number_of_cps=15,
    )
)
# process(
#     config=ModelConfig(
#         preset="lund-probe",
#         number_of_leaf_pairs=number_of_leaf_pairs,
#         number_of_cps=60,
#     )
# )
# process(
#     config=ModelConfig(
#         preset="lund-probe",
#         number_of_leaf_pairs=number_of_leaf_pairs,
#         number_of_cps=90,
#     )
# )

# process(
#     config = ModelConfig(
#         preset="lund-probe",
#     )
# )


"""
CUDA_VISIBLE_DEVICES=1 nohup python scripts/learned_optimize_deviation_val.py --is_comet 0 --is_debug 0 --number_of_cps 180 --batch_size 1 --initial_filters 16 --vae_n_filters 64 --weight_PTV 100 --constraint_mode fixed --is_load_pretrained 1 --downsampling_factor 2 --epochs 300 > d1.log &

CUDA_VISIBLE_DEVICES=2 nohup python scripts/learned_optimize_deviation_val.py --is_comet 0 --is_debug 0 --number_of_cps 180 --batch_size 1 --initial_filters 16 --vae_n_filters 64 --weight_PTV 100 --constraint_mode fixed --is_load_pretrained 1 --downsampling_factor 2 --epochs 300 > d2.log &

CUDA_VISIBLE_DEVICES=0 nohup python scripts/train.py --is_comet 1 --is_debug 0 --batch_size 2 --initial_filters 16 --lr 0.0005 > d0.log &
"""
