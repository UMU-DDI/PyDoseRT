import sys
from pathlib import Path
import os

parent_dir = str(Path(__file__).resolve().parent)
sys.path.append(parent_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pydose_rt.data import MachineConfig
from pydose_rt import DoseEngine
import time

MIN = -200
MAX = 200
config = MachineConfig(
    ct_array_shape=(320, 128, 128),
    resolution=(0.125, 0.3125, 0.3125),
    field_size=(40, 40),
    # field_size=(10.0, 10.0),
    number_of_leaf_pairs=80,
    tpr_20_10=0.72,
    number_of_cps=180,
    mu_scaling=1,
    starting_angle=180,
    dtype=torch.float32,
)
x_ct = np.load("/path/to/CT.npy", allow_pickle=True)
x_ct = np.expand_dims(x_ct, axis=0)

""" # Create air everywhere
x_ct = -1000.0 * np.ones(config.ct_array_shape, dtype=np.float32)

# Dimensions
H, D, W = config.ct_array_shape
cube_h, cube_d, cube_w = 30, 10, 20

# Center positions
center_h = H // 2
center_d = D // 2
center_w = W // 2

# Calculate start and end indices for the cuboid
h_start = center_h - cube_h // 2
h_end = h_start + cube_h
d_start = 115
d_end = 125
w_start = center_w - cube_w // 2
w_end = w_start + cube_w

# Set cuboid to HU=1000 (bone)
x_ct[h_start:h_end, d_start:d_end, w_start:w_end] = 1000.0

# Calculate start and end indices for the cuboid
h_start = center_h - cube_h // 2
h_end = h_start + cube_h
d_start = center_d - cube_d // 2
d_end = d_start + cube_d
w_start = 108
w_end = 128

# Set cuboid to HU=1000 (bone)
x_ct[h_start:h_end, d_start:d_end, w_start:w_end] = 1000.0

# Add batch dimension
x_ct = np.expand_dims(x_ct, axis=0) """

beam = np.load("path/to/Beam.npy", allow_pickle=True)
leaves = beam.item()["POS"]  # shape: [2,180,80,1]

mlc = np.transpose(leaves, (3, 0, 1, 2))  # [1,2,180,80]
# Normalize to 0-1 (field size is 40cm, so -20 to +20)
mlc[:, 0, :, :] = (mlc[:, 0, :, :] - MIN) / (MAX - MIN)  # left leaves
mlc[:, 1, :, :] = (mlc[:, 1, :, :] - MIN) / (MAX - MIN)  # right leaves

mus = beam.item()["METERSET"]
mus = np.squeeze(mus)  # [180]
mus = np.expand_dims(mus, axis=0)  # [1,180]
print("Total MU:", np.sum(mus))
jaws = beam.item()["ASYM"]
jaws = np.squeeze(jaws)  # [2,180]
jaws = np.expand_dims(jaws, axis=0)  # [1,2,180]
jaws[:, 0, :] = (jaws[:, 0, :] - MIN) / (MAX - MIN)  # lower jaw
jaws[:, 1, :] = (jaws[:, 1, :] - MIN) / (MAX - MIN)  # upper jaw


dose_layer = DoseEngine(
    torch.tensor(x_ct, dtype=config.dtype, device=device), config, 11
)
start = time.time()
dose = (
    dose_layer(
        torch.tensor(mlc, dtype=config.dtype, device=device),
        torch.tensor(mus, dtype=config.dtype, device=device),
        torch.tensor(jaws, dtype=config.dtype, device=device),
    )
    .detach()
    .cpu()
    .numpy()
)
end = time.time()
print("Computation time: {:.2f} s".format(end - start))
print(dose.shape)  # [BG, H (z), D (y), W (x)]

# Central indices
central_y = dose.shape[2] // 2  # y (D)
central_z = dose.shape[1] // 2  # z (H)
central_x = dose.shape[3] // 2  # x (W)

# Axis values in cm
y = np.arange(dose.shape[2]) * config.resolution[1]  # y (D)
z = np.arange(dose.shape[1]) * config.resolution[0]  # z (H)
x = np.arange(dose.shape[3]) * config.resolution[2]  # x (W)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))


ct_coronal = x_ct[0, :, :, central_x][::-1, :]
dose_coronal = dose[0, :, :, central_x][::-1, :]
im0_ct = axs[0].imshow(
    ct_coronal,
    cmap="gray",
    extent=[y[0], y[-1], z[0], z[-1]],
    aspect=(y[-1] - y[0]) / (z[-1] - z[0]),
)
im0_dose = axs[0].imshow(
    dose_coronal,
    cmap="jet",
    vmin=np.min(dose),
    vmax=np.max(dose),
    extent=[y[0], y[-1], z[0], z[-1]],
    aspect=(y[-1] - y[0]) / (z[-1] - z[0]),
    alpha=0.5,
)
axs[0].set_xlabel("y [cm]")
axs[0].set_ylabel("z [cm]")
axs[0].set_title("Sagittal (y-z)")
fig.colorbar(im0_dose, ax=axs[0], fraction=0.046, pad=0.04)


ct_sagittal = x_ct[0, :, central_y, :][::-1, :]
dose_sagittal = dose[0, :, central_y, :][::-1, :]
im1_ct = axs[1].imshow(
    ct_sagittal,
    cmap="gray",
    extent=[x[0], x[-1], z[0], z[-1]],
    aspect=(x[-1] - x[0]) / (z[-1] - z[0]),
)
im1_dose = axs[1].imshow(
    dose_sagittal,
    cmap="jet",
    vmin=np.min(dose),
    vmax=np.max(dose),
    extent=[x[0], x[-1], z[0], z[-1]],
    aspect=(x[-1] - x[0]) / (z[-1] - z[0]),
    alpha=0.5,
)
axs[1].set_xlabel("x [cm]")
axs[1].set_ylabel("z [cm]")
axs[1].set_title("Coronal (z-x)")
fig.colorbar(im1_dose, ax=axs[1], fraction=0.046, pad=0.04)


# Axial (y-x): slice through center z (H)
ct_axial = x_ct[0, central_z, :, :]
dose_axial = dose[0, central_z, :, :]
im2_ct = axs[2].imshow(
    ct_axial,
    cmap="gray",
)
im2_dose = axs[2].imshow(
    dose_axial, cmap="jet", vmin=np.min(dose), vmax=np.max(dose), alpha=0.5
)
axs[2].set_xlabel("x [cm]")
axs[2].set_ylabel("y [cm]")
axs[2].set_title("Transversal (y-x)")
fig.colorbar(im2_dose, ax=axs[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
