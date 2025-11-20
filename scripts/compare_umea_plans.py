from sympy import false
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import pydicom
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from pydose_rt import DoseEngine
from pydose_rt.data import MachineConfig, TreatmentConfig, Phantom, loaders

import torch
import numpy as np

def sample_tensor_nearest(dose_calc, voxel_size, iso_center, xyz_mm):
    """
    dose_calc: torch.Tensor, shape (Z, Y, X)
    voxel_size: (dx, dy, dz) in mm
    xyz_mm: np.ndarray of shape (N, 3) with columns [X, Y, Z] in mm
    returns: torch.Tensor of shape (N,) with calculated dose at those points
    """
    Z, Y, X = dose_calc.shape
    dx, dy, dz = voxel_size

    # center index (isocenter at (0,0,0 mm))
    cx = ((X - 1) / 2.0) - iso_center[0]
    cy = 0 # ((Y - 1) / 2.0) - iso_center[1]
    cz = ((Z - 1) / 2.0) - iso_center[2]

    x_mm = xyz_mm[:, 0]
    y_mm = xyz_mm[:, 1]
    z_mm = xyz_mm[:, 2]

    # physical -> index space
    ix = cx + x_mm / dx
    iy = cy + y_mm / dy
    iz = cz + z_mm / dz

    # nearest voxel
    ix = torch.round(torch.from_numpy(ix)).long().clamp(0, X - 1)
    iy = torch.round(torch.from_numpy(iy)).long().clamp(0, Y - 1)
    iz = torch.round(torch.from_numpy(iz)).long().clamp(0, Z - 1)

    # sample
    return dose_calc[iz, iy, ix].cpu().detach().numpy()

do_plot = False
machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/umea.json", ct_array_shape=(500, 500, 500), resolution=(1.0, 1.0, 1.0), number_of_leaf_pairs=60, tpr_20_10=0.673)

treatment_config = TreatmentConfig(field_size=(400, 400), number_of_cps=1, starting_angle=0, iso_center=(0.0, 150.0, 0.0), kernel_size=501)

phantom = Phantom.from_uniform_water(shape=machine_config.ct_array_shape, spacing=machine_config.resolution)

dose_engine = DoseEngine(
    machine_config, 
    treatment_config, 
    permute_ct=False, 
    leafs_centered=False,
    adjust_values=False
)

mlcs, jaws, mus = dose_engine.get_open_parameters(field_size=100)
dose = dose_engine(
    mlcs, 
    mus, 
    jaws, 
    ct_image=phantom.ct_array.to(treatment_config.dtype).to(treatment_config.device))
dose = dose

measurements = loaders.load_asc_measurements("/home/bolo/Documents/PyDoseRT/test_data/6 MV Photons/TrueBeam X6 squares OK.asc", coord_map=("X", "Z", "Y"))
measurements = [measurement for measurement in measurements if measurement["header_dict"]["FSZ"] == ['100', '100']]
# measurements = [measurement for measurement in measurements if (measurement["header_dict"]["STS"][2], measurement["header_dict"]["EDS"][2]) == ('100.0', '100.0')]

if do_plot:
    N = len(measurements)
    cols = 3
    rows = math.ceil(N / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3*rows))
    axes = axes.flatten()
for i, measurement in enumerate(measurements):
    samples = sample_tensor_nearest(dose[0, ...], machine_config.resolution, treatment_config.iso_center, measurement["coords_engine"])
    samples = samples * measurement["dose"].max() / samples.max()
    mape = np.abs(samples - measurement["dose"])
    if (do_plot):
        changing_vars = np.argwhere(np.var(measurement["coords_engine"], 0) != 0)
        ticks = measurement["coords_engine"][:, changing_vars[0]]
        ax = axes[i]
        axis = ["Z", "X", "Y"][changing_vars[0][0]]
        ax.plot(ticks, samples, color="orange", linestyle="solid")
        ax.plot(ticks, measurement["dose"], color="blue", linestyle="dashed")
        ax.set_title(f"{measurement['header_dict']['STS']} - {measurement['header_dict']['EDS']}")
        ax.set_xlabel(f"{axis} [mm]")

if do_plot:
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
    