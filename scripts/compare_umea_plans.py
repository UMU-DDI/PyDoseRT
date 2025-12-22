from sympy import false
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import pydicom
from pydose_rt import DoseEngine
from pydose_rt.data import MachineConfig, Phantom, loaders, Beam
from pydose_rt.utils.utils import sample_tensor_nearest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype=torch.float32

do_plot = True

field_sizes = [50, 100, 200]

raw_measurements = loaders.load_asc_measurements("/home/bolo/Documents/PyDoseRT/test_data/10 MV Photons/TrueBeam X10 Squares OK.asc", coord_map=("X", "Z", "Y"))
results =  []

if do_plot:
    cols = 1
    rows = 1

    fig = plt.figure(figsize=(9, 6))
    # axes = axes.flatten()
for field_size in field_sizes:
    exp_name = f"{field_size}"

    measurements = raw_measurements.copy()
    measurements = [measurement for measurement in measurements if measurement["header_dict"]["FSZ"] == [str(field_size), str(field_size)]]
    
    measurements = [measurement for measurement in measurements if measurement["header_dict"]["EDS"][2] == measurement["header_dict"]["STS"][2]]
    measurements = [measurement for measurement in measurements if measurement["header_dict"]["EDS"][2] == str(100.0)]
    measurements = [measurement for measurement in measurements if measurement["header_dict"]["EDS"][1] == str(0.0)]


    # measurements = [measurement for measurement in measurements if measurement["header_dict"]["EDS"][1] == str(0.0)]
    # measurements = [measurement for measurement in measurements if measurement["header_dict"]["EDS"][0] == str(0.0)]


    # measurements = [measurement for measurement in measurements if (float(measurement["header_dict"]["STS"][1]) == 0.0) and (float(measurement["header_dict"]["EDS"][2]) != 200.0) and (float(measurement["header_dict"]["EDS"][0]) != 0.0)]

    resolution = (1.0, 1.0, 1.0)
    ct_array_shape = (500, 500, 500)
    machine_config = MachineConfig(
        preset="src/pydose_rt/data/machine_presets/umea_10MV.json",
        # profile_corrections=None,
        )
    phantom = Phantom.from_uniform_water(shape=ct_array_shape, spacing=resolution).to(device).to(dtype)
    number_of_beams=1
    starting_angle=0
    iso_center=(250.0, 100.0, 250.0)
    kernel_size=501
    beam = Beam.create(
        gantry_angle_deg=0.0, 
        number_of_leaf_pairs=60, 
        collimator_angle_deg=0.0, 
        field_size_mm=(field_size, field_size),
        iso_center=iso_center, 
        device=device, 
        dtype=dtype)
    dose_engine = DoseEngine(
        machine_config, 
        kernel_size,
        dose_grid_spacing=phantom.resolution,
        dose_grid_shape=phantom.density_image.shape,
        beam_template=beam,
        device=device,
        dtype=dtype,
        adjust_values=False
    )

    dose = dose_engine.compute_dose(
        beam,
        density_image=phantom.density_image).detach()
    dose = dose


    for i, measurement in enumerate(measurements):
        samples = sample_tensor_nearest(dose[0, ...], resolution, np.subtract(iso_center, (0.5, 100.5, 0.5)), measurement["coords_engine"])
        samples = samples * measurement["dose"].max() / samples.max()
        # mape = np.mean(np.abs(samples - measurement["dose"])[measurement["dose"] > 0] /  measurement["dose"][measurement["dose"] > 0])
        mape = np.mean(np.abs(samples - measurement["dose"]))
        results.append(mape)

        if (do_plot):
            coords = measurement["coords_engine"]  # shape (N, 3) presumably

            # Indices of coordinates that actually change
            var_mask = np.var(coords, axis=0) != 0
            changing_idx = np.where(var_mask)[0]

            if changing_idx.size == 0:
                raise ValueError("No changing coordinates found in coords_engine.")

            # Compute physical distance along the line (only over changing components)
            # This works for 1D, 2D (diagonal), or 3D profiles
            diffs = np.diff(coords[:, changing_idx], axis=0)              # (N-1, k)
            seg_len = np.linalg.norm(diffs, axis=1)                       # (N-1,)
            dist = np.concatenate(([0.0], np.cumsum(seg_len)))            # (N,)
            ticks = dist - np.mean(dist)                                  # center at 0


            # Name the axes: your original mapping looked like [Z, X, Y]
            axis_names = np.array(["Z", "X", "Y"])

            if changing_idx.size == 1:
                # Just one axis varying: behave like before
                axis_label = f"{axis_names[changing_idx[0]]} [mm]"
            else:
                # Multiple axes varying: distance along the profile
                varying_str = "+".join(axis_names[changing_idx])
                axis_label = f"Distance along {varying_str} [mm]"

            plt.plot(ticks, samples, color="orange", linestyle="solid", label=f"PyDoseRT - {str(int(field_size / 10))}x{str(int(field_size /10))}cm")
            plt.plot(ticks, measurement["dose"], color="blue", linestyle="dashed", label=f"Measurement - {str(int(field_size / 10))}x{str(int(field_size /10))}cm")

            plt.xlabel(axis_label)
            print(f"Experiment: {exp_name}\t\tResults: {np.mean(results)}")
if do_plot:
    # plt.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"out/profiles.png")
    plt.close()
    

    