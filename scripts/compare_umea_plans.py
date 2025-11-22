from sympy import false
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import pydicom
from pydose_rt import DoseEngine
from pydose_rt.data import MachineConfig, TreatmentConfig, Phantom, loaders
from pydose_rt.utils.utils import sample_tensor_nearest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

do_plot = True

mlc_scatter_amplitudes = [0.07, 0.075, 0.08]
mlc_scatter_range_mms = [80, 90, 100, 110]
field_sizes = [50, 100, 200, 400]

for mlc_scatter_amplitude in mlc_scatter_amplitudes:
    for mlc_scatter_range_mm in mlc_scatter_range_mms:
        results =  []
        for field_size in field_sizes:
            machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/umea_10MV.json", ct_array_shape=(500, 500, 500), resolution=(1.0, 1.0, 1.0), number_of_leaf_pairs=60, mlc_scatter_amplitude=mlc_scatter_amplitude, mlc_scatter_range_mm=mlc_scatter_range_mm)
            treatment_config = TreatmentConfig(field_size=(400, 400), number_of_cps=1, starting_angle=0, iso_center=(0.0, 150.0, 0.0), kernel_size=501)
            phantom = Phantom.from_uniform_water(shape=machine_config.ct_array_shape, spacing=machine_config.resolution)
            dose_engine = DoseEngine(
                machine_config, 
                treatment_config, 
                permute_ct=False, 
                leafs_centered=False,
                adjust_values=False
            )
            dose_engine.eval()

            mlcs, jaws, mus = dose_engine.get_open_parameters(field_size=field_size)
            dose = dose_engine(
                mlcs, 
                mus, 
                jaws, 
                ct_image=phantom.ct_array.to(treatment_config.dtype).to(treatment_config.device))
            dose = dose

            measurements = loaders.load_asc_measurements("/home/bolo/Documents/PyDoseRT/test_data/10 MV Photons/TrueBeam X10 Squares OK.asc", coord_map=("X", "Z", "Y"))
            measurements = [measurement for measurement in measurements if measurement["header_dict"]["FSZ"] == [str(field_size), str(field_size)]]
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
                mape = np.mean(np.abs(samples - measurement["dose"])[measurement["dose"] > 0] /  measurement["dose"][measurement["dose"] > 0])
                results.append(mape)
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
            
            del machine_config, treatment_config, dose_engine, dose, phantom
        print(f"Scatter amplitude: {mlc_scatter_amplitude}\tScatter range: {mlc_scatter_range_mm}\tResults: {np.mean(results)}")

                