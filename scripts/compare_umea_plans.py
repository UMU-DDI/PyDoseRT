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

head_scatter_amplitudes = [0.1, 1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 70.0, 90.0]
mlc_leakage_range_mms = [20]
field_sizes = [50]

for head_scatter_amplitude in head_scatter_amplitudes:
        results =  []
        for field_size in field_sizes:
            resolution = (1.0, 1.0, 1.0)
            ct_array_shape = (500, 500, 500)
            machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/umea_10MV.json", mlc_leakage_amplitude=0.0, head_scatter_amplitude=1.0, head_scatter_range_mm=head_scatter_amplitude)
            phantom = Phantom.from_uniform_water(shape=ct_array_shape, spacing=resolution).to(device).to(dtype)
            number_of_beams=1
            starting_angle=0
            iso_center=(0.0, 150.0, 0.0)
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
                image_template=phantom.density_image,
                beam_template=beam,
                device=device,
                dtype=dtype,
                adjust_values=False
            )

            dose = dose_engine.compute_dose(
                beam,
                ct_image=phantom.density_image).detach()
            dose = dose

            measurements = loaders.load_asc_measurements("/home/bolo/Documents/PyDoseRT/test_data/10 MV Photons/TrueBeam X10 Squares OK.asc", coord_map=("X", "Z", "Y"))
            measurements = [measurement for measurement in measurements if measurement["header_dict"]["FSZ"] == [str(field_size), str(field_size)]]
            measurements = [measurement for measurement in measurements if float(measurement["header_dict"]["STS"][2]) - float(measurement["header_dict"]["EDS"][2]) == 0.0]
            # measurements = [measurement for measurement in measurements if (measurement["header_dict"]["STS"][2], measurement["header_dict"]["EDS"][2]) == ('100.0', '100.0')]

            if do_plot:
                N = len(measurements)
                cols = 3
                rows = math.ceil(N / cols)

                fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3*rows))
                axes = axes.flatten()

            for i, measurement in enumerate(measurements):
                samples = sample_tensor_nearest(dose[0, ...], resolution, iso_center, measurement["coords_engine"])
                samples = samples * measurement["dose"].max() / samples.max()
                mape = np.mean(np.abs(samples - measurement["dose"])[measurement["dose"] > 0] /  measurement["dose"][measurement["dose"] > 0])
                # results.append(mape)
                thr_20 = 0.1 * measurement["dose"].max()
                thr_80 = 0.9 * measurement["dose"].max()
                results.append(np.abs(sum((samples > thr_20) * (samples < thr_80)) - sum((measurement["dose"] > thr_20) * (measurement["dose"] < thr_80))))
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
                plt.savefig(f"out/profiles_{field_size}_{head_scatter_amplitude}.png")
                plt.close()
                # plt.show()
            
            del machine_config, dose_engine, dose, phantom
        print(f"Penumbra size: {head_scatter_amplitude}\t\tResults: {np.mean(results)}")

                