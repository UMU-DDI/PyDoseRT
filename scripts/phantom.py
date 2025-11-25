import torch
import numpy as np
import pydicom
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from pydose_rt import DoseEngine
from pydose_rt.data import MachineConfig, Phantom, Beam

machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/umea_10MV.json", ct_array_shape=(185, 167, 167), resolution=(3.0, 3.0, 3.0), number_of_leaf_pairs=60, tpr_20_10=0.72)

phantom = Phantom.from_uniform_water(shape=machine_config.ct_array_shape, spacing=machine_config.resolution)
dose_engine = DoseEngine(
    machine_config, 
    permute_ct=False, 
    leafs_centered=True
)
phantom = phantom.ct_array.to(dose_engine.dtype).to(dose_engine.device)

beam = Beam.create(gantry_angle_deg=45, machine_config=machine_config)
# mlcs, jaws, mus = dose_engine.get_open_parameters()
dose = dose_engine.forward_single_beam(beam, ct_image=phantom)
dose = dose.cpu().detach().numpy()

print("Dose shape:", dose.shape)
print("Dose max:", dose.max().item())

path ="/home/bolo/Documents/PyDoseRT/test_data/10x10-10MV/RD1.2.752.243.1.1.20240927183310596.8800.73001.dcm"
ds = pydicom.dcmread(path)
ds_ref = ds.pixel_array * float(ds.DoseGridScaling)
ds_ref = np.transpose(ds_ref, (0, 1, 2))[:, 1:-1, 1:-1]

print(np.mean(np.abs(ds_ref - dose)))
