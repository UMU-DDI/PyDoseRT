import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from pydose_rt import DoseEngine
from pydose_rt.data import MachineConfig, TreatmentConfig, Phantom

machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/umea.json", ct_array_shape=(185, 167, 167), resolution=(3.0, 3.0, 3.0), number_of_leaf_pairs=5, tpr_20_10=0.72)

treatment_config = TreatmentConfig(field_size=(100, 100), number_of_cps=1, starting_angle=0, iso_center=(0.0, 150.0, 0.0), kernel_size=55)

phantom = Phantom.from_sphere(shape=machine_config.ct_array_shape, spacing=machine_config.resolution, radius_mm=50.0)

dose_engine = DoseEngine(
    machine_config, 
    treatment_config, 
    permute_ct=False, 
    leafs_centered=True
)

mlcs, jaws, mus = dose_engine.get_open_parameters()
dose = dose_engine(
    mlcs, 
    mus, 
    jaws, 
    ct_image=phantom.ct_array.to(treatment_config.dtype).to(treatment_config.device))

print("Dose shape:", dose.shape)
print("Dose max:", dose.max().item())