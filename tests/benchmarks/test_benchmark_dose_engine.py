import sys
sys.path.append("../../")
import numpy as np
from pydose_rt import DoseEngine
from pydose_rt.data import MachineConfig, TreatmentConfig
import torch

def test_dose_engine_layer(benchmark):
    machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/test.json")
    treatment_config = TreatmentConfig(preset="src/pydose_rt/data/treatment_presets/test.json", kernel_size=5)
    dose_layer = DoseEngine(machine_config, treatment_config)
    

    ct_array = torch.zeros(
        (
            1,
            machine_config.ct_array_shape[0],
            machine_config.ct_array_shape[1],
            machine_config.ct_array_shape[2],
        ),
        dtype=treatment_config.dtype,
        device=treatment_config.device
    )
    y_mlc, y_jaws, y_mus = dose_layer.get_open_parameters()

    benchmark(lambda: dose_layer(y_mlc, y_mus, y_jaws, ct_image=ct_array))