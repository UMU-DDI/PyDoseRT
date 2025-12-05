import sys

from pydose_rt.utils.utils import get_shapes
sys.path.append("../../")
import numpy as np
from pydose_rt import DoseEngine
from pydose_rt.data import MachineConfig, BeamSequence
import torch

def test_dose_engine_layer(benchmark, default_ct_array_shape, default_resolution, default_gantry_angles, default_number_of_beams, default_kernel_size, default_field_size, default_machine_config, default_collimator_angles, default_iso_center, default_sid, default_device, default_dtype):
    machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/test.json")
    shapes = get_shapes(machine_config,
                        default_ct_array_shape,
                        number_of_beams=default_number_of_beams,
                        kernel_size = default_kernel_size,
                        field_size=default_field_size)
    beam_sequence = BeamSequence.from_tensors(torch.zeros(shapes["MLCs"][1:], dtype=default_dtype, device=default_device), torch.ones(shapes["MUs"][1:], dtype=default_dtype, device=default_device), torch.zeros(shapes["jaws"][1:], dtype=default_dtype, device=default_device), default_gantry_angles, default_collimator_angles, default_iso_center, default_sid, default_field_size)

    ct_array = torch.zeros(default_ct_array_shape,
        dtype=default_dtype,
        device=default_device
    )

    dose_layer = DoseEngine(default_machine_config,
                            kernel_size=default_kernel_size,
                            dose_grid_spacing=default_resolution,
                            dose_grid_shape=ct_array.shape,
                            beam_template=beam_sequence,
                            device=default_device,
                            dtype=default_dtype)
    

    benchmark(lambda: dose_layer.compute_dose(beam_sequence, density_image=ct_array))