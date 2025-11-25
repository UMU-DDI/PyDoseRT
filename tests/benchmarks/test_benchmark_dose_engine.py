import sys

from pydose_rt.utils.utils import get_shapes
sys.path.append("../../")
import numpy as np
from pydose_rt import DoseEngine
from pydose_rt.data import MachineConfig, BeamSequence
import torch

def test_dose_engine_layer(benchmark, default_ct_array_shape, default_resolution, default_gantry_angles, default_number_of_cps, default_kernel_size, default_field_size, default_beam_limiting_device_angles, default_iso_center, default_sid, default_device, default_dtype):
    machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/test.json")
    shapes = get_shapes(machine_config,
                        default_ct_array_shape,
                        number_of_cps=default_number_of_cps,
                        kernel_size = default_kernel_size,
                        field_size=default_field_size)
    beam_sequence = BeamSequence.from_tensors(torch.zeros(shapes["MLCs"]), torch.ones(shapes["MUs"]), torch.zeros(shapes["jaws"]), default_gantry_angles, default_beam_limiting_device_angles, default_iso_center, default_sid, default_field_size)
    dose_layer = DoseEngine(default_ct_array_shape, 
                            default_resolution, 
                            machine_config,
                            beam_sequence,
                            default_kernel_size,
                            default_device,
                            default_dtype)
    

    ct_array = torch.zeros(
        (
            1,
            default_ct_array_shape[0],
            default_ct_array_shape[1],
            default_ct_array_shape[2],
        ),
        dtype=dose_layer.dtype,
        device=dose_layer.device
    )

    benchmark(lambda: dose_layer.compute_beam_sequence(beam_sequence, ct_image=ct_array))