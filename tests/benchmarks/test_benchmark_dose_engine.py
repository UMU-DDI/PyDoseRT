import sys
sys.path.append("../../")
import numpy as np
from DoseEngines import DoseEngine
from DoseEngines import ModelConfig
import torch

def test_dose_engine_layer(benchmark):
    config = ModelConfig(preset="test")
    dose_layer = DoseEngine(config, 5)

    ct_array = torch.zeros(
        (
            1,
            config.ct_array_shape[0],
            config.ct_array_shape[1],
            config.ct_array_shape[2],
        ),
        dtype=torch.float32,
        device=config.device
    )
    y_mlc = torch.zeros(config.shape_mlc[0], dtype=torch.float32, device=config.device)
    y_mus = torch.zeros(config.shape_mlc[1], dtype=torch.float32, device=config.device)
    y_jaws = torch.zeros(config.shape_jaws, dtype=torch.float32, device=config.device)

    benchmark(lambda: dose_layer(y_mlc, y_mus, y_jaws, ct_image=ct_array))