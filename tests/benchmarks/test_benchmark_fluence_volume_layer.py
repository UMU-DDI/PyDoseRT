from pathlib import Path
import sys

from pydose_rt.utils.utils import get_shapes
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import torch
from pydose_rt.layers import FluenceVolumeLayer


# ---- Fixtures -----
@pytest.fixture
def fluence_volume_layer(default_machine_config, default_resolution, default_ct_array_shape):
    """Fixture to create a FluenceMapLayer instance"""
    return FluenceVolumeLayer(default_machine_config, default_resolution, default_ct_array_shape)


def test_fluence_volume_benchmark(benchmark, default_machine_config, default_field_size, default_dtype, default_device, default_number_of_beams, fluence_volume_layer):
    shapes = get_shapes(default_machine_config, 
                        number_of_beams=default_number_of_beams,
                        field_size=default_field_size)

    y_mlc = torch.zeros(shapes["fluence_maps"], dtype=default_dtype, device=default_device)
    benchmark(lambda: fluence_volume_layer(y_mlc))
