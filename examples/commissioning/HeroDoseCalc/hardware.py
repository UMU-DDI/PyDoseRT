"""
Hardware and runtime utilities for the dose calculation pipeline.
"""
from typing import Tuple
import torch


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


DEVICE = get_device()
print(f"Running on device: {DEVICE}")


class MemoryManager:
    """Heuristics for estimating VRAM usage before running heavy kernels."""

    @staticmethod
    def check_vram(grid_shape: Tuple[int, int, int], batch_size: int, n_slabs: int) -> bool:
        if not torch.cuda.is_available():
            return True
        nx, ny, nz = grid_shape
        float_size = 4
        n_voxels = nx * ny * nz
        peak_gb = (
            (n_voxels * 2)
            + (batch_size * n_slabs * nx * nz)
            + (batch_size * n_voxels * 3)
        ) * float_size / 1e9
        free_mem, _ = torch.cuda.mem_get_info()
        free_gb = free_mem / 1e9
        if peak_gb > (free_gb * 0.9):
            print(f"⚠️  WARNING: Est. VRAM usage {peak_gb:.2f}GB > 90% of {free_gb:.2f}GB.")
            return False
        return True
