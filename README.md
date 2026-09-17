# PyDoseRT

A **differentiable radiation therapy dose calculation engine** for automated treatment planning, built on PyTorch.

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-red.svg)](https://pytorch.org/)

## Overview

PyDoseRT implements a physics-based **pencil beam convolution model** with full gradient support, enabling gradient-based optimization of radiation therapy treatment plans. The engine is designed for researchers and medical physicists developing automated treatment planning algorithms.

### Key Features

- **Fully Differentiable**: All operations support automatic differentiation for gradient-based optimization
- **Physics-Based Modeling**: Pencil beam convolution with tissue heterogeneity, scatter, and penumbra effects
- **DICOM Integration**: Native support for CT, RTDOSE, RTPLAN, and RTSTRUCT files
  - Load existing treatment plans from TPS systems
  - Import patient CT scans and structure sets
  - Validate calculated dose against reference RTDOSE
- **GPU Accelerated**: CUDA-optimized computations for fast dose calculations
  - Sequential processing mode for memory-efficient computation
  - Parallel processing for maximum speed
- **Treatment Modalities**: Support for VMAT (Volumetric Modulated Arc Therapy), IMRT, and static fields
- **Proton Pencil Beam**: Differentiable ion dose engine with a heterogeneity-aware
  multiple-Coulomb-scattering model and a commissioned kernel table that can itself be
  calibrated by gradient descent
- **Clinical Validation**:
  - Gamma index analysis (2%/2mm, 3%/3mm)
  - DVH constraint evaluation
  - Comparison with TPS dose distributions
- **Gradient-Based Optimization**: Optimize MLC leaf positions and monitor units directly
- **Calibration System**: Ensures accurate absolute dose at reference conditions

## Quick start

For getting started, we have collected some example data and workflows using PyDoseRT in different scenarios:
- [Dose calculation from a DICOM RTPLAN](https://colab.research.google.com/github/UMU-DDI/PyDoseRT/blob/main/examples/rtplan.ipynb)
- [Machine parameter optimization using gradient descent](https://colab.research.google.com/github/UMU-DDI/PyDoseRT/blob/main/examples/optimization.ipynb)
- [PyDoseRT in a deep learning model](https://colab.research.google.com/github/UMU-DDI/PyDoseRT/blob/main/examples/dlmodel.ipynb)
- [Water phantom evaluations](https://colab.research.google.com/github/UMU-DDI/PyDoseRT/blob/main/examples/phantom.ipynb)
- [Pencil beam kernel visualization](https://colab.research.google.com/github/UMU-DDI/PyDoseRT/blob/main/examples/pbkernel.ipynb)

## Installation

### Requirements

- Python 3.11, 3.12, or 3.13
- CUDA-capable GPU (recommended, but CPU supported)
- Linux, macOS, or Windows


### Install through pip

The latest stable release of PyDoseRT is available through pip, and it can be installed by running:

```bash
pip install pydosert
```

### Install from Source

PyDoseRT is currently under active development. Install in editable mode to get the latest updates:

```bash
# Clone the repository
git clone https://github.com/UMU-DDI/PyDoseRT.git
cd PyDoseRT

# Install in editable/development mode (recommended)
pip install -e .

# Or install with test dependencies
pip install -e ".[test]"
```

The `-e` flag installs the package in editable mode, which means changes to the source code are immediately reflected without reinstalling. This is recommended for development and staying up-to-date with the latest improvements.

### Dependencies

PyDoseRT requires the following key packages:
- **PyTorch** (≥2.6.0) - Deep learning framework and autodiff
- **NumPy** (≥1.26.4) - Numerical computing
- **SciPy** (≥1.11.1) - Scientific computing
- **pydicom** (≥2.4.4) - DICOM file handling
- **SimpleITK** (≥2.4.1) - Medical image processing
- (**pymedphys** (≥0.41.0) - Medical physics utilities only used for gamma pass rate evaluations)

See `pyproject.toml` for the complete dependency list.

## Architecture

### Dose Calculation Pipeline

PyDoseRT implements dose calculation as a series of differentiable PyTorch layers that process each beam's contribution:

1. **Fluence Map Layer** - Converts MLC leaf positions and jaw settings to 2D fluence maps, accounting for:
   - Leaf transmission
   - Source penumbra (finite source size)
   - Head scatter from collimators

2. **Fluence Volume Layer** - Projects 2D fluence maps into 3D volumes using divergent beam geometry

3. **Radiological Depth Layer** - Converts CT Hounsfield Units to radiological depth:
   - HU-to-density conversion using calibrated lookup tables
   - Ray-tracing through divergent beam geometry
   - Effective depth calculation for tissue heterogeneity correction

4. **Pencil Beam Kernel Layer** - Generates depth-dependent dose deposition kernels:
   - Primary photon dose component
   - Scatter dose with energy spectrum modeling
   - Lateral scatter based on radiological depth
   - Energy-dependent beam hardening

5. **Beam-wise Convolution Layer** - Applies pencil beam kernels

6. **Beam Rotation Layer** - Rotates dose distribution from beam's-eye-view to patient coordinates using trilinear interpolation

7. **Accumulation** - Sums dose contributions from all control points/beams

### Key Methods

The `DoseEngine` class provides several computation methods:

- **`compute_dose(beam_sequence, density_image)`** - Computes dose for a beam sequence in parallel (GPU memory intensive)
- **`compute_dose(beam_sequence, density_image, beam_chunk_size=N)`** - Processes beams in gradient-checkpointed chunks of `N` to reduce peak memory while preserving gradients. The chunk geometry is cached and reused across calls. `beam_chunk_size` can also be set once on the `DoseEngine(...)` constructor.
- **`calibrate(calibration_mu, original_beam_template)`** - Calibrates the engine to match expected dose output at reference conditions

After initialization, the engine must be calibrated using a reference beam configuration to ensure accurate absolute dose values.

### Repository Structure

```
PyDoseRT/
├── src/pydosert/           # Main source code
│   ├── engine/              # Core dose calculation engine
│   ├── data/                # Data structures and DICOM loaders
│   ├── layers/              # Computation layers (fluence, convolution, etc.)
│   ├── physics/             # Physics models (kernels, attenuation, scatter)
│   ├── geometry/            # Geometric transformations
│   ├── objectives/          # Loss functions and metrics
│   └── utils/               # Utilities and visualization
├── examples/                # Jupyter notebook tutorials
├── scripts/                 # Command-line scripts
├── tests/                   # Test suite
│   ├── unittests/          # Unit tests
│   ├── benchmarks/         # Performance tests
│   └── smoketests/         # Integration tests
└── pyproject.toml          # Package configuration
```

## Machine Configurations

PyDoseRT includes preset configurations for common linear accelerators:

TODO: Offer meaningful template
- **Generic configurations** - Customizable templates

You can create custom machine configurations by providing:
- MLC geometry (leaf widths, positions)
- Source characteristics (SSD, energy)
- Beam quality parameters (TPR 20/10)
- Collimation system parameters

## Physics Model

### Pencil Beam Convolution

The dose calculation uses a parameterized convolution method based on Nyholm et. al. 2006.

```bibtex
@article{Nyholm2006,
   title = {Photon pencil kernel parameterisation based on beam quality index},
   author = {Tufve Nyholm and Jörgen Olofsson and Anders Ahnesjö and Mikael Karlsson},
   doi = {10.1016/j.radonc.2006.02.002},
   journal = {Radiotherapy and Oncology},
   year = {2006}
}
```

For a deeper understanding of the kernel computations, run `examples/kernel.ipynb`.

### Tissue Heterogeneity

CT Hounsfield Units (HU) are converted to radiological depth using:
- Linear density-HU lookup tables
- Ray-tracing through divergent beam geometry
- Effective depth scaling for each beamlet

### Additional Effects

- **MLC scatter and transmission** - Leaf leakage and interleaf effects
- **Head scatter** - Collimator-dependent scatter contribution
- **Source penumbra** - Geometric penumbra from finite source size
- **Tongue-and-groove effect** - MLC interdigitation

## Proton Dose Calculation

PyDoseRT also ships a **differentiable proton pencil-beam engine**, `IonDoseEngine`. It is
independent of the photon pipeline: it shares no layers and no state, and enabling it changes
nothing about photon dose calculation.

### Pipeline

An ion call computes the dose of a *batch of beamlets* which are independent pencil beams, each with its own gantry angle, energy, spot position, spot size and weight. They are processed in a per-beam beam's-eye view (BEV) and are rotated then back into the patient frame:

1. **BEV resampling** - The stopping-power-ratio volume is resampled into each beamlet's BEV
   window and integrated along depth into a per-column water-equivalent depth (WEQ)
2. **Lattice pencil beam** - Each beamlet is split into `n**2` sub-beams on a quarter-FWHM
   grid; every sub-beam looks up its own WEQ column in the kernel table, so both the
   Bragg-peak depth and the lateral sigma respond to the tissue that sub-beam crosses
3. **Lateral model** - A narrow multiple-Coulomb-scattering core per sub-beam plus a broad
   nuclear halo laid down once per beamlet, both cell-integrated with `erf` rather than
   point-sampled
4. **Heterogeneity correction** - A Fermi-Eyges lever-arm variance excess added to the core
   sigma, identically zero in homogeneous water by construction
5. **Correction hook** - An optional `bev_correction` module sees the complete BEV payload
   before it is rotated back; this is the documented injection point for a learned residual
   model
6. **Finalisation** - Rotation into the patient frame and conversion from MeV to Gy

Everything is plain differentiable PyTorch: gradients flow to the beamlet weights, positions,
energies and sigmas, and to the commissioned kernel table itself.

### Minimal example

```python
import torch
from importlib import resources

from pydosert import IonBeamletBatch, IonDoseEngine, IonMachineConfig
from pydosert.physics import IonKernelTable

# 1. Commissioned base data: one .npz per machine, shipped as package data.
table_path = resources.files("pydosert.data").joinpath("machine_presets/protons_doserad.npz")
kernel_table = IonKernelTable.load(table_path, device="cpu", dtype=torch.float32)

# 2. A batch of beamlets: one gantry angle, one energy, one spot each.
grid_shape = (48, 200, 48)          # (H, D, W) voxels
spacing_mm = (1.0, 1.0, 1.0)        # (rh, rd, rw)
energy_mev = 120.4273              # tabulated; kernel_table.available_energies lists all 114
beamlets = IonBeamletBatch.create(
    gantry_angle_deg=[0.0],
    position_mm=[[0.0, 0.0]],
    energy_mev=[energy_mev],
    sigma_mm=[6.0],                 # isotropic initial spot sigma
    weight=[1e8],                   # particles
    iso_center_mm=[23.5, 100.0, 23.5],
    sad_mm=10_000.0,
)

# 3. The engine.
engine = IonDoseEngine(
    machine_config=IonMachineConfig(),
    kernel_table=kernel_table,
    dose_grid_spacing=spacing_mm,
    dose_grid_shape=grid_shape,
    field_size=(32, 32),            # BEV window in voxels, per beamlet
)

# 4. Dose, in Gy, on a water phantom scored everywhere.
density = torch.ones(grid_shape)            # stopping-power ratio, not mass density
dose_mask = torch.ones(grid_shape, dtype=torch.bool)
dose = engine.compute_dose(beamlets, density, dose_mask)

central_axis = dose[0, 24, :, 24]           # depth profile through the beam axis
print(f"dose {tuple(dose.shape)}, peak {float(dose.max()):.4f} Gy, "
      f"Bragg peak at {int(central_axis.argmax())} mm depth")
# dose (1, 48, 200, 48), peak 0.1548 Gy, Bragg peak at 105 mm depth
```

`compute_dose(..., return_per_beamlet=True)` returns one cropped dose per beamlet from the
same single BEV pass, instead of the summed volume.

### Commissioning and calibration

- `commissioning/conversion/convert_proton_mat_to_npz.py` converts a pyRadPlan/matRad proton
  machine `.mat` into the `.npz` kernel table `IonKernelTable` loads. Rows are stored
  padded-rectangular rather than resampled, so they round-trip bit-exactly.
- `commissioning/calibrate_ion_kernel_table.py` calibrates a table against water-phantom
  Monte Carlo **by backpropagation through the engine**. `IonKernelCalibration` holds
  learnable per-row residuals that are exactly zero at initialisation — a table carrying an
  untrained calibration computes bit-identically to the table it was built from — and that
  cannot produce an unphysical curve for any parameter value.


## Examples

### Jupyter Notebooks

Explore the `examples/` directory for interactive tutorials:

- **`phantom.ipynb`** - Basic dose calculations on water phantoms and simple geometries
- **`direct_optimization.ipynb`** - Treatment plan optimization workflows with gradient descent
- **`kernels.ipynb`** - Understanding pencil beam kernel computation and physics models
- **`rtplan_test_1arc.ipynb`** - Loading and validating DICOM RT plans (VMAT example)


## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pydosert

# Run benchmarks
pytest tests/benchmarks/ --benchmark-only
```

## Performance

- **GPU Acceleration**: 10-100x speedup vs CPU for typical cases
- **Memory Efficiency**: Supports cropping to field-of-view and sequential beam processing
- **Mixed Precision**: FP16/FP32 support for memory-constrained scenarios
- **Batch Processing**: Multiple patients/beams in parallel

Typical performance (NVIDIA A100):

| Operation                                   | Time [s] |
| ------------------------------------------- | -------- |
| Single beam dose calculation                | 0.221    |
| VMAT prediction step (forward)              | 2.051    |
| VMAT optimization step (forward + backward) | 5.102    |


## Limitations

- **Pencil beam model**: Less accurate than Monte Carlo for high tissue heterogeneity
- **No electron therapy**: Photons and protons only
- **Protons report dose to water**: no dose-to-medium conversion is applied
- **Simplified MLC model**: Does not include all vendor-specific details
- **Research tool**: Not clinically validated for treatment planning

## Citation

If you use PyDoseRT in your research, please cite:

```bibtex
@article{Simko2025,
      title={A physics-informed, plug-and-play dose engine for gradient-based radiotherapy treatment planning}, 
      author={Attila Simkó and Matthias Kronsteiner and Simon Glatzer and Minh Vu and Josef A. Lundman and Joakim Jonsson and Jörgen Olofsson and Kristina Sandgren and Wolfgang Lechner and Dietmar Georg and Tommy Löfstedt and Tufve Nyholm and Anders Garpebring and Gerd Heilemann},
      year={2025},
      url={https://arxiv.org/abs/2512.18863}, 
}
```

PyDoseRT was developed in collaboration between Umeå University (Department of Diagnostics and Intervention) and the Medical University of Vienna (Department of Radiation Oncology).

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or feature requests:
- Open an issue on [GitHub](https://github.com/UMU-DDI/PyDoseRT/issues)
- Contact the authors via [email](attila.simko@umu.se)

---

**Disclaimer**: PyDoseRT is a research tool and has not been clinically validated. It should not be used for clinical treatment planning without proper validation and regulatory approval.
