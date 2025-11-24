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
- **GPU Accelerated**: CUDA-optimized computations for fast dose calculations
- **Treatment Modalities**: Support for VMAT (Volumetric Modulated Arc Therapy) and other techniques
- **Clinical Constraints**: DVH (Dose-Volume Histogram) analysis and constraint evaluation
- **Flexible API**: Easy integration with PyTorch optimization workflows

## Installation

### Requirements

- Python 3.11, 3.12, or 3.13
- CUDA-capable GPU (recommended, but CPU supported)
- Linux, macOS, or Windows

### Install from Source

```bash
# Clone the repository
git clone https://github.com/UMU-DDI/PyDoseRT.git
cd PyDoseRT

# Install in development mode
pip install -e .

# Or install with test dependencies
pip install -e ".[test]"
```

### Dependencies

PyDoseRT requires the following key packages:
- **PyTorch** (≥2.6.0) - Deep learning framework and autodiff
- **NumPy** (≥1.26.4) - Numerical computing
- **SciPy** (≥1.11.1) - Scientific computing
- **pydicom** (≥2.4.4) - DICOM file handling
- **SimpleITK** (≥2.4.1) - Medical image processing
- **pymedphys** (≥0.41.0) - Medical physics utilities

See `pyproject.toml` for the complete dependency list.

## Quick Start

### Basic Dose Calculation

```python
import torch
from pydose_rt import DoseEngine
from pydose_rt.data import MachineConfig, Phantom, Beam

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load machine configuration (linear accelerator parameters)
machine_config = MachineConfig(
    preset="src/pydose_rt/data/machine_presets/umea_10MV.json",
    number_of_leaf_pairs=60,
    tpr_20_10=0.72
)

# Create a water phantom for testing
phantom = Phantom.from_uniform_water(
    shape=(185, 167, 167),
    spacing=(3.0, 3.0, 3.0)  # mm
)

# Define a beam
beam = Beam.create(
    gantry_angle_deg=0,
    number_of_leaf_pairs=60,
    field_size=(100, 100)  # mm
)

# Initialize dose engine
dose_engine = DoseEngine(
    ct_array_shape=phantom.ct_array.shape,
    resolution=(3.0, 3.0, 3.0),
    machine_config=machine_config,
    beam_input=beam,
    device=device,
    dtype=torch.float32
)

# Calculate dose
dose = dose_engine.compute_single_beam(beam, ct_image=phantom.ct_array)

# Visualize
from pydose_rt.utils.plotting import plot_dose_distribution
plot_dose_distribution(dose, phantom.ct_array)
```

### Working with DICOM Data

```python
from pydose_rt.data import loaders

# Load patient data from DICOM files
patient, beam_sequences = loaders.load_dicom(
    ct_folder="path/to/ct_series/",
    dose_path="path/to/rtdose.dcm",
    plan_path="path/to/rtplan.dcm",
    struct_path="path/to/rtstruct.dcm"
)

# Calculate dose for existing treatment plan
dose = dose_engine.compute_beam_sequence(
    beam_sequences[0],
    patient.ct_array
)

# Compare with reference dose
difference = dose - patient.dose_array
```

### Treatment Plan Optimization

```python
from pydose_rt.data import BeamSequence, OptimizationConfig
from pydose_rt.objectives import compute_loss

# Create optimizable beam sequence
beam_sequence = BeamSequence.create(
    gantry_angles=[0, 90, 180, 270],
    number_of_leaf_pairs=60,
    field_size=(200, 200),
    requires_grad=True  # Enable gradient tracking
)

# Define clinical constraints
opt_config = OptimizationConfig(
    structures={
        "PTV": {"min_dose": 60.0, "max_dose": 66.0},
        "OAR": {"max_dose": 30.0, "type": "organ_at_risk"}
    }
)

# Optimization loop
optimizer = torch.optim.Adam(beam_sequence.parameters(), lr=0.01)

for iteration in range(100):
    optimizer.zero_grad()

    # Forward pass: calculate dose
    dose = dose_engine.compute_beam_sequence(beam_sequence, patient.ct_array)

    # Evaluate loss based on clinical constraints
    loss = compute_loss(dose, patient, opt_config)

    # Backward pass: compute gradients
    loss.backward()

    # Update beam parameters
    optimizer.step()

    if iteration % 10 == 0:
        print(f"Iteration {iteration}: Loss = {loss.item():.4f}")
```

## Architecture

### Dose Calculation Pipeline

PyDoseRT implements dose calculation as a series of differentiable layers:

1. **Fluence Map Layer** - Converts MLC/jaw positions to 2D fluence maps
2. **Fluence Volume Layer** - Projects fluence to 3D with divergent beam geometry
3. **Radiological Depth Layer** - Converts CT to radiological depth maps
4. **Pencil Beam Kernel Layer** - Generates depth-dependent dose kernels
5. **Beam-wise Convolution Layer** - Applies kernels via 3D convolution
6. **CP Rotation Layer** - Transforms dose to patient coordinates
7. **Accumulation** - Sums contributions from all control points

### Repository Structure

```
PyDoseRT/
├── src/pydose_rt/           # Main source code
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

The dose calculation uses a convolution/superposition method based on Nyholm et. al. 2006.

TODO: Add more info

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

## Examples

Explore the `examples/` directory for Jupyter notebooks demonstrating:

- **phantom.ipynb** - Basic dose calculations on a simple water phantom
- **direct_optimization.ipynb** - Treatment plan optimization workflows
- **rtplan_test.ipynb** - DICOM plan import and validation

Run scripts from the `scripts/` directory:

```bash
TODO: Fill up with meaningful scripts
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pydose_rt

# Run benchmarks
pytest tests/benchmarks/ --benchmark-only
```

## Performance

- **GPU Acceleration**: 10-100x speedup vs CPU for typical cases
- **Memory Efficiency**: Supports cropping to field-of-view and sequential beam processing
- **Mixed Precision**: FP16/FP32 support for memory-constrained scenarios
- **Batch Processing**: Multiple patients/beams in parallel

Typical performance (NVIDIA A100):
TODO: Fill in Inference times

## Limitations

- **Pencil beam model**: Less accurate than Monte Carlo for high tissue heterogeneity
- **Photon therapy only**: Electron and proton therapy not currently supported
- **Simplified MLC model**: Does not include all vendor-specific details
- **Research tool**: Not clinically validated for treatment planning

## Citation

If you use PyDoseRT in your research, please cite:

```bibtex
@article{pydosert2025,
TODO: Fill in arxiv article
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Authors

TODO: Come up with author list

**Institution**: Umeå University - Department of Diagnostics and Intervention

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or feature requests:
- Open an issue on [GitHub](https://github.com/UMU-DDI/PyDoseRT/issues)
- Contact the authors via email

---

**Disclaimer**: PyDoseRT is a research tool and has not been clinically validated. It should not be used for clinical treatment planning without proper validation and regulatory approval.