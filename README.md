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
- **Clinical Validation**:
  - Gamma index analysis (2%/2mm, 3%/3mm)
  - DVH constraint evaluation
  - Comparison with TPS dose distributions
- **Gradient-Based Optimization**: Optimize MLC leaf positions and monitor units directly
- **Calibration System**: Ensures accurate absolute dose at reference conditions

## Installation

### Requirements

- Python 3.11, 3.12, or 3.13
- CUDA-capable GPU (recommended, but CPU supported)
- Linux, macOS, or Windows

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
- **pymedphys** (≥0.41.0) - Medical physics utilities

See `pyproject.toml` for the complete dependency list.

## Quick Start

### Basic Dose Calculation from DICOM Data

```python
import torch
from pydose_rt import DoseEngine
from pydose_rt.data import MachineConfig, loaders
from pydose_rt.data.beam import BeamSequence

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Load machine configuration (linear accelerator parameters)
machine_config = MachineConfig(
    preset="src/pydose_rt/data/machine_presets/umea_10MV.json"
)

# Load patient DICOM data (CT, RTPLAN, RTDOSE, RTSTRUCT)
patient, beam_sequences = loaders.load_dicom(
    ct_folder="path/to/ct_series/",
    dose_path="path/to/rtdose.dcm",
    plan_path="path/to/rtplan.dcm",
    struct_path="path/to/rtstruct.dcm",
    struct_names=["CTV", "PTV", "Bladder", "Rectum", "External"],
    use_delivery=True  # Use actual delivery MUs from plan
)

# Combine beam sequences if multiple arcs
beam_sequence = BeamSequence.from_beams(
    [beam for bs in beam_sequences for beam in bs]
)

# Move data to device
patient = patient.to(device).to(dtype)
beam_sequence = beam_sequence.to(device).to(dtype)

# Get density image (masked by external contour)
density_image = torch.where(
    patient.structures["External"],
    patient.density_image,
    0.0
)

# Initialize dose engine
dose_engine = DoseEngine(
    kernel_size=251,  # Size of pencil beam kernel (larger = more accurate, slower)
    dose_grid_spacing=patient.resolution,  # Voxel spacing (mm)
    dose_grid_shape=density_image.shape,  # Grid dimensions
    machine_config=machine_config,
    beam_template=beam_sequence,
    device=device,
    dtype=dtype
)

# Calibrate dose engine to match machine output
dose_engine.calibrate(
    calibration_mu=machine_config.calibration_mu,
    original_beam_template=beam_sequence
)

# Calculate dose
dose_pred = dose_engine.compute_dose_sequential(
    beam_sequence,
    density_image=density_image
)

# Mask dose to external contour
dose_pred = torch.where(patient.structures["External"], dose_pred[0], 0.0)

# Visualize
from pydose_rt.utils.plotting import quick_plot
quick_plot(patient, dose_pred, title="Predicted Dose Distribution")
```

### Dose Validation and Metrics

```python
from pydose_rt.objectives.metrics import result_validation
from pydose_rt.data import OptimizationConfig

# Load clinical objectives from preset
optimization = OptimizationConfig.from_json(
    "src/pydose_rt/data/optimization_presets/gold-atlas.json"
)

# Validate calculated dose against reference
results = result_validation(
    patient,
    machine_config,
    beam_sequence,
    dose_pred,
    optimization,
    compute_gamma=True,  # Gamma index analysis
    compute_clinical_criteria=True,  # DVH constraint checking
    gamma_threshold_distance=2.0,  # mm
    gamma_threshold_dose=2.0  # %
)

# Print results
if "gamma_pass_rate" in results:
    print(f"Gamma pass rate (2%/2mm): {results['gamma_pass_rate']:.2%}")

if "clinical_criteria" in results:
    print(f"Clinical criteria passed: {results['clinical_criteria']['passed_test']:.1%}")

# Compare with TPS dose
import torch
mae = torch.abs(dose_pred - patient.dose).mean()
print(f"Mean Absolute Error: {mae:.3f} Gy")
```

### Treatment Plan Optimization

```python
import torch
from pydose_rt.data import BeamSequence, OptimizationConfig
from pydose_rt.objectives.losses import compute_dvh_loss, scale_loss

# Load optimization objectives
optimization = OptimizationConfig.from_json(
    "src/pydose_rt/data/optimization_presets/gold-atlas.json"
)

# Create optimizable beam sequence with gradient tracking
beam_sequence = BeamSequence.create(
    gantry_angles_deg=[0, 51, 102, 153, 204, 255, 306],  # 7 beams
    number_of_leaf_pairs=60,
    field_size=(400.0, 400.0),  # mm
    iso_center=(0.0, 0.0, 0.0),
    collimator_angles_deg=[0.0] * 7,
    sid=1000.0,  # mm
    open_field_size=100.0,  # Initial aperture
    device=device,
    dtype=dtype,
    requires_grad=True  # Enable gradient tracking for MLC leaves and MUs
)

# Initialize optimizer (AdamW works well for fluence optimization)
optimizer = torch.optim.AdamW(
    beam_sequence.parameters(),
    lr=1.0,
    weight_decay=1e-4
)

# Optimization loop
max_iterations = 100
for iteration in range(max_iterations):
    optimizer.zero_grad()

    # Forward pass: calculate dose with current beam parameters
    dose_pred = dose_engine.compute_dose(
        beam_sequence.to_delivery(),
        density_image=patient.density_image.unsqueeze(0)
    )

    # Compute loss based on clinical constraints
    losses = []

    # PTV prescription (e.g., 60 Gy)
    if "PTV" in patient.structures:
        ptv_loss = torch.mean(
            torch.abs(dose_pred[0][patient.structures["PTV"]] - 60.0)
        )
        losses.append(scale_loss(ptv_loss, optimization.structures["PTV"]["weight"]))

    # OAR sparing (minimize dose to organs at risk)
    for oar_name in ["Bladder", "Rectum", "FemoralHead_L", "FemoralHead_R"]:
        if oar_name in patient.structures:
            oar_loss = torch.mean(
                torch.abs(dose_pred[0][patient.structures[oar_name]])
            )
            losses.append(scale_loss(oar_loss, optimization.structures[oar_name]["weight"]))

    # Total loss
    total_loss = torch.stack(losses).sum()

    # Backward pass and optimization step
    total_loss.backward()
    optimizer.step()

    if iteration % 10 == 0:
        print(f"Iteration {iteration}: Loss = {total_loss.item():.4f}")
```

## Architecture

### Dose Calculation Pipeline

PyDoseRT implements dose calculation as a series of differentiable PyTorch layers that process each beam's contribution:

1. **Beam Validation Layer** - Validates beam geometry and MLC positions
2. **Fluence Map Layer** - Converts MLC leaf positions and jaw settings to 2D fluence maps, accounting for:
   - Leaf transmission
   - Source penumbra (finite source size)
   - Head scatter from collimators

3. **Fluence Volume Layer** - Projects 2D fluence maps into 3D volumes using divergent beam geometry

4. **Radiological Depth Layer** - Converts CT Hounsfield Units to radiological depth:
   - HU-to-density conversion using calibrated lookup tables
   - Ray-tracing through divergent beam geometry
   - Effective depth calculation for tissue heterogeneity correction

5. **Pencil Beam Kernel Layer** - Generates depth-dependent dose deposition kernels:
   - Primary photon dose component
   - Scatter dose with energy spectrum modeling
   - Lateral scatter based on radiological depth
   - Energy-dependent beam hardening

6. **Beam-wise Convolution Layer** - Applies pencil beam kernels via 3D FFT convolution

7. **Beam Rotation Layer** - Rotates dose distribution from beam's-eye-view to patient coordinates using trilinear interpolation

8. **Accumulation** - Sums dose contributions from all control points/beams

### Key Methods

The `DoseEngine` class provides several computation methods:

- **`compute_dose(beam_sequence, density_image)`** - Computes dose for a beam sequence in parallel (GPU memory intensive)
- **`compute_dose_sequential(beam_sequence, density_image)`** - Processes beams sequentially to reduce memory usage
- **`calibrate(calibration_mu, original_beam_template)`** - Calibrates the engine to match expected dose output at reference conditions

After initialization, the engine must be calibrated using a reference beam configuration to ensure accurate absolute dose values.

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

| Operation                                   | Time [s] |
| ------------------------------------------- | -------- |
| Single beam dose calculation                | 0.221    |
| VMAT prediction step (forward)              | 2.051    |
| VMAT optimization step (forward + backward) | 5.102    |


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