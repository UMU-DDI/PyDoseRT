# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
PyDoseRT uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
This changelog was introduced after releasing version 1.3.0.

## [Unreleased]

### Added
- `load_dicom(pad_to_cylinder=True)` pads the axial plane so that no tissue leaves the grid when a slice is rotated about the isocentre, and moves the beams with it. The helpers behind it, `body_cylinder_radius_mm`, `pad_to_cylinder` and `crop_from_cylinder` (which brings dose back to the original grid), are public. Without padding the engine now warns once, on the first dose, how many non-air voxels the rotation drops.
- `pydosert.geometry.conventions` is the single definition of the axis conventions (volume layout, isocentre order/units/origin, gantry rotation plane and direction, units), exported as `pydosert.CONVENTIONS`. Every photon example now opens with the same summary, and `tests/unittests/test_conventions.py` checks each statement against the layer that implements it.
- `IonBeamletBatch` supports indexing (`batch[i]`, slices, index tensors and bool masks, always returning a batch) and `chunks(n)` for walking it in sub-batches. Slices are views, so gradients flow back to the parent.
- `IonDoseEngine` takes `beamlet_chunk_size`, on the constructor or per `compute_dose` call, computing the beamlets in gradient-checkpointed groups. Peak memory becomes flat in the spot count instead of linear (190 MiB at every size from G=64 to G=2048 on a 32x150x32 grid, against 317 MiB to out-of-memory unchunked). The result is unchanged for any chunk size.
- Proton pencil-beam dose calculation. `IonDoseEngine` computes the dose of a batch of ion beamlets on a beam's-eye-view lattice and is differentiable end to end, including an optional `bev_correction` hook for a learned residual model. Supporting types: `IonKernelTable` (commissioned base data in a padded-rectangular `.npz`), `IonBeamletBatch`, `IonMachineConfig`, `fermi_eyges_excess` (heterogeneity-aware multiple-Coulomb-scattering correction, identically zero in water) and the BEV geometry helpers in `pydosert.geometry.bev`. Nothing in the photon pipeline changes.
- `IonKernelCalibration` plus `commissioning/calibrate_ion_kernel_table.py`: learnable per-row residuals over a kernel table, fitted against water-phantom Monte Carlo by backpropagating through the engine. The residuals are exactly zero at initialisation, so an untrained calibration reproduces the input table bit-for-bit.
- `commissioning/conversion/convert_proton_mat_to_npz.py` converts a pyRadPlan/matRad proton machine `.mat` into the kernel-table `.npz`.
- `patient_dose_mask` builds the dose-scoring mask topologically, keeping internal air (trachea, bowel gas, sinuses) that a density threshold would zero. `IonDoseEngine.compute_dose` requires the mask explicitly and applies no threshold of its own.
- `pydosert.exceptions` with the error types the package raises: `ShapeError`, `DeviceDtypeError`, `EngineStateError`, `StructureError` and `GeometryError`, all deriving from `PyDoseRTError` and from the built-in they replace (so `except ValueError` keeps working).
- New composable objective primitives in `pydosert.objectives.losses`: `upper_penalty`, `lower_penalty`, `mean_upper_penalty` and `squared_penalty` (one- and two-sided squared-hinge penalties on voxel doses), plus `geud` for the generalized equivalent uniform dose. `geud` is a reduction rather than a loss, so it composes with the penalties to build EUD objectives (e.g. `upper_penalty(geud(x, 2.5), c)` is a max-EUD constraint).
- New `pydosert.objectives.regularizers` module holding the deliverability regularizers `mus_loss`, `leafs_loss` and `jaws_loss`, each returning a (rate, complexity) pair derived from the machine limits.
- `condition_aperture_pair` and `condition_beam_params` in `pydosert.data.beam` give direct optimization and deep-learning workflows one shared differentiable map from unconstrained variables to physical ordered leaf/jaw pairs and positive MUs. The MU scale is normalized by the control-point count so the raw variables stay ~O(1) regardless of the number of control points.
- New plotting functions: `plot_mu_polar`, `plot_fluence_and_mu`, `plot_dvh`, `plot_profiles` and `plot_kernel`, with the reusable `compute_fluence_maps` and `compute_dvh_curves` helpers behind them.

### Changed
- **Breaking**: `Beam` and `BeamSequence` are frozen dataclasses. Their derive-a-copy methods (`to`, `to_delivery`, `stack`, `__getitem__`) already returned new objects, so rebinding a field was never supported; it now raises `FrozenInstanceError` instead of silently doing nothing -- which on the `Beam` from `seq[0]`, a view into the sequence, was actively misleading. Use `dataclasses.replace`. Autograd is unaffected, and both types are now hashable.
- **Breaking**: the photon engine resolves `device` and `dtype` at construction (explicit value, else the beam template's, else CUDA-if-available / float32) rather than adopting whatever the first input tensor happened to be, and rejects inputs that disagree with `DeviceDtypeError`. A non-floating `dtype` is refused up front.
- **Breaking**: the photon engine, `Patient` and `BeamSequence` raise those types instead of `assert` and bare `Exception`. Asserts disappear under `python -O`, which is exactly when a silent wrong-shape input is worst, and a bare `Exception` cannot be caught selectively. Messages now name the offending value and the way out.
- **Breaking**: the plotting functions have been renamed to a consistent `plot_*` scheme: `print_paper_plot` is now `plot_overview`, `print_comparison_plot` is now `plot_comparison`, and `make_animation` is now `plot_animation`.
- `utils.py` docstrings have been converted to the Google style used elsewhere in the package.
- The example notebooks have been updated to the new objective, conditioning and plotting APIs.

### Fixed
- The radiological-depth ray caster placed the isocentre at `iso / resolution + 0.5`, half a voxel further along every axis than `FluenceVolumeLayer` and `build_rotation_grids`, so the depth ray ran beside the beam axis whose radiological depth it measures rather than along it. All three now use `iso / resolution`. Dose changes most at field edges and in the entrance region, and a four-field box in water sits twice as close to its isocentre (0.81 mm off, now 0.41 mm).
- `Patient.device` and `Patient.dtype` read the density-image tensor directly; they previously referenced a non-existent `.attenuation.data` attribute and raised.
- `load_structures` returns early when no structure set is given, instead of nesting the whole body in a conditional.

### Removed
- **Breaking**: the ad-hoc loss collection in `pydosert.objectives.losses` has been removed in favour of the composable primitives above: `scale_loss`, `constraint_loss`, `compute_l2_loss`, `dose_loss`, `compute_loss`, `compute_dvh_loss`, `compute_mae_loss`, `leaf_range_loss`, `create_sphere_mask`, `cosine_warmup_scheduler`, `dvh_percentile_objective`, `dvh_volume_objective`, `dvh_percentile_loss_with_threshold`, `dvh_volume_loss_with_threshold`, `dvh_Dp_loss` and `dvh_Vx_loss`.
- **Breaking**: `mus_loss`, `leafs_loss` and `jaws_loss` have moved from `pydosert.objectives.losses` to `pydosert.objectives.regularizers`.
- **Breaking**: the `print_results` and `quick_plot` plotting functions have been removed.
- The unused `get_initial_weights` helper has been removed from `utils.py`.

## [1.4.0]

### Added
- Five new example files are now available through the repository. They are suitable for running in a T4 google colab environment.
- The commissioning process has a new setting for kernel size, this will be used throughout. A known limitation when evaluating the engine with a smaller kernel, but it was shown to work well empirically.
- `compute_dose` now accepts a `beam_chunk_size` argument (also settable on the `DoseEngine` constructor) that processes beams in gradient-checkpointed chunks to lower peak memory on large problems while retaining gradients. The per-chunk beam geometry is cached and reused across calls.
- New `PhotonBaseEngine` base class holding the engine scaffolding (construction, device/dtype handling, input validation, `compute_dose` orchestration, beam chunking with geometry caching, and calibration). New photon engines can subclass it and implement the pipeline hooks (`_initialize_layers`, `_full_geometry`, `_build_chunk_geometry`, `_forward_core`).
### Changed
- `DoseEngine` now subclasses `PhotonBaseEngine` and only implements the pencil-beam pipeline hooks. Its public interface is unchanged.
- Machine/Optimization configurations are now built-in to the package, and easier to access. To get a list of all available presets, run `list_machine_presets()` or `list_optimization_presets()`. All related tests have been updated. 
- Changed commissioning pipeline to use json files
### Fixed
- The correct email adresses are now in the pyproject file.
- An axis flip bug was fixed in the pencil beam model.
- Mask out-of-bounds ray points in RadiologicalDepthLayer instead of clamping
- Align the rad-depth ray rotation center with the align_corners=False convention (+0.5 voxel shift).
- Aspect-correct the affine_grid rotation matrices in build_rotation_grids and rotate_2d_images.
- Removed warning for leaf_widths cloning of tensors.
### Removed
- Three unused examples have been removed.
- **Breaking**: The beam validation layer has been removed, due to serious limitations. The `adjust_values` parameter is no longer available for initializing the dose engine.
- **Breaking**: The calibration of the dose engine no longer requires the beam template. Calibration can also be performed automatically during the initialization of the dose engine using the `auto_calibrate` argument.
- **Breaking**: `compute_dose_sequential` has been removed. Use `compute_dose(..., beam_chunk_size=N)` for memory-efficient, gradient-retaining dose computation (`beam_chunk_size=1` reproduces the old beam-by-beam behaviour).