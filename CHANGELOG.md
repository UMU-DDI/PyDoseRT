# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
PyDoseRT uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
This changelog was introduced after releasing version 1.3.0.

## [Unreleased]

### Added
### Changed
### Fixed
### Removed

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