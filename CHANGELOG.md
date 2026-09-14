# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
PyDoseRT uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
This changelog was introduced after releasing version 1.3.0.

## [Unreleased]

### Added
- New composable objective primitives in `pydosert.objectives.losses`: `upper_penalty`, `lower_penalty`, `mean_upper_penalty` and `squared_penalty` (one- and two-sided squared-hinge penalties on voxel doses), plus `geud` for the generalized equivalent uniform dose. `geud` is a reduction rather than a loss, so it composes with the penalties to build EUD objectives (e.g. `upper_penalty(geud(x, 2.5), c)` is a max-EUD constraint).
- New `pydosert.objectives.regularizers` module holding the deliverability regularizers `mus_loss`, `leafs_loss` and `jaws_loss`, each returning a (rate, complexity) pair derived from the machine limits.
- `condition_aperture_pair` and `condition_beam_params` in `pydosert.data.beam` give direct optimization and deep-learning workflows one shared differentiable map from unconstrained variables to physical ordered leaf/jaw pairs and positive MUs. The MU scale is normalized by the control-point count so the raw variables stay ~O(1) regardless of the number of control points.
- New `MultilatticeEngine`, a sibling of `DoseEngine` on the same `PhotonBaseEngine` base. Where `DoseEngine` takes its kernel depth from a single central-axis ray, the multilattice engine partitions the beam's-eye-view fluence into up to `L x L` equal-fluence tiles, convolves each with the kernels of a ray through its own fluence-weighted centroid, and corrects the residual per-voxel depth difference with the depth dose of each tile's own field, `D(d_voxel; R) / D(d_ray; R)`, taken from the same pencil-beam model (`R` is the tile's equivalent radius). It has no free parameter, follows TPR 20/10, is correct through buildup, and adapts to the lattice; a constant attenuation is still available via `mu_eff=`. This fixes the off-axis depth-dose error the central-ray approximation makes in laterally heterogeneous anatomy (lung, most visibly). Configured with `lattice_size`, `cf_clamp`, `tile_chunk` and `ray_supersample`; `lattice_size` trades accuracy against the number of tiles, though each tile's convolution is cropped to its own support plus the kernel halo -- exactly, since the masked source is zero outside it -- so the total convolution work stays close to a single full-field pass.
- New `TermaScalingLayer`, a differentiable TERMA source scaling following Laakkonen, Fan and Harju (Med Phys. 2023): the water fluence is scaled locally by effective field size and smoothed relative density before pencil-beam convolution. It is not enabled anywhere by default; `c1` and `c2_per_mm` default to the values used in the DoseRAD 2026 experiments (0.9 and 0.28 /mm) and can be refitted per machine with `learnable=True`. Plugs into `MultilatticeEngine` via `source_scale_layer=`.
- New plotting functions: `plot_mu_polar`, `plot_fluence_and_mu`, `plot_dvh`, `plot_profiles` and `plot_kernel`, with the reusable `compute_fluence_maps` and `compute_dvh_curves` helpers behind them.

### Changed
- **Breaking**: the plotting functions have been renamed to a consistent `plot_*` scheme: `print_paper_plot` is now `plot_overview`, `print_comparison_plot` is now `plot_comparison`, and `make_animation` is now `plot_animation`.
- `utils.py` docstrings have been converted to the Google style used elsewhere in the package.
- The example notebooks have been updated to the new objective, conditioning and plotting APIs.

### Fixed
- `Patient.device` and `Patient.dtype` read the density-image tensor directly; they previously referenced a non-existent `.attenuation.data` attribute and raised.
- `load_structures` returns early when no structure set is given, instead of nesting the whole body in a conditional.
- `result_validation` computes the fraction-scaled predicted and reference dose unconditionally; the gamma path referenced them while they were only assigned inside the clinical-criteria branch, so requesting gamma without clinical criteria raised `UnboundLocalError`.

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