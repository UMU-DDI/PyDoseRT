# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
PyDoseRT uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
This changelog was introduced after releasing version 1.3.0.

## [Unreleased]

### Added
- Five new example files are now available through the repository. They are suitable for running in a T4 google colab environment.
### Changed
- Machine/Optimization configurations are now built-in to the package, and easier to access. To get a list of all available presets, run `list_machine_presets()` or `list_optimization_presets()`. All related tests have been updated. 
- Changed commissioning pipeline to use json files
### Fixed
- The correct email adresses are now in the pyproject file.
### Removed
- Three unused examples have been removed.
- **Breaking**: The beam validation layer has been removed, due to serious limitations. The `adjust_values` parameter is no longer available for initializing the dose engine.
- **Breaking**: The calibration of the dose engine no longer requires the beam template. Calibration can also be performed automatically during the initialization of the dose engine using the `auto_calibrate` argument.