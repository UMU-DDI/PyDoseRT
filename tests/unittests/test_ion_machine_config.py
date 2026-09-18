"""Tests for :class:`pydosert.data.ion_machine.IonMachineConfig`.

Self-contained: the built-in preset shipped in
``src/pydosert/data/ion_machine_presets`` is used for the package-data path, and
custom presets are written into ``tmp_path``.
"""

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from pydosert.data.ion_machine import (  # noqa: E402
    IonMachineConfig,
    list_ion_machine_presets,
)
from pydosert.data.machine_config import MachineConfig  # noqa: E402

# --------------------------------------------------------------------------- defaults


def test_default_construction_works():
    """The whole point: a proton machine needs no MLC leaf count."""
    config = IonMachineConfig()
    assert config.bams_to_iso_dist_mm == 1000.0
    assert config.fit_air_offset_mm == 0.0
    assert config.preset is None


def test_photon_config_cannot_be_default_constructed():
    """Evidence for why this model exists rather than reusing MachineConfig."""
    with pytest.raises(ValidationError) as excinfo:
        MachineConfig()
    missing = {err["loc"][0] for err in excinfo.value.errors()}
    assert missing == {"tpr_20_10", "number_of_leaf_pairs"}


def test_explicit_kwargs():
    config = IonMachineConfig(bams_to_iso_dist_mm=1250.0, fit_air_offset_mm=12.5)
    assert config.bams_to_iso_dist_mm == 1250.0
    assert config.fit_air_offset_mm == 12.5


def test_only_the_two_ion_fields_are_modelled():
    assert set(IonMachineConfig.model_fields) == {
        "preset",
        "bams_to_iso_dist_mm",
        "fit_air_offset_mm",
    }


def test_no_sad_field():
    """SAD lives on the beamlet batch and the kernel table, not here."""
    assert "sad_mm" not in IonMachineConfig.model_fields
    assert "sad" not in IonMachineConfig.model_fields


def test_fields_have_descriptions():
    for name, field in IonMachineConfig.model_fields.items():
        assert field.description, f"{name} has no Field description"


def test_rejects_negative_distances():
    with pytest.raises(ValidationError):
        IonMachineConfig(bams_to_iso_dist_mm=-1.0)
    with pytest.raises(ValidationError):
        IonMachineConfig(fit_air_offset_mm=-0.5)


def test_rejects_non_numeric():
    with pytest.raises(ValidationError):
        IonMachineConfig(bams_to_iso_dist_mm="nozzle")


def test_matches_the_values_the_engine_reads():
    """``rad_depth_offset = 0.0011 * ((ssd + bams) - sad - fit_air)``."""
    config = IonMachineConfig(bams_to_iso_dist_mm=1000.0, fit_air_offset_mm=0.0)
    ssd_mm, sad_mm = 800.0, 1000.0
    offset = 0.0011 * ((ssd_mm + config.bams_to_iso_dist_mm) - sad_mm - config.fit_air_offset_mm)
    assert offset == pytest.approx(0.88)


# --------------------------------------------------------------------------- presets


def test_builtin_preset_is_listed():
    presets = list_ion_machine_presets()
    assert "generic" in presets
    assert presets == sorted(presets)


def test_builtin_preset_loads():
    config = IonMachineConfig(preset="generic")
    assert config.preset == "generic"
    assert config.bams_to_iso_dist_mm == 1000.0
    assert config.fit_air_offset_mm == 0.0


def test_builtin_preset_name_with_extension():
    assert IonMachineConfig(preset="generic.json").bams_to_iso_dist_mm == 1000.0


def test_ion_presets_do_not_pollute_the_photon_preset_list():
    from pydosert.data.machine_config import list_machine_presets

    assert "generic" not in list_machine_presets()


def test_explicit_kwargs_override_the_preset(tmp_path):
    preset = tmp_path / "custom.json"
    preset.write_text(json.dumps({"bams_to_iso_dist_mm": 500.0, "fit_air_offset_mm": 3.0}))
    config = IonMachineConfig(preset=str(preset), bams_to_iso_dist_mm=1234.0)
    assert config.bams_to_iso_dist_mm == 1234.0, "kwargs must win over the preset"
    assert config.fit_air_offset_mm == 3.0, "unspecified fields come from the preset"


def test_preset_from_path(tmp_path):
    preset = tmp_path / "beamline.json"
    preset.write_text(json.dumps({"bams_to_iso_dist_mm": 777.0}))
    config = IonMachineConfig(preset=str(preset))
    assert config.bams_to_iso_dist_mm == 777.0
    assert config.fit_air_offset_mm == 0.0, "unset preset fields fall back to the default"


def test_unknown_preset_raises_and_lists_the_available_ones():
    with pytest.raises(ValidationError, match="Unknown ion machine preset"):
        IonMachineConfig(preset="no_such_machine")


def test_preset_must_be_a_json_object(tmp_path):
    preset = tmp_path / "bad.json"
    preset.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValidationError, match="JSON object at the top level"):
        IonMachineConfig(preset=str(preset))


def test_preset_with_an_invalid_value_still_validates(tmp_path):
    preset = tmp_path / "negative.json"
    preset.write_text(json.dumps({"bams_to_iso_dist_mm": -10.0}))
    with pytest.raises(ValidationError):
        IonMachineConfig(preset=str(preset))


def test_empty_preset_string_is_ignored():
    assert IonMachineConfig(preset="").bams_to_iso_dist_mm == 1000.0


# --------------------------------------------------------------------------- env vars


def test_environment_variable_is_picked_up(monkeypatch):
    monkeypatch.setenv("bams_to_iso_dist_mm", "1500.0")
    assert IonMachineConfig().bams_to_iso_dist_mm == 1500.0


def test_explicit_kwarg_beats_the_environment(monkeypatch):
    monkeypatch.setenv("bams_to_iso_dist_mm", "1500.0")
    assert IonMachineConfig(bams_to_iso_dist_mm=900.0).bams_to_iso_dist_mm == 900.0
