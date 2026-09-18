import numpy as np
import pytest
import torch

from pydosert.physics.dose_mask import (
    patient_dose_mask,
)

AIR = 0.0012
TISSUE = 1.0


def phantom_with_cavity(shape=(40, 40, 40), cavity=(slice(18, 22), slice(18, 22), slice(18, 22))):
    """A tissue block surrounded by air, with an enclosed air cavity inside it."""
    volume = torch.full(shape, AIR)
    volume[6:-6, 6:-6, 6:-6] = TISSUE
    volume[cavity] = AIR
    return volume



def test_internal_cavity_is_scored_and_external_air_is_not():
    volume = phantom_with_cavity()
    mask = patient_dose_mask(volume)

    assert mask[20, 20, 20], "enclosed air cavity must stay scored"
    assert not mask[0, 0, 0], "air outside the patient must not be scored"
    assert mask[10, 20, 20], "tissue must be scored"


def test_cavity_touching_the_border_is_still_scored():
    """A trachea cut by the first slice touches the border but is not external air."""
    volume = torch.full((40, 40, 40), AIR)
    volume[6:-6, 6:-6, 6:-6] = TISSUE
    volume[0:25, 18:22, 18:22] = AIR  # a lumen open at the h = 0 face

    mask = patient_dose_mask(volume)
    assert mask[20, 20, 20], "the lumen inside the tissue must stay scored"
    assert not mask[0, 0, 0]


def test_falls_back_to_the_threshold_when_the_patient_fills_the_volume():
    volume = torch.full((30, 30, 30), TISSUE)
    volume[10:14, 10:14, 10:14] = AIR  # only air is an internal pocket, no external air

    mask = patient_dose_mask(volume)
    assert mask.all(), "with no external air every voxel is scored"


def test_all_air_volume_scores_nothing():
    """No patient in the volume: every voxel is external air."""
    volume = torch.full((8, 8, 8), AIR)
    assert not patient_dose_mask(volume).any()


def test_solid_phantom_is_all_scored():
    """A water phantom filling the grid has no air at all."""
    volume = torch.full((8, 8, 8), TISSUE)
    assert patient_dose_mask(volume).all()


def test_accepts_a_leading_singleton_axis():
    volume = phantom_with_cavity()
    mask_3d = patient_dose_mask(volume)
    mask_4d = patient_dose_mask(volume.unsqueeze(0))

    assert mask_4d.shape == (1,) + tuple(volume.shape)
    assert torch.equal(mask_4d[0], mask_3d)


def test_rejects_other_shapes():
    with pytest.raises(ValueError, match=r"\[H, D, W\]"):
        patient_dose_mask(torch.ones(4, 4))
    with pytest.raises(ValueError, match=r"\[H, D, W\]"):
        patient_dose_mask(torch.ones(2, 4, 4, 4))


def test_mask_is_boolean_and_keeps_the_input_shape():
    volume = phantom_with_cavity()
    mask = patient_dose_mask(volume)
    assert mask.dtype == torch.bool
    assert mask.shape == volume.shape


def test_scored_fraction_matches_the_body_plus_cavity():
    """The mask is exactly the filled body outline for this phantom."""
    volume = phantom_with_cavity()
    mask = patient_dose_mask(volume).numpy()

    body = np.zeros(volume.shape, dtype=bool)
    body[6:-6, 6:-6, 6:-6] = True
    assert np.array_equal(mask, body)
