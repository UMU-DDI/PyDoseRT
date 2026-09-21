"""The one place where PyDoseRT's axis and coordinate conventions are defined.

Every layer, engine and example follows what is written here. If code and this
module disagree, the code is wrong -- :mod:`tests.unittests.test_conventions`
pins the statements below against the layers that implement them.

Import :data:`CONVENTIONS` to print the summary, e.g. at the top of a notebook.
"""

from __future__ import annotations

__all__ = ["CONVENTIONS"]

#: Human-readable summary of the conventions, suitable for printing in a notebook.
CONVENTIONS = """\
PyDoseRT axis conventions
=========================

Volume layout
    Dose grids, CT and density volumes are (H, D, W); batched they are
    [B, H, D, W], and per-beam intermediates are [B, G, H, D, W] with G the
    control-point axis.

        H  height  patient superior-inferior (the axis the gantry does NOT move in)
        D  depth   the axis the beam runs along at gantry 0
        W  width   the axis the beam runs along at gantry 90

    dose_grid_shape and dose_grid_spacing are both given in this order:
    dose_grid_spacing = (res_H, res_D, res_W) in mm.

Isocentre
    iso_center is (h, d, w) in MILLIMETRES, in the same axis order as the grid,
    measured from the CENTRE of voxel [0, 0, 0] -- SimpleITK's origin
    convention, which is what the DICOM loaders subtract. So voxel i along an
    axis is at i * resolution mm, and the isocentre sits at the fractional
    voxel index

        iso_voxel = iso_center / dose_grid_spacing        (no half-voxel shift)

    Every layer that needs the isocentre uses exactly this: the fluence
    projection, the radiological-depth ray caster and the beam rotation. A
    half-voxel disagreement between them offsets the depth ray from the beam
    axis it belongs to, which shows up as an entrance/build-up error.

Gantry rotation
    The gantry rotates in the (D, W) plane; H is untouched. At gantry angle
    theta the beam travels along

        (cos theta, -sin theta)   in (D, W)

    so at gantry 0 the beam enters at low D and runs towards +D, and at
    gantry 90 it enters at high W and runs towards -W. Angles are stored in
    RADIANS on BeamSequence (gantry_angles); the create() helpers take degrees.

Beam-frame quantities
    Leaf positions are [G, N, 2] ordered (left, right) and jaw positions are
    [G, 2] ordered (lower, upper), both in mm at the isocentre plane. Leaves
    move along W and jaws along H when the collimator angle is 0.

Units
    Distances mm, angles radians (degrees only in the *_deg helpers and
    properties), dose Gy, MU in machine monitor units.
"""


def print_conventions() -> None:
    """Print :data:`CONVENTIONS`. Convenience for notebooks."""
    print(CONVENTIONS)
