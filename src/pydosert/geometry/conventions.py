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
    convention, which is what the DICOM loaders subtract. Voxel i along an axis
    is therefore at i * resolution mm.

    KNOWN DISCREPANCY (open, do not "fix" piecemeal)
        The three layers that convert the isocentre from mm to voxels do not
        agree on a half voxel:

            FluenceVolumeLayer        iso / resolution
            build_rotation_grids      iso / resolution
            radiological-depth rays   iso / resolution + 0.5

        For iso 64 mm at 2 mm spacing that is voxel 32.0 against 32.5, i.e. the
        depth ray runs half a voxel (1 mm) to the side of the beam axis whose
        radiological depth it measures. Errors of this kind surface in the
        entrance/build-up region, where they are easy to mistake for a physics
        problem.

        The +0.5 is deliberate and dates from the 1.4.0 "align the rad-depth ray
        rotation center with the align_corners=False convention" change. Which
        of the two is right has NOT been settled: removing it moves a 4-field
        box centroid measurably closer to the isocentre but changes dose, so it
        needs validating on a real cohort before anything moves.
        test_conventions.py pins the current numbers so the difference stays
        visible and cannot drift further.

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
