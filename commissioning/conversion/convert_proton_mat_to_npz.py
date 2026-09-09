"""Convert a pyRadPlan/matRad proton machine ``.mat`` into the pydosert ``.npz`` kernel table.

Usage:
    python convert_proton_mat_to_npz.py <input_mat> <output_npz>

Example:
    python convert_proton_mat_to_npz.py protons_Generic.mat src/pydosert/data/machine_presets/protons_generic.npz

The ``.mat`` is read with ``scipy.io`` -- this script is the *only* place in the
project that needs scipy for ion base data. The resulting ``.npz`` is a
padded-rectangular table that :class:`pydosert.physics.kernels.ion_kernel_table.IonKernelTable`
loads with numpy alone.

Layout of the produced archive (``E`` energies, ``Dmax`` = longest depth row,
``F`` = number of tabulated focus distances)::

    format_version  ()        int64     == 1
    energy_mev      (E,)      float64   sorted strictly increasing
    n_valid         (E,)      int32     samples actually used in row e
    depth_mm        (E, Dmax) float64   padded with +inf
    idd             (E, Dmax) float64   'Z', padded with 0
    sigma           (E, Dmax) float64   padded with the row's edge value
    sigma1          (E, Dmax) float64   padded with the row's edge value
    sigma2          (E, Dmax) float64   padded with the row's edge value
    weight          (E, Dmax) float64   padded with the row's edge value
    offset_mm       (E,)      float64
    focus_dist_mm   (E, F)    float64   from initFocus.dist
    focus_sigma_mm  (E, F)    float64   from initFocus.sigma
    sad_mm          ()        float64   from machine.meta.SAD
    bams_to_iso_mm  ()        float64   from machine.meta.BAMStoIsoDist

Everything is stored as float64: the engine runs at both float32 and float64 and
the reference implementation carries full float64 internally, so storing float32
would silently degrade the float64 path. Depth rows are *not* resampled onto a
common grid -- resampling to a uniform 0.2 mm grid costs 0.2 % of peak IDD
because the rows above ~230 MeV carry pyRadPlan's variable 1/2 mm spacing.
Padding keeps the round-trip bit-exact.

The converter fails loudly: any missing or malformed field raises rather than
degrading the table (a missing ``sigma1``/``sigma2``/``weight`` used to silently
turn a double-Gaussian machine into a single-Gaussian one).
"""
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

FORMAT_VERSION = 1

#: Per-energy fields that must be present on every ``machine.data`` entry.
REQUIRED_ENTRY_FIELDS = ("energy", "offset", "depths", "Z", "sigma", "sigma1", "sigma2", "weight", "initFocus")

#: Per-energy curves that must share the shape of ``depths``.
DEPTH_CURVE_FIELDS = ("Z", "sigma", "sigma1", "sigma2", "weight")

#: ``machine.meta`` fields that must be present.
REQUIRED_META_FIELDS = ("SAD", "BAMStoIsoDist")


def _require(obj: Any, name: str, context: str) -> Any:
    """Return attribute ``name`` of ``obj`` or raise with a helpful message."""
    value = getattr(obj, name, None)
    if value is None:
        available = getattr(obj, "_fieldnames", None)
        raise ValueError(
            f"{context} is missing the required field '{name}'"
            + (f" (available: {sorted(available)})" if available else "")
        )
    return value


def _as_1d(value: Any, name: str, context: str) -> np.ndarray:
    """Coerce a MATLAB field to a 1-D float64 array, raising on anything unusable."""
    array = np.asarray(value, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError(f"{context} has an empty '{name}'")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{context} has non-finite values in '{name}'")
    return array


def read_machine_mat(input_mat: str | Path) -> Dict[str, Any]:
    """Read a pyRadPlan/matRad machine ``.mat`` into plain numpy arrays.

    Args:
        input_mat: Path to the ``machine`` ``.mat`` file.

    Returns:
        Dictionary with the per-energy rows (as ragged lists) and the scalar meta data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any required field is missing or malformed.
    """
    import scipy.io as sio  # local import: scipy is only needed for conversion

    path = Path(input_mat)
    if not path.is_file():
        raise FileNotFoundError(f"Machine data not found: {path}")

    raw = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "machine" not in raw:
        found = sorted(k for k in raw if not k.startswith("__"))
        raise ValueError(f"{path} does not contain a 'machine' struct (found: {found})")
    machine = raw["machine"]

    meta = _require(machine, "meta", f"{path}: machine")
    for name in REQUIRED_META_FIELDS:
        _require(meta, name, f"{path}: machine.meta")
    sad_mm = float(meta.SAD)
    bams_to_iso_mm = float(meta.BAMStoIsoDist)

    entries = list(np.atleast_1d(_require(machine, "data", f"{path}: machine")))
    if not entries:
        raise ValueError(f"{path}: machine.data is empty")

    rows: List[Dict[str, Any]] = []
    for i, entry in enumerate(entries):
        context = f"{path}: machine.data({i + 1})"
        for name in REQUIRED_ENTRY_FIELDS:
            _require(entry, name, context)

        energy = float(entry.energy)
        context = f"{path}: machine.data({i + 1}) at {energy:g} MeV"
        offset = float(entry.offset)
        if not np.isfinite(offset):
            raise ValueError(f"{context} has a non-finite 'offset'")

        depths = _as_1d(entry.depths, "depths", context)
        order = np.argsort(depths, kind="stable")
        depths = depths[order]
        if np.any(np.diff(depths) <= 0.0):
            raise ValueError(f"{context} has duplicate or non-monotonic 'depths'; interpolation would be ill-defined")

        curves = {}
        for name in DEPTH_CURVE_FIELDS:
            values = _as_1d(getattr(entry, name), name, context)
            if values.shape != depths.shape:
                raise ValueError(
                    f"{context} has mismatched '{name}' shape {values.shape} vs 'depths' {depths.shape}"
                )
            curves[name] = values[order]

        init_focus = entry.initFocus
        focus_dist = _as_1d(_require(init_focus, "dist", f"{context}: initFocus"), "dist", context)
        focus_sigma = _as_1d(_require(init_focus, "sigma", f"{context}: initFocus"), "sigma", context)
        if focus_dist.shape != focus_sigma.shape:
            raise ValueError(
                f"{context} has mismatched initFocus dist {focus_dist.shape} / sigma {focus_sigma.shape}"
            )
        focus_order = np.argsort(focus_dist, kind="stable")
        focus_dist = focus_dist[focus_order]
        focus_sigma = focus_sigma[focus_order]
        if np.any(np.diff(focus_dist) <= 0.0):
            raise ValueError(f"{context} has duplicate or non-monotonic initFocus 'dist'")

        rows.append(
            {
                "energy": energy,
                "offset": offset,
                "depths": depths,
                "focus_dist": focus_dist,
                "focus_sigma": focus_sigma,
                **curves,
            }
        )

    rows.sort(key=lambda r: r["energy"])
    energies = np.array([r["energy"] for r in rows], dtype=np.float64)
    if np.any(np.diff(energies) <= 0.0):
        duplicates = energies[:-1][np.diff(energies) <= 0.0]
        raise ValueError(f"{path}: machine.data has duplicate energies: {duplicates.tolist()}")

    focus_sizes = {r["focus_dist"].size for r in rows}
    if len(focus_sizes) != 1:
        raise ValueError(
            f"{path}: initFocus tables have differing lengths {sorted(focus_sizes)}; "
            "the padded table format requires one focus grid size for the whole machine"
        )

    return {"rows": rows, "sad_mm": sad_mm, "bams_to_iso_mm": bams_to_iso_mm}


def machine_to_arrays(machine: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Pack the ragged per-energy rows into the padded-rectangular arrays.

    Args:
        machine: The dictionary returned by :func:`read_machine_mat`.

    Returns:
        Dictionary of numpy arrays ready for ``np.savez_compressed``.
    """
    rows = machine["rows"]
    n_energies = len(rows)
    n_valid = np.array([r["depths"].size for r in rows], dtype=np.int32)
    d_max = int(n_valid.max())
    n_focus = rows[0]["focus_dist"].size

    depth_mm = np.full((n_energies, d_max), np.inf, dtype=np.float64)
    idd = np.zeros((n_energies, d_max), dtype=np.float64)
    sigma = np.zeros((n_energies, d_max), dtype=np.float64)
    sigma1 = np.zeros((n_energies, d_max), dtype=np.float64)
    sigma2 = np.zeros((n_energies, d_max), dtype=np.float64)
    weight = np.zeros((n_energies, d_max), dtype=np.float64)
    focus_dist_mm = np.zeros((n_energies, n_focus), dtype=np.float64)
    focus_sigma_mm = np.zeros((n_energies, n_focus), dtype=np.float64)

    edge_padded = {"sigma": sigma, "sigma1": sigma1, "sigma2": sigma2, "weight": weight}
    for e, row in enumerate(rows):
        n = int(n_valid[e])
        depth_mm[e, :n] = row["depths"]
        idd[e, :n] = row["Z"]  # padded with 0: there is no dose past the tabulated range
        for name, target in edge_padded.items():
            target[e, :n] = row[name]
            target[e, n:] = row[name][-1]  # edge padding keeps a padded row monotone-safe
        focus_dist_mm[e] = row["focus_dist"]
        focus_sigma_mm[e] = row["focus_sigma"]

    return {
        "format_version": np.int64(FORMAT_VERSION),
        "energy_mev": np.array([r["energy"] for r in rows], dtype=np.float64),
        "n_valid": n_valid,
        "depth_mm": depth_mm,
        "idd": idd,
        "sigma": sigma,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "weight": weight,
        "offset_mm": np.array([r["offset"] for r in rows], dtype=np.float64),
        "focus_dist_mm": focus_dist_mm,
        "focus_sigma_mm": focus_sigma_mm,
        "sad_mm": np.float64(machine["sad_mm"]),
        "bams_to_iso_mm": np.float64(machine["bams_to_iso_mm"]),
    }


def convert_proton_mat_to_npz(input_mat: str | Path, output_npz: str | Path) -> None:
    """Convert a pyRadPlan/matRad proton machine ``.mat`` to the pydosert ``.npz`` table.

    Args:
        input_mat: Path to the input ``machine`` ``.mat`` file.
        output_npz: Path of the ``.npz`` archive to write.
    """
    machine = read_machine_mat(input_mat)
    arrays = machine_to_arrays(machine)

    output = Path(output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        np.savez_compressed(f, **arrays)

    n_energies, d_max = arrays["depth_mm"].shape
    size_mb = output.stat().st_size / 1024**2
    print(f"Converted {input_mat} -> {output}")
    print(f"  energies      : {n_energies} ({arrays['energy_mev'][0]:.4f} .. {arrays['energy_mev'][-1]:.4f} MeV)")
    print(f"  depth samples : {int(arrays['n_valid'].min())} .. {d_max} (padded to {d_max})")
    print(f"  focus grid    : {arrays['focus_dist_mm'].shape[1]} distances")
    print(f"  SAD / BAMS    : {float(arrays['sad_mm']):.1f} mm / {float(arrays['bams_to_iso_mm']):.1f} mm")
    print(f"  file size     : {size_mb:.2f} MiB")


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python convert_proton_mat_to_npz.py <input_mat> <output_npz>")
        print("Example: python convert_proton_mat_to_npz.py protons_Generic.mat protons_generic.npz")
        return 1

    input_mat = sys.argv[1]
    output_npz = sys.argv[2]

    try:
        convert_proton_mat_to_npz(input_mat, output_npz)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
