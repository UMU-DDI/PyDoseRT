import numpy as np
import torch
import random
import copy
import numpy as np
from pydosert.data import Patient, MachineConfig
import torch
import os
import time
import pydicom
from pydosert.data import OptimizationConfig
import os
from pathlib import Path
import pydicom
import os
from pathlib import Path
import pydicom

def find_patient_paths(patient_base: str | Path):
    """
    Given a patient directory, recursively search for:
      - All RTPLAN files
      - All RTDOSE files
      - RTSTRUCT (first found)
      - CT folder (directory whose files are CT dicoms; choose the one with most slices)

    Args:
        patient_base (str | Path): Patient directory to search recursively.

    Returns:
        ct_folder: Path
        rtplan_paths: list[Path]
        rtdose_paths: list[Path]
        rtstruct_path: Path

    Raises:
        FileNotFoundError if any of the above (except multiple plans/doses) cannot be found.
    """
    patient_base = Path(patient_base)

    rtplan_paths: list[Path] = []
    rtdose_paths: list[Path] = []
    rtstruct_path: Path | None = None

    ct_candidates: list[tuple[Path, int]] = []  # (folder, number_of_ct_files)

    for root, dirs, files in os.walk(patient_base):
        root_path = Path(root)

        # --- Find RTPLAN / RTDOSE / RTSTRUCT ---
        for fname in files:
            lname = fname.lower()
            fpath = root_path / fname

            # RTPLAN
            if ("rtplan" in lname) or lname.startswith("rp"):
                rtplan_paths.append(fpath)

            # RTDOSE
            if ("rtdose" in lname) or lname.startswith("rd"):
                rtdose_paths.append(fpath)

            # RTSTRUCT (take first found)
            if (("rtstruct" in lname) or lname.startswith("rs")) and rtstruct_path is None:
                rtstruct_path = fpath

        # --- Check for CT DICOM folders ---
        dicom_files = [
            root_path / f
            for f in files
            if f.lower().endswith(".dcm") or "." not in f
        ]

        if not dicom_files:
            continue

        try:
            ds = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True, force=True)
            modality = getattr(ds, "Modality", "").upper()
        except Exception:
            modality = ""

        if modality == "CT":
            ct_candidates.append((root_path, len(dicom_files)))

    # --- Select CT folder with most slices ---
    ct_folder: Path | None = None
    if ct_candidates:
        ct_folder = max(ct_candidates, key=lambda x: x[1])[0]

    # --- Sanity checks ---
    missing = []
    if ct_folder is None:
        missing.append("ct_folder")
    if not rtplan_paths:
        missing.append("rtplan_paths")
    if not rtdose_paths:
        missing.append("rtdose_paths")
    if rtstruct_path is None:
        missing.append("rtstruct_path")

    if missing:
        raise FileNotFoundError(
            f"Could not find {', '.join(missing)} under {patient_base}"
        )

    return ct_folder, rtplan_paths, rtdose_paths, rtstruct_path

def mae_optimal_scale(A: np.ndarray, P: np.ndarray, mask=None):
    """Find the scalar c minimizing the weighted MAE ||c*A - P||_1 (weighted median of P/A).

    Args:
        A (np.ndarray): Source array of any shape (e.g. [D, H, W]).
        P (np.ndarray): Target array, same shape as A.
        mask (np.ndarray | None): Optional boolean array, same shape as A, selecting
            the voxels to include.

    Returns:
        (float): Optimal scale factor c.
    """
    if mask is not None:
        A = A[mask]
        P = P[mask]

    valid = A > 0  # ignore zero or negative A if intensities are positive
    A = A[valid]
    P = P[valid]

    ratios = P / A
    weights = np.abs(A)

    # Sort ratios by value
    idx = np.argsort(ratios)
    sorted_ratios = ratios[idx]
    sorted_weights = weights[idx]

    # Cumulative weight
    cumulative = np.cumsum(sorted_weights)
    cutoff = cumulative[-1] / 2.0

    # Weighted median = first ratio where cumulative weight >= half total
    median_idx = np.searchsorted(cumulative, cutoff)
    c = sorted_ratios[median_idx]
    return c

def get_shapes(machine: MachineConfig, ct_shape: tuple[int, int, int] = None, number_of_beams: int = None, kernel_size: int = None, field_size: tuple[int, int] = None):
    """Build a dict of expected tensor shapes for the engine's plan/dose arrays.

    Args:
        machine (MachineConfig): Machine config providing number_of_leaf_pairs.
        ct_shape (tuple[int, int, int] | None): CT grid shape (D, H, W).
        number_of_beams (int | None): Number of beams G; returns None if not given.
        kernel_size (int | None): Convolution kernel size.
        field_size (tuple[int, int] | None): Fluence-map field size (rows, cols).

    Returns:
        (dict[str, tuple] | None): Named shapes, e.g. MLCs [1, G, N, 2], jaws [1, G, 2],
            MUs [1, G], fluence_volumes [G, D, H, W, 1], radiological_depths [G, H, 1],
            kernels [kernel_size, kernel_size, G, H], fluence_maps [G, field_size[0],
            field_size[1]]. None if number_of_beams is None.
    """
    shapes = dict()
    if number_of_beams is None:
        return
    
    shapes["MLCs"] = (1, number_of_beams, machine.number_of_leaf_pairs, 2)
    shapes["jaws"] = (1, number_of_beams, 2)
    shapes["MUs"] = (1, number_of_beams)
    if ct_shape is not None:
        shapes["fluence_volumes"] = (number_of_beams, ct_shape[0], ct_shape[1], ct_shape[2], 1)
        shapes["radiological_depths"] = (number_of_beams, ct_shape[1], 1)
        if kernel_size is not None:
            shapes["kernels"] = (kernel_size, kernel_size, number_of_beams, ct_shape[1])
    if field_size is not None:
        shapes["fluence_maps"] = (number_of_beams, field_size[0], field_size[1])

    return shapes

def sample_tensor_nearest(dose_calc, voxel_size, iso_center, xyz_mm):
    """Sample a dose volume at given physical points using nearest-voxel lookup.

    Args:
        dose_calc (torch.Tensor): Dose volume, [Z, Y, X].
        voxel_size (Sequence[float]): Voxel spacing (dx, dy, dz) in mm.
        iso_center (Sequence[float]): Isocenter voxel index (cx, cy, cz).
        xyz_mm (np.ndarray): Query points, [N, 3] with columns [X, Y, Z] in mm.

    Returns:
        (np.ndarray): Sampled dose at each point, [N].
    """
    Z, Y, X = dose_calc.shape
    dx, dy, dz = voxel_size

    # center index (isocenter at (0,0,0 mm))
    cx = iso_center[0]
    cy = iso_center[1]
    cz = iso_center[2]

    x_mm = xyz_mm[:, 0]
    y_mm = xyz_mm[:, 1]
    z_mm = xyz_mm[:, 2]

    # physical -> index space
    ix = cx + x_mm / dx
    iy = cy + y_mm / dy
    iz = cz + z_mm / dz

    # nearest voxel
    ix = torch.round(torch.from_numpy(ix)).long().clamp(0, X - 1)
    iy = torch.round(torch.from_numpy(iy)).long().clamp(0, Y - 1)
    iz = torch.round(torch.from_numpy(iz)).long().clamp(0, Z - 1)

    # sample
    return dose_calc[iz, iy, ix].cpu().detach().numpy()

def export_plan(treatment: OptimizationConfig, input_plan_path, output_plan_path, scaling=400, beam_number="1"):

    """Write MLC positions, jaw positions and MU values to a new RTPLAN DICOM file.

    Leaf/jaw/MU arrays are taken from ``treatment[0]`` (batch index 0).

    Args:
        treatment (OptimizationConfig): Indexable plan; ``treatment[0]`` provides
            leaf_positions [num_cp, 2, num_leaves], jaw_positions [num_cp, 2] and
            mus [num_cp].
        input_plan_path: Path to the original RTPLAN file used as a template.
        output_plan_path: Path where the new RTPLAN file is saved.
        scaling (float): Scaling factor to convert normalized positions back to mm.
        beam_number (str): Beam number to modify (default "1").
    """
    # Load the original plan
    ds = pydicom.dcmread(input_plan_path)
 
    # Remove batch dimension
    leafs = treatment[0].leaf_positions  # (2, num_cp, num_leaves)
    jaws = treatment[0].jaw_positions    # (2, num_cp)
    mus = treatment[0].mus      # (num_cp,)

 
    num_cp = len(mus)
 
    cumulative_mus = np.cumsum(mus)
    cumulative_mus -= cumulative_mus[0]
    cumulative_mus /= np.sum(mus)
    total_mu = np.sum(mus)
    cumulative_weights = cumulative_mus / cumulative_mus.max()
    

    # Find the beam to modify
    beam_found = False
    for beam in ds.BeamSequence:
        if str(beam.BeamNumber) == beam_number:
            beam_found = True
 
            # Update beam meterset in FractionGroupSequence
            for ref_seq in ds.FractionGroupSequence[0].ReferencedBeamSequence:
                if str(ref_seq.ReferencedBeamNumber) == beam_number:
                    ref_seq.BeamMeterset = float(total_mu)
 
            # Update control points
            num_existing_cp = len(beam.ControlPointSequence)
            expected_cp = num_cp
 
            if num_existing_cp != expected_cp:
                print(f"Warning: Expected {expected_cp} control points but found {num_existing_cp}")
 
            for index, cps in enumerate(beam.ControlPointSequence):
                if index >= expected_cp:
                    break
 
                # Update cumulative meterset weight
                if index == 0:
                    cps.CumulativeMetersetWeight = 0.0
                else:
                    cps.CumulativeMetersetWeight = float(cumulative_weights[index])
 
                # Update MLC and jaw positions
                if "BeamLimitingDevicePositionSequence" in cps:
                    for sequence in cps.BeamLimitingDevicePositionSequence:
                        if sequence.RTBeamLimitingDeviceType == "MLCX":
                            # Combine higher and lower banks
                            mlc_positions = np.concatenate([
                                leafs[index, :, 0],
                                leafs[index, :, 1]
                            ])
                            mlc_positions = [float(x) for x in mlc_positions]
                            sequence.LeafJawPositions = mlc_positions
 
                        elif sequence.RTBeamLimitingDeviceType == "ASYMX":
                            jaw_positions = [
                                float(jaws[index, 0]),
                                float(jaws[index, 1])
                            ]
                            sequence.LeafJawPositions = jaw_positions
 
            break
 
    if not beam_found:
        raise ValueError(f"Beam number {beam_number} not found in plan")
 
    # Save the modified plan
    ds.save_as(output_plan_path)
    print(f"Plan saved to {output_plan_path}")

def get_model_input(patient: Patient, machine: MachineConfig):
    """Stack the CT and per-structure bound/weight matrices into a model input.

    Args:
        patient (Patient): Provides ct_array [D, H, W] and structures masks [D, H, W].
        machine (MachineConfig): Provides per-structure lower/higher bound (Gy and %)
            and weight maps.

    Returns:
        (np.ndarray): Stacked input, [6, D, H, W]; channels are scaled CT, lower/higher
            bound Gy, lower/higher bound percent, and weights.
    """
    structures = patient.structures
    lower_bound_gys = create_bound_weight_matrix(structures, machine.lower_bound_gys)
    higher_bound_gys = create_bound_weight_matrix(structures, machine.higher_bound_gys)
    lower_bound_percents = create_bound_weight_matrix(structures, machine.lower_bound_percents)
    higher_bound_percents = create_bound_weight_matrix(structures, machine.higher_bound_percents)
    weights = create_bound_weight_matrix(structures, machine.weights)
    return np.stack([patient.ct_array / 1000,
                     lower_bound_gys,
                     higher_bound_gys,
                     lower_bound_percents,
                     higher_bound_percents,
                     weights])

def create_bound_weight_matrix(structures, bound):
    """Paint per-structure scalar bounds/weights into a single voxel matrix.

    Args:
        structures (dict[str, np.ndarray]): Structure masks keyed by id, each [D, H, W].
        bound (dict[str, float]): Per-structure scalar value to assign within its mask.

    Returns:
        (np.ndarray): Accumulated value matrix, [D, H, W].
    """
    first_structure = next(iter(structures.values()))
    bound_matrix = np.zeros_like(first_structure, dtype=np.float32)
    for structure_id, array in structures.items():
        if structure_id in bound:
            bound_matrix += array * bound[structure_id]
    return bound_matrix

def prune_patients(patient_list):
    """Filter patient directories to those containing CT.npy and StructureSet.npy.

    Args:
        patient_list (Iterable[str]): Candidate patient directory paths.

    Returns:
        (list[str]): Directories that exist and contain both required files.
    """
    pruned_list = []
    for patient in patient_list:
        if not os.path.isdir(patient):
            continue

        if (("CT.npy" in os.listdir(patient)) and ("StructureSet.npy" in os.listdir(patient))):
            pruned_list.append(patient)
    return pruned_list
     
def normalize_weights(constraints, sum_value=100):  #
    """
    Normalizes the values in the 'weight' sub-dictionary of the constraints
    so that their sum equals sum_value.

    Args:
        constraints (dict): The constraints dictionary containing the 'weight' key.
        sum_value (float): Target sum for the normalized weights (default 100).

    Returns:
        dict: The modified constraints dictionary with normalized weights.
    """
    weights = constraints.get("weight")
    if not weights:
        return constraints  # Return original if 'weight' key is missing

    total_weight = sum(weights.values())
    if total_weight == 0:
        total_weight = 1e-6

    normalized_weights = {}
    for roi, weight in weights.items():
        normalized_weights[roi] = (weight / total_weight) * sum_value

    constraints["weight"] = normalized_weights
    return constraints

def get_initial_weights():
    """Return the default (currently all-zero) loss-term weight dict.

    Returns:
        (dict[str, float]): Initial scalar weight per machine-regularization loss term.
    """
    min_int_range = -3
    max_int_range = 2
    weights = {
        "l2_loss_oars_and_background": 0.0, # 10**np.random.randint(-3, 1), # 0.01,
        "mu_reg_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "mu_complexity_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "leaf_reg_loss": 0.0,# 10**np.random.randint(-5, 2), # 10**np.random.randint(min_int_range, max_int_range),
        "leaf_complexity_loss": 0.0,# 10**np.random.randint(-5, 2), # 10**np.random.randint(-2, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "jaw_reg_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "jaw_complexity_loss": 0.0, # 10**np.random.randint(-3, 5), # 10**np.random.randint(min_int_range, max_int_range),
    }
    
    return weights

