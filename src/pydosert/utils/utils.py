import numpy as np
import torch
from pydosert.data import Patient, MachineConfig
import os
import pydicom
from pydosert.data import OptimizationConfig
from pathlib import Path

def find_patient_paths(patient_base: str | Path):
    """
    Recursively locate the DICOM files needed to load a patient.

    Walks ``patient_base`` and collects every RTPLAN and RTDOSE file, the first
    RTSTRUCT, and the CT folder (the DICOM directory holding the most CT slices).

    Args:
        patient_base (str | Path): Root directory to search.

    Returns:
        tuple: ``(ct_folder, rtplan_paths, rtdose_paths, rtstruct_path)`` where
            ``ct_folder`` and ``rtstruct_path`` are ``Path`` and the plan and dose
            entries are ``list[Path]``.

    Raises:
        FileNotFoundError: If the CT folder, any RTPLAN, any RTDOSE, or the
            RTSTRUCT cannot be found under ``patient_base``.
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
    """
    Find the scalar ``c`` minimising the MAE between ``c * A`` and ``P``.

    Voxels where ``A <= 0`` are ignored. The optimum is the weighted median of the
    per-voxel ratios ``P / A``, with weights ``|A|``.

    Args:
        A (np.ndarray): Reference array (e.g. calculated dose).
        P (np.ndarray): Target array (e.g. measured dose), same shape as ``A``.
        mask (np.ndarray, optional): Boolean mask selecting voxels to include.

    Returns:
        float: Optimal scaling factor ``c``.
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
    """
    Return the expected tensor shapes for the dose-calculation pipeline.

    Only the shapes derivable from the given arguments are included.

    Args:
        machine (MachineConfig): Machine configuration (for the leaf-pair count).
        ct_shape (tuple[int, int, int], optional): Dose grid shape (D, H, W).
        number_of_beams (int, optional): Number of beams / control points.
        kernel_size (int, optional): Pencil-beam kernel size.
        field_size (tuple[int, int], optional): Fluence-map size (H, W) in pixels.

    Returns:
        dict: Mapping of tensor name to its shape, or ``None`` when
            ``number_of_beams`` is not given.
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
    """
    Nearest-neighbour sample of a dose grid at physical (mm) points.

    Points are given in mm relative to the isocenter and converted to voxel
    indices using ``voxel_size``.

    Args:
        dose_calc (torch.Tensor): Dose grid [Z, Y, X].
        voxel_size (tuple[float, float, float]): Voxel spacing (dx, dy, dz) in mm.
        iso_center (tuple[float, float, float]): Isocenter voxel index (x, y, z).
        xyz_mm (np.ndarray): Query points [N, 3] as columns [X, Y, Z] in mm.

    Returns:
        np.ndarray: Sampled dose [N] at the nearest voxel to each point.
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

    """
    Write MLC positions and MU values into a copy of an RTPLAN DICOM file.

    Uses ``input_plan_path`` as a template, overwrites the control points of the
    selected beam with the plan carried by ``treatment``, and saves the result.

    Args:
        treatment (OptimizationConfig): Plan whose first beam supplies the leaf,
            jaw and MU values to write out.
        input_plan_path: Path to the original RTPLAN file used as a template.
        output_plan_path: Path where the modified RTPLAN is saved.
        scaling (int): Position scaling factor.
        beam_number (str): Beam number to modify.

    Raises:
        ValueError: If ``beam_number`` is not present in the plan.
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
    """
    Build the multi-channel model input for a patient.

    Stacks the (scaled) CT with per-voxel bound and weight maps derived from the
    machine's optimisation constraints.

    Args:
        patient (Patient): Patient providing the CT and structure masks.
        machine (MachineConfig): Source of the per-structure bounds and weights.

    Returns:
        np.ndarray: Stacked input channels [C, D, H, W].
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
    """
    Paint per-structure scalar values into a single voxel grid.

    Each structure mask is multiplied by its value in ``bound`` and the results
    are summed into one volume.

    Args:
        structures (dict): Mapping of structure name to binary mask array.
        bound (dict): Mapping of structure name to the scalar to assign.

    Returns:
        np.ndarray: Accumulated value grid with the shape of a structure mask.
    """
    first_structure = next(iter(structures.values()))
    bound_matrix = np.zeros_like(first_structure, dtype=np.float32)
    for structure_id, array in structures.items():
        if structure_id in bound:
            bound_matrix += array * bound[structure_id]
    return bound_matrix

def prune_patients(patient_list):
    """
    Keep only patient directories that contain the required preprocessed files.

    Args:
        patient_list (list): Candidate patient directory paths.

    Returns:
        list: Directories that exist and contain both ``CT.npy`` and
            ``StructureSet.npy``.
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
    Normalise the per-ROI values in ``constraints["weight"]`` to a fixed sum.

    Args:
        constraints (dict): Constraints dictionary containing a ``"weight"`` sub-dict.
        sum_value (float): Target sum for the normalised weights.

    Returns:
        dict: The constraints dictionary with ``"weight"`` normalised (returned
            unchanged if no ``"weight"`` key is present).
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
