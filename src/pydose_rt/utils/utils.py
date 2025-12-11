import numpy as np
import torch
import random
import copy
import numpy as np
from pydose_rt.data import Patient, MachineConfig
import torch
import os
import time
import pydicom
from pydose_rt.data import OptimizationConfig
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
    """
    Finds scalar c that minimizes MAE(||c*A - P||_1).
    A, P : numpy arrays of same shape (3D or any shape)
    mask : optional boolean array (same shape) to include only specific voxels
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
    dose_calc: torch.Tensor, shape (Z, Y, X)
    voxel_size: (dx, dy, dz) in mm
    xyz_mm: np.ndarray of shape (N, 3) with columns [X, Y, Z] in mm
    returns: torch.Tensor of shape (N,) with calculated dose at those points
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
    Writes MLC positions and MU values to a new RTPLAN DICOM file.
 
    Args:
        input_plan_path: Path to the original RTPLAN file to use as template
        output_plan_path: Path where the new RTPLAN file will be saved
        leafs: MLC leaf positions, shape (1, 2, num_control_points, num_leaves)
               where dim 1 is [higher, lower] banks
        jaws: Jaw positions, shape (1, 2, num_control_points)
              where dim 1 is [lower, higher]
        mus: MU values, shape (1, num_control_points)
        scaling: Scaling factor to convert normalized positions back to mm
        beam_number: Beam number to modify (default "1")
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
    first_structure = next(iter(structures.values()))
    bound_matrix = np.zeros_like(first_structure, dtype=np.float32)
    for structure_id, array in structures.items():
        if structure_id in bound:
            bound_matrix += array * bound[structure_id]
    return bound_matrix

def prune_patients(patient_list):
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
    so that their sum is 100.

    Args:
        constraints (dict): The constraints dictionary containing the 'weight' key.

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


def compute_valid_leaf_mask_minh(
    ptv_mask,  # [B, W, D, H] boolean PTV mask in voxel-indices
    config,
    leaf_width=1,
    voxel_sizes=(1, 1, 1),
    margin_mm: float = 0,
) -> torch.BoolTensor:
    """
    Returns a (B, number_of_beams, num_leafs) mask marking which leaves ever intercept the PTV for each batch.
    Assumes leaves move along the z-axis (H axis).
    """
    if ptv_mask.ndim == 3:
        ptv_mask = ptv_mask.unsqueeze(0)  # [1, W, D, H]
    B = ptv_mask.shape[0]
    number_of_beams = config.number_of_beams
    num_leafs = config.number_of_leaf_pairs

    (H, D, W) = config.ct_array_shape
    dx, dy, dz = voxel_sizes

    iso_x = ((W - 1) / 2) * dx
    iso_y = ((D - 1) / 2) * dy
    iso_z = ((H - 1) / 2) * dz

    isocenter = (iso_x, iso_y, iso_z)

    device = ptv_mask.device

    all_valid_leaf = torch.zeros(
        (B, number_of_beams, num_leafs), dtype=torch.uint8, device=device
    )

    for b in range(B):
        # 1) Gather PTV voxel centers (in mm)
        coords = torch.nonzero(
            ptv_mask[b], as_tuple=False
        ).float()  # [N, 3] indices: [w, d, h]
        if coords.shape[0] == 0:
            continue  # No PTV in this batch
        pts_mm = coords * torch.tensor([dx, dy, dz], device=device)  # [N,3] in mm

        # 2) Project all PTV points onto the z-axis (leaf direction)
        v_coord = pts_mm[:, 2]  # z in mm

        # 3) Compute leaf centers along z-axis (centered at isocenter z)
        z_leaf_centers = (
            torch.linspace(
                -(num_leafs / 2 - 0.5) * leaf_width,
                (num_leafs / 2 - 0.5) * leaf_width,
                num_leafs,
                device=device,
            )
            + iso_z
        )  # [num_leafs]

        # 4) For each beam, mark leaves whose center is within the PTV z-range
        z_min = v_coord.min().item()
        z_max = v_coord.max().item()

        valid_leaf_1d = (z_leaf_centers >= (z_min - margin_mm)) & (
            z_leaf_centers <= (z_max + margin_mm)
        )
        valid_leaf_per_beam = (
            valid_leaf_1d.unsqueeze(0).expand(number_of_beams, -1).clone()
        )
        all_valid_leaf[b] = valid_leaf_per_beam

    return all_valid_leaf  # shape: (B, number_of_beams, num_leafs)



def compute_valid_leaf_mask(
    dose_engine,
    dose_model,
    ct,  # Tensor of shape [1, Z, Y, X, 1]
    ptv_mask,  # Tensor of shape [1, Z, Y, X, 1], binary {0,1}
    n_cps: int,
    n_leafs: int,
    eps=1e-6,
    device=None,
):
    """
    Identifies MLC leaves that do not affect the PTV dose.

    Args:
        dose_model: A PyTorch module that computes dose from CT, MLC, and MU.
        ct: 5D CT image tensor [1, Z, Y, X, 1].
        ptv_mask: Binary mask of PTV region, same shape as ct.
        mlc: MLC positions [1, n_cps, n_leafs, 2] (left/right).
        mus: Monitor units per control point [1, n_cps].
        eps: Threshold for considering a gradient to be effectively zero.

    Returns:
        out_of_range: Boolean mask [1, n_leafs] — True if the leaf does not affect the PTV dose.
    """
    # Prevent gradients for dose_engine parameters
    for param in dose_model.parameters():
        param.requires_grad = False

    B = ct.shape[0]
    n_sides = 2  # MLC has 2 sides

    if device is None:
        device = ct.device if ct.device is not None else torch.device("cpu")

    # --- Create dummy mlc and mus ---
    mlc = torch.zeros((B, n_cps, n_leafs, n_sides), device=device)
    mlc[:, :, :, 1] = 1
    if dose_engine in ["attila", "matthias"]:
        mlc = mlc.permute(0, 3, 1, 2)
    elif dose_engine == "minh":  # does not work for now
        pass
    mus = torch.ones((B, n_cps), device=device)

    # Clone mlc and set requires_grad=True
    mlc = mlc.clone().requires_grad_(True)

    # Perform the forward+backward in an enabled-grad block even if outer context is no_grad()
    with torch.enable_grad():
        # Forward pass
        if dose_engine == "attila":
            dose_pred = dose_model(ct, mlc, mus)  # Predict 3D dose
        elif dose_engine == "matthias":
            dose_pred = dose_model(mlc, mus, jaw_positions=None, ct_image=ct * 1000)

        ptv_dose = dose_pred * ptv_mask  # Isolate PTV dose
        ptd = torch.sum(ptv_dose)  # Total dose in PTV

        # Backward pass
        ptd.backward()

        # Get gradients
        grads = mlc.grad  # [1, 2, n_cps, n_leafs]

        grads = grads.sum(axis=1)

        out_of_range = grads < eps

    valid_leaf = ~out_of_range
    return valid_leaf
