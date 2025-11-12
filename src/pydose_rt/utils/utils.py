import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.ndimage as ndi
import random
import copy
import numpy as np
from pydose_rt.data import PatientData, MachineConfig
import torch
import os
import time
import pydicom

def export_plan(config, input_plan_path, output_plan_path, scaling=400, beam_number="1"):

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
    leafs = config.patient.plan_mlcs[0]  # (2, num_cp, num_leaves)
    jaws = config.patient.plan_jaws[0]    # (2, num_cp)
    mus = config.patient.plan_mus[0]      # (num_cp,)
 
    # Reverse the scaling transformation
    leafs = leafs * scaling - (scaling / 2)
    jaws = jaws * scaling - (scaling / 2)
 
    mus = np.hstack([0.0, mus])  # (2, num_cp)
    # Split leafs back into higher and lower banks
    beam_higher = np.vstack([leafs[1][1:2, :], leafs[1]])  # (num_cp, num_leaves)
    beam_lower = np.vstack([leafs[0][1:2, :], leafs[0]])   # (num_cp, num_leaves)
    jaw_lower = np.hstack([jaws[0][1:2], jaws[0]])    # (num_cp,)
    jaw_higher = np.hstack([jaws[1][1:2], jaws[1]])    # (num_cp,)
 
    num_cp = len(mus)
    multi_cp = num_cp > 1
 
    # Convert differential MUs back to cumulative if multi-control point
    if multi_cp:
        cumulative_mus = np.cumsum(mus) / np.sum(mus)
        total_mu = np.sum(mus)
        cumulative_weights = cumulative_mus / cumulative_mus.max()
 
        # For multi-CP, we need to convert averaged positions back to actual control point positions        # This reverses the averaging done in fetch_plan_data

        actual_beam_higher = np.zeros((num_cp, beam_higher.shape[1]))
        actual_beam_lower = np.zeros((num_cp, beam_lower.shape[1]))
        actual_jaw_lower = np.zeros(num_cp)
        actual_jaw_higher = np.zeros(num_cp)
 
        # Extrapolate first control point backwards from first two midpoints
        # If m[0] and m[1] are midpoints, assume linear progression:
        # p[0] = 2*m[0] - m[1] ensures proper reconstruction
        actual_beam_higher[0] = beam_higher[0]
        actual_beam_lower[0] = beam_lower[0]
        actual_jaw_lower[0] = jaw_lower[0]
        actual_jaw_higher[0] = jaw_higher[0]
 
        # Reconstruct subsequent control points using midpoint formula: p[i+1] = 2*m[i] - p[i]
        for i in range(num_cp):
            actual_beam_higher[i] = beam_higher[i]
            actual_beam_lower[i] = beam_lower[i]
            actual_jaw_lower[i] = jaw_lower[i]
            actual_jaw_higher[i] = jaw_higher[i]
    else:
        total_mu = mus[0]
        cumulative_weights = [1.0]
        actual_beam_higher = beam_higher
        actual_beam_lower = beam_lower
        actual_jaw_lower = jaw_lower
        actual_jaw_higher = jaw_higher
 
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
                if multi_cp:
                    if index == 0:
                        cps.CumulativeMetersetWeight = 0.0
                    else:
                        cps.CumulativeMetersetWeight = float(cumulative_weights[index])
                else:
                    if hasattr(cps, "CumulativeMetersetWeight"):
                        cps.CumulativeMetersetWeight = 1.0
 
                # Update MLC and jaw positions
                if "BeamLimitingDevicePositionSequence" in cps:
                    for sequence in cps.BeamLimitingDevicePositionSequence:
                        if sequence.RTBeamLimitingDeviceType == "MLCX":
                            # Combine higher and lower banks
                            mlc_positions = np.concatenate([
                                actual_beam_lower[index],
                                actual_beam_higher[index]
                            ])
                            mlc_positions = [float(x) for x in mlc_positions]
                            sequence.LeafJawPositions = mlc_positions
 
                        elif sequence.RTBeamLimitingDeviceType == "ASYMX":
                            jaw_positions = [
                                float(actual_jaw_lower[index]),
                                float(actual_jaw_higher[index])
                            ]
                            sequence.LeafJawPositions = jaw_positions
 
            break
 
    if not beam_found:
        raise ValueError(f"Beam number {beam_number} not found in plan")
 
    # Save the modified plan
    ds.save_as(output_plan_path)
    print(f"Plan saved to {output_plan_path}")

def get_model_input(patient: PatientData, machine: MachineConfig):
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
     
def randomize_weights(constraints):  #
    """
    Creates a new dictionary with the same structure as the input constraints,
    but with randomized weight values (between 1 and 100).

    Args:
        constraints (dict): The original constraints dictionary.

    Returns:
        dict: A new dictionary with randomized weights.
    """
    new_constraints = copy.deepcopy(constraints)
    for roi in new_constraints["weight"]:
        new_constraints["weight"][roi] = random.uniform(0.01, 1.0)
    return new_constraints


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
        "loss_lower_bound_gy": 1.0, # 10**np.random.randint(min_int_range, max_int_range),
        "loss_higher_bound_gy": 1.0, #10**np.random.randint(min_int_range, max_int_range),
        "loss_lower_bound_target": 0.0, # 10**np.random.randint(min_int_range, max_int_range),
        "loss_higher_bound_target": 0.0, # 10**np.random.randint(min_int_range, max_int_range),
        "l2_loss_oars_and_background": 10**np.random.randint(-3, 1), # 0.01,
        "mu_rate_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "mu_complexity_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "leaf_reg_loss": 10**np.random.randint(-3, 5), # 10**np.random.randint(min_int_range, max_int_range),
        "leaf_complexity_loss": 10**np.random.randint(-3, 5), # 10**np.random.randint(-2, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "jaw_opening_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
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
    Returns a (B, number_of_cps, num_leafs) mask marking which leaves ever intercept the PTV for each batch.
    Assumes leaves move along the z-axis (H axis).
    """
    if ptv_mask.ndim == 3:
        ptv_mask = ptv_mask.unsqueeze(0)  # [1, W, D, H]
    B = ptv_mask.shape[0]
    number_of_cps = config.number_of_cps
    num_leafs = config.number_of_leaf_pairs

    (H, D, W) = config.ct_array_shape
    dx, dy, dz = voxel_sizes

    iso_x = (W // 2) * dx
    iso_y = (D // 2) * dy
    iso_z = (H // 2) * dz

    isocenter = (iso_x, iso_y, iso_z)

    device = ptv_mask.device

    all_valid_leaf = torch.zeros(
        (B, number_of_cps, num_leafs), dtype=torch.uint8, device=device
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
            valid_leaf_1d.unsqueeze(0).expand(number_of_cps, -1).clone()
        )
        all_valid_leaf[b] = valid_leaf_per_beam

    return all_valid_leaf  # shape: (B, number_of_cps, num_leafs)



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
