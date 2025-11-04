import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.ndimage as ndi
import random
import copy
import numpy as np
from pydose_rt.data import PatientConfig, MachineConfig
import torch
import os
import time

def get_model_input(patient: PatientConfig, machine: MachineConfig):
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

def get_initial_weights():
    min_int_range = -3
    max_int_range = 2
    weights = {
        "loss_lower_bound_gy": 1.0, # 10**np.random.randint(min_int_range, max_int_range),
        "loss_higher_bound_gy": 1.0, #10**np.random.randint(min_int_range, max_int_range),
        "loss_lower_bound_target": 0.0, # 10**np.random.randint(min_int_range, max_int_range),
        "loss_higher_bound_target": 0.0, # 10**np.random.randint(min_int_range, max_int_range),
        "l2_loss_oars_and_background": 10**np.random.randint(-3, 1),
        "mu_rate_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "mu_complexity_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "leaf_reg_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "leaf_complexity_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(-2, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "jaw_opening_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "jaw_complexity_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
    }
    
    return weights

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
        "l2_loss_oars_and_background": 10**np.random.randint(-3, 1),
        "mu_rate_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "mu_complexity_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "leaf_reg_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "leaf_complexity_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(-2, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "jaw_opening_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
        "jaw_complexity_loss": 0.0, #10**np.random.randint(-3, 0), # 10**np.random.randint(min_int_range, max_int_range),
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

    (W, D, H) = config.ct_array_shape
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

