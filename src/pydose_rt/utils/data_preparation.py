import random
import copy
import numpy as np
import torch
import os
import time
from pydose_rt.utils.config import config as PARAMS

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


def load_prepocessed_patient(file_ID, constraints_template, constraint_mode="fixed", epoch_number=0, is_normalize_weight=False, weight_ptv=None):
    constraints = get_constraints(constraints_template, constraint_mode, epoch_number, is_normalize_weight, weight_ptv)
        
    ct = None       
    while ct is None:
        try:
            ct = np.transpose((np.load(os.path.join(file_ID, "CT.npy"), allow_pickle=True) / 1000), (1, 2, 0))
        except Exception:
            time.sleep(random.random())

    ptv = None
    while ptv is None:
        try:
            structs = np.load(os.path.join(file_ID, "StructureSet.npy"), allow_pickle=True).astype(np.float32)
            ptv = np.transpose(structs[0, ...], (1, 2, 0))
            oar_penile = np.clip(np.transpose(structs[1, ...], (1, 2, 0)) - (ptv), 0, 1)
            oar_fem_l = np.clip(np.transpose(structs[2, ...], (1, 2, 0)) - (ptv + oar_penile), 0, 1)
            oar_fem_r = np.clip(np.transpose(structs[3, ...], (1, 2, 0)) - (ptv + oar_penile + oar_fem_l), 0, 1)
            oar_bladder = np.clip(np.transpose(structs[4, ...], (1, 2, 0)) - (ptv + oar_penile + oar_fem_l + oar_fem_r), 0, 1)
            oar_rectum = np.clip(np.transpose(structs[5, ...], (1, 2, 0)) - (ptv + oar_penile + oar_fem_l + oar_fem_r + oar_bladder), 0, 1)
            background = np.clip(np.transpose(structs[6, ...], (1, 2, 0)) - (ptv + oar_penile + oar_fem_l + oar_fem_r + oar_bladder + oar_rectum), 0, 1)
            # background = np.clip(np.ones_like(ptv) - (ptv + oar_penile + oar_fem_l + oar_fem_r + oar_bladder + oar_rectum), 0, 1)
        except Exception:
            time.sleep(random.random())

    # Dose (optional)
    dose = None
    if (os.path.exists(os.path.join(file_ID, "Dose.npy"))):
        while dose is None:
            try:
                dose = np.transpose(np.load(os.path.join(file_ID, "Dose.npy"), allow_pickle=True), (1, 2, 0))
            except Exception:
                time.sleep(random.random())
    else:
        dose = np.zeros_like(ct)
    dose_tensor = torch.from_numpy(dose)

    # --- Now generate bound/weight matrices (AFTER augmentation) ---
    structures = {
        "PTV": ptv,
        "ROI1": oar_penile,
        "ROI2": oar_fem_l,
        "ROI3": oar_fem_r,
        "ROI4": oar_bladder,
        "ROI5": oar_rectum,
        "ROI6": background,
    }

    lower_bound_gy = create_bound_weight_matrix(
        structures, bound=constraints["lower_bound_gy"]
    )
    higher_bound_gy = create_bound_weight_matrix(
        structures, bound=constraints["higher_bound_gy"]
    )
    lower_bound_target_percent = create_bound_weight_matrix(
        structures, bound=constraints["lower_bound_target_percent"]
    )
    higher_bound_target_percent = create_bound_weight_matrix(
        structures, bound=constraints["higher_bound_target_percent"]
    )
    region_weight = create_bound_weight_matrix(
        structures, bound=constraints["weight"], factor=1
    )

    # Stack inputs (channel-first): expected shape -> (C, W, D, H)
    images = np.stack(
        [
            ct,
            lower_bound_gy,
            higher_bound_gy,
            lower_bound_target_percent,
            higher_bound_target_percent,
            region_weight,
        ],
        axis=0,
    )

    mask = np.stack(
        [
            ptv,
            oar_penile,
            oar_fem_l,
            oar_fem_r,
            oar_bladder,
            oar_rectum,
            background,
        ],
        axis=0,
    )

    images_tensor = torch.from_numpy(images.astype(np.float32))
    masks_tensor = torch.from_numpy(mask.astype(np.float32))

    region_weights_tensor = torch.from_numpy(region_weight.astype(np.float32))

    return (
        images_tensor,
        dose_tensor,
        masks_tensor,
        region_weights_tensor,
        constraints,
    )
            
def create_bound_weight_matrix(structures, bound, factor=1):
    first_structure = next(iter(structures.values()))
    bound_matrix = np.zeros_like(first_structure, dtype=np.float32)
    for structure_id, array in structures.items():
        if structure_id in bound:
            bound_matrix += array * bound[structure_id]
    return bound_matrix / factor

def get_constraints(constraints_template, constraint_mode, epoch_number, is_normalize_weight, weight_ptv):
    epoch = epoch_number
    p_augment = min(epoch * 0.1, 1.0)
    if constraint_mode == "fixed":
        constraints = copy.deepcopy(constraints_template)
    elif constraint_mode == "rconstraint":
        if random.random() < p_augment:
            constraints = copy.deepcopy(random.choice(PARAMS.list_constraints))
        else:
            constraints = copy.deepcopy(constraints_template)
        if weight_ptv is not None:
            for key, value in constraints["weight"].items():
                if value == 10:
                    constraints["weight"][key] = weight_ptv
    elif constraint_mode == "rweight":
        if random.random() < p_augment:
            constraints = randomize_weights(
                copy.deepcopy(constraints_template)
            )
        else:
            constraints = copy.deepcopy(constraints_template)

    if is_normalize_weight:
        constraints = normalize_weights(constraints)

    constraints["weight"]["ROI1"] = 10**np.random.randint(0, 2)
    constraints["weight"]["ROI2"] = 10**np.random.randint(0, 3)
    constraints["weight"]["ROI3"] = 10**np.random.randint(0, 3)
    constraints["weight"]["ROI4"] = 10**np.random.randint(0, 3)
    constraints["weight"]["ROI5"] = 10**np.random.randint(0, 3)
    constraints["weight"]["ROI6"] = 10**np.random.randint(0, 3)

    return constraints

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
