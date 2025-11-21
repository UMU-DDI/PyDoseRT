import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from pydose_rt.data import MachineConfig, TreatmentConfig, Patient
from pydose_rt import DoseEngine
import copy
import pymedphys
import torch
from typing import Dict, List, Tuple, Optional

def exponentially_weighted_difference(x, y1, y2, alpha=5.0):
    """
    Calculate exponentially weighted difference between two curves.
    
    Parameters:
    x: array of x values (0-1 range)
    y1, y2: arrays of y values for the two curves
    alpha: exponent factor to control weighting strength
    
    Returns:
    The exponentially weighted difference
    """
    # Exponential weighting
    weights = np.exp(alpha * x)
    
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    # Calculate differences
    diff = y1 - y2
    
    # Apply weights and sum
    exp_diff = np.sum(weights * diff)
    
    return exp_diff

def weighted_area_difference(x, y1, y2, weight_func=None):
    """
    Calculate weighted area difference between two curves.
    
    Parameters:
    x: array of x values (0-1 range)
    y1, y2: arrays of y values for the two curves
    weight_func: function that takes x and returns weights
    
    Returns:
    The weighted area difference
    """
    if weight_func is None:
        # Default weight function: linear increase with x
        weights = x
    else:
        weights = weight_func(x)
    
    # Calculate point-wise differences
    diff = y1 - y2
    
    # Apply weights to differences
    weighted_diff = diff * weights
    
    # Calculate the area using trapezoid rule
    area = trapezoid(weighted_diff, x)
    
    return area



def dose_at_volume_percent(dose_array: np.ndarray,
                           structure_mask: np.ndarray,
                           volume_percent: float) -> float:
    """
    Calculate the dose (Gy) received by a given percentage of the structure volume.
    This computes Dx% - the dose at x% of the volume.
    Parameters:
    -----------
    dose_array : np.ndarray
        3D dose distribution (Gy)
    structure_mask : np.ndarray
        3D binary mask for the structure
    volume_percent : float
        Percentage of volume (0-100)
    Returns:
    --------
    float
        Dose in Gy at the specified volume percentage.
        For example, D95% with volume_percent=95 returns the dose covering 95% of the volume.
    """
    # Extract doses within the structure
    structure_doses = dose_array[structure_mask > 0]

    if len(structure_doses) == 0:
        return 0.0

    # Sort doses in descending order
    sorted_doses = np.sort(structure_doses)[::-1]

    # Calculate the index corresponding to the volume percentage
    # volume_percent% of volume means we want the dose that covers this percentage
    idx = int(np.ceil(len(sorted_doses) * volume_percent / 100.0)) - 1
    idx = max(0, min(idx, len(sorted_doses) - 1))

    return float(sorted_doses[idx])


def dose_at_volume_cc(dose_array: np.ndarray,
                      structure_mask: np.ndarray,
                      volume_cc: float,
                      voxel_volume_cc: float) -> float:
    """
    Calculate the dose (Gy) received by a given absolute volume (cc) of the structure.
    This computes Dx cc - the minimum dose to the hottest x cc of the structure.
    Parameters:
    -----------
    dose_array : np.ndarray
        3D dose distribution (Gy)
    structure_mask : np.ndarray
        3D binary mask for the structure
    volume_cc : float
        Volume in cubic centimeters
    voxel_volume_cc : float
        Volume of a single voxel in cc
    Returns:
    --------
    float
        Dose in Gy at the specified volume.
    """
    # Extract doses within the structure
    structure_doses = dose_array[structure_mask > 0]

    if len(structure_doses) == 0:
        return 0.0

    # Sort doses in descending order
    sorted_doses = np.sort(structure_doses)[::-1]

    # Calculate number of voxels corresponding to the volume
    n_voxels = int(np.ceil(volume_cc / voxel_volume_cc))
    n_voxels = max(1, min(n_voxels, len(sorted_doses)))

    # Return the dose at the n_voxels-th hottest voxel
    return float(sorted_doses[n_voxels - 1])


def volume_at_dose(dose_array: np.ndarray,
                   structure_mask: np.ndarray,
                   dose_threshold: float) -> float:
    """
    Calculate the percentage of structure volume receiving at least a given dose.
    This computes Vx Gy - the volume % receiving at least x Gy.
    Parameters:
    -----------
    dose_array : np.ndarray
        3D dose distribution (Gy)
    structure_mask : np.ndarray
        3D binary mask for the structure
    dose_threshold : float
        Dose threshold in Gy
    Returns:
    --------
    float
        Percentage of volume (0-100) receiving at least the threshold dose.
    """
    # Extract doses within the structure
    structure_doses = dose_array[structure_mask > 0]

    if len(structure_doses) == 0:
        return 0.0

    # Calculate the fraction of volume receiving at least the threshold dose
    volume_fraction = np.sum(structure_doses >= dose_threshold) / len(structure_doses)

    return float(volume_fraction * 100.0)


def validate_clinical_criteria(patient: Patient,
                               treatment_config: TreatmentConfig,
                               pred_dose: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Validate predicted dose against clinical criteria defined in treatment_config.
    Returns a ratio for each criterion where:
    - ratio < 1.0 means the criterion is met (passed)
    - ratio = 1.0 means the criterion is exactly at the threshold
    - ratio > 1.0 means the criterion is violated (failed)
    Parameters:
    -----------
    patient : Patient
        Patient data including structure masks
    treatment_config : TreatmentConfig
        Treatment configuration with clinical criteria
    pred_dose : np.ndarray
        Predicted dose distribution (Gy), shape (1, D, H, W) or (D, H, W)
    Returns:
    --------
    Dict[str, Dict[str, float]]
        Nested dictionary with structure names as keys, and for each structure:
        - 'criteria': List of criterion results
        - Each criterion has: 'type', 'description', 'value', 'threshold', 'ratio', 'passed'
    """
    # Handle dose array shape
    if pred_dose.ndim == 4:
        dose = pred_dose[0, ...]
    else:
        dose = pred_dose

    # Calculate voxel volume in cc
    voxel_spacing_mm = patient.voxel_spacing_mm
    voxel_volume_cc = np.prod(voxel_spacing_mm) / 1000.0  # Convert mm³ to cc

    results = {}

    # Process each structure with clinical criteria
    for struct in treatment_config.structures:
        structure_name = struct.name

        # Skip if structure not in patient masks
        if structure_name not in patient.structures:
            continue

        structure_mask = patient.structures[structure_name]

        structure_results = {
            'criteria': []
        }

        # Process explicit clinical criteria if defined
        if hasattr(struct, 'clinical_criteria') and struct.clinical_criteria:
            for clin_criterion in struct.clinical_criteria:
                criterion_result = _evaluate_clinical_criterion(
                    dose, structure_mask, clin_criterion, voxel_volume_cc,
                    prescription_gy=treatment_config.prescription_gy
                )
                structure_results['criteria'].append(criterion_result)

        # If no explicit criteria, fall back to generating criteria from constraints
        elif hasattr(struct, 'constraints'):
            constraints = struct.constraints

            # Criterion 1: Lower bound dose at lower bound target percent (D_x% >= threshold)
            # This is typically for targets: D95% >= prescription dose
            if constraints.lower_bound_target_percent > 0 and constraints.lower_bound_gy > 0:
                actual_dose = dose_at_volume_percent(
                    dose, structure_mask, constraints.lower_bound_target_percent
                )
                threshold_dose = constraints.lower_bound_gy

                # For "at least" constraints, ratio = threshold / actual
                # If actual > threshold, ratio < 1 (passed)
                if actual_dose > 0:
                    ratio = threshold_dose / actual_dose
                else:
                    ratio = float('inf') if threshold_dose > 0 else 1.0

                criterion = {
                    'type': f'D{constraints.lower_bound_target_percent:.2f}%',
                    'description': f'At least {threshold_dose:.2f} Gy dose at {constraints.lower_bound_target_percent:.2f} % volume',
                    'value': actual_dose,
                    'threshold': threshold_dose,
                    'ratio': ratio,
                    'passed': ratio <= 1.0
                }
                structure_results['criteria'].append(criterion)

            # Criterion 2: Higher bound dose at higher bound target percent (D_x% <= threshold)
            # This checks that no more than x% of volume receives more than threshold dose
            if constraints.higher_bound_target_percent < 100 and constraints.higher_bound_gy > 0:
                # D_{100 - higher_bound_target_percent}% <= threshold
                # If higher_bound_target_percent = 2%, we check D2% (dose to hottest 2%)
                actual_dose = dose_at_volume_percent(
                    dose, structure_mask, 100 - constraints.higher_bound_target_percent
                )
                threshold_dose = constraints.higher_bound_gy

                # For "at most" constraints, ratio = actual / threshold
                # If actual < threshold, ratio < 1 (passed)
                if threshold_dose > 0:
                    ratio = actual_dose / threshold_dose
                else:
                    ratio = float('inf') if actual_dose > 0 else 1.0

                criterion = {
                    'type': f'D{100 - constraints.higher_bound_target_percent:.2f}%',
                    'description': f'At most {threshold_dose:.2f} Gy dose at {100 - constraints.higher_bound_target_percent:.2f} % volume',
                    'value': actual_dose,
                    'threshold': threshold_dose,
                    'ratio': ratio,
                    'passed': ratio <= 1.0
                }
                structure_results['criteria'].append(criterion)

            # Criterion 3: Volume at higher bound dose (V_x Gy <= threshold %)
            # This is for OARs: V_32Gy <= 35% (no more than 35% receives 32 Gy)
            if constraints.higher_bound_gy > 0 and constraints.higher_bound_target_percent < 100:
                actual_volume_percent = volume_at_dose(
                    dose, structure_mask, constraints.higher_bound_gy
                )
                threshold_volume_percent = constraints.higher_bound_target_percent

                # For "at most" constraints, ratio = actual / threshold
                if threshold_volume_percent > 0:
                    ratio = actual_volume_percent / threshold_volume_percent
                else:
                    ratio = float('inf') if actual_volume_percent > 0 else 1.0

                criterion = {
                    'type': f'V{constraints.higher_bound_gy:.2f}Gy',
                    'description': f'At most {threshold_volume_percent:.2f} % volume at {constraints.higher_bound_gy:.2f} Gy dose',
                    'value': actual_volume_percent,
                    'threshold': threshold_volume_percent,
                    'ratio': ratio,
                    'passed': ratio <= 1.0
                }
                structure_results['criteria'].append(criterion)

        results[structure_name] = structure_results

    return results


def _evaluate_clinical_criterion(dose: np.ndarray,
                                 structure_mask: np.ndarray,
                                 criterion: 'ClinicalCriterion',
                                 voxel_volume_cc: float,
                                 prescription_gy: Optional[float] = None) -> Dict:
    """
    Evaluate a single clinical criterion and return the result.
    Parameters:
    -----------
    dose : np.ndarray
        3D dose distribution (Gy)
    structure_mask : np.ndarray
        3D binary mask for the structure
    criterion : ClinicalCriterion
        The clinical criterion to evaluate
    voxel_volume_cc : float
        Volume of a single voxel in cc
    prescription_gy : Optional[float]
        Prescription dose in Gy (needed if criterion uses dose_percent)
    Returns:
    --------
    Dict
        Dictionary with 'type', 'description', 'value', 'threshold', 'ratio', 'passed'
    """

    # Resolve dose threshold: use dose_percent if available, otherwise dose_gy
    def get_dose_threshold(crit) -> float:
        if crit.dose_percent is not None:
            if prescription_gy is None:
                raise ValueError("Criterion uses dose_percent but prescription_gy not provided")
            return crit.dose_percent * prescription_gy / 100.0
        elif crit.dose_gy is not None:
            return crit.dose_gy
        else:
            raise ValueError("Criterion must specify either dose_gy or dose_percent")

    if criterion.criterion_type == 'dose_at_volume':
        # Dx% - dose at x% of volume
        actual_value = dose_at_volume_percent(
            dose, structure_mask, criterion.volume_percent
        )
        threshold = get_dose_threshold(criterion)

        # Determine ratio based on constraint type
        if criterion.constraint_type == 'at_least':
            # D95% >= 38.43 Gy
            ratio = threshold / actual_value if actual_value > 0 else float('inf')
            type_str = f'D{criterion.volume_percent:.2f}%'
            desc = criterion.description or f'At least {threshold:.2f} Gy dose at {criterion.volume_percent:.2f} % volume'
        else:  # at_most
            # D2% <= 45.69 Gy
            ratio = actual_value / threshold if threshold > 0 else float('inf')
            type_str = f'D{criterion.volume_percent:.2f}%'
            desc = criterion.description or f'At most {threshold:.2f} Gy dose at {criterion.volume_percent:.2f} % volume'

    elif criterion.criterion_type == 'dose_at_volume_cc':
        # Dx cc - dose at x cubic centimeters
        actual_value = dose_at_volume_cc(
            dose, structure_mask, criterion.volume_cc, voxel_volume_cc
        )
        threshold = get_dose_threshold(criterion)

        # For dose constraints, typically "at_most"
        if criterion.constraint_type == 'at_most':
            ratio = actual_value / threshold if threshold > 0 else float('inf')
            type_str = f'D{criterion.volume_cc:.2f}cc'
            desc = criterion.description or f'At most {threshold:.2f} Gy dose at {criterion.volume_cc:.2f} cm³ volume'
        else:  # at_least (rare for absolute volume)
            ratio = threshold / actual_value if actual_value > 0 else float('inf')
            type_str = f'D{criterion.volume_cc:.2f}cc'
            desc = criterion.description or f'At least {threshold:.2f} Gy dose at {criterion.volume_cc:.2f} cm³ volume'

    elif criterion.criterion_type == 'volume_at_dose':
        # Vx Gy - volume % receiving at least x Gy
        dose_threshold = get_dose_threshold(criterion)
        actual_value = volume_at_dose(
            dose, structure_mask, dose_threshold
        )
        threshold = criterion.volume_percent

        # Determine ratio based on constraint type
        if criterion.constraint_type == 'at_most':
            # V38.5Gy <= 15%
            ratio = actual_value / threshold if threshold > 0 else float('inf')
            type_str = f'V{dose_threshold:.2f}Gy'
            desc = criterion.description or f'At most {threshold:.2f} % volume at {dose_threshold:.2f} Gy dose'
        else:  # at_least
            # V42.7Gy >= 95%
            ratio = threshold / actual_value if actual_value > 0 else float('inf')
            type_str = f'V{dose_threshold:.2f}Gy'
            desc = criterion.description or f'At least {threshold:.2f} % volume at {dose_threshold:.2f} Gy dose'

    else:
        raise ValueError(f"Unknown criterion type: {criterion.criterion_type}")

    return {
        'type': type_str,
        'description': desc,
        'value': actual_value,
        'threshold': threshold,
        'ratio': ratio,
        'passed': ratio <= 1.0
    }

def result_validation(patient: Patient,
                      machine_config: MachineConfig,
                      treatment_config: TreatmentConfig,
                      pred_dose: np.array,
                      pred_mlc: np.array,
                      pred_jaws: np.array,
                      pred_mus: np.array,
                      compute_gamma: bool = False,                      
                      compute_clinical_criteria: bool = True,
                      global_normalisation = None):
    results = {}
    
    # Validate clinical criteria if requested
    if compute_clinical_criteria:
        clinical_results = validate_clinical_criteria(
            patient, treatment_config, pred_dose
        )
        results['clinical_criteria'] = clinical_results
        
    if compute_gamma:
        axes = tuple(
            np.arange(patient.dose.shape[i]) * patient.voxel_spacing_mm[i]
            for i in range(3)
        )
        
        # Compute dose cutoff value (10% of max dose)
        
        dose_cutoff = 10.0
        if global_normalisation is None:
            global_normalisation = patient.dose.max()
        dose_cutoff_value = dose_cutoff / 100 * global_normalisation
        dose_threshold = 3.0
        distance_threshold = 3.0
        max_gamma = 2.0
        
        # Create mask for evaluation (only where dose > cutoff)
        
        # Compute gamma
        gamma_map = pymedphys.gamma(
            axes_reference=axes,
            dose_reference=patient.dose,
            axes_evaluation=axes,
            dose_evaluation=pred_dose[0, ...],
            dose_percent_threshold=dose_threshold,
            distance_mm_threshold=distance_threshold,
            lower_percent_dose_cutoff=dose_cutoff,
            interp_fraction=10,  # Interpolation resolution
            max_gamma=max_gamma,
            global_normalisation=global_normalisation,
            local_gamma=False,  # Global gamma (% of max dose)
            quiet=True
        )
        
        # Calculate pass rate
        mask = patient.dose > dose_cutoff_value
        # mask = config.patient.structures["External"] > 0
        gamma_valid = gamma_map[mask]
        gamma_valid = gamma_valid[~np.isnan(gamma_valid)]
        pass_rate = np.sum(gamma_valid <= 1.0) / len(gamma_valid) * 100
        mean_gamma = np.mean(gamma_valid)

        results["gamma_pass_rate"] = pass_rate
        results["mean_gamma"] = mean_gamma

    # Start with values in the predictions
    if (pred_dose.min() < 0):
        results["check_min_dose_pass"] = 0
    else:
        results["check_min_dose_pass"] = 1

    if (pred_mlc.min() < 0) or (pred_mlc.max() > 1):
        results["check_mlc_bounds_pass"] = 0
    else:
        results["check_mlc_bounds_pass"] = 1

    if (pred_jaws.min() < 0) or (pred_jaws.max() > 1):
        results["check_jaws_bounds_pass"] = 0
    else:
        results["check_jaws_bounds_pass"] = 1

    if (pred_mus.min() < 0):
        results["check_mus_bounds_pass"] = 0
    else:
        results["check_mus_bounds_pass"] = 1

    if (((pred_mlc[0, 1, :, :] - pred_mlc[0, 0, :, :]).min() * treatment_config.field_size[0]).item() < machine_config.minimum_leaf_overlap):
        results["check_mlc_collision_pass"] = 0
    else:
        results["check_mlc_collision_pass"] = 1

    if (((pred_mlc[0, 0, :, :].max() - pred_mlc[0, 0, :, :].min()) * treatment_config.field_size[0]).item() > 150.0 or \
        ((pred_mlc[0, 1, :, :].max() - pred_mlc[0, 1, :, :].min()) * treatment_config.field_size[0]).item() > 150.0):
        results["maximum_leaf_tip_difference"] = 0
    else:
        results["maximum_leaf_tip_difference"] = 1
    

    return results

def validate_unit_dose(machine: MachineConfig, treatment: TreatmentConfig, target_mu: int):
    # Create config for 20x20x20 cm phantom with 2mm resolution
    # 200mm / 2mm = 100 voxels per dimension
    treatment = copy.deepcopy(treatment)
    device = treatment.device
    # config.machine.ct_array_shape = tuple(np.divide((200, 200, 200), config.machine.resolution).astype(np.int32))
    treatment.number_of_cps = 1
    treatment.starting_angle = 0
    
    # config.machine.downsampling_factor = (1,1,1)
    center_x, center_y, center_z = np.divide(machine.ct_array_shape, 2).astype(np.int32)
    iso_y = - (100 - center_y * machine.resolution[1])
    center_y_iso = center_y - int(iso_y / machine.resolution[1])
    treatment.iso_center = (0.0, iso_y, 0.0)
 
    # Create water phantom (HU = 0 for water)
    x_ct = 0.0 * np.expand_dims(np.ones(machine.ct_array_shape), 0)
 
    # Set up MLC positions for full 10x10 field
    # Positions are normalized: 0.5 and 1.0 create a centered field
    y_mlc = np.zeros((1, 2, treatment.number_of_cps, machine.number_of_leaf_pairs))
    y_mlc[:, 0, :, :] = - 100.0
    y_mlc[:, 1, :, :] = 100.0
 
    # Set up jaw positions for 10x10 field
    y_jaws = np.zeros((1, 2, treatment.number_of_cps))
    y_jaws[:, 0, :] = - 100.0
    y_jaws[:, 1, :] = 100.0
 
    # Set monitor units
    mus = target_mu * np.ones((1, treatment.number_of_cps), dtype=np.float32)
 
    # Create dose engine
    dose_layer = DoseEngine(machine, treatment, permute_ct=False, leafs_centered=False)
 
    # Calculate dose
    dose = dose_layer(
        torch.tensor(y_mlc, dtype=treatment.dtype, device=device),
        torch.tensor(mus, dtype=treatment.dtype, device=device),
        jaw_positions=torch.tensor(y_jaws, dtype=treatment.dtype, device=device),
        ct_image=torch.tensor(x_ct, dtype=treatment.dtype, device=device)
    )

    # Get center dose (at 10cm depth - index 50 for 100 voxels)
    center_dose = dose[0, center_x, center_y_iso, center_z].detach().cpu().numpy()

    # Calculate calibration factor
    # This gives the factor to normalize to 1 Gy per MU at reference conditions
    calibration_factor = machine.mean_photon_energy_MeV / center_dose

    return center_dose, calibration_factor