import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from pydose_rt.data import MachineConfig, OptimizationConfig, Patient, Beam
from pydose_rt.data.beam import BeamSequence
import copy
import pymedphys
import torch
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import binary_fill_holes, binary_erosion

# DVH calculation functions moved to pydose_rt.data.metrics_helpers

def dose_at_volume_max(
    dose_array: np.ndarray,
    structure_mask: np.ndarray,
    volume_percent: float,
    dose_threshold_percent: float,
    prescription_gy: float
) -> float:
    """
    Check if dose at a given volume percentage is at most the threshold.
    Parameters:
    -----------
    dose_array : np.ndarray
        3D dose distribution (Gy)
    structure_mask : np.ndarray
        3D binary mask for the structure
    volume_percent : float
        Percentage of volume (0-100). Use 0.01 for max dose checks.
    dose_threshold_percent : float
        Maximum allowed dose as percentage of prescription (0-200+)
    prescription_gy : float
        Prescription dose in Gy
    Returns:
    --------
    float
        Ratio where < 1.0 = passed, > 1.0 = failed
        ratio = actual_dose / threshold_dose
    Example:
    --------
    # Check if D2% <= 107% of prescription (hot spot constraint)
    ratio = dose_at_volume_max(dose, ptv_mask, 2.0, 107.0, 42.7)
    # ratio = 0.95 means actual D2% is 95% of allowed maximum (passed)
    # Check if max dose <= 105% of prescription
    ratio = dose_at_volume_max(dose, bladder_mask, 0.01, 105.0, 42.7)
    """
    actual_dose = dose_at_volume_percent(dose_array, structure_mask, volume_percent)
    threshold_dose = dose_threshold_percent / 100.0 * prescription_gy
    return actual_dose / threshold_dose if threshold_dose > 0 else float('inf')


def dose_at_volume_min(
    dose_array: np.ndarray,
    structure_mask: np.ndarray,
    volume_percent: float,
    dose_threshold_percent: float,
    prescription_gy: float
) -> float:
    """
    Check if dose at a given volume percentage is at least the threshold.
    Parameters:
    -----------
    dose_array : np.ndarray
        3D dose distribution (Gy)
    structure_mask : np.ndarray
        3D binary mask for the structure
    volume_percent : float
        Percentage of volume (0-100)
    dose_threshold_percent : float
        Minimum required dose as percentage of prescription (0-100)
    prescription_gy : float
        Prescription dose in Gy
    Returns:
    --------
    float
        Ratio where < 1.0 = passed, > 1.0 = failed
        ratio = threshold_dose / actual_dose
    Example:
    --------
    # Check if D95% >= 90% of prescription (coverage constraint)
    ratio = dose_at_volume_min(dose, ptv_mask, 95.0, 90.0, 42.7)
    # ratio = 0.98 means actual D95% is above minimum (passed)
    # Check if D99% >= 95% of prescription
    ratio = dose_at_volume_min(dose, ptv_mask, 99.0, 95.0, 42.7)
    """
    actual_dose = dose_at_volume_percent(dose_array, structure_mask, volume_percent)
    threshold_dose = dose_threshold_percent / 100.0 * prescription_gy
    return threshold_dose / actual_dose if actual_dose > 0 else float('inf')


def volume_at_dose_max(
    dose_array: np.ndarray,
    structure_mask: np.ndarray,
    dose_threshold_percent: float,
    volume_threshold_percent: float,
    prescription_gy: float
) -> float:
    """
    Check if volume receiving a given dose is at most the threshold.
    Parameters:
    -----------
    dose_array : np.ndarray
        3D dose distribution (Gy)
    structure_mask : np.ndarray
        3D binary mask for the structure
    dose_threshold_percent : float
        Dose level as percentage of prescription (0-100+)
    volume_threshold_percent : float
        Maximum allowed volume percentage (0-100)
    prescription_gy : float
        Prescription dose in Gy
    Returns:
    --------
    float
        Ratio where < 1.0 = passed, > 1.0 = failed
        ratio = actual_volume / threshold_volume
    Example:
    --------
    # Check if V(90% of Rx) <= 15% (OAR sparing constraint)
    ratio = volume_at_dose_max(dose, rectum_mask, 90.0, 15.0, 42.7)
    # ratio = 0.85 means 12.75% of volume receives the dose (passed)
    # Check if V(75% of Rx) <= 35%
    ratio = volume_at_dose_max(dose, bladder_mask, 75.0, 35.0, 42.7)
    """
    threshold_dose = dose_threshold_percent / 100.0 * prescription_gy
    actual_volume = volume_at_dose(dose_array, structure_mask, threshold_dose)
    return actual_volume / volume_threshold_percent if volume_threshold_percent > 0 else float('inf')


def volume_at_dose_min(
    dose_array: np.ndarray,
    structure_mask: np.ndarray,
    dose_threshold_percent: float,
    volume_threshold_percent: float,
    prescription_gy: float
) -> float:
    """
    Check if volume receiving a given dose is at least the threshold.
    Parameters:
    -----------
    dose_array : np.ndarray
        3D dose distribution (Gy)
    structure_mask : np.ndarray
        3D binary mask for the structure
    dose_threshold_percent : float
        Dose level as percentage of prescription (0-100+)
    volume_threshold_percent : float
        Minimum required volume percentage (0-100)
    prescription_gy : float
        Prescription dose in Gy
    Returns:
    --------
    float
        Ratio where < 1.0 = passed, > 1.0 = failed
        ratio = threshold_volume / actual_volume
    Example:
    --------
    # Check if V(100% of Rx) >= 99% (target coverage constraint)
    ratio = volume_at_dose_min(dose, ptv_mask, 100.0, 99.0, 42.7)
    # ratio = 0.99 means 100% of volume receives the dose (passed)
    """
    threshold_dose = dose_threshold_percent / 100.0 * prescription_gy
    actual_volume = volume_at_dose(dose_array, structure_mask, threshold_dose)
    return volume_threshold_percent / actual_volume if actual_volume > 0 else float('inf')

def result_validation(patient: Patient,
                      machine_config: MachineConfig,
                      beam_sequence: BeamSequence,
                      pred_dose: torch.Tensor,
                      optimization_config: OptimizationConfig = None,
                      compute_gamma: bool = False,                      
                      compute_clinical_criteria: bool = True,
                      global_normalisation = None,
                      gamma_threshold_dose: float = 3.0,
                      gamma_threshold_distance: float = 3.0
                      ) -> Dict[str, float]:
    results = {}
    patient = patient.to('cpu')
    
    # Validate clinical criteria if requested
    if compute_clinical_criteria:
        pred_dose_np = patient.number_of_fractions * pred_dose.cpu().detach().numpy()
        patient_dose_np = patient.number_of_fractions * patient.dose.cpu().detach().numpy()

        validation_results = optimization_config.validate(pred_dose_np, patient)
        clinical_results = dict(sum([[(k + "_" +  v['type'], v['ratio']) for v in v_list['criteria']] for k, v_list in validation_results.items()], []))
        clinical_results["passed_test"] = np.mean(np.array(list(clinical_results.values())) < 1.0)
        results['clinical_criteria'] = clinical_results

        dose_50_percent = 0.5 * optimization_config.prescription_gy
        dose_max = optimization_config.prescription_gy
        dose_95_percent = 0.95 * optimization_config.prescription_gy
    
        dose_pred_50 = pred_dose_np > dose_50_percent
        dose_true_50 = patient_dose_np > dose_50_percent
        results['dice_50'] = 2 * np.sum(dose_pred_50 * dose_true_50) / (np.sum(dose_pred_50) + np.sum(dose_true_50))


        dose_pred_95 = pred_dose_np > dose_95_percent
        dose_true_95 = patient_dose_np > dose_95_percent
        results['dice_95'] = 2 * np.sum(dose_pred_95 * dose_true_95) / (np.sum(dose_pred_95) + np.sum(dose_true_95))

        results['mean_dose_diff'] = np.mean(pred_dose_np - patient_dose_np)
        results['mean_abs_dose_diff'] = np.mean(np.abs(pred_dose_np - patient_dose_np))
        results['mean_95th_percentile_diff'] = np.percentile(np.abs(pred_dose_np - patient_dose_np), 95.0)

        for mask_name in patient.structures.keys():
            results[f"{mask_name}_D_max"] = np.abs(patient.dose[patient.structures[mask_name]].cpu().detach().numpy().max() - pred_dose[patient.structures[mask_name]].cpu().detach().numpy().max())
            results[f"{mask_name}_D_mean"] = np.abs(patient.dose[patient.structures[mask_name]].cpu().detach().numpy().mean() - pred_dose[patient.structures[mask_name]].cpu().detach().numpy().mean())
        for mask_name in patient.structures.keys():
            for percent in [0.98, 0.5, 0.02]:
                results[f"{mask_name}_D_{percent}%"] = dose_at_volume_percent(patient_dose_np, patient.structures[mask_name].cpu().detach().numpy(), percent) - dose_at_volume_percent(pred_dose_np, patient.structures[mask_name].cpu().detach().numpy(), percent)

        volume_cc = np.prod(patient.resolution) / 1000.0
        for mask_name in patient.structures.keys():
            for cc in [2, 0.5]:
                results[f"{mask_name}_D_{cc}_cc"] = dose_at_volume_cc(patient_dose_np, patient.structures[mask_name].cpu().detach().numpy(), cc,  volume_cc) - dose_at_volume_cc(pred_dose_np, patient.structures[mask_name].cpu().detach().numpy(), cc, volume_cc)

        for mask_name in patient.structures.keys():
            for vv in [np.round(0.5 * optimization_config.prescription_gy, 2), 
                    np.round(0.36 * optimization_config.prescription_gy, 2), 
                    np.round(0.4 * optimization_config.prescription_gy)]:
                results[f"{mask_name}_V_{vv}_%"] = volume_at_dose(patient_dose_np, patient.structures[mask_name].cpu().detach().numpy(), vv) - volume_at_dose(pred_dose_np, patient.structures[mask_name].cpu().detach().numpy(), vv)

            
    if compute_gamma:
        axes = tuple(
            np.arange(patient.dose.shape[i]) * patient.resolution[i]
            for i in range(3)
        )
        
        # Compute dose cutoff value (10% of max dose)
        gamma_dose_ref = patient_dose_np
        gamma_dose_eval = pred_dose_np
        dose_cutoff = 10.0
        if global_normalisation is None:
            global_normalisation = gamma_dose_ref.max()
        dose_cutoff_value = dose_cutoff / 100 * global_normalisation
        gamma_mask = gamma_dose_ref > dose_cutoff_value
        dose_threshold = gamma_threshold_dose
        distance_threshold = gamma_threshold_distance
        max_gamma = 2.0
        
        # Create mask for evaluation (only where dose > cutoff)
        
        # Compute gamma
        gamma_map = pymedphys.gamma(
            axes_reference=axes,
            dose_reference=gamma_dose_ref,
            axes_evaluation=axes,
            dose_evaluation=gamma_dose_eval,
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
        if "Body" in patient.structures.keys():
            external_mask = binary_erosion(binary_fill_holes(patient.structures["Body"].cpu().detach().numpy()), np.ones((3, 3, 3)), iterations=5)
        else:
            external_mask = binary_erosion(binary_fill_holes(patient.structures["External"].cpu().detach().numpy()), np.ones((3, 3, 3)), iterations=5)
        gamma_valid = gamma_map[gamma_mask * external_mask]
        gamma_valid = gamma_valid[~np.isnan(gamma_valid)]
        pass_rate = np.sum(gamma_valid <= 1.0) / len(gamma_valid) * 100
        mean_gamma = np.mean(gamma_valid)

        results["gamma_pass_rate"] = pass_rate
        results["mean_gamma"] = mean_gamma

    pred_mlc = beam_sequence.leaf_positions.unsqueeze(0)
    pred_mus = beam_sequence.mus.unsqueeze(0)
    pred_jaws = beam_sequence.jaw_positions.unsqueeze(0)
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

    if (((pred_mlc[0, 1, :, :] - pred_mlc[0, 0, :, :]).min() * beam_sequence.field_size[0]).item() < machine_config.minimum_leaf_opening):
        results["check_mlc_collision_pass"] = 0
    else:
        results["check_mlc_collision_pass"] = 1

    if (((pred_mlc[0, 0, :, :].max() - pred_mlc[0, 0, :, :].min()) * beam_sequence.field_size[0]).item() > 150.0 or \
        ((pred_mlc[0, 1, :, :].max() - pred_mlc[0, 1, :, :].min()) * beam_sequence.field_size[0]).item() > 150.0):
        results["maximum_leaf_tip_difference"] = 0
    else:
        results["maximum_leaf_tip_difference"] = 1
    

    return results

"""
Helper functions for dose-volume histogram calculations.

These functions are used by OptimizationConfig for clinical criteria validation.
"""


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