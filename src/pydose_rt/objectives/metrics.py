import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from pydose_rt.data import DoseConfig
import pymedphys

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

def result_validation(config: DoseConfig, 
                      pred_dose: np.array, 
                      pred_mlc: np.array, 
                      pred_jaws: np.array, 
                      pred_mus: np.array):
    
    axes = tuple(
        np.arange(config.patient.dose.shape[i]) * config.patient.voxel_spacing_mm[i]
        for i in range(3)
    )
    
    # Compute dose cutoff value (10% of max dose)
    dose_cutoff = 10.0
    dose_cutoff_value = dose_cutoff / 100 * np.max(config.patient.dose)
    dose_threshold = 3.0
    distance_threshold = 3.0
    max_gamma = 2.0
    
    # Create mask for evaluation (only where dose > cutoff)
    
    # Compute gamma
    gamma_map = pymedphys.gamma(
        axes_reference=axes,
        dose_reference=config.patient.dose,
        axes_evaluation=axes,
        dose_evaluation=pred_dose[0, ...],
        dose_percent_threshold=dose_threshold,
        distance_mm_threshold=distance_threshold,
        lower_percent_dose_cutoff=dose_cutoff,
        interp_fraction=10,  # Interpolation resolution
        max_gamma=max_gamma,
        local_gamma=False,  # Global gamma (% of max dose)
        quiet=True
    )
    
    # Calculate pass rate
    # mask = config.patient.dose > dose_cutoff_value
    mask = config.patient.structures["External"] > 0
    gamma_valid = gamma_map[mask]
    gamma_valid = gamma_valid[~np.isnan(gamma_valid)]
    pass_rate = np.sum(gamma_valid <= 1.0) / len(gamma_valid) * 100
    mean_gamma = np.mean(gamma_valid)

    print(pass_rate)
    print(mean_gamma)
    results = {}

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

    

    return results

# Example usage
if __name__ == "__main__":
    # Generate sample data
    x = np.linspace(0, 1, 100)
    y1 = np.sin(2 * np.pi * x) + 0.5
    y2 = np.sin(2 * np.pi * x + 0.5) + 0.3
    
    # Define exponential weight function
    def exp_weight(x, alpha=5.0):
        return np.exp(alpha * x)
    
    # Calculate the differences
    exp_diff = exponentially_weighted_difference(x, y1, y2, alpha=5.0)
    area_diff = weighted_area_difference(x, y1, y2, 
                                        weight_func=lambda x: exp_weight(x, alpha=5.0))
    
    # Print results
    print(f"Exponentially Weighted Difference: {exp_diff:.6f}")
    print(f"Exponentially Weighted Area Difference: {area_diff:.6f}")
    
    # Visualize the curves and weights
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
    # Plot curves
    ax1.plot(x, y1, 'b-', label='Curve 1')
    ax1.plot(x, y2, 'r-', label='Curve 2')
    ax1.set_title('Sample Curves')
    ax1.legend()
    ax1.grid(True)
    
    # Plot differences
    diff = y1 - y2
    ax2.plot(x, diff, 'g-')
    ax2.fill_between(x, 0, diff, alpha=0.3, color='g')
    ax2.set_title('Raw Difference')
    ax2.grid(True)
    
    # Plot weighted differences
    weights = exp_weight(x)
    weights_norm = weights / np.sum(weights)
    weighted_diff = diff * weights_norm
    
    ax3.plot(x, weighted_diff, 'b-', label='Weighted Difference')
    ax3.fill_between(x, 0, weighted_diff, alpha=0.3, color='b')
    
    # Add weight function on secondary y-axis
    ax3_twin = ax3.twinx()
    ax3_twin.plot(x, weights_norm, 'r--', label='Weight Factor')
    ax3_twin.set_ylabel('Weight Factor', color='r')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    
    ax3.set_title('Exponentially Weighted Difference')
    ax3.grid(True)
    
    plt.tight_layout()
    # plt.show()
    
    # Create a figure to show how alpha affects the weighting
    plt.figure(figsize=(10, 6))
    alphas = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    for alpha in alphas:
        weights = exp_weight(x, alpha)
        weights_norm = weights / np.sum(weights)
        plt.plot(x, weights_norm, label=f'α = {alpha}')
    
    plt.title('Effect of α on Exponential Weighting')
    plt.xlabel('x')
    plt.ylabel('Normalized Weight')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    
    # Create a figure to show how alpha affects the weighted area difference
    plt.figure(figsize=(10, 6))
    alpha_range = np.linspace(0.1, 15, 50)
    area_diffs = []
    
    for alpha in alpha_range:
        area = weighted_area_difference(x, y1, y2, 
                                      weight_func=lambda x, a=alpha: exp_weight(x, a))
        area_diffs.append(area)
    
    plt.plot(alpha_range, area_diffs)
    plt.title('Effect of α on Exponentially Weighted Area Difference')
    plt.xlabel('α')
    plt.ylabel('Weighted Area Difference')
    plt.grid(True)
    plt.tight_layout()
    plt.show()