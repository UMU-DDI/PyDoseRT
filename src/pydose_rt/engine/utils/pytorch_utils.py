import os
import torch
import re
import glob


def print_first_n_params(model, n=10, title="Model Parameters"):
    """
    Prints the first N parameters of a PyTorch model.
    """
    print(f"\n--- {title} ---")
    param_count = 0
    for name, param in model.named_parameters():
        if param_count >= n:
            break
        # Print the first few values for brevity, along with shape
        print(f"  {name}: {param.data.flatten()[:5]}... (shape: {param.shape})")
        param_count += 1
    if param_count == 0:
        print("  No trainable parameters found in the model.")


def save_model(model, path="my_model.keras"):
    torch.save(model.state_dict(), path)
    print()
    print()
    print(f">> Model at {path} saved successfully!")
    print()
    print()


def load_model(model, path="my_model.keras"):
    model.load_state_dict(torch.load(path, weights_only=True))

    print()
    print()
    print(f">> Model at {path} loaded successfully!")
    print()
    print()

    return model


def find_and_rank_model_files(directory, base_experiment_name):
    """
    Finds and ranks "model_without_dose_layer" files in subdirectories containing base_experiment_name.

    Args:
        directory: The root directory to search in.
        base_experiment_name: The string to search for in subdirectory names.

    Returns:
        The path of the last ranked file, or None if no matching files are found.
    """

    matching_subdirs = []
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path) and base_experiment_name in subdir:
            matching_subdirs.append(subdir_path)

    if not matching_subdirs:
        return None

    all_model_files = []
    for subdir_path in matching_subdirs:
        model_files = glob.glob(os.path.join(subdir_path, "*model_wo_dose_*"))
        all_model_files.extend(model_files)

    if not all_model_files:
        return None

    ranked_files = sorted(all_model_files)

    return ranked_files[-1] if ranked_files else None


def get_latest_model_path(
    experiment_name, pretrained_folder="database/pretrained/pytorch"
):
    pattern = r"b\d+_bs\d+_f\d+_v\d+_d\d+_ptv\d+"
    pattern = r"b\d+_bs\d+_f\d+_v\d+_d\d+_de\w+_lw\d+_ptv\d+"

    match = re.search(pattern, experiment_name)
    base_experiment_name = match.group(0) + "_"
    return find_and_rank_model_files(pretrained_folder, base_experiment_name)


if __name__ == "__main__":
    experiment_name = "test_b090_bs01_f16_v16_d04_ptv100_rfixed_01"
    # file_path = "database/pretrained/b015_bs01_f16_v16_d04_ptv10_fixed_03/model_without_dose_layer_10.keras"
    file_path = get_latest_model_path(experiment_name)
    loaded_model = load_model(file_path)
    a = 2
