import os
import tensorflow as tf
import re
import glob


def set_fixed_gpu_memory_limit_mb(memory_limit_mb):
    """
    Sets a fixed memory limit for TensorFlow GPU in megabytes.

    Args:
        memory_limit_mb (int): The fixed memory limit in megabytes.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)],
            )
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set at program startup
            print(e)


def save_model(model, path="my_model.keras"):
    model.save(path, save_format="tf")
    print("Model saved successfully!")


def load_model(path="my_model.keras"):
    loaded_model = tf.keras.models.load_model(path)

    print()
    print()
    print(f">> Model at {path} loaded successfully!")
    print()
    print()

    return loaded_model

    # Continue training
    # ... (your data loading and processing code) ...

    # loaded_model.fit(
    #     x_train,
    #     y_train,
    #     epochs=10,  # Continue for 10 more epochs
    #     initial_epoch=loaded_model.optimizer.iterations.numpy(),  # Important
    #     validation_data=(x_val, y_val),
    # )

    # print("Training resumed successfully!")


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
        model_files = glob.glob(os.path.join(subdir_path, "*model_without_dose_layer*"))
        all_model_files.extend(model_files)

    if not all_model_files:
        return None

    ranked_files = sorted(all_model_files)

    return ranked_files[-1] if ranked_files else None


def get_latest_model_path(experiment_name, pretrained_folder="database/pretrained"):
    pattern = r"b\d+_bs\d+_f\d+_v\d+_d\d+_ptv\d+"
    match = re.search(pattern, experiment_name)
    base_experiment_name = match.group(0) + "_"
    return find_and_rank_model_files(pretrained_folder, base_experiment_name)


if __name__ == "__main__":
    experiment_name = "test_b090_bs01_f16_v16_d04_ptv100_rfixed_01"
    # file_path = "database/pretrained/b015_bs01_f16_v16_d04_ptv10_fixed_03/model_without_dose_layer_10.keras"
    file_path = get_latest_model_path(experiment_name)
    loaded_model = load_model(file_path)
    a = 2
