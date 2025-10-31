import os
import numpy as np
from pydose_rt.engine.data import DataGenerator
from pydose_rt import ModelConfig


def pad_str(s, width):
    s1 = s + " " * (
        width - len("This is string 1 with a fixed length of 100 characters.")
    )


def debug_print(name, tensor, elapsed_time=None, verbose=False):

    if verbose:
        import tensorflow as tf

        name_str = f"{name}:"
        shape_str = f"shape={tensor.shape}"
        min_str = f"min={tf.reduce_min(tensor):.4f}"
        max_str = f"max={tf.reduce_max(tensor):.4f}"
        mean_str = f"mean={tf.reduce_mean(tensor):.4f}"

        if elapsed_time is not None:
            time_str = f"time={elapsed_time:.4f} seconds"
        else:
            time_str = f"time=n/a seconds"

        s = "{:<65s} {:<35s} {:<15s} {:<15s} {:<15s} {:<30s}".format(
            name_str, shape_str, min_str, max_str, mean_str, time_str
        )

        print(s)


class TestSetup:
    def __init__(
        self,
        parent_dir=None,
        filedir="database/testsetup/b180_bs02_f16_v16_d04_ptv100_fixed_01/data_20_c1.npz",
    ):
        data_path = (
            os.path.join(parent_dir, "database/AUTORPT/")
            if parent_dir is not None
            else "database/AUTORPT/"
        )

        file_path = (
            os.path.join(
                parent_dir,
                filedir,
            )
            if parent_dir is not None
            else filedir
        )
        with np.load(file_path, allow_pickle=True) as npzdata_load:
            npzdata = {key: npzdata_load[key] for key in npzdata_load.files}

        self.gen = DataGenerator(data_path, "plotting", True, 1)

        self.npzdata = npzdata
        self.data = self.npzdata

        self.x = self.npzdata["x"]
        self.ct = np.array(self.x[0, ..., 0])

        self.masks = self.npzdata["masks"]

    def create_real(self):
        self.config = ModelConfig(
            ct_array_shape=(128, 128, 320),
            resolution=(0.3, 0.3, 0.3),
            field_size=(40, 40),
            number_of_leaf_pairs=60,
            tpr_20_10=0.72,
            downsampling_factor=(2, 2, 2),
            number_of_cps=180,
        )

        self.dose = self.npzdata["dose"]
        self.dose = np.array(self.dose)[0, ...]

        self.leafs = self.npzdata["leafs"]
        self.mus = self.npzdata["mus"]

    def create_dummy(self, number_of_leaf_pairs=60, number_of_cps=180):
        self.config = ModelConfig(
            ct_array_shape=(128, 128, 320),
            resolution=(0.3, 0.3, 0.3),
            field_size=(40, 40),
            number_of_leaf_pairs=number_of_leaf_pairs,
            tpr_20_10=0.72,
            downsampling_factor=(2, 2, 2),
            number_of_cps=number_of_cps,
        )

        np.random.seed(42)

        self.dose = None

        self.leafs = np.random.rand(*self.config.shape_mlc[0]).astype(np.float32)
        self.mus = np.random.rand(*self.config.shape_mlc[1]).astype(np.float32)


# print(TestSetup.config)


def check_for_nan(**tensors_with_names):
    """
    Checks a dictionary of TensorFlow tensors for NaN values.

    Args:
        **tensors_with_names: Keyword arguments where keys are the variable names
                             (strings) and values are the corresponding TensorFlow
                             tensors.

    Raises:
        ValueError: If any of the input tensors contain one or more NaN values.
                    The error message lists the names of the tensors containing NaNs.
        TypeError: If any of the provided values are not TensorFlow tensors
                   or cannot be converted to one.
    """
    nan_found_in = []
    import tensorflow as tf

    for name, tensor in tensors_with_names.items():
        try:
            # Ensure it's a tensor or convert it
            if not isinstance(tensor, tf.Tensor):
                # Try converting common types like numpy arrays or lists
                # Allow None to pass through without error, but maybe print a warning
                if tensor is None:
                    print(f"Warning: Input '{name}' is None, skipping NaN check.")
                    continue
                try:
                    tensor = tf.convert_to_tensor(
                        tensor, dtype_hint=tf.float32
                    )  # Added dtype_hint
                    # print(
                    #     f"Warning: Input '{name}' was not a tf.Tensor, but was converted."
                    # )
                except (ValueError, TypeError) as conversion_error:
                    raise TypeError(
                        f"Input '{name}' could not be converted to a tf.Tensor. Original error: {conversion_error}"
                    )

            # Check for NaNs - Only check float/complex types for NaN
            if tensor.dtype.is_floating or tensor.dtype.is_complex:
                if tf.reduce_any(tf.math.is_nan(tensor)):
                    print(
                        f"NaN detected in tensor: {name}"
                    )  # Print immediately when found
                    nan_found_in.append(name)
            # Optional: print a message for non-float types if desired
            # else:
            #    print(f"Skipping NaN check for tensor '{name}' with non-float dtype: {tensor.dtype}")

        except Exception as e:
            # Catch potential issues during checking (e.g., unexpected types)
            print(f"Error checking tensor '{name}': {e}")
            # Decide if you want to raise immediately or collect names
            # Collecting names might be better to report all issues at once
            nan_found_in.append(f"{name} (Error during check: {e})")

    # After checking all tensors, raise an error if any NaNs were found
    if nan_found_in:
        error_message = (
            f"NaN values detected in the following tensors: {', '.join(nan_found_in)}"
        )
        raise ValueError(error_message)
    else:
        # print("Checked all provided tensors. No NaN values found.")
        pass


if __name__ == "__main__":
    T = TestSetup()
    T.create_real()
