import os
import time, random
import numpy as np
import torch
import torch.nn.functional as F
import copy
import json
import scipy.ndimage
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter
from pydose_rt.utils.config import config as PARAMS


# -------------------------
# Utility functions (adapted for PyTorch)
# -------------------------


def conv3d_pytorch_same_padding(input_volume_np, kernel_np):
    """
    Performs 3D convolution of a 3D input volume with a 3D kernel using PyTorch,
    with 'same' padding to ensure the output volume has the same shape as the input volume.

    Args:
        input_volume_np (numpy.ndarray): A 3D NumPy array representing the input volume (shape: (depth, height, width)).
        kernel_np (numpy.ndarray): A 3D NumPy array representing the convolution kernel (shape: (kernel_depth, kernel_height, kernel_width)).

    Returns:
        numpy.ndarray: A 3D NumPy array representing the convolved output volume with 'same' padding (shape: same as input_volume_np).
    """
    # Convert NumPy arrays to PyTorch tensors
    # PyTorch conv3d expects input: (N, C_in, D_in, H_in, W_in)
    # PyTorch conv3d expects kernel: (C_out, C_in, D_k, H_k, W_k)

    # Add batch and channel dimensions to input (1, 1, D, H, W)
    input_tensor = torch.from_numpy(input_volume_np).float().unsqueeze(0).unsqueeze(0)

    # Add input and output channel dimensions to kernel (1, 1, D_k, H_k, W_k)
    kernel_tensor = torch.from_numpy(kernel_np).float().unsqueeze(0).unsqueeze(0)

    # Get dimensions
    input_depth, input_height, input_width = input_volume_np.shape
    kernel_depth, kernel_height, kernel_width = kernel_np.shape

    # Calculate padding for 'same' convolution
    # Pytorch's padding is for (W, H, D)
    pad_d = kernel_depth // 2
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    output_tensor = F.conv3d(
        input=input_tensor,
        weight=kernel_tensor,
        padding=(pad_d, pad_h, pad_w),  # PyTorch padding is (D, H, W)
    )

    # Remove batch and channel dimensions and convert back to NumPy array
    # output_tensor shape will be (1, 1, D, H, W)
    output_volume = output_tensor.squeeze().numpy()

    return output_volume


def fix_roi6_randomize_weights_and_swap_rows(constraints):  #
    """
    Fixes ROI6 values, randomizes weights with specific ranges, and swaps the entire row of constraint values
    between other keys using index shuffling.

    Args:
        constraints (dict): The dictionary containing the constraints.

    Returns:
        dict: The modified constraints dictionary.

    the loss appears to be unstable if weight is random
    """
    rois_to_swap = [roi for roi in constraints["weight"].keys() if roi != "ROI6"]
    constraint_types_to_swap = [
        "lower_bound_gy",
        "higher_bound_gy",
        "lower_bound_target_percent",
        "higher_bound_target_percent",
        "weight",
    ]

    # Randomize weights (excluding ROI6) with specific ranges
    for roi in rois_to_swap:
        if roi == "PTV":
            constraints["weight"][roi] = random.randint(10, 1000)
        else:
            constraints["weight"][roi] = random.randint(1, 10)

    # Swap the whole row of values between ROIs (excluding ROI6) using index shuffling
    num_rois_to_swap = len(rois_to_swap)
    indices = list(range(num_rois_to_swap))
    random.shuffle(indices)

    for i in range(num_rois_to_swap):
        # Get the original ROI name using the current index
        roi1 = rois_to_swap[i]
        # Get the ROI name to swap with using the shuffled index
        roi2_index = indices[i]
        roi2 = rois_to_swap[roi2_index]

        # Perform the swap for all constraint types
        for constraint_type in constraint_types_to_swap:
            temp_value = constraints[constraint_type][roi1]
            constraints[constraint_type][roi1] = constraints[constraint_type][roi2]
            constraints[constraint_type][roi2] = temp_value

    return constraints


def fix_roi6_and_swap_rows(constraints):  #
    """
    Fixes ROI6 values, randomizes weights with specific ranges, and swaps the entire row of constraint values
    between other keys using index shuffling.

    Args:
        constraints (dict): The dictionary containing the constraints.

    Returns:
        dict: The modified constraints dictionary.
    """
    rois_to_swap = [roi for roi in constraints["weight"].keys() if roi != "ROI6"]
    constraint_types_to_swap = [
        "lower_bound_gy",
        "higher_bound_gy",
        "lower_bound_target_percent",
        "higher_bound_target_percent",
        "weight",
    ]

    # Swap the whole row of values between ROIs (excluding ROI6) using index shuffling
    num_rois_to_swap = len(rois_to_swap)
    indices = list(range(num_rois_to_swap))
    random.shuffle(indices)

    for i in range(num_rois_to_swap):
        # Get the original ROI name using the current index
        roi1 = rois_to_swap[i]
        # Get the ROI name to swap with using the shuffled index
        roi2_index = indices[i]
        roi2 = rois_to_swap[roi2_index]

        # Perform the swap for all constraint types
        for constraint_type in constraint_types_to_swap:
            temp_value = constraints[constraint_type][roi1]
            constraints[constraint_type][roi1] = constraints[constraint_type][roi2]
            constraints[constraint_type][roi2] = temp_value

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


class DataGenerator(
    Dataset
):  # Changed from keras.utils.Sequence to torch.utils.data.Dataset
    def __init__(
        self,
        data_path,
        cohort,
        transform=None,  # <--- new: pass a torchio.Compose(...) if you want augmentation
        transform_mask=None,
        shuffle=False,
        batch_size=1,
        synthetic_dose=False,
        downsampling_factor=1,
        is_debug=0,
        constraints=PARAMS.constraints,
        is_return_gaussian_ptv_oars=False,
        weight_ptv=None,
        constraint_mode="fixed",
        is_normalize_weight=False,
        verbose=False,
        is_return_file_name=False,
    ):
        self.synthetic_dose = synthetic_dose
        self.data_path = data_path
        self.downsampling_factor = downsampling_factor
        files = os.listdir(data_path)
        self.file_list = [file for file in files if file.endswith(".npz")]
        self.file_list.sort()
        self.cohort = cohort
        if cohort == "training":
            self.file_list = [
                file for file in self.file_list if (not (file.startswith("val_")))
            ]
            if is_debug == 1:
                self.file_list = self.file_list[:100]
            elif is_debug == 2:
                self.file_list = self.file_list[:1]
            else:
                pass
        elif cohort == "validating":
            self.file_list = [
                file for file in self.file_list if (file.startswith("val_"))
            ]
            if is_debug == 1:
                self.file_list = self.file_list[:10]
            elif is_debug == 2:
                self.file_list = self.file_list[:1]
            else:
                pass
        elif cohort == "plotting":
            self.file_list = [
                file for file in self.file_list if (file.startswith("val_"))
            ][0:1]
        elif cohort == "all":  # generate dose samples
            pass

        self.indexes = np.arange(len(self.file_list))
        self.shuffle = shuffle  # Dataloader handles this, but kept for on_epoch_end
        self.batch_size = batch_size  # Will be overridden by DataLoader's batch_size
        self.is_return_gaussian_ptv_oars = is_return_gaussian_ptv_oars
        self.constraints_template = copy.deepcopy(
            constraints
        )  # Store original constraints
        self.weight_ptv = weight_ptv  # control weight here for the "ptv"
        self.constraint_mode = constraint_mode
        self.is_normalize_weight = is_normalize_weight
        self.verbose = verbose
        self.is_return_file_name = is_return_file_name

        self.epoch_number = 0  # Initialize epoch number

        self.gaussian_kernel_3d = self.gaussian_kernel_3d(size=9, sigma=1.5)
        # self.on_epoch_end() # Removed as DataLoader handles shuffling per epoch

        print(f"Number of files: {len(self.file_list)} in {cohort} cohort")

    def gaussian_kernel_3d(self, size=9, sigma=1.5):  #
        """Create a normalized 3D Gaussian kernel."""
        ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
        xx, yy, zz = np.meshgrid(ax, ax, ax)  # Create 3D grid
        kernel = np.exp(
            -(xx**2 + yy**2 + zz**2) / (2 * sigma**2)
        )  # 3D Gaussian formula
        return kernel / np.sum(kernel)  # Normalize the 3D kernel

    def __len__(self):
        return len(self.file_list)  # Length is simply the number of files

    def __getitem__(self, index):
        # Data generation for a single item
        file_ID = self.file_list[self.indexes[index]]  # Use self.indexes for shuffling
        return self._data_generation_single(file_ID)

    def on_epoch_end(self):
        # This method is typically called by a DataLoader if `shuffle=True`
        # in the DataLoader itself. For the Dataset, we just update the internal indexes.
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def set_epoch(self, epoch):  #
        self.epoch_number = epoch

    def create_bound_weight_matrix(self, structures, bound, factor=100):  #
        """
        Creates a dose matrix based on the provided structures and bounds.

        Args:
            structures: Dictionary mapping structure names to NumPy arrays.
                    Example: {'PTV': ptv_array, 'ROI1': oar_penile_array, ...}
            bound: Dictionary containing bound values for each structure.
                Example: {'PTV': 70, 'ROI1': 50, ...}

        Returns:
            NumPy array representing the combined dose matrix (normalized to 0-1).
        """
        # Use the first structure to determine shape for initialization
        first_structure = next(iter(structures.values()))
        bound_matrix = np.zeros_like(first_structure, dtype=np.float32)

        # Apply bounds for each structure
        for structure_id, array in structures.items():
            if structure_id in bound:
                bound_matrix += array * bound[structure_id]

        # Normalize to 0-1 range
        return bound_matrix / factor

    def downsample_ct(self, ct_data, downsample_factor=2, sigma=1.0):
        """
        Downsample a 3D CT volume by a specified factor using Gaussian smoothing.

        Args:
            ct_data (np.ndarray): 3D CT volume with shape [W, D, H]
            downsample_factor (int): Factor to downsample by (default: 2)
            sigma (float): Standard deviation for Gaussian filter (default: 1.0)

        Returns:
            np.ndarray: Downsampled CT volume with shape [W//downsample_factor, D//downsample_factor, H//downsample_factor]
        """
        # Apply Gaussian smoothing to prevent aliasing
        smoothed_ct = gaussian_filter(ct_data, sigma=sigma, mode="constant")

        # Downsample by selecting every nth voxel
        downsampled_ct = smoothed_ct[
            :: downsample_factor[0], :: downsample_factor[1], :: downsample_factor[2]
        ]

        return downsampled_ct

    def downsample_ct_new(self, ct_data, downsample_factor=2, order=3):
        """
        Downsample CT with interpolation (cubic by default).

        Args:
            ct_data (np.ndarray): 3D CT volume [W, D, H].
            downsample_factor (int): Downsampling factor.
            order (int): Spline interpolation order (0=nearest, 1=linear, 3=cubic).

        Returns:
            np.ndarray: Resampled CT.
        """
        zoom_factors = np.divide(1, downsample_factor)
        return scipy.ndimage.zoom(ct_data, zoom_factors, order=order)

    def _data_generation_single(self, file_ID):
        file_path = self.data_path + file_ID

        # constraints logic
        epoch = self.epoch_number  #
        p_augment = min(epoch * 0.1, 1.0)  #

        if self.verbose and self.constraint_mode != "fixed":
            print("epoch: ", epoch, "\t p_augment: ", p_augment)

        if self.constraint_mode == "fixed":
            if self.verbose:
                print("fixed")
            constraints = copy.deepcopy(self.constraints_template)  # Use template
        elif self.constraint_mode == "rconstraint":
            if random.random() < p_augment:
                if self.verbose:
                    print("rconstraint -- random")
                constraints = copy.deepcopy(random.choice(PARAMS.list_constraints))
            else:
                if self.verbose:
                    print("constraint -- not random")
                constraints = copy.deepcopy(self.constraints_template)
            if self.weight_ptv is not None:
                for key, value in constraints["weight"].items():
                    if value == 10:
                        constraints["weight"][key] = self.weight_ptv
        elif self.constraint_mode == "rweight":
            if random.random() < p_augment:
                if self.verbose:
                    print("rweight -- random")
                constraints = randomize_weights(
                    copy.deepcopy(self.constraints_template)
                )
            else:
                if self.verbose:
                    print("rweight -- not random")
                constraints = copy.deepcopy(self.constraints_template)

        if self.is_normalize_weight:
            constraints = normalize_weights(constraints)
            if self.verbose:
                print("normalize_weights:")
                print(json.dumps(constraints, indent=4))

        # data loading
        npzdata = None
        while npzdata is None:
            try:
                with np.load(file_path, allow_pickle=True) as npzdata_load:
                    npzdata = {key: npzdata_load[key] for key in npzdata_load.files}
            except Exception as e:
                time.sleep(random.random())

        ct = np.interp(npzdata["CT"], (0, 255), (-1, 1)).astype(np.float32)
        ptv = npzdata["PTV"].astype(np.float32)
        oar_penile = npzdata["ROI1"].astype(np.float32)
        oar_fem_l = npzdata["ROI2"].astype(np.float32)
        oar_fem_r = npzdata["ROI3"].astype(np.float32)
        oar_bladder = npzdata["ROI4"].astype(np.float32)
        oar_rectum = npzdata["ROI5"].astype(np.float32)
        background = npzdata["ROI6"].astype(np.float32)

        if self.downsampling_factor and self.downsampling_factor != (1, 1, 1):
            factor = self.downsampling_factor
            # Downsample continuous-valued CT image
            # ct = scipy.ndimage.zoom(ct, zoom=zoom_factor, order=3, mode="nearest")
            ct = self.downsample_ct(ct, factor)
            zoom_factor = np.divide(1, factor)
            # Downsample binary masks with nearest-neighbor interpolation
            ptv = scipy.ndimage.zoom(ptv, zoom=zoom_factor, order=0, mode="nearest")
            oar_penile = scipy.ndimage.zoom(
                oar_penile, zoom=zoom_factor, order=0, mode="nearest"
            )
            oar_fem_l = scipy.ndimage.zoom(
                oar_fem_l, zoom=zoom_factor, order=0, mode="nearest"
            )
            oar_fem_r = scipy.ndimage.zoom(
                oar_fem_r, zoom=zoom_factor, order=0, mode="nearest"
            )
            oar_bladder = scipy.ndimage.zoom(
                oar_bladder, zoom=zoom_factor, order=0, mode="nearest"
            )
            oar_rectum = scipy.ndimage.zoom(
                oar_rectum, zoom=zoom_factor, order=0, mode="nearest"
            )
            background = scipy.ndimage.zoom(
                background, zoom=zoom_factor, order=0, mode="nearest"
            )

            for mask in [
                ptv,
                oar_penile,
                oar_fem_l,
                oar_fem_r,
                oar_bladder,
                oar_rectum,
                background,
            ]:
                assert np.all(
                    np.logical_or(mask == 0, mask == 1)
                ), "Mask contains non-binary values"

            # Check that the sum of all masks equals 1 everywhere
            masks = [
                ptv,
                oar_penile,
                oar_fem_l,
                oar_fem_r,
                oar_bladder,
                oar_rectum,
                background,
            ]
            mask_sum = np.sum(masks, axis=0)
            assert np.all(
                mask_sum == 1
            ), f"Mask sum is not 1 everywhere: min={mask_sum.min()}, max={mask_sum.max()}"

        gaussian_ptv_oar = None
        if self.is_return_gaussian_ptv_oars:
            gaussian_ptv_oar = conv3d_pytorch_same_padding(  # Changed to PyTorch conv
                input_volume_np=ptv
                + oar_penile
                + oar_fem_l
                + oar_fem_r
                + oar_bladder
                + oar_rectum,
                kernel_np=self.gaussian_kernel_3d,
            )

        structures = {  #
            "PTV": ptv,
            "ROI1": oar_penile,
            "ROI2": oar_fem_l,
            "ROI3": oar_fem_r,
            "ROI4": oar_bladder,
            "ROI5": oar_rectum,
            "ROI6": background,
        }

        # lower and higher bounds gy
        lower_bound_gy = self.create_bound_weight_matrix(
            structures, bound=constraints["lower_bound_gy"]
        )
        higher_bound_gy = self.create_bound_weight_matrix(
            structures, bound=constraints["higher_bound_gy"]
        )

        # lower and higher bounds percent target
        lower_bound_target_percent = self.create_bound_weight_matrix(
            structures, bound=constraints["lower_bound_target_percent"]
        )
        higher_bound_target_percent = self.create_bound_weight_matrix(
            structures,
            bound=constraints["higher_bound_target_percent"],
        )

        region_weight = self.create_bound_weight_matrix(
            structures,
            bound=constraints["weight"],
            factor=1,
        )

        # Stack input features (CT and constraint maps)
        # PyTorch expects (C, D, H, W) for convolutions
        images = np.stack(
            [
                ct,
                lower_bound_gy,
                higher_bound_gy,
                lower_bound_target_percent,
                higher_bound_target_percent,
                region_weight,
            ],
            axis=0,  # Stack along a new channel dimension
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
            axis=0,  # Stack along a new channel dimension
        )

        dose = np.zeros_like(ct)
        if "Dose" in npzdata:
            dose = npzdata["Dose"]
        if "dose" in npzdata:
            dose = 50.0 * npzdata["dose"] / 255
        if dose.size == 1:
            dose = np.zeros_like(ct)

        # Convert all NumPy arrays to PyTorch tensors
        images_tensor = torch.from_numpy(images).float()
        dose_tensor = torch.from_numpy(dose).float()
        masks_tensor = torch.from_numpy(mask).float()
        region_weights_tensor = torch.from_numpy(region_weight).float()

        if self.is_return_gaussian_ptv_oars:
            gaussian_ptv_oars_tensor = torch.from_numpy(gaussian_ptv_oar).float()
            if self.is_return_file_name:
                return (
                    images_tensor,
                    dose_tensor,
                    masks_tensor,
                    region_weights_tensor,
                    constraints,  # Constraints dict stays as dict
                    gaussian_ptv_oars_tensor,
                    file_ID,  # Use file_ID directly for consistency
                )
            else:
                return (
                    images_tensor,
                    dose_tensor,
                    masks_tensor,
                    region_weights_tensor,
                    constraints,
                    gaussian_ptv_oars_tensor,
                )
        else:
            if self.is_return_file_name:
                return (
                    images_tensor,
                    dose_tensor,
                    masks_tensor,
                    region_weights_tensor,
                    constraints,
                    file_ID,
                )
            else:
                return (
                    images_tensor,
                    dose_tensor,
                    masks_tensor,
                    region_weights_tensor,
                    constraints,
                )


# Example Usage (adapted for PyTorch DataLoader):
if __name__ == "__main__":
    # Instantiate the Dataset
    train_dataset = DataGenerator(
        "database/AUTORPT/",
        "training",
        shuffle=True,  # Shuffle here for the DataLoader
        batch_size=1,  # This will be ignored by DataLoader
        constraints=PARAMS.constraints,
        is_debug=0,
        weight_ptv=1000,
        # constraint_mode="rweight",  # Using rweight from original example
        constraint_mode="fixed",
        downsampling_factor=2,
        verbose=1,
        is_normalize_weight=True,  # Example from original
    )

    # Instantiate the DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,  # Define your desired batch size here
        shuffle=True,  # Important for training; calls on_epoch_end internally
        num_workers=0,  # Set to >0 for multiprocessing data loading
        pin_memory=True,  # For faster GPU transfers
    )

    print(f"\nNumber of batches per epoch: {len(train_loader)}")

    for epoch in range(20):
        print(f"\n--- Epoch {epoch+1} ---")
        train_dataset.set_epoch(epoch)  # Call set_epoch for constraint logic

        for i, batch_data in enumerate(train_loader):
            images, y_dose, masks, region_weights, constraints = batch_data

            print(f"Batch {i+1}:")
            print("  images shape:", images.shape)  # (Batch, C, D, H, W)
            print("  y_dose shape:", y_dose.shape)  # (Batch, D, H, W)
            print("  masks shape:", masks.shape)  # (Batch, C, D, H, W)
            print("  region_weights shape:", region_weights.shape)  # (Batch, D, H, W)
            # Constraints is now a list of dicts or a dict depending on collate_fn
            # Default collate_fn will make it a list of dicts for `constraints`
            # print("  constraints (first item):", json.dumps(constraints[0], indent=4))

            # Break after a few batches for demonstration
            if i >= 1:
                break
        if epoch >= 0:  # Only run one epoch for quick test
            break
