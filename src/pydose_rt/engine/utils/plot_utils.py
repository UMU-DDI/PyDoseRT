import numpy as np
import torch
from PIL import Image
import io
import base64
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from scipy import ndimage as ndi
import SimpleITK as sitk
from typing import Union, List, Tuple


import engine.utils.path_utils as path_utils
from engine.config import config as PARAMS

from engine.utils.utils import (
    compute_valid_leaf_mask_minh,
    compute_valid_leaf_mask,
    animate_ct_and_doses,
)
from engine.utils.mask_utils import get_body_mask_from_normalized_ct
from DoseEngines.DoseEngine import DoseEngine


# Function to rotate a NumPy array
def rotate_numpy_array(np_array, angle):
    return np.array(
        Image.fromarray(np_array).rotate(angle, resample=Image.BICUBIC, expand=True)
    )


# Helper function to convert NumPy array to a Plotly-compatible image
def numpy_to_base64_image(np_array, cmap="gray"):
    colormapped_data = plt.cm.get_cmap(cmap)(np_array)

    colormapped_image = (colormapped_data * 255).astype(np.uint8)
    # Convert NumPy array to an image using Pillow
    pil_image = Image.fromarray(colormapped_image)

    # Save the image to a BytesIO object and encode as base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{base64_image}"


class Plotter:
    """
    Make plots for DVH and dose distribution gif.
    """

    def __init__(
        self,
        dose_engine_config,
        gen_val,
        experiment,
        number_of_cps,
        num_fluence_plots,
        dose_engine,
        fluence_layer=None,
        ## Added by Minh
        scaled_mu=1,  # for Minh scaled mu from 0-1 to get dose 0-1. Now times 100 to get correct
        out_path="database/log_plots",
        ## Added by Minh
    ):
        self.dose_engine_config = dose_engine_config
        self.experiment = experiment
        self.number_of_cps = number_of_cps
        self.gen_val = gen_val
        self.idx = 0
        self.num_fluence_plots = num_fluence_plots
        self.sz = 2
        self.fps = 16
        self.fluence_layer = fluence_layer

        ## Added by Minh
        self.roi_colors = PARAMS.roi_colors
        self.scaled_mu = scaled_mu
        self.dose_engine = dose_engine
        ## Added by Minh

        self.out_path = out_path

        path_utils.make_dir(self.out_path)

    def create_dvh(self, dose, epoch, text_constraint=None, is_without_model=False):
        dose = dose * 100  # scale 0-1 to 0-100

        if is_without_model:
            # Correct way to create a figure and a single subplot (Axes object)
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            # Correct way for the other case as well
            fig, ax = plt.subplots(figsize=(10, 6))

        for idx, (key, value) in enumerate(PARAMS.structure_names.items()):
            roi_name = value
            roi = self.masks[idx]

            # Extract dose values

            dose_values = dose[roi > 0.0]  # scale 0-1 to 0-100

            # Compute the histogram
            bins = np.linspace(0, np.max(dose), 1000)
            hist, bin_edges = np.histogram(dose_values, bins=bins, density=False)

            # Compute the cumulative sum (cumulative histogram)
            cumulative_hist = np.cumsum(hist[::-1])[::-1]  # Reverse cumulative sum
            cumulative_hist_normalized = (
                cumulative_hist / cumulative_hist.max()
            )  # Normalize

            # Plot the DVH
            plt.plot(
                bin_edges[:-1],
                cumulative_hist_normalized,
                label=roi_name,
                color=self.roi_colors[key],
            )

        # if not is_without_model:
        plt.xlabel("Dose (Gy)")
        plt.ylabel("Volume Fraction")
        plt.title("Dose Volume Histogram (DVH)")
        plt.grid(True)

        # else:
        #     # Remove x-axis labels and ticks
        #     ax.set_xticks([])
        #     ax.set_xticklabels([])

        #     # Remove y-axis labels and ticks
        #     ax.set_yticks([])
        #     ax.set_yticklabels([])

        #     # Remove the plot frame (spines)
        #     ax.spines["top"].set_visible(False)
        #     ax.spines["right"].set_visible(False)
        #     ax.spines["bottom"].set_visible(False)
        #     ax.spines["left"].set_visible(False)

        plt.legend()
        if text_constraint is None:
            figpath = f"{self.out_path}/dvh_{epoch:02d}.png"
        else:
            figpath = f"{self.out_path}/dvh_{epoch:02d}_{text_constraint}.png"

        plt.savefig(figpath)

        if self.experiment is not None:
            self.experiment.log_image(figpath, step=epoch, overwrite=False)

    def create_animation(
        self,
        ct_array,
        dose_array,
        epoch,
        text_constraint=None,
        is_show_contour=True,
    ):
        # animate_ct_and_doses(
        #     ct_np=ct_array,
        #     dose_list=[
        #         dose_array,
        #     ],
        #     out_path=f"database/temp/dose_movie_batch0.mp4",
        #     fps=10,
        #     dpi=80,
        # )

        # List to store frames for the GIF
        frames = []
        num_slices = ct_array.shape[-1]

        # find mask
        mask_img_or_np, intermediates = get_body_mask_from_normalized_ct(ct_array)
        final_mask = intermediates["final_mask"]
        dose_array = dose_array * final_mask

        # OPTIMIZATION 2: Reduce the figure size
        fig_size = (6, 6)  # Smaller figure size

        for i in range(num_slices):
            fig, ax = plt.subplots(figsize=fig_size, dpi=50)
            im_ct = ax.imshow(
                ct_array[:, :, i], cmap="gray", vmin=-1, vmax=1, alpha=0.3
            )
            dose_norm = dose_array[:, :, i]
            # im_dose = ax.imshow(dose_norm, cmap="jet", vmin=0, vmax=1, alpha=0.2)
            # im_dose = ax.imshow(dose_norm, cmap="jet", vmin=0, vmax=100, alpha=1.0)
            im_dose = ax.imshow(dose_norm, cmap="jet", vmin=0, vmax=1, alpha=0.6)

            # Plot PTV using the custom color
            ptv_mask = self.masks[0]
            if not is_show_contour:
                mask_display = np.ma.masked_where(
                    ptv_mask[:, :, i] == 0, ptv_mask[:, :, i]
                )
                ptv_cmap = ListedColormap([self.roi_colors["PTV"]])
                im_mask = ax.imshow(
                    mask_display, cmap=ptv_cmap, alpha=1.0, vmin=0, vmax=1
                )
            else:
                ax.contour(
                    ptv_mask[:, :, i],
                    levels=[0.5],
                    colors=[self.roi_colors["PTV"]],
                    linewidths=4.0,
                    alpha=1.0,
                    zorder=5,
                )

            # Plot other ROIs (indices 1 to 5 correspond to ROI1 to ROI5)
            roi_keys = ["ROI1", "ROI2", "ROI3", "ROI4", "ROI5"]
            for j, key in enumerate(roi_keys, start=1):
                other_mask = self.masks[j]
                if not is_show_contour:
                    mask_display_other = np.ma.masked_where(
                        other_mask[:, :, i] == 0, other_mask[:, :, i]
                    )
                    roi_cmap = ListedColormap([self.roi_colors[key]])
                    im_mask_other = ax.imshow(
                        mask_display_other, cmap=roi_cmap, alpha=1.0, vmin=0, vmax=1
                    )
                else:
                    ax.contour(
                        other_mask[:, :, i],
                        levels=[0.5],
                        colors=[self.roi_colors[key]],
                        linewidths=1.0,
                        alpha=1.0,
                        zorder=5,
                    )

            ax.axis("off")
            fig.tight_layout(pad=0)

            # Convert the current figure to a numpy array (RGB image)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
            plt.close(fig)

        # Save the frames as an MP4 video
        if text_constraint is None:
            figpath = f"{self.out_path}/dose_{epoch:02d}.mp4"
        else:
            figpath = f"{self.out_path}/dose_{epoch:02d}_{text_constraint}.mp4"

        fps = max(1, int(num_slices / 15))
        fps = 20
        fps = 1
        with imageio.get_writer(
            figpath, format="FFMPEG", fps=fps, codec="libx264"
        ) as writer:
            for frame in frames:
                writer.append_data(frame)

        if self.experiment is not None:
            self.experiment.log_video(figpath, step=epoch, overwrite=False)

    def create_leafs_animation(self, data, epoch, text_constraint=None):
        try:
            leafs = data["leafs"].numpy()
        except:
            leafs = data["leafs"]

        if self.dose_engine == "minh":
            leafs_left = leafs[:, :, 0]  # Directly select the control point data
            leafs_right = leafs[:, :, 1]  # Directly select the control point data
        else:
            center, width = leafs[:, :, 0], leafs[:, :, 1]
            leafs_left = center - 0.5 * width
            leafs_right = center + 0.5 * width

        # Determine global x-axis limits.
        # margin = 1.0  # extra margin
        # global_left = np.min(leafs_left) - margin
        # global_right = np.max(leafs_right) + margin

        margin = 1.0  # extra margin
        global_left = 0
        global_right = 1

        n_cp = leafs_left.shape[0]
        n_leaves = leafs_left.shape[1]

        fps = max(1, int(n_cp / 15))
        frames = []

        fig_size = (6, 6)  # Smaller figure size

        for cp in range(n_cp):
            fig, ax = plt.subplots(figsize=fig_size, dpi=50)
            for i in range(n_leaves):
                y_center = i
                height = 0.8  # Thickness of the bar for visualization

                left_edge = leafs_left[cp, i]
                right_edge = leafs_right[cp, i]

                # Left
                ax.add_patch(
                    plt.Rectangle(
                        (global_left, y_center - height / 2),
                        left_edge - global_left,
                        height,
                        color="black",
                    )
                )
                # Right
                ax.add_patch(
                    plt.Rectangle(
                        (right_edge, y_center - height / 2),
                        global_right - right_edge,
                        height,
                        color="black",
                    )
                )

                # dashed lines to indicate the full field width.
                ax.plot(
                    [global_left, global_right],
                    [y_center - height / 2, y_center - height / 2],
                    color="gray",
                    linestyle="--",
                    linewidth=0.5,
                )
                ax.plot(
                    [global_left, global_right],
                    [y_center + height / 2, y_center + height / 2],
                    color="gray",
                    linestyle="--",
                    linewidth=0.5,
                )

            ax.set_xlim(global_left, global_right)
            ax.set_ylim(-1, n_leaves)
            ax.set_xlabel("Position")
            ax.set_ylabel("Leaf Pair Index")
            ax.set_title(f"Control Point {cp + 1}")
            ax.invert_yaxis()

            fig.tight_layout(pad=0)

            # Convert the current figure to a numpy array (RGB image)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
            plt.close(fig)

        # Save the frames as an MP4 video
        if text_constraint is None:
            figpath = f"{self.out_path}/leafs_{epoch:02d}.mp4"
        else:
            figpath = f"{self.out_path}/leafs_{epoch:02d}_{text_constraint}.mp4"

        with imageio.get_writer(
            figpath, format="FFMPEG", fps=fps, codec="libx264"
        ) as writer:
            for frame in frames:
                writer.append_data(frame)

        if self.experiment is not None:
            self.experiment.log_video(figpath, step=epoch, overwrite=False)

    def plot_leafs_at_cp(
        self,
        data,
        epoch,
        cp=0,
        text_constraint=None,
        fig_size=(6, 6),
    ):
        """
        Plot a single MLC leaf configuration at a specified control point.

        Args:
            data (dict): Dictionary containing 'leafs' (np.ndarray or torch.Tensor)
                        [n_cp, n_leafs, 2] array with left/right positions.
            epoch (int): Current epoch number.
            cp (int): Index of the control point to visualize.
            text_constraint (str or None): Optional text to add to the filename.
            fig_size (tuple): Size of the output figure.
        """
        try:
            leafs = data["leafs"].numpy()
        except AttributeError:  # More specific exception for torch.Tensor
            leafs = data["leafs"]  # Assume it's already a NumPy array if .numpy() fails

        if self.dose_engine == "minh":
            leafs_left = leafs[cp, :, 0]  # Directly select the control point data
            leafs_right = leafs[cp, :, 1]  # Directly select the control point data
        else:
            center, width = leafs[cp, :, 0], leafs[cp, :, 1]
            leafs_left = center - 0.5 * width
            leafs_right = center + 0.5 * width

        global_left = 0
        global_right = 1
        n_leaves = leafs_left.shape[
            0
        ]  # Get n_leaves from the shape of the cp-specific data

        # n_cp check is removed as we're directly indexing leafs[cp, :, :]
        # If data['leafs'] itself is just [n_leafs, 2], then n_cp=1 and cp must be 0

        fig, ax = plt.subplots(figsize=fig_size, dpi=100)
        height = 0.8

        for i in range(n_leaves):
            y_center = i
            left_edge = leafs_left[i]
            right_edge = leafs_right[i]

            # Left block
            ax.add_patch(
                plt.Rectangle(
                    (global_left, y_center - height / 2),
                    left_edge - global_left,
                    height,
                    color="black",
                )
            )
            # Right block
            ax.add_patch(
                plt.Rectangle(
                    (right_edge, y_center - height / 2),
                    global_right - right_edge,
                    height,
                    color="black",
                )
            )

            # Dashed lines to mark field height
            ax.plot(
                [global_left, global_right],
                [y_center - height / 2] * 2,
                color="gray",
                linestyle="--",
                linewidth=0.5,
            )
            ax.plot(
                [global_left, global_right],
                [y_center + height / 2] * 2,
                color="gray",
                linestyle="--",
                linewidth=0.5,
            )

        ax.set_xlim(global_left, global_right)
        ax.set_ylim(-1, n_leaves)

        # --- Changes Start Here ---

        # Remove x-axis labels and ticks
        ax.set_xticks([])
        ax.set_xticklabels([])

        # Remove y-axis labels and ticks
        ax.set_yticks([])
        ax.set_yticklabels([])

        # Remove the plot frame (spines)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # --- Changes End Here ---

        # Keep title if you still want it inside the plot area
        # ax.set_title(f"MLC Leaf Configuration at CP {cp + 1}", fontsize=12)

        fig.tight_layout(pad=0)  # pad=0 helps remove extra whitespace around the plot

        if text_constraint is None:
            figpath = f"{self.out_path}/leafs_{epoch:02d}_{cp:02d}.png"
        else:
            figpath = (
                f"{self.out_path}/leafs_{epoch:02d}_{cp:02d}_{text_constraint}.png"
            )

        plt.savefig(figpath)
        plt.close(fig)  # Close the figure to free up memory

    def create_mus(self, data, epoch, text_constraint=None):
        try:
            mus = data["mus"].numpy()
        except:
            mus = data["mus"]

        mus = mus[:]

        # plt.figure(figsize=(10, 6))  # Adjust figure size as needed
        plt.figure(figsize=(6, 6))  # Adjust figure size as needed
        plt.plot(mus, color="black")
        plt.xlabel("Control point")
        plt.ylabel("MU")
        plt.title("Monitoring Unit")
        # plt.grid(True)
        plt.legend()

        if text_constraint is None:
            figpath = f"{self.out_path}/mu_{epoch:02d}.png"
        else:
            figpath = f"{self.out_path}/mu_{epoch:02d}_{text_constraint}.png"

        plt.savefig(figpath)

        if self.experiment is not None:
            self.experiment.log_image(figpath, step=epoch, overwrite=False)

    def make(self, model, epoch, text_constraint=None):
        self.x, self.y, self.masks, self.region_weights, self.constraints, _ = next(
            iter(self.gen_val)
        )

        # Ensure self.x is on the same device as the model
        device = next(model.parameters()).device
        self.x = self.x.to(device)
        # If you use self.masks, self.region_weights, etc. in model, move them too

        if self.dose_engine == "minh":
            valid_leaf = compute_valid_leaf_mask_minh(
                self.masks[:, 0],
                self.dose_engine_config,
            )
            self.dose, leafs, mus, dose_bypass = model(self.x, valid_leaf.to(device))
        elif self.dose_engine == "matthias":
            # valid_leaf = compute_valid_leaf_mask_attila(
            #     self.masks[:, 0],
            #     self.dose_engine_config,
            # )
            # self.dose, leafs, mus, dose_bypass = model(self.x, valid_leaf.to(device))

            valid_leaf = compute_valid_leaf_mask(
                "matthias",
                dose_model=DoseEngine(
                    self.dose_engine_config,
                    kernel_size=15,
                    permute_ct=True,
                    leafs_centered=True,
                ),
                ct=self.x[:, 0, :, :, :].to(device),
                ptv_mask=self.masks[:, 0, :, :, :].to(device),
                n_cps=int(self.dose_engine_config.number_of_cps),
                n_leafs=self.dose_engine_config.number_of_leaf_pairs,
            )
            self.dose, leafs, mus, dose_bypass = model(
                self.x,
                valid_leaf.to(device),
            )

        data_dict = {
            "dose": self.dose[0].detach().cpu().numpy(),
            "leafs": leafs[0].detach().cpu().numpy(),
            "mus": mus[0].detach().cpu().numpy(),
            "dose_bypass": dose_bypass[0].detach().cpu().numpy(),
            "x": self.x[0].detach().cpu().numpy(),
            "masks": self.masks[0].detach().cpu().numpy(),
        }
        self.data = data_dict
        self.dose = self.dose[0].detach().cpu().numpy()
        self.leafs = leafs[0].detach().cpu().numpy()
        self.mus = mus[0].detach().cpu().numpy()
        self.dose_bypass = dose_bypass[0].detach().cpu().numpy()
        self.x = self.x[0].detach().cpu().numpy()
        self.masks = self.masks[0].detach().cpu().numpy()
        self.ct = np.array(self.x[0])

        if text_constraint is None:
            data_dict_path = f"{self.out_path}/data_{epoch:02d}.npz"
        else:
            data_dict_path = f"{self.out_path}/data_{epoch:02d}_{text_constraint}.npz"

        np.savez_compressed(data_dict_path, **data_dict)

        print(">> create_leafs_animation")
        self.create_leafs_animation(self.data, epoch, text_constraint)

        print(">> create_mus")
        self.create_mus(self.data, epoch, text_constraint)

        print(">> create_dvh")
        self.create_dvh(self.dose, epoch, text_constraint)

        print(">> create_animation")
        self.create_animation(self.ct, self.dose, epoch, text_constraint)

    def make_without_model_gen(
        self,
        x,
        masks,
        dose,
        leafs,
        mus,
        dose_bypass,
        epoch,
        file_name,
        text_constraint=None,
    ):
        data_dict = {
            "dose": dose,
            "leafs": leafs,
            "mus": mus,
            "dose_bypass": dose_bypass,
            "x": x,
            "masks": masks,
        }
        self.data = data_dict
        self.dose = dose
        self.masks = masks

        if text_constraint is None:
            data_dict_path = f"{self.out_path}/data_{epoch:02d}.npz"
        else:
            data_dict_path = f"{self.out_path}/data_{epoch:02d}_{text_constraint}.npz"

        np.savez_compressed(data_dict_path, **data_dict)

        print(">> create_leafs_animation")
        self.create_leafs_animation(self.data, epoch, text_constraint)

        print(">> leafs at cp")
        self.plot_leafs_at_cp(self.data, epoch, cp=0, text_constraint=text_constraint)

        print(">> create_mus")
        self.create_mus(self.data, epoch, text_constraint)

        print(">> create_dvh")
        self.create_dvh(self.dose, epoch, text_constraint, is_without_model=True)

        # ct = np.array(x[0, ..., 0])
        # dose = np.array(dose)[0, ...]

        ct = np.array(x[0])

        print(">> create_animation")
        self.create_animation(ct, dose, epoch, text_constraint)


if __name__ == "__main__":
    p = Plotter(
        dose_engine_config=None,
        gen_val=None,
        experiment=None,
        number_of_cps=None,
        fluence_layer=None,
        num_fluence_plots=None,
        dose_engine="matthias",
    )

    file_path = "/home/rd/Documents/github/autoplan/database/experiments/b180_bs02_f16_v16_d04_ptv10_02/data_0.npz"
    # file_path = "/home/rd/Documents/github/autoplan/database/experiments/test_b003_bs01_f01_v01_d04_ptv10_fixed_01/data_00_c1.npz"
    # file_path = "/home/rd/Documents/github/autoplan/database/experiments/b090_bs01_f16_v16_d04_ptv100_fixed_01/data_00_c1.npz"
    # file_path = "/home/rd/Documents/github/autoplan/database/experiments/b180_bs01_f24_v24_d04_ptv300_fixed_03/"
    file_path = "/mnt/SSD/github/autoplan/database/experiments/b180_bs02_f16_v16_d04_ptv10_02/data_0.npz"
    file_path = "/home/rd/Documents/github/autoplan/database/experiments/b060_bs01_f16_v64_d04_dematthias_lw03_ptv100_ds222_aug00_fixed_01/data_00_c1.npz"
    with np.load(file_path, allow_pickle=True) as npzdata_load:
        npzdata = {key: npzdata_load[key] for key in npzdata_load.files}
    # p.create_leafs_animation(npzdata, 0, "")
    # p.create_mus(npzdata, 0, "")
    dose = npzdata["dose"]
    p.masks = npzdata["masks"]
    p.create_dvh(dose, 0)
