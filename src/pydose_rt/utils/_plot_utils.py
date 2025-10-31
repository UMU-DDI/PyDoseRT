import numpy as np
from PIL import Image
import io
import base64
import imageio
import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FFMpegWriter


import pydose_rt.utils.path_utils as path_utils
from pydose_rt.utils.config import config as PARAMS


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
        config,
        gen_val,
        experiment,
        number_of_cps,
        fluence_layer,
        num_fluence_plots,
        ## Added by Minh
        scaled_mu=1,  # for Minh scaled mu from 0-1 to get dose 0-1. Now times 100 to get correct
        out_path="database/log_plots",
        ## Added by Minh
    ):
        self.config = config
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
        ## Added by Minh

        self.out_path = out_path

        path_utils.make_dir(self.out_path)

    def create_dvh(self, dose, epoch, text_constraint=None):
        plt.figure(figsize=(10, 6))
        for idx, (key, value) in enumerate(PARAMS.structure_names.items()):
            roi_name = value
            roi = self.masks[..., idx]

            # Extract dose values for the GTV
            dose_values = dose[roi > 0.0]

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
        plt.xlabel("Dose (Gy)")
        plt.ylabel("Volume Fraction")
        plt.title("Dose Volume Histogram (DVH)")
        plt.grid(True)
        plt.legend()

        if text_constraint is None:
            figpath = f"{self.out_path}/dvh_{epoch:02d}.png"
        else:
            figpath = f"{self.out_path}/dvh_{epoch:02d}_{text_constraint}.png"

        plt.savefig(figpath)

        if self.experiment is not None:
            self.experiment.log_image(figpath, step=epoch, overwrite=False)

    def create_animation(self, ct_array, dose_array, epoch, text_constraint=None):
        # List to store frames for the GIF
        frames = []
        num_slices = ct_array.shape[-1]

        # OPTIMIZATION 2: Reduce the figure size
        fig_size = (6, 6)  # Smaller figure size

        for i in range(num_slices):
            fig, ax = plt.subplots(figsize=fig_size, dpi=50)
            im_ct = ax.imshow(ct_array[:, :, i], cmap="gray", vmin=-1, vmax=1)
            dose_norm = dose_array[:, :, i]
            im_dose = ax.imshow(dose_norm, cmap="jet", vmin=0, vmax=1, alpha=0.2)

            # Plot PTV using the custom color
            ptv_mask = self.masks[0, ..., 0]
            mask_display = np.ma.masked_where(ptv_mask[:, :, i] == 0, ptv_mask[:, :, i])
            ptv_cmap = ListedColormap([self.roi_colors["PTV"]])
            im_mask = ax.imshow(mask_display, cmap=ptv_cmap, alpha=1.0, vmin=0, vmax=1)

            # Plot other ROIs (indices 1 to 5 correspond to ROI1 to ROI5)
            roi_keys = ["ROI1", "ROI2", "ROI3", "ROI4", "ROI5"]
            for j, key in enumerate(roi_keys, start=1):
                other_mask = self.masks[0, ..., j]
                mask_display_other = np.ma.masked_where(
                    other_mask[:, :, i] == 0, other_mask[:, :, i]
                )
                roi_cmap = ListedColormap([self.roi_colors[key]])
                im_mask_other = ax.imshow(
                    mask_display_other, cmap=roi_cmap, alpha=1.0, vmin=0, vmax=1
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

        with imageio.get_writer(
            figpath, format="FFMPEG", fps=20, codec="libx264"
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

        centers = leafs[0, 0, :, :]
        widths = leafs[0, 1, :, :]
        n_cp = centers.shape[0]

        # Compute left and right positions for each leaf pair.
        leafs_left = centers - widths / 2.0
        leafs_right = centers + widths / 2.0

        # Determine global x-axis limits.
        margin = 1.0  # extra margin
        global_left = np.min(leafs_left) - margin
        global_right = np.max(leafs_right) + margin

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

    def make(self, model, epoch, text_constraint=None):
        self.x, self.y, self.masks, self.region_weights, self.constraints = (
            self.gen_val[self.idx]
        )  # Minh's generator

        try:
            self.dose, leafs, mus, dose_bypass = model(self.x)
        except:
            self.dose = model(self.x)

        # print(
        #     leafs.numpy().min(),
        #     leafs.numpy().max(),
        #     mus.numpy().min(),
        #     mus.numpy().max(),
        #     self.dose.numpy().min(),
        #     self.dose.numpy().max(),
        # )
        data_dict = {
            "dose": self.dose,
            "leafs": leafs,
            "mus": mus,
            "dose_bypass": dose_bypass,
            "x": self.x,
            "masks": self.masks,
        }
        self.data = data_dict

        if text_constraint is None:
            data_dict_path = f"{self.out_path}/data_{epoch:02d}.npz"
        else:
            data_dict_path = f"{self.out_path}/data_{epoch:02d}_{text_constraint}.npz"

        np.savez_compressed(data_dict_path, **data_dict)

        print(">> create_leafs_animation")
        self.create_leafs_animation(self.data, epoch, text_constraint)

        print(">> create_dvh")
        self.create_dvh(self.dose, epoch, text_constraint)

        self.ct = np.array(self.x[0, ..., 0])
        self.dose = np.array(self.dose)[0, ...]

        print(">> create_animation")
        self.create_animation(self.ct, self.dose, epoch, text_constraint)


if __name__ == "__main__":
    p = Plotter(
        config=None,
        gen_val=None,
        experiment=None,
        number_of_cps=None,
        fluence_layer=None,
        num_fluence_plots=None,
    )

    file_path = "/home/rd/Documents/github/autoplan/database/experiments/b180_bs02_f16_v16_d04_ptv10_02/data_0.npz"
    file_path = "/home/rd/Documents/github/autoplan/database/experiments/test_b003_bs01_f01_v01_d04_ptv10_fixed_01/data_00_c1.npz"
    with np.load(file_path, allow_pickle=True) as npzdata_load:
        npzdata = {key: npzdata_load[key] for key in npzdata_load.files}
    p.create_leafs_animation(npzdata, 0, "")
