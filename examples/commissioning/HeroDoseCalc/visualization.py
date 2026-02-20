"""
Dose visualization helpers.
"""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from .data import Phantom


class VisualizeDose:
    """Plot axial/coronal/sagittal slices with dose overlay."""

    def __init__(self, phantom: "Phantom", dose_grid: np.ndarray, isocenter: Tuple[float, float, float]):
        self.phantom = phantom
        self.density = phantom.data.cpu().numpy()
        self.dose = dose_grid
        self.iso = isocenter

        ox, oy, oz = (
            phantom.origin[0].item(),
            phantom.origin[1].item(),
            phantom.origin[2].item(),
        )
        res = phantom.res
        nx, ny, nz = self.density.shape

        self.x_centers = ox + (np.arange(nx) + 0.5) * res
        self.y_centers = oy + (np.arange(ny) + 0.5) * res
        self.z_centers = oz + (np.arange(nz) + 0.5) * res

        self.x_ext = [ox, ox + nx * res]
        self.y_ext = [oy, oy + ny * res]
        self.z_ext = [oz, oz + nz * res]

        self.cx = np.abs(self.x_centers - self.iso[0]).argmin()
        self.cy = np.abs(self.y_centers - self.iso[1]).argmin()
        self.cz = np.abs(self.z_centers - self.iso[2]).argmin()

    def plot(self):
        """Render three orthogonal dose overlays."""
        views = [
            {
                "title": "Axial (XY)",
                "rho": self.density[:, :, self.cz],
                "dose": self.dose[:, :, self.cz],
                "extent": [self.x_ext[0], self.x_ext[1], self.y_ext[1], self.y_ext[0]],
                "xlabel": "R <-> L",
                "ylabel": "Ant <-> Post",
                "origin": "upper",
                "aspect": "equal",
            },
            {
                "title": "Coronal (XZ)",
                "rho": self.density[:, self.cy, :],
                "dose": self.dose[:, self.cy, :],
                "extent": [self.x_ext[0], self.x_ext[1], self.z_ext[0], self.z_ext[1]],
                "xlabel": "R <-> L",
                "ylabel": "Inf <-> Sup",
                "origin": "lower",
                "aspect": "equal",
            },
            {
                "title": "Sagittal (YZ)",
                "rho": self.density[self.cx, :, :],
                "dose": self.dose[self.cx, :, :],
                "extent": [self.y_ext[0], self.y_ext[1], self.z_ext[0], self.z_ext[1]],
                "xlabel": "Ant <-> Post",
                "ylabel": "Inf <-> Sup",
                "origin": "lower",
                "aspect": "equal",
            },
        ]

        size_x = self.x_ext[1] - self.x_ext[0]
        size_y = self.y_ext[1] - self.y_ext[0]
        size_z = self.z_ext[1] - self.z_ext[0]

        scale = 1.0 / 25.0
        fig_w = max(12, min(24, (size_x * 2 + size_y) * scale))
        fig_h = max(5, min(12, max(size_y, size_z) * scale + 2))

        fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), constrained_layout=True)

        for ax, view in zip(axes, views):
            rho_img = view["rho"].T
            dose_img = view["dose"].T

            ax.imshow(
                rho_img,
                origin=view["origin"],
                cmap="gray",
                extent=view["extent"],
                aspect=view["aspect"],
            )

            d_max = self.dose.max()
            if d_max > 0:
                alpha = dose_img / d_max
                alpha[alpha < 0.05] = 0.0
                ax.imshow(
                    dose_img,
                    origin=view["origin"],
                    cmap="jet",
                    alpha=alpha,
                    extent=view["extent"],
                    aspect=view["aspect"],
                )

            ax.set_title(view["title"], fontsize=12, fontweight="bold")
            ax.set_xlabel(view["xlabel"])
            ax.set_ylabel(view["ylabel"])
            self._add_orientation_markers(ax, view["title"])

        plt.show()

    def _add_orientation_markers(self, ax, title):
        props = dict(transform=ax.transAxes, color="cyan", weight="bold", fontsize=10)
        if title == "Axial (XY)":
            ax.text(0.5, 0.98, "A", ha="center", va="top", **props)
            ax.text(0.5, 0.02, "P", ha="center", va="bottom", **props)
            ax.text(0.02, 0.5, "R", va="center", ha="left", **props)
            ax.text(0.98, 0.5, "L", va="center", ha="right", **props)
        elif title == "Coronal (XZ)":
            ax.text(0.5, 0.98, "H", ha="center", va="top", **props)
            ax.text(0.5, 0.02, "F", ha="center", va="bottom", **props)
            ax.text(0.02, 0.5, "R", va="center", ha="left", **props)
            ax.text(0.98, 0.5, "L", va="center", ha="right", **props)
        elif title == "Sagittal (YZ)":
            ax.text(0.5, 0.98, "H", ha="center", va="top", **props)
            ax.text(0.5, 0.02, "F", ha="center", va="bottom", **props)
            ax.text(0.02, 0.5, "A", va="center", ha="left", **props)
            ax.text(0.98, 0.5, "P", va="center", ha="right", **props)
