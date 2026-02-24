"""Commissioning dashboard plotter."""
from __future__ import annotations

import os
from typing import Any, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .commissioning_parser import MeasurementParser
from .commissioning_toolkit import OutputFactorFitResult, PenumbraFitResult, ProfileCorrectionResult
from .commissioning_types import MeasuredProfile


class CommissioningDashboard:
    def __init__(self) -> None:
        plt.ion()
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.patch.set_facecolor("#0b0d0f")
        self.gs = self.fig.add_gridspec(2, 3, wspace=0.22, hspace=0.25)

        self.ax_console = self.fig.add_subplot(self.gs[0, 0])
        self.ax_penumbra = self.fig.add_subplot(self.gs[0, 1])
        self.ax_profile = self.fig.add_subplot(self.gs[0, 2])
        self.ax_loss = self.fig.add_subplot(self.gs[1, 0])
        self.ax_scatter = self.fig.add_subplot(self.gs[1, 1])
        self.ax_of = self.fig.add_subplot(self.gs[1, 2])

        for ax, title in [
            (self.ax_console, "CONSOLE"),
            (self.ax_penumbra, "PENUMBRA"),
            (self.ax_profile, "PROFILE"),
            (self.ax_scatter, "SCATTER"),
            (self.ax_loss, "LOSS FUNCTION"),
            (self.ax_of, "OF RESIDUAL CURVE"),
        ]:
            self._style_axis(ax, title)

        self.log_lines: List[str] = []
        self.max_log_lines = 18
        self.max_line_chars = 45
        self.console_text = self.ax_console.text(
            0.02,
            0.98,
            "",
            transform=self.ax_console.transAxes,
            va="top",
            ha="left",
            fontfamily="monospace",
            fontsize=9,
            color="#d7dee5",
        )
        self.ax_console.set_xticks([])
        self.ax_console.set_yticks([])

        self.pen_lines = []
        self.profile_lines = []
        self.loss_line = None
        self.of_lines = []

    def _style_axis(self, ax, title: str) -> None:
        ax.set_facecolor("#14181b")
        ax.set_title(title, color="#e6edf3", fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors="#b7c1c8", labelsize=9)
        ax.grid(True, color="#2a2f33", alpha=0.5, linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_color("#2a2f33")

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        vmax = float(np.max(values))
        return values / vmax if vmax != 0.0 else values

    def log(self, message: str) -> None:
        trimmed = message[: self.max_line_chars]
        self.log_lines.append(trimmed)
        self.log_lines = self.log_lines[-self.max_log_lines :]
        self.console_text.set_text("\n".join(self.log_lines))
        self._redraw(self.ax_console)

    def update_penumbra(
        self,
        pos_x: Iterable[float],
        meas_x: np.ndarray,
        sim_x: np.ndarray,
        pos_y: Iterable[float],
        meas_y: np.ndarray,
        sim_y: np.ndarray,
        loss_history: List[float],
        *,
        title_extra: str = "",
    ) -> None:
        pos_x = -np.abs(np.asarray(pos_x))
        pos_y = np.abs(np.asarray(pos_y))
        meas_x = np.asarray(meas_x)
        sim_x = np.asarray(sim_x)
        meas_y = np.asarray(meas_y)
        sim_y = np.asarray(sim_y)
        if meas_x.size:
            meas_x = meas_x / (meas_x[np.argmin(np.abs(pos_x))] or 1.0)
        if sim_x.size:
            sim_x = sim_x / (sim_x[np.argmin(np.abs(pos_x))] or 1.0)
        if meas_y.size:
            meas_y = meas_y / (meas_y[np.argmin(np.abs(pos_y))] or 1.0)
        if sim_y.size:
            sim_y = sim_y / (sim_y[np.argmin(np.abs(pos_y))] or 1.0)

        if not self.pen_lines:
            self.pen_lines = [
                self.ax_penumbra.plot(pos_x, meas_x, "o", color="#6fb1ff", markersize=2, alpha=0.5, label="Meas X")[0],
                self.ax_penumbra.plot(pos_x, sim_x, "-", color="#8ecbff", linewidth=1.6, label="Sim X")[0],
                self.ax_penumbra.plot(pos_y, meas_y, "o", color="#ff8fa3", markersize=2, alpha=0.5, label="Meas Y")[0],
                self.ax_penumbra.plot(pos_y, sim_y, "-", color="#ffb7c5", linewidth=1.6, label="Sim Y")[0],
            ]
            self.ax_penumbra.legend(loc="upper right", fontsize=8, framealpha=0.1)
        else:
            self.pen_lines[0].set_data(pos_x, meas_x)
            self.pen_lines[1].set_data(pos_x, sim_x)
            self.pen_lines[2].set_data(pos_y, meas_y)
            self.pen_lines[3].set_data(pos_y, sim_y)

        self.ax_penumbra.set_title("PENUMBRA", color="#e6edf3", fontsize=12, fontweight="bold", pad=10)
        self._autoscale(self.ax_penumbra)

    def update_profile(
        self,
        pos: Iterable[float],
        meas: np.ndarray,
        sim: np.ndarray,
        *,
        title_extra: str = "",
        axis: str | None = None,
    ) -> None:
        pos = np.asarray(pos)
        if axis:
            axis = axis.upper()
        if axis == "X":
            pos = -np.abs(pos)
        elif axis == "Y":
            pos = np.abs(pos)
        meas = np.asarray(meas)
        sim = np.asarray(sim)
        if meas.size:
            meas = meas / (meas[np.argmin(np.abs(pos))] or 1.0)
        if sim.size:
            sim = sim / (sim[np.argmin(np.abs(pos))] or 1.0)
        if not self.profile_lines:
            self.profile_lines = [
                self.ax_profile.plot(pos, meas, "o", color="#9be9a8", markersize=2, alpha=0.5, label="Measured")[0],
                self.ax_profile.plot(pos, sim, "-", color="#b7f7d1", linewidth=1.6, label="Sim")[0],
            ]
            self.ax_profile.legend(loc="upper right", fontsize=8, framealpha=0.1)
        else:
            self.profile_lines[0].set_data(pos, meas)
            self.profile_lines[1].set_data(pos, sim)

        self.ax_profile.set_title("PROFILE", color="#e6edf3", fontsize=12, fontweight="bold", pad=10)
        self._autoscale(self.ax_profile)

    def update_scatter_multi(
        self,
        pos_x_list: Sequence[np.ndarray],
        meas_x_list: Sequence[np.ndarray],
        sim_x_list: Sequence[np.ndarray],
        pos_y_list: Sequence[np.ndarray],
        meas_y_list: Sequence[np.ndarray],
        sim_y_list: Sequence[np.ndarray],
    ) -> None:
        self.ax_scatter.clear()
        self._style_axis(self.ax_scatter, "SCATTER")
        def _normalize_series(pos: np.ndarray, values: np.ndarray) -> np.ndarray:
            if values.size == 0:
                return values
            cax_idx = int(np.argmin(np.abs(pos)))
            norm = values[cax_idx] if values[cax_idx] != 0 else values.max()
            return values / norm if norm != 0 else values

        for idx, (pos_x, meas_x, sim_x) in enumerate(zip(pos_x_list, meas_x_list, sim_x_list)):
            meas_x = _normalize_series(pos_x, meas_x)
            sim_x = _normalize_series(pos_x, sim_x)
            label_meas = "Meas X" if idx == 0 else None
            label_sim = "Sim X" if idx == 0 else None
            self.ax_scatter.plot(pos_x, meas_x, "o", color="#6fb1ff", markersize=2, alpha=0.5, label=label_meas)
            self.ax_scatter.plot(pos_x, sim_x, "-", color="#8ecbff", linewidth=1.6, label=label_sim)
        for idx, (pos_y, meas_y, sim_y) in enumerate(zip(pos_y_list, meas_y_list, sim_y_list)):
            meas_y = _normalize_series(pos_y, meas_y)
            sim_y = _normalize_series(pos_y, sim_y)
            label_meas = "Meas Y" if idx == 0 else None
            label_sim = "Sim Y" if idx == 0 else None
            self.ax_scatter.plot(pos_y, meas_y, "o", color="#ff8fa3", markersize=2, alpha=0.5, label=label_meas)
            self.ax_scatter.plot(pos_y, sim_y, "-", color="#ffb7c5", linewidth=1.6, label=label_sim)
        if pos_x_list or pos_y_list:
            self.ax_scatter.legend(loc="upper right", fontsize=8, framealpha=0.1)
        self._autoscale(self.ax_scatter)

    def update_loss(self, loss_history: List[float]) -> None:
        xs = np.arange(1, len(loss_history) + 1)
        ys = np.array(loss_history, dtype=float)
        if self.loss_line is None:
            self.loss_line = self.ax_loss.plot(xs, ys, "-", color="#ffd166", linewidth=1.8)[0]
        else:
            self.loss_line.set_data(xs, ys)
        self._autoscale(self.ax_loss)

    def update_of_residual(
        self,
        sizes_mm: np.ndarray,
        of_meas: np.ndarray,
        of_model: np.ndarray,
    ) -> None:
        sort_idx = np.argsort(sizes_mm)
        sizes_mm = sizes_mm[sort_idx]
        of_meas = of_meas[sort_idx]
        of_model = of_model[sort_idx]
        if not self.of_lines:
            self.of_lines = [
                self.ax_of.plot(sizes_mm, of_meas, "o", color="#6fb1ff", markersize=3, label="OF meas")[0],
                self.ax_of.plot(sizes_mm, of_model, "-", color="#ff8fa3", linewidth=1.6, label="OF model")[0],
            ]
            self.ax_of.legend(loc="upper right", fontsize=8, framealpha=0.1)
        else:
            self.of_lines[0].set_data(sizes_mm, of_meas)
            self.of_lines[1].set_data(sizes_mm, of_model)
        self._autoscale(self.ax_of)

    def _autoscale(self, ax) -> None:
        ax.relim()
        ax.autoscale_view()
        self._redraw(ax)

    def _redraw(self, ax) -> None:
        ax.figure.canvas.draw_idle()
        ax.figure.canvas.flush_events()


class CommissioningPlotter:
    def __init__(self, *, show: bool = True):
        self.show = show
        self.dashboard = CommissioningDashboard() if show else None

    def log(self, message: str) -> None:
        if self.dashboard is not None:
            self.dashboard.log(message)

    def update_penumbra(
        self,
        pos_x: Iterable[float],
        meas_x: np.ndarray,
        sim_x: np.ndarray,
        pos_y: Iterable[float],
        meas_y: np.ndarray,
        sim_y: np.ndarray,
        loss_history: List[float],
        *,
        title_extra: str = "",
    ) -> None:
        if self.dashboard is not None:
            self.dashboard.update_penumbra(
                pos_x, meas_x, sim_x, pos_y, meas_y, sim_y, loss_history, title_extra=title_extra
            )

    def update_profile(
        self,
        pos: Iterable[float],
        meas: np.ndarray,
        sim: np.ndarray,
        *,
        title_extra: str = "",
        axis: str | None = None,
    ) -> None:
        if self.dashboard is not None:
            self.dashboard.update_profile(pos, meas, sim, title_extra=title_extra, axis=axis)

    def update_scatter_multi(
        self,
        pos_x_list: Sequence[np.ndarray],
        meas_x_list: Sequence[np.ndarray],
        sim_x_list: Sequence[np.ndarray],
        pos_y_list: Sequence[np.ndarray],
        meas_y_list: Sequence[np.ndarray],
        sim_y_list: Sequence[np.ndarray],
    ) -> None:
        if self.dashboard is not None:
            self.dashboard.update_scatter_multi(
                pos_x_list, meas_x_list, sim_x_list, pos_y_list, meas_y_list, sim_y_list
            )
    def update_loss(self, loss_history: List[float]) -> None:
        if self.dashboard is not None:
            self.dashboard.update_loss(loss_history)

    def update_of_residual(
        self, sizes_mm: np.ndarray, of_meas: np.ndarray, of_model: np.ndarray
    ) -> None:
        if self.dashboard is not None:
            self.dashboard.update_of_residual(sizes_mm, of_meas, of_model)

    def plot_penumbra_fit(self, result: PenumbraFitResult) -> Any:
        self.update_penumbra(
            result.crossline_meas_pos_mm,
            result.crossline_meas_dose,
            result.crossline_sim_dose,
            result.inline_meas_pos_mm,
            result.inline_meas_dose,
            result.inline_sim_dose,
            [],
            title_extra="",
        )
        return self.dashboard.fig if self.dashboard is not None else None

    def plot_profile_correction(self, result: ProfileCorrectionResult) -> Any:
        self.update_profile(result.position_mm, result.meas_norm, result.sim_norm, title_extra="")
        return self.dashboard.fig if self.dashboard is not None else None

    def plot_output_factor_fit(self, result: OutputFactorFitResult) -> Any:
        meas = [m for m in result.measurements if abs(m.field_x_mm - m.field_y_mm) <= 1.0]
        meas.sort(key=lambda m: m.field_x_mm)
        equiv = np.array([m.field_x_mm for m in meas])
        of_meas = np.array([m.value for m in meas])
        of_model = np.array([m.sc_model * m.sp for m in meas])
        self.update_of_residual(equiv, of_meas, of_model)
        return self.dashboard.fig if self.dashboard is not None else None

    def generate_report(self, *, toolkit: Any, measurement_files: List[str], output_dir: str) -> None:
        message_start = f"Report generation started: {output_dir}"
        print(message_start)
        self.log(message_start)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        all_profiles: List[MeasuredProfile] = []
        for f in measurement_files:
            all_profiles.extend(MeasurementParser.parse_rfa300(f))

        unique_db: dict[Tuple[str, Tuple[int, int], str, str, int], MeasuredProfile] = {}
        for p in all_profiles:
            fs_key = (round(p.field_size_mm[0]), round(p.field_size_mm[1]))
            d_key = round(p.depth_mm) if p.depth_mm is not None else -1
            key = (p.energy, fs_key, p.scan_type, p.axis, d_key)
            unique_db[key] = p

        clean_profiles = list(unique_db.values())

        sim_groups: dict[Tuple[str, Tuple[int, int]], List[MeasuredProfile]] = {}
        for p in clean_profiles:
            fs_key = (round(p.field_size_mm[0]), round(p.field_size_mm[1]))
            sim_groups.setdefault((p.energy, fs_key), []).append(p)

        for (energy, fs), profiles in sim_groups.items():
            fig = self._create_report_figure_fast2d(toolkit, energy, fs, profiles)
            fname = f"Report_{energy}_{fs[0]}x{fs[1]}.png"
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(os.path.join(output_dir, fname), dpi=100)
            plt.close(fig)
        message_done = f"Report generation finished: {output_dir}"
        print(message_done)
        self.log(message_done)

    def _create_report_figure_fast2d(
        self,
        toolkit: Any,
        energy: str,
        field_size_tuple: Tuple[int, int],
        profiles: Sequence[MeasuredProfile],
    ) -> Any:
        depth_map: dict[int, dict[str, MeasuredProfile]] = {}
        for p in profiles:
            if p.scan_type != "PRO":
                continue
            if p.depth_mm is None:
                continue
            axis = p.axis.upper()
            if axis not in ("X", "Y"):
                continue
            depth_key = int(round(p.depth_mm))
            depth_map.setdefault(depth_key, {})[axis] = p

        depths = sorted(depth_map.keys())
        if not depths:
            fig = plt.figure(figsize=(12, 6))
            ax = plt.subplot(1, 1, 1)
            ax.text(0.5, 0.5, "No profile data for report", ha="center")
            return fig

        profiles_to_sim: List[MeasuredProfile] = []
        for entry in depth_map.values():
            profiles_to_sim.extend([p for p in entry.values() if p is not None])
        sim_map = toolkit.simulate_profiles_for_report(profiles_to_sim, res_mm=1.0)

        fig = plt.figure(figsize=(14, max(4, 3 * len(depths))))
        fig.suptitle(f"Commissioning: {energy} - Field {field_size_tuple}", fontsize=16)

        for row_idx, depth in enumerate(depths, start=1):
            ax = plt.subplot(len(depths), 1, row_idx)
            ax.set_title(f"Depth {depth} mm")
            entry = depth_map[depth]
            p_x = entry.get("X")
            p_y = entry.get("Y")

            if p_x is None and p_y is None:
                ax.text(0.5, 0.5, "No X/Y profiles", ha="center")
                continue

            if p_x is not None:
                key_x = (
                    p_x.energy,
                    int(p_x.id),
                    p_x.axis.upper(),
                    round(float(p_x.depth_mm or 0.0), 3),
                    round(float(p_x.field_size_mm[0]), 3),
                    round(float(p_x.field_size_mm[1]), 3),
                    int(p_x.position_mm.shape[0]),
                )
                sim_x = sim_map.get(key_x)
                if sim_x is not None:
                    cax_idx = int(np.argmin(np.abs(p_x.position_mm)))
                    m_norm = p_x.dose_values[cax_idx] if p_x.dose_values[cax_idx] > 0 else p_x.dose_values.max()
                    s_norm = sim_x[cax_idx] if sim_x[cax_idx] > 0 else sim_x.max()
                    x_pos = -np.abs(p_x.position_mm)
                    ax.plot(x_pos, p_x.dose_values / m_norm * 100, ".", color="tab:blue", alpha=0.4, markersize=2)
                    ax.plot(x_pos, sim_x / s_norm * 100, "-", color="tab:blue", label="Crossline")

            if p_y is not None:
                key_y = (
                    p_y.energy,
                    int(p_y.id),
                    p_y.axis.upper(),
                    round(float(p_y.depth_mm or 0.0), 3),
                    round(float(p_y.field_size_mm[0]), 3),
                    round(float(p_y.field_size_mm[1]), 3),
                    int(p_y.position_mm.shape[0]),
                )
                sim_y = sim_map.get(key_y)
                if sim_y is not None:
                    cax_idx = int(np.argmin(np.abs(p_y.position_mm)))
                    m_norm = p_y.dose_values[cax_idx] if p_y.dose_values[cax_idx] > 0 else p_y.dose_values.max()
                    s_norm = sim_y[cax_idx] if sim_y[cax_idx] > 0 else sim_y.max()
                    y_pos = np.abs(p_y.position_mm)
                    ax.plot(y_pos, p_y.dose_values / m_norm * 100, ".", color="tab:orange", alpha=0.4, markersize=2)
                    ax.plot(y_pos, sim_y / s_norm * 100, "-", color="tab:orange", label="Inline")

            ax.axvline(0.0, color="k", linewidth=0.8, alpha=0.4)
            ax.set_xlabel("Off-Axis (mm)")
            ax.set_ylabel("% Dose")
            ax.set_ylim(0, 110)
            ax.grid(True, alpha=0.3)
            ax.legend()

        return fig
