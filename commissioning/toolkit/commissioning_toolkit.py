"""Commissioning toolkit using the PyDoseRT dose engine.

Fitting steps: geometric penumbra → off-axis profile correction → head scatter
and output factors.  Configuration state is held in memory (``self._config_data``)
throughout the pipeline; the final results are exported as PyDoseRT-format JSON
files via ``export_config``.

Dose planes are simulated using PyDoseRT layers directly:

  * Aperture + penumbra + profile + head scatter – ``FluenceMapLayer`` at
    1 mm/px native resolution.
  * Nyholm pencil-beam kernel – ``PencilBeamModel`` evaluated at the exact
    target depth.
"""
from __future__ import annotations

import json
import os
import re
import types
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.special import erf

from pydosert.physics.kernels.pencil_beam_model import PencilBeamModel
from pydosert.layers.FluenceMapLayer import FluenceMapLayer
from pydosert.data.machine_config import MachineConfig as PydoseRTMachineConfig

from .commissioning_parser import MeasurementParser
from .commissioning_types import MeasuredProfile, OutputFactorMeasurement
from .config import MachineConfig


# ---------------------------------------------------------------------------
# Device selection (mirrors pydosert.layers behaviour)
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


DEVICE = _get_device()


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PenumbraFitResult:
    energy: str
    target_field_mm: Tuple[float, float]
    target_depth_mm: float
    geometric_penumbra_mm: Tuple[float, float]
    crossline_meas_pos_mm: np.ndarray
    crossline_meas_dose: np.ndarray
    crossline_sim_dose: np.ndarray
    inline_meas_pos_mm: np.ndarray
    inline_meas_dose: np.ndarray
    inline_sim_dose: np.ndarray


@dataclass
class ProfileCorrectionResult:
    energy: str
    profile_id: int
    field_size_mm: Tuple[float, float]
    depth_mm: float
    axis: str
    position_mm: np.ndarray
    meas_norm: np.ndarray
    sim_norm: np.ndarray
    profile_curve: List[Tuple[float, float]]


@dataclass
class OutputFactorFitResult:
    energy: str
    head_scatter_magnitude: float
    head_scatter_sigma_mm: Tuple[float, float]
    output_factor_curve: List[List[float]]
    measurements: List[OutputFactorMeasurement]


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

def _fine_profile(
    pos_mm: np.ndarray, dose_values: np.ndarray, *, samples: int = 2000
) -> Tuple[np.ndarray, np.ndarray] | None:
    d_max = float(np.max(dose_values)) if dose_values is not None else 0.0
    if d_max <= 0.0:
        return None
    pos_mm = np.asarray(pos_mm, dtype=float)
    dose_pct = (np.asarray(dose_values, dtype=float) / d_max) * 100.0
    sort_idx = np.argsort(pos_mm)
    pos_s = pos_mm[sort_idx]
    dose_s = dose_pct[sort_idx]
    f = interp1d(pos_s, dose_s, kind="linear", fill_value="extrapolate")
    fine_pos = np.linspace(float(pos_s.min()), float(pos_s.max()), samples)
    return fine_pos, f(fine_pos)


def calculate_penumbra_width(pos_mm: np.ndarray, dose_values: np.ndarray) -> float:
    fine = _fine_profile(pos_mm, dose_values)
    if fine is None:
        return 0.0
    fine_pos, fine_dose = fine
    crossings = np.where(np.diff(np.sign(fine_dose - 50.0)))[0]
    widths: List[float] = []
    for c_idx in crossings:
        start = max(0, c_idx - 200)
        end = min(len(fine_dose), c_idx + 200)
        local_pos = fine_pos[start:end]
        local_dose = fine_dose[start:end]
        if len(local_dose) < 5:
            continue
        try:
            if local_dose[0] < local_dose[-1]:
                p20 = np.interp(20.0, local_dose, local_pos)
                p80 = np.interp(80.0, local_dose, local_pos)
            else:
                p20 = np.interp(20.0, local_dose[::-1], local_pos[::-1])
                p80 = np.interp(80.0, local_dose[::-1], local_pos[::-1])
            w = abs(p80 - p20)
            if 0.5 < w < 20.0:
                widths.append(w)
        except Exception:
            continue
    return float(np.mean(widths)) if widths else 0.0


# ---------------------------------------------------------------------------
# CommissioningToolkit
# ---------------------------------------------------------------------------

class CommissioningToolkit:
    def __init__(
        self,
        config_path: str,
        *,
        device: torch.device = DEVICE,
        verbose: bool = False,
        log_callback: Callable[[str], None] | None = None,
        kernel_size_mm: float = 100.0,
    ):
        self.config_path = config_path
        self._config_data: Dict[str, Any] = self.load_json(config_path)
        self.device = device
        self.verbose = verbose
        self.log_callback = log_callback
        self.jaw_offset_mm = (0.0, 0.0)
        self.jaw_scale = (1.0, 1.0)
        self.kernel_size_mm: float = float(kernel_size_mm)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
        if self.log_callback is not None:
            self.log_callback(message)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _axis_offset_mm(self, profile: MeasuredProfile) -> float:
        axis = profile.axis.upper()
        if axis == "X":
            return float(self.jaw_offset_mm[0])
        if axis == "Y":
            return float(self.jaw_offset_mm[1])
        if axis == "D":
            return 0.5 * (float(self.jaw_offset_mm[0]) + float(self.jaw_offset_mm[1]))
        return 0.0

    def _adjust_positions_for_offset(self, profile: MeasuredProfile) -> np.ndarray:
        return np.asarray(profile.position_mm, dtype=float) - self._axis_offset_mm(profile)

    @staticmethod
    def _interp_if_needed(
        sim: np.ndarray, sim_pos: np.ndarray, target_pos: np.ndarray
    ) -> np.ndarray:
        if sim.shape[0] != target_pos.shape[0] or not np.array_equal(sim_pos, target_pos):
            return np.interp(target_pos, sim_pos, sim)
        return sim

    def _field_size_scaled(
        self,
        field_size_mm: Tuple[float, float],
        *,
        depth_mm: float | None,
        ssd_mm: float,
        sad_mm: float = 1000.0,
    ) -> Tuple[float, float]:
        scaled = (field_size_mm[0] * self.jaw_scale[0], field_size_mm[1] * self.jaw_scale[1])
        if depth_mm is None or ssd_mm <= 0.0:
            return scaled
        mag = (ssd_mm + depth_mm) / sad_mm
        return (scaled[0] * mag, scaled[1] * mag)

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _hs_config_for_energy(self, energy: str) -> MachineConfig:
        """Build config from in-memory state and blank out the OF residual curve (used during HS fitting)."""
        cfg = MachineConfig._from_dict(self._config_data, energy=energy)
        cfg.output_factor_curve = [(0.0, 1.0), (500.0, 1.0)]
        return cfg

    @classmethod
    def _format_json_compact(cls, value: Any, *, indent: int, level: int) -> List[str]:
        pad = " " * (indent * level)
        if isinstance(value, dict):
            lines = [pad + "{"]
            items = list(value.items())
            for i, (key, val) in enumerate(items):
                child_lines = cls._format_json_compact(val, indent=indent, level=level + 1)
                if child_lines:
                    child_lines[0] = (
                        " " * (indent * (level + 1))
                        + json.dumps(str(key))
                        + ": "
                        + child_lines[0].lstrip()
                    )
                if i != len(items) - 1:
                    child_lines[-1] += ","
                lines.extend(child_lines)
            lines.append(pad + "}")
            return lines
        if isinstance(value, list):
            if not value:
                return [pad + "[]"]
            if all(not isinstance(v, (dict, list)) for v in value):
                return [pad + json.dumps(value)]
            lines = [pad + "["]
            for i, item in enumerate(value):
                item_lines = cls._format_json_compact(item, indent=indent, level=level + 1)
                if len(item_lines) == 1:
                    line = item_lines[0]
                    if i != len(value) - 1:
                        line += ","
                    lines.append(line)
                else:
                    lines.extend(item_lines[:-1])
                    last = item_lines[-1]
                    if i != len(value) - 1:
                        last += ","
                    lines.append(last)
            lines.append(pad + "]")
            return lines
        return [pad + json.dumps(value)]

    @classmethod
    def save_json_compact(cls, path: str, data: Dict[str, Any], *, indent: int = 4) -> None:
        lines = cls._format_json_compact(data, indent=indent, level=0)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def export_config(
        self,
        *,
        output_dir: str | None = None,
        output_prefix: str = "machine_config",
    ) -> Dict[str, str]:
        """Convert the in-memory commissioning config to per-energy PyDoseRT-compatible flat JSONs.

        The commissioning format is a nested, multi-energy document.  PyDoseRT's
        ``MachineConfig`` expects a flat, single-energy JSON.  This method performs
        the conversion for every energy present in ``self._config_data`` and writes
        one output file per energy.

        Field mapping
        -------------
        * ``tpr20_10``                        → ``tpr_20_10`` (direct)
        * ``mlc.leaf_boundaries``             → ``leaf_widths`` (differences) +
                                               ``number_of_leaf_pairs`` (count)
        * ``mlc.transmission``               → ``mlc_transmission`` (direct)
        * ``source.geometric_penumbra_mm``   → ``penumbra_fwhm``
          Commissioning stores Gaussian sigma (mm); PyDoseRT expects FWHM (mm at
          1 mm/px resolution), so each value is multiplied by 2.355.
        * ``source.head_scatter_magnitude``  → ``head_scatter_amplitude``
          PyDoseRT uses ``[amplitude, 1.0]`` so that the FluenceMapLayer formula
          ``(1 - amp_mlc)*fluence + amp_mlc*scatter/amp_jaw`` reduces to the
          commissioning blending ``(1-w)*fluence + w*scatter``.
        * ``source.head_scatter_sigma_mm``   → ``head_scatter_sigma`` (÷ 10, mm→cm)
        * ``reference_mu``                   → ``calibration_mu`` (direct)
        * ``profile.curve``                  → ``profile_corrections``
          Pairs ``[[r, ratio], …]`` → two lists ``[[r, …], [ratio, …]]``.
        * ``output_factors.curve``           → ``output_factors``
          Same transposition as profile corrections.

        Parameters
        ----------
        output_dir:
            Directory to write the per-energy files.  Defaults to current directory.
        output_prefix:
            Filename stem prefix; files are named ``{prefix}_{energy}.json``.

        Returns
        -------
        dict
            Mapping of energy key → output file path.
        """
        data = self._config_data

        # ---- MLC geometry (shared across energies) ---------------------------
        mlc = data.get("mlc", {})
        leaf_boundaries: List[float] = mlc.get("leaf_boundaries", [])
        leaf_widths = [
            round(float(leaf_boundaries[i + 1]) - float(leaf_boundaries[i]), 4)
            for i in range(len(leaf_boundaries) - 1)
        ]
        number_of_leaf_pairs = len(leaf_widths)
        mlc_transmission = float(mlc.get("transmission", 0.0))

        # ---- Per-energy export -----------------------------------------------
        if output_dir is None:
            output_dir = "."
        os.makedirs(output_dir, exist_ok=True)

        output_paths: Dict[str, str] = {}
        for energy_key, e_data in (data.get("energies") or {}).items():
            src = e_data.get("source", {})

            # Skip energies that still have placeholder (unfitted) parameters.
            # head_scatter_magnitude is always > 0 after a successful Step 3 fit
            # (amplitude bounds are 0.03–0.15); 0.0 means the energy was never
            # commissioned in this session.
            if float(src.get("head_scatter_magnitude", 0.0)) == 0.0:
                self._log(
                    f"Skipping energy '{energy_key}': head_scatter_magnitude is 0 "
                    "(energy not commissioned in this run)"
                )
                continue

            # Penumbra: sigma (mm) → FWHM (px at 1 mm/px)
            geo_pen = src.get("geometric_penumbra_mm", [0.0, 0.0])
            penumbra_fwhm = [round(float(v) * 2.355, 4) for v in geo_pen]

            # Head scatter amplitude: scalar → [amp, 1.0]
            hs_magnitude = float(src.get("head_scatter_magnitude", 0.0))
            head_scatter_amplitude = [round(hs_magnitude, 6), 1.0]

            # Head scatter sigma: mm (at iso) → cm
            hs_sigma_mm = src.get("head_scatter_sigma_mm", [0.0, 0.0])
            head_scatter_sigma = [round(float(v) / 10.0, 5) for v in hs_sigma_mm]

            # Profile corrections: [[r, ratio], …] → [[r, …], [ratio, …]]
            profile_curve = (e_data.get("profile") or {}).get("curve") or []
            profile_corrections: List[List[float]] | None = None
            if profile_curve:
                profile_corrections = [
                    [float(p[0]) for p in profile_curve],
                    [float(p[1]) for p in profile_curve],
                ]

            # Output factors: [[s, OF], …] → [[s, …], [OF, …]]
            of_curve = (e_data.get("output_factors") or {}).get("curve") or []
            output_factors: List[List[float]] | None = None
            if of_curve:
                output_factors = [
                    [float(p[0]) for p in of_curve],
                    [float(p[1]) for p in of_curve],
                ]

            # Mean photon energy: derive from energy key (e.g. "10MV" → 10.0)
            _match = re.search(r"(\d+(?:\.\d+)?)", energy_key)
            mean_photon_energy_MeV = float(_match.group(1)) if _match else 10.0

            out_cfg: Dict[str, Any] = {
                "tpr_20_10": float(e_data.get("tpr20_10", 0.7)),
                "number_of_leaf_pairs": number_of_leaf_pairs,
                "mlc_transmission": mlc_transmission,
                "calibration_mu": float(e_data.get("reference_mu", 100.0)),
                "mean_photon_energy_MeV": mean_photon_energy_MeV,
                "leaf_widths": leaf_widths,
                "penumbra_fwhm": penumbra_fwhm,
                "head_scatter_amplitude": head_scatter_amplitude,
                "head_scatter_sigma": head_scatter_sigma,
                "head_scatter_ssd_mm": 50.0,
            }
            if profile_corrections is not None:
                out_cfg["profile_corrections"] = profile_corrections
            if output_factors is not None:
                out_cfg["output_factors"] = output_factors

            out_path = os.path.join(output_dir, f"{output_prefix}_{energy_key}.json")
            self.save_json_compact(out_path, out_cfg)
            output_paths[energy_key] = out_path
            self._log(f"Exported machine config ({energy_key}): {out_path}")

        return output_paths

    def finalize_config(self) -> None:
        """Clean up numerical precision in ``self._config_data`` in-place."""
        data = self._config_data
        for _, e_data in (data.get("energies") or {}).items():
            prof = (e_data.get("profile") or {}).get("curve")
            if isinstance(prof, list) and prof:
                prof_sorted = sorted(prof, key=lambda x: x[0])
                clean: list = []
                last_r = -20.0
                for r, val in prof_sorted:
                    r, val = float(r), float(val)
                    if r == 0.0 or r >= 495.0 or (r - last_r) >= 10.0:
                        clean.append([round(r, 1), round(val, 4)])
                        last_r = r
                e_data.setdefault("profile", {})["curve"] = clean

            of_curve = (e_data.get("output_factors") or {}).get("curve")
            if isinstance(of_curve, list) and of_curve:
                e_data.setdefault("output_factors", {})["curve"] = [
                    [round(float(s), 1), round(float(v), 4)] for s, v in of_curve
                ]

            src = e_data.get("source")
            if isinstance(src, dict):
                for key in ("geometric_penumbra_mm", "head_scatter_sigma_mm"):
                    if key in src:
                        src[key] = [round(float(x), 4) for x in src[key]]
                if "head_scatter_magnitude" in src:
                    src["head_scatter_magnitude"] = round(float(src["head_scatter_magnitude"]), 5)

    # ------------------------------------------------------------------
    # Core simulation (PyDoseRT engine)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_engine_config(cfg: MachineConfig) -> PydoseRTMachineConfig:
        """Convert a commissioning MachineConfig to a PyDoseRT MachineConfig.

        The result is always at 1 mm/px (native FluenceMapLayer resolution).
        ``output_factors`` is intentionally left as ``None`` so that the Sc +
        OF-residual logic in ``_simulate_dose_plane_fast`` is applied manually,
        keeping the commissioning physics model intact.
        """
        mlc = cfg.mlc
        if mlc and mlc.leaf_boundaries:
            lb = mlc.leaf_boundaries
            leaf_widths = [float(lb[i + 1]) - float(lb[i]) for i in range(len(lb) - 1)]
            mlc_transmission = mlc.transmission
        else:
            leaf_widths = None
            mlc_transmission = cfg.mlc_transmission
        number_of_leaf_pairs = len(leaf_widths) if leaf_widths else 60

        # sigma (mm) → FWHM (px) at 1 mm/px: fwhm_px = sigma_mm * 2.355
        penumbra_fwhm = [v * 2.355 for v in cfg.geometric_penumbra_mm]

        # head scatter amplitude: scalar → [amp, 1.0]  (matches FluenceMapLayer blend)
        w = float(cfg.head_scatter_magnitude)
        head_scatter_amplitude = [w, 1.0] if w > 0.0 else None
        # head scatter sigma: mm (at iso) → cm for precompute_head_scatter_kernel
        sc_sig = cfg.head_scatter_sigma_mm
        head_scatter_sigma = [float(sc_sig[0]) / 10.0, float(sc_sig[1]) / 10.0] if w > 0.0 else None

        # profile corrections: [(r, ratio), …] → [[r, …], [ratio, …]]
        profile_corrections = None
        if cfg.profile_curve and len(cfg.profile_curve) >= 2:
            profile_corrections = [
                [float(p[0]) for p in cfg.profile_curve],
                [float(p[1]) for p in cfg.profile_curve],
            ]

        return PydoseRTMachineConfig(
            tpr_20_10=cfg.tpr20_10,
            number_of_leaf_pairs=number_of_leaf_pairs,
            mlc_transmission=mlc_transmission,
            calibration_mu=cfg.reference_mu,
            mean_photon_energy_MeV=10.0,
            leaf_widths=leaf_widths,
            penumbra_fwhm=penumbra_fwhm if any(v > 1e-3 for v in penumbra_fwhm) else None,
            head_scatter_amplitude=head_scatter_amplitude,
            head_scatter_sigma=head_scatter_sigma,
            head_scatter_ssd_mm=50.0,
            profile_corrections=profile_corrections,
            output_factors=None,  # applied manually in _simulate_dose_plane_fast
        )

    def _simulate_dose_plane_fast(
        self,
        config: MachineConfig,
        field_size_mm: Tuple[float, float],
        depth_mm: float,
        *,
        res_mm: float,
        grid_span_mm: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate a 2D dose plane at *depth_mm* using PyDoseRT layers directly.

        The fluence map (aperture, penumbra, profile correction, head scatter) is
        computed by ``FluenceMapLayer`` at its native 1 mm/px resolution, which
        matches the standard PyDoseRT treatment-planning setup and is consistent
        with the parameters exported by ``export_config``.

        Steps
        -----
        1. FluenceMapLayer (1 mm/px)  – aperture, penumbra, profile, head scatter
        2. Sc + OF residual            – analytical formula + lookup table
        3. Nyholm pencil-beam kernel   – PyDoseRT ``PencilBeamModel`` at exact depth
        4. Downsample to *res_mm*      – bilinear interpolation to target resolution
        """
        device = self.device
        dtype = torch.float32
        fw, fh = float(field_size_mm[0]), float(field_size_mm[1])

        # ---- 1. FluenceMapLayer at 1 mm/px -----------------------------------
        engine_cfg = self._build_engine_config(config)
        leaf_widths = engine_cfg.leaf_widths
        n_full = int(sum(leaf_widths)) if leaf_widths else 400
        if n_full % 2 == 0:
            n_full += 1

        fluence_layer = FluenceMapLayer(
            engine_cfg, field_size=(n_full, n_full), device=device, dtype=dtype
        )

        # Leaf/jaw positions in mm = pixels at 1 mm/px
        N = engine_cfg.number_of_leaf_pairs
        leaf_pos = torch.zeros(1, 1, N, 2, device=device, dtype=dtype)
        leaf_pos[..., 0] = -fw / 2.0
        leaf_pos[..., 1] =  fw / 2.0
        jaw_pos = torch.zeros(1, 1, 2, device=device, dtype=dtype)
        jaw_pos[..., 0] = -fh / 2.0
        jaw_pos[..., 1] =  fh / 2.0

        with torch.no_grad():
            fluence = fluence_layer(leaf_pos, jaw_pos)  # [1, n_full, n_full]
        fluence = fluence.unsqueeze(0)  # [1, 1, n_full, n_full]

        # ---- 2. Output factor (Sc analytical + OF-residual curve) -----------
        w = float(config.head_scatter_magnitude)
        sc_sig_x = float(config.head_scatter_sigma_mm[0])
        sc_sig_y = float(config.head_scatter_sigma_mm[1])
        sc = 1.0
        if w > 0.0 and sc_sig_x > 0.0 and sc_sig_y > 0.0:
            t10_x = erf(100.0 / (2.0 * np.sqrt(2.0) * sc_sig_x))
            t10_y = erf(100.0 / (2.0 * np.sqrt(2.0) * sc_sig_y))
            norm_sc = 1.0 + w * t10_x * t10_y
            tx = erf(fw / (2.0 * np.sqrt(2.0) * sc_sig_x))
            ty = erf(fh / (2.0 * np.sqrt(2.0) * sc_sig_y))
            sc = (1.0 + w * tx * ty) / norm_sc

        of_residual = 1.0
        if config.output_factor_curve is not None and len(config.output_factor_curve) >= 2:
            of_pts = sorted(config.output_factor_curve, key=lambda p: p[0])
            s_eq = (2.0 * fw * fh) / (fw + fh + 1e-6)
            of_residual = float(
                np.interp(s_eq, [p[0] for p in of_pts], [p[1] for p in of_pts])
            )

        fluence = fluence * (sc * of_residual)

        # ---- 3. Nyholm pencil-beam kernel at 1 mm/px -------------------------
        # Extract the central grid_span_mm region before convolution so the
        # kernel has enough context (from the full fluence) to be accurate.
        n_roi_1mm = min(int(grid_span_mm), n_full)
        if n_roi_1mm % 2 == 0:
            n_roi_1mm += 1
        c = n_full // 2
        start = c - n_roi_1mm // 2
        fluence_roi = fluence[:, :, start : start + n_roi_1mm, start : start + n_roi_1mm]


        # Kernel size is fixed by kernel_size_mm and is independent of the
        # simulation grid span.  It must not exceed the extracted ROI (which
        # itself is bounded by n_full, the physical leaf-pair extent).
        kernel_1mm = int(self.kernel_size_mm)
        if kernel_1mm % 2 == 0:
            kernel_1mm += 1
        pbm = PencilBeamModel((1.0, 1.0, 1.0), config.tpr20_10, kernel_1mm)
        d_t = torch.tensor([[[[float(depth_mm)]]]], dtype=dtype, device=device)
        with torch.no_grad():
            kernel = pbm.get_pencil_beam(
                d_t, pbm.rs[None, None].to(device=device, dtype=dtype), normalize=True
            )  # [1, 1, kH, kW]

        kH, kW = pbm.kernel_size_h, pbm.kernel_size_w
        with torch.no_grad():
            if device.type == "cpu" and kH >= 32:
                fft_h = n_roi_1mm + kH - 1
                fft_w = n_roi_1mm + kW - 1
                f2d = fluence_roi[0, 0]
                k2d = torch.flip(kernel[0, 0], dims=(0, 1))
                conv_full = torch.fft.irfft2(
                    torch.fft.rfft2(f2d, s=(fft_h, fft_w))
                    * torch.fft.rfft2(k2d, s=(fft_h, fft_w)),
                    s=(fft_h, fft_w),
                )
                dose_1mm = conv_full[kH // 2 : kH // 2 + n_roi_1mm, kW // 2 : kW // 2 + n_roi_1mm]
            else:
                dose_1mm = F.conv2d(
                    fluence_roi, kernel.to(device), padding=(kH // 2, kW // 2)
                )[0, 0]

        # ---- 4. Downsample to target res_mm ----------------------------------
        n_out = max(1, int(round(grid_span_mm / res_mm)))
        if n_out % 2 == 0:
            n_out += 1
        if abs(res_mm - 1.0) < 1e-6:
            dose_2d = dose_1mm.cpu().numpy() if isinstance(dose_1mm, torch.Tensor) else dose_1mm
            n_out = n_roi_1mm
        else:
            dose_t = dose_1mm.unsqueeze(0).unsqueeze(0) if isinstance(dose_1mm, torch.Tensor) else torch.from_numpy(dose_1mm).unsqueeze(0).unsqueeze(0)
            dose_2d = (
                F.interpolate(dose_t.float(), size=(n_out, n_out), mode="bilinear", align_corners=False)[0, 0]
                .cpu()
                .numpy()
            )

        pos = (np.arange(n_out) - n_out / 2 + 0.5) * res_mm
        return dose_2d, pos

    # ------------------------------------------------------------------
    # Profile extraction helpers
    # ------------------------------------------------------------------

    def _sample_profile_from_plane(
        self, dose_2d: np.ndarray, pos: np.ndarray, profile: MeasuredProfile
    ) -> np.ndarray:
        axis = profile.axis.upper()
        center = dose_2d.shape[0] // 2
        if axis == "X":
            return dose_2d[center, :]
        if axis == "Y":
            return dose_2d[:, center]
        if axis == "D":
            adj_pos = self._adjust_positions_for_offset(profile)
            diag_mm = adj_pos / np.sqrt(2.0)
            idx = diag_mm / (pos[1] - pos[0]) + center
            coords = np.vstack([idx, idx])
            return scipy.ndimage.map_coordinates(dose_2d, coords, order=1, mode="nearest")
        return dose_2d[:, center]

    def _extract_profiles_from_plane(
        self, dose_2d: np.ndarray, pos: np.ndarray, profiles: List[MeasuredProfile]
    ) -> List[np.ndarray]:
        extracted: List[np.ndarray] = []
        for profile in profiles:
            sim_prof = self._sample_profile_from_plane(dose_2d, pos, profile)
            adj_pos = self._adjust_positions_for_offset(profile)
            if profile.axis.upper() == "D":
                extracted.append(sim_prof)
            else:
                extracted.append(self._interp_if_needed(sim_prof, pos, adj_pos))
        return extracted

    def _simulate_plane_and_extract_profiles(
        self,
        config: MachineConfig,
        profiles: List[MeasuredProfile],
        *,
        res_mm: float = 0.5,
    ) -> List[np.ndarray]:
        if not profiles:
            return []
        ref = profiles[0]
        for profile in profiles[1:]:
            if profile.depth_mm != ref.depth_mm or profile.field_size_mm != ref.field_size_mm:
                raise ValueError(
                    "All profiles must share depth and field size to reuse one dose plane."
                )
        scaled_field = self._field_size_scaled(
            ref.field_size_mm, depth_mm=ref.depth_mm, ssd_mm=ref.ssd_mm
        )
        field_size = max(scaled_field)
        grid_span = max(field_size + 40.0, self.kernel_size_mm)
        dose_2d, pos = self._simulate_dose_plane_fast(
            config,
            scaled_field,
            float(ref.depth_mm),
            res_mm=res_mm,
            grid_span_mm=grid_span,
        )
        return self._extract_profiles_from_plane(dose_2d, pos, profiles)

    def simulate_profiles_for_report(
        self,
        profiles: Sequence[MeasuredProfile],
        *,
        res_mm: float = 1.0,
    ) -> Dict[Tuple[str, int, str, float, float, float, int], np.ndarray]:
        if not profiles:
            return {}
        config_cache: Dict[str, MachineConfig] = {}
        groups: Dict[Tuple[float, float, float, float, str], List[MeasuredProfile]] = {}
        for profile in profiles:
            key = (
                round(float(profile.depth_mm or 0.0), 3),
                round(float(profile.field_size_mm[0]), 3),
                round(float(profile.field_size_mm[1]), 3),
                round(float(profile.ssd_mm or 0.0), 3),
                profile.energy,
            )
            groups.setdefault(key, []).append(profile)

        sim_map: Dict[Tuple[str, int, str, float, float, float, int], np.ndarray] = {}
        for profiles_group in groups.values():
            energy = profiles_group[0].energy
            config = config_cache.get(energy)
            if config is None:
                config = MachineConfig._from_dict(self._config_data, energy=energy)
                config_cache[energy] = config
            sims = self._simulate_plane_and_extract_profiles(
                config, profiles_group, res_mm=res_mm
            )
            for profile, sim in zip(profiles_group, sims):
                key = (
                    profile.energy,
                    int(profile.id),
                    profile.axis.upper(),
                    round(float(profile.depth_mm or 0.0), 3),
                    round(float(profile.field_size_mm[0]), 3),
                    round(float(profile.field_size_mm[1]), 3),
                    int(profile.position_mm.shape[0]),
                )
                sim_map[key] = sim
        return sim_map

    # ------------------------------------------------------------------
    # Step 1: geometric penumbra
    # ------------------------------------------------------------------

    def fit_geometric_penumbra(
        self,
        profiles: List[MeasuredProfile],
        *,
        target_field_mm: Tuple[float, float] = (100.0, 100.0),
        target_depth_mm: float = 100.0,
        plotter: Any | None = None,
    ) -> PenumbraFitResult:
        prof_x = MeasurementParser.find_profile(
            profiles, target_field_mm, target_depth_mm, axis="X"
        )
        prof_y = MeasurementParser.find_profile(
            profiles, target_field_mm, target_depth_mm, axis="Y"
        )
        if not prof_x or not prof_y:
            raise ValueError("Could not find Crossline/Inline profiles for penumbra fitting")

        energy_key = prof_x.energy.replace(" ", "")
        sim_config = MachineConfig._from_dict(self._config_data, energy=prof_x.energy)

        final_values = [
            float(sim_config.geometric_penumbra_mm[0]),
            float(sim_config.geometric_penumbra_mm[1]),
        ]

        def center_and_half(profile: MeasuredProfile) -> Tuple[float, float]:
            fine = _fine_profile(profile.position_mm, profile.dose_values)
            if fine is None:
                return 0.0, max(profile.field_size_mm) / 2.0
            fine_pos, fine_dose = fine
            crossings = np.where(np.diff(np.sign(fine_dose - 50.0)))[0]
            if len(crossings) < 2:
                return 0.0, max(profile.field_size_mm) / 2.0
            left = float(fine_pos[crossings[0]])
            right = float(fine_pos[crossings[-1]])
            return 0.5 * (left + right), 0.5 * (right - left)

        cx, hx = center_and_half(prof_x)
        cy, hy = center_and_half(prof_y)
        nominal_hx = prof_x.field_size_mm[0] / 2.0 if prof_x.field_size_mm[0] > 0 else 1.0
        nominal_hy = prof_y.field_size_mm[1] / 2.0 if prof_y.field_size_mm[1] > 0 else 1.0
        sx = hx / nominal_hx if nominal_hx > 0 else 1.0
        sy = hy / nominal_hy if nominal_hy > 0 else 1.0

        self.jaw_offset_mm = (cx, cy)
        self.jaw_scale = (sx, sy)
        self._log(
            f"Jaw offset: x={self.jaw_offset_mm[0]:.2f} mm y={self.jaw_offset_mm[1]:.2f} mm"
        )
        self._log(f"Jaw scale: x={self.jaw_scale[0]:.2f} y={self.jaw_scale[1]:.2f}")

        target_width_x = calculate_penumbra_width(prof_x.position_mm, prof_x.dose_values)
        target_width_y = calculate_penumbra_width(prof_y.position_mm, prof_y.dose_values)
        self._log(f"Target penumbra: x={target_width_x:.2f} y={target_width_y:.2f}")

        penumbra_res_mm = 1.0
        loss_history: List[float] = []
        eval_count = 0

        def _update_penumbra_plot(
            sim_x: np.ndarray, sim_y: np.ndarray, *, log_message: str | None = None
        ) -> None:
            if plotter is None:
                return
            if log_message:
                self._log(log_message)
            plotter.update_penumbra(
                prof_x.position_mm,
                prof_x.dose_values,
                sim_x,
                prof_y.position_mm,
                prof_y.dose_values,
                sim_y,
                loss_history,
            )
            plotter.update_loss(loss_history)

        def objective(vals: np.ndarray) -> float:
            nonlocal eval_count
            pen_x, pen_y = float(vals[0]), float(vals[1])
            original = sim_config.geometric_penumbra_mm
            sim_config.geometric_penumbra_mm = (pen_x, pen_y)
            try:
                sim_x, sim_y = self._simulate_plane_and_extract_profiles(
                    sim_config, [prof_x, prof_y], res_mm=penumbra_res_mm
                )
            finally:
                sim_config.geometric_penumbra_mm = original
            w_x = calculate_penumbra_width(prof_x.position_mm, sim_x)
            w_y = calculate_penumbra_width(prof_y.position_mm, sim_y)
            loss = (w_x - target_width_x) ** 2 + (w_y - target_width_y) ** 2
            loss_history.append(float(loss))
            eval_count += 1
            if plotter is not None and eval_count % 2 == 0:
                _update_penumbra_plot(
                    sim_x,
                    sim_y,
                    log_message=f"#{eval_count}, Loss: {loss:.6f}, Sigma: [{pen_x:.2f}, {pen_y:.2f}]",
                )
            return loss

        res = minimize(
            objective,
            x0=np.array(final_values, dtype=float),
            bounds=[(0.0, 5.0), (0.0, 5.0)],
            method="L-BFGS-B",
            options={"ftol": 1e-4, "maxiter": 50},
        )
        if res.success:
            final_values = [float(res.x[0]), float(res.x[1])]
            sim_config.geometric_penumbra_mm = (final_values[0], final_values[1])
            self._log(f"Penumbra fit: [{final_values[0]:.2f}, {final_values[1]:.2f}]")
        else:
            self._log(f"Penumbra fit failed: {res.message}")

        self._config_data["energies"][energy_key]["source"]["geometric_penumbra_mm"] = final_values

        tuned_x, tuned_y = self._simulate_plane_and_extract_profiles(
            sim_config, [prof_x, prof_y], res_mm=penumbra_res_mm
        )
        _update_penumbra_plot(
            tuned_x,
            tuned_y,
            log_message=f"Penumbra final: [{final_values[0]:.2f}, {final_values[1]:.2f}]",
        )

        return PenumbraFitResult(
            energy=prof_x.energy,
            target_field_mm=target_field_mm,
            target_depth_mm=target_depth_mm,
            geometric_penumbra_mm=(final_values[0], final_values[1]),
            crossline_meas_pos_mm=prof_x.position_mm,
            crossline_meas_dose=prof_x.dose_values,
            crossline_sim_dose=tuned_x,
            inline_meas_pos_mm=prof_y.position_mm,
            inline_meas_dose=prof_y.dose_values,
            inline_sim_dose=tuned_y,
        )

    # ------------------------------------------------------------------
    # Step 2: off-axis profile correction
    # ------------------------------------------------------------------

    def fit_profile_correction(
        self,
        profiles: List[MeasuredProfile],
        *,
        iterations: int = 2,
        sim_res_mm: float = 3.0,
        plateau_dose_threshold: float = 0.75,
        plateau_position_fraction: float = 0.85,
        diagonal_cutoff_deg: float = 13.0,
        plotter: Any | None = None,
    ) -> ProfileCorrectionResult:
        candidates = [p for p in profiles if p.axis == "D"]
        if candidates:
            target_profile = max(candidates, key=lambda p: p.field_size_mm[0])
        else:
            x_profiles = [p for p in profiles if p.axis == "X" and p.scan_type == "PRO"]
            if not x_profiles:
                raise ValueError("No Diagonal or Crossline profiles found for profile correction")
            target_profile = max(x_profiles, key=lambda p: p.field_size_mm[0])

        sim_config = MachineConfig._from_dict(self._config_data, energy=target_profile.energy)
        sim_config.profile_curve = ((0.0, 1.0), (500.0, 1.0))

        def _apply_diagonal_taper(
            curve: List[Tuple[float, float]]
        ) -> List[Tuple[float, float]]:
            if target_profile.axis.upper() != "D":
                return curve
            ssd_mm = float(target_profile.ssd_mm or 1000.0)
            depth_mm = float(target_profile.depth_mm or 0.0)
            cutoff_mm = np.tan(np.deg2rad(diagonal_cutoff_deg)) * (ssd_mm + depth_mm)
            start_mm = 0.95 * cutoff_mm
            if cutoff_mm <= 0.0 or cutoff_mm <= start_mm:
                return curve
            tapered: List[Tuple[float, float]] = []
            for r, f in curve:
                if r >= cutoff_mm:
                    tapered.append((r, 0.0))
                elif r >= start_mm:
                    frac = 1.0 - (r - start_mm) / (cutoff_mm - start_mm)
                    tapered.append((r, float(f) * max(0.0, frac)))
                else:
                    tapered.append((r, f))
            return tapered

        correction_curve: List[Tuple[float, float]] = [(0.0, 1.0), (500.0, 1.0)]
        loss_history: List[float] = []
        meas_norm = sim_norm = None

        for _ in range(max(1, iterations)):
            sim_dose = self._simulate_plane_and_extract_profiles(
                sim_config, [target_profile], res_mm=sim_res_mm
            )[0]

            cax_idx = int(np.argmin(np.abs(target_profile.position_mm)))
            meas_val = target_profile.dose_values
            sim_val = sim_dose

            meas_norm = meas_val / meas_val[cax_idx]
            sim_norm = sim_val / sim_val[cax_idx]
            loss_history.append(float(np.mean(np.abs(sim_norm - meas_norm))))
            if plotter is not None:
                plotter.update_profile(
                    target_profile.position_mm,
                    meas_norm,
                    sim_norm,
                    title_extra="",
                    axis=target_profile.axis,
                )
                plotter.update_loss(loss_history)

            # Restrict the profile-correction fit to the plateau region so that
            # the penumbra transition of the reference field does not contaminate
            # the correction curve.  Any positional mismatch between the
            # simulated and measured penumbra produces a large spurious
            # meas/sim ratio right at the field edge.  If that artefact is
            # written into the correction curve, it distorts the apparent field
            # width whenever a *different* field's penumbra falls at the same
            # radial distance.
            #
            # Strategy: keep only plateau points (dose > 75 % of CAX) AND
            # points within 85 % of the reference field's half-width.
            plateau_mask = (meas_norm > plateau_dose_threshold) & (sim_norm > plateau_dose_threshold)

            # Compute reference field half-width at the measurement point.
            ssd_ref = float(target_profile.ssd_mm or 1000.0)
            depth_ref = float(target_profile.depth_mm or 0.0)
            mag_ref = (ssd_ref + depth_ref) / 1000.0
            ax = target_profile.axis.upper()
            if ax == "D":
                fx_r = float(target_profile.field_size_mm[0]) * mag_ref
                fy_r = float(target_profile.field_size_mm[1]) * mag_ref
                ref_half_mm = np.sqrt((fx_r / 2.0) ** 2 + (fy_r / 2.0) ** 2)
            elif ax == "X":
                ref_half_mm = float(target_profile.field_size_mm[0]) / 2.0 * mag_ref
            else:
                ref_half_mm = float(target_profile.field_size_mm[1]) / 2.0 * mag_ref

            inside_plateau = np.abs(target_profile.position_mm) <= plateau_position_fraction * ref_half_mm
            mask = plateau_mask & inside_plateau
            valid_pos = np.abs(target_profile.position_mm[mask])
            valid_ratio = meas_norm[mask] / sim_norm[mask]

            sort_idx = np.argsort(valid_pos)
            radial_dist = valid_pos[sort_idx]
            correction_factors = valid_ratio[sort_idx]

            correction_curve_step: List[Tuple[float, float]] = [(0.0, 1.0)]
            last_r = 0.0
            for r, f in zip(radial_dist, correction_factors):
                if r - last_r > 2.0:
                    correction_curve_step.append((float(r), float(f)))
                    last_r = float(r)
            correction_curve_step.append((500.0, correction_curve_step[-1][1]))

            base_r = np.array([p[0] for p in correction_curve], dtype=float)
            base_f = np.array([p[1] for p in correction_curve], dtype=float)
            step_r = np.array([p[0] for p in correction_curve_step], dtype=float)
            step_f = np.array([p[1] for p in correction_curve_step], dtype=float)
            combined = step_f * np.interp(step_r, base_r, base_f)
            correction_curve = list(zip(step_r.tolist(), combined.tolist()))
            correction_curve = _apply_diagonal_taper(correction_curve)
            sim_config.profile_curve = correction_curve

        energy_key = target_profile.energy.replace(" ", "")
        self._config_data["energies"][energy_key]["profile"]["curve"] = correction_curve

        return ProfileCorrectionResult(
            energy=target_profile.energy,
            profile_id=target_profile.id,
            field_size_mm=target_profile.field_size_mm,
            depth_mm=float(target_profile.depth_mm or 0.0),
            axis=target_profile.axis,
            position_mm=target_profile.position_mm,
            meas_norm=meas_norm,
            sim_norm=sim_norm,
            profile_curve=correction_curve,
        )

    # ------------------------------------------------------------------
    # Step 3: output factors / head scatter
    # ------------------------------------------------------------------

    def _calculate_sp_factors(
        self,
        measurements: List[OutputFactorMeasurement],
        energy: str,
        *,
        res_mm: float,
        grid_span_mm: float,
    ) -> List[OutputFactorMeasurement]:
        config = MachineConfig._from_dict(self._config_data, energy=energy)
        config.head_scatter_magnitude = 0.0
        config.output_factor_curve = [(0.0, 1.0), (500.0, 1.0)]

        def get_dose(fx: float, fy: float) -> float:
            scaled_field = self._field_size_scaled((fx, fy), depth_mm=100.0, ssd_mm=900.0)
            dose_2d, _ = self._simulate_dose_plane_fast(
                config,
                scaled_field,
                100.0,
                res_mm=res_mm,
                grid_span_mm=grid_span_mm,
            )
            center = dose_2d.shape[0] // 2
            return float(dose_2d[center, center])

        ref_dose = get_dose(100.0, 100.0)
        for m in measurements:
            d = get_dose(m.field_x_mm, m.field_y_mm)
            m.sp = d / ref_dose
            m.sc_meas = m.value / m.sp
        return measurements

    def _calculate_of_residual_curve(
        self,
        measurements: List[OutputFactorMeasurement],
        amp: float,
        sx_iso: float,
        sy_iso: float,
    ) -> List[List[float]]:
        t10_x = erf(100.0 / (2.0 * np.sqrt(2.0) * sx_iso))
        t10_y = erf(100.0 / (2.0 * np.sqrt(2.0) * sy_iso))
        norm = 1.0 + amp * t10_x * t10_y
        curve: List[List[float]] = []
        for m in measurements:
            tx = erf(m.field_x_mm / (2.0 * np.sqrt(2.0) * sx_iso))
            ty = erf(m.field_y_mm / (2.0 * np.sqrt(2.0) * sy_iso))
            m.sc_model = (1.0 + amp * tx * ty) / norm
            of_model_total = m.sc_model * m.sp
            m.residual = (m.value / of_model_total) if of_model_total > 0 else 1.0
            ratio = max(m.field_x_mm, m.field_y_mm) / (min(m.field_x_mm, m.field_y_mm) + 1e-6)
            if ratio < 1.2:
                equiv = float(np.sqrt(m.field_x_mm * m.field_y_mm))
                val = max(0.95, min(1.05, float(m.residual)))
                curve.append([equiv, val])
        curve.sort(key=lambda x: x[0])
        return curve

    def _select_tail_profiles(
        self,
        profiles: List[MeasuredProfile],
        *,
        axes: Sequence[str],
        depths_mm: Sequence[float],
        fields_mm: Sequence[Tuple[float, float]],
    ) -> List[MeasuredProfile]:
        if not profiles:
            return []
        axes_set = {a.upper() for a in axes} if axes else {"X", "Y"}
        depth_set = {round(float(d)) for d in depths_mm} if depths_mm else None
        field_set = (
            {(round(float(fx)), round(float(fy))) for fx, fy in fields_mm}
            if fields_mm
            else None
        )
        selected: List[MeasuredProfile] = []
        for p in profiles:
            if p.scan_type != "PRO":
                continue
            if p.axis.upper() not in axes_set:
                continue
            if depth_set and p.depth_mm is not None:
                if round(float(p.depth_mm)) not in depth_set:
                    continue
            if field_set:
                fs_key = (round(float(p.field_size_mm[0])), round(float(p.field_size_mm[1])))
                if fs_key not in field_set:
                    continue
            selected.append(p)
        return selected

    def _full_field_log_residuals(
        self,
        config: MachineConfig,
        profiles: List[MeasuredProfile],
        *,
        linear_threshold: float,
        linear_weight: float,
        bands_pct: Sequence[Tuple[float, float]] | None,
        band_weights: Sequence[float] | None,
        sim_res_mm: float,
        sim_cache: Dict[Any, Any] | None,
        return_cache: bool = False,
    ) -> Tuple[
        np.ndarray,
        Dict[Tuple[int, str, int, int, int], Tuple[np.ndarray, np.ndarray]] | None,
    ]:
        if not profiles:
            return np.array([], dtype=float), None
        eps = 1e-6
        diffs: List[float] = []
        plane_cache = None
        plot_cache: (
            Dict[Tuple[int, str, int, int, int], Tuple[np.ndarray, np.ndarray]] | None
        ) = ({} if return_cache else None)
        if sim_cache is not None:
            plane_cache = sim_cache.setdefault("plane_cache", {})

        for p in profiles:
            meas = p.dose_values
            adj_pos = self._adjust_positions_for_offset(p)
            cax_idx = int(np.argmin(np.abs(adj_pos)))
            meas_norm = meas / (meas[cax_idx] if meas[cax_idx] != 0 else meas.max())
            scaled_field = self._field_size_scaled(
                p.field_size_mm, depth_mm=p.depth_mm, ssd_mm=p.ssd_mm
            )

            sim_key = (
                int(p.id),
                p.axis.upper(),
                round(float(p.depth_mm or 0.0), 3),
                round(float(p.field_size_mm[0]), 3),
                round(float(p.field_size_mm[1]), 3),
                round(float(sim_res_mm), 3),
            )
            sim = None
            if sim_cache is not None:
                sim = sim_cache.get(sim_key)
            if sim is None:
                field_size = max(scaled_field)
                field_half = (
                    scaled_field[0] / 2.0 if p.axis.upper() == "X" else scaled_field[1] / 2.0
                )
                sigma_pad = 3.0 * max(config.head_scatter_sigma_mm)
                pos_span = 2.0 * max(float(np.abs(adj_pos).max()), 0.0) + 40.0
                sigma_span = 2.0 * (field_half + sigma_pad + 20.0)
                grid_span = max(field_size + 40.0, pos_span, sigma_span, self.kernel_size_mm)
                plane_key = (
                    round(float(scaled_field[0]), 3),
                    round(float(scaled_field[1]), 3),
                    round(float(p.depth_mm or 0.0), 3),
                    round(float(sim_res_mm), 3),
                )
                cached_plane = (
                    plane_cache.get(plane_key) if plane_cache is not None else None
                )
                if cached_plane is None or cached_plane[2] + 1e-6 < grid_span:
                    dose_2d, pos = self._simulate_dose_plane_fast(
                        config,
                        scaled_field,
                        float(p.depth_mm or 0.0),
                        res_mm=sim_res_mm,
                        grid_span_mm=grid_span,
                    )
                    if plane_cache is not None:
                        plane_cache[plane_key] = (dose_2d, pos, grid_span)
                else:
                    dose_2d, pos = cached_plane[0], cached_plane[1]
                sim = self._sample_profile_from_plane(dose_2d, pos, p)
                if plot_cache is not None:
                    depth_key = int(round(float(p.depth_mm or 0.0)))
                    fx = int(round(float(p.field_size_mm[0])))
                    fy = int(round(float(p.field_size_mm[1])))
                    plot_cache[(int(p.id), p.axis.upper(), depth_key, fx, fy)] = (
                        dose_2d,
                        pos,
                    )
                if sim_cache is not None:
                    sim_cache[sim_key] = sim

            sim = self._interp_if_needed(sim, pos, adj_pos)
            sim_norm = sim / (sim[cax_idx] if sim[cax_idx] != 0 else sim.max())

            mask = (meas_norm > 0) & (sim_norm > 0)
            weights = np.ones_like(meas_norm, dtype=float)
            if bands_pct:
                abs_pos = np.abs(adj_pos)
                width = scaled_field[0] if p.axis.upper() == "X" else scaled_field[1]
                half_width = width / 2.0
                if half_width > 0:
                    pos_pct = (abs_pos / half_width) * 100.0
                    band_mask = np.zeros_like(abs_pos, dtype=bool)
                    if band_weights is not None and len(band_weights) != len(bands_pct):
                        raise ValueError(
                            "band_weights length must match bands_pct length"
                        )
                    for band_idx, (start, end) in enumerate(bands_pct):
                        mask_i = (pos_pct >= start) & (pos_pct <= end)
                        band_mask |= mask_i
                        if band_weights is not None:
                            weights = np.where(
                                mask_i, float(band_weights[band_idx]), weights
                            )
                    mask &= band_mask
            if not np.any(mask):
                continue

            log_diff = np.log(sim_norm[mask] + eps) - np.log(meas_norm[mask] + eps)
            w = weights[mask]
            high_mask = meas_norm[mask] >= linear_threshold
            if np.any(high_mask):
                lin_diff = (
                    sim_norm[mask][high_mask] - meas_norm[mask][high_mask]
                ) * linear_weight
                log_diff = log_diff * w
                log_diff[high_mask] = lin_diff * w[high_mask]
            else:
                log_diff = log_diff * w
            diffs.extend(log_diff.tolist())

        return np.array(diffs, dtype=float), plot_cache

    def _extract_from_cache(
        self,
        plot_cache: (
            Dict[Tuple[int, str, int, int, int], Tuple[np.ndarray, np.ndarray]] | None
        ),
        prof_x: MeasuredProfile,
        prof_y: MeasuredProfile,
        *,
        res_mm: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not plot_cache:
            return self._simulate_plane_and_extract_profiles(
                self._hs_config_for_energy(prof_x.energy), [prof_x, prof_y], res_mm=res_mm
            )
        key_x = (
            int(prof_x.id),
            "X",
            int(round(float(prof_x.depth_mm or 0.0))),
            int(round(float(prof_x.field_size_mm[0]))),
            int(round(float(prof_x.field_size_mm[1]))),
        )
        key_y = (
            int(prof_y.id),
            "Y",
            int(round(float(prof_y.depth_mm or 0.0))),
            int(round(float(prof_y.field_size_mm[0]))),
            int(round(float(prof_y.field_size_mm[1]))),
        )
        dose_x, pos_x = plot_cache.get(key_x, (None, None))
        dose_y, pos_y = plot_cache.get(key_y, (None, None))
        if dose_x is None or dose_y is None:
            return self._simulate_plane_and_extract_profiles(
                self._hs_config_for_energy(prof_x.energy), [prof_x, prof_y], res_mm=res_mm
            )
        sim_x = self._interp_if_needed(
            self._sample_profile_from_plane(dose_x, pos_x, prof_x),
            pos_x,
            self._adjust_positions_for_offset(prof_x),
        )
        sim_y = self._interp_if_needed(
            self._sample_profile_from_plane(dose_y, pos_y, prof_y),
            pos_y,
            self._adjust_positions_for_offset(prof_y),
        )
        return sim_x, sim_y

    def fit_output_factors(
        self,
        measurements: List[OutputFactorMeasurement],
        *,
        energy: str = "10MV",
        tail_profiles: List[MeasuredProfile] | None = None,
        bands_pct: Sequence[Tuple[float, float]] | None = None,
        band_weights: Sequence[float] | None = None,
        axes: Sequence[str] | None = None,
        depths_mm: Sequence[float] | None = None,
        fields_mm: Sequence[Tuple[float, float]] | None = None,
        plateau_window: int = 6,
        plateau_rtol: float = 1e-4,
        plateau_max_restarts: int = 3,
        jitter_amp: float = 0.005,
        jitter_sigma_mm: float = 1.0,
        plotter: Any | None = None,
    ) -> OutputFactorFitResult:
        col_geo = self._config_data.get("collimator_geometry", {})
        z_x = float(col_geo.get("x_jaw_z_mm", 366.0))
        z_y = float(col_geo.get("y_jaw_z_mm", 257.0))
        z_sc = 100.0

        tail_selected = self._select_tail_profiles(
            tail_profiles or [],
            axes=axes or ("X", "Y"),
            depths_mm=depths_mm or (100.0,),
            fields_mm=fields_mm or ((200.0, 200.0),),
        )
        if not tail_selected:
            self._log("Full-field log fit: no profiles matched filters.")

        amp_bounds = (0.03, 0.15)
        sigma_src_bounds_mm = (5.0, 35.0)
        mid_amp = 0.5 * (amp_bounds[0] + amp_bounds[1])
        mid_sig = 0.5 * (sigma_src_bounds_mm[0] + sigma_src_bounds_mm[1])
        x0 = np.array([mid_amp, mid_sig], dtype=float)

        class _RestartFit(Exception):
            pass

        iter_count = 0
        restarts_used = 0
        best_loss = float("inf")
        best_x = x0.copy()
        last_loss = float("inf")
        loss_history: List[float] = []
        plateau_streak = 0

        def _sigmas_from_params(
            params: np.ndarray,
        ) -> Tuple[float, float, float]:
            amp, sig_src = params
            denom_x = z_x - z_sc
            denom_y = z_y - z_sc
            if denom_x <= 1.0 or denom_y <= 1.0:
                return float(amp), 0.0, 0.0
            sx_iso = sig_src * (1000.0 - z_x) / denom_x
            sy_iso = sig_src * (1000.0 - z_y) / denom_y
            return float(amp), float(sx_iso), float(sy_iso)

        def _update_plot_from_cache(
            plot_cache: (
                Dict[Tuple[int, str, int, int, int], Tuple[np.ndarray, np.ndarray]] | None
            ),
        ) -> None:
            if plotter is None or plot_cache is None:
                return
            pos_x_list: List[np.ndarray] = []
            meas_x_list: List[np.ndarray] = []
            sim_x_list: List[np.ndarray] = []
            pos_y_list: List[np.ndarray] = []
            meas_y_list: List[np.ndarray] = []
            sim_y_list: List[np.ndarray] = []
            for p in tail_selected:
                axis = p.axis.upper()
                if axis not in ("X", "Y"):
                    continue
                depth_key = int(round(float(p.depth_mm or 0.0)))
                fx = int(round(float(p.field_size_mm[0])))
                fy = int(round(float(p.field_size_mm[1])))
                key = (int(p.id), axis, depth_key, fx, fy)
                dose_2d, pos = plot_cache.get(key, (None, None))
                if dose_2d is None or pos is None:
                    continue
                sim = self._interp_if_needed(
                    self._sample_profile_from_plane(dose_2d, pos, p),
                    pos,
                    self._adjust_positions_for_offset(p),
                )
                if axis == "X":
                    pos_x_list.append(-np.abs(self._adjust_positions_for_offset(p)))
                    meas_x_list.append(p.dose_values)
                    sim_x_list.append(sim)
                else:
                    pos_y_list.append(np.abs(self._adjust_positions_for_offset(p)))
                    meas_y_list.append(p.dose_values)
                    sim_y_list.append(sim)
            plotter.update_scatter_multi(
                pos_x_list, meas_x_list, sim_x_list,
                pos_y_list, meas_y_list, sim_y_list,
            )

        def _compute_hs_loss(
            params: np.ndarray,
            *,
            return_cache: bool,
            sim_cache: Dict[Any, Any] | None,
        ) -> Tuple[
            float,
            np.ndarray,
            Dict[Tuple[int, str, int, int, int], Tuple[np.ndarray, np.ndarray]] | None,
        ]:
            amp, sx_iso, sy_iso = _sigmas_from_params(params)
            if sx_iso <= 0.0 or sy_iso <= 0.0:
                return 1e9, np.array([1e3], dtype=float), None
            cfg = self._hs_config_for_energy(energy)
            cfg.head_scatter_magnitude = float(amp)
            cfg.head_scatter_sigma_mm = (float(sx_iso), float(sy_iso))
            full_res, plot_cache = self._full_field_log_residuals(
                cfg,
                tail_selected,
                linear_threshold=0.5,
                linear_weight=1.0,
                bands_pct=bands_pct,
                band_weights=band_weights,
                sim_res_mm=3.0,
                sim_cache=sim_cache,
                return_cache=return_cache,
            )
            res_arr = np.array(full_res if full_res.size else [0.0], dtype=float)
            if not np.all(np.isfinite(res_arr)):
                res_arr = np.nan_to_num(res_arr, nan=1e3, posinf=1e3, neginf=-1e3)
            loss = float(np.mean(res_arr**2))
            return loss, res_arr, plot_cache

        def objective(params: np.ndarray) -> float:
            nonlocal iter_count, best_loss, best_x, restarts_used, plateau_streak, last_loss
            loss, _res_arr, plot_cache = _compute_hs_loss(
                params, return_cache=plotter is not None, sim_cache=None
            )
            iter_count += 1
            loss_history.append(loss)
            last_loss = loss
            if loss < best_loss:
                best_loss = loss
                best_x = np.array(params, dtype=float)
            sig_src = float(params[1])
            self._log(
                f"#{iter_count}, Loss: {loss:.6f}, "
                f"Sigma: [{sig_src:.2f}, {sig_src:.2f}], "
                f"Amp: {float(params[0]):.4f}"
            )
            if (
                plateau_window > 0
                and len(loss_history) >= plateau_window
                and restarts_used < plateau_max_restarts
            ):
                recent = loss_history[-plateau_window:]
                spread = max(recent) - min(recent)
                tol = max(plateau_rtol * max(1.0, recent[-1]), 1e-6)
                if spread <= tol:
                    plateau_streak += 1
                    best_x = np.array(params, dtype=float)
                    best_loss = loss
                    restarts_used += 1
                    self._log(
                        f"Plateau detected. Restart {restarts_used}/{plateau_max_restarts}"
                    )
                    raise _RestartFit
                else:
                    plateau_streak = 0
            if plotter is not None:
                plotter.update_loss(loss_history)
                if iter_count % 2 == 0:
                    _update_plot_from_cache(plot_cache)
            return float(loss)

        def probe_loss(params: np.ndarray, *, plot: bool = False) -> float:
            loss, _res_arr, plot_cache = _compute_hs_loss(
                params, return_cache=plot, sim_cache=None
            )
            if plotter is not None and plot:
                plotter.update_loss(loss_history + [loss])
                _update_plot_from_cache(plot_cache)
            return loss

        lower_bounds = np.array([amp_bounds[0], sigma_src_bounds_mm[0]], dtype=float)
        upper_bounds = np.array([amp_bounds[1], sigma_src_bounds_mm[1]], dtype=float)
        minimize_bounds = [
            (amp_bounds[0], amp_bounds[1]),
            (sigma_src_bounds_mm[0], sigma_src_bounds_mm[1]),
        ]

        # Warm-start: probe a few random candidates before running the optimiser.
        try:
            rng = np.random.default_rng(0)
            candidates = [x0] + [rng.uniform(lower_bounds, upper_bounds) for _ in range(6)]
            best_cand = x0
            best_loss = probe_loss(x0, plot=True)
            for cand in candidates[1:]:
                loss = probe_loss(np.array(cand, dtype=float), plot=True)
                if loss < best_loss:
                    best_loss = loss
                    best_cand = np.array(cand, dtype=float)
            x0 = best_cand
            self._log(
                f"Start: amp={x0[0]:.4f}, sigma=[{x0[1]:.2f}, {x0[1]:.2f}], "
                f"loss={best_loss:.6f}"
            )
        except Exception:
            pass

        res = None
        stop_after_plateau = False
        while True:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    module=r"scipy\.optimize\._lsq\.(common|trf)",
                )
                try:
                    res = minimize(
                        objective,
                        x0=x0,
                        bounds=minimize_bounds,
                        method="Powell",
                        options={"maxiter": 200, "xtol": 1e-5, "ftol": 1e-6},
                    )
                    break
                except _RestartFit:
                    if restarts_used > plateau_max_restarts:
                        break
                    base = best_x if np.isfinite(best_loss) else x0
                    jitter_candidates = [
                        base,
                        np.array([base[0] + jitter_amp, base[1] + jitter_sigma_mm]),
                        np.array([base[0] + jitter_amp, base[1] - jitter_sigma_mm]),
                        np.array([base[0] - jitter_amp, base[1] + jitter_sigma_mm]),
                        np.array([base[0] - jitter_amp, base[1] - jitter_sigma_mm]),
                    ]
                    best_cand_j = base
                    best_cand_loss = best_loss
                    for cand in jitter_candidates:
                        cand = np.clip(cand, lower_bounds, upper_bounds)
                        cand_loss = probe_loss(cand, plot=True)
                        if cand_loss < best_cand_loss:
                            best_cand_loss = cand_loss
                            best_cand_j = cand
                    if best_cand_loss >= last_loss - 1e-12:
                        self._log("No improvement after jitter, stopping.")
                        stop_after_plateau = True
                        break
                    x0 = best_cand_j
                    best_loss = best_cand_loss
                    plateau_streak = 0
                    continue
            if stop_after_plateau:
                break

        if res is None and stop_after_plateau:
            res = types.SimpleNamespace(
                x=best_x,
                status=0,
                message="Stopped after plateau without improvement",
                nfev=iter_count,
            )
        elif res is None:
            res = minimize(
                objective,
                x0=x0,
                bounds=minimize_bounds,
                method="Powell",
                options={"maxiter": 1, "xtol": 1e-4, "ftol": 1e-6},
            )

        final_params = best_x if np.isfinite(best_loss) else np.array(res.x, dtype=float)
        amp = float(final_params[0])
        sig_src = float(final_params[1])
        sx_iso = sig_src * (1000.0 - z_x) / (z_x - z_sc)
        sy_iso = sig_src * (1000.0 - z_y) / (z_y - z_sc)

        max_field = (
            max(max(m.field_x_mm, m.field_y_mm) for m in measurements) if measurements else 0.0
        )
        sigma_pad = 3.0 * max(sx_iso, sy_iso)
        grid_span = max(max_field + 2.0 * sigma_pad, self.kernel_size_mm)
        measurements = self._calculate_sp_factors(
            measurements, energy, res_mm=3.0, grid_span_mm=grid_span
        )

        curve = self._calculate_of_residual_curve(measurements, amp, sx_iso, sy_iso)
        energy_key = energy.replace(" ", "")
        self._config_data["energies"][energy_key]["source"]["head_scatter_magnitude"] = amp
        self._config_data["energies"][energy_key]["source"]["head_scatter_sigma_mm"] = [sx_iso, sy_iso]
        self._config_data["energies"][energy_key]["output_factors"]["curve"] = curve

        if plotter is not None:
            meas_sq = [m for m in measurements if abs(m.field_x_mm - m.field_y_mm) <= 1.0]
            meas_sq.sort(key=lambda m: m.field_x_mm)
            sizes = np.array([m.field_x_mm for m in meas_sq])
            of_meas_arr = np.array([m.value for m in meas_sq])
            of_model_arr = np.array([m.sc_model * m.sp for m in meas_sq])
            plotter.update_of_residual(sizes, of_meas_arr, of_model_arr)

        return OutputFactorFitResult(
            energy=energy,
            head_scatter_magnitude=amp,
            head_scatter_sigma_mm=(sx_iso, sy_iso),
            output_factor_curve=curve,
            measurements=measurements,
        )
