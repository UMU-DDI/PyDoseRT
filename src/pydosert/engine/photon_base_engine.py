"""
Abstract base class for differentiable photon dose-calculation engines.

PhotonBaseEngine provides the pipeline-agnostic machinery shared by photon
engines: construction and device/dtype handling, input validation, the
``compute_dose`` orchestration, beam chunking with gradient checkpointing and
geometry caching, and reference calibration.

Concrete engines subclass it and implement the pipeline-specific hooks:
    - ``_initialize_layers``:    build the engine's layers from a beam template
    - ``_full_geometry``:        the geometry context for the full beam set
    - ``_build_chunk_geometry``: per-chunk geometry contexts for chunked dose
    - ``_forward_core``:         run the pipeline for one (full or partial) beam set

The "geometry context" is an opaque object produced by ``_full_geometry`` /
``_build_chunk_geometry`` and consumed by ``_forward_core``. The base class never
inspects it, so each engine is free to decide what it contains (e.g. the
beam-count-dependent layers that must vary per chunk).
"""
from dataclasses import replace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from pydosert.data import MachineConfig, Beam, BeamSequence
from pydosert.exceptions import DeviceDtypeError, EngineStateError, ShapeError


class PhotonBaseEngine(nn.Module):
    """
    Base class implementing the shared scaffolding for photon dose engines.

    Subclass this and implement ``_initialize_layers``, ``_full_geometry``,
    ``_build_chunk_geometry`` and ``_forward_core`` to define a concrete engine.

    Attributes:
        machine_config (MachineConfig): Machine physics parameters.
        device (torch.device): PyTorch device for computation.
    """
    machine_config: MachineConfig | None = None
    dose_grid_shape: tuple[int, int, int] | None = None
    dose_grid_spacing: tuple[float, float, float] | None = None
    number_of_beams: int | None = None
    layers_initialized: bool = False
    gantry_angles: torch.Tensor | None = None
    collimator_angles: torch.Tensor | None = None
    field_size: tuple[float, float] | None = None
    SID: float | None = None
    iso_center: tuple[float, float, float] | None = None

    def __init__(
        self,
        machine_config: MachineConfig,
        kernel_size: int,
        dose_grid_spacing: tuple[float, float, float],
        dose_grid_shape: tuple[int, int, int],
        beam_template: BeamSequence | Beam | None = None,
        auto_calibrate: bool = False,
        adjust_values: bool = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = None,
        verbose: bool = False,
        beam_chunk_size: int | None = None,
    ) -> "PhotonBaseEngine":
        """
        Initializes the engine.

        Args:
            machine_config: Machine physics and MLC specifications.
            kernel_size: Size of the pencil beam dose kernel.
            dose_grid_spacing: Voxel spacing in mm (depth, height, width).
            dose_grid_shape: Shape of the output grid tensor (depth, height, width) in pixels.
            beam_template: Optional Beam or BeamSequence defining the treatment geometry.
                If omitted, the engine is unconfigured until the first compute_dose call.
            auto_calibrate: Run calibration immediately after construction (default: False).
            adjust_values: Deprecated adjustment of beam parameters.
            device: PyTorch device for computation.
            dtype: Data type for tensors.
            verbose: Enable verbose output (default: False).
            beam_chunk_size: Default number of beams processed per gradient-checkpointed
                chunk in compute_dose. None (default) processes all beams in a single
                pass (lowest runtime, highest peak memory). Set a positive value to trade
                runtime for lower peak memory on large problems; this can be overridden per
                call. The per-chunk beam geometry is cached and reused across calls.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.beam_chunk_size = beam_chunk_size
        self._chunk_geometry_cache = None
        self._chunk_geometry_cache_key = None

        # Resolve device and dtype now rather than on the first forward pass: layers
        # allocate buffers at construction, so leaving them unset silently builds the
        # geometry on one device and the physics on another.
        self.device = self._resolve_device(device, beam_template)
        self.dtype = self._resolve_dtype(dtype, beam_template)
        self.verbose = verbose

        self.machine_config = machine_config
        self.dose_grid_spacing = dose_grid_spacing
        self.dose_grid_shape = dose_grid_shape
        self._initialize_layers(beam_template)

        if adjust_values is not None:
            raise ValueError("The `adjust_values` argument, together with the beam validation layer has been removed due to major limitations.")

        if auto_calibrate:
            self.calibrate(verbose=verbose)
            self._initialize_layers(beam_template)

    @staticmethod
    def _resolve_device(device, beam_template) -> torch.device:
        """Device to build on: the explicit one, else the template's, else CUDA if present.

        Args:
            device (torch.device | str | None): Explicitly requested device, or None.
            beam_template (BeamSequence | Beam | None): Template to borrow from.

        Returns:
            torch.device: The device the engine will build its layers on.
        """
        if device is None:
            device = (beam_template.device if beam_template is not None
                      else ("cuda" if torch.cuda.is_available() else "cpu"))
        # Resolve through a tensor so "cuda" becomes "cuda:0": the two are the same
        # device but compare unequal, which would reject every input.
        return torch.empty(0, device=device).device

    @staticmethod
    def _resolve_dtype(dtype, beam_template) -> torch.dtype:
        """Dtype to build in: the explicit one, else the template's, else float32.

        Args:
            dtype (torch.dtype | None): Explicitly requested dtype, or None.
            beam_template (BeamSequence | Beam | None): Template to borrow from.

        Returns:
            torch.dtype: The floating dtype the engine will compute in.

        Raises:
            DeviceDtypeError: If the requested dtype is not a floating type.
        """
        if dtype is None:
            dtype = beam_template.dtype if beam_template is not None else torch.float32
        dtype = torch.empty(0, dtype=dtype).dtype
        if not dtype.is_floating_point:
            raise DeviceDtypeError(f"dtype must be a floating point type, got {dtype}.")
        return dtype

    def _check_inputs_match_engine(self, *tensors: torch.Tensor) -> None:
        """Reject inputs that are not on the engine's device and dtype.

        Args:
            *tensors (torch.Tensor): Inputs to check; None entries are skipped.

        Raises:
            DeviceDtypeError: If any input disagrees with the engine.
        """
        for tensor in tensors:
            if tensor is None:
                continue
            if tensor.device != self.device:
                raise DeviceDtypeError(
                    f"Input is on {tensor.device} but the engine was built on {self.device}. "
                    "Move it with tensor.to(engine.device), or build the engine with "
                    f"device='{tensor.device}'.")
            if tensor.is_floating_point() and tensor.dtype != self.dtype:
                raise DeviceDtypeError(
                    f"Input has dtype {tensor.dtype} but the engine computes in {self.dtype}. "
                    "Cast it with tensor.to(engine.dtype), or build the engine with "
                    f"dtype={tensor.dtype}.")

    @property
    def iso_center_voxel(self) -> tuple[int, int, int]:
        """Isocenter location as clamped voxel indices.

        Returns:
            tuple[int, int, int]: Isocenter index (ix, iy, iz) into the
                [D, H, W] dose grid, or None if no isocenter has been set.
        """
        if self.iso_center is None:
            return None

        sx, sy, sz = self.dose_grid_shape
        rx, ry, rz = self.dose_grid_spacing
        X, Y, Z = self.iso_center  # physical coords, origin at isocenter corner
        X_center, Y_center, Z_center = (X, Y, Z)

        # Convert physical coords to voxel indices and round to nearest voxel
        ix = int(X_center / rx)
        iy = int(Y_center / ry)
        iz = int(Z_center / rz)

        # Optionally clamp to valid voxel range
        ix = max(0, min(sx - 1, ix))
        iy = max(0, min(sy - 1, iy))
        iz = max(0, min(sz - 1, iz))

        return (ix, iy, iz)

    def _assert_sizes(self, density_image, leaf_positions, jaw_positions, mus, fluence_maps=None):
        """Validate input tensor shapes, devices and dtypes for a forward pass.

        Args:
            density_image (torch.Tensor): CT/density volume [B, D, H, W].
            leaf_positions (torch.Tensor | None): Leaf positions [B, G, N, 2].
                Required unless fluence_maps is provided.
            jaw_positions (torch.Tensor | None): Jaw positions [B, G, 2], or None.
            mus (torch.Tensor | None): Monitor units [B, G]. Optional when
                fluence_maps is provided.
            fluence_maps (torch.Tensor | None): Pre-computed fluence maps
                [B, G, H, W] or [B*G, H, W]. When given, leaf/jaw positions are
                not validated.

        Where B is the batch size, G the number of beams, N the number of leaf
        pairs, and (D, H, W) the dose-grid shape.
        """

        G = self.number_of_beams

        if fluence_maps is not None:
            # Derive B from fluence_maps; mus is optional in this path
            fm_h, fm_w = self.field_size
            if fluence_maps.dim() == 4:
                B = fluence_maps.shape[0]
                expected_fm = (B, G, fm_h, fm_w)
                if not (fluence_maps.shape == expected_fm):
                    raise ShapeError(f"Fluence maps shape mismatch: expected {expected_fm}, got {fluence_maps.shape}")
            elif fluence_maps.dim() == 3:
                if not (fluence_maps.shape[0] % G == 0):
                    raise ShapeError(f"Fluence maps leading dim {fluence_maps.shape[0]} is not divisible by G={G}")
                B = fluence_maps.shape[0] // G
                expected_fm = (B * G, fm_h, fm_w)
                if not (fluence_maps.shape == expected_fm):
                    raise ShapeError(f"Fluence maps shape mismatch: expected {expected_fm}, got {fluence_maps.shape}")
            else:
                raise ValueError(
                    f"fluence_maps must be 3D [B*G, H, W] or 4D [B, G, H, W], got {fluence_maps.dim()}D"
                )

            # Validate mus only when provided
            if mus is not None:
                if not (mus.dim() == 2):
                    raise ShapeError(f"MUs needs 2 dimensions [B, G], got {mus.dim()}D: {mus.shape}")
                expected_mus = (B, G)
                if not (mus.shape == expected_mus):
                    raise ShapeError(f"MUs shape mismatch: expected {expected_mus}, got {mus.shape}")

            devices = {fluence_maps.device}
            dtypes = {fluence_maps.dtype}
            if mus is not None:
                devices.add(mus.device)
                dtypes.add(mus.dtype)
        else:
            B = leaf_positions.shape[0]
            if not (leaf_positions.dim() == 4):
                raise ShapeError(f"Leaf positions needs 4 dimensions [B, 2, CP, N], got {leaf_positions.dim()}D: {leaf_positions.shape}")
            if not (mus.dim() == 2):
                raise ShapeError(f"MUs needs 2 dimensions [B, CP], got {mus.dim()}D: {mus.shape}")

            if not (leaf_positions.shape[0] == B and mus.shape[0] == B):
                raise ShapeError(f"Batch size mismatch: ct={B}, leaf_positions={leaf_positions.shape[0]}, mus={mus.shape[0]}")

            expected_leaf = (B, G, self.machine_config.number_of_leaf_pairs, 2)
            if not (leaf_positions.shape == expected_leaf):
                raise ShapeError(f"Leaf positions shape mismatch: expected {expected_leaf}, got {leaf_positions.shape}")

            expected_mus = (B, G)
            if not (mus.shape == expected_mus):
                raise ShapeError(f"MUs shape mismatch: expected {expected_mus}, got {mus.shape}")

            if jaw_positions is not None:
                if not (jaw_positions.dim() == 3):
                    raise ShapeError(f"Jaw positions needs 3 dimensions [B, 2, CP], got {jaw_positions.dim()}D: {jaw_positions.shape}")

                if not (jaw_positions.shape[0] == B):
                    raise ShapeError(f"Batch size mismatch: ct={B}, jaw_positions={jaw_positions.shape[0]}")

                expected_jaw = (B, G, 2)
                if not (jaw_positions.shape == expected_jaw):
                    raise ShapeError(f"Jaw positions shape mismatch: expected {expected_jaw}, got {jaw_positions.shape}")

            devices = {leaf_positions.device, mus.device}
            if jaw_positions is not None:
                devices.add(jaw_positions.device)
            dtypes = {leaf_positions.dtype, mus.dtype}
            if jaw_positions is not None:
                dtypes.add(jaw_positions.dtype)

        if density_image is None:
            raise ValueError("CT image must be provided.")
        if not (density_image.dim() == 4):
            raise ShapeError(f"CT image needs 4 dimensions [B, D, H, W], got {density_image.dim()}D: {density_image.shape}")

        expected_ct = (B, *self.dose_grid_shape)
        if not (density_image.shape == expected_ct):
            raise ShapeError(f"CT shape mismatch: expected {expected_ct}, got {density_image.shape}")

        devices.add(density_image.device)
        dtypes.add(density_image.dtype)

        if len(devices) != 1:
            raise DeviceDtypeError(
                f"All inputs must be on one device, got {sorted(str(d) for d in devices)}. "
                "Move them with tensor.to(engine.device).")

        if len(dtypes) != 1:
            raise DeviceDtypeError(
                f"All inputs must share one dtype, got {sorted(str(d) for d in dtypes)}. "
                "Cast them with tensor.to(engine.dtype).")

    def forward(
        self,
        leaf_positions: torch.Tensor | None,
        mus: torch.Tensor | None,
        jaw_positions: torch.Tensor | None,
        density_image: torch.Tensor,
        return_intermediates: bool = False,
        fluence_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Runs the full dose calculation pipeline over all beams.

        Args:
            leaf_positions: Leaf positions [B, G, N, 2]. Not required when fluence_maps is provided.
            mus: Monitor units [B, G]. Optional when fluence_maps is provided; if supplied the dose
                is scaled by MUs, if omitted the fluence maps are used as-is.
            jaw_positions: Jaw positions [B, G, 2]. Not required when fluence_maps is provided.
            density_image: CT image tensor [B, D, H, W].
            return_intermediates: If True, also return intermediate tensors.
            fluence_maps: Optional pre-computed fluence maps [B, G, H, W] or [B*G, H, W].
                If provided, the FluenceMapLayer is skipped and leaf_positions/jaw_positions
                are ignored. The maps are used directly as input to the FluenceVolumeLayer.

        Returns:
            Dose tensor [B, D, H, W].
        """
        self._check_inputs_match_engine(fluence_maps, leaf_positions, jaw_positions, mus, density_image)

        if not self.layers_initialized:
            raise EngineStateError(
                "Layers have not been initialized, so no dose can be computed. Pass a beam "
                "template to the constructor, or call compute_dose(beam_input, ...) which "
                "initializes them from the beams.")

        self._assert_sizes(density_image, leaf_positions, jaw_positions, mus, fluence_maps=fluence_maps)

        return self._forward_core(
            leaf_positions,
            mus,
            jaw_positions,
            density_image,
            self._full_geometry(),
            self.collimator_angles,
            self.number_of_beams,
            return_intermediates,
            fluence_maps,
        )

    def compute_dose(
        self,
        beam_input: BeamSequence | Beam,
        density_image: torch.Tensor | None = None,
        return_intermediates: bool = False,
        overwrite: bool = False,
        fluence_maps: torch.Tensor | None = None,
        beam_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """
        Compute dose from a BeamSequence or Beam.

        Args:
            beam_input: BeamSequence (shapes: mus [CP], leaf_positions [CP, N, 2], jaw_positions [CP, 2])
                or a single Beam. Always required for geometry (gantry angles, iso center, etc.).
            density_image: CT image tensor [1, D, H, W].
            return_intermediates: If True, also return intermediate tensors.
            overwrite: Re-initialize layers even if already set up.
            fluence_maps: Optional pre-computed fluence maps [1, G, H, W] or [G, H, W].
                If provided, the FluenceMapLayer is skipped and leaf/jaw positions from
                beam_input are ignored. G must equal the number of beams in beam_input.
            beam_chunk_size: Number of beams per gradient-checkpointed chunk for this call,
                overriding the engine default set at construction. When set (and the beam
                count exceeds it) the beams are processed in chunks whose intermediates are
                recomputed one chunk at a time during backward, lowering peak memory while
                preserving gradients. The per-chunk geometry is cached and reused across
                calls with the same beam layout. Ignored when return_intermediates is True.

        Returns:
            Dose tensor [1, D, H, W].
        """
        self._initialize_layers(beam_input, overwrite)

        # Add batching dimension to parameters
        if density_image is not None:
            ct_tensor = density_image
            if ct_tensor.dim() == 3:
                ct_tensor = ct_tensor.unsqueeze(0)
        else:
            ct_tensor = None

        if isinstance(beam_input, Beam):
            leaf_positions = beam_input.leaf_positions.unsqueeze(0).unsqueeze(0)
            mus = beam_input.mu.unsqueeze(0).unsqueeze(0)
            jaw_positions = beam_input.jaw_positions.unsqueeze(0).unsqueeze(0)
        elif isinstance(beam_input, BeamSequence):
            leaf_positions = beam_input.leaf_positions.unsqueeze(0)
            mus = beam_input.mus.unsqueeze(0)
            jaw_positions = beam_input.jaw_positions.unsqueeze(0)

        # Normalise fluence_maps to [1, G, H, W] so forward() can reshape to [B*G, H, W]
        if fluence_maps is not None and fluence_maps.dim() == 3:
            fluence_maps = fluence_maps.unsqueeze(0)  # [G, H, W] -> [1, G, H, W]

        chunk_size = beam_chunk_size if beam_chunk_size is not None else self.beam_chunk_size
        if (
            chunk_size is not None
            and not return_intermediates
            and self.number_of_beams > chunk_size
        ):
            return self._compute_dose_chunked(
                leaf_positions=leaf_positions,
                mus=mus,
                jaw_positions=jaw_positions,
                density_image=ct_tensor,
                fluence_maps=fluence_maps,
                chunk_size=chunk_size,
                overwrite=overwrite,
            )

        return self.forward(
            leaf_positions=leaf_positions,
            mus=mus,
            jaw_positions=jaw_positions,
            density_image=ct_tensor,
            return_intermediates=return_intermediates,
            fluence_maps=fluence_maps,
        )

    def _chunk_geometry_key(self, chunk_size: int) -> tuple:
        """Build the cache key identifying the per-chunk geometry contexts.

        Args:
            chunk_size (int): Number of beams per chunk.

        Returns:
            tuple: Hashable key combining chunk size, beam count, gantry angles,
                isocenter, dose-grid shape/spacing, device and dtype.
        """
        return (
            chunk_size,
            self.number_of_beams,
            tuple(self.gantry_angles.detach().cpu().tolist()),
            tuple(self.iso_center),
            tuple(self.dose_grid_shape),
            tuple(self.dose_grid_spacing),
            str(self.device),
            str(self.dtype),
        )

    def _compute_dose_chunked(
        self,
        leaf_positions: torch.Tensor,
        mus: torch.Tensor,
        jaw_positions: torch.Tensor | None,
        density_image: torch.Tensor,
        fluence_maps: torch.Tensor | None,
        chunk_size: int,
        overwrite: bool,
    ) -> torch.Tensor:
        """
        Compute dose in beam chunks to limit backward-pass peak memory.

        Each chunk is independently checkpointed, so only one chunk's intermediates
        are recomputed at a time during backward. The per-chunk geometry contexts are
        cached so they are not rebuilt on every call.

        Args:
            leaf_positions (torch.Tensor): Leaf positions [B, G, N, 2].
            mus (torch.Tensor): Monitor units [B, G].
            jaw_positions (torch.Tensor | None): Jaw positions [B, G, 2], or None.
            density_image (torch.Tensor): CT/density volume [B, D, H, W].
            fluence_maps (torch.Tensor | None): Pre-computed fluence maps [B, G, H, W], or None.
            chunk_size (int): Number of beams processed per checkpointed chunk.
            overwrite (bool): Force a rebuild of the cached per-chunk geometry.

        Returns:
            torch.Tensor: Dose tensor [B, D, H, W] summed over all beams.
        """
        key = self._chunk_geometry_key(chunk_size)
        if overwrite or self._chunk_geometry_cache_key != key:
            self._chunk_geometry_cache = self._build_chunk_geometry(chunk_size)
            self._chunk_geometry_cache_key = key

        dose = None
        for start, end, geometry in self._chunk_geometry_cache:
            jaw_chunk = jaw_positions[:, start:end] if jaw_positions is not None else None
            fluence_chunk = fluence_maps[:, start:end] if fluence_maps is not None else None
            chunk_dose = checkpoint(
                self._forward_core,
                leaf_positions[:, start:end],
                mus[:, start:end],
                jaw_chunk,
                density_image,
                geometry,
                self.collimator_angles[start:end],
                end - start,
                False,
                fluence_chunk,
                use_reentrant=False,
            )
            dose = chunk_dose if dose is None else dose + chunk_dose
        return dose

    def calibrate(self,
                  calibration_mu: float = None,
                  original_beam_template: BeamSequence | None = None, # Deprecated setting
                  verbose: bool = True) -> None:
        """
        Calibrates the model by normalizing the output dose so that the pre-defined MU value corresponds to 1Gy.

        Args:
            calibration_mu: The MU where the delivered dose should correspond to 1Gy in water at 10cm depth.
            original_beam_template: A depracated argument for setting the template back to the engine.
            verbose: Enable verbose output (default: False).

        Returns:
            None
        """
        if self.machine_config is None:
            raise EngineStateError(
                "Machine physics is required to calibrate: machine_config is None. Set it on the engine before calling calibrate()."
            )
        if self.dose_grid_shape is None:
            raise EngineStateError(
                "The dose grid shape is required to calibrate: dose_grid_shape is None. Set it on the engine before calling calibrate()."
            )
        if self.dose_grid_spacing is None:
            raise EngineStateError(
                "The dose grid spacing is required to calibrate: dose_grid_spacing is None. Set it on the engine before calling calibrate()."
            )

        if original_beam_template is not None:
            print("The argument `original_beam_template` is now deprecated and will not be used for calibration")
        center_x, _, center_z = torch.tensor(self.dose_grid_spacing) * (torch.tensor(self.dose_grid_shape)) / 2
        iso_center = (center_x.item(), 100.0, center_z.item())
        beam = Beam.create(0.0, self.machine_config.number_of_leaf_pairs, 0.0, (100.0, 100.0), iso_center=iso_center, device=self.device, dtype=self.dtype)
        if calibration_mu is None:
            calibration_mu = self.machine_config.calibration_mu

        beam = replace(beam, mu=calibration_mu * beam.mu)
        water_attenuation = torch.ones(self.dose_grid_shape).to(self.device).to(self.dtype)

        self.layers_initialized = False

        dose = self.compute_dose(
            beam,
            density_image=water_attenuation,
            overwrite=True
        )

        # Get center dose (at 10cm depth - index 50 for 100 voxels)
        center_dose = dose[0, *self.iso_center_voxel].detach().cpu().numpy().item()

        # Calculate calibration factor
        # This gives the factor to normalize to 1 Gy per MU at reference conditions
        calibration_factor = self.machine_config.mean_photon_energy_MeV / center_dose

        if abs(center_dose - 1.0) > 0.001:
            if verbose:
                print(f"Calibration failed. Adjusting calibration factor to: {calibration_factor}")
            self.machine_config.mean_photon_energy_MeV = calibration_factor

        # Reset layers to apply new beam sequence
        self.layers_initialized = False

    # ------------------------------------------------------------------
    # Pipeline-specific hooks (implemented by concrete engines)
    # ------------------------------------------------------------------

    def _initialize_layers(self, new_beam_data: BeamSequence | Beam, overwrite: bool = False) -> None:
        """
        Build (or refresh) the engine's layers from a beam template.

        Implementations must set the geometry attributes the base class relies on
        (number_of_beams, gantry_angles, collimator_angles, field_size, SID,
        iso_center) and set ``self.layers_initialized = True`` once ready.
        """
        raise NotImplementedError

    def _full_geometry(self):
        """
        Return the geometry context for the full beam set.

        The returned object is passed unchanged to ``_forward_core``; the base class
        does not inspect it.
        """
        raise NotImplementedError

    def _build_chunk_geometry(self, chunk_size: int) -> list[tuple[int, int, object]]:
        """
        Build the per-chunk geometry contexts for chunked dose computation.

        Returns a list of ``(start, end, geometry)`` tuples, one per beam chunk,
        where ``geometry`` is the context passed to ``_forward_core`` for that chunk.
        Implementations should cache-friendly: the result is stored and reused across
        calls with the same beam layout (see ``_chunk_geometry_key``).
        """
        raise NotImplementedError

    def _forward_core(
        self,
        leaf_positions: torch.Tensor | None,
        mus: torch.Tensor | None,
        jaw_positions: torch.Tensor | None,
        density_image: torch.Tensor,
        geometry: object,
        collimator_angles: torch.Tensor,
        number_of_beams: int,
        return_intermediates: bool = False,
        fluence_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run the dose pipeline for a (possibly partial) set of beams.

        ``geometry`` is the context produced by ``_full_geometry`` (full beam set) or
        ``_build_chunk_geometry`` (a single chunk). ``number_of_beams`` is the number
        of beams in this call, which may be smaller than ``self.number_of_beams`` when
        chunking. Must return a dose tensor [B, D, H, W] summed over the given beams.
        """
        raise NotImplementedError