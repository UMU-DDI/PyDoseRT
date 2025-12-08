"""
Beam and BeamSequence data structures for radiotherapy treatment planning.

These classes provide a clean abstraction for beam parameters while maintaining
full differentiability for deep learning workflows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, TYPE_CHECKING, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from pydose_rt.data import MachineConfig
    from pydose_rt.engine.dose_engine import DoseEngine


@dataclass
class Beam:
    """
    A single control point (beam) in a treatment arc for a single sample.

    This is the atomic unit for beam representation - NO batch dimension.
    For batched processing, stack multiple BeamSequences.

    Attributes:
        gantry_angle: Gantry angle in radians
        mu: Monitor units (scalar tensor)
        leaf_positions: MLC leaf positions [N, 2] where 2=(left, right)
        jaw_positions: Jaw positions [2] where 2=(lower, upper)
    """
    gantry_angle: float  # radians
    collimator_angle: float # radians
    mu: torch.Tensor     # scalar or [1]
    leaf_positions: torch.Tensor  # [N, 2]
    jaw_positions: torch.Tensor   # [2]
    field_size: tuple[int, int] = (400, 400)
    iso_center: tuple[float, float, float] = (0, 0, 0)
    sid: float = 1000.0
    ssd: float = None

    @classmethod
    def create(
        cls,
        gantry_angle_deg: float,
        number_of_leaf_pairs: int,
        collimator_angle_deg:float = 0.0,
        field_size_mm: tuple[int, int] = (400, 400),
        iso_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        device: torch.device | str = 'cuda',
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = True,
    ) -> Beam:
        """
        Create a single beam at a specific gantry angle.

        Args:
            gantry_angle_deg: Gantry angle in degrees            
            number_of_leaf_pairs: Number of MLC leaf pairs
            field_size_mm: Field size (width, height) in mm. Default (400, 400)
            iso_center: Isocenter position (x, y, z) in mm. Default (0, 0, 0)0)
            device: PyTorch device
            dtype: Data type for tensors
            requires_grad: Whether tensors require gradients (for optimization)

        Returns:
            Beam with initialized parameters (fully open field)

        Example:            
            >>> beam = Beam.create(90.0, number_of_leaf_pairs=60, requires_grad=True)
            >>> dose = dose_engine.compute_single_beam(beam, ct_image)
            >>> loss.backward()  # Gradients flow to beam parameters
        """
        field_w, field_h = field_size_mm

        # Initialize leaves at field edges (fully open) [N, 2]
        leaf_positions = torch.zeros(number_of_leaf_pairs, 2, device=device, dtype=dtype)
        leaf_positions[:, 0] = -field_w / 2  # Left leaves
        leaf_positions[:, 1] = field_w / 2   # Right leaves

        # Initialize jaws at field edges [2]
        jaw_positions = torch.zeros(2, device=device, dtype=dtype)
        jaw_positions[0] = -field_h / 2  # Lower jaw
        jaw_positions[1] = field_h / 2   # Upper jaw

        # MU initialized to 1.0 (scalar)
        mu = torch.ones(1, device=device, dtype=dtype).squeeze()

        if requires_grad:
            leaf_positions = leaf_positions.requires_grad_(True)
            jaw_positions = jaw_positions.requires_grad_(True)
            mu = mu.requires_grad_(True)

        return cls(
            gantry_angle=math.radians(gantry_angle_deg),
            collimator_angle=math.radians(collimator_angle_deg),
            mu=mu,
            ssd=1000.0,
            leaf_positions=leaf_positions,
            jaw_positions=jaw_positions,
            iso_center=iso_center
        )

    @property
    def gantry_angle_deg(self) -> float:
        """Gantry angle in degrees."""
        return math.degrees(self.gantry_angle)

    @property
    def device(self) -> torch.device:
        """Device of the underlying tensors."""
        return self.leaf_positions.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the underlying tensors."""
        return self.leaf_positions.dtype

    @property
    def num_leaf_pairs(self) -> int:
        """Number of MLC leaf pairs."""
        return self.leaf_positions.shape[0]

    @property
    def requires_grad(self) -> bool:
        """Whether any tensor requires gradients."""
        return (
            self.leaf_positions.requires_grad or
            self.jaw_positions.requires_grad or
            self.mu.requires_grad
        )

    def detach(self) -> Beam:
        """Return a new Beam with detached tensors (no gradient tracking)."""
        return Beam(
            gantry_angle=self.gantry_angle,
            collimator_angle=self.collimator_angle,
            mu=self.mu.detach(),
            leaf_positions=self.leaf_positions.detach(),
            jaw_positions=self.jaw_positions.detach(),
            field_size=self.field_size,
            iso_center=self.iso_center,
            sid=self.sid,
            ssd=self.ssd
        )

    def clone(self) -> Beam:
        """Return a deep copy of this Beam."""
        return Beam(
            gantry_angle=self.gantry_angle,
            collimator_angle=self.collimator_angle,
            mu=self.mu.clone(),
            leaf_positions=self.leaf_positions.clone(),
            jaw_positions=self.jaw_positions.clone(),
            field_size=self.field_size,
            iso_center=self.iso_center,
            sid=self.sid,
            ssd=self.ssd
        )

    def to(self, device: torch.device | str) -> Beam:
        """Move beam tensors to a different device."""
        return Beam(
            gantry_angle=self.gantry_angle,
            collimator_angle=self.collimator_angle,
            mu=self.mu.to(device),
            leaf_positions=self.leaf_positions.to(device),
            jaw_positions=self.jaw_positions.to(device),
            field_size=self.field_size,
            iso_center=self.iso_center,
            sid=self.sid,
            ssd=self.ssd
        )


@dataclass
class BeamSequence:
    """
    A sequence of control points (beams) for a single treatment arc.

    NO batch dimension - represents a single sample's treatment.
    For batched processing, use BeamSequence.stack() to combine multiple sequences.

    When you index into a BeamSequence (e.g., `beam_seq[0]`), you get a `Beam`
    whose tensors are VIEWS into the original data - gradients flow back.

    Attributes:
        mus: Monitor units [CP]
        leaf_positions: MLC positions [CP, N, 2] where 2=(left, right)
        jaw_positions: Jaw positions [CP, 2] where 2=(lower, upper)
        gantry_angles: Gantry angles in radians [CP], or None to use engine's angles

    Example - From DICOM:
        >>> beam_seq = BeamSequence.from_treatment_config(treatment_config)
        >>> for beam in beam_seq:
        ...     print(f"Beam at {beam.gantry_angle_deg}deg")

    Example - Batching multiple sequences:
        >>> sequences = [seq1, seq2, seq3]
        >>> batched_leafs, batched_mus, batched_jaws = BeamSequence.stack(sequences)
        >>> dose = engine.forward(batched_leafs, batched_mus, batched_jaws, ct_batch)
    """
    mus: torch.Tensor             # [CP]
    leaf_positions: torch.Tensor  # [CP, N, 2]
    jaw_positions: torch.Tensor   # [CP, 2]
    field_size: tuple[int, int]
    iso_center: tuple[float, float, float]
    sid: float
    gantry_angles: Optional[torch.Tensor] = None  # [CP] in radians, or None to use engine's
    collimator_angles: Optional[torch.Tensor] = None

    @property
    def has_gantry_angles(self) -> bool:
        """Whether this BeamSequence has explicit gantry angles."""
        return self.gantry_angles is not None

    def __post_init__(self):
        """Validate tensor shapes."""
        CP = self.leaf_positions.shape[0]  # [CP, N, 2] -> CP is dim 0

        if self.gantry_angles is not None:
            assert len(self.gantry_angles) == CP, \
                f"gantry_angles length {len(self.gantry_angles)} doesn't match CP count {CP}"

        assert self.mus.shape == (CP,), \
            f"mus shape {self.mus.shape} doesn't match expected ({CP},)"
        assert self.leaf_positions.shape[0] == CP and self.leaf_positions.shape[2] == 2, \
            f"leaf_positions shape should be [CP, N, 2], got: {self.leaf_positions.shape}"
        assert self.jaw_positions.shape == (CP, 2), \
            f"jaw_positions shape {self.jaw_positions.shape} doesn't match expected ({CP}, 2)"

    @staticmethod
    def stack(sequences: list[BeamSequence]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stack multiple BeamSequences into batched tensors for the dose engine.

        Args:
            sequences: List of BeamSequence objects (must have same CP count and leaf count)

        Returns:
            Tuple of (leaf_positions, mus, jaw_positions) with batch dimension:
            - leaf_positions: [B, CP, N, 2]
            - mus: [B, CP]
            - jaw_positions: [B, CP, 2]

        Example:
            >>> batched_leafs, batched_mus, batched_jaws = BeamSequence.stack([seq1, seq2])
            >>> dose = engine.forward(batched_leafs, batched_mus, batched_jaws, ct_batch)
        """
        if not sequences:
            raise ValueError("Cannot stack empty list of BeamSequences")

        leaf_positions = torch.stack([s.leaf_positions for s in sequences], dim=0)  # [B, CP, N, 2]
        mus = torch.stack([s.mus for s in sequences], dim=0)  # [B, CP]
        jaw_positions = torch.stack([s.jaw_positions for s in sequences], dim=0)  # [B, CP, 2]

        return leaf_positions, mus, jaw_positions
    
    @classmethod
    def create(
        cls,
        gantry_angles_deg: list[float] | torch.Tensor,
        number_of_leaf_pairs: int,
        field_size: tuple[int, int],
        iso_center: tuple[float, float, float],
        collimator_angles_deg: list[float] | torch.Tensor | None = None,
        sid: float = 1000.0,
        open_field_size: float = 0.0,
        device: torch.device | str = 'cuda',
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = True,
    ) -> BeamSequence:
        """
        Create a BeamSequence with initialized parameters.
        
        Args:
            gantry_angles: Gantry angles in degrees (list or tensor)
            number_of_leaf_pairs: Number of MLC leaf pairs
            field_size: Field size (width, height) in mm
            iso_center: Isocenter position (x, y, z) in mm
            collimator_angles: BLD angles in degrees, or None for all zeros
            sid: Source to isocenter distance in mm
            open_field_size: Size of the open field in mm (0.0=closed)
            device: PyTorch device
            dtype: Data type for tensors
            requires_grad: Whether tensors require gradients
            
        Returns:
            BeamSequence with initialized parameters
            
        Example:
            >>> angles = [0, 90, 180, 270]
            >>> beam_seq = BeamSequence.create(
            ...     gantry_angles_deg=angles,
            ...     number_of_leaf_pairs=60,
            ...     field_size=(400, 400),
            ...     iso_center=(0, 0, 0),            
            ...     open_field_size=200.0,  # 200mm open field
            ... )
        """
        # Convert gantry angles to tensor in radians
        if isinstance(gantry_angles_deg, list):
            gantry_angles = torch.tensor(gantry_angles_deg, dtype=dtype, device=device)
        else:
            gantry_angles = gantry_angles_deg.to(dtype=dtype, device=device)
            # Assume already in radians if tensor
        gantry_angles = torch.deg2rad(gantry_angles)

        num_cps = len(gantry_angles)
        field_w, field_h = field_size
        
        # Initialize leaf positions [CP, N, 2]
        leaf_positions = torch.zeros(num_cps, number_of_leaf_pairs, 2, device=device, dtype=dtype)
        leaf_positions[:, :, 0] = -open_field_size / 2  # Left leaves
        leaf_positions[:, :, 1] = open_field_size / 2   # Right leaves
        
        # Initialize jaw positions [CP, 2]
        jaw_positions = torch.zeros(num_cps, 2, device=device, dtype=dtype)
        jaw_positions[:, 0] = -open_field_size / 2  # Lower jaw
        jaw_positions[:, 1] = open_field_size / 2   # Upper jaw
        
        # Initialize MUs [CP]
        mus = torch.ones(num_cps, device=device, dtype=dtype)
        
        # Handle beam limiting device angles
        if collimator_angles_deg is None:
            collimator_angles = torch.zeros(num_cps, device=device, dtype=dtype)
        elif isinstance(collimator_angles_deg, list):
            collimator_angles = torch.tensor(collimator_angles_deg, dtype=dtype, device=device)
        else:
            collimator_angles = collimator_angles_deg.to(dtype=dtype, device=device)
        collimator_angles = torch.deg2rad(collimator_angles)
        
        # Set requires_grad
        if requires_grad:
            leaf_positions.requires_grad_(True)
            jaw_positions.requires_grad_(True)
            mus.requires_grad_(True)
        
        return cls(
            mus=mus,
            leaf_positions=leaf_positions,
            jaw_positions=jaw_positions,
            gantry_angles=gantry_angles,
            collimator_angles=collimator_angles,
            field_size=field_size,
            iso_center=iso_center,
            sid=sid,
        )

    @classmethod
    def prepare_for_engine(
        cls,
        leaf_positions: torch.Tensor,
        mus: torch.Tensor,
        jaw_positions: torch.Tensor,
        dose_engine: 'DoseEngine',
    ) -> BeamSequence:
        """
        Create a BeamSequence from tensors, filling metadata from the dose engine.
        This is useful for deep learning applications where you have predicted or
        optimized tensors but don't have the original beam specifications. The dose
        engine provides all necessary geometric and machine parameters.
        Args:
            leaf_positions: MLC positions [CP, N, 2] where N is number of leaf pairs
            mus: Monitor units [CP] where CP is number of control points
            jaw_positions: Jaw positions [CP, 2] where 2=(lower, upper)
            dose_engine: DoseEngine instance to extract metadata from
        Returns:
            BeamSequence ready to use with the dose engine (gradients flow through)
        Raises:
            ValueError: If tensor shapes don't match dose engine expectations
        Example:
            >>> # After training a model that predicts beam parameters
            >>> predicted_leafs = model(input)  # [CP, N, 2]
            >>> predicted_mus = mu_model(input)  # [CP]
            >>> predicted_jaws = jaw_model(input)  # [CP, 2]
            >>>
            >>> beam_seq = BeamSequence.prepare_for_engine(
            ...     leaf_positions=predicted_leafs,
            ...     mus=predicted_mus,
            ...     jaw_positions=predicted_jaws,
            ...     dose_engine=engine
            ... )
            >>> dose = engine.compute_beam_sequence(beam_seq, ct_image)
        """
        # Validate shapes
        expected_cp = dose_engine.number_of_beams
        expected_leafs = dose_engine.machine_config.number_of_leaf_pairs

        # Check leaf_positions shape [CP, N, 2]
        if leaf_positions.dim() != 3:
            raise ValueError(
                f"leaf_positions must be 3D [CP, N, 2], got {leaf_positions.dim()}D: {leaf_positions.shape}"
            )
        if leaf_positions.shape[0] != expected_cp:
            raise ValueError(
                f"leaf_positions CP count mismatch: expected {expected_cp}, got {leaf_positions.shape[0]}"
            )
        if leaf_positions.shape[1] != expected_leafs:
            raise ValueError(
                f"leaf_positions leaf pair count mismatch: expected {expected_leafs}, got {leaf_positions.shape[1]}"
            )
        if leaf_positions.shape[2] != 2:
            raise ValueError(
                f"leaf_positions last dimension must be 2 (left, right), got {leaf_positions.shape[2]}"
            )

        # Check mus shape [CP]
        if mus.dim() != 1:
            raise ValueError(
                f"mus must be 1D [CP], got {mus.dim()}D: {mus.shape}"
            )
        if mus.shape[0] != expected_cp:
            raise ValueError(
                f"mus CP count mismatch: expected {expected_cp}, got {mus.shape[0]}"
            )

        # Check jaw_positions shape [CP, 2]
        if jaw_positions.dim() != 2:
            raise ValueError(
                f"jaw_positions must be 2D [CP, 2], got {jaw_positions.dim()}D: {jaw_positions.shape}"
            )
        if jaw_positions.shape[0] != expected_cp:
            raise ValueError(
                f"jaw_positions CP count mismatch: expected {expected_cp}, got {jaw_positions.shape[0]}"
            )
        if jaw_positions.shape[1] != 2:
            raise ValueError(
                f"jaw_positions last dimension must be 2 (lower, upper), got {jaw_positions.shape[1]}"
            )

        return cls.from_tensors(
            leaf_positions=leaf_positions,
            mus=mus,
            jaw_positions=jaw_positions,
            gantry_angles=dose_engine.gantry_angles,
            collimator_angles=dose_engine.collimator_angles,
            iso_center=dose_engine.iso_center,
            sid=dose_engine.SID,
            field_size=dose_engine.field_size,
        )
    
    @classmethod
    def from_tensors(
        cls,
        leaf_positions: torch.Tensor,
        mus: torch.Tensor,
        jaw_positions: torch.Tensor,
        gantry_angles: torch.Tensor,
        collimator_angles: torch.Tensor,
        iso_center: float,
        sid: float,
        field_size: tuple[float, float]

    ) -> BeamSequence:
        """
        Create a BeamSequence from raw tensors.

        Args:
            leaf_positions: MLC positions [CP, N, 2]
            mus: Monitor units [CP]
            jaw_positions: Jaw positions [CP, 2]
            gantry_angles: Gantry angles in radians [CP], or None to use engine's angles

        Returns:
            BeamSequence wrapping the provided tensors (no copy, gradients flow through)
        """
        return cls(
            mus=mus,
            leaf_positions=leaf_positions,
            jaw_positions=jaw_positions,
            gantry_angles=gantry_angles,
            collimator_angles=collimator_angles,
            iso_center=iso_center,
            sid=sid,
            field_size=field_size
        )

    @staticmethod
    def _compute_gantry_angles(
        num_cps: int,
        starting_angle_deg: float,
        clockwise: bool,
        device: torch.device | str = 'cuda',
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Compute gantry angles for a given number of control points."""
        import math
        import numpy as np

        start = math.radians(starting_angle_deg)
        if num_cps == 1:
            return torch.tensor([start], dtype=dtype, device=device)

        if clockwise:
            end = start + math.radians(360)
        else:
            end = start - math.radians(360)

        # Match the gantry_angles computation
        angles = np.linspace(start, end, num_cps + 2, endpoint=False)[:-2] % (2 * math.pi)
        return torch.tensor(angles, dtype=dtype, device=device)

    @classmethod
    def from_beams(cls, beams: list[Beam]) -> BeamSequence:
        """
        Stack individual Beam objects into a BeamSequence.

        Note: This creates NEW tensors by stacking, so gradients will flow
        to the stacked tensor, not the original Beam tensors.

        Args:
            beams: List of Beam objects (must have same leaf count)

        Returns:
            BeamSequence with stacked parameters
        """
        if not beams:
            raise ValueError("Cannot create BeamSequence from empty list")

        gantry_angles = torch.tensor(
            [b.gantry_angle for b in beams],
            dtype=beams[0].dtype,
            device=beams[0].device,
        )

        collimator_angles = torch.tensor(
            [b.collimator_angle for b in beams],
            dtype=beams[0].dtype,
            device=beams[0].device,
        )

        # Stack along CP dimension (dim 0)
        # mu: scalar -> [CP]
        mus = torch.stack([b.mu for b in beams], dim=0)  # [CP]

        # leaf_positions: [N, 2] -> [CP, N, 2]
        leaf_positions = torch.stack([b.leaf_positions for b in beams], dim=0)  # [CP, N, 2]

        # jaw_positions: [2] -> [CP, 2]
        jaw_positions = torch.stack([b.jaw_positions for b in beams], dim=0)  # [CP, 2]

        if np.all([np.all(b.iso_center == beams[0].iso_center) for b in beams]):
            iso_center = beams[0].iso_center
        else:
            raise Exception("Isocenters are different for different beams. This will not work.")
        
        if np.all([np.all(b.sid == beams[0].sid) for b in beams]):
            sid = beams[0].sid
        else:
            raise Exception("SID are different for different beams. This will not work.")
        
        if np.all([np.all(b.field_size == beams[0].field_size) for b in beams]):
            field_size = beams[0].field_size
        else:
            raise Exception("Field sizes are different for different beams. This will not work.")
        

        return cls(
            mus=mus,
            leaf_positions=leaf_positions,
            jaw_positions=jaw_positions,
            gantry_angles=gantry_angles,
            collimator_angles=collimator_angles,
            iso_center=iso_center,
            sid=sid,
            field_size=field_size
        )

    def __len__(self) -> int:
        """Number of control points in the sequence."""
        return self.leaf_positions.shape[0]  # [CP, N, 2] -> CP is dim 0

    def __getitem__(self, idx: int | slice) -> Beam:
        """
        Get a single Beam at the specified index.

        The returned Beam contains VIEWS into the original tensors,
        not copies. Gradients flow back to the original BeamSequence tensors.

        Args:
            idx: Control point index (0-based)

        Returns:
            Beam with views into this sequence's tensors

        Raises:
            ValueError: If gantry_angles is None (use engine's angles instead)
        """
        if isinstance(idx, slice):
            return BeamSequence(
                mus=self.mus[idx],
                leaf_positions=self.leaf_positions[idx, :, :],
                jaw_positions=self.jaw_positions[idx, :],
                gantry_angles=self.gantry_angles[idx] if self.gantry_angles is not None else None,
                collimator_angles=self.collimator_angles[idx] if self.collimator_angles is not None else None,
                field_size=self.field_size,
                iso_center=self.iso_center,
                sid=self.sid,
            )
        else:
            if idx < 0:
                idx = len(self) + idx
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of range for BeamSequence of length {len(self)}")

            if self.gantry_angles is None:
                raise ValueError(
                    "Cannot index into BeamSequence without gantry_angles. "
                    "This BeamSequence relies on the engine's gantry angles."
                )

            return Beam(
                gantry_angle=self.gantry_angles[idx].item(),
                collimator_angle=self.collimator_angles[idx].item(),
                mu=self.mus[idx],                    # scalar
                leaf_positions=self.leaf_positions[idx, :, :],  # [N, 2]
                jaw_positions=self.jaw_positions[idx, :],       # [2]
                field_size=self.field_size,
                iso_center=self.iso_center,
                sid=self.sid
            )

    def __iter__(self) -> Iterator[Beam]:
        """Iterate over all beams in the sequence."""
        for i in range(len(self)):
            yield self[i]

    @property
    def num_beams(self) -> int:
        """Number of control points (alias for __len__)."""
        return len(self)

    @property
    def num_leaf_pairs(self) -> int:
        """Number of MLC leaf pairs."""
        return self.leaf_positions.shape[1]  # [CP, N, 2] -> N is dim 1

    @property
    def device(self) -> torch.device:
        """Device of the underlying tensors."""
        return self.leaf_positions.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the underlying tensors."""
        return self.leaf_positions.dtype

    @property
    def requires_grad(self) -> bool:
        """Whether any tensor requires gradients."""
        return (
            self.leaf_positions.requires_grad or
            self.jaw_positions.requires_grad or
            self.mus.requires_grad
        )

    @property
    def gantry_angles_deg(self) -> Optional[np.ndarray]:
        """Gantry angles in degrees as numpy array, or None if not set."""
        if self.gantry_angles is None:
            return None
        return np.degrees(self.gantry_angles.cpu().numpy())

    def detach(self) -> BeamSequence:
        """Return a new BeamSequence with detached tensors."""
        return BeamSequence(
            mus=self.mus.detach(),
            leaf_positions=self.leaf_positions.detach(),
            jaw_positions=self.jaw_positions.detach(),
            gantry_angles=self.gantry_angles.detach() if self.gantry_angles is not None else None,
            collimator_angles=self.collimator_angles.detach() if self.collimator_angles is not None else None,
            field_size=self.field_size,
            iso_center=self.iso_center,
            sid=self.sid
        )

    def clone(self) -> BeamSequence:
        """Return a deep copy of this BeamSequence."""
        return BeamSequence(
            mus=self.mus.clone(),
            leaf_positions=self.leaf_positions.clone(),
            jaw_positions=self.jaw_positions.clone(),
            gantry_angles=self.gantry_angles.clone() if self.gantry_angles is not None else None,
            collimator_angles=self.collimator_angles.clone() if self.collimator_angles is not None else None,
            field_size=self.field_size,
            iso_center=self.iso_center,
            sid=self.sid
        )

    def to(self, device: torch.device | str) -> BeamSequence:
        """Move all tensors to a different device."""
        return BeamSequence(
            mus=self.mus.to(device),
            leaf_positions=self.leaf_positions.to(device),
            jaw_positions=self.jaw_positions.to(device),
            gantry_angles=self.gantry_angles.to(device) if self.gantry_angles is not None else None,
            collimator_angles=self.collimator_angles.to(device) if self.collimator_angles is not None else None,
            field_size=self.field_size,
            iso_center=self.iso_center,
            sid=self.sid
        )

    def slice(self, start: int, end: int) -> BeamSequence:
        """
        Get a contiguous slice of control points as a new BeamSequence.

        The returned BeamSequence contains VIEWS into the original tensors.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            BeamSequence with the specified range of control points
        """
        return BeamSequence(
            mus=self.mus[start:end],                              # [CP_slice]
            leaf_positions=self.leaf_positions[start:end, :, :],  # [CP_slice, N, 2]
            jaw_positions=self.jaw_positions[start:end, :],       # [CP_slice, 2]
            gantry_angles=self.gantry_angles[start:end] if self.gantry_angles is not None else None,
            collimator_angles=self.collimator_angles[start:end] if self.collimator_angles is not None else None,
            field_size=self.field_size,
            iso_center=self.iso_center,
            sid=self.sid
        )

    def to_delivery(self) -> BeamSequence:
        """
        Convert control points to delivery positions by averaging adjacent points.

        DICOM RT plans store N+1 control points, but dose is delivered at N
        intermediate positions between them. This method computes those
        intermediate positions.

        The returned BeamSequence has N control points (one less than original),
        where each value is the average of adjacent control points:
            delivery[i] = (control_point[i] + control_point[i+1]) / 2

        Gradients flow back to the original control points.

        Returns:
            BeamSequence with N averaged delivery positions
        """
        if len(self) < 2:
            raise ValueError("Need at least 2 control points to compute delivery positions")

        # Average adjacent control points - gradients flow through
        # leaf_positions: [CP, N, 2] -> [CP-1, N, 2]
        avg_leaf_positions = (
            self.leaf_positions[:-1, :, :] + self.leaf_positions[1:, :, :]
        ) / 2

        # mus: [CP] -> [CP-1]
        avg_mus = (self.mus[:-1] + self.mus[1:]) / 2

        # jaw_positions: [CP, 2] -> [CP-1, 2]
        avg_jaw_positions = (
            self.jaw_positions[:-1, :] + self.jaw_positions[1:, :]
        ) / 2

        two_pi = 2 * math.pi
        avg_gantry_angles = None
        if self.gantry_angles is not None:
            a = self.gantry_angles[:-1]
            b = self.gantry_angles[1:]
            # shortest signed difference in (-π, π]
            delta = (b - a + math.pi) % two_pi - math.pi
            # go halfway along that shortest arc and wrap back to [0, 2π)
            avg_gantry_angles = (a + 0.5 * delta + two_pi) % two_pi

        avg_collimator_angles = None
        if self.collimator_angles is not None:
            a = self.collimator_angles[:-1]
            b = self.collimator_angles[1:]
            delta = (b - a + math.pi) % two_pi - math.pi
            avg_collimator_angles = (a + 0.5 * delta + two_pi) % two_pi

        return BeamSequence(
            mus=avg_mus,
            leaf_positions=avg_leaf_positions,
            jaw_positions=avg_jaw_positions,
            gantry_angles=avg_gantry_angles,
            collimator_angles=avg_collimator_angles,
            field_size=self.field_size,
            iso_center=self.iso_center,
            sid=self.sid
        )

    @property
    def control_points(self) -> BeamSequence:
        """Alias for self - the original control point representation."""
        return self

    @property
    def delivery(self) -> BeamSequence:
        """
        Property alias for to_delivery().

        Convenient for chaining:
            dose = engine.forward_beam_sequence(beam_seq.delivery)
        """
        return self.to_delivery()
    def parameters(self) -> list[torch.Tensor]:
        """Return list of optimizable parameters."""
        return [self.leaf_positions, self.jaw_positions, self.mus]

    def requires_grad_(self, requires_grad: bool = True) -> BeamSequence:
        """Set requires_grad on all tensors."""
        self.leaf_positions.requires_grad_(requires_grad)
        self.jaw_positions.requires_grad_(requires_grad)
        self.mus.requires_grad_(requires_grad)
        return self