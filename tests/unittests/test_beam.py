import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import torch
import math
from pydose_rt.data import Beam, BeamSequence


class TestBeam:
    """Tests for Beam class"""

    def test_beam_create(self, default_device, default_dtype):
        """Test basic beam creation"""
        beam = Beam.create(
            gantry_angle_deg=90.0,
            number_of_leaf_pairs=60,
            field_size_mm=(400, 400),
            device=default_device,
            dtype=default_dtype,
            requires_grad=True
        )

        assert beam is not None
        assert beam.gantry_angle == pytest.approx(math.radians(90.0))
        assert beam.num_leaf_pairs == 60
        assert beam.leaf_positions.shape == (60, 2)
        assert beam.jaw_positions.shape == (2,)
        assert beam.mu.shape == ()  # scalar

    def test_beam_gantry_angle_conversion(self, default_device):
        """Test gantry angle conversion between degrees and radians"""
        for angle_deg in [0.0, 45.0, 90.0, 180.0, 270.0, 360.0]:
            beam = Beam.create(
                gantry_angle_deg=angle_deg,
                number_of_leaf_pairs=10,
                device=default_device
            )
            assert beam.gantry_angle_deg == pytest.approx(angle_deg)
            assert beam.gantry_angle == pytest.approx(math.radians(angle_deg))

    def test_beam_requires_grad(self, default_device):
        """Test that beam tensors can require gradients"""
        beam = Beam.create(
            gantry_angle_deg=0.0,
            number_of_leaf_pairs=10,
            device=default_device,
            requires_grad=True
        )

        assert beam.leaf_positions.requires_grad
        assert beam.jaw_positions.requires_grad
        assert beam.mu.requires_grad
        assert beam.requires_grad

    def test_beam_no_grad(self, default_device):
        """Test beam creation without gradients"""
        beam = Beam.create(
            gantry_angle_deg=0.0,
            number_of_leaf_pairs=10,
            device=default_device,
            requires_grad=False
        )

        assert not beam.leaf_positions.requires_grad
        assert not beam.jaw_positions.requires_grad
        assert not beam.mu.requires_grad
        assert not beam.requires_grad

    def test_beam_detach(self, default_device):
        """Test beam detach creates new beam without gradients"""
        beam = Beam.create(
            gantry_angle_deg=45.0,
            number_of_leaf_pairs=10,
            device=default_device,
            requires_grad=True
        )

        detached = beam.detach()

        assert not detached.leaf_positions.requires_grad
        assert not detached.jaw_positions.requires_grad
        assert not detached.mu.requires_grad
        assert detached.gantry_angle == beam.gantry_angle
        assert detached.num_leaf_pairs == beam.num_leaf_pairs

    def test_beam_clone(self, default_device):
        """Test beam clone creates independent copy"""
        beam = Beam.create(
            gantry_angle_deg=45.0,
            number_of_leaf_pairs=10,
            device=default_device,
            requires_grad=True
        )

        cloned = beam.clone()

        assert cloned.gantry_angle == beam.gantry_angle
        assert torch.allclose(cloned.leaf_positions, beam.leaf_positions)
        assert torch.allclose(cloned.jaw_positions, beam.jaw_positions)
        assert torch.allclose(cloned.mu, beam.mu)

        # Modify cloned beam should not affect original
        cloned.leaf_positions[0, 0] = 999.0
        assert not torch.allclose(cloned.leaf_positions, beam.leaf_positions)

    def test_beam_to_device(self, default_dtype):
        """Test moving beam to different device"""
        beam = Beam.create(
            gantry_angle_deg=0.0,
            number_of_leaf_pairs=10,
            device='cpu',
            dtype=default_dtype
        )

        assert beam.device == torch.device('cpu')

        # Try to move to CUDA if available
        if torch.cuda.is_available():
            beam_cuda = beam.to('cuda')
            assert beam_cuda.device.type == 'cuda'
            assert beam.device == torch.device('cpu')  # Original unchanged

    def test_beam_initial_positions(self, default_device):
        """Test that beam is initialized with fully open field"""
        field_size = (400, 400)
        beam = Beam.create(
            gantry_angle_deg=0.0,
            number_of_leaf_pairs=10,
            field_size_mm=field_size,
            device=default_device
        )

        # Left leaves should be at -field_width/2
        assert torch.allclose(beam.leaf_positions[:, 0], torch.tensor(-field_size[0] / 2))
        # Right leaves should be at +field_width/2
        assert torch.allclose(beam.leaf_positions[:, 1], torch.tensor(field_size[0] / 2))

        # Lower jaw should be at -field_height/2
        assert beam.jaw_positions[0].cpu().detach().numpy() == pytest.approx(-field_size[1] / 2)
        # Upper jaw should be at +field_height/2
        assert beam.jaw_positions[1].cpu().detach().numpy() == pytest.approx(field_size[1] / 2)

        # MU should be initialized to 1.0
        assert beam.mu.cpu().detach().numpy() == pytest.approx(1.0)

    def test_beam_dtype_preservation(self):
        """Test that beam preserves dtype"""
        for dtype in [torch.float32, torch.float64]:
            beam = Beam.create(
                gantry_angle_deg=0.0,
                number_of_leaf_pairs=10,
                device='cpu',
                dtype=dtype
            )
            assert beam.dtype == dtype
            assert beam.leaf_positions.dtype == dtype
            assert beam.jaw_positions.dtype == dtype
            assert beam.mu.dtype == dtype


class TestBeamSequence:
    """Tests for BeamSequence class"""

    def test_beam_sequence_create(self, default_device, default_dtype):
        """Test basic beam sequence creation"""
        angles = [0.0, 90.0, 180.0, 270.0]
        beam_seq = BeamSequence.create(
            gantry_angles_deg=angles,
            number_of_leaf_pairs=60,
            field_size=(400, 400),
            iso_center=(0.0, 0.0, 0.0),
            device=default_device,
            dtype=default_dtype,
            requires_grad=True
        )

        assert beam_seq is not None
        assert beam_seq.leaf_positions.shape == (4, 60, 2)
        assert beam_seq.jaw_positions.shape == (4, 2)
        assert beam_seq.mus.shape == (4,)
        assert beam_seq.gantry_angles.shape == (4,)

    def test_beam_sequence_indexing(self, default_device, default_dtype):
        """Test indexing into beam sequence returns Beam objects"""
        angles = [0.0, 90.0, 180.0]
        beam_seq = BeamSequence.create(
            gantry_angles_deg=angles,
            number_of_leaf_pairs=10,
            field_size=(400, 400),
            iso_center=(0.0, 0.0, 0.0),
            device=default_device,
            dtype=default_dtype
        )

        beam = beam_seq[0]
        assert isinstance(beam, Beam)
        assert beam.gantry_angle_deg == pytest.approx(0.0)
        assert beam.num_leaf_pairs == 10

    def test_beam_sequence_iteration(self, default_device, default_dtype):
        """Test iterating over beam sequence"""
        angles = [0.0, 90.0, 180.0, 270.0]
        beam_seq = BeamSequence.create(
            gantry_angles_deg=angles,
            number_of_leaf_pairs=10,
            field_size=(400, 400),
            iso_center=(0.0, 0.0, 0.0),
            device=default_device,
            dtype=default_dtype
        )

        collected_angles = []
        for beam in beam_seq:
            assert isinstance(beam, Beam)
            collected_angles.append(beam.gantry_angle_deg)

        assert len(collected_angles) == 4
        assert collected_angles == pytest.approx(angles)

    def test_beam_sequence_length(self, default_device, default_dtype):
        """Test beam sequence length"""
        for num_beams in [1, 5, 10, 20]:
            angles = list(range(0, 360, 360 // num_beams))[:num_beams]
            beam_seq = BeamSequence.create(
                gantry_angles_deg=angles,
                number_of_leaf_pairs=10,
                field_size=(400, 400),
                iso_center=(0.0, 0.0, 0.0),
                device=default_device,
                dtype=default_dtype
            )
            assert len(beam_seq) == num_beams

    def test_beam_sequence_stack(self, default_device, default_dtype):
        """Test stacking multiple beam sequences"""
        seq1 = BeamSequence.create(
            gantry_angles_deg=[0.0, 90.0],
            number_of_leaf_pairs=10,
            field_size=(400, 400),
            iso_center=(0.0, 0.0, 0.0),
            device=default_device,
            dtype=default_dtype
        )

        seq2 = BeamSequence.create(
            gantry_angles_deg=[180.0, 270.0],
            number_of_leaf_pairs=10,
            field_size=(400, 400),
            iso_center=(0.0, 0.0, 0.0),
            device=default_device,
            dtype=default_dtype
        )

        leaf_positions, mus, jaw_positions = BeamSequence.stack([seq1, seq2])

        assert leaf_positions.shape == (2, 2, 10, 2)  # [B, CP, N, 2]
        assert mus.shape == (2, 2)  # [B, CP]
        assert jaw_positions.shape == (2, 2, 2)  # [B, CP, 2]

    def test_beam_sequence_stack_single(self, default_device, default_dtype):
        """Test stacking single beam sequence"""
        seq = BeamSequence.create(
            gantry_angles_deg=[0.0, 90.0, 180.0],
            number_of_leaf_pairs=10,
            field_size=(400, 400),
            iso_center=(0.0, 0.0, 0.0),
            device=default_device,
            dtype=default_dtype
        )

        leaf_positions, mus, jaw_positions = BeamSequence.stack([seq])

        assert leaf_positions.shape == (1, 3, 10, 2)  # [1, CP, N, 2]
        assert mus.shape == (1, 3)  # [1, CP]
        assert jaw_positions.shape == (1, 3, 2)  # [1, CP, 2]

    def test_beam_sequence_stack_empty_fails(self):
        """Test that stacking empty list raises error"""
        with pytest.raises(ValueError, match="Cannot stack empty list"):
            BeamSequence.stack([])

    def test_beam_sequence_requires_grad(self, default_device):
        """Test beam sequence with gradient tracking"""
        beam_seq = BeamSequence.create(
            gantry_angles_deg=[0.0, 90.0],
            number_of_leaf_pairs=10,
            field_size=(400, 400),
            iso_center=(0.0, 0.0, 0.0),
            device=default_device,
            requires_grad=True
        )

        assert beam_seq.leaf_positions.requires_grad
        assert beam_seq.jaw_positions.requires_grad
        assert beam_seq.mus.requires_grad

    def test_beam_sequence_gradient_flow(self, default_device, default_dtype):
        """Test that gradients flow through indexed beams"""
        beam_seq = BeamSequence.create(
            gantry_angles_deg=[0.0, 90.0],
            number_of_leaf_pairs=10,
            field_size=(400, 400),
            iso_center=(0.0, 0.0, 0.0),
            device=default_device,
            dtype=default_dtype,
            requires_grad=True
        )

        beam = beam_seq[0]
        loss = beam.leaf_positions.sum()
        loss.backward()

        # Gradients should flow back to the original sequence
        assert beam_seq.leaf_positions.grad is not None
        assert beam_seq.leaf_positions.grad[0].sum() > 0

    def test_beam_sequence_has_gantry_angles(self, default_device, default_dtype):
        """Test has_gantry_angles property"""
        beam_seq = BeamSequence.create(
            gantry_angles_deg=[0.0, 90.0],
            number_of_leaf_pairs=10,
            field_size=(400, 400),
            iso_center=(0.0, 0.0, 0.0),
            device=default_device,
            dtype=default_dtype
        )

        assert beam_seq.has_gantry_angles

    def test_beam_sequence_dtype_preservation(self):
        """Test that beam sequence preserves dtype"""
        for dtype in [torch.float32, torch.float64]:
            beam_seq = BeamSequence.create(
                gantry_angles_deg=[0.0, 90.0],
                number_of_leaf_pairs=10,
                field_size=(400, 400),
                iso_center=(0.0, 0.0, 0.0),
                device='cpu',
                dtype=dtype
            )

            assert beam_seq.leaf_positions.dtype == dtype
            assert beam_seq.jaw_positions.dtype == dtype
            assert beam_seq.mus.dtype == dtype