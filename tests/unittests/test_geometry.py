import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import torch
import math
from pydose_rt.geometry.projections import soft_max, soft_min, fractional_box_overlap, resample_fluence_map
from pydose_rt.geometry.rotations import get_radiological_depth_indices, rotate_2d_images


class TestProjections:
    """Tests for projection functions"""

    def test_soft_max_approximates_max(self):
        """Test that soft_max approximates torch.max"""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([2.0, 1.0, 3.5])

        result = soft_max(a, b, sharpness=100.0)  # High sharpness for close approximation
        expected = torch.maximum(a, b)

        assert torch.allclose(result, expected, atol=0.01)

    def test_soft_min_approximates_min(self):
        """Test that soft_min approximates torch.min"""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([2.0, 1.0, 3.5])

        result = soft_min(a, b, sharpness=100.0)
        expected = torch.minimum(a, b)

        assert torch.allclose(result, expected, atol=0.01)

    def test_soft_max_gradients(self):
        """Test that soft_max provides gradients for both inputs"""
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([2.0, 1.0], requires_grad=True)

        result = soft_max(a, b)
        loss = result.sum()
        loss.backward()

        assert a.grad is not None
        assert b.grad is not None
        assert torch.any(a.grad != 0)
        assert torch.any(b.grad != 0)

    def test_soft_min_gradients(self):
        """Test that soft_min provides gradients for both inputs"""
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([2.0, 1.0], requires_grad=True)

        result = soft_min(a, b)
        loss = result.sum()
        loss.backward()

        assert a.grad is not None
        assert b.grad is not None

    def test_fractional_box_overlap_full_overlap(self):
        """Test fractional overlap with full overlap"""
        d = torch.tensor([0.0])  # Bin center
        left = torch.tensor([-1.0])  # Left edge
        right = torch.tensor([1.0])  # Right edge

        overlap = fractional_box_overlap(d, left, right)

        # Should have full overlap (1.0)
        assert overlap.item() == pytest.approx(1.0, abs=0.01)

    def test_fractional_box_overlap_no_overlap(self):
        """Test fractional overlap with no overlap"""
        d = torch.tensor([0.0])  # Bin center
        left = torch.tensor([2.0])  # Left edge far right
        right = torch.tensor([3.0])  # Right edge even further

        overlap = fractional_box_overlap(d, left, right)

        # Should have no overlap (0.0)
        assert overlap.item() == pytest.approx(0.0, abs=0.01)
    
    @pytest.mark.xfail
    def test_fractional_box_overlap_partial(self):
        """Test fractional overlap with partial overlap"""
        d = torch.tensor([0.0])  # Bin center at 0
        left = torch.tensor([0.0])  # Left edge at center
        right = torch.tensor([1.0])  # Right edge to the right

        overlap = fractional_box_overlap(d, left, right, sharpness=None)

        # Should have partial overlap (approximately 0.5)
        assert 0.3 < overlap.item() < 0.7  # Rough range for partial overlap

    def test_fractional_box_overlap_gradients(self):
        """Test that gradients flow through fractional_box_overlap"""
        d = torch.tensor([0.0])
        left = torch.tensor([0.0], requires_grad=True)
        right = torch.tensor([1.0], requires_grad=True)

        overlap = fractional_box_overlap(d, left, right)
        loss = overlap.sum()
        loss.backward()

        assert left.grad is not None
        assert right.grad is not None

    def test_resample_fluence_map_shape(self):
        """Test that resample_fluence_map produces correct output shape"""
        B = 2
        W = 100
        N = 60  # Number of leaves
        field_size = 400
        leaf_widths = [5.0] * N  # Uniform leaf widths

        values = torch.randn(B, W, N, 1)
        result = resample_fluence_map(values, leaf_widths, field_size, torch.float32)

        expected_shape = (B, W, field_size, 1)
        assert result.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {result.shape}"

    def test_resample_fluence_map_uniform_input(self):
        """Test resample_fluence_map with uniform input values"""
        B = 1
        W = 10
        N = 20
        field_size = 100
        leaf_widths = [5.0] * N

        # All leaves have value 1.0
        values = torch.ones(B, W, N, 1)
        result = resample_fluence_map(values, leaf_widths, field_size, torch.float32)

        # Output should be approximately all ones
        assert torch.allclose(result, torch.ones_like(result), atol=0.1)

    def test_resample_fluence_map_gradients(self):
        """Test that gradients flow through resample_fluence_map"""
        B = 2
        W = 10
        N = 20
        field_size = 100
        leaf_widths = [5.0] * N

        values = torch.randn(B, W, N, 1, requires_grad=True)
        result = resample_fluence_map(values, leaf_widths, field_size, torch.float32)
        loss = result.sum()
        loss.backward()

        assert values.grad is not None
        assert torch.any(values.grad != 0)


class TestRotations:
    """Tests for rotation functions"""

    def test_get_radiological_depth_indices_shape(self):
        """Test that get_radiological_depth_indices produces correct output shape"""
        input_shape = (64, 64, 64)  # (H, D, W)
        angles_rad = [0.0, math.pi/2, math.pi]

        indices = get_radiological_depth_indices(
            input_shape,
            angles_rad,
            torch.float32
        )

        # Expected shape: [1, G, D, 3]
        expected_shape = (1, 3, 64, 3)
        assert indices.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {indices.shape}"

    def test_get_radiological_depth_indices_single_angle(self):
        """Test get_radiological_depth_indices with single angle"""
        input_shape = (32, 32, 32)
        angles_rad = [0.0]

        indices = get_radiological_depth_indices(
            input_shape,
            angles_rad,
            torch.float32
        )

        assert indices.shape == (1, 1, 32, 3)

    def test_get_radiological_depth_indices_zero_angle(self):
        """Test that zero angle produces expected ray through center"""
        input_shape = (64, 64, 64)
        angles_rad = [0.0]

        indices = get_radiological_depth_indices(
            input_shape,
            angles_rad,
            torch.float32
        )

        # At zero angle, x should be constant (at center)
        x_coords = indices[0, 0, :, 0]
        assert torch.allclose(x_coords, x_coords[0] * torch.ones_like(x_coords)), \
            "X coordinates should be constant for zero angle"

        # Y should vary from 0 to D-1
        y_coords = indices[0, 0, :, 1]
        assert y_coords[0] == pytest.approx(0.0, abs=0.1)
        assert y_coords[-1] == pytest.approx(63.0, abs=0.1)

    def test_get_radiological_depth_indices_with_isocenter(self):
        """Test get_radiological_depth_indices with custom isocenter"""
        input_shape = (64, 64, 64)
        angles_rad = [0.0]
        iso_center = (96.0, 96.0, 96.0)  # Custom isocenter in mm
        resolution = (3.0, 3.0, 3.0)  # 3mm voxels

        indices = get_radiological_depth_indices(
            input_shape,
            angles_rad,
            torch.float32,
            iso_center=iso_center,
            resolution=resolution
        )

        assert indices.shape == (1, 1, 64, 3)
        # Isocenter should affect the ray position
        # At isocenter (96, 96, 96) mm with 3mm voxels, voxel coords should be ~32
        center_x = indices[0, 0, 0, 0]
        assert 30.0 < center_x < 34.0  # Should be around voxel 32

    def test_get_radiological_depth_indices_multiple_angles(self):
        """Test get_radiological_depth_indices with multiple angles"""
        input_shape = (32, 32, 32)
        angles_rad = [0.0, math.pi/4, math.pi/2, math.pi]

        indices = get_radiological_depth_indices(
            input_shape,
            angles_rad,
            torch.float32
        )

        assert indices.shape == (1, 4, 32, 3)

        # Each angle should produce different ray coordinates
        for i in range(len(angles_rad)):
            for j in range(i + 1, len(angles_rad)):
                # Rays at different angles should have different coordinates
                ray_i = indices[0, i]
                ray_j = indices[0, j]
                assert not torch.allclose(ray_i, ray_j), \
                    f"Rays at angles {angles_rad[i]} and {angles_rad[j]} should differ"

    def test_rotate_2d_images_shape(self, default_device, default_dtype):
        """Test that rotate_2d_images produces correct output shape"""
        BG = 4
        H = 64
        W = 64
        angles_rad = [0.0, math.pi/4, math.pi/2, 3*math.pi/4]

        images = torch.randn(BG, H, W, device=default_device, dtype=default_dtype)
        rotated = rotate_2d_images(images, angles_rad, default_device, default_dtype)

        assert rotated.shape == images.shape, \
            f"Expected shape {images.shape}, got {rotated.shape}"

    def test_rotate_2d_images_zero_angle(self, default_device, default_dtype):
        """Test that zero angle produces minimal rotation"""
        BG = 1
        H = 32
        W = 32
        angles_rad = [0.0]

        images = torch.randn(BG, H, W, device=default_device, dtype=default_dtype)
        rotated = rotate_2d_images(images, angles_rad, default_device, default_dtype)

        # Zero rotation should produce similar output (may have small interpolation artifacts)
        assert torch.allclose(rotated, images, atol=0.1), \
            "Zero angle rotation should preserve image approximately"

    def test_rotate_2d_images_gradients(self, default_device, default_dtype):
        """Test that gradients flow through rotate_2d_images"""
        BG = 2
        H = 32
        W = 32
        angles_rad = [math.pi/4, math.pi/2]

        images = torch.randn(BG, H, W, device=default_device, dtype=default_dtype, requires_grad=True)
        rotated = rotate_2d_images(images, angles_rad, default_device, default_dtype)
        loss = rotated.sum()
        loss.backward()

        assert images.grad is not None
        assert torch.any(images.grad != 0)

    def test_rotate_2d_images_dtype_preservation(self, default_device):
        """Test that rotate_2d_images preserves dtype"""
        for dtype in [torch.float32, torch.float64]:
            BG = 2
            H = 32
            W = 32
            angles_rad = [0.0, math.pi/2]

            images = torch.randn(BG, H, W, device=default_device, dtype=dtype)
            rotated = rotate_2d_images(images, angles_rad, default_device, dtype)

            assert rotated.dtype == dtype, f"Expected dtype {dtype}, got {rotated.dtype}"