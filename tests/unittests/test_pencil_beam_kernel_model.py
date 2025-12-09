from pathlib import Path
import sys
import torch

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.absolute()))

import pytest

from pydose_rt.physics.kernels.pencil_beam_model import coeffs as COEFFICIENTS, PencilBeamModel

@pytest.fixture(scope="function")
def pencil_beam_kernel_model(default_machine_config, default_resolution) -> PencilBeamModel:
    return PencilBeamModel(default_resolution, default_machine_config.tpr_20_10, kernel_size=25)


class TestPencilBeamModel:
    @pytest.mark.parametrize(
        "parameter,tpr,expected_value",
        [
            ("A1", 0.72, 0.0038774804816691143),
            ("A2", 0.72, -1.66547181525417),
            ("A3", 0.72, -0.04446500542538754),
            ("A4", 0.72, 0.00010907329942527776),
            ("A5", 0.72, 0.18035601817463812),

            ("B1", 0.72, 0.01283405969914142),
            ("B2", 0.72, -0.009536489677424459),
            ("B3", 0.72, -0.0455082974876262),
            ("B4", 0.72, 0.00011329967902720304),
            ("B5", 0.72, 1.4489094077514082),

            ("a1", 0.72, -0.011260807962593282),
            ("a2", 0.72, 3.496748644927365),

            ("b1", 0.72, 0.2049180292602557),
            ("b2", 0.72, -0.26165547931400246),
            ("b3", 0.72, -0.03606398942758915),
            ("b4", 0.72, 0.0002465905465857121),
            ("b5", 0.72, 1.1061211405778435),

            ("a1", 0.69, -0.010664844189712705),
            ("a2", 0.69, 4.278268380623299)
        ],
        ids=[
            "Param: A1, TPR20,10: 0.72", "Param: A2, TPR20,10: 0.72", "Param: A3, TPR20,10: 0.72",
            "Param: A4, TPR20,10: 0.72", "Param: A5, TPR20,10: 0.72",
            "Param: B1, TPR20,10: 0.72", "Param: B2, TPR20,10: 0.72", "Param: B3, TPR20,10: 0.72",
            "Param: B4, TPR20,10: 0.72", "Param: B5, TPR20,10: 0.72",
            "Param: a1, TPR20,10: 0.72", "Param: a2, TPR20,10: 0.72",
            "Param: b1, TPR20,10: 0.72", "Param: b2, TPR20,10: 0.72", "Param: b3, TPR20,10: 0.72",
            "Param: b4, TPR20,10: 0.72", "Param: b5, TPR20,10: 0.72",
            "Param: a1, TPR20,10: 0.69", "Param: a2, TPR20,10: 0.69",]
    )
    def test_get_param_should_return_the_value_for_specified_parameter_according_to_eq_7(
            self, pencil_beam_kernel_model, parameter, tpr, expected_value):
        # Arrange

        # Act
        actual = pencil_beam_kernel_model.get_param(parameter, tpr)

        # Assert
        assert actual == pytest.approx(expected_value, rel=1e-6, abs=1e-6)

    @pytest.mark.parametrize(
        "depth, tpr, expected_value",
        [
            (0.1, 0.72, 0.003922572180517211),
            (1.0, 0.72, 0.010548557846173284),
            (10.0, 0.72, 0.00850405219550931),
            (30.0, 0.72, 0.003559545901112395),
            (10.0, 0.69, 0.010740152278099201),
        ],
        ids=[
            "Depth: 0.0, TPR20,10: 0.72",
            "Depth: 1.0, TPR20,10: 0.72",
            "Depth: 10.0, TPR20,10: 0.72",
            "Depth: 30.0, TPR20,10: 0.72",
            "Depth: 10.0, TPR20,10: 0.69",
        ]
    )
    def test_depth_A_returns_correct_value_for_A_given_specific_depth_and_tpr_according_to_eq_3_and_5(self, pencil_beam_kernel_model, depth, tpr, expected_value):
        # Arrange
        pencil_beam_kernel_model.params = {k: pencil_beam_kernel_model.get_param(k, tpr) for k in COEFFICIENTS.keys()}
        # Act
        actual = pencil_beam_kernel_model.depth_A(d=torch.tensor(depth)).item()
        # Assert
        assert actual == pytest.approx(expected_value, rel=1e-6, abs=1e-6)

    @pytest.mark.parametrize(
        "depth, tpr, expected_value",
        [
            (0.0, 0.72, 9.069366384703944e-06),
            (1.0, 0.72, 1.304167098582641e-05),
            (10.0, 0.72, 0.00010281336868026025),
            (30.0, 0.72, 7.83347858857436e-05),
            (10.0, 0.69, 0.00012425384299934782),
        ],
        ids=[
            "Depth: 0.0, TPR20,10: 0.72",
            "Depth: 1.0, TPR20,10: 0.72",
            "Depth: 10.0, TPR20,10: 0.72",
            "Depth: 30.0, TPR20,10: 0.72",
            "Depth: 10.0, TPR20,10: 0.69",
        ]
    )
    def test_depth_B_returns_correct_value_for_A_given_specific_depth_and_tpr_according_to_eq_4_and_6(self, pencil_beam_kernel_model, depth, tpr, expected_value):
        # Arrange
        pencil_beam_kernel_model.params = {k: pencil_beam_kernel_model.get_param(k, tpr) for k in COEFFICIENTS.keys()}

        # Act
        actual = pencil_beam_kernel_model.depth_B(d=torch.tensor(depth))

        # Assert
        assert actual == pytest.approx(expected_value, rel=1e-6, abs=1e-6)

    @pytest.mark.parametrize(
        "depth, tpr, expected_value",
        [
            (0.0, 0.69, 4.278268380623299),
            (1.0, 0.69, 4.267603536433586),
            (10.0, 0.69, 4.171619938726172),
            (30.0, 0.69, 3.9583230549319177),
            (10.0, 0.72, 3.3841405653014323),
        ],
        ids=[
            "a given - Depth: 0.0, TPR20,10: 0.69",
            "a given - Depth: 1.0, TPR20,10: 0.69",
            "a given - Depth: 10.0, TPR20,10: 0.69",
            "a given - Depth: 30.0, TPR20,10: 0.69",
            "a given - Depth: 10.0, TPR20,10: 0.72",
        ]
    )
    def test_depth_a_returns_correct_value_for_a_given_specific_depth_and_tpr_according_to_eq_5(self, pencil_beam_kernel_model, depth, tpr, expected_value):
        """ NB! This test is testing the calculation according to the correction to the equation that can be found at
        https://www.sciencedirect.com/science/article/pii/S0167814010007231
         
        In the correction the equation is specified as:
                a(a_{1..2};d) = a2 + a1 * d^2
        """""
        # Arrange
        pencil_beam_kernel_model.params = {k: pencil_beam_kernel_model.get_param(k, tpr) for k in COEFFICIENTS.keys()}

        # Act
        actual = pencil_beam_kernel_model.depth_a(depth)

        # Assert
        assert actual == pytest.approx(expected_value, rel=1e-12, abs=1e-12)

    @pytest.mark.parametrize(
        "depth, tpr, expected_value",
        [
            (0.0, 0.69, 0.04843348476172036),
            (1.0, 0.69, 0.06178661500993746),
            (10.0, 0.69, 0.13090298651610577),
            (30.0, 0.69, 0.07975528382389455),
            (10.0, 0.72, 0.13591314449262962),
        ],
        ids=[
            "b given - Depth: 0.0, TPR20,10: 0.69",
            "b given - Depth: 1.0, TPR20,10: 0.69",
            "b given - Depth: 10.0, TPR20,10: 0.69",
            "b given - Depth: 30.0, TPR20,10: 0.69",
            "b given - Depth: 10.0, TPR20,10: 0.72",
        ]
    )
    def test_depth_b_returns_correct_value_for_b_given_specific_depth_and_tpr_according_to_eq_6(self, pencil_beam_kernel_model, depth, tpr, expected_value):
        # Arrange
        pencil_beam_kernel_model.params = {k: pencil_beam_kernel_model.get_param(k, tpr) for k in COEFFICIENTS.keys()}

        # Act
        actual = pencil_beam_kernel_model.depth_b(torch.tensor(depth))

        # Assert
        assert actual == pytest.approx(expected_value, rel=1e-6, abs=1e-6)

    @pytest.mark.xfail
    def test_get_pencil_beam_returns_a_numpy_array_of_length_four(self, pencil_beam_kernel_model):
        # Arrange
        expected_shape = (4, )

        radiological_depth = 10.0
        tpr = 0.72
        normalization = "finite"

        batch_size = 1
        number_of_gantry_angles = 2
        number_of_sampled_depths: int = 10

        radiological_depths = np.ndarray(shape=(batch_size * number_of_gantry_angles, number_of_sampled_depths, 1))
        for ind in range(number_of_sampled_depths):
            radiological_depths[:, ind, 0] = ind

        # Act
        actual = pencil_beam_kernel_model.get_pencil_beam(
            d=radiological_depths, r=pencil_beam_kernel_model.get_rs(kernel_size=(10, 10)), normalize=normalization
        )

        # Assert
        assert isinstance(actual, np.ndarray)
        assert actual.shape == expected_shape
