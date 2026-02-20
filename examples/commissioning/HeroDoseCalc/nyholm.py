"""
Nyholm-inspired beam model for kernel generation.
"""
import numpy as np
import torch

from .hardware import DEVICE


class NyholmBeamModel:
    """Generates depth-dependent convolution kernels for photon beams."""

    def __init__(self, tpr20_10: float, resolution_mm: float, device=DEVICE):
        self.device, self.res, self.tpr = device, resolution_mm, tpr20_10
        self.max_depth_mm = 400.0
        self.slab_depths = torch.tensor(
            [0, 2, 5, 10, 20, 40, 80, 120, 160, 200, 250, 300, 350, 400],
            dtype=torch.float32,
            device=device,
        )
        self.params = self._calculate_depth_parameters()
        self.kernel_weights = self._generate_kernels()

    def _poly_val(self, coeffs):
        val = 0.0
        for i, c in enumerate(coeffs):
            val += c * (self.tpr ** i)
        return val

    def _calculate_depth_parameters(self):
        COEFFS = {
            "A1": [0.0128018, -0.0577391, 0.1790839, -0.2467955, 0.1328192, -0.0194684],
            "A2": [16.7815028, -279.4672663, 839.0016549, -978.4915013, 470.5317337, -69.2485573],
            "A3": [-0.0889669, -0.2587584, 0.7069203, -0.3654033, 0.0029760, -0.0003786],
            "A4": [0.0017089, -0.0169150, 0.0514650, -0.0639530, 0.0324490, -0.0049121],
            "A5": [0.1431447, -0.2134626, 0.5825546, -0.2969273, -0.0011436, 0.0002219],
            "B1": [-42.7607523, 264.3424720, -633.4540368, 731.5311577, -402.5280374, 82.4936551],
            "B2": [0.2428359, -2.5029336, 7.6128101, -9.5273454, 4.8249840, -0.7097852],
            "B3": [-0.0910420, -0.2621605, 0.7157244, -0.3664126, 0.0000930, -0.0000232],
            "B4": [0.0017284, -0.0172146, 0.0522109, -0.0643946, 0.0322177, -0.0047015],
            "B5": [-30.4609625, 354.2866078, -1073.2952368, 1315.2670101, -656.3702845, 96.5983711],
            "a1": [-0.0065985, 0.0242136, -0.0647001, 0.0265272, 0.0072169, -0.0020479],
            "a2": [-26.3337419, 435.6865552, -1359.8342546, 1724.6602381, -972.7565415, 200.3468023],
            "b1": [-80.7027159, 668.1710175, -2173.2445309, 3494.2393490, -2784.4670834, 881.2276510],
            "b2": [3.4685991, -41.2468479, 124.9729952, -153.2610078, 76.5242757, -11.2624113],
            "b3": [-39.6550497, 277.7202038, -777.0749505, 1081.5724508, -747.1056558, 204.5432666],
            "b4": [0.6514859, -4.7179961, 13.6742202, -19.7521659, 14.1873606, -4.0478845],
            "b5": [0.4695047, -3.6644336, 10.0039321, -5.1195905, -0.0007387, 0.0002360],
        }
        p = {key: self._poly_val(val) for key, val in COEFFS.items()}
        d_cm = self.slab_depths / 10.0
        A_over_a = p["A1"] * (1.0 - torch.exp(p["A2"] * torch.sqrt(d_cm**2 + p["A5"] ** 2))) * torch.exp(
            p["A3"] * d_cm + p["A4"] * (d_cm**2)
        )
        B_over_b = p["B1"] * (1.0 - torch.exp(p["B2"] * torch.sqrt(d_cm**2 + p["B5"] ** 2))) * torch.exp(
            p["B3"] * d_cm + p["B4"] * (d_cm**2)
        )
        a_val = p["a2"] + p["a1"] * d_cm
        b_val = p["b1"] * (1.0 - torch.exp(p["b2"] * torch.sqrt(d_cm**2 + p["b5"] ** 2))) * torch.exp(
            p["b3"] * d_cm + p["b4"] * (d_cm**2)
        )
        return {"A": A_over_a * a_val, "B": B_over_b * b_val, "a": a_val, "b": b_val}

    def _generate_kernels(self):
        k_size, k_half = 301, 150
        coords_cm = (torch.arange(-k_half, k_half + 1, dtype=torch.float32, device=self.device) * self.res) / 10.0
        grid_x, grid_y = torch.meshgrid(coords_cm, coords_cm, indexing="ij")
        r_cm = torch.sqrt(grid_x**2 + grid_y**2)
        r_safe = r_cm.clone()
        r_safe[r_safe < 1e-6] = 1e-6
        kernels = torch.zeros((len(self.slab_depths), 1, k_size, k_size), device=self.device)
        res_cm = self.res / 10.0
        R_eq = res_cm / np.sqrt(np.pi)
        for i in range(len(self.slab_depths)):
            A, B, a, b = self.params["A"][i], self.params["B"][i], self.params["a"][i], self.params["b"][i]
            val = (A * torch.exp(-a * r_safe) + B * torch.exp(-b * r_safe)) / r_safe
            term_A = (A / a) * (1.0 - torch.exp(-a * R_eq))
            term_B = (B / b) * (1.0 - torch.exp(-b * R_eq))
            val[r_cm < (res_cm * 0.1)] = (2 * np.pi * (term_A + term_B)) / (res_cm**2)
            kernels[i, 0] = val
        return kernels
