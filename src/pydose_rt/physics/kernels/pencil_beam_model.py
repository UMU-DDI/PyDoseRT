#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module provides functions and classes for generating pencil beam kernels used in radiotherapy dose calculation.

It includes precomputed polynomial coefficients for kernel parameterization, utility functions 
for Gaussian kernel generation, and the PencilBeamModel class for calculating dose kernels 
based on radiological depth and beam parameters.
"""

import numpy as np
from pydose_rt.data.machine_config import MachineConfig

coeffs = {
    "A1": [
        0.0128018,
        -0.0577391,
        0.1790839,
        -0.2467955,
        0.1328192,
        -0.0194684,
    ],
    "A2": [
        16.7815028,
        -279.4672663,
        839.0016549,
        -978.4915013,
        470.5317337,
        -69.2485573,
    ],
    "A3": [
        -0.0889669,
        -0.2587584,
        0.7069203,
        -0.3654033,
        0.0029760,
        -0.0003786,
    ],
    "A4": [
        0.0017089,
        -0.0169150,
        0.0514650,
        -0.0639530,
        0.0324490,
        -0.0049121,
    ],
    "A5": [
        0.1431447,
        -0.2134626,
        0.5825546,
        -0.2969273,
        -0.0011436,
        0.0002219,
    ],
    "B1": [
        -42.7607523,
        264.3424720,
        -633.4540368,
        731.5311577,
        -402.5280374,
        82.4936551,
    ],
    "B2": [
        0.2428359,
        -2.5029336,
        7.6128101,
        -9.5273454,
        4.8249840,
        -0.7097852,
    ],
    "B3": [
        -0.0910420,
        -0.2621605,
        0.7157244,
        -0.3664126,
        0.0000930,
        -0.0000232,
    ],
    "B4": [
        0.0017284,
        -0.0172146,
        0.0522109,
        -0.0643946,
        0.0322177,
        -0.0047015,
    ],
    "B5": [
        -30.4609625,
        354.2866078,
        -1073.2952368,
        1315.2670101,
        -656.3702845,
        96.5983711,
    ],
    "a1": [
        -0.0065985,
        0.0242136,
        -0.0647001,
        0.0265272,
        0.0072169,
        -0.0020479,
    ],
    "a2": [
        -26.3337419,
        435.6865552,
        -1359.8342546,
        1724.6602381,
        -972.7565415,
        200.3468023,
    ],
    "b1": [
        -80.7027159,
        668.1710175,
        -2173.2445309,
        3494.2393490,
        -2784.4670834,
        881.2276510,
    ],
    "b2": [
        3.4685991,
        -41.2468479,
        124.9729952,
        -153.2610078,
        76.5242757,
        -11.2624113,
    ],
    "b3": [
        -39.6550497,
        277.7202038,
        -777.0749505,
        1081.5724508,
        -747.1056558,
        204.5432666,
    ],
    "b4": [
        0.6514859,
        -4.7179961,
        13.6742202,
        -19.7521659,
        14.1873606,
        -4.0478845,
    ],
    "b5": [
        0.4695047,
        -3.6644336,
        10.0039321,
        -5.1195905,
        -0.0007387,
        0.0002360,
    ],
}


class PencilBeamModel:
    """
    Model for generating pencil beam dose kernels for radiotherapy dose calculation.

    Attributes:
        tpr (float): Tissue phantom ratio (TPR 20/10) for beam quality.
        config (object): Configuration object containing resolution and TPR.
        kernel_size (int): Size of the kernel (number of pixels).
        params (dict): Precomputed kernel parameters for the given TPR.
        rs (np.ndarray): Radial distance grid for kernel calculation.
    """
    def __init__(self, config: MachineConfig, kernel_size: int):
        """
        Initialize the PencilBeamModel.

        Args:
            config (MachineConfig): Configuration object with TPR and resolution.
            kernel_size (int): Size of the kernel (number of pixels) in the dimension with smaller pixel size.
        """
        self.tpr = config.tpr_20_10
        self.config = config
        self.res_h, self.res_w = self.config.resolution[0] / 10, self.config.resolution[2] / 10

        # Determine which dimension has smaller pixel size

        if self.res_h <= self.res_w:
            kernel_size_w = kernel_size
            kernel_size_h = int(round(kernel_size * (self.res_w / self.res_h)))
        else:
            kernel_size_w = int(round(kernel_size * (self.res_h / self.res_w)))
            kernel_size_h = kernel_size

        # Ensure both kernel sizes are odd for more efficient convolution
        if kernel_size_h % 2 == 0:
            kernel_size_h += 1
        if kernel_size_w % 2 == 0:
            kernel_size_w += 1

        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.params = {k: self.get_param(k, self.tpr) for k in coeffs.keys()}
        self.rs = self.get_rs(
            [self.kernel_size_h, self.kernel_size_w]
        )  # Calculate the radial distance
        d_10cm = 10.0 * np.ones((1, 1, 1, 1)) # 100mm depth
        self.norm = np.sum(self.get_pencil_beam(d=d_10cm, r=self.rs[np.newaxis, np.newaxis, :, :], normalize=False))

    def get_param(self, parameter: str, TPR: float) -> float:
        """
        Calculate parameter value for a given TPR using polynomial coefficients.

        Args:
            parameter (str): Parameter name (key in coeffs).
            TPR (float): Tissue phantom ratio.

        Returns:
            float: Computed parameter value.
        """
        return sum(c * TPR**i for i, c in enumerate(coeffs[parameter]))

    def depth_A(self, d: float) -> float:
        """
        Compute the A component of the kernel at depth d.

        Args:
            d (float or np.ndarray): Depth in cm.

        Returns:
            float or np.ndarray: A component value.
        """
        return self.depth_A_per_a(d) * self.depth_a(d)

    def depth_B(self, d: float) -> float:
        """
        Compute the B component of the kernel at depth d.

        Args:
            d (float or np.ndarray): Depth in cm.

        Returns:
            float or np.ndarray: B component value.
        """
        return self.depth_B_per_b(d) * self.depth_b(d)

    def depth_A_per_a(self, d: float | np.ndarray) -> float:
        """
        Compute the A/a term for the kernel at depth d.

        Args:
            d (float or np.ndarray): Depth in cm.

        Returns:
            float or np.ndarray: A/a term value.
        """
        return (
            self.params["A1"]
            * (1 - np.exp(self.params["A2"] * np.sqrt(d**2 + self.params["A5"] ** 2)))
            * np.exp(self.params["A3"] * d + self.params["A4"] * d**2)
        )

    def depth_B_per_b(self, d: float | np.ndarray) -> float:
        """
        Compute the B/b term for the kernel at depth d.

        Args:
            d (float or np.ndarray): Depth in cm.

        Returns:
            float or np.ndarray: B/b term value.
        """
        return (
            self.params["B1"]
            * (1 - np.exp(self.params["B2"] * np.sqrt(d**2 + self.params["B5"] ** 2)))
            * np.exp(self.params["B3"] * d + self.params["B4"] * d**2)
        )

    def depth_a(self, d: float | np.ndarray) -> float:
        """
        Compute the a parameter for the kernel at depth d.

        NB: The equation from the original print had the parameters a1 and a2 flipped in this equation according to
        the corrigendum published in  Radiotherapy & oncology Volume 98, Issue 2p286February 2011

        Args:
            d (float or np.ndarray): Depth in cm.

        Returns:
            float or np.ndarray: a parameter value.
        """
        return self.params["a2"] + self.params["a1"] * d

    def depth_b(self, d: float | np.ndarray) -> float:
        """
        Compute the b parameter for the kernel at depth d.

        Args:
            d (float or np.ndarray): Depth in cm.

        Returns:
            float or np.ndarray: b parameter value.
        """
        return (
            self.params["b1"]
            * (
                1
                - np.exp(self.params["b2"] * np.sqrt((d**2) + (self.params["b5"] ** 2)))
            )
            * np.exp((self.params["b3"] * d) + (self.params["b4"] * d**2))
        )

    def get_pencil_beam(self, d: np.ndarray, r: np.ndarray, normalize: bool = True,
                        add_source_blur: bool = False, src_fwhm_mm_iso: float = 2.5,
                        SAD_cm: float = 100.0, SSD_cm: float = 100.0) -> np.ndarray:
        """
        Generate pencil beam kernel for given depths and radial grid.

        Args:
            d (np.ndarray): Radiological depth [mm], shape (B*G, N, 1).  # TODO: Fix this documentation as it does not correspond with the implementation
            r (np.ndarray): Radial grid [mm], shape (Hk, Wk).
            normalize (bool): Normalize to the unit kernel at 10cm radiological depth.
            add_source_blur (bool): Whether to add geometric penumbra (source blur).
            src_fwhm_mm_iso (float): Source FWHM at isocenter [mm].
            SAD_cm (float): Source-to-axis distance [mm].
            SSD_cm (float): Source-to-surface distance [mm].

        Returns:
            np.ndarray: Pencil beam kernel, shape (B*G, N, Hk, Wk).
        """
        # shapes
        d = np.asarray(d, float)
        r2 = np.asarray(r, float)                # (Hk, Wk)

        # Convert radiological depth to cm from mm
        d /= 10

        BG, N, _, _ = d.shape
        _, _, Hk, Wk = r2.shape
        mask = (r2 > 0.0)


        # depth-dependent pieces (broadcast over BG,N)
        depth_a = self.depth_a(d)              # (BG,N,1,1)
        depth_b = self.depth_b(d)               # (BG,N,1,1)
        A_over_a = self.depth_A_per_a(d)        # (BG,N,1,1)
        B_over_b = self.depth_B_per_b(d)        # (BG,N,1,1)
        depth_A = A_over_a * depth_a
        depth_B = B_over_b * depth_b

        # numerator everywhere
        exact_num = (depth_A * np.exp(-depth_a * r2)) + (depth_B * np.exp(-depth_b * r2))  # (BG,N,Hk,Wk)


        # safe divide for r>0
        exact = np.empty_like(exact_num)
        np.divide(exact_num, r2, out=exact, where=mask)

        # center pixel: area-average over a disk whose area = one pixel
        dx = float(self.config.resolution[0] / 10.0)
        dy = float(self.config.resolution[2] / 10.0)
        r_h = np.sqrt(dx * dy / np.pi)
        center_val = (2.0 / (r_h * r_h)) * (
            A_over_a * (1.0 - np.exp(-depth_a * r_h)) +
            B_over_b * (1.0 - np.exp(-depth_b * r_h))
        )                                                    # (BG,N,1,1)

        K = np.where(mask, exact, center_val)               # (BG,N,Hk,Wk)

        # ---- optional source blur (geometric penumbra) ----
        if add_source_blur:
            # magnification at depth
            M = (SSD_cm + d)[..., None] / SAD_cm            # (BG,N,1,1)
            sigma_iso_cm = (src_fwhm_mm_iso / 10.0) / (2.0 * np.sqrt(2.0*np.log(2.0)))
            sigma_cm = np.maximum(sigma_iso_cm * M, 1e-8)   # (BG,N,1,1)

            # build Gaussian per (BG,N)
            yy, xx = np.mgrid[-(Hk//2):(Hk - Hk//2), -(Wk//2):(Wk - Wk//2)]
            xx = xx[None, None, :, :] / dx
            yy = yy[None, None, :, :] / dy
            sig_pix = sigma_cm / np.array([dx, dy])[None, None, None, :]  # (BG,N,1,2)
            sigx = sig_pix[..., 0]
            sigy = sig_pix[..., 1]
            G = np.exp(-0.5 * ((xx / sigx) ** 2 + (yy / sigy) ** 2))
            G /= G.sum(axis=(-2, -1), keepdims=True)

            # FFT conv (same size)
            padH, padW = Hk + Hk - 1, Wk + Wk - 1
            F_K = np.fft.rfftn(K, s=(padH, padW), axes=(-2, -1))
            F_G = np.fft.rfftn(G, s=(padH, padW), axes=(-2, -1))
            K_full = np.fft.irfftn(F_K * F_G, s=(padH, padW), axes=(-2, -1))
            i0, j0 = (Hk - 1) // 2, (Wk - 1) // 2
            K = K_full[..., i0:i0+Hk, j0:j0+Wk]

        # Normalize to 10cm depth kernel integral
        if normalize:
            K /= self.norm

        return np.array(K, dtype=np.float32)  # (BG, N, Hk, Wk)

    def apply_flattening(self, kernel: np.ndarray, r: np.ndarray, max_radius: float = 16, alpha: float = 0.2, beta: float = 2.0) -> np.ndarray:
        """
        Apply flattening filter to the kernel for beam profile correction.

        Args:
            kernel (np.ndarray): Kernel to be flattened.
            r (np.ndarray): Radial grid.
            max_radius (float): Maximum radius for flattening.
            alpha (float): Flattening parameter.
            beta (float): Flattening parameter.

        Returns:
            np.ndarray: Flattened kernel.
        """
        flattening_factor = alpha - beta * (r / max_radius) ** 2
        flattening_factor = np.clip(
            flattening_factor, 0.5, 1.0
        )  # Clip for stability
        return kernel * flattening_factor

    def get_rs(self, kernel_size: tuple) -> np.ndarray:
        """
        Compute radial distance grid for kernel calculation.

        Args:
            kernel_size (list or tuple): Size of the kernel [H, W].

        Returns:
            np.ndarray: Radial distance grid in cm.
        """
        
        h = np.arange(0, kernel_size[0], dtype=np.int32)
        w = np.arange(0, kernel_size[1], dtype=np.int32)
        h -= kernel_size[0] // 2
        w -= kernel_size[1] // 2
        w, h = np.meshgrid(w, h)

        dh = np.abs(h.astype(np.float32)) * self.res_h
        dw = np.abs(w.astype(np.float32)) * self.res_w

        rs = np.sqrt(dh**2 + dw**2)
        
        return rs

    def get_nested_kernels(self, radiological_depth: np.ndarray) -> np.ndarray:
        """
        Generate kernels for nested radiological depths.

        Args:
            radiological_depth (np.ndarray): Array of radiological depths.

        Returns:
            np.ndarray: Nested kernels for all depths.
        """
        return self.get_pencil_beam(
            d=radiological_depth[..., 0, np.newaxis, np.newaxis],
            r=self.rs[np.newaxis, np.newaxis, :, :],
        )

    def R_limit(self, d: float, F: float) -> float:
        """
        Compute the kernel support radius limit for a given depth and field size.

        Args:
            d (float): Depth in cm.
            F (float): Field size parameter.

        Returns:
            float: Radius limit for kernel support.
        """
        # TODO: Define the origin of the "magic" variables/values here
        return 0.561 * ((90 + d) / (90 + 10)) * F

