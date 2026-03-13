"""
This module provides functions and classes for generating pencil beam kernels used in radiotherapy dose calculation.

It includes precomputed polynomial coefficients for kernel parameterization, utility functions 
for Gaussian kernel generation, and the PencilBeamModel class for calculating dose kernels 
based on radiological depth and beam parameters.
"""

import torch

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
        tpr (torch.Tensor): Tissue phantom ratio (TPR 20/10) for beam quality.
        config (object): Configuration object containing resolution and TPR.
        kernel_size (int): Size of the kernel (number of pixels).
        params (dict): Precomputed kernel parameters for the given TPR.
        rs (torch.Tensor): Radial distance grid for kernel calculation.
    """
    def __init__(self, 
                 resolution, 
                 tpr_20_10, 
                 kernel_size: int) -> 'PencilBeamModel':
        """
        Initialize the PencilBeamModel.

        Args:
            config (MachineConfig): Configuration object with TPR and resolution.
            kernel_size (int): Size of the kernel (number of pixels) in the dimension with smaller pixel size.
        """
        self.tpr = tpr_20_10
        self.resolution = resolution
        self.res_h, self.res_w = resolution[0] / 10, resolution[2] / 10

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
        d_100mm = 100.0 * torch.ones((1, 1, 1, 1)) # 100mm depth
        self.norm = torch.max(self.get_pencil_beam(d=d_100mm, r=self.rs[torch.newaxis, torch.newaxis, :, :], normalize=False))

    def get_param(self, parameter: str, TPR: torch.Tensor) -> torch.Tensor:
        """
        Calculate parameter value for a given TPR using polynomial coefficients.

        Args:
            parameter (str): Parameter name (key in coeffs).
            TPR (torch.Tensor): Tissue phantom ratio.

        Returns:
            Tensor: Computed parameter value.
        """
        return sum(c * TPR**i for i, c in enumerate(coeffs[parameter]))

    def depth_A(self, d: torch.Tensor) -> torch.Tensor:
        """
        Compute the A component of the kernel at depth d.

        Args:
            d (torch.Tensor): Depth in cm.

        Returns:
            torch.Tensor: A component value.
        """
        return self.depth_A_per_a(d) * self.depth_a(d)

    def depth_B(self, d: torch.Tensor) -> torch.Tensor:
        """
        Compute the B component of the kernel at depth d.

        Args:
            d (torch.Tensor): Depth in cm.

        Returns:
            torch.Tensor: B component value.
        """
        return self.depth_B_per_b(d) * self.depth_b(d)

    def depth_A_per_a(self, d: torch.Tensor) -> torch.Tensor:
        """
        Compute the A/a term for the kernel at depth d.

        Args:
            d (torch.Tensor): Depth in cm.

        Returns:
            torch.Tensor: A/a term value.
        """
        return (
            self.params["A1"]
            * (1 - torch.exp(self.params["A2"] * torch.sqrt(d**2 + self.params["A5"] ** 2)))
            * torch.exp(self.params["A3"] * d + self.params["A4"] * d**2)
        )

    def depth_B_per_b(self, d: torch.Tensor) -> torch.Tensor:
        """
        Compute the B/b term for the kernel at depth d.

        Args:
            d (torch.Tensor): Depth in cm.

        Returns:
            torch.Tensor: B/b term value.
        """
        return (
            self.params["B1"]
            * (1 - torch.exp(self.params["B2"] * torch.sqrt(d**2 + self.params["B5"] ** 2)))
            * torch.exp(self.params["B3"] * d + self.params["B4"] * d**2)
        )

    def depth_a(self, d: torch.Tensor) -> torch.Tensor:
        """
        Compute the a parameter for the kernel at depth d.

        NB: The equation from the original print had the parameters a1 and a2 flipped in this equation according to
        the corrigendum published in  Radiotherapy & oncology Volume 98, Issue 2p286February 2011

        Args:
            d (torch.Tensor): Depth in cm.

        Returns:
            torch.Tensor: a parameter value.
        """
        return self.params["a2"] + self.params["a1"] * d

    def depth_b(self, d: torch.Tensor) -> torch.Tensor:
        """
        Compute the b parameter for the kernel at depth d.

        Args:
            d (torch.Tensor): Depth in cm.

        Returns:
            torch.Tensor: b parameter value.
        """
        return (
            self.params["b1"]
            * (
                1
                - torch.exp(self.params["b2"] * torch.sqrt((d**2) + (self.params["b5"] ** 2)))
            )
            * torch.exp((self.params["b3"] * d) + (self.params["b4"] * d**2))
        )

    def get_pencil_beam(self, 
                        d: torch.Tensor, 
                        r: torch.Tensor, 
                        normalize: bool = True,
                        apply_circular_mask: bool = False, 
                        mask_radius_cm: float = None,
                        depth_threshold_mm: float = 0.5) -> torch.Tensor:
        """
        Generate pencil beam kernel for given depths and radial grid.

        Args:
            d (torch.Tensor): Radiological depth [mm], shape (B*G, N, 1).  # TODO: Fix this documentation as it does not correspond with the implementation
            r (torch.Tensor): Radial grid [mm], shape (Hk, Wk).
            normalize (bool): Normalize to the unit kernel at 10cm radiological depth.            
            depth_threshold_mm (float): Minimum radiological depth [mm] below which kernel is zero. Default is 0.5mm.

        Returns:
            torch.Tensor: Pencil beam kernel, shape (B*G, N, Hk, Wk).
        """
        # ensure r2 has same dtype/device as d and a broadcastable shape (1,1,Hk,Wk)
        r2 = r.to(d.dtype).to(d.device)
        if r2.ndim == 2:
            r2 = r2.view(1, 1, *r2.shape)  # (1,1,Hk,Wk)
        BG, N, _, _ = d.shape
        _, _, Hk, Wk = r2.shape

        # fast return if everything below original mm threshold (d given in mm)
        if torch.all(d < depth_threshold_mm):
            return torch.zeros((BG, N, Hk, Wk), dtype=torch.float32, device=d.device)

        # keep caller's tensor unchanged: work on local copy for unit conversion
        d = d / 10.0  # convert mm -> cm, shape (BG, N, 1, 1)
        mask = (r2 > 0.0)  # (1,1,Hk,Wk) boolean

        # depth-dependent scalars (broadcastable as (BG,N,1,1))
        depth_a = self.depth_a(d)       # (BG,N,1,1)
        depth_b = self.depth_b(d)       # (BG,N,1,1)
        A_over_a = self.depth_A_per_a(d)  # (BG,N,1,1)
        B_over_b = self.depth_B_per_b(d)  # (BG,N,1,1)
        depth_A = A_over_a * depth_a
        depth_B = B_over_b * depth_b

        # allocate numerator buffer (BG, N, Hk, Wk) to avoid extra temporaries
        K_numer = torch.empty((BG, N, Hk, Wk), dtype=depth_A.dtype, device=depth_A.device)

        # compute first term in-place: K_numer := depth_A * exp(-depth_a * r2)
        # broadcasting: (BG,N,1,1) * (1,1,Hk,Wk) -> (BG,N,Hk,Wk)
        torch.exp((-depth_a) * r2, out=K_numer)  # K_numer = exp(-depth_a * r2)
        K_numer.mul_(depth_A)                    # K_numer = depth_A * exp(...)

        # compute second term into a temporary then add: tmp := depth_B * exp(-depth_b * r2); K_numer += tmp
        tmp = torch.empty((BG, N, Hk, Wk), dtype=depth_B.dtype, device=depth_B.device)
        torch.exp((-depth_b) * r2, out=tmp)  # tmp = exp(-depth_b * r2)
        tmp.mul_(depth_B)                     # tmp = depth_B * exp(...)
        K_numer.add_(tmp)                     # K_numer = term1 + term2
        del tmp                                # release reference quickly

        # safe division by r2 for r > 0:
        # build a denominator that is r2 where mask True, else 1.0 where mask False.
        # denom has shape (1,1,Hk,Wk) and will broadcast over BG,N.
        r2_f = r2.to(K_numer.dtype)
        denom = torch.where(mask, r2_f, torch.ones_like(r2_f))

        # in-place division uses broadcasting: divides every BG,N plane by denom
        K_numer = K_numer / denom  # produces correct division for r>0; for r==0 we will overwrite below

        # center pixel: area-average over the pixel disk (BG,N,1,1)
        dx = torch.tensor(self.resolution[0] / 10.0, dtype=torch.float32, device=d.device)
        dy = torch.tensor(self.resolution[2] / 10.0, dtype=torch.float32, device=d.device)
        r_h = torch.sqrt(dx * dy / torch.pi)
        center_val = (2.0 / (r_h * r_h)) * (
            A_over_a * (1.0 - torch.exp(-depth_a * r_h)) +
            B_over_b * (1.0 - torch.exp(-depth_b * r_h))
        )  # (BG, N, 1, 1)

        # find center pixel coordinates (could be more than one if the grid has multiple zeros)
        center_locations = torch.nonzero(~mask[0, 0], as_tuple=False)  # tensor of (ih, iw) rows
        if center_locations.numel() > 0:
            center_val_flat = center_val.view(BG, N)  # (BG, N)
            # write center_val into each center pixel location
            for (ih, iw) in center_locations:
                K_numer[:, :, ih.item(), iw.item()] = center_val_flat

        # now K_numer is the kernel before normalization/masking/depth threshold
        K = K_numer

        # normalize if requested
        if normalize:
            # do an in-place style division via /= to avoid allocating a new large tensor
            K = K / self.norm

        # optional circular mask: zero-out values beyond mask_radius_cm
        if apply_circular_mask:
            if mask_radius_cm is None:
                mask_radius_cm = min(Hk, Wk) * max(self.res_h, self.res_w) / 2.0
            circular_mask = (r2 <= mask_radius_cm)  # (1,1,Hk,Wk) boolean
            # multiplication broadcasts circular_mask across BG,N
            K = K * circular_mask

        # zero out kernels where radiological depth is below threshold (d currently in cm)
        depth_threshold_cm = depth_threshold_mm / 10.0
        depth_mask = (d >= depth_threshold_cm)  # (BG, N, 1, 1)
        K = K * depth_mask  # broadcast over Hk,Wk

        return K.to(torch.float32)
    
    def get_rs(self, kernel_size: tuple) -> torch.Tensor:
        """
        Compute radial distance grid for kernel calculation.

        Args:
            kernel_size (list or tuple): Size of the kernel [H, W].

        Returns:
            torch.Tensor: Radial distance grid in cm.
        """
        
        h = torch.arange(0, kernel_size[0], dtype=torch.int32)
        w = torch.arange(0, kernel_size[1], dtype=torch.int32)
        h -= kernel_size[0] // 2
        w -= kernel_size[1] // 2
        w, h = torch.meshgrid(w, h, indexing="ij")

        dh = torch.abs(h.to(torch.float32)) * self.res_h
        dw = torch.abs(w.to(torch.float32)) * self.res_w

        rs = torch.sqrt(dh**2 + dw**2)
        
        return rs

    def get_nested_kernels(self, radiological_depth: torch.Tensor) -> torch.Tensor:
        """
        Generate kernels for nested radiological depths.

        Args:
            radiological_depth (torch.Tensor): Array of radiological depths.

        Returns:
            torch.Tensor: Nested kernels for all depths.
        """
        return self.get_pencil_beam(
            d=radiological_depth[..., 0, torch.newaxis, torch.newaxis],
            r=self.rs[torch.newaxis, torch.newaxis, :, :],
        )

    def R_limit(self, d: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel support radius limit for a given depth and field size.

        Args:
            d (torch.Tensor): Depth in cm.
            F (torch.Tensor): Field size parameter.

        Returns:
            Tensor: Radius limit for kernel support.
        """
        # TODO: Define the origin of the "magic" variables/values here
        return 0.561 * ((90 + d) / (90 + 10)) * F

