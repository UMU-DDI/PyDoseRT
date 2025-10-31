# Mask projections on MLC plan - without SSD

To get the 2D outline of each 3D structure (PTV or OAR) as seen from a given beam angle, you can:

1. **Define your MLC plane axes** $(u,v)$ exactly as in your dose engine.
2. **Collect the physical coordinates** of every voxel in your 3D mask.
3. **Project** those coordinates onto the $(u,v)$ plane via dot‐products.
4. **Rasterize** the resulting $(u,v)$ samples into a 2D occupancy image.
5. **Extract the boundary** of that 2D blob (e.g. with `skimage.measure.find_contours`) and draw it as a solid line.

Below is a step‐by‐step implementation. I’ll use NumPy + PyTorch for the math and `scikit-image` for the contour extraction.

```python
import numpy as np
import torch
from skimage import measure
import matplotlib.pyplot as plt

def project_structures_to_mlc_plane(
    masks_3d,              # torch.BoolTensor [C, W, D, H]
    voxel_sizes,           # tuple (dx,dy,dz) in mm
    isocenter,             # iterable (x0,y0,z0) in mm
    beam_direction,        # iterable (dx,dy,dz) unit vector
    plane_size=(200,200),  # (n_u, n_v) pixels for output image
    plane_extent=((-100,100),(-100,100))  # ((u_min,u_max),(v_min,v_max)) in mm
):
    """
    Projects each 3D mask into the MLC (u,v) plane at isocenter, and
    returns a list of 2D contours for each mask.
    """
    device = masks_3d.device
    C, W, D, H = masks_3d.shape
    dx, dy, dz = voxel_sizes
    isoc = torch.tensor(isocenter, device=device, dtype=torch.float32)

    # 1) compute beam plane axes (u,v) orthonormal to beam_direction
    bdir = torch.tensor(beam_direction, device=device, dtype=torch.float32)
    bdir = bdir / bdir.norm()
    # pick a reference non-collinear vector
    ref = torch.tensor([0,0,1], device=device)
    if torch.allclose(bdir, ref, atol=1e-6):
        ref = torch.tensor([0,1,0], device=device)
    u = torch.cross(bdir, ref); u /= u.norm()
    v = torch.cross(bdir, u);   v /= v.norm()

    contours_per_struct = []

    # grid of voxel centers (in mm)
    # we'll only enumerate where mask==1
    coords = torch.nonzero(masks_3d, as_tuple=False)  # [N,4]: [c, i,j,k]
    for c_id in range(C):
        # pick only this structure
        idx = coords[:,0] == c_id
        pts_vox = coords[idx,1:].float()  # [Nc, 3]
        if pts_vox.numel()==0:
            contours_per_struct.append([])
            continue

        # convert to physical mm
        pts_mm = pts_vox * torch.tensor([dx,dy,dz], device=device)

        # center about isocenter
        rel = pts_mm - isoc.unsqueeze(0)  # [Nc,3]

        # project onto (u,v)
        u_coords = (rel * u).sum(dim=1).cpu().numpy()  # [Nc]
        v_coords = (rel * v).sum(dim=1).cpu().numpy()

        # 2) rasterize into a 2D binary image
        (u_min,u_max),(v_min,v_max) = plane_extent
        Nu, Nv = plane_size
        # map continuous coords to pixel indices
        u_idx = ((u_coords - u_min)/(u_max-u_min)*(Nu-1)).astype(int)
        v_idx = ((v_coords - v_min)/(v_max-v_min)*(Nv-1)).astype(int)
        # clamp
        mask2d = np.zeros((Nu,Nv), dtype=np.uint8)
        valid = (u_idx>=0)&(u_idx<Nu)&(v_idx>=0)&(v_idx<Nv)
        mask2d[u_idx[valid], v_idx[valid]] = 1

        # 3) extract contour(s)
        # skimage.measure.find_contours expects mask with 1s
        contours = measure.find_contours(mask2d, level=0.5)
        # convert pixel contours back to mm for plotting
        conts_mm = []
        for cnt in contours:
            # cnt is an array [M,2] giving (row, col) floats in pixel coords
            # row->u_idx, col->v_idx
            u_pix = cnt[:,0]
            v_pix = cnt[:,1]
            u_mm = u_min + (u_pix/(Nu-1))*(u_max-u_min)
            v_mm = v_min + (v_pix/(Nv-1))*(v_max-v_min)
            conts_mm.append(np.stack([u_mm,v_mm],axis=1))
        contours_per_struct.append(conts_mm)

    return contours_per_struct


# -----------------------------
# Example usage & plotting
# -----------------------------
# masks_3d: torch.BoolTensor [C,W,D,H]
# let's say C=2 (PTV, OAR)
#
# define your parameters:
voxel_sizes = (1.0,1.0,1.0)
isocenter    = (64.0,64.0,64.0)  # mm
beam_dir     = (0.0, -1.0, 0.0)  # direct anterior beam

contours = project_structures_to_mlc_plane(
    masks_3d,
    voxel_sizes,
    isocenter,
    beam_dir,
    plane_size=(200,200),
    plane_extent=((-100,100),(-100,100))
)

fig, ax = plt.subplots()
for c_id, struct_conts in enumerate(contours):
    for cnt in struct_conts:
        ax.plot(cnt[:,0], cnt[:,1],
                linestyle='-', linewidth=2,
                label=f"struct {c_id}" if cnt is struct_conts[0] else None)
ax.set_aspect('equal')
ax.set_xlabel('u (mm)')
ax.set_ylabel('v (mm)')
ax.legend()
plt.show()
```

### What it does

1. **Voxel enumeration**
   Finds every $(i,j,k)$ where a mask is 1 and converts it to world‐mm $(x,y,z)$.

2. **Plane axes**
   Computes two orthonormal vectors `u` and `v` perpendicular to your beam direction.

3. **Projection**
   Projects each point’s $(x,y,z)$ vector into scalars

   $$
     u = (r - \text{isocenter})\cdot \mathbf{u},\quad
     v = (r - \text{isocenter})\cdot \mathbf{v}.
   $$

4. **Rasterization**
   Maps continuous $(u,v)$ to integer pixel indices in a 2D array.

5. **Contour extraction**
   Uses `skimage.measure.find_contours` on the binary image to find the outer boundary.

6. **Plotting**
   Converts pixel‐space contours back to $(u,v)$-mm and draws solid lines.

—now you’ll get crisp, solid‐line outlines of PTV and each OAR in the MLC plane.



---
# Mask projections on MLC plan - with SSD

## Detailed Commentary

1. **Module docstring**

   ```python
   """
   mlc_mask_projection.py

   Project 3D structure masks (PTV, OARs) into the MLC (u,v) planes for multiple gantry angles,
   accounting for a finite source-to-isocenter distance (SSD), compute convex-hull contours,
   and plot them in a grid of subplots.
   """
   ```

   – Explains the purpose and usage.

2. **Imports**

   * `numpy` for numeric arrays,
   * `torch` for tensor and mask handling,
   * `skimage.measure` (not strictly used here, but kept for completeness),
   * `matplotlib.pyplot` for plotting,
   * `math` for trigonometric calculations,
   * `scipy.spatial.ConvexHull` for boundary extraction,
   * `TestSetup` from your utilities to generate dummy data.

3. **`get_beam_axes(beam_dir)`**

   * Takes a **unit‐length** 3D torch tensor `beam_dir`.
   * If `beam_dir ≈ (0, 1, 0)`, set `(u,v) = ((1,0,0), (0,0,1))`.
   * Otherwise pick a “reference” vector (`ref`) that’s not colinear with `beam_dir` (use `(0,0,1)` unless `beam_dir.z` is near ±1).
   * Compute `u = normalize(cross(beam_dir, ref))` and `v = normalize(cross(beam_dir, u))`.
     – Returns two orthonormal axes `u` and `v` spanning the MLC plane.

4. **`project_mask_to_uv_divergent(...)`**

   * **Arguments**:

     * `mask_3d`: 3D binary mask `[W, D, H]`.
     * `voxel_sizes`: `(dx,dy,dz)` in mm.
     * `isocenter`: `(x0,y0,z0)` in mm.
     * `beam_dir`: 3D unit vector.
     * `u, v`: in‐plane axes from `get_beam_axes`.
     * `ssd`: source‐to‐isocenter distance in mm.
   * **Steps**:

     1. Find all nonzero voxels: `coords = torch.nonzero(mask_3d) → [N,3] = (i,j,k)`.
     2. Convert `(i,j,k)` to physical mm: `pts_mm = coords * [dx,dy,dz]`.
     3. Compute `source = isocenter - ssd * beam_dir`.
     4. For each `P = pts_mm[n]`, define ray `R(t) = S + t*(P - S)`. Solve `(R(t) - I)·D = 0`.

        * `(I - S)·D` is a scalar, call it `num`.
        * `(P - S)·D` is a scalar per point, call it `den`.
        * `t = num/den` for `den ≠ 0`.
        * `R = S + t*(P - S)`.
     5. Now `(R - I)` is purely in the MLC plane. Project onto `u` and `v`:

        $$
          u_i = (R_i - I)·u,\quad v_i = (R_i - I)·v.
        $$

        Return an `N×2` numpy array of `(u_i, v_i)`.

5. **`plot_struct_contours_for_beams(...)`**

   * **Arguments**:

     * `masks_3d`: `[B,W,D,H,C]` (we assume `B=1`).
     * `struct_keys`: list of length `C` naming each channel, e.g. `["PTV","ROI1",…]`.
     * `voxel_sizes`: `(dx,dy,dz)`.
     * `isocenter`: `(x0,y0,z0)` in mm.
     * `ssd`: source distance in mm.
     * `number_of_cps`: how many gantry angles (evenly spaced 0..360°).
     * `cols`: how many columns in the subplot grid.
     * `structure_names`: dict `key→display_name`.
     * `roi_colors`: dict `key→color` (matplotlib).
   * **Body**:

     1. Assert batch size `B == 1`.
     2. If no custom names or colors provided, default to key or `None`.
     3. Convert `isocenter` to a torch tensor on `device`.
     4. Create a grid of subplots: `rows = ceil(number_of_cps/cols)`.
     5. Loop over `i = 0..number_of_cps-1`:

        * Compute `angle_deg = i*(360/number_of_cps)`.
        * Build `beam_dir = (sinθ, cosθ, 0)`, normalize.
        * Get `(u,v)` from `get_beam_axes(beam_dir)`.
        * For each structure channel `c`:
          • Extract binary mask `[W,D,H]`
          • Call `project_mask_to_uv_divergent(...)` to get `proj[N,2]`.
          • If `N < 3`, skip (cannot form hull).
          • Otherwise compute `hull = ConvexHull(proj)`, get `vertices = hull.vertices`, form `poly = proj[vertices]`, then close it: `poly = vstack([poly, poly[0]])`.
          • Plot `poly[:,0], poly[:,1]` with label and color.
        * Show legend.
     6. Turn off any unused axes (if `number_of_cps < rows*cols`).
     7. Add a suptitle. `plt.show()`.

6. **`if __name__ == "__main__":`**

   * Uses `TestSetup()` to generate dummy masks (argument to your test harness).
   * Converts `masks` → `torch.Tensor(masks)` on GPU/CPU.
   * Defines `struct_keys`, `structure_names`, `roi_colors`, `voxel_sizes`, `isocenter`, and a realistic `ssd`.
   * Calls `plot_struct_contours_for_beams(...)` with `number_of_cps=8`, `cols=4`.

---

### 📋 How SSD Affects the Projection

* **Without SSD** (orthographic projection), one would simply do `(P − I)·u` and `(P − I)·v`. That implicitly assumes rays are parallel and originate at infinity.
* **With SSD** (divergent beam), each ray actually emanates from a point `S = I − D·ssd`. Then each voxel `P` lies somewhere between source and plane. We find where the line from `S` through `P` meets the plane at `I`. The math solves

  $$
    R(t) = S + t \,(P - S), \quad\text{s.t.}\;(R(t) - I)\cdot D = 0 
    \;\Rightarrow\; t = \frac{(I-S)\cdot D}{(P-S)\cdot D}.
  $$

  This `R(t)` is guaranteed to lie in the plane (perpendicular to `D` through `I`). Finally, `(R - I)` is purely in-plane, so projecting onto `(u,v)` gives the correct geometric divergence.

---

#### Script Name

We’ve named it:

```
mlc_mask_projection.py
```

because it clearly communicates that this script **projects** 3D **masks** into the MLC plane (u,v) for plotting.

---

With this in place, your contour lines in each MLC subplot will be properly **“closed”** and incorporate true beam divergence through SSD.
