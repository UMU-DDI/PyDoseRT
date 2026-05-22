"""
Patient configuration - CT dimensions and geometric parameters.
"""
# from pydantic import BaseModel, Field, model_validator
from typing import List
import torch
from pathlib import Path
import numpy as np
from pydosert.data.utils.dicom_utils import load_ct_series, load_structures, load_dose, fetch_plan_data
from pydosert.data import Patient, BeamSequence
import SimpleITK as sitk
from typing import List, Dict, Any, Tuple

def load_dicom(
    ct_folder: Path,
    dose_path: List[Path] | Path | None,
    plan_path: Path | None,
    struct_path: Path | None,
    struct_names: List[str] | None = None,
    use_delivery: bool = False,
    new_spacing: tuple[float, float, float] = (2.0, 2.0, 2.0),
    crop_volume: bool = True,
    device: torch.device | str = 'cuda',
    dtype: torch.dtype = torch.float32,
) -> tuple['Patient', 'BeamSequence']:
    """    
    Load DICOM data and create Patient and BeamSequence.
    
    Args:
        ct_folder (Path): Path to folder containing CT DICOM files.
        dose_path (List[Path] | Path | None): Path(s) to RTDOSE file(s).
        plan_path (Path | None): Path to RTPLAN file.
        struct_path (Path | None): Path to RTSTRUCT file.
        struct_names (List[str] | None): List of structure names to load (None = all).
        use_delivery (bool): If True, configure for delivery positions (N averaged).
            If False (default), configure for raw control points (N+1 from DICOM).
        new_spacing (tuple[float, float, float]): Target voxel spacing (z, y, x) in mm.
        crop_volume (bool): If True, center-crop the axial plane to 40 cm.
        device (torch.device | str): Device for BeamSequence tensors.
        dtype (torch.dtype): Data type for BeamSequence tensors.
    Returns:
        tuple[Patient, list[BeamSequence]]: Patient (CT/dose/structure tensors of
            shape [D, H, W]) and the list of per-beam BeamSequence objects.
    Note:
        When use_delivery=True:
        - BeamSequence contains N delivery positions (averaged from N+1 control points)
        - DoseEngine can be created directly with this config
        When use_delivery=False:
        - BeamSequence contains N+1 raw control points from DICOM
        - Call beam_seq.to_delivery() before dose calculation
    """
    ct_series, ref = load_ct_series(ct_folder)
    structures = load_structures(ct_series, ct_folder, struct_path, struct_names=struct_names)

    if isinstance(dose_path, Path):
        dose_path = [ dose_path ]

    if isinstance(plan_path, Path):
        plan_path = [ plan_path ]

    doses = dict()
    for path in dose_path:
        dose, plan_ref = load_dose(path)
        doses[plan_ref] = dose

    # If RTPLAN is available, use it to determine isocenter
    if plan_path is not None:
        plans = fetch_plan_data(plan_path[0])
    
    new_spacing_sitk = (new_spacing[2], new_spacing[1], new_spacing[0])  # sitk uses (x,y,z)
    dose_ref = list(doses.keys())[0]
    dose = doses[dose_ref]
    _, num_fractions = list(plans.values())[0]
    ct_resampled = resample_image_to_spacing(
        ct_series,
        new_spacing=new_spacing_sitk,
        interpolator=sitk.sitkLinear,
    )

    if (crop_volume):
        ct_resampled = center_crop_axial(ct_resampled, max_size_cm=40.0)

    # 2. Resample all structures to the CT grid (use nearest-neighbor!)
    resampled_structures_torch = {}
    for name, struct_img in structures.items():
        struct_resampled = sitk.Resample(
            struct_img,
            ct_resampled,              # reference image
            sitk.Transform(),
            sitk.sitkNearestNeighbor,  # important for labels
            0,                         # default value
            struct_img.GetPixelID(),
        )

        struct_array = sitk.GetArrayFromImage(struct_resampled) > 0  # (z, y, x), bool
        resampled_structures_torch[name] = torch.from_numpy(struct_array)

    # 3. Resample dose to CT grid (linear interpolation)
    dose_resampled = sitk.Resample(
        dose,
        ct_resampled,              # reference image
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        dose.GetPixelID(),
    )
    dose_array = sitk.GetArrayFromImage(dose_resampled) / float(num_fractions)
    dose_tensor = torch.from_numpy(dose_array)

    # 4. Convert CT to torch
    ct_array = sitk.GetArrayFromImage(ct_resampled)  # (z, y, x)
    CT = torch.from_numpy(ct_array)

    # 5. Compute resolution and origin in your preferred order (z, x, y) or (z, y, x)
    # SimpleITK: spacing/origin are always (x, y, z)
    origin_xyz = ct_resampled.GetOrigin()

    # If your tensors are (z, y, x), you usually want spacing/origin in (z, y, x) too:
    origin = [origin_xyz[2], origin_xyz[1], origin_xyz[0]]

    # If you *really* wanted (z, x, y) for some reason, you can change the index order above.

    # 6. Build the Patient object
    patient = Patient(
        ct_tensor=CT,
        structures=resampled_structures_torch,
        dose=dose_tensor,
        resolution=new_spacing,
        number_of_fractions=num_fractions
    )
    

    # Create BeamSequence from raw control points
    beam_sequences = []
    for key, (seq, _) in plans.items():
        beam_sequence = BeamSequence.from_beams(seq).to(device).to(dtype)

        if dose_ref in plans.keys():
            if dose_ref != key:
                continue

        beam_sequence.iso_center = tuple(np.array(beam_sequence.iso_center) - np.array(origin))
        if use_delivery:
            # Convert to delivery positions and update treatment config
            beam_sequence = beam_sequence.to_delivery()
        beam_sequences.append(beam_sequence)

        

    return patient, beam_sequences

def resample_image_to_spacing(image, new_spacing, interpolator=sitk.sitkLinear):
    """
    Resample a SimpleITK image to a new spacing, keeping the same physical extent.
    
    Args:
        image (sitk.Image): Image to resample.
        new_spacing (tuple[float, float, float]): Target spacing (sx, sy, sz) in mm.
        interpolator: SimpleITK interpolator (default sitk.sitkLinear).
    Returns:
        sitk.Image: Resampled image with the requested spacing.
    """
    original_spacing = image.GetSpacing()   # (sx, sy, sz)
    original_size = image.GetSize()         # (nx, ny, nz)

    # Compute new size so that physical size stays (approximately) the same
    new_size = [
        int(round(osz * (osp / nsp)))
        for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)
    ]

    resampled = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0.0,                 # default pixel value
        image.GetPixelID(),
    )
    return resampled

def center_crop_axial(image, max_size_cm=40.0):
    """
    Crop the axial plane (x, y) of a SimpleITK image to a maximum physical size.
    Args:
        image: SimpleITK image to crop
        max_size_cm: Maximum physical size in cm for x and y dimensions
    Returns:
        Cropped SimpleITK image with updated origin
    """
    max_size_mm = max_size_cm * 10.0  # Convert cm to mm

    spacing = image.GetSpacing()  # (x, y, z)
    size = image.GetSize()  # (nx, ny, nz)
    origin = image.GetOrigin()  # (x, y, z)

    # Calculate physical size in mm for x and y
    physical_size_x = size[0] * spacing[0]
    physical_size_y = size[1] * spacing[1]

    # Determine crop size in voxels
    new_size_x = min(size[0], int(max_size_mm / spacing[0]))
    new_size_y = min(size[1], int(max_size_mm / spacing[1]))
    new_size_z = size[2]  # Keep z unchanged

    # If no cropping needed, return original image
    if new_size_x == size[0] and new_size_y == size[1]:
        return image

    # Calculate crop start indices (center crop)
    start_x = (size[0] - new_size_x) // 2
    start_y = (size[1] - new_size_y) // 2
    start_z = 0

    # Update origin to account for cropping
    new_origin = (
        origin[0] + start_x * spacing[0],
        origin[1] + start_y * spacing[1],
        origin[2]
    )

    # Extract region of interest
    cropped = sitk.RegionOfInterest(
        image,
        size=[new_size_x, new_size_y, new_size_z],
        index=[start_x, start_y, start_z]
    )

    # Update origin
    cropped.SetOrigin(new_origin)

    return cropped

def load_asc_measurements(path: str,
                          coord_map: Tuple[str, str, str] = ("X", "Y", "Z")):
    """
    Load a BDS-style .asc file and split it into measurements.

    Args:
        path (str): Path to the .asc measurement file.
        coord_map (Tuple[str, str, str]): Mapping from engine (x, y, z) to ASC
            axes; must be a permutation of ("X", "Y", "Z"). For example
            coord_map=("X", "Z", "Y") maps engine_x=ASC.X, engine_y=ASC.Z,
            engine_z=ASC.Y.
    Raises:
        ValueError: If coord_map is not a permutation of ("X", "Y", "Z").
    Returns:
        list[dict]: One dict per measurement, each with:
            - 'measurement_number': int or None
            - 'header_dict': parsed % / : lines, e.g. {'DAT': '09-07-2015', ...}
            - 'header_lines': raw header lines
            - 'data_raw': np.ndarray of shape (N, 4) [X_file, Y_file, Z_file, Dose]
            - 'coords_asc': np.ndarray of shape (N, 3) [X_file, Y_file, Z_file]
            - 'coords_engine': np.ndarray of shape (N, 3) [x_eng, y_eng, z_eng]
            - 'dose': np.ndarray of shape (N,)
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    # validate coord_map
    valid_axes = {"X", "Y", "Z"}
    if set(coord_map) != valid_axes:
        raise ValueError(
            f"coord_map must be a permutation of ('X','Y','Z'), got {coord_map}"
        )

    measurements: List[Dict[str, Any]] = []

    current_number = None
    current_data_lines: List[str] = []
    current_header_lines: List[str] = []
    current_header_dict: Dict[str, str] = {}

    def finalize_block():
        """Finalize current measurement block into measurements list."""
        if current_number is None:
            return

        if current_data_lines:
            data = np.loadtxt(current_data_lines, usecols=(1, 2, 3, 4))
            if data.ndim == 1:  # single row special case
                data = data[None, :]
        else:
            data = np.empty((0, 4), dtype=float)

        coords_asc = data[:, :3]          # [X_file, Y_file, Z_file]
        dose = data[:, 3]

        # map ASC -> engine coords
        name_to_idx = {"X": 0, "Y": 1, "Z": 2}
        idxs = [name_to_idx[name] for name in coord_map]
        coords_engine = coords_asc[:, idxs]

        measurements.append(
            {
                "measurement_number": current_number,
                "header_dict": current_header_dict.copy(),
                "header_lines": current_header_lines.copy(),
                "data_raw": data,
                "coords_asc": coords_asc,
                "coords_engine": coords_engine,
                "dose": dose,
            }
        )

    for line in lines:
        if "Measurement number" in line:
            # close previous measurement
            finalize_block()

            # extract number from line
            num = None
            for token in line.split():
                if token.isdigit():
                    num = int(token)

            current_number = num
            current_data_lines = []
            current_header_lines = [line]
            current_header_dict = {}
            if num is not None:
                current_header_dict["MeasurementNumber"] = str(num)

        else:
            if current_number is None:
                # global header: ignore
                continue

            stripped = line.lstrip()

            if stripped.startswith("="):
                # data row
                current_data_lines.append(line)
            else:
                # header/meta row
                current_header_lines.append(line)

                # strip inline comments after '#'
                stripped_comment = stripped.split("#", 1)[0].rstrip()
                if not stripped_comment:
                    continue

                if stripped_comment[0] in ("%", ":"):
                    body = stripped_comment[1:].strip()
                    if not body:
                        continue
                    parts = body.split(None, 1)
                    key = parts[0]
                    value = parts[1].strip() if len(parts) > 1 else ""
                    current_header_dict[key] = [val.strip() for val in value.split("\t")]

    # last measurement
    finalize_block()

    return measurements
