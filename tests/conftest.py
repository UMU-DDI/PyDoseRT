import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
import torch
from pydose_rt.data import MachineConfig, Beam, BeamSequence
import os
from dotenv import load_dotenv
load_dotenv()  # will look for .env in project root
 
def pytest_addoption(parser):
    parser.addoption(
        "--rtp-data-dir",
        action="store",
        default=None,
        help="Path to the RTP dataset root (optional)."
    )
    parser.addoption(
        "--rtp-dose-path",
        action="store",
        default=None,
        help="Path to the RTP Dose file (optional)."
    )
    parser.addoption(
        "--rtp-struct-path",
        action="store",
        default=None,
        help="Path to the RTP Struct file (optional)."
    )
    parser.addoption(
        "--rtp-plan-path",
        action="store",
        default=None,
        help="Path to the RTP plan file (optional)."
    )
 
@pytest.fixture(scope="session")
def rtp_struct_path(pytestconfig):
    opt = pytestconfig.getoption("--rtp-struct-path") or os.getenv("RTP_STRUCT_PATH")
    if not opt:
        pytest.skip("No RTP struct path provided (--rtp-struct-path or RTP_STRUCT_PATH). Skipping integration test.")
    p = Path(opt)
    if not p.exists():
        pytest.fail(f"Provided RTP struct path does not exist: {p}")
    return p

@pytest.fixture(scope="session")
def rtp_dose_path(pytestconfig):
    opt = pytestconfig.getoption("--rtp-dose-path") or os.getenv("RTP_DOSE_PATH")
    if not opt:
        pytest.skip("No RTP dose path provided (--rtp-dose-path or RTP_DOSE_PATH). Skipping integration test.")
    p = Path(opt)
    if not p.exists():
        pytest.fail(f"Provided RTP dataset path does not exist: {p}")
    return p
 
@pytest.fixture(scope="session")
def rtp_plan_path(pytestconfig):
    opt = pytestconfig.getoption("--rtp-plan-path") or os.getenv("RTP_PLAN_PATH")
    if not opt:
        pytest.skip("No RTP dataset provided (--rtp-plan-path or RTP_PLAN_PATH). Skipping integration test.")
    p = Path(opt)
    if not p.exists():
        pytest.fail(f"Provided RTP dataset path does not exist: {p}")
    return p
 
@pytest.fixture(scope="session")
def rtp_data_dir(pytestconfig):
    opt = pytestconfig.getoption("--rtp-data-dir") or os.getenv("RTP_DATA_DIR")
    if not opt:
        pytest.skip("No RTP dataset provided (--rtp-data-dir or RTP_DATA_DIR). Skipping integration test.")
    p = Path(opt)
    if not p.exists():
        pytest.fail(f"Provided RTP dataset path does not exist: {p}")
    return p
 
@pytest.fixture
def default_machine_config():
    """Fixture for the default MachineConfig"""
    return MachineConfig(
        preset="src/pydose_rt/data/machine_presets/test.json",

        head_scatter_amplitude=None,
        head_scatter_sigma=None,
        penumbra_fwhm=None,
        profile_corrections=None,
    )
 
@pytest.fixture
def default_device():
    """Fixture for the default device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
@pytest.fixture
def default_dtype():
    """Fixture for the default dtype"""
    return torch.float32
 
@pytest.fixture
def default_ct_array_shape():
    """Fixture for default CT array shape"""
    return (64, 64, 64)
 
@pytest.fixture
def default_resolution():
    """Fixture for default resolution"""
    return (3.0, 3.0, 3.0)
 
@pytest.fixture
def default_field_size():
    """Fixture for default field size"""
    return (400, 400)
 
@pytest.fixture
def default_iso_center():
    """Fixture for default isocenter"""
    return (0.0, 0.0, 0.0)
 
@pytest.fixture
def default_sid():
    """Fixture for default SID"""
    return 1000.0
 
@pytest.fixture
def default_kernel_size():
    """Fixture for default kernel size"""
    return 15

@pytest.fixture
def default_number_of_beams():
    """Fixture for default number of control points"""
    return 1
 
@pytest.fixture
def default_gantry_angles(default_number_of_beams, default_device, default_dtype):
    """Fixture for default gantry angles"""
    return torch.zeros(default_number_of_beams, device=default_device, dtype=default_dtype)

@pytest.fixture
def default_collimator_angles(default_number_of_beams, default_device, default_dtype):
    """Fixture for default gantry angles"""
    return torch.zeros(default_number_of_beams, device=default_device, dtype=default_dtype)
 
@pytest.fixture
def default_beam(default_machine_config, default_field_size, default_iso_center, default_device, default_dtype):
    """Fixture for a default single Beam"""
    return Beam.create(
        gantry_angle_deg=0.0,
        number_of_leaf_pairs=default_machine_config.number_of_leaf_pairs,
        field_size_mm=default_field_size,
        iso_center=default_iso_center,
        device=default_device,
        dtype=default_dtype,
        requires_grad=True
    )
 
@pytest.fixture
def default_beam_sequence(default_machine_config, default_number_of_beams, default_field_size, default_iso_center, default_sid, default_device, default_dtype):
    """Fixture for a default BeamSequence"""
    return BeamSequence.create(
        gantry_angles_deg=[0.0] * default_number_of_beams,
        number_of_leaf_pairs=default_machine_config.number_of_leaf_pairs,
        field_size=default_field_size,
        iso_center=default_iso_center,
        sid=default_sid,
        device=default_device,
        dtype=default_dtype,
        requires_grad=True
    )