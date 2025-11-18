import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import pytest
from pydose_rt.data import MachineConfig, TreatmentConfig
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
        "--rtp-plan-path",
        action="store",
        default=None,
        help="Path to the RTP plan file (optional)."
    )

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
    )

@pytest.fixture
def default_treatment_config():
    """Fixture for the default TreatmentConfig"""
    return TreatmentConfig(
        preset="src/pydose_rt/data/treatment_presets/test.json",
    )