"""
Public API for the HeroDoseCalc package.
"""
from .hardware import DEVICE, get_device, MemoryManager
from .data import (
    MLCConfig,
    ControlPoint,
    MachineConfig,
    Phantom,
    DicomPhantom,
    ImportRTPlan,
    RTDoseExporter,
    HAS_PYDICOM,
)
from .nyholm import NyholmBeamModel
from .fluence import FluenceGenerator
from .engine import DoseEngine, DoseCalibrator
from .visualization import VisualizeDose
# Commissioning (preferred API)
from .commissioning_parser import MeasurementParser
from .commissioning_toolkit import CommissioningToolkit
from .commissioning_plotter import CommissioningPlotter
from .commissioning_types import MeasuredProfile, OutputFactorMeasurement

try:
    # Commissioning (compatibility layer)
    from .commissioning import (  # type: ignore
        RFA300Parser,
        CommissioningTool,
        CommissioningReport,
        CommissioningPipeline,
        OutputFactorParser,
    )
except ModuleNotFoundError:
    RFA300Parser = None  # type: ignore
    CommissioningTool = None  # type: ignore
    CommissioningReport = None  # type: ignore
    CommissioningPipeline = None  # type: ignore
    OutputFactorParser = None  # type: ignore

__all__ = [
    "DEVICE",
    "get_device",
    "MemoryManager",
    "MLCConfig",
    "ControlPoint",
    "MachineConfig",
    "Phantom",
    "DicomPhantom",
    "ImportRTPlan",
    "RTDoseExporter",
    "HAS_PYDICOM",
    "NyholmBeamModel",
    "FluenceGenerator",
    "DoseEngine",
    "DoseCalibrator",
    "VisualizeDose",
    "MeasurementParser",
    "CommissioningToolkit",
    "CommissioningPlotter",
    "MeasuredProfile",
    "OutputFactorMeasurement",
]

if RFA300Parser is not None:
    __all__ += [
        "RFA300Parser",
        "CommissioningTool",
        "CommissioningReport",
        "CommissioningPipeline",
        "OutputFactorParser",
    ]
