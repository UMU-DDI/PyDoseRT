from pydosert.engine.photon_base_engine import PhotonBaseEngine
from pydosert.engine.dose_engine import DoseEngine
from pydosert.data import MachineConfig, OptimizationConfig, Phantom, Patient, Beam, BeamSequence
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pydosert")
except PackageNotFoundError:
    __version__ = "0.0.0"
    
__all__ = ['DoseEngine', 
           'PhotonBaseEngine',
           'MachineConfig', 
           'OptimizationConfig', 
           'Phantom', 
           'Patient', 
           'Beam', 
           'BeamSequence']