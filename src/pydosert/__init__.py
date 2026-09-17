from pydosert.engine.photon_base_engine import PhotonBaseEngine
from pydosert.engine.dose_engine import DoseEngine
from pydosert.engine.ion_dose_engine import IonDoseEngine
from pydosert.physics.dose_mask import patient_dose_mask
from pydosert.geometry.conventions import CONVENTIONS, print_conventions
from pydosert.data import (
    MachineConfig,
    IonMachineConfig,
    OptimizationConfig,
    Phantom,
    Patient,
    Beam,
    BeamSequence,
    IonBeamletBatch,
)
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pydosert")
except PackageNotFoundError:
    __version__ = "0.0.0"
    
__all__ = ['DoseEngine', 
           'PhotonBaseEngine',
           'IonDoseEngine',
           'MachineConfig', 
           'IonMachineConfig',
           'OptimizationConfig', 
           'Phantom', 
           'Patient', 
           'Beam', 
           'BeamSequence',
           'IonBeamletBatch',
           'patient_dose_mask',
           'CONVENTIONS',
           'print_conventions']
