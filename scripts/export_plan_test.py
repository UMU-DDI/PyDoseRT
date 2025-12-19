import sys
sys.path.append('../')
sys.path.append('../../')
import pydicom
from IPython.display import clear_output
import time
import math

from pydicom.data import get_testdata_file
from pydose_rt.data import MachineConfig, Patient, loaders
from pydose_rt.utils.utils import find_patient_paths
import numpy as np
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, rotate
from pydose_rt import DoseEngine
from pydose_rt.utils.utils import export_plan
import torch
import csv
import ast
from io import StringIO

# Set paths
base_path = '/home/bolo/Documents/PyDoseRT/test_data/GoldAtlasPlans/10X/P01/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ct_folder, rtplan_path, rtdose_path, rtstruct_path = find_patient_paths(base_path)

patient, beam_sequences = loaders.load_dicom(
            ct_folder=ct_folder, 
            dose_path=rtdose_path, 
            plan_path=rtplan_path, 
            struct_path=rtstruct_path,
            struct_names=["CTV", "PTV", "PenileBulb", "Prostate", "FemoralHead_L", "FemoralHead_R", "Bladder", "Rectum", "SeminalVesicles", "External"],
            use_delivery=False
            )



mu_path = '/home/bolo/Documents/PyDoseRT/out/mu_values-150.npy'
mlc_path = '/home/bolo/Documents/PyDoseRT/out/mlc_positions-150.npy'

with open(mu_path, "r") as f:
    mus = np.array(ast.literal_eval(f.read()))

with open(mlc_path, "r") as f:
    leaf_positions = np.array(ast.literal_eval(f.read()))


beam_sequences[0].mus = mus
beam_sequences[0].leaf_positions = leaf_positions

export_plan(beam_sequences, rtplan_path[0], "out/plan.dcm")