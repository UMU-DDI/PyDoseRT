import sys
sys.path.append('../')
sys.path.append('../../')
import pydicom
from IPython.display import clear_output
import time
import math

from pydicom.data import get_testdata_file
from pydose_rt.data import MachineConfig, Patient, loaders
# from pydose_rt.data import MachineConfig
from pydose_rt.objectives.metrics import result_validation, validate_unit_dose
import numpy as np
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, rotate
from pydose_rt import DoseEngine
from pydose_rt.utils.utils import export_plan
import SimpleITK as sitk
from pydose_rt.utils.plotting import print_results, make_animation
import torch
import csv
import ast
from io import StringIO

# Set paths
ct_folder = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21/"
rtplan_path = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21_plans/1ARC/RP1.2.752.243.1.1.20251031145134399.7000.37887.dcm"
rtdose_path = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP_umea/0e54d72a21_plans/1ARC/RD1.2.752.243.1.1.20251031145134399.8000.21005.dcm"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kernel_size = 55

patient, treatment = loaders.load_dicom(
            ct_folder=ct_folder, 
            dose_path=rtdose_path, 
            plan_path=rtplan_path, 
            struct_names=["CTV", "PTVT_42.7", "FemoralHead_L", "FemoralHead_R", "Bladder", "External"],
            treatment_preset="src/pydose_rt/data/optimization_presets/umea.json",
            dtype=torch.float16,
            device=device
            )

treatment.kernel_size = 75
treatment.device = device
treatment.dtype = torch.float16

machine_config = MachineConfig(preset="src/pydose_rt/data/machine_presets/umea.json", resolution=patient.voxel_spacing_mm, ct_array_shape=patient.ct_array.shape)


mu_path = '/home/bolo/Documents/PyDose/out/mu_values-3000.npy'
mlc_path = '/home/bolo/Documents/PyDose/out/mlc_positions-3000.npy'

with open(mu_path, "r") as f:
    mus = np.array(ast.literal_eval(f.read()))

with open(mlc_path, "r") as f:
    mlcs = np.array(ast.literal_eval(f.read()))


treatment.plan_mus = 10 * mus
treatment.plan_mlcs = mlcs

export_plan(treatment, rtplan_path, "out/plan.dcm")