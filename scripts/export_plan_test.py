import sys
sys.path.append('../')
sys.path.append('../../')
import pydicom
from IPython.display import clear_output
import time
import math

from pydicom.data import get_testdata_file
from pydose_rt.data import MachineConfig, PatientData, DoseConfig
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

config = DoseConfig.from_dicom(
    ct_folder=ct_folder, 
    dose_path=rtdose_path,
    plan_path=rtplan_path,
    struct_names=["External", "CTV", "FemoralHead_R", "FemoralHead_L", "Bladder", "PTVT_42.7"],
    machine_preset="umea",
    downsampling_factor=(1, 2, 2),
    dtype=torch.float32,
    device=device
)

mu_path = '/home/bolo/Documents/PyDose/out/mu_values-3000.npy'
mlc_path = '/home/bolo/Documents/PyDose/out/mlc_positions-3000.npy'

# with open(mu_path, "r") as f:
#     mus = np.array(ast.literal_eval(f.read()))

# with open(mlc_path, "r") as f:
#     mlcs = np.array(ast.literal_eval(f.read()))


# config.patient.plan_mus = 10 * mus
# config.patient.plan_mlcs = mlcs
export_plan(config, rtplan_path, "out/plan.dcm")