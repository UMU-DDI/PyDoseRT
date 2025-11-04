import random
import copy
import numpy as np
import torch
import os
import time

def load_files(file_path):
    ct = None       
    while ct is None:
        try:
            ct = (np.load(os.path.join(file_path, "CT.npy"), allow_pickle=True))
        except Exception:
            time.sleep(random.random())

    structures = dict()
    while len(structures) == 0:
        try:
            structs = np.load(os.path.join(file_path, "StructureSet.npy"), allow_pickle=True).astype(np.float32)
            structures["PTV"] = structs[0, ...]
            structures["PenileBulb"] = np.clip(structs[1, ...] - (structures["PTV"]), 0, 1)
            structures["FemoralHead_L"] = np.clip(structs[2, ...] - (structures["PTV"] + structures["PenileBulb"]), 0, 1)
            structures["FemoralHead_R"] = np.clip(structs[3, ...] - (structures["PTV"] + structures["PenileBulb"] + structures["FemoralHead_L"]), 0, 1)
            structures["Bladder"] = np.clip(structs[4, ...] - (structures["PTV"] + structures["PenileBulb"] + structures["FemoralHead_L"] + structures["FemoralHead_R"]), 0, 1)
            structures["Rectum"] = np.clip(structs[5, ...] - (structures["PTV"] + structures["PenileBulb"] + structures["FemoralHead_L"] + structures["FemoralHead_R"] + structures["Bladder"]), 0, 1)
            structures["Background"] = np.clip(structs[6, ...] - (structures["PTV"] + structures["PenileBulb"] + structures["FemoralHead_L"] + structures["FemoralHead_R"] + structures["Bladder"] + structures["Rectum"]), 0, 1)
        except Exception:
            time.sleep(random.random())

    # Dose (optional)
    dose = None
    if (os.path.exists(os.path.join(file_path, "Dose.npy"))):
        while dose is None:
            try:
                dose = np.load(os.path.join(file_path, "Dose.npy"), allow_pickle=True)
            except Exception:
                time.sleep(random.random())
    else:
        dose = np.zeros_like(ct)

    return ct, structures, dose
