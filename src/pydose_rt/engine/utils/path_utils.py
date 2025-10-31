# -*- coding: utf-8 -*-
"""
Created on Tue Dec 04
Copyright (c) 2023, Vu Hoang Minh. All rights reserved.
@author:  Vu Hoang Minh
@email:   minh.vu@umu.se
@license: BSD 3-clause.
"""

import os
import ntpath
from engine.utils.print_utils import print_separator
import glob
import argparse


def get_project_dir(path, project_name):
    paths = path.split(project_name)
    return paths[0] + project_name


def split_dos_path_into_components(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)

        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)

            break

    folders.reverse()
    return folders


def get_parent_dir(path):
    return os.path.abspath(os.path.join(path, os.pardir))


def get_filename(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_filename_without_extension(path):
    filename = get_filename(path)
    return os.path.splitext(filename)[0]


def make_dir(dir):
    if not os.path.exists(dir):
        print_separator()
        print("making dir", dir)
        os.makedirs(dir)


def get_hyperopt_path(project_name, database_path="database", folder="optimization"):
    save_path = database_path + f"/{folder}"
    make_dir(save_path)
    file_name = project_name + ".hyperopt"
    return os.path.join(save_path, file_name)


def get_folder(args):
    """Generates a short experiment name from arguments with 2-digit padding."""
    if isinstance(args, argparse.Namespace):
        params = {
            "ds": args.dataset,
            "b": args.number_of_cps,
            "bs": args.batch_size,
            "f": args.initial_filters,
            "v": args.vae_n_filters,
            "d": args.depth,
            "de": args.dose_engine,
            "lw": args.leaf_width,
            "ptv": args.weight_PTV,
            "df": args.downsampling_factor,
            "aug": args.is_augmentation,
            "constraint_mode": args.constraint_mode,
        }
    else:
        params = args

    padded_params = {}
    for key, value in params.items():
        if key == "constraint_mode":
            continue
        if key in ["ds", "de"]:
            padded_params[key] = f"{value}"
        elif key == "b":
            padded_params[key] = f"{value:03d}"  # Pad 'b' value to 3 digits
        elif key == "df":
            padded_params[key] = "".join(map(str, value))
        else:
            padded_params[key] = f"{value:02d}"  # Pad other values to 2 digits

    base_name = "_".join([f"{key}{value}" for key, value in padded_params.items()])
    constraint_mode_str = f'_{params["constraint_mode"]}'  # cm for constraint_mode
    return base_name + constraint_mode_str


def contains_file_with_string(directory, search_string):
    # Ensure directory exists
    if not os.path.isdir(directory):
        return False

    # Iterate over all items in the directory
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        # Check if it's a file (not a directory) and contains the string
        if os.path.isfile(full_path) and search_string in item:
            return True

    return False


def main():
    output_dir = "/mnt/sda2/3DUnetCNN_BRATS/projects/pros/database/prediction/pros_2018_is-256-256-128_crop-0_bias-0_denoise-0_norm-11_hist-0_ps-128-128-128_segnet3d_crf-0_loss-dice_xent_aug-1_model/validation_case_956"


if __name__ == "__main__":
    main()
