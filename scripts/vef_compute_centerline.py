#!/usr/bin/env python3

import os
import sys
import argparse

import vascular_encoding_framework as vef
from vascular_encoding_framework import messages as msg

from case_io import load_vascular_mesh, save_centerline
from config.readers import read_centerline_config
from config.writers import write_centerline_config


def compute_centerline(case_dir, params=None, debug=False, overwrite=False):
    """
    Function to make a vef case directory at path provided in case_dir argument.
    Additionally, the filename of a mesh can be passed, and it is copied and saved
    in Meshes directory inside the case. If the mesh_fname is passed, the module also
    attempts to compute the boundaries and save them at the Meshes directory.
    """

    vmesh = load_vascular_mesh(case_dir, suffix='_input')

    print(vmesh.boundaries)

    if params is None:
        params = read_centerline_config(case_dir)

    vef.centerline.extract_centerline(vmesh=vmesh,
                                      params=params,
                                      params_domain=params['params_domain'],
                                      params_path=params['params_path'])

    save_centerline(case_dir)

#






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Script to compute the centerline of a vascular
    structure using the Vascular Encoding Framework conventions. The computation of the centerline
    requires a vascular mesh with _input suffix to exist in the case. Parameters can be tuned using
    a centerline.json file stored in the case directory, otherwise default values are used.""")

    parser.add_argument('-w',
                        action='store_true',
                        help="""Force overwriting if it already exists.""")

    parser.add_argument('case',
                        action='store',
                        default=None,
                        help="""The path/name with the case. If none is provided,
                        a directory called vef_case.""")

    args = parser.parse_args()

    compute_centerline(case_dir=args.case, overwrite=args.w)
    #
