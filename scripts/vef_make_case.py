#!/usr/bin/env python3

import os
import sys
import argparse

import vascular_encoding_framework as vef
from vascular_encoding_framework import messages as msg

from .case_io import load_vascular_mesh


def handle_mesh_and_case_name(mesh, case, ow=False):
    """
    Auxiliar function to handle paths as stated in script description.

    Arguments:
    -------------

        mesh, case : str
            The mesh and case paths.

        ow : bool
            Whether to overwrite existing files or stop.

    Returns:
    ---------
        mesh, case : str
            The updated strings.
    """

    base_dir = os.getcwd()
    if mesh is not None:
        base_dir = os.path.dirname(mesh)
        if case is None:
            case = 'vef_case'
        case = os.path.join(base_dir, case)

    if os.path.exists(case) and not ow:
        msg.warning_message(f"the case: {case} already exists and overwritting is set to False. Nothing will be created.")
        return None, None

    return mesh, case
#

def make_case(case_dir, mesh_fname=None, hierarchy=None):
    """
    Function to make a vef case directory at path provided in case_dir argument.
    Additionally, the filename of a mesh can be passed, and it is copied and saved
    in Meshes directory inside the case. If the mesh_fname is passed, the module also
    attempts to compute the boundaries and save them at the Meshes directory.
    """

    os.makedirs(case_dir, exist_ok=True)

    meshes_dir = os.path.join(case_dir, 'Meshes')
    if mesh_fname is not None or hierarchy is not None:
        os.makedirs(meshes_dir)

    if mesh_fname is not None:
        vmesh = load_vascular_mesh(path=mesh_fname, abs_path=True)
        if hierarchy is not None:
            vmesh.set_boundary_data(hierarchy)
#






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Script to generate a directory according to \
    the Vascular Encoding Framework case requirements. The directory path can be provided using
    the -c or --case flag. If no mesh file is provided the current directory is used.""")

    parser.add_argument('-w',
                        action='store_true',
                        help="""Force overwriting if it already exists.""")

    parser.add_argument('-m',
                        '-mesh',
                        dest='mesh',
                        type=str,
                        default=None,
                        help = """Path to an existing mesh.""")

    parser.add_argument('-c',
                        '--case',
                        dest='case',
                        action='store',
                        default='None',
                        help="""The path/name with the case to be created. If none is provided,
                        a directory called vef_case.""")

    args = parser.parse_args()

    mesh_path, case_path = handle_mesh_and_case_name(args.case, args.mesh, ow=args.w)
    if case_path is None:
        sys.exit(0)

    make_case(case_path, mesh_path)
#
