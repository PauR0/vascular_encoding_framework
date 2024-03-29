#!/usr/bin/env python3

import os
import sys
import argparse

from vascular_encoding_framework import messages as msg

from case_io import load_vascular_mesh, save_vascular_mesh


def handle_case_and_mesh_name(case, mesh, ow=False):
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
    if case is None:
        case_dir = 'vef_case'
    else:
        case_dir = case

    if mesh is not None:
        base_dir = os.path.dirname(mesh)
        if case is None:
            case_dir = os.path.join(base_dir, case_dir)

    if os.path.exists(case_dir) and not ow:
        msg.warning_message(f"The case: {case_dir} already exists and overwritting is set to False. Nothing will be created.")
        return None, None

    return mesh, case
#

def make_case(case_dir, mesh_fname=None, hierarchy=None, show_bounds=False, overwrite=False):
    """
    Function to make a vef case directory at path provided in case_dir argument.
    Additionally, the filename of a mesh can be passed, and it is copied and saved
    in Meshes directory inside the case. If the mesh_fname is passed, the module also
    attempts to compute the boundaries and save them at the Meshes directory.
    """

    os.makedirs(case_dir, exist_ok=True)

    meshes_dir = os.path.join(case_dir, 'Meshes')
    if mesh_fname is not None or hierarchy is not None:
        os.makedirs(meshes_dir, exist_ok=True)

    if mesh_fname is not None:
        vmesh = load_vascular_mesh(path=mesh_fname, abs_path=True)
        if hierarchy is not None:
            vmesh.set_boundary_data(hierarchy)
        if show_bounds:
            vmesh.plot_boundaries_ids()

        save_vascular_mesh(vmesh, case_dir, suffix="_input", binary=True, ext='vtk', overwrite=overwrite)
#






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Script to generate a directory according to \
    the Vascular Encoding Framework case requirements. The script works as follows. With no arguments
    a directory called, vef_case is created. If a mesh is pased, then the vef_case is created in the same
    directory where the mesh is located and the mesh is set as the input mesh at Meshes directory. If a case
    path is provided, it is used (in combination or not with the mesh). All this cases check existence of the
    files/directories before creating them, and if overwritting is set to false, the creation is skipped printing a message to terminal.""")

    parser.add_argument('--plot-boundaries',
                        dest="plot_bounds",
                        action='store_true',
                        help="""Plot the estimated boundaries with its ids. If no mesh
                        have been passed, this flag is ignored.""")

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
                        default=None,
                        help="""The path/name with the case to be created. If none is provided,
                        a directory called vef_case.""")

    args = parser.parse_args()

    mesh_path, case_path = handle_case_and_mesh_name(case=args.case, mesh=args.mesh, ow=args.w)
    if case_path is None:
        sys.exit(0)

    make_case(case_path, mesh_path, overwrite=args.w)
#
