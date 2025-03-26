import argparse
import os
import sys

from vascular_encoding_framework import messages as msg

from .case_io import load_vascular_mesh, save_vascular_mesh
from .config.readers import read_centerline_config, read_encoding_config
from .config.writers import write_centerline_config, write_encoding_config


def handle_case_and_mesh_name(case, mesh, ow=False):
    """
    Auxiliary function to handle paths as stated in script description.

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
        msg.warning_message(
            f'The case: {case_dir} already exists and overwriting is set to False. Nothing will be created.')
        return None, None

    return case_dir, mesh
#


def make_case(
        case_dir,
        mesh_fname=None,
        vmesh=None,
        show_boundaries=False,
        overwrite=False,
        cl_params=None,
        ec_params=None):
    """
    Function to make a vef case directory at path provided in case_dir argument.
    Additionally, the filename of a mesh can be passed, and it is copied and saved
    in Meshes directory inside the case. If the mesh_fname is passed, the module also
    attempts to compute the boundaries and save them at the Meshes directory.
    """

    case_dir, mesh_fname = handle_case_and_mesh_name(
        case_dir, mesh_fname, ow=overwrite)

    if cl_params is None:
        cl_params = read_centerline_config(case_dir)
    if ec_params is None:
        ec_params = read_encoding_config(case_dir)

    os.makedirs(case_dir, exist_ok=True)
    write_centerline_config(case_dir, cl_params)
    write_encoding_config(case_dir, ec_params)

    if vmesh is None and mesh_fname is not None:
        vmesh = load_vascular_mesh(path=mesh_fname, abs_path=True)
    elif vmesh is not None and mesh_fname is not None:
        msg.warning_message(
            f'Using vmesh provided to make the case. mesh_fname {mesh_fname} is being ignored.')

    if vmesh is not None:

        if show_boundaries:
            vmesh.plot_boundary_ids()

        meshes_dir = os.path.join(case_dir, 'Meshes')
        os.makedirs(meshes_dir, exist_ok=True)
        save_vascular_mesh(
            vmesh,
            case_dir,
            suffix='_input',
            binary=True,
            overwrite=overwrite)
    return case_dir
#
