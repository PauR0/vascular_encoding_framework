"""
Input/output module for vef case directories.
"""

import os

import vascular_encoding_framework as vef
from vascular_encoding_framework import messages as msg


def in_case(case_dir, subdir, name):
    """
    Modify the provided path according to fit to case directory structure.


    Arguments:
    -----------

        case_dir : str
            The case directory.

        subdir : str or list[str]
            the actual subdirectory the path must end. It can be either a string
            or a list of strings as in ['foo', 'bar'] -> case_dir/foo/bar/

        name : str
            The filename.

    Returns:
    ----------
        :  str
            The appropriate path/to/file fulfilling the case directory criteria.
    """

    if isinstance(subdir, list):
        return os.path.join(case_dir, *subdir, name)

    return os.path.join(case_dir, subdir, name)
#

def load_vascular_mesh(path, suffix="", ext="vtk", abs_path=False):
    """
    Load a vascular mesh with all the available data at a given case directory.


    Parameters
    ----------
        path : string
            The path to the wall mesh.

        suffix : string
            A string indicating a suffix in the mesh name. E.g. suffix="_input"
            means mesh_input.vtk

        ext : str opt,
            Default vtk. File extension compatible with pyvista.read.

        abs_path : bool, optional
            Default False. Whether the path argument if the path-name to the file containing
            the vascular mesh or to a case directory.
    """


    if abs_path:
        mesh_fname   = path
        bounds_fname = None
    else:
        mesh_fname   = in_case(path, "Meshes", f"mesh{suffix}.{ext}")
        bounds_fname = in_case(path, "Meshes", f"boundaries{suffix}.json")

    try:
        vmesh = vef.VascularMesh.read(filename=mesh_fname, boundaries_fname=bounds_fname)
        return vmesh

    except FileNotFoundError:
        msg.error_message(f"Wrong path given. Cannot find {mesh_fname}.")
        return None
#

