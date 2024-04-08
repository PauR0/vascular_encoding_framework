"""
Input/output module for vef case directories.
"""

import os

import pyvista as pv

import vascular_encoding_framework as vef
from vascular_encoding_framework.utils._io import is_writable
from vascular_encoding_framework import messages as msg


def in_case(case_dir, subdir, name):
    """
    Modify the provided path according to fit to case directory structure.


    Arguments
    -----------

        case_dir : str
            The case directory.

        subdir : str or list[str]
            the actual subdirectory the path must end. It can be either a string
            or a list of strings as in ['foo', 'bar'] -> case_dir/foo/bar/

        name : str
            The filename.

    Returns
    ----------
        :  str
            The appropriate path/to/file fulfilling the case directory criteria.
    """

    if isinstance(subdir, list):
        return os.path.join(case_dir, *subdir, name)

    return os.path.join(case_dir, subdir, name)
#

def make_subdirs(case_dir, subdir):
    """
    Make the subdir at case dir


    Arguments
    ---------

        case_dir : str
            The case directory

        subdir : str, list[str]
            The subdirectory to be created
    """


    sd = os.path.join(case_dir, subdir)
    if isinstance(subdir, list):
        sd = os.path.join(case_dir, *subdir)

    os.makedirs(sd, exist_ok=True)
#

def load_vascular_mesh(path, suffix="", ext="vtk", abs_path=False):
    """
    Load a vascular mesh with all the available data at a given case directory.


    Arguments
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

def save_vascular_mesh(vmesh, path, suffix="", binary=True, ext="vtk", abs_path=False, overwrite=False):
    """
    Save a vascular mesh with all the available data at a given path. By default
    this function assumes path is a case directory and uses default name convention.


    Arguments
    ----------

        path : string
            The path to the wall mesh.

        suffix : string
            A string indicating a suffix in the mesh name. E.g. suffix="_input"
            means mesh_input.vtk

        binary : bool, opt.
            Default True. Wheteher to save vtk files in binary format.

        ext : str opt,
            Default vtk. File extension compatible with pyvista.read.

        abs_path : bool, optional
            Default False. Whether the path argument if the path-name to the file containing
            the vascular mesh or to a case directory.

        overwrite : bool, opt
            Default False. Whether to overwirte existing files.
    """

    if not abs_path:
        make_subdirs(path, 'Meshes')
        mesh_fname   = os.path.join(path, 'Meshes', f'mesh{suffix}.{ext}')
        bounds_fname = os.path.join(path, 'Meshes', f'boundaries{suffix}')

    else:
        dirname, nameext = os.path.split(path)
        name, _ = os.path.splitext(nameext)
        mesh_fname   = path
        bounds_fname = os.path.join(dirname, f"{name}_boundaries")

    if is_writable(mesh_fname, overwrite=overwrite) and is_writable(bounds_fname, overwrite=overwrite):
        vmesh.save(fname=mesh_fname, binary=binary, boundaries_fname=bounds_fname)

    else:
        msg.warning_message(f"Overwritting is set to false, and {mesh_fname} or {bounds_fname} already exists. Nothig will be written.")
#

def load_centerline_domain(case_dir):
    """
    Load the centerline domain under the case directory convention.

    Using UNIX path format, the centerline domain is expected to be at subdir Centerline, with name,
    domain.vtk, i.e.:

    case_dir/Centerline/domain.vtk

    Arguments
    ---------
        case_dir : str
            The path to the case directory.

    Returns
    -------
        cl_domain : pv.UnstructuredGrid
            The loaded lumen discretization.
    """

    fname = in_case(case_dir, 'Centerline', 'domain.vtk')
    cl_domain = None
    if os.path.exists(fname):
        cl_domain = pv.read(fname)

    return cl_domain
#

def save_centerline_domain(case_dir, cl_domain, binary=True, overwrite=False):
    """
    Save the centerline domain under the case directory convention.

    Using UNIX path format, the centerline domain is expected to be at subdir Centerline, with name,
    domain.vtk, i.e.:

        case_dir/Centerline/domain.vtk

    Arguments
    ---------

        case_dir : str
            The path to the case directory.

        cl_domain : pv.UnstructuredGrid
            The computed centerline domain as a pyvista UnstructuredGrid.

        binary : bool, opt.
            Default True. Wheteher to save vtk files in binary format.

        overwrite : bool, opt
            Default False. Whether to overwrite existing files.
    """

    make_subdirs(case_dir, 'Centerline')
    fname = in_case(case_dir, 'Centerline', 'domain.vtk')
    message = f"{fname} exists and overwritting is set to False."
    if is_writable(fname, overwrite=overwrite, message=message):
        cl_domain.save(fname, binary=binary)
#

def load_centerline_path(case_dir):
    """
    Load the centerline path under the case directory convention.

    Using UNIX path format, the centerline path is expected to be at subdir Centerline, with name,
    path.vtm, i.e.:

    case_dir/Centerline/path.vtm

    Due to the MultiBlock save format of vtk, a directory called path is expected to be
    found at Centerline subdir containing the path data.

    Arguments
    ---------
        case_dir : str
            The path to the case directory.

    Returns
    -------
        cl_paths : pv.MultiBlock
            The loaded paths optimizing the distance to the wall.
    """

    fname = in_case(case_dir, 'Centerline', 'path.vtm')

    cl_paths = None
    if os.path.exists(fname):
        cl_paths = pv.read(fname)

    return cl_paths
#

def save_centerline_path(case_dir, cl_path, binary=True, overwrite=False):
    """
    Save the centerline path under the case directory convention.

    Using UNIX path format, the centerline path is expected to be at subdir Centerline, with name,
    path.vtm, i.e.:

        case_dir/Centerline/path.vtm

    Due to the MultiBlock save format of vtk, a directory called path is also created at Centerline
    subdir containing the path data.

    Arguments
    ---------

        case_dir : str
            The path to the case directory.

        cl_path : pv.MultiBlock
            The computed centerline paths as a pyvista MultiBlock.

        binary : bool, opt.
            Default True. Wheteher to save vtk files in binary format.

        overwrite : bool, opt
            Default False. Whether to overwrite existing files.
    """

    make_subdirs(case_dir, 'Centerline')
    fname = in_case(case_dir, 'Centerline', 'path.vtm')
    message = f"{fname} exists and overwritting is set to False."
    if is_writable(fname, overwrite=overwrite, message=message):
        cl_path.save(fname, binary=binary)
#

def load_centerline(case_dir):
    """
    Load the centerline network under the case directory convention.

    Using UNIX path format, the centerline path is expected to be at subdir Centerline with name
    centerline.vtm, i.e.:

        case_dir/Centerline/centerline.vtm

    Due to the MultiBlock save format of vtk, a directory called centerline is also expecterd to be
    at Centerline subdir containing the centerline data.

    Arguments
    ---------

        case_dir : str
            The path to the case directory.

        cl_net : vef.CenterlineNetwork
            The computed centerline paths as a pyvista MultiBlock.

        binary : bool, opt.
            Default True. Wheteher to save vtk files in binary format.

        overwrite : bool, opt
            Default False. Whether to overwrite existing files.
    """

    fname = in_case(case_dir, 'Centerline', 'centerline.vtm')

    cl_mb = None
    if os.path.exists(fname):
        cl_mb = pv.read(fname)

    return cl_mb
#

def save_centerline(case_dir, cl_net, binary=True, overwrite=False):
    """
    Save the centerline network under the case directory convention.

    Using UNIX path format, the centerline path is expected to be at subdir Centerline, with name,
    centerline.vtm, i.e.:

        case_dir/Centerline/centerline.vtm

    Due to the MultiBlock save format of vtk, a directory called centerline is also created at
    Centerline subdir containing the centerline data.

    Arguments
    ---------

        case_dir : str
            The path to the case directory.

        cl_net : vef.CenterlineNetwork
            The computed centerline paths as a pyvista MultiBlock.

        binary : bool, opt.
            Default True. Wheteher to save vtk files in binary format.

        overwrite : bool, opt
            Default False. Whether to overwrite existing files.
    """

    make_subdirs(case_dir, 'Centerline')
    fname = in_case(case_dir, 'Centerline', 'centerline.vtm')
    message = f"{fname} exists and overwritting is set to False."
    cl_mb = cl_net.to_multiblock()
    if is_writable(fname=fname, overwrite=overwrite, message=message):
        cl_mb.save(fname, binary)
#