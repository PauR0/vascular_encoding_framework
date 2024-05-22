"""
Input/output module for vef case directories.

TODO: Implement an alternative to prevent code duplication for loaders and savers.
A function-factory seems to be incompatible with docstring.
"""

import os
from copy import deepcopy

import pyvista as pv

import vascular_encoding_framework as vef
from vascular_encoding_framework.utils._io import is_writable
from vascular_encoding_framework import messages as msg


_case_convention = {
    "boundaries" : ["Meshes",     "boundaries.json"],
    "mesh"       : ["Meshes",     "mesh.vtk"],
    "cl_domain"  : ["Centerline", "domain.vtk"],
    "cl_path"    : ["Centerline", "path.vtm"],
    "centerline" : ["Centerline", "centerline.vtm"],
    "encoding"   : ["Encoding",   "encoding.vtm"],
}
#

def get_case_convention(obj, suffix="", case_dir=""):
    """
    Get the path relative to a case directory for the main objects of vef_scripts module.

    Arguments
    ---------

        obj : {"boundaries", "mesh", "domain", "cl_path", "centerline", "encoding"}
            The name of the object to get the path.

        suffix : str, optional
            Defaulting to empty string. An string to append to filename before the extension.

        case_dir : str, optional
            Defaulting to empty string. A path to a case directory to be pre-appended.

    Returns
    -------

        cfname : str
            The path, relative to top case directory level, of the object with the provided suffix.

    """

    cfname = deepcopy(_case_convention[obj])
    name, ext = cfname[-1].split('.')
    cfname[-1] = f"{name}{suffix}.{ext}"
    cfname = os.path.join(*cfname)
    return os.path.join(case_dir, cfname)
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

def load_vascular_mesh(path, suffix="", abs_path=False):
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
        mesh_fname   = get_case_convention("mesh",       suffix=suffix, case_dir=path)
        bounds_fname = get_case_convention("boundaries", suffix=suffix, case_dir=path)

    try:
        vmesh = vef.VascularMesh.read(filename=mesh_fname, boundaries_fname=bounds_fname)
        return vmesh

    except FileNotFoundError:
        msg.error_message(f"Wrong path given. Cannot find {mesh_fname}.")
        return None
#

def save_vascular_mesh(vmesh, path, suffix="", binary=True, abs_path=False, overwrite=False):
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
        mesh_fname   = get_case_convention("mesh",       suffix=suffix, case_dir=path)
        bounds_fname = get_case_convention("boundaries", suffix=suffix, case_dir=path)

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

def load_centerline_domain(case_dir, suffix=""):
    """
    Load the centerline domain under the case directory convention.

    Using UNIX path format, the centerline domain is expected to be at subdir Centerline, with name,
    domain.vtk, i.e.:

    case_dir/Centerline/domain.vtk

    Arguments
    ---------
        case_dir : str
            The path to the case directory.

        suffix : string
            A string indicating a suffix in the mesh name. E.g. suffix="_input"
            means domain_input.vtk

    Returns
    -------
        cl_domain : pv.UnstructuredGrid
            The loaded lumen discretization.
    """

    fname = get_case_convention(obj='cl_domain', suffix=suffix, case_dir=case_dir)
    cl_domain = None
    if os.path.exists(fname):
        cl_domain = pv.read(fname)

    return cl_domain
#

def save_centerline_domain(case_dir, cl_domain, suffix="", binary=True, overwrite=False):
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

        suffix : string
            A string indicating a suffix in the mesh name. E.g. suffix="_input"
            means domain_input.vtk

        binary : bool, opt.
            Default True. Wheteher to save vtk files in binary format.

        overwrite : bool, opt
            Default False. Whether to overwrite existing files.
    """

    make_subdirs(case_dir, 'Centerline')
    fname = get_case_convention(obj='cl_domain', suffix=suffix, case_dir=case_dir)
    message = f"{fname} exists and overwritting is set to False."
    if is_writable(fname, overwrite=overwrite, message=message):
        cl_domain.save(fname, binary=binary)
#

def load_centerline_path(case_dir, suffix=""):
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

        suffix : string
            A string indicating a suffix in the mesh name. E.g. suffix="_input"
            means domain_input.vtk


    Returns
    -------
        cl_paths : pv.MultiBlock
            The loaded paths optimizing the distance to the wall.
    """

    fname = get_case_convention(obj='cl_path', suffix=suffix, case_dir=case_dir)

    cl_paths = None
    if os.path.exists(fname):
        cl_paths = pv.read(fname)

    return cl_paths
#

def save_centerline_path(case_dir, cl_path, suffix="", binary=True, overwrite=False):
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

        suffix : string
            A string indicating a suffix in the mesh name. E.g. suffix="_input"
            means domain_input.vtk

        binary : bool, opt.
            Default True. Wheteher to save vtk files in binary format.

        overwrite : bool, opt
            Default False. Whether to overwrite existing files.
    """

    make_subdirs(case_dir, 'Centerline')
    fname = get_case_convention(obj='cl_path', suffix=suffix, case_dir=case_dir)
    message = f"{fname} exists and overwritting is set to False."
    if is_writable(fname, overwrite=overwrite, message=message):
        cl_path.save(fname, binary=binary)
#

def load_centerline(case_dir, suffix=""):
    """
    Load the centerline network under the case directory convention.

    Using UNIX path format, the centerline network is expected to be at subdir Centerline with name
    centerline.vtm, i.e.:

        case_dir/Centerline/centerline.vtm

    Due to the MultiBlock save format of vtk, a directory called centerline is also expecterd to be
    at Centerline subdir containing the centerline data.

    Arguments
    ---------

        case_dir : str
            The path to the case directory.

        suffix : str, opt
            Default an empty string. A suffix to be added before the extension.

    Returns
    -------

        cl_net : vef.CenterlineNetwork
            The loaded centerline
    """

    fname = get_case_convention(obj='centerline', suffix=suffix, case_dir=case_dir)

    cl_net = None
    if os.path.exists(fname):
        cl_mb = pv.read(fname)
        cl_net = vef.CenterlineNetwork().from_multiblock(mb=cl_mb)
    else:
        msg.warning_message(f"No centerline was found in the case at '{case_dir}' .")

    return cl_net
#

def save_centerline(case_dir, cl_net, suffix="", binary=True, overwrite=False):
    """
    Save the centerline network under the case directory convention.

    Using UNIX path format, the centerline network is expected to be at subdir Centerline, with name,
    centerline.vtm, i.e.:

        case_dir/Centerline/centerline.vtm

    Due to the MultiBlock save format of vtk, a directory called centerline is also created at
    Centerline subdir containing the centerline data.

    Arguments
    ---------

        case_dir : str
            The path to the case directory.

        cl_net : vef.CenterlineNetwork
            The computed centerline network.

        suffix : str, opt
            Default an empty string. A suffix to be added before the extension.

        binary : bool, opt.
            Default True. Wheteher to save vtk files in binary format.

        overwrite : bool, opt
            Default False. Whether to overwrite existing files.
    """

    make_subdirs(case_dir, 'Centerline')
    fname = get_case_convention(obj='centerline', suffix=suffix, case_dir=case_dir)
    message = f"{fname} exists and overwritting is set to False."
    cl_mb = cl_net.to_multiblock(add_attributes=True)
    if is_writable(fname=fname, overwrite=overwrite, message=message):
        cl_mb.save(fname, binary)
#

def load_vascular_encoding(case_dir, suffix=""):
    """
    Load the vascular encoding under the case directory convention.

    Using UNIX path format, the vascular encoding is expected to be at subdir Encoding with name
    encoding.vtm, i.e.:

        case_dir/Encoding/encoding.vtm

    Due to the MultiBlock save format of vtk, a directory called encoding is also expected to be
    at Encoding subdir containing the vascular encoding data.

    Arguments
    ---------

        case_dir : str
            The path to the case directory.

        suffix : str, opt
            Default an empty string. A suffix to be added before the extension.

    Returns
    -------

        vsc_enc : vef.VascularEncoding
            The loaded vascular encoding
    """

    fname = get_case_convention(obj='encoding', suffix=suffix, case_dir=case_dir)

    vsc_enc = None
    if os.path.exists(fname):
        enc_mb = pv.read(fname)
        vsc_enc = vef.VascularEncoding.from_multiblock(vsc_mb=enc_mb)

    return vsc_enc
#

def save_vascular_encoding(case_dir, vsc_enc, suffix="", binary=True, overwrite=False):
    """
    Save the vascular encoding under the case directory convention.

    Using UNIX path format, the vascular encoding is saved at subdir Encoding with name
    encoding.vtm, i.e.:

        case_dir/Encoding/encoding.vtm

    Due to the MultiBlock save format of vtk, a directory called encoding is also created
    at Encoding subdir containing the vascular encoding data.

    Arguments
    ---------

        case_dir : str
            The path to the case directory.

        vsc_enc : vef.VascularEncoding
            The computed vascular encoding.

        binary : bool, opt.
            Default True. Wheteher to save vtk files in binary format.

        overwrite : bool, opt
            Default False. Whether to overwrite existing files.
    """

    make_subdirs(case_dir, 'Encoding')
    fname = get_case_convention(obj='encoding', suffix=suffix, case_dir=case_dir)
    message = f"{fname} exists and overwritting is set to False."
    enc_mb = vsc_enc.to_multiblock()
    if is_writable(fname=fname, overwrite=overwrite, message=message):
        enc_mb.save(fname, binary)
#
