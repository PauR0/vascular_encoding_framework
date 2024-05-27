
import numpy as np

import vascular_encoding_framework as vef
from vascular_encoding_framework import messages as msg

from .config.readers import read_centerline_config
from .config.writers import write_centerline_config

from .case_io import (load_centerline_domain, save_centerline_domain,
                    load_centerline_path, save_centerline_path,
                    load_vascular_mesh, save_centerline)


def set_v1_from_boundary_roots(vmesh, params):
    """
    This function checks the v1 in the boundary roots of the boundaries tree of a VascularMesh.
    Then if it's not None, the v1 is set on the centerlines starting on the roots.

    Arguments
    ---------

        vmesh : VascularMesh
            The vascular mesh with boundaries defined.

        params : dict
            The parameter dictionary.

    Returns
    -------

        params : dict
            The parameter dictionary with the new entries with the v1 appropriately defined.
    """

    for rid in vmesh.boundaries.roots:
        b_root = vmesh.boundaries[rid]
        if b_root.v1 is not None:
            for chid in b_root.children:
                if not chid in params:
                    params[chid] = {}
                params[chid]['p'] = np.copy(b_root.v1)
                params[chid]['pt_mode'] = 'project'

    return params
#

def compute_centerline(case_dir, params=None, binary=True, overwrite=False, force=False, debug=False):
    """
    Given a vef case directory with an existing mesh with the '_input' suffix, this function-script
    allows the computation of the centerline and its storing at the Centerline subdir.

    By default this function wont overwrite any file, however overwritting can be handled with the
    overwrite and force arguments. If a preexisting centerline exists, the overwrite argument allow
    to overwrite the centerline file. However, if domain and path files already exists, to force
    the recomputaion the argument force must be used.


    Arguments
    ---------

        case_dir : str
            The case directory under the vef convention.

        params : dict, opt
            Default None. The parameters for the centerline computation. If None, params are read
            from centerline.json at case_dir. If centerline.json is not found, default parameters
            are assumed.

        binary : bool, opt
            Default True. Whether to write vtk files in binary mode. Binary is recomended to save
            disk space.

        overwrite : bool, opt
            Default False. Whether to overwrite centerline.vtm file

        force : bool, opt
            Default False. Whether to force recomputation even if files exist at case_dir.
            WARNING: Forcing recomputation does not imply overwritting!

        debug : bool, opt
            Default False. Whether to run the script in debug mode.

    Return
    ------
        cl_net : CenterlineNetwork
            The computed centerline.
    """

    vmesh = load_vascular_mesh(case_dir, suffix='_input')
    if vmesh is None:
        return None

    if params is None:
        params = read_centerline_config(case_dir)

    set_v1_from_boundary_roots(vmesh=vmesh, params=params)

    cl_domain = load_centerline_domain(case_dir)
    if cl_domain is None or force:
        msg.computing_message("cenerline domain")
        cl_domain = vef.centerline.extract_centerline_domain(vmesh=vmesh,
                                                             params=params['params_domain'],
                                                             debug=debug)
        msg.done_message("cenerline domain")
        save_centerline_domain(case_dir=case_dir, cl_domain=cl_domain, binary=binary, overwrite=overwrite)

    cl_path = load_centerline_path(case_dir)
    if cl_path is None or force:
        msg.computing_message("centerline paths")
        cl_path = vef.centerline.extract_centerline_path(vmesh=vmesh,
                                                          cl_domain=cl_domain,
                                                          params=params['params_path'],
                                                          debug=debug)
        msg.done_message("centerline paths")
        save_centerline_path(case_dir=case_dir, cl_path=cl_path, binary=binary, overwrite=overwrite)

    cl_net = vef.CenterlineNetwork.from_multiblock_paths(cl_path,
                                                         **params)

    write_centerline_config(path=case_dir, data=params)
    save_centerline(case_dir=case_dir, cl_net=cl_net, binary=binary, overwrite=overwrite)

    return cl_net
#
