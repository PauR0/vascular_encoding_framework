#!/usr/bin/env python3

import os
import sys
import argparse

import vascular_encoding_framework as vef
from vascular_encoding_framework import messages as msg

from config.readers import read_centerline_config
from config.writers import write_centerline_config

from case_io import (load_centerline_domain, save_centerline_domain,
                    load_centerline_path, save_centerline_path,
                    load_vascular_mesh, save_centerline)


def compute_centerline(case_dir, params=None, binary=True, debug=False, overwrite=False):
    """
    Function to make a vef case directory at path provided in case_dir argument.
    Additionally, the filename of a mesh can be passed, and it is copied and saved
    in Meshes directory inside the case. If the mesh_fname is passed, the module also
    attempts to compute the boundaries and save them at the Meshes directory.
    """

    vmesh = load_vascular_mesh(case_dir, suffix='_input')

    if params is None:
        params = read_centerline_config(case_dir)

    cl_domain = load_centerline_domain(case_dir)
    if cl_domain is None or overwrite:
        msg.computing_message("cenerline domain")
        cl_domain = vef.centerline.extract_centerline_domain(vmesh=vmesh,
                                                             params=params['params_domain'],
                                                             debug=debug)
        msg.done_message("cenerline domain")
        save_centerline_domain(case_dir=case_dir, cl_domain=cl_domain, binary=binary, overwrite=overwrite)

    cl_path = load_centerline_path(case_dir)
    if cl_path is None or overwrite:
        msg.computing_message("centerline paths")
        cl_path = vef.centerline.extract_centerline_path(vmesh=vmesh,
                                                          cl_domain=cl_domain,
                                                          params=params['params_path'])
        msg.done_message("centerline paths")
        save_centerline_path(case_dir=case_dir, cl_path=cl_path, binary=binary, overwrite=overwrite)

    cl_net = vef.CenterlineNetwork.from_multiblock_paths(cl_path,
                                                         knots=params['knots'],
                                                         graft_rate=params['graft_rate'],
                                                         force_tangent=params['force_tangent'])

    write_centerline_config(path=case_dir, data=params)
    save_centerline(case_dir=case_dir, cl_net=cl_net, binary=binary, overwrite=overwrite)
#






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Script to compute the centerline of a vascular
    structure using the Vascular Encoding Framework conventions. The computation of the centerline
    requires a vascular mesh with _input suffix to exist in the case. Parameters can be tuned using
    a centerline.json file stored in the case directory, otherwise default values are used.""")

    parser.add_argument('-w',
                        action='store_true',
                        help="""Force overwriting if it already exists.""")

    parser.add_argument('--ascii',
                        dest='binary',
                        action='store_false',
                        help="""Whether to write ouput files in binary. Default is binary mode.""")

    parser.add_argument('-p',
                        '--params',
                        dest='params',
                        type=str,
                        default=None,
                        help = """Alternative json file with the desired params.""")

    parser.add_argument('case',
                        action='store',
                        default=None,
                        help="""The path/name with the case. If none is provided,
                        a directory called vef_case.""")

    args = parser.parse_args()

    params = None
    if args.params is not None:
        params = read_centerline_config(path=args.params, abs_path=True)

    #compute_centerline(case_dir=args.case, params=params, overwrite=args.w)
    compute_centerline(case_dir='/Users/pau/tmp/test_case', params=None, overwrite=False)
    #
