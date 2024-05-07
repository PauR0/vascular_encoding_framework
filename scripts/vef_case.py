#! /usr/bin/env python3

import os
import argparse

from vascular_encoding_framework.utils._io import read_json
import vascular_encoding_framework.messages as msg

from .config.readers import (read_centerline_config,
                             read_encoding_config)

from .case_io import in_case, load_vascular_mesh, save_vascular_mesh
from .vef_compute_centerline import compute_centerline
from .vef_encode import encode

def update_ids(case_dir, new_ids):
    """
    This function updates the input boundary ids.

    The new_ids dict is expected to follow the logic {"old_id":"new_id"}. Note that ids must be strings
    and existing ones are not allowed. Due to the sequential nature of the id changing, to make id
    permutation, an "aux" id can be used as follows. Say that the ids "foo" and "bar" are in use and
    a permutation is required, then the following dictionary would do the trick:
                                {"foo":"aux", "bar":"foo", "aux":"bar"}

    TODO: Current version only changes the ids in the boundary_input* files. It would be interesting
    to change the ids in all the derived objects such as other param files or centerline and encoding.

    Arguments
    ---------

        case_dir : str
            The case directory.

        new_ids : {True, dict}
            If None, it is expected to be found at case_dir with name new_ids.json
    """

    if not isinstance(new_ids, dict):
        nids_fname = in_case(case_dir, subdir='', name='new_ids.json')
        new_ids = read_json(nids_fname)

    vmesh = load_vascular_mesh(path=case_dir, suffix='_input')

    for oid, nid in new_ids.items():
        vmesh.boundaries.change_node_id(old_id=oid, new_id=nid)
    save_vascular_mesh(vmesh=vmesh, path=case_dir, suffix='_input', overwrite=True)
#

def show_boundaries(case_dir):
    """
    Load the input mesh and show a plot with the boundaries ids.

    Arguments
    ---------

        case_dir : str
            The case directory.
    """

    vmesh = load_vascular_mesh(path=case_dir, suffix='_input')
    vmesh.plot_boundary_ids(print_data=True)
#


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""This is a general script that allows to perform
    most of the Vascular Encoding Framework core operations such as centerline computation or
    encoding on a given case. It also allows to run basic routines as changing boundary ids.""")

    parser.add_argument('--cc',
                        '--cl-config',
                        '--centerline-config',
                        dest="cl_config",
                        type=str,
                        action='store',
                        default=None,
                        help="""Path to a json with the configuration parameters for the centerline
                        computation.""")

    parser.add_argument('--ec',
                        '--enc-config',
                        '--encoding-config',
                        dest="enc_config",
                        type=str,
                        action='store',
                        default=None,
                        help="""Path to a json with the configuration parameters for the encoding
                        computation.""")

    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        help="""Run in debug mode. This mode shows some plots at different stages
                        of the process.""")


    parser.add_argument('--update-ids',
                        dest="upids",
                        type=str,
                        nargs="?",
                        action='store',
                        const=True,
                        help="""Path to a json file with the ids to be changed with format {"old_id":"new_id"}.
                        If no argument is provided, the file new_ids.json will be loaded from case dir.""")

    parser.add_argument('--show-boundaries',
                        dest="show_boundaries",
                        action='store_true',
                        help="""Display the input mesh and boundaries stored on Meshes dir.""")

    parser.add_argument('-w',
                        action='store_true',
                        help="""Force overwriting of files.""")

    parser.add_argument('-c',
                        '--compute-cl',
                        '--compute-centerline',
                        dest='compute_centerline',
                        action='store_true',
                        help="""Run centerline computation. If it already exists, it wont be
                        overwritten unless -w flag is also raised.""")

    parser.add_argument('-e',
                        '--encode',
                        dest='encode',
                        action='store_true',
                        help="""Run encoding algorithm. If it already exists, it wont be overwritten
                        unless -w flag is also raised.""")

    parser.add_argument('case',
                        action='store',
                        nargs='?',
                        type=str,
                        default=os.getcwd(),
                        help="""The path to a case directory.""")

    args = parser.parse_args()

    if not os.path.exists(args.case) or not os.path.isdir(args.case):
        msg.error_message(f"Wrong path given.... {args.case}")


    if args.upids:
        update_ids(case_dir=args.case, new_ids=args.upids)

    if args.show_boundaries:
        show_boundaries(case_dir=args.case)

    if args.compute_centerline:
        if args.cl_params is not None:
            cl_params = read_centerline_config(path=args.cl_params, abs_path=True)
        compute_centerline(case_dir=args.case,
                           params=cl_params,
                           binary=True,
                           overwrite=args.w,
                           force=False,
                           debug=args.debug)

    if args.encode:
        if args.enc_params is not None:
            enc_params = read_encoding_config(path=args.enc_params, abs_path=True)
        encode(case_dir  = args.case,
               params    = enc_params,
               binary    = True,
               debug     = args.debug,
               overwrite = args.w,
               force     = False)
