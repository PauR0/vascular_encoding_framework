"""
Vascular Encoding Framework scripts package commandline interface

"""

import os
import sys
import argparse

from vascular_encoding_framework.messages import error_message

from .config.readers import *
from .config.writers import *

from .vef_make_case import make_case
from .vef_case import update_ids, show_boundaries
from .vef_compute_centerline import compute_centerline
from .vef_encode import encode


MODES = {'make_case', 'run_case', 'GPA', 'SSM'}


parser = argparse.ArgumentParser(description="""Commandline interface of the vascular encoding
framework script package.""") #TODO: Documentation.

parser.add_argument('--debug',
                    dest='debug',
                    action='store_true',
                    help="""Run in debug mode. This mode shows some plots at different stages of the process.""")

parser.add_argument('-f',
                    '--force',
                    dest='force',
                    action='store_true',
                    help="""Force the recomputation of intermediate steps of centerline.""")

parser.add_argument('-w',
                    action='store_true',
                    help="""Force overwriting if it already exists.""")

parser.add_argument('-cp',
                    '--cl-params',
                    '--centerline-params',
                    dest='cl_params',
                    type=str,
                    default=None,
                    help = """Alternative json file with the desired centerline params.""")

parser.add_argument('-ep',
                    '--ec-params',
                    '--encoding-params',
                    dest='ec_params',
                    type=str,
                    default=None,
                    help = """Alternative json file with the desired encoding params.""")

parser.add_argument('--update-ids',
                    dest="upids",
                    type=str,
                    nargs="?",
                    action='store',
                    const=True,
                    help="""Path to a json file with the ids to be changed with format {"old_id":"new_id"}.
                    If no argument is provided, the file new_ids.json will be loaded from case dir.""")

parser.add_argument('--ascii',
                    dest='binary',
                    action='store_false',
                    help="""Whether to write ouput files in binary. Default is binary mode.""")

parser.add_argument('--show-boundaries',
                    dest="show_boundaries",
                    action='store_true',
                    help="""Display the input mesh and boundaries stored on Meshes dir.""")

parser.add_argument('-e',
                    '-encode',
                    dest='encode',
                    action='store_true',
                    help="""Ignored if mode is not run_case. Run vascular encoding. Requires
                    mesh_input, boundaries_input and centerline to exist""")

parser.add_argument('-c',
                    '-centerline',
                    dest='centerline',
                    action='store_true',
                    help="""Ignored if mode is not run_case. Run vascular centerline computation.
                    Requires mesh_input, boundaries_input to exist and to have a propper""")

parser.add_argument('--mesh',
                    dest='mesh',
                    type=str,
                    default=None,
                    help = """Ignored if mode is not make_case. Path to an existing mesh file.""")

parser.add_argument('--case',
                    dest='case',
                    action='store',
                    default=None,
                    help="""The path/name with the case to be created/run. In make_case mode, if
                    not provided, a directory called vef_case is created at current working
                    directory. In run_case mode, if not provided, current working directory is used.""")

parser.add_argument('mode',
                    action='store',
                    default=None,
                    help=f"""The mode to run. It must be in [{MODES}].""")

args = parser.parse_args()

if args.mode not in MODES:
    error_message(f"Wrong value for mode argument. Given is {args.mode}. Must be in {MODES}")
    sys.exit(-1)


cl_params=None
if args.cl_params is not None:
    cl_params = read_centerline_config(path=args.cl_params, abs_path=True)

ec_params=None
if args.ec_params is not None:
    ec_params = read_encoding_config(path=args.ec_params, abs_path=True)


if args.mode == 'make_case':
    make_case(case_dir=args.case, mesh_fname=args.mesh, show_boundaries=args.show_boundaries, overwrite=args.w, cl_params=cl_params, ec_params=ec_params)
    sys.exit(0)
#

if args.mode == 'run_case':

    case = args.case if args.case is not None else os.getcwd()

    if args.upids:
        update_ids(case_dir=case, new_ids=args.upids)

    if args.show_boundaries:
        show_boundaries(case_dir=case)

    if args.centerline:
        compute_centerline(case_dir=case, params=cl_params, binary=args.binary, overwrite=args.w, force=args.force, debug=args.debug)

    if args.encode:
        encode(case_dir=case, params=ec_params, binary=args.binary, debug=args.debug, overwrite=args.w)

    sys.exit(0)
#

if args.mode in ['GPA', 'SSM']:
    print(f"Sorry mode {args.mode} is not implemented yet...")
    sys.exit(0)
#