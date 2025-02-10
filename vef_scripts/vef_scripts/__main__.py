"""
Vascular Encoding Framework scripts package commandline interface

"""
import argparse
import os
import sys

from vascular_encoding_framework.messages import error_message

from .config.readers import *
from .config.writers import *
from .vef_align import align_encodings
from .vef_case import show_adapted_frame, show_boundaries, update_ids
from .vef_cohort import _Centerliner, _Encoder, _Updater, cohort_run
from .vef_compute_centerline import compute_centerline
from .vef_encode import encode
from .vef_make_case import make_case

MODES = {'case', 'cohort'}


parser = argparse.ArgumentParser(
    description="""Commandline interface of the vascular encoding
framework script package.""")  # TODO: Documentation.

parser.add_argument(
    '--debug',
    dest='debug',
    action='store_true',
    help="""Run in debug mode. This mode shows some plots at different stages of the process.""")

parser.add_argument(
    '-f',
    '--force',
    dest='force',
    action='store_true',
    help="""Force the recomputation of intermediate steps of centerline computation.""")

parser.add_argument('-w',
                    action='store_true',
                    help="""Overwrite existing files.""")

parser.add_argument(
    '--cp',
    '--cl-params',
    '--centerline-params',
    dest='cl_params',
    type=str,
    default=None,
    help="""Path to an alternative json file with the desired centerline params.""")

parser.add_argument(
    '--ep',
    '--ec-params',
    '--encoding-params',
    dest='ec_params',
    type=str,
    default=None,
    help="""Path to an alternative json file with the desired encoding params.""")

parser.add_argument(
    '--gpa-params',
    dest='gpa_params',
    type=str,
    default=None,
    help="""Path to an alternative json file with the desired alignment parameters.""")

parser.add_argument(
    '--align',
    dest='align',
    action='store_true',
    help="""For cohort mode only. Run the GPA algorithm on the encodings of a cohort.""")

parser.add_argument(
    '--update-ids',
    dest='upids',
    type=str,
    nargs='?',
    action='store',
    const=True,
    help="""Path to a json file with the ids to be changed with format {"old_id":"new_id"}.
                    If no argument is provided, the file new_ids.json will be loaded from case dir. WARNING:
                    If an argument is provided in cohort mode, all the cases will update the boundaries with
                    the provided json file.""")

parser.add_argument(
    '--ascii',
    dest='binary',
    action='store_false',
    help="""Whether to write ouput files in ascii. Default is binary mode.""")

parser.add_argument(
    '--show-boundaries',
    dest='show_boundaries',
    action='store_true',
    help="""Display the input mesh and boundaries stored on Meshes dir.""")

parser.add_argument(
    '--show-adapted',
    '--show-adapted-frame',
    dest='show_adapted',
    action='store_true',
    help="""Display the parallel transport of the adapte frame along centerline network.""")

parser.add_argument(
    '-e',
    '--encode',
    dest='encode',
    action='store_true',
    help="""Run vascular encoding. Requires mesh_input, boundaries_input and
                    centerline to exist at case directory.""")

parser.add_argument(
    '-c',
    '--centerline',
    dest='centerline',
    action='store_true',
    help="""Run vascular centerline computation. Requires mesh_input, boundaries_input
                    to exist and to have a propper hierarchy.""")

parser.add_argument(
    '--make',
    dest='make',
    action='store_true',
    help="""Make the case directory using the arugments provided.""")

parser.add_argument(
    '--mesh',
    dest='mesh',
    type=str,
    default=None,
    help="""Ignored if --make is not raised. Path to an existing mesh file to
                    set as input mesh..""")

parser.add_argument(
    '--exclude',
    dest='exclude',
    type=str,
    nargs='+',
    default=None,
    help="""Ignored if mode is not cohort. List of directories at cohort directory
                    to be excluded.""")

parser.add_argument(
    '--n-proc',
    dest='n_proc',
    type=int,
    default=1,
    help="""Default is 1. Number of parallel procesess to use in cohort mode routines.""")

parser.add_argument(
    '--path',
    dest='path',
    action='store',
    default=None,
    help="""If mode is case:\n\t A path/name with the case to be created/run can be
                    provided. If --make is raised and --case is not provided, the name vef_case is
                    used to create a case directory at current working directory. If --make is not
                    raised and --case is not passed, the current working directory is assumed to be
                    a case directory and it is used to run the routines.\n
                    If mode is cohort:\n\t A path to a cohort directory can be passed. If it not
                    raised in cohort mode, the current working directory will be assumed to be a
                    cohort directory and used to run the routines.""")

parser.add_argument(
    'mode',
    action='store',
    default=None,
    help=f"""The mode to run. It must be in [{MODES}]. If mode is case, the command
                    order is: make -> upadte_ids -> show_boundaries -> centerline comp -> encoding.
                    If mode is cohort, then the order is: upadte_ids -> centerline comp -> encoding
                     -> alignment -> SSM""")

args = parser.parse_args()

if args.mode not in MODES:
    error_message(
        f'Wrong value for mode argument. Given is {args.mode}. Must be in {MODES}')
    sys.exit(-1)


if args.mode == 'case':

    cl_params = None
    if args.cl_params is not None:
        cl_params = read_centerline_config(path=args.cl_params, abs_path=True)

    ec_params = None
    if args.ec_params is not None:
        ec_params = read_encoding_config(path=args.ec_params, abs_path=True)

    if args.make:
        case = make_case(
            case_dir=args.path,
            mesh_fname=args.mesh,
            overwrite=args.w,
            cl_params=cl_params,
            ec_params=ec_params)

    else:
        case = args.path if args.path is not None else os.getcwd()

    if args.upids:
        update_ids(case_dir=case, new_ids=args.upids)

    if args.show_boundaries:
        show_boundaries(case_dir=case)

    if args.centerline:
        compute_centerline(
            case_dir=case,
            params=cl_params,
            binary=args.binary,
            overwrite=args.w,
            force=args.force,
            debug=args.debug)

    if args.show_adapted:
        show_adapted_frame(case_dir=case, suffix='')

    if args.encode:
        encode(
            case_dir=case,
            params=ec_params,
            binary=args.binary,
            debug=args.debug,
            overwrite=args.w)

    sys.exit(0)
#

if args.mode == 'cohort':

    cohort_dir = args.path if args.path is not None else os.getcwd()

    if args.upids:
        cohort_run(cohort_dir=cohort_dir,
                   routine=_Updater(new_ids=args.upids),
                   exclude=args.exclude,
                   n_proc=args.n_proc,
                   desc='Cohort mode: updating boundaries')

    if args.centerline:
        cl_params_fname = args.cl_params if args.cl_params is not None else os.path.join(
            cohort_dir, 'centerline.json')
        cl_params = read_centerline_config(
            cohort_dir) if os.path.exists(cl_params_fname) else None
        cohort_run(cohort_dir=cohort_dir,
                   routine=_Centerliner(params=cl_params,
                                        binary=args.binary,
                                        overwrite=args.w,
                                        force=args.force,
                                        debug=False),
                   exclude=args.exclude,
                   n_proc=args.n_proc,
                   desc='Cohort mode: computing centerline')

    if args.encode:
        ec_params_fname = args.ec_params if args.ec_params is not None else os.path.join(
            cohort_dir, 'encoding.json')
        ec_params = read_encoding_config(
            cohort_dir) if os.path.exists(ec_params_fname) else None
        cohort_run(cohort_dir=cohort_dir,
                   routine=_Encoder(params=ec_params,
                                    binary=args.binary,
                                    overwrite=args.w,
                                    debug=False),
                   exclude=args.exclude,
                   n_proc=args.n_proc,
                   desc=f'Cohort mode: Encoding')

    if args.align:
        gpa_params = None
        if args.gpa_params is not None:
            gpa_params = read_alignment_config(path=args.gpa_params)
        align_encodings(
            cohort_dir=cohort_dir,
            params=gpa_params,
            exclude=args.exclude,
            overwrite=args.w)

    sys.exit(0)
#
