"""
Vascular Encoding Framework scripts package commandline interface

"""

import os
import sys
import argparse

from vascular_encoding_framework.messages import error_message

from .vef_make_case import make_case
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

parser.add_argument('--ascii',
                    dest='binary',
                    action='store_false',
                    help="""Whether to write ouput files in binary. Default is binary mode.""")

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


parser.add_argument('mode',
                    action='store',
                    default=None,
                    help=f"""The mode to run. It must be in [{MODES}].""")

args = parser.parse_args()

if args.mode not in MODES:
    error_message(f"Wrong value for mode argument. Given is {args.mode}. Must be in {MODES}")
    sys.exit(-1)

params = None
if args.params is not None:
    params = read_centerline_config(path=args.params, abs_path=True)

compute_centerline(case_dir=args.case, params=params, overwrite=args.w, force=args.force, debug=args.debug)
#
