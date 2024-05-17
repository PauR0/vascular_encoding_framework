

import argparse

import vascular_encoding_framework as vef

from .config.readers import read_encoding_config
from .config.writers import write_encoding_config

from .case_io import (load_vascular_mesh, load_centerline,
                     save_vascular_encoding)


def encode(case_dir, params=None, binary=True, debug=False, overwrite=False, force=False):
    """
    Given a vef case directory where there is a vascular mesh with the '_input' suffix, whose
    centerline is also stored at the Centerline subdir. This function-script allows the computation
    of the vascular encoding, stores it at the Encoding subdir.

    By default this function wont overwrite any file, however overwritting can be handled with the
    overwrite and force arguments. If a preexisting centerline exists, the overwrite argument allow
    to overwrite the centerline file. However, if domain and path files already exists, to force
    the recomputaion the argument force must be used.


    Arguments
    ---------

        case_dir : str
            The case directory under the vef convention.

        params : dict, opt
            Default None. The parameters for the encoding computation. If None, params are read
            from encoding.json at case_dir. If encoding.json is not found, default parameters
            are assumed.

        binary : bool, opt
            Default True. Whether to write vtk files in binary mode. Binary is recomended to save
            disk space.

        overwrite : bool, opt
            Default False. Whether to existing files

        force : bool, opt
            Default False. Whether to force recomputation even if files exist at case_dir.
            WARNING: Forcing recomputation does not imply overwritting!

    Return
    ------
        cl_net : vef.VascularEncoding
            The computed vascular encoding.
    """

    vmesh = load_vascular_mesh(case_dir, suffix='_input')
    if vmesh is None:
        return None

    cl_net = load_centerline(case_dir=case_dir)
    if cl_net is None:
        return None


    if params is None:
        params = read_encoding_config(case_dir)

    vsc_enc = vef.encode_vascular_mesh(vmesh=vmesh, cl_net=cl_net, params=params, debug=debug)

    write_encoding_config(path=case_dir, data=params)
    save_vascular_encoding(case_dir=case_dir, vsc_enc=vsc_enc, binary=binary, overwrite=overwrite)

    return vsc_enc
#






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Script to compute the encoding of a vascular
    structure using the Vascular Encoding Framework conventions. The computation of the encoding
    requires a vascular mesh with _input suffix to exist in the case. A centerline is also required
    to be stored in the Centerline subdir. Parameters can be tuned by means of an encoding.json
    file stored in the case directory, otherwise default values are used.""")

    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        help="""Run in debug mode. This mode shows some plots at different stages of the process.""")

    parser.add_argument('-f',
                        '--force',
                        dest='force',
                        action='store_true',
                        help="""Force the computation of domain and path.""")

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
        params = read_encoding_config(path=args.params, abs_path=True)

    encode(case_dir=args.case, params=params, overwrite=args.w, force=args.force, debug=args.debug)
    #
