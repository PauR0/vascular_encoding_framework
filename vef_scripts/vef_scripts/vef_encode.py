import vascular_encoding_framework as vef

from .case_io import load_centerline, load_vascular_mesh, save_vascular_encoding
from .config.readers import read_encoding_config
from .config.writers import write_encoding_config


def encode(case_dir, params=None, binary=True, debug=False, overwrite=False):
    """
    Given a vef case directory where there is a vascular mesh with the '_input' suffix, whose
    centerline is also stored at the Centerline subdir. This function-script allows the computation
    of the vascular anatomy encoding, stores it at the Encoding subdir.

    By default this function wont overwrite any file, however overwriting can be handled with the
    overwrite and force arguments. If a preexisting centerline exists, the overwrite argument allow
    to overwrite the centerline file. However, if domain and path files already exists, to force
    the recomputation the argument force must be used.


    Arguments:
    ---------
        case_dir : str
            The case directory under the vef convention.

        params : dict, opt
            Default None. The parameters for the encoding computation. If None, params are read
            from encoding.json at case_dir. If encoding.json is not found, default parameters
            are assumed.

        binary : bool, opt
            Default True. Whether to write vtk files in binary mode. Binary is recommended to save
            disk space.

        overwrite : bool, opt
            Default False. Whether to existing files


    Return:
    ------
        vsc_enc : vef.VascularAnatomyEncoding
            The computed vascular anatomy encoding.
    """

    vmesh = load_vascular_mesh(case_dir, suffix="_input")
    if vmesh is None:
        return None

    cl_tree = load_centerline(case_dir=case_dir)
    if cl_tree is None:
        return None

    if params is None:
        params = read_encoding_config(case_dir)

    vsc_enc = vef.encode_vascular_mesh(vmesh=vmesh, cl_tree=cl_tree, debug=debug, **params)

    write_encoding_config(path=case_dir, data=params)
    save_vascular_encoding(case_dir=case_dir, vsc_enc=vsc_enc, binary=binary, overwrite=overwrite)

    return vsc_enc


#
