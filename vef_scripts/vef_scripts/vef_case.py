

import os

from vascular_encoding_framework.utils._io import read_json
import vascular_encoding_framework.messages as msg


from .vef_compute_centerline import compute_centerline
from .vef_encode import encode

from .config.readers import (read_centerline_config,
                            read_encoding_config)

from .case_io import load_vascular_mesh, save_vascular_mesh


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
        nids_fname = os.path.join(case_dir, 'new_ids.json')
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