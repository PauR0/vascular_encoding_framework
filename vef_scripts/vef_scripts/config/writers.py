
__all__ = [
    "write_centerline_config",
    "write_encoding_config",
]

import os
import json

from .readers import read_json
from . import _defaults_dir

def pretty_write(j, fname, write_replacements=None):
    """
    Format an write json files.

    Arguments:
    -------------

        j : str
            The json in string format.

        fname : string
            The file name to write.

        write_replacements : list[list[]], opt
            A list with the replacements.
    """

    if write_replacements is None:
        write_replacements = [[',', ',\n'], ['}}', '}\n }'],
                              ['{"', '{\n "'], ['"}', '"\n}']]

    for r in write_replacements:
        j = j.replace(r[0],r[1])
    j += "\n"

    with open(fname, 'w', encoding='utf-8') as f:
        f.write(j)
#

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)

def get_json_writer(default_fname):

    params = read_json(os.path.join(_defaults_dir, default_fname))
    fname = default_fname

    def write_json(path, data=None, abs_path=False):

        if abs_path:
            json_file = path
        else:
            json_file = os.path.join(path, fname)

        try:
            if data:
                for k in data:
                    params[k] = data[k]
            pretty_write(json.dumps(params, cls=NumpyEncoder), json_file)
        except FileNotFoundError:
            pass
        #

        return params
    #

    return write_json
#


write_centerline_config = get_json_writer('centerline.json')
write_encoding_config = get_json_writer('encoding.json')
#