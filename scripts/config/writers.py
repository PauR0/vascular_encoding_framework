

import json

from .readers import read_json
from ..config import _defaults_dir

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


def get_json_writer(default_fname, template_file):

    params = read_json(template_file)
    fname = default_fname

    def write_json(path, data=None, abs_path=False):

        if abs_path:
            json_file = path
        else:
            json_file = path + "/" + fname

        try:
            if data:
                for k in data['metadata']:
                    params['metadata'][k] = data['metadata'][k]
                for k in data['data']:
                    params['data'][k] = data['data'][k]
            pretty_write(json.dumps(params), json_file)
        except FileNotFoundError:
            pass
        #

        return params
    #

    return write_json
#