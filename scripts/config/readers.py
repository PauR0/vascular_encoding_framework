
import os
import json
from copy import deepcopy

from . import _defaults_dir

def read_json(file):
    """
    Read a json from file


    TODO: Use utils._io.read_json
    Arguments:
    -----------

        file : str

    Returns:
    ---------

        params : dict
    """
    params = None
    with open(file, 'r') as param_file:
        params = json.load(param_file)

    return params
#

def param_from_file(params, path):
    """
    Provide support to redirect parameter to another json to prevent nested jsons.
    """

    for k, v in params.items():
        if isinstance(v, str):
            if v.startswith("FILE::"): #Two semicolon -> _defaults_dir
                params[k] = read_json(os.path.join(_defaults_dir, v.removeprefix("FILE::")))
            elif v.startswith("FILE:"): #One semicolon -> same_dir
                params[k] = read_json(v.removeprefix("FILE:"))
        elif isinstance(v, dict):
            params[k] = param_from_file(v, path)

    return params
#

def get_json_reader(default_name):
    """
    Function factory to get specific parameter readers.

    Arguments:
    -----------

            default_name : str
                The default name that reader

            temp_fname : str
                The name of the template json, typically stored in
                defaults/ dir.

    Returns:
    ---------

        json_reader : function
            A json reader with a predefine behaviour such as a default name,
            and a template.
    """

    params = read_json(os.path.join(_defaults_dir, default_name))
    fname = default_name

    def json_reader(path=None,abs_path=False):

        new_params = deepcopy(params)
        try:
            if abs_path:
                json_file = path
                path = os.path.dirname(json_file)
            else:
                json_file = os.path.join(path, fname)

            read_params = read_json(json_file)
            for k in read_params:
                    new_params[k] = read_params[k]
        except (FileNotFoundError, TypeError):
            print(f"Could not find {json_file}. Assuming default parameters.")

        new_params = param_from_file(params=new_params, path=path)
        return new_params
    #

    return json_reader
#


read_centerline_config = get_json_reader(default_name="centerline.json")
#
