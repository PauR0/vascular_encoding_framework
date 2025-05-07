import json
import os

from ..messages import error_message


def is_writable(fname, overwrite=True, message=None):
    """
    Check if file exists or should be overwritten.

    Warning: This function assume overwrite by default.

    Parameters
    ----------
    fname : str
        The file name to check.
    overwrite : bool, opt
        Default True. Whether to overwrite or not.
    message : str, opt
        Default None. If passed, it is printed as the description
        of an error message.

    Returns
    -------
    out : bool
        True if fname can be written False otherwise.
    """

    out = True
    if os.path.exists(fname) and not overwrite:
        if message is not None:
            error_message(message)
        out = False

    return out


def write_json(fname, data, indent=4, overwrite=True):
    """
    Write a dictionary to a json file. Before saving, this function checks if there exist a file
    with the same name, and overwriting can be prevented using the overwrite argument. All the
    dictionary entries have to be json-serializable.

    Parameters
    ----------
    fname : str
        The filename to be saved. If does not end with .json extension, it is added.
    data : dict
        The dictionary to be written.
    indent : int, opt
        Default 4. Whether to add indentation levels to entries in the json file.
    overwrite : bool, opt
        Default False. Whether to overwrite an already existing file.

    """

    if is_writable(fname, overwrite=overwrite):
        with open(fname, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=indent))


def read_json(file):
    """
    Read a json from file.

    Parameters
    ----------
    file : str

    Returns
    -------
    params : dict

    See Also
    --------
    save

    """

    params = None
    with open(file, "r", encoding="utf-8") as param_file:
        params = json.load(param_file)

    return params
