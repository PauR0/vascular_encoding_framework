
import os
import json


def is_writable(fname, overwrite=True):
    """
    Check if file exists or should be overwritten.

    Warning: This function assume overwrite by default.

    Arguments:
    -----------

        fname : str
            The file name to check.

        overwrite : bool, opt
            Default True. Whether to overwrite or not.

    Returns:
    ----------
        True if fname can be written False otherwise.

    """

    if os.path.exists(fname) and not overwrite:
        return False

    return True
#

def write_json(fname, data, indent=4, overwrite=True):
    """
    Write a dictionary to a json file. Before saving, this function checks if there exist a file
    with the same name, and overwritting can be prevented using the overwrite argument. All the
    dictionary entries have to be json-serializable.

    Arguments:
    -----------

        fname : str
            The filename to be saved. If does not end with .json extension, it is added.

        data : dict
            The dictionary to be written.

        indent : int, opt
            Default 4. Whether to add indentation levels to entries in the json file.

        overwrite : bool, opt
            Default False. Whether to overwritte an already existing file.

    """


    if is_writable(fname, overwrite=overwrite):
        with open(fname, 'w', encoding="utf-8") as f:
            f.write(json.dumps(data, indent=indent))
#