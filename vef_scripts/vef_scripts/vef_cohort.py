"""
Module for cohort input/output.

A cohort directory is but a directory containing case directories. It may
contain other kind of files and directories, and most of the input functions
will have an argument to exclude some desired cases.
"""
import os
from copy import deepcopy
from inspect import signature
from multiprocessing import Pool

from tqdm.contrib.concurrent import process_map

import vascular_encoding_framework as vef
from vascular_encoding_framework.messages import error_message

from .case_io import (get_case_convention, load_centerline,
                      load_vascular_encoding, load_vascular_mesh,
                      save_centerline, save_vascular_encoding,
                      save_vascular_mesh)
from .vef_case import update_ids
from .vef_compute_centerline import compute_centerline
from .vef_encode import encode


def get_case_directories(
        cohort_dir,
        exclude=None,
        required=None,
        suffix='',
        cohort_relative=True):
    """
    Get a list with the cases directories in a cohort dir.

    A subdirectory will be considered a case if contains the required object.
    If no required object has been passed. All the subdirectories, except from
    ['Log', 'SSM', '.DS_Store'] and those excluded by argument will be considered
    as a case.

    Arguments
    ---------

        cohort_dir : str
            The path to the directory where the cohort is stored.

        exclude : list[str]. optional
            Default None. A list with the subdirectories to be excluded.

        required : {None, "mesh", "centerline", "encoding"}, optional
            Default None. An object required to exist for a directory to be considered as a case.

        suffix : str, optional
            Default None. A suffix to be appended to filename, before extension in required files.

        cohort_relative : bool, optional
            Default True. If False, the cohort dir is preapended in case_dirs elements.

    Returns
    -------

        case_dirs : list[str]
    """

    if exclude is None:
        exclude = []

    exclude += ['Log', 'SSM', '.DS_Store']

    case_dirs = []
    for d in sorted(os.listdir(cohort_dir)):
        full_d = os.path.join(cohort_dir, d)
        req = '.' if required is None else get_case_convention(
            required, suffix=suffix, case_dir=full_d)  # '.' Should always exist
        if os.path.isdir(full_d) and os.path.exists(req) and d not in exclude:
            if cohort_relative:
                case_dirs.append(d)
            else:
                case_dirs.append(full_d)

    return case_dirs
#


def load_cohort_object(
        cohort_dir,
        which,
        exclude=None,
        keys_from_dirs=True,
        suffix=''):
    """
    Load a certain object from a cohort directory.

    Arguments
    ---------

        cohort_dir : str
            The path to the directory where the cohort is stored.

        which : {'centerline', 'encoding'}
            The object to be loaded.

        exclude : list[str]
            Default is None. A list containing the name of the directories to be exlcuded. By names
            what is meant is the paths relative to the cohort_dir.

        keys_from_dirs : bool or callable, optional
            Default True. Whether to use the directory names as the keys of the dict. A callable
            can be passed to filter the directory names, i.e. lambda s: s.replace('substring', ''),
            removing the 'substring' from the directory names in the ids. If False, the index is
            set by the order after sorting the case drectries.

        suffix : str, optional
            Default an empty string. The suffix used when saving the object.

    Returns
    -------

        cohort : Dict[str:VascularEncoding]
            The dictionary containig the vascular encoding objects.

    """

    assert which in {'encoding', 'centerline'}, error_message(
        f"Can't load object {which} from cohort. Available options are {{'centerline', 'encoding'}}.", prnt=False)

    if exclude is None:
        exclude = []

    if which == 'encoding':
        loader = load_vascular_encoding
    elif which == 'centerline':
        loader = load_centerline
    elif which == 'mesh':
        def loader(
            case_dir,
            suffix): return load_vascular_mesh(
            path=case_dir,
            suffix=suffix)

    cases = get_case_directories(
        cohort_dir=cohort_dir,
        exclude=exclude,
        required=which,
        suffix=suffix,
        cohort_relative=False)
    cohort = {}
    for case in cases:
        obj = loader(case_dir=case, suffix=suffix)
        if obj is not None:
            k = str(len(cohort))
            if callable(keys_from_dirs):
                k = keys_from_dirs(case)
            else:
                k = case
            cohort[k] = obj

    return cohort
#


def save_cohort_object(
        cohort_dir,
        cohort,
        suffix='',
        binary=False,
        overwrite=False):
    """
    Save a certain object at case directories contained in a cohort directory.

    Arguments
    ---------

        cohort_dir : str
            The path to the cohort directory.

        cohort : dict[obj]
            A dictionary containing the name of the case directories as keys (relative to cohort_dir)
            and as value the object selected. Currently supported classes for the objects are
            {VascularEncoding, CenterlineNetwork, VascularMesh}.

        suffix : str, optional
            Default an empty string. The suffix used when saving the object.

        binary : bool, optional
            Default False. Whether to overwrite existing files.

        overwrite : bool, optional
            Default False. Whether to overwrite existing file.

    """

    for case, obj in cohort.items():

        case_dir = os.path.join(cohort_dir, case)
        if not os.path.exists(case_dir):
            error_message(
                f'Wrong case_dir at: {case_dir}. The {type(obj)} object wont be saved.')

        if not isinstance(
            obj,
            (vef.VascularEncoding,
             vef.CenterlineNetwork,
             vef.VascularMesh)):
            error_message(
                'Only VascularEncoding, CenterlineNetwork adn VascularMesh objects are supported for cohort saving.')
            return

        if isinstance(obj, vef.CenterlineNetwork):
            save_centerline(
                case_dir=case_dir,
                cl_net=obj,
                suffix=suffix,
                binary=binary,
                overwrite=overwrite)

        elif isinstance(obj, vef.VascularEncoding):
            save_vascular_encoding(
                case_dir=case_dir,
                vsc_enc=obj,
                suffix=suffix,
                binary=binary,
                overwrite=overwrite)

        elif isinstance(obj, vef.VascularMesh):
            save_vascular_mesh(
                vmesh=obj,
                path=case_dir,
                suffix=suffix,
                binary=binary,
                overwrite=overwrite)

        else:
            error_message(
                f'Wrong object type {type(obj)}. Only {{VascularEncoding, CenterlineNetwork, VascularMesh}} are supported.')
#


def cohort_run(cohort_dir, routine, exclude=None, n_proc=1, desc=None):
    """
    Function to execute a functionality across the cohort directory.

    Arguments
    ---------

        cohort_dir : str
            The directory where the cohort is stored.

        routine : callable
            The routine to be run on each case. A callable that take the path to a
            case directory.

        exclude : list[str]
            A list of dictionaries to avoid running.

        n_proc : int
            Number of parallel processes in which to run the routine across the cohort.

        desc : str
            The description of the routine to be displayed in the progress bar.
    """

    if not callable(routine):
        return error_message(
            f'Wrong value for kind argument in cohort_run. Passed is {routine}, \
                             allowed are callable objects that take a path to a case directory as input.')

    if desc is None:
        desc = f'Running {routine.__class__.__name__}'

    case_dirs = get_case_directories(
        cohort_dir=cohort_dir,
        exclude=exclude,
        cohort_relative=False)
    process_map(routine, case_dirs, max_workers=n_proc, desc=desc)

    # with Pool(processes=n_proc) as pool:
    #    pool.map(routine, case_dirs)
#


class _Routinizer:
    """
    Auxiliar class to make callable objects suitable for Pool parallelization.
    """

    def __init__(self, func, **kwargs) -> None:

        self.__func = func
        self.__params = {}

        sgntr = signature(func)
        self.__params = {name: deepcopy(kwargs.get(name, param.default))
                         for name, param in sgntr.parameters.items()
                         if param.default is not param.empty}
    #

    def __call__(self, case_dir) -> None:
        self.__func(case_dir, **self.__params)
    #
#


class _Updater(_Routinizer):

    def __init__(self, **kwargs) -> None:
        super().__init__(func=update_ids, **kwargs)
    #
#


class _Centerliner(_Routinizer):

    def __init__(self, **kwargs) -> None:
        super().__init__(func=compute_centerline, **kwargs)
    #
#


class _Encoder(_Routinizer):

    def __init__(self, **kwargs) -> None:
        super().__init__(func=encode, **kwargs)
    #
#
