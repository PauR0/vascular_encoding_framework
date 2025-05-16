__all__ = [
    "check_specific",
    "filter_specific",
    "broadcast_kwargs",
    "is_sequence",
    "is_numeric",
    "is_arrayable",
    "attribute_checker",
    "attribute_setter",
]

from copy import deepcopy
from typing import Any

import numpy as np


def check_specific(params: dict, nid: str, arg: str, default: Any):
    """
    Check if the params dict contains params[nid][arg].
    Then if it exists it is returned, otherwise default argument is returned.

    This function is meant to be used to filter out node specific parameters in
    in functions applied to a Tree.

    Parameters
    ----------
    params : dict
        The Node specific parameter dict.
    nid : str
        The node id to check for.
    arg : str
        The argument name to check for.
    default : Any
        The default value the param arg must have.

    Returns
    -------
    : Any
        Either params[nid][arg] or default.
    """

    try:
        return params[nid][arg]
    except KeyError:
        return default


def filter_specific(params: dict, nid: str, exclude: list[str] = None) -> dict:
    """
    Resolve the priority of the parameters for a given id.

    Essentially consist in checking if for a parameter p in params, there's a node specific value
    given and set it right.


    Parameters
    ----------
    parms: dict
        A dictionary typically, the kwargs from another function.
    nid: str
        The id of a node element for which to sort the parameters.
    exclude: list[str], optional
        Name of node-specific elements not to be included in the resulting dictionary.


    Returns
    -------
    out_params : dict
        The dictionary with the sorted ou parameters.
    """

    # There's no node-specific parameter
    out_params = deepcopy(params)
    if nid not in params:
        return out_params

    if not isinstance(params[nid], dict):
        raise ValueError(
            f"Wrong type for specific arguments. Expected dict, passed {params[nid].__class__.__name__}"
        )

    exclude = [] if exclude is None else exclude
    out_params.update({k: deepcopy(v) for k, v in out_params[nid].items() if k not in exclude})
    return out_params


def broadcast_kwargs(**kwargs) -> dict[str, np.ndarray]:
    """
    Broadcast all input kwargs to the same shape.

    Parameters
    ----------
    **kwargs : float or ndarray
        Arbitrary keyword arguments that must be either floats or numpy arrays.

    Returns
    -------
    : dict
        A dictionary with the same keys as kwargs, where all values
        are numpy arrays of the same shape.

    Raises
    ------
    TypeError
        If any input is not a float or numpy array.
    ValueError
        If arrays of different shapes are provided and cannot be broadcast together.

    Examples
    --------
    >>> broadcast_kwargs(a=1.0, b=np.array([2.0, 3.0]))
    {'a': array([1., 1.]), 'b': array([2., 3.])}

    >>> broadcast_kwargs(x=np.ones((2, 3)), y=5.0)
    {'x': array([[1., 1., 1.], [1., 1., 1.]]), 'y': array([[5., 5., 5.], [5., 5., 5.]])}
    """
    # Validate input types and collect array shapes
    array_shapes = []

    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            array_shapes.append(value.shape)
        elif not isinstance(value, float):
            raise TypeError(f"Input '{key}' must be a float or numpy array, got {type(value)}")

    # If no arrays provided, return dictionary of 1D arrays with single elements
    if not array_shapes:
        return {k: np.array([v]) for k, v in kwargs.items()}

    # Determine target shape from all arrays
    try:
        target_shape = np.broadcast_shapes(*array_shapes)
    except ValueError:
        raise ValueError("Input arrays have incompatible shapes that cannot be broadcast together")

    # Create output dictionary
    result = {}

    for key, value in kwargs.items():
        if isinstance(value, float):
            result[key] = np.full(target_shape, value)
        else:  # Must be a numpy array
            try:
                result[key] = np.broadcast_to(value, target_shape).copy()
            except ValueError:
                raise ValueError(
                    f"Could not broadcast input '{key}' to target shape {target_shape}"
                )

    return result


def is_sequence(obj) -> bool:
    """
    Check wether an object is a sequence.

    Parameters
    ----------
    obj : any
        The object to be checked.

    Returns
    -------
    : bool
    """
    if isinstance(obj, str) and len(obj) < 2:
        return False

    return hasattr(obj, "__iter__") and callable(getattr(obj, "__iter__"))


def is_numeric(obj) -> bool:
    """
    Check whether a object is numeric.

    Parameters
    ----------
    seq : iterable object.
        The sequence to test.

    Returns
    -------
    bool
    """

    numeric = (int, float)
    if not isinstance(obj, numeric):
        return False
    return True


def is_arrayable(seq) -> bool:
    """
    Check whether a sequence is all numeric and safe to be casted it to a numpy array. This function
    is used to parse float list as numpy arrays but preventing strings and other actual arrayable
    functions to be a numpy array.

    Parameters
    ----------
    seq : any
        The object to be tested.

    Returns
    -------
    bool
    """

    if not is_sequence(seq):
        return False

    for elmnt in seq:
        if is_sequence(elmnt):
            if not is_arrayable(elmnt):
                return False
        else:
            if not is_numeric(elmnt):
                return False

    return True


def attribute_checker(obj, atts, info=None, opts=None) -> None:
    """
    Check if attribute has been set otherwise raise an AttributeError.

    Parameters
    ----------
    obj : any,
        The object the attributes of which will be checked.
    atts : list[str]
        The names of the attributes to be checked for.
    info : str, opt
        An information string to be added to error message before
        'Attribute {att} is None....'. If None, no message is printed.
    opts : List[Any], optional.
        Default None. A list containing accepted values for attribute.

    Raises
    ------
        AttributeError
            If the attributes are None, or not in opts.
    """

    if info is None:
        info = ""
    if opts is None:
        for att in atts:
            if getattr(obj, att) is None:
                raise AttributeError(f"{info}. Attribute {att} is {getattr(obj, att)}....")

    else:
        for att, opt in zip(atts, opts):
            if getattr(obj, att) not in opt:
                raise ValueError(
                    f"{info}. Attribute {att} is {getattr(obj, att)}, and it must be in [{opt}]...."
                )


def attribute_setter(obj, **kwargs):
    """Set attributes passed in a dict-way."""
    for k, v in kwargs.items():
        setattr(obj, k, v)
