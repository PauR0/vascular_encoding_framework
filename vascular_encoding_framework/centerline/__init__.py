from __future__ import annotations

__all__ = [
    "Centerline",
    "CenterlineTree",
    "ParallelTransport",
    "extract_centerline",
    "Seekers",
    "Flux",
    "extract_centerline_domain",
    "CenterlinePathExtractor",
    "extract_centerline_path",
]
from typing import Literal as _Literal

from .centerline import Centerline
from .centerline_tree import CenterlineTree
from .domain_extractors import Flux, Seekers, extract_centerline_domain
from .parallel_transport import ParallelTransport
from .path_extractor import CenterlinePathExtractor, extract_centerline_path


def extract_centerline(
    vmesh,
    n_knots: int = 10,
    curvature_penatly: float = 1.0,
    graft_rate: float = 0.5,
    force_extremes: bool | _Literal["ini", "end"] = True,
    pt_mode="project",
    p=None,
    params_domain=None,
    params_path=None,
    debug=False,
    **kwargs,
) -> CenterlineTree:
    """
    Compute the CenterlineTree of a provided a VascularMesh object with properly defined boundaries.

    Parameters
    ----------
    vmesh : VascularMesh
        The VascularMesh object where centerline is to be computed.
    n_knots : int
        The number of knots to perform the fitting. To use a specific value per branch, read
        the kwargs documentation.
    graft_rate : float, opt
        Default is 0.5. A parameter to control the grafting insertion. Represent a distance
        proportional to the radius traveled towards the parent branch inlet along the centerline
        at the junction. To use a specific value per branch, read the kwargs documentation.
    force_extremes : {False, True, 'ini', 'end'}
        Default True. Whether to force the centerline to interpolate the boundary behavior
        of the approximation. If True the first and last point are interpolated and its
        tangent is approximated by finite differences using the surrounding points. If 'ini',
        respectively 'end', only one of both extremes is forced.
    params_domain : dict, opt
        The parameters for the domain extraction algorithm. More information about it in the
        domain_extractors module.
    params_path : dict
        The parameters for the path extraction algorithm. More information about it in the
        path_extractor module.
    debug : bool, opt
        Defaulting to False. Running in debug mode shows some plots at certain steps.
    **kwargs : dict
        The above described arguments can be provided per branch using the kwargs. Say there
        exist a path with id AUX in the passed multiblock, to set specific parameters for the
        branch AUX, one can pass the dictionary AUX={n_knots:20}, setting the number of knots to
        20 and assuming the default values for the rest of the parameters.


    Returns
    -------
    cl_tree : CenterlineTree
        The computed Centerline
    """

    cl_domain = extract_centerline_domain(vmesh=vmesh, params=params_domain, debug=debug)
    cl_paths = extract_centerline_path(vmesh=vmesh, cl_domain=cl_domain, params=params_path)
    cl_tree = CenterlineTree.from_multiblock_paths(
        cl_paths,
        n_knots=n_knots,
        curvature_penatly=curvature_penatly,
        graft_rate=graft_rate,
        force_extremes=force_extremes,
        pt_mode=pt_mode,
        p=p,
        **kwargs,
    )
    return cl_tree
