#! /usr/bin/env python3

import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA

import messages as msg

def planar_coordinates(points, c0, v1, v2):
    M = np.array([v1.T,v2.T])
    points2d = np.dot(M, points-c0.reshape((3,1)))
    return points2d
#

def polar_to_cart(x_pol):
    """
    Polar to cartesian 2D coordinates

    Arguments:
    -----------

        x_pol : np.ndarray, (2, N)
            Points in polar coordinates.

    Returns:
    ---------

        x_cart : np.ndarray (2,N)
            The points in cartesian 2D coordinates.
    """
    return x_pol[1]*np.array((np.cos(x_pol[0]),np.sin(x_pol[0])))
#

def cart_to_polar(x_cart, sort=True):
    """
    Cartesian 2D to polar coordinates with angular coord in [0, 2\pi]

    Arguments:
    ------------

        x_cart : np.ndarray (2, N)
            The array of points in Cartesian coordinates.

        sort : bool, opt.
            Default True. Whether to sort or not the points by its angular coord.

    Returns:
    ---------

        x_pol : np.ndarray
            Points in polar coordinates.
    """
    x_pol = np.array( (np.arctan2(x_cart[1], x_cart[0]),
                       np.linalg.norm(x_cart,axis=0)) )
    x_pol[0][x_pol[0]<0] += 2*np.pi

    #Sorting the array points by the value in the first row!
    if sort:
        x_pol = x_pol[:,np.argsort(x_pol[0,:])]

    return x_pol
#

def get_theta_coord(points, c, v1, v2):

    u1, u2 = planar_coordinates(points.T, c0=c, v1=v1, v2=v2)
    th = np.arctan2(u2,u1)
    th[th < 0] += 2*np.pi
    return th
#

def normalize(v):
    """ Normalize a vector """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
#

def compute_ref_from_points(points, message=None):
    """ Compute a local reference frame by means of a principal component analysis.
    """

    if message is None:
        message='local PCA frame'

    msg.computing_message(task=message)
    pca = PCA()
    pca.fit(points)
    center = pca.mean_
    e1 = pca.components_[0]
    e2 = pca.components_[1]
    e3 = pca.components_[2]
    msg.done_message(task=message)

    return center, e1, e2, e3
#

def sort_glob_ids_by_angle(gids, points, c, v1, v2):

    if not isinstance(gids, np.ndarray):
        gids = np.array(gids)

    th = get_theta_coord(points, c, v1, v2)
    ids = th.argsort()
    return gids[ids]
#

def triangulate_cross_section(cross_section, method='connected', n=None):
    """
    This function triangulare a cross section list of points.

    Arguments:
    -----------

        cs : pv.PolyData
            The object containing the cross section to be triangulated.

        method : str {'connected', 'unconnected'}

        n : np.ndarray, optional
            The normal of the best fitting plane. If not provided, a PCA is
            used to compute it as the component of least variance.

    """

    new_cs=None
    if method == 'connected':
        new_cs = triangulate_connected_cross_section(cs=cross_section)

    elif method == 'unconnected':
        new_cs = triangulate_unconnected_cross_section(cs=cross_section, n=n)
    else:
        msg.error_message(f"Cannot triangulate cross section with unknown method {method}."+\
                            "Available options are {'connected', 'unconnected'}.")

    return new_cs
#

def triangulate_unconnected_cross_section(cs, n=None):
    """
    This function builds the triangulation of a cross section PolyData
    without requiring lines connecting the points. It assumes points can be
    projected onto the best fitting plane and ordered by a given angular origin
    arbitrarily defined.

    Caveats: There are no warranties that the triangulation
    has the same orientation as the mesh the cross section was extracted from.

    Arguments:
    -----------

        cs : pv.PolyData (or simmilar)
            The object containing the attributes points and n_points of the
            cross section to be triangulated.

        n : np.ndarray, optional
            The normal of the best fitting plane. If not provided, a PCA is
            used to compute it as the component of least variance.

    Returns:
    ---------

        new_cs : pv.PolyData
            The PolyData containing the points of the given cross section
            with the faces defined.

    """

    if n is None:
        pca = PCA(n_components=3).fit(cs.points)
        n = normalize(pca.components_[2])
    else:
        n = normalize(n)

    i = 0
    E = np.eye(3)
    while np.cross(n, E[i]).sum() == 0:
        i += 1

    v1 = normalize(np.cross(n, E[i]))
    v2 = normalize(np.cross(n, v1))

    n_pts_range = np.arange(cs.n_points, dtype=int)
    sorted_ids = sort_glob_ids_by_angle(n_pts_range,
                                        cs.points,
                                        np.array(cs.center),
                                        v1, v2)

    sorted_points = cs.points[sorted_ids]
    faces = np.array([cs.n_points] + list(range(cs.n_points)), dtype=int)

    new_cs = pv.PolyData(sorted_points, faces=faces).triangulate()
    return new_cs
#

def triangulate_connected_cross_section(cs):
    """
    Triangulate a cross section using the lines defining it to
    build triangle faces using the center of de cs. This way
    geometrical orientation should be preserved.

    Arguments:
    ------------

        cs : pv.PolyData
            The cross section with topology defined by using lines.

    Returns:
    ----------
        new_cs : pv.Polydata
            The new triangulated cross section
    """

    if cs.lines.size == 0:
        msg.error_message("Cannot triangulate cross section with connected method."+\
                          "The given cross section has no lines. Try unconnected method instead...")

    new_cs = cs.copy(deep=True)
    new_cs.points = np.vstack([cs.points, cs.center])
    lines = cs.lines.reshape(-1, 3)
    lines[:, 0] = cs.n_points-1
    faces = np.full(shape=(lines.shape[0], 4), fill_value=4, dtype=int)
    faces[:, 1:] = lines
    cs.faces = faces
    return cs
#

def knots_list(s0, s1, n, mode='complete', ext=None, k=3):
    """ Generates a B-Spline uniform list of knots

        Given s0 and s1 it returns a numpy array of nodes for a B-Spline

        Parameters
        ----------

        s0 : float
            Starting parameter value

        s1 : float
            Ending parameter value

        n : int
            Number of B-Spline knots from s0 to s1.

        ext : int
            Defaulting to k+1. The times to extend the knot vector from
            both ends preserving the separation between nodes.

        k : int
            The degree of the spline.

        mode : {'simple','complete', 'extended', 'periodic'} , optional
            If mode='simple' then l is a numpy linspace from s0 to s1.

            If mode='complete' l contains [s0,s0,s0] at the beginning and
            [s1,s1,s1] at the end. Needed for scipy.interpolation.BSpline.

            If mode = 'extended' and ext is not none, it extends ext times the
            knot vector from both ends preserving the spacing.

            If mode = 'periodic' then the list of knots is ready for periodic
            B-Splines fitted with splrep. It is the same as setting mode='extended'
            and ext=k+1.

        Output
        ------
            l: the list of knots
    """
    l = np.linspace(s0,s1,n+2)
    d = (s1-s0)/(n+1)

    if mode == 'periodic':
        mode = 'extended'
        ext = k+1

    if mode == 'simple':
        l = l[1:-1]
    elif mode == 'extended':
        if ext is None:
            ext = k+1
        l = np.concatenate([l[0] + np.arange(-ext, 0)*d, l, l[-1] + np.arange(ext+1)[1:]*d])[1:-1]

    else:
        initial  = np.array( [s0]*3 )
        terminal = np.array( [s1]*3 )
        l = np.concatenate((initial,l,terminal))

    return l
#

def attribute_checker(obj, atts, extra_info=''):
    """
    Function to check if attribute has been set and print error message.

    Arguments:
    ------------

        obj : any,
            The object the attributes of which will be checked.

        atts : list[str]
            The names of the attributes to be checked for.

    Returns:
    --------
        True if all the attributes are different to None. False otherwise.
    """


    for att in atts:
        if getattr(obj, att) is None:
            msg.error_message(info=f"{extra_info}. Attribute {att} is None....")
            return False

    return True
#

def attribute_setter(obj, **kwargs):
    """
    Function to set attributes passed in a dict-way.
    """
    for k, v in kwargs.items():
        setattr(obj, k, v)
#
