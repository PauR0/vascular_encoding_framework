

import numpy as np

from sklearn.decomposition import PCA

from ..messages import *

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

def get_theta_coord(points, c, v1, v2, deg=False):
    """
    Get the theta coordinate for a list of points in a cross
    section.

    Arguments:
    ------------

        points : np.ndarray (3,) or (N, 3)
            The points belonging to the same cross section

        c, v1, v2 : np.ndarray (3,)
            The center, v1, and v2 of the cross section respectively.

        deg : bool, opt
            Default False. Whether to return theta coord in degrees instead of radians.
    """

    if len(points.shape) == 1:
        points = points[None, :] #Adding a dimension

    u1, u2 = planar_coordinates(points.T, c0=c, v1=v1, v2=v2)
    th = np.arctan2(u2,u1)
    th[th < 0] += 2*np.pi

    if deg:
        th = radians_to_degrees(r=th)

    if len(th) == 1:
        return th[0]

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
    """
    Compute a local reference frame by means of a principal component analysis.

    Arguments:
    -----------

        points : array-like (N, 3)
            The point array.

    Returns:
    ---------

        center : np.array
            The average position of the points

        e1, e2, e3 : np.ndarray (3,)
            The vectors sorted by variance.

    """

    if message is None:
        message='local PCA frame'

    computing_message(info=message)
    pca = PCA()
    pca.fit(points)
    center = pca.mean_
    e1 = pca.components_[0]
    e2 = pca.components_[1]
    e3 = pca.components_[2]
    done_message(info=message)

    return center, e1, e2, e3
#

def sort_glob_ids_by_angle(gids, points, c, v1, v2):

    if not isinstance(gids, np.ndarray):
        gids = np.array(gids)

    th = get_theta_coord(points, c, v1, v2)
    ids = th.argsort()
    return gids[ids]
#

def radians_to_degrees(r):
    """
    Convert from radians to degrees.

    Arguments:
    ----------

        r : float or np.ndarray
            The radians.

    Returns:
    ----------

        deg : float or np.ndarray
            The degrees.

    """
    deg = 180/np.pi * r
    return deg
#
