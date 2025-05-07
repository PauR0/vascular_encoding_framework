import numpy as np
from sklearn.decomposition import PCA

from ..messages import *


def planar_coordinates(points, c0, v1, v2):
    """
    Compute planar coordinates of points.

    Given a set of 3D points and a plane defined by a center and two
    orthonormal basis vectors, compute the 2D coordinates of the points
    projected onto the plane.

    Parameters
    ----------
    points : array_like
        (3, N) array of 3D points, where N is the number of points.
    c0 : array_like
        (3,) array representing the center of the plane.
    v1 : array_like
        (3,) array representing the first orthonormal basis vector of the plane.
    v2 : array_like
        (3,) array representing the second orthonormal basis vector of the plane.

    Returns
    -------
    points2d : ndarray
        (2, N) array of 2D coordinates corresponding to the projected points.
    """
    M = np.array([v1.T, v2.T])
    points2d = np.dot(M, points - c0.reshape((3, 1)))
    return points2d


def polar_to_cart(x_pol):
    """
    Polar to cartesian 2D coordinates.

    Parameters
    ----------
    x_pol : np.ndarray, (2, N)
        Points in polar coordinates.

    Returns
    -------
    x_cart : np.ndarray (2,N)
        The points in cartesian 2D coordinates.
    """
    return x_pol[1] * np.array((np.cos(x_pol[0]), np.sin(x_pol[0])))


#


def cart_to_polar(x_cart, sort=True):
    """
    Cartesian 2D to polar coordinates with angular coord in [0, 2pi].

    Parameters
    ----------
    x_cart : np.ndarray (2, N)
        The array of points in Cartesian coordinates.
    sort : bool, opt.
        Default True. Whether to sort or not the points by its angular coord.

    Returns
    -------
    x_pol : np.ndarray
        Points in polar coordinates.
    """
    x_pol = np.array((np.arctan2(x_cart[1], x_cart[0]), np.linalg.norm(x_cart, axis=0)))
    x_pol[0][x_pol[0] < 0] += 2 * np.pi

    # Sorting the array points by the value in the first row!
    if sort:
        x_pol = x_pol[:, np.argsort(x_pol[0, :])]

    return x_pol


def get_theta_coord(points, c, v1, v2, deg=False):
    """
    Get the theta coordinate for a list of points in a cross
    section.

    Parameters
    ----------
    points : np.ndarray (3,) or (N, 3)
        The points belonging to the same cross section
    c, v1, v2 : np.ndarray (3,)
        The center, v1, and v2 of the cross section respectively.
    deg : bool, opt
        Default False. Whether to return theta coord in degrees instead of radians.
    """

    if len(points.shape) == 1:
        points = points[None, :]  # Adding a dimension

    u1, u2 = planar_coordinates(points.T, c0=c, v1=v1, v2=v2)
    th = np.arctan2(u2, u1)
    th[th < 0] += 2 * np.pi

    if deg:
        th = radians_to_degrees(r=th)

    if len(th) == 1:
        return th[0]

    return th


def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def compute_ref_from_points(points):
    """
    Compute a local reference frame by means of a principal component analysis.

    Parameters
    ----------
    points : array-like (N, 3)
        The point array.

    Returns
    -------
    center : np.array
        The average position of the points
    e1, e2, e3 : np.ndarray (3,)
        The vectors sorted by variance.

    """

    pca = PCA()
    pca.fit(points)
    center = pca.mean_
    e1 = pca.components_[0]
    e2 = pca.components_[1]
    e3 = pca.components_[2]

    return center, e1, e2, e3


def sort_glob_ids_by_angle(gids, points, c, v1, v2):
    """
    Sort global IDs based on the angular coordinate of corresponding points.

    Given a set of global IDs and their corresponding 3D points, sort the IDs
    based on the angular coordinate of the points in a plane defined by a center
    and two orthonormal basis vectors.

    Parameters
    ----------
    gids : array_like
        (N,) array of global IDs.
    points : array_like
        (3, N) array of 3D points corresponding to the global IDs.
    c : array_like
        (3,) array representing the center of the plane.
    v1 : array_like
        (3,) array representing the first orthonormal basis vector of the plane.
    v2 : array_like
        (3,) array representing the second orthonormal basis vector of the plane.

    Returns
    -------
    sorted_gids : ndarray
        (N,) array of global IDs sorted by the angular coordinate of the
        corresponding points.

    See Also
    --------
    get_theta_coord : Compute the angular coordinate of points in a plane.
    """
    if not isinstance(gids, np.ndarray):
        gids = np.array(gids)

    th = get_theta_coord(points, c, v1, v2)
    ids = th.argsort()
    return gids[ids]


def radians_to_degrees(r):
    """
    Convert from radians to degrees.

    Parameters
    ----------
    r : float or np.ndarray
        The radians.

    Returns
    -------
    deg : float or np.ndarray
        The degrees.

    """
    deg = 180 / np.pi * r
    return deg


def decompose_transformation_matrix(matrix):
    """
    Decompose a transformation matrix in its translation, scale and rotation components.

    Parameters
    ----------
    matrix: numpy.array (4,4)
        The transformation matrix to be decomposed.

    Returns
    -------
    t: numpy.array (3,1)
        The translation vector.
    s: numpy.array (3,1)
        The scale factor of each dimension.
    r: numpy.array (3,3)
        The rotation matrix.
    """

    # Translation
    t = matrix[0:-1, 3]
    # Scale
    s = np.linalg.norm(matrix[[0, 1, 2], :3], axis=0)
    # Rotation
    r = matrix[0:-1, 0:-1]
    for i in range(3):
        r[:, i] = r[:, i] * (1 / s[i])

    return t, s, r


def compose_transformation_matrix(t=None, s=None, r=None):
    """
    Build a transformation matrix based on a translation vector, a scale vector and a rotation matrix.

    Parameters
    ----------
    t : np.ndarray (3,), opt
        Default None. Translation vector.
    s : np.ndarray (3,), opt
        Default None. Scale vector.
    r : np.ndarray (3, 3)
        Default None. The rotation matrix.

    Returns
    -------
    matrix: numpy.array (4,4)
        The transformation matrix.

    """

    matrix = np.eye(4)

    if r is not None:
        matrix[0:-1, 0:-1] = r

    if s is not None:
        matrix[:3, :3] *= s

    if t is not None:
        matrix[0:-1, 3] = t

    return matrix


def transform_point_array(points, t=None, s=None, r=None):
    """
    Apply affine transformation to a numpy array of 3D points.

    Parameters
    ----------
    points : np.ndarray (n, 3)
        The array of 3D points.
    t : np.ndarray (3,), opt
        Default None. Translation vector.
    s : np.ndarray (3,), opt
        Default None. Scale vector.
    r : np.ndarray (3, 3)
        Default None. The rotation matrix.

    Returns
    -------
    points_tr : np.ndarray (n, 3)
        The transformed points
    """

    pts_ext = np.hstack([points, np.ones((points.shape[0], 1))])
    matrix = compose_transformation_matrix(t=t, s=s, r=r)
    points_tr = (matrix @ pts_ext.T).T
    points_tr = points_tr[:, :3]

    return points_tr
