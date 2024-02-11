"""
The utils submodule to store computational geometry stuff.

"""


import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA

from utils.spatial import normalize
import messages as msg


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

    #Append the center at the end of the point array
    points = np.vstack([cs.points, cs.center])
    faces  = np.full(shape=(cs.lines.shape[0]//3, 4), fill_value=3, dtype=int)
    faces[:, 1] = cs.n_points #The first id of all the triangles is the center
    faces[:, 2:] = cs.lines.reshape(-1, 3)[:, 1:]

    new_cs = pv.PolyData(points, faces)

    return new_cs
#
