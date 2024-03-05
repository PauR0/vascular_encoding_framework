"""
The utils submodule to store computational geometry stuff.

"""


import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA

from utils.spatial import normalize
import messages as msg


def extract_section(mesh, normal, origin, min_perim=None, triangulate=False):
    """
    Given a vascular mesh, a normal and an origin this method extracts the clossest component of
    the intersection of the mesh and the plane with provided normal and origin. A minimum perimeter can be
    provided to discard spurious small intersections.

    From each connected component (cc) of the mesh-plane intersection, we use ray-trace to compute the amount of
    intersections between the cc and the segment [origin - cc.center]. The cross section with no intersection
    should be the sought one.

    Arguments:
    ------------

        mesh : pv.PolyData
            The vascular mesh domain.

        normal : np.ndarray (3,)
            The normal of the plane

        origin : np.ndarray (3,)
            The origin of the plane

        min_perim : float, opt
            Default None. A inferior perimeter threshold to remove small spurious intersections.

        triangulate : bool, opt
            Default False. Whether to triangulate the resulting section.

    Returns:
    ---------

        sect : pv.Polydata
            The section extracted.
    """

    sect = mesh.slice(normal=normal, origin=origin)
    sect = sect.connectivity()
    conn_comps = [sect.extract_points(sect.get_array('RegionId', preference='point') == comp) for comp in np.unique(sect['RegionId'])]
    comp_centers = np.array([cn.center for cn in conn_comps])
    dist = np.linalg.norm(comp_centers - origin, axis=1)
    for i, comp in enumerate(conn_comps):
        intersect_points, _ = mesh.ray_trace(origin, comp_centers[i])
        if intersect_points.shape[0] > 0:
            dist[i] = np.inf
        if min_perim is not None:
            comp = comp.compute_cell_sizes()
            if comp['Length'].sum() < min_perim:
                dist[i] = np.inf
    selected_comp = dist.argmin()
    sect = conn_comps[selected_comp].extract_surface()

    if triangulate:
        sect = triangulate_cross_section(sect)

    return sect
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

    #Append the center at the end of the point array
    points = np.vstack([cs.points, cs.center])
    faces  = np.full(shape=(cs.lines.shape[0]//3, 4), fill_value=3, dtype=int)
    faces[:, 1] = cs.n_points #The first id of all the triangles is the center
    faces[:, 2:] = cs.lines.reshape(-1, 3)[:, 1:]

    new_cs = pv.PolyData(points, faces)

    return new_cs
#

def polyline_from_points(points):
    """
    Make a pyvista PolyData consisting of points and
    lines joining them.

    Arguments:
    ------------

        points : array-like
            The list of points.

    Returns:
    ---------

        pline : pv.PolyData
            The polyline.
    """

    if not isinstance(points, np.ndarray):
        points = np.array(points)

    pline = pv.PolyData(points)
    pline.lines = np.array([(2, i, i+1) for i in range(pline.n_points-1)], dtype=int).ravel()
    return pline
#
