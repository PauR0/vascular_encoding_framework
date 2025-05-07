"""The utils submodule to store computational geometry stuff."""

from typing import Literal

import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA

from ..messages import *
from .spatial import normalize, sort_glob_ids_by_angle


def approximate_cross_section(
    point: np.ndarray,
    mesh: pv.PolyData,
    theta_res: int = 30,
    phi_res: int = 30,
    n0: np.ndarray = None,
    max_d: float = None,
    min_perim: float = None,
    debug: bool = False,
):
    """
    Approximate the cross section of a point in the lumen of a vascular mesh.

    This function computes the cross section as the plane-mesh intersection whose perimeter/area is minimal.
    The intersection optimization is approximated using a discrete amount of possible planes, the resolution can
    be controlled by means of the theta_res and phi_res arguments.


    Parameters
    ----------
    point : np.ndarray (3,)
        The point to compute the cross section
    mesh : pv.PolyData
        The vascular mesh.
    theta_res, phi_res : int, opt
        The resolution in spherical coordinates for the testing directions.
    n0 : np.ndarray (3,)
        Default (0, 0, 1). A normalized direction to filter the sphere of
        normals.
    max_d : float, opt
        The maximum distance allowed between the provided center and the
        center of the computed section.
    min_perim : float, opt
        Default None. A minimum threshold for the available perimeters. It
        can be used to filter out small intersections. If the point provided
        is near the centerline, and let d be the distance between point and mesh,
        a good value for min_perim is (d/2)*2*pi = d*pi, which is the perimeter of
        the circumference with radius half the distance from point to mesh.

    Returns
    -------
    cs_opt : pv.PolyData
        The computed cross section. It is triangulated using the center.
    """

    def perimeter(n):
        cs = extract_section(mesh=mesh, normal=n, origin=point, min_perim=min_perim)

        pts, _ = mesh.ray_trace(point, cs.center, first_point=True)
        if pts.shape[0] > 0:
            return np.inf

        if max_d is not None:
            if np.linalg.norm(point - np.array(cs.center)) > max_d:
                return np.inf

        cs = cs.compute_cell_sizes(length=True, area=False, volume=False)
        return cs["Length"].sum()

    if n0 is None:
        # Testing only on half the sphere, due to periodicity
        n0 = np.array([0, 0, 1])

    normals = pv.Sphere(theta_resolution=theta_res, phi_resolution=phi_res)
    aligned = normals.points.dot(n0) >= -0.0
    normals = normals.extract_points(aligned, adjacent_cells=True)
    perimeters = np.array([perimeter(n) for n in normals.points])
    perimeters[perimeters <= 0.0] = np.inf
    id_opt = np.argmin(perimeters)
    n_opt = normals.points[id_opt]

    cs_opt = extract_section(mesh=mesh, normal=n_opt, origin=point, triangulate=True)

    if debug:
        arrow = pv.Arrow(start=point, direction=n_opt)
        p = pv.Plotter()
        p.add_mesh(mesh, opacity=0.4)
        p.add_mesh(mesh.slice(normal=n_opt, origin=cs_opt.center), color="r")
        p.add_mesh(cs_opt, color="g")
        p.add_mesh(point, color="g", render_points_as_spheres=True, point_size=10)
        p.add_mesh(arrow)
        p.show()

    return cs_opt


def extract_section(
    mesh: pv.PolyData,
    normal: np.ndarray,
    origin: np.ndarray,
    min_perim: float = None,
    triangulate: bool = False,
):
    """
    Given a vascular mesh, a normal and an origin this method extracts the closest component of
    the intersection of the mesh and the plane with provided normal and origin. A minimum perimeter can be
    provided to discard spurious small intersections.

    From each connected component (cc) of the mesh-plane intersection, we use ray-trace to compute the amount of
    intersections between the cc and the segment [origin - cc.center]. The cross section with no intersection
    should be the sought one.

    Parameters
    ----------
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

    Returns
    -------
        sect : pv.Polydata
            The section extracted.
    """

    sect = mesh.slice(normal=normal, origin=origin)
    sect = sect.connectivity()
    conn_comps = [
        sect.extract_points(sect.get_array("RegionId", preference="point") == comp)
        for comp in np.unique(sect["RegionId"])
    ]
    comp_centers = np.array([cn.center for cn in conn_comps])
    dist = np.linalg.norm(comp_centers - origin, axis=1)
    for i, comp in enumerate(conn_comps):
        intersect_points, _ = mesh.ray_trace(origin, comp_centers[i])
        if intersect_points.shape[0] > 0:
            dist[i] = np.inf
        if min_perim is not None:
            comp = comp.compute_cell_sizes()
            if comp["Length"].sum() < min_perim:
                dist[i] = np.inf
    selected_comp = dist.argmin()
    sect = conn_comps[selected_comp].extract_surface()

    if triangulate:
        sect = triangulate_cross_section(sect)

    return sect


def triangulate_cross_section(
    cross_section: pv.PolyData,
    method: Literal["connected", "unconnected"] = "connected",
    n: np.ndarray = None,
):
    """
    Triangulate a cross section list of points.

    Parameters
    ----------
    cs : pv.PolyData
        The object containing the cross section to be triangulated.
    method : str {'connected', 'unconnected'}, optional
        If 'connected' the points are assumed to be connected by lines, and the topology
        is preserved, if 'unconnected' the topology is inferred by polar coordinates.
    n : np.ndarray, optional
        The normal of the best fitting plane. If not provided, a PCA is
        used to compute it as the component of least variance.

    """

    new_cs = None
    if method == "connected":
        new_cs = triangulate_connected_cross_section(cs=cross_section)

    elif method == "unconnected":
        new_cs = triangulate_unconnected_cross_section(cs=cross_section, n=n)
    else:
        error_message(
            f"Cannot triangulate cross section with unknown method {method}."
            + "Available options are {'connected', 'unconnected'}."
        )

    return new_cs


def triangulate_unconnected_cross_section(cs, n=None):
    """
    Build the triangulation of a cross section PolyData without requiring any line connecting
    the points.

    It assumes points can be projected onto the best fitting plane and ordered by
    a given angular origin arbitrarily defined.

    Caveats: There is no warranty that the triangulation has the same orientation as the mesh
    the cross section is extracted from.

    Parameters
    ----------
    cs : pv.PolyData (or similar)
        The object containing the attributes points and n_points of the
        cross section to be triangulated.
    n : np.ndarray, optional
        The normal of the best fitting plane. If not provided, a PCA is
        used to compute it as the component of least variance.

    Returns
    -------
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
    sorted_ids = sort_glob_ids_by_angle(n_pts_range, cs.points, np.array(cs.center), v1, v2)

    sorted_points = cs.points[sorted_ids]
    faces = np.array([cs.n_points] + list(range(cs.n_points)), dtype=int)

    new_cs = pv.PolyData(sorted_points, faces=faces).triangulate()
    return new_cs


def triangulate_connected_cross_section(cs):
    """
    Triangulate a cross section using the lines defining it to
    build triangle faces using the center of de cs. This way
    geometrical orientation should be preserved.

    Parameters
    ----------
    cs : pv.PolyData
        The cross section with topology defined by using lines.

    Returns
    -------
    new_cs : pv.Polydata
        The new triangulated cross section
    """

    if cs.lines.size == 0:
        error_message(
            "Cannot triangulate cross section with connected method."
            + "The given cross section has no lines. Try unconnected method instead..."
        )

    # Append the center at the end of the point array
    points = np.vstack([cs.points, cs.center])
    faces = np.full(shape=(cs.lines.shape[0] // 3, 4), fill_value=3, dtype=int)
    # The first id of all the triangles is the center
    faces[:, 1] = cs.n_points
    faces[:, 2:] = cs.lines.reshape(-1, 3)[:, 1:]

    new_cs = pv.PolyData(points, faces)

    return new_cs


def polyline_from_points(points):
    """
    Make a pyvista PolyData consisting of points and lines joining them.

    Parameters
    ----------
    points : array-like
        The list of points.

    Returns
    -------
    pline : pv.PolyData
        The polyline.
    """

    if not isinstance(points, np.ndarray):
        points = np.array(points)

    pline = pv.PolyData(points)
    pline.lines = np.array([(2, i, i + 1) for i in range(pline.n_points - 1)], dtype=int).ravel()
    return pline
