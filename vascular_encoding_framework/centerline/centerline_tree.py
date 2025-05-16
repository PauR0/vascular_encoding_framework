from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pyvista as pv

from .._base import EncodingTree, SpatialObject, check_specific
from ..utils.spatial import normalize, radians_to_degrees
from .centerline import Centerline
from .domain_extractors import extract_centerline_domain
from .parallel_transport import ParallelTransport
from .path_extractor import extract_centerline_path


class CenterlineTree(EncodingTree[Centerline], SpatialObject):
    """Class for the centerline of branched vascular geometries."""

    def __init__(self):
        EncodingTree.__init__(self=self, _node_type=Centerline)

    def __setitem__(self, __key, cl: Centerline) -> None:
        """
        Set items as in dictionaries. However, to belong to a CenterlineTree
        requires consistency in the adapted frames.
        """

        if cl.parent is not None and cl._spl is not None:
            cl.set_data(join_t=self[cl.parent].get_projection_parameter(cl(cl.t0), method="scalar"))
            v1 = ParallelTransport.parallel_rotation(
                t0=self[cl.parent].get_tangent(cl.join_t),
                t1=cl.get_tangent(cl.t0),
                v=self[cl.parent].v1(cl.join_t),
            )
            cl.compute_adapted_frame(p=v1, mode="as_is")

        super().__setitem__(__key, cl)

    def get_centerline_association(self, p, n=None, method="scalar", thrs=30):
        """
        Compute the centerline association of a point in space.

        If no normal is None, the branch is decided based on the distance to a rough approximation
        on the point projection. If n is provided, let q the projection of p onto the nearest
        centerline branch, if the angles between vectors q2p and n are greater than _thrs_, the next
        nearest branch will be tested. If non satisfy the criteria, a warning message will be output
        and the point will be assigned to the nearest branch.

        Warning: normal is expected to be used as the surface normal of a point. However, normals
        are sensible to high frequency noise in the mesh, try smoothing it before using the normals
        in the computation of the centerline association.

        Parameters
        ----------
        p : np.ndarray
            The point in space
        n : np.ndarray, opt
            Default is None. The normal of the points. Specially useful to preserve
            topology
        method : Literal {'scalar', 'vec', 'jac-vec', 'sample'}
            Default scalar. The method use to compute the projection.
            Note: 'sample' method is the fastest, but the least accurate.
        thrs : float, opt
            Default is 30. The maximum angle (in degrees) allowed between q2p and n.

        Returns
        -------
        bid : str
            The branch id.
        """

        ids, dists, angles = [], [], []
        for cid, cl in self.items():
            q, _, d = cl.get_projection_point(p, method=method, full_output=True)
            ids.append(cid)
            dists.append(d)
            if n is not None:
                q2p = normalize(p - q)
                angles.append((np.arccos(n.dot(q2p))))

        min_i = np.argmin(dists)
        minid = ids[min_i]

        if n is None:
            return minid

        angles = radians_to_degrees(np.array(angles)).tolist()
        while ids:
            i = np.argmin([dists])
            if angles[i] < thrs:
                minid = ids[i]
                break
            _, _, _ = ids.pop(i), dists.pop(i), angles.pop(i)

        return minid

    def get_projection_parameter(
        self, p, cl_id=None, n=None, method="scalar", thrs=30, full_output=False
    ):
        """
        Get the parameter of the projection onto the centerline tree.
        If centerline id (cl_id) argument is not provided it is computed
        using get_centerline_association.

        Parameters
        ----------
        p : np.ndarray (3,)
            The 3D point.
        cl_id : str, opt
            Default None. The id of the centerline of the tree to project
            the point. If None, it is computed using get_centerline_association
            method.
        n : np.ndarray, opt
            Default None. A normal direction at the point, useful if the point
            belongs to the surface of the vascular domain, its normal can be used.
        method : Literal {'scalar', 'vec', 'jac-vec', 'sample'}
            The method use to compute the projection.
        full_output : bool
            Whether to return the distance and centerline membership with the parameter
            or not. Default is False.

        Returns
        -------
        t : float
            The value of the parameter.
        d : float, opt
            The distance from p to the closest point in the centerline
        cl_id : str
            The id of the centerline it belongs to

        """

        if cl_id is None:
            cl_id = self.get_centerline_association(p, n=n, thrs=thrs)

        t, d = self[cl_id].get_projection_parameter(p=p, method=method)

        if full_output:
            return t, d, cl_id

        return t

    def get_projection_point(
        self, p, cl_id=None, n=None, method="scalar", thrs=30, full_output=False
    ):
        """
        Get the point projection onto the centerline tree.
        If centerline id (cl_id) argument is not provided it is computed
        using get_centerline_association.

        Parameters
        ----------
        p : np.ndarray (3,)
            The 3D point.
        cl_id : str, opt
            Default None. The id of the centerline in the tree to project the point.  If None,
            it is computed using get_centerline_association method.
        n : np.ndarray, opt
            Default None. A normal direction at the point, useful if the point belongs to the
            surface of the vascular domain, its normal can be used.
        method : Literal {'scalar', 'vec', 'jac-vec', 'sample'}
            The method use to compute the projection.
        full_output : bool
            Whether to return the parameter value, distance and the centerline association or
            not. Default is False.

        Returns
        -------
        p : np.ndarray (3,)
            The projection of the point in the centerline
        t : float, opt
            The value of the parameter.
        d : float, opt
            The distance from p to the closest point in the centerline
        cl_id : str, opt
            The id of the centerline it belongs to
        """

        if cl_id is None:
            cl_id = self.get_centerline_association(p, n=n, thrs=thrs)

        p, t, d = self[cl_id].get_projection_point(p=p, method=method, full_output=True)

        if full_output:
            return p, t, cl_id, d

        return p

    def cartesian_to_vcs(self, p, cl_id=None, n=None, method="scalar", thrs=30, full_output=False):
        """
        Given a 3D point p expressed in cartesian coordinates, this method computes its expression
        in the Vessel Coordinate System (VCS) of the centerline it has been associated to.

        Parameters
        ----------
        p : np.ndarray (3,)
            The 3D point.
        cl_id : str, opt
            Default None. The id of the centerline in the tree to project the point. If None, it is
            computed using get_centerline_association method.
        n : np.ndarray, opt
            Default None. A normal direction at the point, useful if the point belongs to the
            surface of the vascular domain, its normal can be used.
        method : Literal {'scalar', 'vec', 'jac-vec', 'sample'}
            The method use to compute the projection.
        full_output : bool, opt
            Default False. Whether to add the cl_id to the returns.

        Returns
        -------
        p_vcs : np.ndarray (3,)
            The (tau, theta, rho) coordinates of the given point.
        cl_id : str, opt
            The id of the centerline the point has been associated to.

        """

        if cl_id is None:
            cl_id = self.get_centerline_association(p=p, n=n, method=method, thrs=thrs)

        if full_output:
            return self[cl_id].cartesian_to_vcs(p=p), cl_id

        return self[cl_id].cartesian_to_vcs(p=p)

    def plot_adapted_frame(
        self,
        vmesh: pv.PolyData = None,
        plotter: pv.Plotter = None,
        scale: float = None,
        show: bool = True,
    ):
        """
        Plot the parallel transported frame.

        Parameters
        ----------
        vmesh : VascularMesh or pv.PolyData
            The vascular mesh used to compute the centerline.
        plotter : pv.Plotter
            Default None. If passed, parallel_transport is displayed there.
        scale : float, opt
            By default no scale is applied. The scale of the arrows used to plot the adapted frame
            vectors.
        show : bool, opt
            Default True. Whether to show the plot or not.
        """

        if plotter is None:
            plotter = pv.Plotter()

        for i, cl in enumerate(self.values()):
            if i == len(self) - 1:
                cl.plot_adapted_frame(vmesh=vmesh, plotter=plotter, scale=scale, show=show)
            else:
                cl.plot_adapted_frame(plotter=plotter, scale=scale, show=False)

    def to_multiblock(self, add_attributes: bool = False) -> pv.MultiBlock:
        """
        Return a pyvista MultiBlock with the centerline branches as pyvista PolyData objects.

        Parameters
        ----------
        add_attributes : bool, opt
            Default False. Whether to add all the required attributes to built the
            CenterlineTree back.

        Returns
        -------
        mb : pv.MultiBlock
            The multiblock with the polydata paths.

        See Also
        --------
        Centerline.to_polydata
        """

        mb = pv.MultiBlock()
        for i, cl in self.items():
            mb[i] = cl.to_polydata(tau_res=None, add_attributes=add_attributes)
        return mb

    @staticmethod
    def from_multiblock(mb: pv.MultiBlock):
        """
        Make a CenterlineTree object from a pyvista MultiBlock made polydatas.

        As the counterpart of to_multiblock, this static method is meant for building CenterlineTree
        objects from a pyvista MultiBlock, where each element of the MultiBlock is a PolyData
        with the information required to build the Tree structure and the Spline information.

        Parameters
        ----------
        mb : pv.MultiBlock
            The multiblock containing the required data.

        Returns
        -------
        cl_tree : CenterlineTree
            The centerline tree extracted from the passed MultiBlock.
        """

        if not mb.is_all_polydata:
            raise ValueError(
                "Can't make CenterlineTree. Some elements of the MultiBlock are not PolyData type."
            )

        cl_dict = {cid: Centerline().from_polydata(poly=mb[cid]) for cid in mb.keys()}
        roots = [cid for cid, cl in cl_dict.items() if cl.parent in [None, "None"]]

        cl_tree = CenterlineTree()

        def add_to_tree(i):
            cl_tree[i] = cl_dict[i]
            for chid in cl_dict[i].children:
                add_to_tree(chid)

        for rid in roots:
            add_to_tree(rid)

        return cl_tree

    @staticmethod
    def from_multiblock_paths(
        paths: pv.MultiBlock,
        n_knots: int = 10,
        curvature_penatly: float = 1.0,
        graft_rate: float = 0.5,
        force_extremes: bool | Literal["ini", "end"] = True,
        pt_mode="project",
        p=None,
        **kwargs,
    ) -> CenterlineTree:
        """
        Create a CenterlineTree from a pyvista MultiBlock made polydatas with
        points joined by lines, basically like the output of CenterlinePathExtractor.


        Each polydata must have the corresponding metadata as user_dict.

        Parameters
        ----------
        paths : pv.MultiBlock
            The multiblock containing the centerline paths. All the elements in paths must be of
            PolyData type. Their block name will be interpreted as their ID. Each of these polydatas
            must have the centerline metadata defined as user_dict.
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
        **kwargs : dict
            The above described arguments can be provided per branch using the kwargs. Say there
            exist a path with id AUX in the passed multiblock, to set specific parameters for the
            branch AUX, one can pass the dictionary AUX={n_knots:20}, setting the number of knots to
            20 and assuming the default values for the rest of the parameters.

        Returns
        -------
        cl_tree : CenterlineTree
            The centerline tree extracted from the passed MultiBlock.
        """

        if not paths.is_all_polydata:
            raise ValueError(
                "Can't make CenterlineTree. Some elements of the MultiBlock are not PolyData type "
            )

        cl_tree = CenterlineTree()

        cl_ids = paths.keys()
        parents = {i: paths[i].user_dict["parent"] for i in paths.keys()}

        def add_to_tree(nid):
            nonlocal n_knots, force_extremes, curvature_penatly, graft_rate

            points = paths[nid].points
            if parents[nid] is not None:
                pcl = cl_tree[parents[nid]]
                pre_joint = paths[nid].points[0]
                pre_tau_joint = pcl.get_projection_parameter(pre_joint)
                gr = check_specific(kwargs, nid, "graft_rate", graft_rate)
                if gr:
                    tau_joint = pcl.travel_distance_parameter(
                        d=-paths[nid]["radius"][0] * gr, a=pre_tau_joint
                    )
                    joint = pcl(tau_joint)
                    ids = np.linalg.norm(points - joint, axis=1) > paths[nid]["radius"][0] * gr
                    points = np.concatenate(
                        [
                            [joint, pcl((tau_joint + pre_tau_joint) / 2)],
                            paths[nid].points[ids],
                        ]
                    )
                else:
                    tau_joint = pre_tau_joint

            cl = Centerline.from_points(
                points,
                n_knots=check_specific(kwargs, nid, "n_knots", n_knots),
                force_extremes=check_specific(kwargs, nid, "force_extremes", force_extremes),
                curvature_penalty=check_specific(
                    kwargs, nid, "curvature_penalty", curvature_penatly
                ),
                pt_mode=check_specific(kwargs, nid, "pt_mode", pt_mode),
                p=check_specific(kwargs, nid, "p", p),
            )

            cl.id = nid
            if parents[nid] is not None:
                cl.parent = parents[nid]
                cl.tau_joint = tau_joint
            cl_tree[nid] = cl

            for cid in cl_ids:
                if parents[cid] == nid:
                    add_to_tree(cid)

        for rid in paths.keys():
            if parents[rid] is None:
                add_to_tree(rid)

        return cl_tree

    def from_feature_vector(self, fv: np.ndarray, hp: dict[str, Any] = None) -> CenterlineTree:
        """
        Build an CenterlineTree object from a feature vector.

        > Note that while hyperparameters argument is optional it must have been previously set or
        passed.

        Parameters
        ----------
        fv : np.ndarray
            The centelrine tree feature vector.
        hp : dict[str, Any], optional
            The hyperparameter dictionary for the  centerline tree object.

        Returns
        -------
        self : CenterlineTree
            The object itself with the elements updated/built with the fv.

        See Also
        --------
        get_hyperparameters
        set_hyperparameters
        to_feature_vector
        """
        return super().from_feature_vector(fv=fv, hp=hp)

    def translate(self, t):
        """
        Translate the CenterlineTree object, translating all the Centerline objects, with the
        translation vector t.

        Parameters
        ----------
        t : np.ndarray (3,)
            The translation vector.

        See Also
        --------
        Centerline.translate
        """

        for _, cl in self.items():
            cl.translate(t)

    def scale(self, s):
        """
        Scale the CenterlineTree object, scaling all the Centerline objects, by a scalar factor s.

        Parameters
        ----------
        s : float
            The scale factor.

        See Also
        --------
        Centerline.scale
        """

        for _, cl in self.items():
            cl.scale(s)

    def rotate(self, r):
        """
        Rotate the CenterlineTree, rotating all the Centerline objects, with the provided rotation
        matrix r.

        Parameters
        ----------
        r : np.ndarray (3, 3)
            The rotation matrix.

        See Also
        --------
        Centerline.rotate
        """

        for _, cl in self.items():
            cl.rotate(r)


def extract_centerline(
    vmesh, params, params_domain=None, params_path=None, debug=False
) -> CenterlineTree:
    """
    Compute the CenterlineTree of a provided a VascularMesh object with properly defined boundaries.

    Parameters
    ----------
    vmesh : VascularMesh
        The VascularMesh object where centerline is to be computed.
    params : dict
        The parameters for the spline approximation for each boundary, together with the grafting
        rate and tangent forcing parameters.
    params_domain : dict, opt
        The parameters for the domain extraction algorithm. More information about it in the
        domain_extractors module.
    params_path : dict
        The parameters for the path extraction algorithm. More information about it in the
        path_extractor module.
    debug : bool, opt
        Defaulting to False. Running in debug mode shows some plots at certain steps.


    Returns
    -------
    cl_tree : CenterlineTree
        The computed Centerline
    """

    cl_domain = extract_centerline_domain(vmesh=vmesh, params=params_domain, debug=debug)
    cl_paths = extract_centerline_path(vmesh=vmesh, cl_domain=cl_domain, params=params_path)
    cl_tree = CenterlineTree.from_multiblock_paths(
        cl_paths,
        knots=params["knots"],
        graft_rate=params["graft_rate"],
        force_extremes=params["force_extremes"],
    )
    return cl_tree
