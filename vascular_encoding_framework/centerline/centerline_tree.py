from __future__ import annotations

import numpy as np
import pyvista as pv

from ..messages import error_message
from ..utils._code import Tree, check_specific
from ..utils.spatial import normalize, radians_to_degrees
from .centerline import Centerline
from .domain_extractors import extract_centerline_domain
from .parallel_transport import ParallelTransport
from .path_extractor import extract_centerline_path


class CenterlineTree(Tree):
    """Class for the centerline of branched vascular geometries."""

    def __setitem__(self, __key, cl: Centerline) -> None:
        """
        Set items as in dictionaries. However, to belong to a CenterlineTree
        requires consistency in the adapted frames.
        """
        # Checking it has parent attribute.
        if not hasattr(cl, "parent"):
            error_message(
                f"Aborted insertion of branch with id: {__key}. It has no parent attribute. Not even None."
            )
            return

        if cl.parent is not None:
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
        Given a 3D point p expressed in cartesian coordinates, this method
        computes its expression in the Vessel Coordinate System (VCS) of the
        centerline it has been associated to.

        Parameters
        ----------
        p : np.ndarray (3,)
            The 3D point.
        cl_id : str, opt
            Default None. The id of the centerline in the tree to project
            the point. If None, it is computed using get_centerline_association
            method.
        n : np.ndarray, opt
            Default None. A normal direction at the point, useful if the point
            belongs to the surface of the vascular domain, its normal can be used.
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

    def to_multiblock(self, add_attributes=False):
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
    def from_multiblock(mb):
        """
        Make a CenterlineTree object from a pyvista MultiBlock made polydatas.

        As the counterpart of to_multiblock, this static method is meant for building
        CenterlineTree objects from a pyvista MultiBlock, where each element of the MultiBlock
        is a PolyData with the information required to build the Tree structure and the Spline
        information.

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
            error_message(
                "Can't make CenterlineTree. Some elements of the MultiBlock are not PolyData type."
            )
            return None

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
        paths, n_knots=10, curvature_penatly=1, graft_rate=0.5, force_extremes=True, **kwargs
    ) -> CenterlineTree:
        """
        Create a CenterlineTree from a pyvista MultiBlock made polydatas with
        points joined by lines, basically like the output of CenterlinePathExtractor.


        Each polydata must have a field_data called 'parent' and has to be a list with
        a single id (present in the multiblock names).

        Parameters
        ----------
        paths : pv.MultiBlock
            The multiblock containing the centerline paths. All the elements in the paths
            have to be of PolyData type. Each of these polydatas must have a field_data
            called 'parent', that has to be a list with a single id (present in the multiblock names).
            The names of the polydatas must be separable in "path_" + "id" as in path_AsAo
        n_knots : dict[str]
            A dictionary with the knots to perform the spline curve least squares fitting of each polydata.
            The id is accessed by the centerline id, and the value can be the list of knots to use, or a int
            in the latter, a uniform spline is built with the number provided.
        graft_rate : float, opt
            Default is 0.5. A parameter to control the grafting insertion. Represent a distance proportional to the radius
            traveled towards the parent branch inlet along the centerline at the junction.
        force_extremes : {False, True, 'ini', 'end'}
            Default True. Whether to force the centerline to interpolate the boundary behavior
            of the approximation. If True the first and last point are interpolated and its
            tangent is approximated by finite differences using the surrounding points. If
            'ini', respectively 'end', only one of both extremes is forced.
        **kwargs : dict
            The above described arguments can be provided per branch using the kwargs. Say there exist a
            path_AUX in the passed multiblock, to set specific parameters for the branch AUX, one can pass
            the dictionary AUX={n_knots:20}, setting the number of knots to 20 and assuming the default
            values for the rest of the parameters.

        Returns
        -------
        cl_tree : CenterlineTree
            The centerline tree extracted from the passed MultiBlock.
        """

        if not paths.is_all_polydata:
            error_message(
                "Can't make CenterlineTree. Some elements of the MultiBlock are not PolyData type "
            )
            return None

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
                pt_mode=check_specific(kwargs, nid, "pt_mode", "project"),
                p=check_specific(kwargs, nid, "p", None),
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

    def translate(self, t, update=True):
        """
        Translate the CenterlineTree object, translating all the Centerline objects, with the translation vector t.

        Parameters
        ----------
        t : np.ndarray (3,)
            The translation vector.
        update : bool, optional
            Default True. Whether to rebuild the splines after the transformation.

        See Also
        --------
        Centerline.translate
        """

        for _, cl in self.items():
            cl.translate(t, update=update)

    def scale(self, s, update=True):
        """
        Scale the CenterlineTree object, scaling all the Centerline objects, by a scalar factor s.

        Parameters
        ----------
        s : float
            The scale factor.
        update : bool, optional
            Default True. Whether to rebuild the splines after the transformation.

        See Also
        --------
        Centerline.scale
        """

        for _, cl in self.items():
            cl.scale(s, update=update)

    def rotate(self, r, update=True):
        """
        Rotate the CenterlineTree, rotating all the Centerline objects, with the provided rotation matrix r.

        Parameters
        ----------
        r : np.ndarray (3, 3)
            The rotation matrix.
        update : bool, optional
            Default True. Whether to rebuild the splines after the transformation.

        See Also
        --------
        Centerline.rotate
        """

        for _, cl in self.items():
            cl.rotate(r, update=update)


def extract_centerline(
    vmesh, params, params_domain=None, params_path=None, debug=False
) -> CenterlineTree:
    """
    Compuete the CenterlineTree of a provided a VascularMesh object with propperly defined
    boundaries.

    Parameters
    ----------
    vmesh : VascularMesh
        The VascularMesh object where centerline is to be computed.
    params : dict
        The parameters for the spline approximation for each boundary, together with the grafting rate
        and tangent forcing parameters.
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
