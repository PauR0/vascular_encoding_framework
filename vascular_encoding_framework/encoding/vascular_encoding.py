from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

from .._base import EncodingTree, SpatialObject, check_specific
from .remesh import VascularMeshing
from .vessel_encoding import VesselAnatomyEncoding

if TYPE_CHECKING:
    from ..centerline import Centerline, CenterlineTree
    from ..vascular_mesh import VascularMesh


class VascularAnatomyEncoding(EncodingTree[VesselAnatomyEncoding], VascularMeshing, SpatialObject):
    """Vascular anatomy encoding class."""

    def __init__(self):
        EncodingTree.__init__(self=self, _node_type=VesselAnatomyEncoding)

    def encode_vascular_mesh(
        self,
        vmesh: VascularMesh,
        cl_tree: CenterlineTree,
        tau_knots: int = 15,
        theta_knots: int = 15,
        laplacian_penalty: float = 1.0,
        insertion: float = 1.0,
        uncouple: bool = True,
        debug: bool = False,
        **kwargs,
    ) -> VascularAnatomyEncoding:
        """
        Encode a Vascular Mesh.

        TODO: Extend documentation.

        Parameters
        ----------
        vmesh : VascularMesh
            The vascular surface to encode.
        cl_tree : CenterlineTree
            The centerlines of the vascular surface.
        tau_knots, theta_knots : int
            The amount of internal knots in for each component of the radius function.
        laplacian_penalty : float, optional
            Default 1.0.
        insertion : float, optional
            Default 1.0.
        uncouple: bool, optional
            Default True. Whether to encode from parent-child interserction or at the cross section
            of the junction.
        debug : bool, optional
            A mode running mode that display plots of the process.
        **kwargs : dict
            The above described parameters can be provided per vessel using the kwargs. Say there
            exist a Vessel whose id is AUX, to set specific parameters AUX, one can pass the
            argument AUX={tau_knots:50}, to set a specific amount of knots and assuming the default
            values on the other parameters.

        Returns
        -------
        self : VascularAnatomyEncoding
            The vascular mesh encoded in a VascularAnatomyEncoding object.

        See Also
        --------
        encode_vascular_mesh
        """

        def remove_centerline_graft(bid):
            nonlocal insertion
            cl: Centerline = cl_tree[bid]
            pve: VesselAnatomyEncoding = self[cl.parent]
            tau = pve.compute_centerline_intersection(cl, mode="parameter")
            r = vmesh.kdt.query(cl(tau))[0] * check_specific(kwargs, bid, "insertion", insertion)
            # Traveling a radius distance towards inlet
            tau_ = cl.travel_distance_parameter(-1 * r, tau)
            cl = cl.trim(tau_0=tau_)
            return cl

        def extract_and_encode_vessel(bid):
            nonlocal tau_knots, theta_knots, laplacian_penalty, uncouple
            cl = cl_tree[bid]
            if cl.parent is not None and check_specific(kwargs, bid, "uncouple", uncouple):
                cl = remove_centerline_graft(bid)

            ve = VesselAnatomyEncoding()
            ve.set_centerline(cl)
            vsl_mesh = ve.extract_vessel_from_network(vmesh, debug=debug)
            ve.encode_vessel_mesh(
                vsl_mesh,
                tau_knots=check_specific(kwargs, bid, "tau_knots", tau_knots),
                theta_knots=check_specific(kwargs, bid, "theta_knots", theta_knots),
                laplacian_penalty=check_specific(
                    kwargs, bid, "laplacian_penalty", laplacian_penalty
                ),
                debug=debug,
            )

            self[bid] = ve
            for cid in cl.children:
                extract_and_encode_vessel(bid=cid)

        for rid in cl_tree.roots:
            extract_and_encode_vessel(bid=rid)

        return self

    def make_triangulated_surface_mesh(
        self,
        tau_res: int = 100,
        theta_res: int = 50,
        join_at_surface: bool = False,
        **kwargs,
    ) -> pv.PolyData:
        """
        Make a triangle mesh of the encoded vascular network.

        Parameters
        ----------
        tau_res, theta_res : int, optional
            The amount of points to use for longitudinal and angular discretization

        join_at_surface : bool, optional
            Whether to project the inlet points to closest points on parent mesh.

        kwargs : dict, optional
            By means of the kwargs, branch-specific parameters can be used by passing a dictionary
            with them. For instance, let us assume that exists a branch with id 'B1' which is
            shorter, and the general longitudinal resolution is a bit excessive, then we can pass an
            extra argument B1={'tau_resolution'=20} and only 20 points will be used on that branch.

        Returns
        -------
        vmesh : pv.PolyData
            The reconstructed wall meshes for each encoded vessel. Note that at current
            state, meshes are not "sewed" together.
        """

        vmesh = pv.PolyData()

        def append_vessel(vid):
            nonlocal vmesh
            ve = self[vid]

            if ve.parent is not None:
                vsl = ve.tube(
                    tau_res=check_specific(kwargs, vid, "tau_res", tau_res),
                    theta_resolution=check_specific(kwargs, vid, "theta_res", theta_res),
                )

                if check_specific(kwargs, vid, "join_at_surface", join_at_surface):
                    kdt = KDTree(vmesh.points)
                    ids = vsl["tau"] == vsl["tau"].min()
                    _, sids = kdt.query(vsl.points[ids])
                    vsl.points[ids] = vmesh.points[sids]
                vmesh += vsl
            else:
                vsl = ve.tube(
                    tau_resolution=check_specific(kwargs, vid, "tau_res", tau_res),
                    theta_resolution=check_specific(kwargs, vid, "theta_res", theta_res),
                )
                vmesh += vsl

            for cid in ve.children:
                append_vessel(cid)

        for rid in self.roots:
            append_vessel(rid)

        return vmesh

    def to_multiblock(
        self, add_attributes: bool = True, tau_res: int = 100, theta_res: int = 50, **kwargs
    ) -> pv.MultiBlock:
        """
        Make a multiblock composed of other multiblocks from each encoded vessel of the vascular
        structure.

        Parameters
        ----------
        add_attributes : bool, optional
            Default True. Whether to add all the attributes required to convert the multiblock
            back to a VesselAnatomyEncoding object.
        tau_res, theta_res : int, optional
            The resolution to build all the vessel walls. Defaulting to make_surface_mesh method
            default values.
        **kwargs
            tau_res and theta_res arguments can be provided per branch using the kwargs. Given a
            branch with id Bk belonging to the vascular anatomy encoding object, to set specific
            discretization parameters for the branch Bk, one can pass the argument
            Bk={theta_res:20}, setting the angular resolution to 20 for Bk and assuming the default
            values for the rest of the parameters.

        Returns
        -------
        vsc_mb : pv.MultiBlock
            The built multiblock object.

        See Also
        --------
        from_multiblock
        VesselAnatomyEncoding.to_multiblock
        VesselAnatomyEncoding.from_multiblock
        Centerline.to_polydata
        Centerline.from_polydata
        """

        vsc_mb = pv.MultiBlock()
        for vid, vsl_enc in self.items():
            vsc_mb[vid] = vsl_enc.to_multiblock(
                add_attributes=add_attributes,
                tau_res=check_specific(kwargs, vid, "tau_res", tau_res),
                theta_res=check_specific(kwargs, vid, "theta_res", theta_res),
            )

        return vsc_mb

    @staticmethod
    def from_multiblock(vsc_mb: pv.MultiBlock) -> VascularAnatomyEncoding:
        """
        Make a VascularAnatomyEncoding object from a pyvista MultiBlock.

        The MultiBlock is expected to contain each vessel as a multiblock itself. The
        hyperparameters and feature vectors must be stored in user_dicts of each block.

        Parameters
        ----------
        vsc_mb : pv.MultiBlock
            The pyvista multiblock with each element.

        Returns
        -------
        vsc_enc : VascularAnatomyEncoding
            The VascularAnatomyEncoding object built from the passed multiblock.

        See Also
        --------
        to_multiblock
        VesselAnatomyEncoding.to_multiblock
        VesselAnatomyEncoding.from_multiblock
        """

        enc_dict = {
            vid: VesselAnatomyEncoding.from_multiblock(vsl_mb=vsc_mb[vid]) for vid in vsc_mb.keys()
        }
        roots = [vid for vid, enc in enc_dict.items() if enc.parent in [None, "None"]]

        vsc_enc = VascularAnatomyEncoding()

        def add_to_tree(i):
            vsc_enc[i] = enc_dict[i]
            for chid in enc_dict[i].children:
                add_to_tree(chid)

        for rid in roots:
            add_to_tree(rid)

        return vsc_enc

    def set_hyperparameters(self, hp: dict[str, Any]):
        """
        Set the hyperparameters of a VascularAnatomyEncoding object.

        Note that this will initialize or modify the required VesselAnatomyEncoding objects
        according to the dictionary keys.

        Parameters
        ----------
        hp : dict[str, Any]
            The hyperparameter dictionary.

        See Also
        --------
        get_hyperparameters
        """

        # Hierarchy is stored in vsl_enc centerline hp
        roots = {rid for rid, _hp in hp.items() if _hp["centerline"]["parent"] is None}
        super().set_hyperparameters(hp=hp, roots=roots)

    def to_feature_vector(self, mode="full") -> np.ndarray:
        """
        Convert the VascularAnatomyEncoding to a feature vector.

        The feature vector version of a VascularAnatomyEncoding consist in the appending of its
        VesselAnatomyEncoding objects in a alphabetic-inductive order. This is, the first root
        branch is picked in alphabetic order, then its first children in alphabetic order, and so
        on, and so on.

        Parameters
        ----------
        mode : {'full', 'centerline', 'radius'}
            The mode to build the feature vector of the VesselAnatomyEncoding objects. _Warning_:
            Only "full" allows the posterior rebuilding of the encoding.

        Returns
        -------
        fv : np.ndarray (N,)
            The feature vector with the selected data.

        See Also
        --------
        from_feature_vector
        VesselAnatomyEncoding.to_feature_vector
        VesselAnatomyEncoding.from_feature_vector
        """
        return super().to_feature_vector(mode=mode)

    def from_feature_vector(
        self, fv: np.ndarray, hp: dict[str, Any] = None
    ) -> VascularAnatomyEncoding:
        """
        Build a VascularAnatomyEncoding object from a feature vector.

        > Note that while hyperparameters argument is optional it must have been previously set or
        passed.

        Parameters
        ----------
        fv : np.ndarray
            The feature vector.
        hp : dict[str, Any], optional
            The hyperparameter dictionary for the VascularAnatomyEncoding object.

        Returns
        -------
        vsc_enc : VascularAnatomyEncoding
            The vascular anatomy encoding built from the fv.

        See Also
        --------
        get_hyperparameters
        set_hyperparameters
        to_feature_vector
        """
        return super().from_feature_vector(fv=fv, hp=hp)

    def translate(self, t):
        """
        Translate the VascularAnatomyEncoding object, translating all the VesselAnatomyEncoding
        objects, with the translation vector t.

        Parameters
        ----------
        t : np.ndarray (3,)
            The translation vector.

        See Also
        --------
        VesselAnatomyEncoding.translate
        """

        for _, ve in self.items():
            ve.translate(t)

    def scale(self, s):
        """
        Scale the VascularAnatomyEncoding object, scaling all the VesselAnatomyEncoding objects, by
        a scalar factor s.

        Parameters
        ----------
        s : float
            The scale factor.

        See Also
        --------
        VesselAnatomyEncoding.scale
        """

        for _, ve in self.items():
            ve.scale(s)

    def rotate(self, r):
        """
        Rotate the VascularAnatomyEncoding, rotating all the VesselAnatomyEncoding objects, with the
        provided rotation matrix r.

        Parameters
        ----------
        r : np.ndarray (3, 3)
            The rotation matrix.

        See Also
        --------
        VesselAnatomyEncoding.rotate
        """

        for _, ve in self.items():
            ve.rotate(r)


def encode_vascular_mesh(
    vmesh: VascularMesh,
    cl_tree: CenterlineTree,
    tau_knots: int = 15,
    theta_knots: int = 15,
    laplacian_penalty: float = 1.0,
    insertion: float = 1.0,
    uncouple: bool = True,
    debug: bool = False,
    **kwargs,
):
    """
    Encode a vascular mesh using the provided parameters.

    Parameters
    ----------
    vmesh : VascularMesh
        The vascular mesh to be encoded.
    cl_tree : CenterlineTree
        The centerline tree of the vascular mesh.
    tau_knots, theta_knots : int
        The amount of internal knots in for each component of the radius function.
    laplacian_penalty : float, optional
        Default 1.0.
    insertion : float, optional
        Default 1.0.
    uncouple: bool, optional
        Default True. Whether to encode from parent-child interserction or at the cross section
        of the junction.
    debug : bool, optional
        A mode running mode that display plots of the process.
    **kwargs : dict
        The above described parameters can be provided per vessel using the kwargs. Say there
        exist a Vessel whose id is AUX, to set specific parameters AUX, one can pass the
        argument AUX={tau_knots:50}, to set a specific amount of knots and assuming the default
        values on the other parameters.

    Returns
    -------
    vsc_enc : VascularAnatomyEncoding
        The vascular anatomy encoding object.
    """

    vsc_enc = VascularAnatomyEncoding()

    vsc_enc.encode_vascular_mesh(
        vmesh,
        cl_tree,
        tau_knots=tau_knots,
        theta_knots=theta_knots,
        laplacian_penalty=laplacian_penalty,
        insertion=insertion,
        uncouple=uncouple,
        debug=debug,
        **kwargs,
    )

    return vsc_enc
