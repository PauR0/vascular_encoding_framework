from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

from .._base import Encoding, SpatialObject, Tree, check_specific
from .remesh import VascularMeshing
from .vessel_encoding import VesselAnatomyEncoding

if TYPE_CHECKING:
    from ..centerline import Centerline, CenterlineTree
    from ..vascular_mesh import VascularMesh


class VascularAnatomyEncoding(Tree, Encoding, VascularMeshing, SpatialObject):
    """Vascular anatomy encoding class."""

    def __init__(self):
        Tree.__init__(self=self)

        self.kind = ["uncoupled"]
        self._hyperparameters = []

    def encode_vascular_mesh(
        self,
        vmesh: VascularMesh,
        cl_tree: CenterlineTree,
        tau_knots: int = 15,
        theta_knots: int = 15,
        laplacian_penalty: float = 1.0,
        **kwargs,
    ) -> VascularAnatomyEncoding:
        """
        Encode a VascularMesh using a centerline tree.

        This method encodes the vascular mesh as it currently is. This may lead to redundance in
        the encoding coefficients if centerline children branches are born at the centerline father
        rather than being born at the surface of the parent's vessel wall.

        Parameters
        ----------
            vmesh : VascularMesh
                The vascular surface to encode.

            cl_tree : CenterlineTree
                The centerlines of the vascular surface.

            tau_knots, theta_knots : int, optional
                Default value is 15 for both. The amount of internal knots in for each component of
                the radius function.

            laplacian_penalty : float, optional
                Default 1.0.

            **kwargs : dict
                The above described parameters can be provided per vessel using the kwargs. Say
                there exist a Vessel whose id is AUX, to set specific parameters AUX, one can pass
                the argument AUX={tau_knots}, to set a specific amount of knots and assuming the
                default values on the other parameters.

        Returns
        -------
            self : VascularAnatomyEncoding
                The vascular mesh encoded in a VascularAnatomyEncoding object.

        See Also
        --------
        encode_vascular_mesh_decoupling

        """

        def encode_and_add_vessel(bid):
            nonlocal tau_knots, theta_knots, laplacian_penalty
            vsl_enc = VesselAnatomyEncoding()
            vsl_enc.set_centerline(cl=cl_tree[bid])
            vsl_mesh = vsl_enc.extract_vessel_from_network(vmesh=vmesh)
            vsl_enc.encode_vessel_mesh(
                vsl_mesh=vsl_mesh,
                tau_knots=check_specific(kwargs, bid, "tau_knots", tau_knots),
                theta_knots=check_specific(kwargs, bid, "theta_knots", theta_knots),
                laplacian_penalty=check_specific(
                    kwargs, bid, "laplacian_penalty", laplacian_penalty
                ),
            )

            for cid in vsl_enc.children:
                encode_and_add_vessel(cid)

        for rid in cl_tree.roots:
            encode_and_add_vessel(rid)

    def encode_vascular_mesh_decoupling(
        self,
        vmesh: VascularMesh,
        cl_tree: CenterlineTree,
        tau_knots: int = 15,
        theta_knots: int = 15,
        laplacian_penalty: float = 1.0,
        insertion: float = 1.0,
        debug: bool = False,
        **kwargs,
    ) -> VascularAnatomyEncoding:
        """
        Encode a vascular mesh decoupling each branch as an independent vessel.

        Although using this method returns independent branches, the encoding keeps
        the track of the vessel junctions by mean of the vessel coordinates of the
        inlet in the parent branch.

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
        debug : bool, optional
            A mode running mode that display plots of the process.
        **kwargs : dict
            The above described parameters can be provided per vessel using the kwargs. Say there
            exist a Vessel whose id is AUX, to set specific parameters AUX, one can pass the
            argument AUX={tau_knots}, to set a specific amount of knots and assuming the default
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

        def decouple_and_encode_vessel(bid):
            nonlocal tau_knots, theta_knots, laplacian_penalty
            cl = cl_tree[bid]
            if cl.parent is not None:
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
                decouple_and_encode_vessel(bid=cid)

        for rid in cl_tree.roots:
            decouple_and_encode_vessel(bid=rid)

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
            By means of the kwargs, branch-specific parameters can be used by passing
            a dictionary with them. For instance, let us assume that exists a branch
            with id 'B1' which is shorter, and the general longitudinal resolution is
            a bit excessive, then we can pass an extra argument B1={'tau_resolution'=20}
            and only 20 points will be used on that branch.

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
        self, add_attributes: bool = True, tau_res: int = 100, theta_res: int = 50
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
                add_attributes=add_attributes, tau_res=tau_res, theta_res=theta_res
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

    def get_hyperparameters(self) -> dict[str, Any]:
        """
        Get the dict containing the hyperparameters of the VascularAnatomyEncoding object.

        Returns
        -------
        hp : dict[str, Any]
            The json serializable dictionary with the hyperparameters of the encoding.
        """

        self._hyperparameters = list(self.keys())
        return super().get_hyperparameters(**self)

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

        def set_branch_hp(bid):
            vsl_enc = self[bid] if bid in self else VesselAnatomyEncoding()
            vsl_enc.set_hyperparameters(hp=hp[bid])
            self[bid] = vsl_enc

            for cid in vsl_enc.children:
                set_branch_hp(cid)

        for rid, _hp in hp.items():
            # Hierarchy is stored in vsl_enc centerline hp
            if _hp["centerline"]["parent"] is None:
                set_branch_hp(rid)

        return

    def get_feature_vector_length(self):
        """
        Return the length of the feature vector.

        The length of a VascularAnatomyEncoding feature vector is the sum of the length of all
        the VesselAnatomyEncoding feature vectors contained in it.

        Returns
        -------
        n : int
            The length of the centerline feature vector.
        """
        n = 0
        for vsl_enc in self.values():
            n += vsl_enc.get_feature_vector_length()

        return n

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

        fv = []

        def append_fv(vid):
            fv.append(self[vid].to_feature_vector(mode=mode))
            for cid in sorted(self[vid].children):
                append_fv(cid)

        for rid in sorted(self.roots):
            append_fv(rid)

        fv = np.concatenate(fv)
        return fv

    @staticmethod
    def from_feature_vector(hp: dict[str, Any], fv: np.ndarray) -> VascularAnatomyEncoding:
        """
        Build a VascularAnatomyEncoding object from a feature vector.

        Parameters
        ----------
        hp : dict[str, Any]
            The hyperparameter dictionary for the VascularAnatomyEncoding object.
        fv : np.ndarray
            The feature vector.

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

        vsc_enc = VascularAnatomyEncoding()
        vsc_enc.set_hyperparameters(hp=hp)
        n = vsc_enc.get_feature_vector_length()
        if len(fv) != n:
            raise ValueError(
                "Cannot build a VascularAnatomyEncoding object from feature vector. Expected a"
                + f"feature vector of length {n} and the one provided has {len(fv)} elements."
            )

        ini = 0

        def extract_vessel_fv(vid):
            nonlocal ini
            vsl_enc: VesselAnatomyEncoding = vsc_enc[vid]
            end = ini + vsl_enc.get_feature_vector_length()
            vsl_enc.update_from_feature_vector(fv=fv[ini:end])
            ini = end

            for cid in sorted(vsl_enc.children):
                extract_vessel_fv(cid)

        for rid in sorted(vsc_enc.roots):
            extract_vessel_fv(rid)

        return vsc_enc

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


def encode_vascular_mesh(vmesh: VascularMesh, cl_tree: CenterlineTree, params: dict, debug: bool):
    """
    Encode a vascular mesh using the provided parameters.

    Parameters
    ----------
    vmesh : VascularMesh
        The vascular mesh to be encoded.
    cl_tree : CenterlineTree
        The centerline tree of the vascular mesh.
    params : dict
        A dictionary containing all the parameters to compute the encoding.

    Returns
    -------
    vsc_enc : VascularAnatomyEncoding
        The vascular anatomy encoding object.
    """

    vsc_enc = VascularAnatomyEncoding()

    if params["method"] == "decoupling":
        vsc_enc.encode_vascular_mesh_decoupling(vmesh, cl_tree, debug=debug, **params)

    elif params["method"] == "at_joint":
        vsc_enc.encode_vascular_mesh(vmesh, cl_tree, **params)

    else:
        raise ValueError(
            "Wrong value for encoding method argument."
            + f"Available options are {{'decoupling', 'at_joint'}} and given is {params['method']}"
        )

    return vsc_enc
