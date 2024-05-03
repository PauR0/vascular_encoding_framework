

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

from ..messages import error_message
from ..utils._code import Tree, check_specific
from ..utils.misc import split_metadata_and_fv

from .encoding import Encoding
from .vessel_encoding import VesselEncoding

class VascularEncoding(Tree, Encoding):

    def __init__(self):

        Tree.__init__(self=self)
        Encoding.__init__(self=self)
    #

    def encode_vascular_mesh(self, vmesh, cl_net, tau_knots=15, theta_knots=15, laplacian_penalty=1.0, **kwargs):
        """
        Encode a VascularMesh using a centerline network.

        This method encodes the vascular mesh as it currently is. This may lead to redundance in
        the encoding coefficients if centerline children branches are born at the centerline father
        rather than being born at the surface of the parent's vessel wall.

        Arguments
        ---------

            vmesh : VascularMesh
                The vascular network to encode.

            cl_net : CenterlineNetwork
                The centerlines of the vascular network.

            tau_knots, theta_knots : int, optional
                Default value is 15 for both. The amount of internal knots in for each component of the radius function.

            laplacian_penalty : float, optional
                Default 1.0.

            **kwargs : dict
                The above described parameters can be provided per vessel using the kwargs. Say there exist a
                Vessel whose id is AUX, to set specific parameters AUX, one can pass the argument AUX={tau_knots},
                to set a specific amount of knots and assuming the default values on the other parameters.

        Returns
        -------
            self : VascularEncoding
                The vascular mesh encoded in a Vascular Encoding object.

        See Also
        --------
        :py:meth:`encode_vascular_mesh_decoupling`

        """

        def encode_and_add_vessel(bid):
            nonlocal tau_knots, theta_knots, laplacian_penalty
            vsl_enc = VesselEncoding()
            vsl_enc.set_centerline(cl=cl_net[bid])
            vsl_mesh = vsl_enc.extract_vessel_from_network(vmesh=vmesh)
            vsl_enc.encode_vessel_mesh(vsl_mesh          = vsl_mesh,
                                       tau_knots         = check_specific(kwargs, bid, 'tau_knots', tau_knots),
                                       theta_knots       = check_specific(kwargs, bid, 'theta_knots', theta_knots),
                                       laplacian_penalty = check_specific(kwargs, bid, 'laplacian_penalty', laplacian_penalty))

            for cid in vsl_enc.children:
                encode_and_add_vessel(cid)

        for rid in cl_net.roots:
            encode_and_add_vessel(rid)
    #

    def encode_vascular_mesh_decoupling(self, vmesh, cl_net, tau_knots=15, theta_knots=15, laplacian_penalty=1.0, insertion=1.0, debug=False, **kwargs):
        """
        Encode a vascular mesh decoupling each branch as an independent vessel.

        Although using this method returns independent branches, the encoding keeps
        the track of the vessel junctions by mean of the vessel coordinates of the
        inlet in the parent branch.

        Arguments
        ---------
            vmesh : VascularMesh
                The vascular network to encode.

            cl_net : CenterlineNetwork
                The centerlines of the vascular network.

            tau_knots, theta_knots : int
                The amount of internal knots in for each component of the radius function.

            laplacian_penalty : float, optional
                Default 1.0.

            insertion : float, optional
                Default 1.0.

            debug : bool, optional
                A mode running mode that display plots of the process.

            **kwargs : dict
                The above described parameters can be provided per vessel using the kwargs. Say there exist a
                Vessel whose id is AUX, to set specific parameters AUX, one can pass the argument AUX={tau_knots},
                to set a specific amount of knots and assuming the default values on the other parameters.

        Returns
        -------
            self : VascularEncoding
                The vascular mesh encoded in a Vascular Encoding object.


        See Also
        --------
        :py:meth:`encode_vascular_mesh`

        """

        def remove_centerline_graft(bid):
            nonlocal insertion
            cl = cl_net[bid]
            pve = self[cl.parent]
            tau = pve.compute_centerline_intersection(cl, mode='parameter')
            r = vmesh.kdt.query(cl(tau))[0] * check_specific(kwargs, bid, 'insertion', insertion)
            tau_ = cl.travel_distance_parameter(-1*r, tau) #Traveling a radius distance towards inlet
            cl = cl.trim(t0_=tau_)
            return cl

        def decouple_and_encode_vessel(bid):
            nonlocal tau_knots, theta_knots, laplacian_penalty
            cl = cl_net[bid]
            if cl.parent is not None:
                cl = remove_centerline_graft(bid)

            ve = VesselEncoding()
            ve.set_centerline(cl)
            vsl_mesh = ve.extract_vessel_from_network(vmesh, debug=debug)
            ve.encode_vessel_mesh(vsl_mesh,
                                  tau_knots   = check_specific(kwargs, bid, 'tau_knots', tau_knots),
                                  theta_knots = check_specific(kwargs, bid, 'theta_knots', theta_knots),
                                  laplacian_penalty = check_specific(kwargs, bid, 'laplacian_penalty', laplacian_penalty),
                                  debug=debug)

            self[bid] = ve
            for cid in cl.children:
                decouple_and_encode_vessel(bid=cid)

        for rid in cl_net.roots:
            decouple_and_encode_vessel(bid=rid)

        return self
    #

    def make_surface_mesh(self, params=None, join_at_surface=False):
        """
        Make a triangle mesh of the encoded vascular network.

        Arguments:
        -----------

            params : dict, optional
                Default None. A dictionary where the keys are each vessel encoding id, and
                the values are the pairs (tau_res, theta_res), i.e. the resolution
                for each dimension in the parametrization. If None, de VesselEncoding default
                is assumed.

            join_at_surface : bool, optional.
                Whether to project the inlet points to closest points on parent mesh.

        Returns:
        ---------

            vsl_mesh : VascularMesh
        """

        vmesh = pv.PolyData()

        def append_vessel(vid):

            nonlocal vmesh
            ve = self[vid]

            tau_res, theta_res = None, None
            if params is not None:
                tau_res, theta_res = params[vid]

            if ve.parent is not None:
                pve = self[ve.parent]
                tau_ini = ve.centerline.t0
                if join_at_surface:
                    tau_ini = pve.compute_centerline_intersection(ve.centerline, mode='parameter')
                vsl = ve.make_surface_mesh(tau_res, theta_res, tau_ini=tau_ini)
                if join_at_surface:
                    kdt = KDTree(vmesh.points)
                    ids = vsl['tau'] == vsl['tau'].min()
                    _, sids = kdt.query(vsl.points[ids])
                    vsl.points[ids] = vmesh.points[sids]
                vmesh += vsl
            else:
                vsl = ve.make_surface_mesh(tau_res, theta_res)
                vmesh += vsl

            for cid in ve.children:
                append_vessel(cid)

        for rid in self.roots:
            append_vessel(rid)

        return vmesh
    #

    def to_multiblock(self, add_attributes=True, tau_res=None, theta_res=None):
        """
        Make a multiblock composed of other multiblocks from each encoded vessel of the vascular
        structure.

        Arguments
        ---------

            add_attributes : bool, optional
                Default True. Whether to add all the attributes required to convert the multiblock
                back to a VesselEncoding object.

            tau_res, theta_res : int, optional
                The resolution to build all the vessel walls. Defaulting to make_surface_mesh method
                default values.

        Return
        ------
            vsc_mb : pv.MultiBlock
                The built multiblock object.

        See Also
        --------
        :py:meth:`from_multiblock`
        :py:meth:`VesselEncoding.to_multiblock`
        :py:meth:`VesselEncoding.from_multiblock`
        :py:meth:`Centerline.to_polydata`
        :py:meth:`Centerline.from_polydata`
        """

        vsc_mb = pv.MultiBlock()
        for vid, vsl_enc in self.items():
            vsc_mb[vid] = vsl_enc.to_multiblock(add_attributes=add_attributes, tau_res=tau_res, theta_res=theta_res)

        return vsc_mb
    #

    @staticmethod
    def from_multiblock(vsc_mb):
        """
        Make a VascularEncoding object from a pyvista MultiBlock.

        The MultiBlock is expected to contain each vessel as a multiblock itself whose data is stored as
        field data.

        Arguments
        ---------

            add_attributes : bool, optional
                Default True. Whether to add all the attributes required to convert the multiblock
                back to a VesselEncoding object.

            tau_res, theta_res : int, optional
                The resolution to build all the vessel walls. Defaulting to make_surface_mesh method
                default values.

        Return
        ------
            vsc_enc : VascularEncoding
                The VascularEncoding object built from the passed multiblock.

        See Also
        --------
        :py:meth:`to_multiblock`
        :py:meth:`VesselEncoding.to_multiblock`
        :py:meth:`VesselEncoding.from_multiblock`
        :py:meth:`Centerline.to_polydata`
        :py:meth:`Centerline.from_polydata`
        """

        enc_dict = {vid:VesselEncoding.from_multiblock(vsl_mb=vsc_mb[vid]) for vid in vsc_mb.keys()}
        roots = [vid for vid, enc in enc_dict.items() if enc.parent in [None, 'None']]

        vsc_enc = VascularEncoding()
        def add_to_tree(i):
            vsc_enc[i] = enc_dict[i]
            for chid in enc_dict[i].children:
                add_to_tree(chid)

        for rid in roots:
            add_to_tree(rid)

        return vsc_enc
    #

    def get_metadata(self, exclude=None):
        """
        This method returns a copy of the metadata array.

        The metadata array of a VascularEncoding object is composed by the metadata arrays of the
        VesselEncoding objects it contains. The first element is the total length of the metadata
        array, then the number of VesselEncoding objects stored in it, and finally the metadata
        arrays of the VesselEncoding objects.

        Returns
        -------
            md : np.ndarray
                The metadata array.

        See Also
        --------
            :py:meth:`set_metadata`
            :py:meth:`VesselEncoding.get_metadata`
            :py:meth:`to_feature_vector`
            :py:meth:`from_feature_vector`
        """

        if exclude is None:
            exclude=[]

        md = []
        def append_md(vid):
            if vid not in exclude:
                md.append(self[vid].get_metadata())
                for cid in sorted(self[vid].children):
                    append_md(cid)

        for rid in sorted(self.roots):
            append_md(rid)

        nve = len(md) #Number of VesselEncodings stored
        n   = (md[0][0])*nve + 2 #Total Amount of metadata elements
        md = np.concatenate(md)
        md = np.concatenate([[n, nve], md])
        return md
    #

    def set_metadata(self, md):
        """
        This method extracts and sets the attributes from a the metadata array.

        See get_metadata method's documentation for further information on the expected format.

        Arguments
        ---------
            md : np.ndarray
                The metadata array.

        See Also
        --------
            :py:meth:`get_metadata`
            :py:meth:`VesselEncoding.get_metadata`
            :py:meth:`VesselEncoding.set_metadata`
            :py:meth:`to_feature_vector`
            :py:meth:`from_feature_vector`
        """

        nve = round(md[1]) #Number of VesselEncodings
        ini = 2

        for i in range(nve):
            end = ini + round(md[ini])
            ve_md = md[ini:end]
            vsl = VesselEncoding()
            vsl.id = f"{i}"
            vsl.set_metadata(ve_md)
            self[str(i)] = vsl
            ini = end
    #

    def get_feature_vector_length(self, exclude=None):
        """
        This method returns the length of the feature vector.

        The length of a VascularEncoding feature vector is the sum of the length of all the VesselEncoding feature vectors contained in it.

        Returns
        -------

            n : int
                The length of the centerline feature vector.

        """

        if len(self)<1:
            return 0

        if exclude is None:
            exclude = []

        n = 0
        def add_length(vid):
            nonlocal n
            if vid not in exclude:
                n += self[vid].get_feature_vector_length()
                for cid in self[vid].children:
                    add_length(cid)

        for rid in self.roots:
            add_length(rid)

        return n
    #

    def to_feature_vector(self, mode='full', exclude=None, add_metadata=True):
        """
        Convert the VascularEncoding to a feature vector.

        The feature vector version of a VascularEncoding consist in appending the feature vector
        representations of all the VesselEncoding objects of the network. To read about the feature
        vector format of each vessel read VesselEncoding.to_feature_vector documentation.

        For consistency reasons the order for appending each vessel in the VascularEncoding follows
        a tree-sorted scheme. Hence, it starts with the first root alphabetically, and continues with
        the first child alphabetically and so on.

        Arguments
        ---------

            mode : {'full', 'centerline', 'radius', 'image'}
                The mode to build the feature vector of the VesselEncoding objects. See
                 `VesselEncoding.to_feature_vector` for further information on each mode.

            exclude : list[str], optional
                Default None. A list with the id of vessels to be excluded when building the feature vector.
                Warning: Note that not only the provided vessel will be excluded but also all the subtree rooted in it.

            add_metadata : bool, optional
                Default True. Whether to add the metadata array at the beginning of the feature vector.

        Return
        ------
            fv : np.ndarray (N,)
                The feature vector with the selected data.

        See Also
        --------
        :py:meth:`from_feature_vector`
        :py:meth:`VesselEncoding.to_feature_vector`
        :py:meth:`VesselEncoding.from_feature_vector`
        """

        if exclude is None:
            exclude = []

        fv = []
        def append_fv(vid):
            if vid not in exclude:
                fv.append(self[vid].to_feature_vector(mode=mode, add_metadata=False))
                for cid in sorted(self[vid].children):
                    append_fv(cid)

        for rid in sorted(self.roots):
            append_fv(rid)

        fv = np.concatenate(fv)

        md = self.get_metadata() if add_metadata else []
        fv = np.concatenate([md, fv])

        return fv
    #

    @staticmethod
    def from_feature_vector(fv, md=None):
        """
        Build a VascularEncoding object from a feature vector.

        Warning: This method only works if the feature vector has the metadata at the beggining or it
        is passed using the md argument.

        Warning: Due to the lack of hierarchical data of the feature vector mode the returned
        VascularEncoding object will only have root nodes whose ids correspond to the its order in
        the feature vector.


        Arguments
        ---------

            fv : np.ndarray or array-like (N,)
                The feature vector with the metadata array at the begining.

            md : np.ndarray, optional
                Default None. If fv does not contain the metadata array at the beggining it can be
                passed through this argument.

        Returns
        -------
            vsc_enc : VascularEncoding
                The vascular encoding built from the fv.

        See Also
        --------
        :py:meth:`get_metadata`
        :py:meth:`set_metadata`
        :py:meth:`to_feature_vector`
        """

        if md is None:
            md, fv = split_metadata_and_fv(fv)

        vsc_enc = VascularEncoding()
        vsc_enc.set_metadata(md)
        n = vsc_enc.get_feature_vector_length()
        if len(fv) != n:
            error_message(f"Cannot build a VascularEncoding object from feature vector. Expected a feature vector of length {n} and the one provided has {len(fv)} elements.")
            return None

        ini = 0
        for _, vsl in vsc_enc.items():
            end = ini + vsl.get_feature_vector_length()
            vsl.extract_from_feature_vector(fv[ini:end])
            ini = end

        return vsc_enc
    #

    def translate(self, t, update=True):
        """
        Translate the VascularEncoding object, translating all the VesselEncoding objects, with the translation vector t.

        Arguments
        ---------

            t : np.ndarray (3,)
                The translation vector.

            update : bool, optional
                Default True. Whether to rebuild the splines after the transformation.

        See Also
        --------
        :py:meth:`VesselEncoding.translate`
        """

        for _, ve in self.items():
            ve.translate(t, update=update)
    #

    def scale(self, s, update=True):
        """
        Scale the VascularEncoding object, scaling all the VesselEncoding objects, by a scalar factor s.

        Arguments
        ---------

            s : float
                The scale factor.

            update : bool, optional
                Default True. Whether to rebuild the splines after the transformation.

        See Also
        --------
        :py:meth:`VesselEncoding.scale`
        """

        for _, ve in self.items():
            ve.scale(s, update=update)
    #

    def rotate(self, r, update=True):
        """
        Rotate the VascularEncoding, rotating all the VesselEncoding objects, with the provided rotation matrix r.

        Arguments
        ---------

            r : np.ndarray (3, 3)
                The rotation matrix.

            update : bool, optional
                Default True. Whether to rebuild the splines after the transformation.

        See Also
        --------
        :py:meth:`VesselEncoding.rotate`
        """

        for _, ve in self.items():
            ve.rotate(r, update=update)
    #
#

def encode_vascular_mesh(vmesh, cl_net, params, debug):
    """
    Encode a vascular mesh using the provided parameters.

    Arguments
    ---------

        vmesh : VascularMesh
            The vascular mesh to be encoded.

        cl_net : CenterlineNetwork
            The centerline network of the vascular mesh.

        params : dict
            A dictionary containing all the parameters to compute the vascular encoding.

    Return
    ------

        vsc_enc : VascularEncoding
            The vascular encoding object.

    """

    vsc_enc = VascularEncoding()

    if params['method'] == 'decoupling':
        vsc_enc.encode_vascular_mesh_decoupling(vmesh, cl_net, debug=debug, **params)

    elif params['method'] == 'at_joint':
        vsc_enc.encode_vascular_mesh(vmesh, cl_net, **params)

    else:
        error_message(f"Wrong value for encoding method argument. Available options are {{ 'decouplin', 'at_joint' }} and given is {params['method']}")

    return vsc_enc