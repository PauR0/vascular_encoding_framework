

import pyvista as pv
from scipy.spatial import KDTree

from ..centerline import CenterlineNetwork
from ..utils._code import Tree

from .vessel_encoding import VesselEncoding

class VascularEncoding(Tree):

    def __init__(self):
        super().__init__()
        self.cl_net : CenterlineNetwork = None
    #

    def encode_vascular_mesh(self, vmesh, cl_net, params):
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

            params : dict.
                A dictionary with the parameters for the encoding.

        Returns
        -------
            self : VascularEncoding
                The vascular mesh encoded in a Vascular Encoding object.

        See Also
        --------
        :py:meth:`encode_vascular_mesh_decoupling`

        """

        def encode_and_add_vessel(bid):
            vsl_enc = VesselEncoding()
            vsl_enc.set_centerline(cl=cl_net[bid])
            vsl_mesh = vsl_enc.extract_vessel_from_network(vmesh=vmesh)
            vsl_enc.encode_vessel_mesh(vsl_mesh    = vsl_mesh,
                                       tau_knots   = params[bid]['tau_knots'],
                                       theta_knots = params[bid]['theta_knots'])
            for cid in vsl_enc.children:
                encode_and_add_vessel(cid)

        for rid in cl_net.roots:
            encode_and_add_vessel(rid)
    #

    def encode_vascular_mesh_decoupling(self, vmesh, cl_net, params, debug=False):
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

            params : dict.
                A dictionary with the parameters for the encoding.

            debug : bool, optional
                A mode running mode that display plots of the process.

        Returns
        -------
            self : VascularEncoding
                The vascular mesh encoded in a Vascular Encoding object.


        See Also
        --------
        :py:meth:`encode_vascular_mesh`

        """

        def remove_centerline_graft(bid):
            cl = cl_net[bid]
            pve = self[cl.parent]
            tau = pve.compute_centerline_intersection(cl, mode='parameter')
            r, _ = vmesh.kdt.query(cl(tau))
            tau_ = cl.travel_distance_parameter(-1*r, tau) #Traveling a radius distance towards inlet
            cl = cl.trim(t0_=tau_)
            return cl

        def decouple_and_encode_vessel(bid):

            cl = cl_net[bid]
            if cl.parent is not None:
                cl = remove_centerline_graft(bid)

            ve = VesselEncoding()
            ve.set_centerline(cl)
            vsl_mesh = ve.extract_vessel_from_network(vmesh, debug=debug)
            ve.encode_vessel_mesh(vsl_mesh,
                                  tau_knots=params[bid]['tau_knots'],
                                  theta_knots=params[bid]['theta_knots'],
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

            params : dict, opt
                Default None. A dictionary where the keys are each vessel encoding id, and
                the values are the pairs (tau_res, theta_res), i.e. the resolution
                for each dimension in the parametrization. If None, de VesselEncoding default
                is assumed.

            join_at_surface : bool, opt.
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
    def from_multiblock():
        """
        TODO
        """
    #

#
