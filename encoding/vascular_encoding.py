


from centerline import CenterlineNetwork
from utils._code import Tree

from .vessel_encoding import VesselEncoding

class VascularEncoding(Tree):

    def __init__(self):
        super().__init__()
        self.cl_net : CenterlineNetwork = None
    #

    def encode_vascular_mesh(self, vmesh, cl_net, params):
        """
        Encode a VascularMesh.


        Arguments:
        -------------

                vmesh : VascularMesh
                    The vascular network to encode.

                cl_net : CenterlineNetwork
                    The centerlines of the vascular network.

                params : dict.
                    A dictionary with the parameters for the encoding.

        Returns:
        ---------
            self : VascularEncoding
                The vascular mesh encoded in a Vascular Encoding object.

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

        Arguments:
        ------------
            TBD.

        Returns:
        ---------
            TBD.
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
    #
#
