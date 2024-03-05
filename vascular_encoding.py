

import pyvista as pv
import numpy as np

from centerline import Centerline, CenterlineNetwork
from utils._code import Tree, Node, attribute_checker
from utils.splines import BiSpline, semiperiodic_LSQ_bivariate_approximation
from utils.spatial import normalize, radians_to_degrees

class Radius(BiSpline):

    def __init__(self):

        super().__init__()
        self.x0 = 0
        self.x1 = 1
        self.y0 = 0
        self.y1 = 2*np.pi
    #

    def set_parameters_from_centerline(self, cl):
        """
        This method set the radius bounds equal to the Centerline object
        passed.

        Arguments:
        -----------

            cl : Centerline
                The Centerline of the vessel
        """

        self.x0 = cl.t0
        self.x1 = cl.t1
    #
#

class VesselEncoding(Node):
    """
    The class for encoding a single branch vessel.
    """

    def __init__(self):

        super().__init__()

        self.centerline : Centerline = None
        self.radius     : Radius   = None
    #

    def set_data(self, **kwargs):
        """
        Method to set attributes using kwargs and the setattr function.
        """

        if 'centerline' in kwargs:
            self.set_centerline(cl=kwargs['centerline'])
            kwargs.pop('centerline')

        return super().set_data(**kwargs)
    #

    def set_centerline(self, cl):
        """
        Set the centerline attribute. Note that the VesselEncoding object inherits
        the node attributes from the centerline, in addition if joint_t is defined,
        it is also inherited.
        """

        self.centerline = cl
        self.set_data_from_other_node(cl)
        if hasattr(cl, 'joint_t'):
            self.set_data(joint_t=cl.joint_t)
        #
    #

    def build(self):

        self.centerline.build()
        self.radius.build()
    #

    def extract_vessel_from_network(self, vmesh, thrs=5, use_normal=True, normal_thrs=30, cl=None, debug=True):
        """
        This method extracts the vessel mesh from a vascular structure
        based on the centerline. It works similarly to the centerline
        association method of the CenterlineNetwork class, however, in
        thiss other method each point is associated to a single branch,
        and this method does not care for other branches, allowing points
        to belong to different vessels.

        The vessel is extracted as follows:
        For each point, p, in the the mesh, its projection, q, is computed. Assuming
        wall normals are pointing outwards,
        Then, the deviation of the point to the cross section it belongs is measured by
        the angle that makes the tangent t, with vector q2p. If a points belong to a cross
        section, the angle between t and q2p should be 90. Then, points whose deviation
        is over thrs argument are rejected. Once points have been identified, they are
        extracted from the mesh and the largest connected component is consiedered as
        the vessel of interest.

        If use_normal is True, instead of considering the angle between t and q2p,
        the angle considered is t and the surface normal of p, \hatN(p).

        This method requires self.centerline. Warning: If argument cl is passed, the
        centerline object is set as self.centerline.

        Arguments:
        -----------

            vmesh : pv.PolyData
                The vascular network surface mesh.

            thrs : list[float],opt
                Defaulting to 10. The angle allowed between the tangent and q2p, can be seen
                as the longitudinal deviation from a cross section, for instance, thrs = 20
                allows points whose fulfilling that 70<angle(t, q2p)<110.

            cl : Centerline
                The Vessel centerline.

        Returns:
        ---------
            vsl_mesh : pv.PolyData
                The vessel polydata extracted.
        """

        if cl is not None:
            self.set_centerline(cl=cl)

        if not attribute_checker(self, ['centerline'], extra_info="cannot extract Vessel from network."):
            return False

        if 'Normals' not in vmesh.point_data:
            vmesh.compute_normals(inplace=True)
        normals = vmesh.get_array('Normals', preference='point')

        ids = np.zeros((vmesh.n_points,))
        for i in range(vmesh.n_points):
            p = vmesh.points[i]
            q, t, _ = self.centerline.get_projection_point(p, full_output=True)
            int_pts, _ = vmesh.ray_trace(q, p, first_point=False)
            if int_pts.shape[0] < 2:
                q2p = normalize(p-q)
                if q2p.dot(normals[i]) > 0:
                    tg = self.centerline.get_tangent(t)
                    angle = radians_to_degrees(np.arccos(q2p.dot(tg)))
                    if abs(angle-90) < thrs:
                        if use_normal:
                            angle = radians_to_degrees(np.arccos(normals[i].dot(q2p)))
                            if angle < normal_thrs:
                                ids[i] = 1
                        else:
                            ids[i] = 1

        vsl_mesh = vmesh.extract_points(ids.astype(bool), adjacent_cells=True, include_cells=True).extract_largest()
        if debug:
            p = pv.Plotter()
            p.add_mesh(vmesh, scalars=ids, n_colors=2, opacity=0.6)
            p.add_mesh(self.centerline.as_polydata(), render_lines_as_tubes=True, color='g', line_width=10)
            p.show()

        return vsl_mesh
    #

    def encode_vessel_mesh(self, vsl_mesh, tau_knots, theta_knots, cl=None, debug=False):
        """
        Encode a vessel using the centerline and the anisotropic radius.
        If the centerline have hierarchical data like its parent or joint_t
        it is also set as a parameter for the branch.

        This method requires self.centerline to be set or passed.
        Warning: If argument cl is passed, the centerline object is set as
        self.centerline what may overwrite possible existing data.


        Arguments:
        -----------

            vsl_mesh : pv.PolyData
                The mesh representing the vessel.

            knots_tau, knots_theta : int
                The amount of divisions to build the uniform knot vector.
                TODO: Add support for non-uniform splines.

            cl : Centerline, opt
                Default None. The centerline of said vessel. If passed is stored
                at self.centerline and node data is copied from it.

        Returns:
        ---------
            self : VesselEncoding
                The VesselEncoding object.
        """

        if cl is not None:
            self.set_centerline(cl=cl)

        points_vcs = np.array([self.centerline.cartesian_to_vcs(p) for p in vsl_mesh.points])

        self.radius = Radius()
        self.radius.set_parameters_from_centerline(self.centerline)
        self.radius.set_parameters(n_knots_x = tau_knots,
                                   n_knots_y = theta_knots,
                                   coeffs    = semiperiodic_LSQ_bivariate_approximation(x=points_vcs[:,0],
                                                                             y=points_vcs[:,1],
                                                                             z=points_vcs[:,2],
                                                                             nx=tau_knots,
                                                                             ny=theta_knots,
                                                                             kx=self.radius.kx,
                                                                             ky=self.radius.ky,
                                                                             debug=True))
        self.build()
    #
#

class VascularEncoding(Tree):

    def __init__(self):
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
#
