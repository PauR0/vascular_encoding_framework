

import pyvista as pv
import numpy as np
from scipy.optimize import minimize_scalar


from centerline import Centerline
import messages as msg
from utils._code import Node, attribute_checker
from utils.spatial import normalize, radians_to_degrees

from .radius import Radius

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

    def cartesian_to_vcs(self, p, rho_norm=False, method='scalar'):
        """
        Given a 3D point p expressed in cartesian coordinates, this method
        computes its expression in the Vessel Coordinate System (VCS). The method
        requires the attribute centerline to be set, additionally if rho normalization
        is desired, the radius spline attributes must have been built.

        Arguments:
        -------------

            p : np.ndarray (3,)
                A 3D point in cartesian coordinates.

            rho_norm : bool, opt
                Default False. If radius attribute is built, and rho_norm
                is True, the radial coordinate is normalized by the expression:
                rho_n = rho /rho_w(tau, theta)

            method : Literal{'scalar', 'vec', 'vec_jac'}, opt
                The minimization method to use. See get_projection_parameter
                for more infor.

        Returns:
        ---------

            p_vcs : np.ndarray(3,)
                The coordinates of the point in the VCS.

        """

        if not attribute_checker(self, atts=['centerline'], extra_info="cant compute VCS."):
            return False

        tau, theta, rho = self.centerline.cartesian_to_vcs(p=p, method=method)
        if rho_norm:
            if not attribute_checker(self, atts=['radius'], extra_info="cant compute normalized VCS."):
                return False
            rho /= self.radius(tau, theta)

        return np.array((tau, theta, rho))
    #

    def vcs_to_cartesian(self, tau, theta, rho, rho_norm=True, grid=False, full_output=False):
        """
        Given a point expressed in Vessel Coordinate System (VCS), this method
        computes its cartesian coordinates.

        Using numpy broadcasting this metho allows working with arrays of vessel
        coordinates.

        Arguments:
        ----------

            tau : float or arraylike (N,)
                The longitudinal coordinate of the point

            theta : float or arraylike (N,)
                Angular coordinate of the point

            rho : float or arraylike (N,)
                The radial coordinate of the point

            rho_norm : bool, opt
                Default False. Whether the rho passed is normalized or not.

            grid : bool
                Default False. If true, the method returns the cartesian representation of the
                grid tau x theta x rho.

            full_output : bool, false
                Default False. Whether to return the as well the vcs. Useful in combination with grid.
        Returns
        -------
            p : np.ndarray (N, 3)
                The point in cartesian coordinates.

            tau, theta, rho : np.ndarray (N, ), opt.
                If full_output is True, the vessel coordinates of the points are returned.
        """

        if grid:
            gr = np.meshgrid(tau, theta, rho, indexing='ij')
            tau   = gr[0].ravel()
            theta = gr[1].reshape(-1, 1)
            rho   = gr[2].reshape(-1, 1)

        if rho_norm:
            rho *= self.radius(tau, np.ravel(theta)).reshape(rho.shape)

        p = self.centerline.vcs_to_cartesian(tau, theta, rho)
        if full_output:
            return p, tau, theta, rho

        return p
    #

    def extract_vessel_from_network(self, vmesh, thrs=5, use_normal=True, normal_thrs=30, cl=None, debug=False):
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
                            angle = radians_to_degrees(np.arccos(np.clip(normals[i].dot(q2p), -1, 1)))
                            if angle < normal_thrs:
                                ids[i] = 1
                        else:
                            ids[i] = 1

        vsl_mesh = vmesh.extract_points(ids.astype(bool), adjacent_cells=True, include_cells=True).extract_largest()
        if debug:
            p = pv.Plotter()
            p.add_mesh(vmesh, scalars=ids, n_colors=2, opacity=0.4)
            p.add_mesh(vsl_mesh, color='g', opacity=0.7)
            p.add_mesh(self.centerline.as_polydata(), render_lines_as_tubes=True, color='g', line_width=10)
            p.show()

        return vsl_mesh
    #

    def encode_vessel_mesh(self, vsl_mesh, tau_knots, theta_knots, cl=None, debug=True):
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

        if not 'vcs' in vsl_mesh.point_data:
            points_vcs = np.array([self.centerline.cartesian_to_vcs(p) for p in vsl_mesh.points])
        else:
            points_vcs = vsl_mesh['vcs']

        self.radius = Radius.from_points(points=points_vcs, tau_knots=tau_knots, theta_knots=theta_knots, cl=cl, debug=debug)
    #

    def compute_centerline_intersection(self, cl, mode='point'):
        """
        Given a centerline that intersects the vessel wall, this method computes the location of
        said intersection. Depending on the mode selected it can return either the intersection
        point or the parameter value of the intersection in the provided centerline.

        Warning: If the passed centerline intersects more than one time, only the first found will
        be returned.

        Arguments:
        --------------

            cl : Centerline
                The intersecting centerline.

            mode : {'point', 'parameter'}, opt
                Default 'point'. What to return.

        Returns:
        ---------
            : np.ndarray or float
                The intersection (parameter or point).

        """

        mode_opts = ['point', 'parameter']
        if mode not in mode_opts:
            msg.error_message(f"Wrong value for mode argument. It must be in {mode_opts} ")

        def intersect(t):
                vcs = self.cartesian_to_vcs(cl(t), rho_norm=True)
                return abs(1-vcs[2])

        res = minimize_scalar(intersect, bounds=(cl.t0, cl.t1), method='bounded') #Parameter at intersection

        if mode == 'parameter':
            return res.x

        return cl(res.x)
    #

    def make_surface_mesh(self, tau_res=None, theta_res=None, tau_ini=None, tau_end=None, theta_ini=None, theta_end=None, vcs=True):
        """
        Make a triangle mesh of the encoded vessel.

        Arguments:
        -----------

            tau_res : int, opt
                The number of longitudinal discretizations.

            theta_res : int, opt
                The number of angular discretizations.

            tau_ini, tau_end, theta_ini, theta_end : float, opt
                Default None. The lower and upper extrema of the interval to build,
                for the longitudinal and angular dimensions respectively. If None,
                the whole definition interval is used.

            vcs : bool
                Defaulting to True. Whether to add the VCS coordinates of
                each point as a point array.

        Returns:
        ---------

            vsl_mesh : VascularMesh
        """

        if tau_res is None:
            tau_res=100

        if theta_res is None:
            theta_res=100

        if tau_ini is None:
            tau_ini = self.centerline.t0,
        if tau_end is None:
            tau_end = self.centerline.t1

        if theta_ini is None:
            theta_ini = self.radius.y0
        if theta_end is None:
            theta_end = self.radius.y1

        close=True
        if theta_end != self.radius.y1:
            close=False

        taus   = np.linspace(tau_ini, tau_end, tau_res)
        thetas = np.linspace(theta_ini, theta_end, theta_res)
        rhos   = [1.0]

        points, tau, theta, rho = self.vcs_to_cartesian(tau=taus, theta=thetas, rho=rhos, grid=True, full_output=True)
        triangles = []

        for i in range(tau_res):
            if i > 0:
                for j in range(theta_res):
                    if j == theta_res-1:
                        if close:
                            triangles.append([3, i*theta_res + j, (i-1)*theta_res + j, (i-1)*theta_res ])
                            triangles.append([3, i*theta_res + j,     i*theta_res,     (i-1)*theta_res ])
                    else:
                        triangles.append([3, i*theta_res + j, (i-1)*theta_res + j,   (i-1)*theta_res + j+1 ])
                        triangles.append([3, i*theta_res + j,     i*theta_res + j+1, (i-1)*theta_res + j+1 ])

        vsl_mesh = pv.PolyData(points, triangles)
        if vcs:
            vsl_mesh['tau']   = tau
            vsl_mesh['theta'] = theta
            vsl_mesh['rho']   = rho
            vsl_mesh['rho_n']   = rho / self.radius(tau, theta)

        return vsl_mesh
    #
#
