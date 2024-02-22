

import numpy as np

from scipy.interpolate import BSpline, make_lsq_spline
from scipy.spatial import KDTree
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.transform import Rotation
from scipy.integrate import quad
from scipy.misc import derivative

import messages as msg
from utils._code import Tree, Node, attribute_checker
from utils.spatial import normalize, compute_ref_from_points, get_theta_coord, radians_to_degrees
from utils.splines import Spline, lsq_spline_smoothing


class ParallelTransport(Spline):

    def __init__(self) -> None:

        super().__init__()

        #The initial vector to be transported.
        self.v0 : np.ndarray = None

    @staticmethod
    def compute_parallel_transport_on_centerline(cl, v0):

        """
        This function build the parallel transport of a given vector v0 along a
        centerline object. The parallel transported vector is interpolated
        using the parameters of the centerline.

        It is build according to the algorithm from:
            https://legacy.cs.indiana.edu/ftp/techreports/TR425.pdf

        Briefly described, given a initial vector, orthogonal to the tangent of a curve.
        A parallel transport of given vector can be obtained by applying the rotation
        required by the curvature to remain normal.

        Arguments:
        -----------

            cl : Centerline,
                The input centerline along which the parallel transport will be computed.

            v0 : np.ndarray (3,)
                The initial vector to be transported.

        """

        #Build the Parallel and inherit spline curve parameters.
        pt = ParallelTransport()
        pt.set_parameters(v0=v0, t0=cl.t0, t1=cl.t1, k=cl.k, knots=cl.knots, n_knots=cl.n_knots)
        param_samples = np.linspace(pt.t0, pt.t1, num=cl.n_samples)

        tg = cl.get_tangent(cl.t0)
        V = []
        for t in param_samples:
            tg_next = cl.get_tangent(t)
            v0 = ParallelTransport.parallel_rotation(t0=tg, t1=tg_next, v=v0)
            V.append(v0)
            tg = tg_next

        #Build
        V = np.array(V)
        pt.set_parameters(build=True,
                          coeffs=make_lsq_spline(x=param_samples, y=V, t=pt.knots, k=pt.k).c)

        return pt
    #

    @staticmethod
    def parallel_rotation(t0, t1, v):
        t0dott1 = np.clip(t0.dot(t1), -1.0, 1.0)
        rot_vec = normalize(np.cross(t0, t1)) * np.arccos(t0dott1)
        R = Rotation.from_rotvec(rot_vec)
        return R.apply(v)
    #
#

class Centerline(Spline, Node):
    """
    The centerline class contains the main attributes and methods of a Bspline
    curve that models the centerline of a branch.
    """

    def __init__(self):

        Node.__init__(self=self)
        Spline.__init__(self=self)

        #Object reference frame
        self.center : np.array = None
        self.e1     : np.array = None
        self.e2     : np.array = None
        self.e3     : np.array = None

        # Spline
        self.tangent : BSpline = None
        self.v1      : ParallelTransport = None
        self.v2      : ParallelTransport = None

        # k-d tree for distance computation
        self.kdt               : KDTree     = None
        self.n_samples         : int        = 100
        self.samples           : np.ndarray = None
        self.parameter_samples : np.ndarray = None
    #

    def __str__(self):
        return Node.__str__(self=self)
    #

    def get_tangent(self, t, normalized=True):
        """
        Get the tangent of the centerline at given parameter values. Values are
        clipped to parameter domain, as in constant extrapolation.

        Arguments:
        --------------

            t : float, array-like
                The parameter values to be evaluated.

            normalized : bool
                Default True. Whether to normalize or not the tangents.
        """

        tt = np.clip(t, a_min=self.t0, a_max=self.t1)
        tg = np.array(self.tangent(tt))

        if normalized:

            if tg.shape == (3,):
                tg /= np.linalg.norm(tg)

            else:
                tg = (tg.T * 1/np.linalg.norm(tg, axis=1)).T

        return tg
    #

    def compute_samples(self, n_samples=None):
        """
        Computes a set of centerline samples and stores it in the
        attribute self.centerline_samples. The samples are also included
        in a k-d tree for fast closest point query.

        Parameters
        ----------

            n_samples : int, optional
                Number of samples to compute. If passed, the attribute
                self.n_samples_centerline is updated. The default is None.
        """
        if n_samples is not None:
            self.n_samples = n_samples

        self.parameter_samples = np.linspace(self.t0, self.t1, num=self.n_samples)
        self.samples = self.evaluate(t=self.parameter_samples)

        self.kdt = KDTree(self.samples)
    #

    def compute_local_ref(self):
        """
        Compute the object local axes.

        """

        if self.samples is None:
            self.compute_samples()
        c, e1, e2, e3 = compute_ref_from_points(points=self.samples)
        self.center = c
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
    #

    def compute_parallel_transport(self, p=None, mode='project'):
        """
        This method allows the build of the adapted frame in several ways.

        If mode == 'project':
            - If a point is passed, the vector p-c(t0) is projected onto the normal plane
              at t0 and used as initial contition for parallel transport.
            - If no point is passed, the mass center of the centerline is used as p.

        if mode == 'as_is':
            - The argument p must be the vector to be parallely transported.

        Arguments:
        ------------

            p : np.ndarray (3,)
                The point/vector to use.    i2p = normalize(c0 - self.evaluate(self.t0))

            mode : Literal['project', 'as_is']
                The chosen mode to use.

        Returns:
        -----------
            ParallelTransport
        """

        if mode == 'project':

            if p is None:
                p = self.center

            i2p = normalize(p - self.evaluate(self.t0))
            t_0 = self.get_tangent(self.t0)
            v0 = normalize(i2p - t_0.dot(i2p)*t_0)

        elif mode == 'as_is':
            if p is None:
                msg.error_message(f"Cannot build parallel transport with mode: {mode} and p: {p}")

            else:
                v0 = p
        else:
            msg.error_message(f"Wrong mode passed: mode = {mode}. Available options are {'project', 'as_is'}.")
            return False

        v = ParallelTransport.compute_parallel_transport_on_centerline(cl=self, v0=v0)
        return v
    #

    def compute_adapted_frame(self, p=None, mode='project'):
        """
        Compute a parallel transported adapted frame. This frame {t, v1, v2} is
        an estable alternative to Frenet frame and has multiple purposes. The
        argument p can be used to provide a preferred direction for v1. In turn,
        v2 is the cross product of t and v1 for orientation and orthonormality
        reasons. This method uses compute_parallel_transport method, you may be
        interested in checking documentation.

        Arguments:
        -----------

            p : np.ndarray (3,)
                A reference point used to define the initial v1.

            mode : Literal['project', 'as_is']
                The mode used to built the adapted frame. Check compute_parallel_transport.
        """

        if p is None:
            if self.e3 is None:
                self.compute_local_ref()
            p = normalize(np.cross(self.get_tangent(self.t0), self.e3))
            aux = normalize(self.center - self(self.t0))
            if p.dot(aux) < 0:
                p *= -1

        self.v1 = self.compute_parallel_transport(mode=mode, p=p)
        v2_0 = normalize(np.cross(self.get_tangent(self.t0), self.v1.v0))
        self.v2 = self.compute_parallel_transport(mode='as_is', p=v2_0)
        #
    #

    def build(self):
        """
        This method builds the splines and sets up other useful attributes.
        """
        if not attribute_checker(self, ['knots', 'coeffs'], extra_info="cant build splines."):
            return False

        super().build()
        self.tangent = self._spline.derivative()

        #Update functions that depend on centerline.
        self.compute_samples()
        self.compute_local_ref()
        self.compute_adapted_frame(mode='project', p=None)
    #

    def get_projection_parameter(self, p, method='scalar', full_output=False):
        """
        Computes the value of the parameter for the point in the centerline
        closest to p.


        Arguments:
        -----------

            p : np.array
                Point from which to compute the distance.

            method : Literal{'scalar', 'vec', 'vec_jac', 'sample'}, opt
                The minimization method to use.
                - 'scalar' : treats the optimization variable as a scalar, using
                scipy.optimize.minimize_scalar.
                - 'vec' : treats the optimization variable as a 1-dimensional
                vector, using scipy.optimize.minimize.
                - 'vec_jac' : treats the optimization variable as a 1-dimensional
                vector, using scipy.optimize.minimize. The Jacobian is provided.
                In all cases, constrained minimization is used to force the
                value of the parameter to be in [self.t0, self.t1]. The default is 'scalar'.
                - 'sample' : the optimization is avoided by keeping the closest
                sampled centerline point.

            full_output : bool
                Whether to return the distance and the value of the parameter
                or not. Default is False.

        Returns:
        ---------

            t : float
                The value of the parameter.

            d : float, opt
                The distance from p to the closest point in the centerline

        """

        def dist_to_centerline_point(t_):
            c = self.evaluate(t_)
            return np.linalg.norm(c-p)

        def deriv(t_):
            c = self.evaluate(t_)
            d = normalize(c - p.reshape(3,1))
            return d.T.dot(self.get_tangent(t_)).reshape(1)

        if self.kdt is None:
            self.compute_samples()

        d, i = self.kdt.query(p)
        t = self.parameter_samples[i]
        if method.startswith("vec") or method == 'sample':
            if method == 'vec_jac':
                res = minimize(dist_to_centerline_point, t, jac=deriv, method='trust-constr', bounds=[(self.t0, self.t1)])
                d = float(res.fun)
                x = float(res.x)
            elif method == 'vec':
                res = minimize(dist_to_centerline_point, t, method='trust-constr', bounds=[(self.t0, self.t1)])
                d = float(res.fun)
                x = float(res.x)
            else:
                x = t
        else:
            if i == 0:
                s0 = t
            else:
                s0 = self.parameter_samples[i-1]

            if i == len(self.parameter_samples)-1:
                s1 = t
            else:
                s1 = self.parameter_samples[i+1]

            res = minimize_scalar(dist_to_centerline_point, method='bounded', bounds=[s0, s1])
            d = float(res.fun)
            x = float(res.x)

        if full_output:
            return x, d

        return x
    #

    def get_projection_point(self, p, method='scalar', full_output=False):
        """
        Computes the point in the centerline closest to p.


        Arguments:
        ----------

            p : np.array
                Point from which to compute the distance.

            method : Literal{'scalar', 'vec', 'vec_jac', 'sample'}, opt
                The minimization method to use.
                - 'scalar' : treats the optimization variable as a scalar, using
                scipy.optimize.minimize_scalar.
                - 'vec' : treats the optimization variable as a 1-dimensional
                vector, using scipy.optimize.minimize.
                - 'vec_jac' : treats the optimization variable as a 1-dimensional
                vector, using scipy.optimize.minimize. The Jacobian is provided.
                - 'sample' : the optimization is avoided by keeping the closest
                sampled centerline point.
                In all cases, constrained minimization is used to force the
                value of the parameter to be in [0,1]. The default is 'scalar'.

            full_output : bool
                Whether to return the distance and the value of the parameter
                or not. Default is False.

        Returns
        -------

            p : np.array
                The closest point to p in the centerline.

            t : float, optional
                The value of the parameter.

            d : float, optional
                The distance from p to the closest point in the centerline.

        """
        t, d = self.get_projection_parameter(p, method=method, full_output=True)

        if full_output:
            return self.evaluate(t), t, d

        return self.evaluate(t)
    #

    def get_adapted_frame(self, t):
        """
        Get the adapted frame at a centerline point of parameter t

        The apted frame is defined as:

                    {t, v1, v2}

        where v1 and v2 are the parallel transported vectors and t, the tangent.

        Arguments:
        ----------
        t : float
            The parameter value for evaluation

        Returns:
        ---------

        t_  : np.ndarray
            The tangent.

        v1 : numpy.array
            The v1 vector of the adapted frame.

        v2 : numpy.array
            The v2 vector of the adapted frame.

        """

        if not attribute_checker(self, ['tangent', 'v1', 'v2'], extra_info='Cant compute adapted frame: '):
            return False

        t_  = self.get_tangent(t)
        v1 = self.v1(t)
        v2 = self.v2(t)

        return t_, v1, v2
    #

    def cartesian_to_vcs(self, p, method='scalar'):
        """
        Given a 3D point p expressed in cartesian coordinates, this method
        computes its expression in the Vessel Coordinate System (VCS).

        Arguments:
        -------------

            p : np.ndarray (3,)
                A 3D point in cartesian coordinates.

            method : Literal{'scalar', 'vec', 'vec_jac'}, opt
                The minimization method to use. See get_projection_parameter
                for more infor.

        Returns:
        ---------

            p_vcs : np.ndarray(3,)
                The coordinates of the point in the VCS.

        """

        tau, rho = self.get_projection_parameter(p, method=method, full_output=True)
        theta = get_theta_coord(p, self(tau), self.v1(tau), self.v2(tau))
        return np.array((tau, theta, rho))
    #

    def vcs_to_cartesian(self, tau, theta, rho):
        """
        Given a point expressed in Vessel Coordinate System (VCS), this method
        computes its cartesian coordinates.


        Arguments:
        ----------

            tau : float
                The longitudinal coordinate of the point

            theta : float, np.array (N,1)
                Angular coordinate of the point

            rho : float, np.array (N,1), optional.
                The radial coordinate of the point

        Returns
        -------
            p : np.ndarray
                The point in cartesian coordinates.

        """

        p = self(tau) + rho * (self.v1(tau)*np.cos(theta) + self.v2(tau)*np.sin(theta))

        return np.array(p)
    #

    def get_frenet_normal(self, t):
        """
        Get the normal vector of the frenet frame at centerline point of parameter t.

        Returns n_ computed as:

                n_ = b_ x t_

        where b and t_ are the binormal and tangent respectively and x is the
        cross product.


        Arguments:
        ----------

            t : float
                The parameter where normal is to be computed.

        Returns:
        --------
            n_ : numpy.array
                The normal of the centerline curve.

        """

        b = self.get_frenet_binormal(t)
        t = self.get_tangent(t)

        return np.cross(b,t)
    #

    def get_frenet_binormal(self, t):
        """
        Get the binormal vector of the frenet frame at centerline point

        Returns b computed as:

                    b_ =  C' x C'' /|| C' x C''||,

        where C is the parametrization of the centerline curve and x the
        cross product.


        Arguments:
        ----------

            height : float
                The parameter where binormal is to be computed.

        Returns:
        --------

            b : numpy.array
                The binormal of the centerline

        """

        cp = self.get_tangent(t, normalized=False)
        cpp = self.tangent(t, nu=1)
        b = np.cross(cp,cpp)

        return normalize(b)
    #

    def get_parametrization_velocity(self, t):
        """
        Compute the velocity of the centerline parametrization, C(t), as ||C'(t)||.

        Arguments:
        -----------
            t : float, array-like
                The parameter where velocity is to be computed.

        Returns:
        --------
            velocity : float, np.ndarray
        """

        if isinstance(t, (float, int)):
            velocity = np.linalg.norm(self.get_tangent(t, normalized=False))
        else:
            velocity = np.linalg.norm(self.get_tangent(t, normalized=False), axis=1)

        return velocity
    #

    def get_arc_length(self, b=None, a=None):
        """
        Compute the arclength of the centerline, with the formula:

                    L_c(h) = int_0^h ||C'(t)|| dt.

        Since the centerline is a piecewise polynomial (spline curve)
        each integration is carried out in each polynomial segment
        to improve accuracy.

        Arguments:
        -----------

            b : float
                Default t1. The upper parameter to compute length
            a : float
                Default t0. The lower parameter to compute length

        Returns:
        --------
        l : float
            centerline arc length
        """

        if a is None:
            a = self.t0
        if b is None:
            b = self.t1

        segments = self.get_knot_segments(a=a, b=b)

        #Compute the length at the segments
        l = 0
        for i in range(len(segments)-1):
            l += quad(self.get_parametrization_velocity, segments[i], segments[i+1])[0]

        return l
    #

    def get_curvature(self, t):
        """
        Get the curvature of the centerline at a given parameter value.
        The curvature is computed assuming the general formula,

                    k = || C' x C''|| / ||C'||^3,

        where C is the parametrization of the centerline.

        Arguments:
        -----------

            t : float
                Parameter on the centerline domain.

        Returns:
        --------

            k : float
                Curvature of the centerline at given parameter.
        """

        C_p = self.get_tangent(t)
        C_pp = self.tangent(t, nu=1)
        num = np.linalg.norm(np.cross(C_p, C_pp))
        den = np.linalg.norm(C_p)**3
        k = num / den
        return k
    #

    def get_torsion(self, t, dt=1e-4):
        """
        Get the torsion of the centerline at a certain distance from the valve.
        The torsion is computed by numerical differentiation of the binormal
        vector of the Frenet frame,

                    t = ||b'||,

        where b is the binormal vector.

        Arguments:
        -----------
        height : float
            Distance from the valve or in other words, point on the centerline domain.

        Returns:
        --------
        t : float
            torsion of the centerline at height point.
        """

        t = - np.linalg.norm(derivative(self.get_frenet_binormal, t, dx=dt, n=1))
        return t
    #

    def get_mean_curvature(self, a=None, b=None):
        """
        Get the mean curavture of the centerline in the segment
        defined from a to b. The mean curvature is computed as the
        defined integral of the curvature from a to b, divided by the
        arc length of the centerline from a to b.

                bar{k}_a^b = L_c([a,b]) * int_a^b k(t)dt.

        Since the centerline is a piecewise
        polynomial (spline curve) each integration is carried out
        in each polynomial segment to improve accuracy.


        Arguments:
        ----------
        a : float
            Default t0. The lower bound of the interval.
            Must be greater than or equal to 0 and lower than b.
        b : Default t1. The upper bound of the interval.
            Must be lower than or equal to 1 and greater than a.

        Returns:
        ---------
        k : float
            The mean curvature estimated.

        """

        if a is None:
            a=self.t0
        if b is None:
            b=self.t1

        if a < 0:
            raise ValueError(f'Value of a {a} is lower than 0')
        if b < 0:
            raise ValueError(f'Value of b {b} is greater than 0')
        if b < a:
            raise ValueError(f'Value of a:{a} is greater than value of b:{b}')


        #Get the segments
        segments = self.get_knot_segments(a=a, b=b)

        #Compute the curvatures at the segments
        k = 0
        for i in range(len(segments)-1):
            k += quad(self.get_curvature, segments[i], segments[i+1])[0]

        k /= self.get_arc_length(b=b, a=a)
        return k
    #

    @staticmethod
    def from_points(points, knots, cl=None, pt_mode='project', p=None, force_tangent=True, norm_param=True):
        """
        Function to build a Centerline object from a list of points. The amount
        knots to perform the LSQ approximation must be provided. An optional
        vector p can be passed to build the adapted frame.

        Arguments:
        ------------

            points : np.ndarray (N, 3)
                The 3D-point array to be approximated.

            knots : int or array-like
                The knot vector used to perform the LSQ spline approximation.
                If an int is passed a uniform knot vector is build.

            cl : Centerline
                A Centerline object to be used. The all the data will be overwritten.

            pt_mode : str
                The mode option to build the adapted frame by parallel transport.
                If p is not passed pt_mode must be 'project'. See compute_parallel_transport
                method for extra documentation.

            p : np.ndarray
                The initial v1. If pt_mode == 'project' it is projected onto inlet plane.

            force_tangent : Literal = {False, True, 'ini', 'end'}
                Whether to add extra weighting to the first and last points in
                the lsq approximation. If 'ini', resp. 'end', only initial (ending)
                points are weighted to force the tangent.

            norm_params : bool, opt
                Default True. Whether to normalize the parameter domain interval to
                [0, 1].

        Returns:
        ----------
            cl : Centerline
                The centerline object built from the points passed.
        """

        wini=1
        wend=1
        if force_tangent:
            wini=2
            wend=2
            if force_tangent == 'ini':
                wini=2
                wend=0
            elif force_tangent == 'else':
                wini=0
                wend=2

        spl = lsq_spline_smoothing(points=points,
                                   knots=knots,
                                   norm_param=norm_param,
                                   n_weighted_ini=wini,
                                   n_weighted_end=wend,
                                   weight_ratio=4)

        if cl is None:
            cl = Centerline()

        cl.set_parameters(
            build   = True,
            t0      = spl.t[0],
            t1      = spl.t[-1],
            k       = spl.k,
            knots   = spl.t,
            coeffs  = spl.c,
            n_knots = len(spl.t) - 2*(spl.k+1), #We only account for internal knots This should be explained somewhere....
            extra   = 'linear')

        cl.compute_adapted_frame(mode=pt_mode, p=p)

        return cl
        #
#


class CenterlineNetwork(Tree):
    """
    Class for the centerline of branched vascular geometries.
    """

    #No constructor is needed, the super() will be executed.

    def __setitem__(self, __key, cl: Centerline) -> None:
        """
        Setting items as in dictionaries. However, to belong to a CenterlineNetwork
        requieres consistency in the adapted frames.
        """
        #Checking it has parent attribute.
        if not hasattr(cl, 'parent'):
            msg.error_message(f"Aborted insertion of branch with id: {__key}. It has no parent attribute. Not even None.")
            return

        if cl.parent is not None:
            cl.set_data(join_t = self[cl.parent].get_projection_parameter(cl(cl.t0), method='scalar'))
            v1 = ParallelTransport.parallel_rotation(t0=self[cl.parent].get_tangent(cl.join_t), t1=cl.get_tangent(cl.t0), v=self[cl.parent].v1(cl.join_t))
            cl.compute_adapted_frame(p=v1, mode='as_is')

        super().__setitem__(__key, cl)
    #

    def get_centerline_association(self, p, n=None, method='scalar', thrs=30):
        """
        Given a point in space (with optional normal n) this method computes
        the branch it can be associated to. If no normal is None, the branch
        is decided based on the distance to a rough approximation on the point
        projection. If n is provided, let q the projection of p onto the nearest
        centerline branch, if the angles between vectors q2p and n are greater
        than thrs, the next nearest branch will be tested. If non satisfy the
        criteria, a warnin message will be outputed and the point will be assigned
        to the nearest branch.

        Warning: normal is expected to be used as the surface normal of a point.
        However, normales are sensible to high frequency noise in the mesh, try
        smoothing it before using the normals in the computation of the centerline
        association.

        Arguments:
        -------------

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

        Returns:
        ---------
            bid : any
                The branch id.

        """

        ids, dists, angles = [], [], []
        for cid, cl in self.items():
            q, t, d = cl.get_projection_point(p, method=method, full_output=True)
            ids.append(cid)
            dists.append(d)
            if n is not None:
                q2p = normalize(p-q)
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
    #

    def get_projection_parameter(self, p, cl_id=None, n=None, method='scalar', thrs=30, full_output=False):
        """
        Get the parameter of the projection onto the centerline tree.
        If centerline id (cl_id) argument is not provided it is computed
        using get_centerline_association.

        Arguments:
        -----------

            p : np.ndarray (3,)
                The 3D point.

            cl_id : str, opt
                Default None. The id of the centerline of the network to project
                the point. If None, it is computed using get_centerline_membership
                method.

            n : np.ndarray, opt
                Default None. A normal direction at the point, useful if the point
                belongs to the surface of the vascular domain, its normal can be used.

            method : Literal {'scalar', 'vec', 'jac-vec', 'sample'}
                The method use to compute the projection.

            full_output : bool
                Whether to return the distance and centerline membership with the parameter
                or not. Default is False.

        Returns:
        --------

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
    #

    def get_projection_point(self, p, cl_id=None, n=None, method='scalar', thrs=30, full_output=False):
        """
        Get the point projection onto the centerline tree.
        If centerline id (cl_id) argument is not provided it is computed
        using get_centerline_membership.

        Arguments:
        -----------

            p : np.ndarray (3,)
                The 3D point.

            cl_id : str, opt
                Default None. The id of the centerline of the network to project
                the point. If None, it is computed using get_centerline_membership
                method.

            n : np.ndarray, opt
                Default None. A normal direction at the point, useful if the point
                belongs to the surface of the vascular domain, its normal can be used.

            method : Literal {'scalar', 'vec', 'jac-vec', 'sample'}
                The method use to compute the projection.

            full_output : bool
                Whether to return the parameter value, distance and the centerline association
                or not. Default is False.

        Returns:
        --------

            pr : np.ndarray (3,)
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
            return t, cl_id, d

        return t
    #

    def cartesian_to_vcs(self, p, cl_id=None, n=None, method='scalar', thrs=30, full_output=False):
        """
        Given a 3D point p expressed in cartesian coordinates, this method
        computes its expression in the Vessel Coordinate System (VCS) of the
        centerline it has been associated to.

        Arguments:
        -----------

            p : np.ndarray (3,)
                The 3D point.

            cl_id : str, opt
                Default None. The id of the centerline of the network to project
                the point. If None, it is computed using get_centerline_membership
                method.

            n : np.ndarray, opt
                Default None. A normal direction at the point, useful if the point
                belongs to the surface of the vascular domain, its normal can be used.

            method : Literal {'scalar', 'vec', 'jac-vec', 'sample'}
                The method use to compute the projection.

            full_output : bool, opt
                Default False. Whether to add the cl_id to the returns.

        Returns:
        --------

            p_vcs : np.ndarray (3,)
                The (tau, theta, rho) coordinates of the given point.

            cl_id : str, opt
                The id of the centerline the point has been associated to.

        """


        if cl_id is None:
            cl_id = self.get_centerline_association(p=p, n=n, method=method, thrs=thrs)

        return self[cl_id].cartesian_to_vcs(p=p)
    #

    @staticmethod
    def from_multiblock_paths(paths, knots):
        """
        Create a CenterlineTree from a pyvista MultiBlock made polydatas with
        points joined by lines, basically like the ouput of CenterlinePathExtractor.
        Each polydata must have a field_data called 'parent' and has to be a list with
        a single id (present in the multiblock names).

        Arguments:
        ------------

            paths : pv.MultiBlock
                The multiblock containing the centerline paths. All the elements in the paths
                have to be of PolyData type. Each of these polydatas must have a field_data
                called 'parent', that has to be a list with a single id (present in the multiblock names).
                The names of the polydatas must be separable in "path_" + "id" as in path_AsAo

            n_knots : dict[str]
                A dictionary with the knots to perform the spline curve least squares fitting of each polydata.
                The id is accessed by the centerline id, and the value can be the list of knots to use, or a int
                in the latter, a uniform spline is built with the number provided.

        Returns:
        -----------

            clnet : CenterlineNetwork
                The centerline network extracted from the passed MultiBlock.
        """

        if not paths.is_all_polydata:
            msg.error_message("Can make CenterlineNetwork. Some elements of the MulitBlock are not PolyData type ")
            return None


        cl_net = CenterlineNetwork()

        cl_ids  = [s.replace('path_', '') for s in paths.keys()]
        parents = {i : paths[f"path_{i}"].field_data['parent'][0] for i in cl_ids}


        def add_to_network(nid):
            force_tangent = True if parents[nid] == 'None' else 'end'
            cl = Centerline.from_points(paths[f'path_{nid}'].points, knots=knots[nid], force_tangent=force_tangent)
            cl.id = nid
            if parents[nid] != 'None':
                cl.parent = parents[nid]
            cl_net[nid] = cl

            for cid in cl_ids:
                if parents[cid] == nid:
                    add_to_network(cid)

        for rid in cl_ids:
            if parents[rid] == 'None':
                add_to_network(rid)

        return cl_net
    #
#
