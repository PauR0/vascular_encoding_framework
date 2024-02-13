

import numpy as np

from scipy.interpolate import BSpline, make_lsq_spline
from scipy.spatial import KDTree
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.transform import Rotation
from scipy.integrate import quad
from scipy.misc import derivative

import messages as msg
from utils._code import attribute_setter, attribute_checker
from utils.spatial import normalize, compute_ref_from_points
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
            tdottn = np.clip(tg.dot(tg_next), -1.0, 1.0)
            rot_vec = normalize(np.cross(tg, tg_next)) *  np.arccos(tdottn)
            R = Rotation.from_rotvec(rot_vec)
            v0 = R.apply(v0)
            V.append(v0)
            tg = tg_next

        #Build
        V = np.array(V)
        pt.set_parameters(_spline = make_lsq_spline(x=param_samples, y=V, t=pt.knots, k=pt.k))

        return pt


class Centerline(Spline):
    """
    The centerline class contains the main attributes and methods of a Bspline
    curve that models the centerline of a branch.
    """

    def __init__(self):

        super().__init__()

        #Object reference frame
        self.center : np.array = np.zeros(3)
        self.e1     : np.array = np.zeros(3)
        self.e2     : np.array = np.zeros(3)
        self.e3     : np.array = np.zeros(3)

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

    def compute_parallel_transport(self, mode='project', p=None):
        """
        This method allows the build of the adapted frame in several ways.

        If mode == 'project':
            - If a point is passed, the projection of the vector p-c(t0) onto the plane
              normal to the tangent at t0 is used.
            - If no point is passed, the center of masses of the centerline is used as p.
        if mode == 'as_is':
            - The argument p must be the vector to be parallely transported.

        Arguments:
        ------------

            mode : Literal['project', 'as_is']
                The chosen mode to use.

            p : np.ndarray (3,)
                The point/vector to use.

        Returns:
        -----------
            ParallelTransport
        """

        if mode == 'project':

            if p is None:
                c0 = self.center

            i2p = normalize(c0 - self.evaluate(self.t0))
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


        #Reference frame
        v1 = np.cross(t_0, self.e3)
        v1 = normalize(v1)

        v2 = np.cross(self.t_0, v1)
        v2 = normalize(v2)

        if v1.dot(cmp) < 0:
            v1 *= -1
            v2 *= -1

        self.v1_0 = v1
        self.v2_0 = v2
    #

    def compute_adapted_frame(self, mode, p):
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

    def get_projection_parameter(self, p, method='scalar'):
        """
        Computes the value of the parameter for the point in the centerline
        closest to p.


        Arguments:
        -----------

            p : np.array
                Point from which to compute the distance.

            method : str, optional
                Minimzation method. It can be one of 'scalar', 'vec' and 'vec_jac'.
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

        Returns:
        ---------

            t : float
                The value of the parameter.

            d : float
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

        return x, d
    #

    def get_centerline_closest_point(self, p, method='scalar', opt_ret=False):
        """
        Computes the point in the centerline closest to p.


        Arguments:
        ----------
        p : np.array
            Point from which to compute the distance.
        method : str, optional
            Minimzation method. It can be one of 'scala', 'vec' and 'vec_jac'.
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
        opt_ret : bool
            Whether to return the distance and the value of the parameter
            or not. Default is False.

        Returns
        -------
        p : np.array
            The closest point to p in the centerline.
        d : float, optional
            The distance from p to the closest point in the centerline.
        t : float, optional
            The value of the parameter.

        """
        t, d = self.get_projection_parameter(p,method=method)

        if opt_ret:
            return self.evaluate(t), d, t

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

    def get_arc_length(self, b=1, a=0):
        """
        Compute the arclength of the centerline, with the formula:

                    L_c(h) = int_0^h ||C'(t)|| dt.

        Since the centerline is a piecewise polynomial (spline curve)
        each integration is carried out in each polynomial segment
        to improve accuracy.

        Arguments:
        -----------

            b : float
                Default 1. The upper parameter to compute length
            a : float
                Default 0. The lower parameter to compute length

        Returns:
        --------
        l : float
            centerline arc length
        """

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

    def get_mean_curvature(self, a=0, b=1):
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
            Default 0. The lower bound of the interval.
            Must be greater than or equal to 0 and lower than b.
        b : Default 1. The upper bound of the interval.
            Must be lower than or equal to 1 and greater than a.

        Returns:
        ---------
        k : float
            The mean curvature estimated.

        """

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

        k /= self.get_arc_length(t=b)
        return k
    #

    @staticmethod
    def from_points(points, knots, pt_mode=None, p=None, force_tangent=True, norm_param=True):
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

            pt_mode : str
                The mode option to build the adapted frame by parallel transport.
                If p is not passed pt_mode must be 'project'. See compute_parallel_transport
                method for extra documentation.

            p : np.ndarray
                The initial v1. If pt_mode == 'project' it is projected onto inlet plane.

            force_tangent : bool
                Whether to add extra weighting to the first and last points in
                the lsq approximation.

            norm_params : bool, opt
                Default True. Whether to normalize the parameter domain interval to
                [0, 1].

        Returns:
        ----------
            cl : Centerline
                The centerline object built from the points passed.
        """

        n_weighted=1
        if force_tangent:
            n_weighted=2

        spl = lsq_spline_smoothing(points=points,
                                   knots=knots,
                                   norm_param=norm_param,
                                   n_weighted_ini=n_weighted,
                                   n_weighted_end=n_weighted,
                                   weight_ratio=4)

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

        if p is not None:
            cl.compute_adapted_frame(mode=pt_mode, p=p)

        return cl
        #