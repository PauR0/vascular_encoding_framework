

import numpy as np

from scipy.interpolate import BSpline
from scipy.spatial import KDTree
from scipy.optimize import minimize, minimize_scalar


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

class Centerline:
    """
    The centerline class contains the main attributes and methods of a Bspline
    curve that models the centerline of a branch.
    """

    def __init__(self):

        #Extrema of the parameter domain.
        self.t0 : float = 0
        self.t1 : float = 1

        #Spline params
        self.k       : int = 3
        self.n_knots : int = None
        self.coeffs  : np.ndarray = None #Shape (3, n_knots+k+1)

        #Object reference frame
        self.center : np.array = np.zeros(3)
        self.e1     : np.array = np.zeros(3)
        self.e2     : np.array = np.zeros(3)
        self.e3     : np.array = np.zeros(3)

        # Spline Curves
        self.curve   : BSpline = None
        self.tangent : BSpline = None

        # k-d tree for distance computation
        self.kdt               : KDTree     = None
        self.n_samples         : int        = 100
        self.samples           : np.ndarray = None
        self.parameter_samples : np.ndarray = None
    #

    def __call__(self, t):
        """
        Evaluate the centerline at given parameter values. Values are clipped
        to parameter domain, as in constant extrapolation.
        Arguments:
        -----------

            t : float or array-like
        """

        return self.evaluate(t)
    #

    def evaluate(self, t):
        """
        Evaluate the spline curve at values provided in t.
        Values are clipped to parameter domain, as in constant
        extrapolation.

        Arguments:
        --------------

            t : float, array-like
                The parameter values to be evaluated.
        """
        tt = np.clip(t, a_min=self.t0, a_max=self.t1)
        return self.curve(tt)
    #

    def get_tangent(self, t, norm=True):
        """
        Get the tangent of the centerline at given parameter values. Values are
        clipped to parameter domain, as in constant extrapolation.

        Arguments:
        --------------

            t : float, array-like
                The parameter values to be evaluated.

            norm : bool
                Default True. Whether to normalize or not the tangents.
        """
        tt = np.clip(t, a_min=self.t0, a_max=self.t1)
        tgt = self.tangent(tt)

        if norm:
            return (tgt.T * 1/np.linalg.norm(tgt, axis=1)).T

        return tgt
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

    def set_parameters(self, **kwargs):
        """
        Set centerline parameters and attributes by kwargs.
        """

        attribute_setter(self, **kwargs)

        if  'k' in kwargs or 'n_knots' in kwargs or 'coeffs' in kwargs:
            if attribute_checker(self, ['k', 'n_knots', 'coeffs']):
                self.build_splines()
    #

    def build_splines(self):

        if not attribute_checker(self, ['n_knots', 'coeffs'], extra_info="cant build splines."):
            return False

        knots = knots_list(self.t0, self.t1, self.n_knots)

        self.curve = BSpline(t=knots,
                             c=self.coeffs,
                             k=3)

        self.tangent = self.curve.derivative()

        #Update functions that depend on centerline.
        self.compute_samples()
        self.build_parallel_transport()
        self.compute_local_ref()
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


        Parameters
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
