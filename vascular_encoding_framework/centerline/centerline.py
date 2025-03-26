from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.integrate import quad
from scipy.interpolate import BSpline, make_lsq_spline
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from ..messages import error_message
from ..splines.splines import UniSpline, uniform_penalized_spline
from ..utils._code import Node, Tree, attribute_checker, check_specific
from ..utils.geometry import polyline_from_points
from ..utils.misc import split_metadata_and_fv
from ..utils.spatial import (compute_ref_from_points, get_theta_coord,
                             normalize, radians_to_degrees)
from .domain_extractors import extract_centerline_domain
from .path_extractor import extract_centerline_path


class ParallelTransport(UniSpline):

    def __init__(self) -> None:

        super().__init__()

        # The initial vector to be transported.
        self.v0: np.ndarray = None
    #

    @staticmethod
    def compute_parallel_transport_on_centerline(cl, v0) -> ParallelTransport:
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

        # Build the Parallel and inherit spline curve parameters.
        pt = ParallelTransport()
        pt.set_parameters(
            v0=v0,
            t0=cl.t0,
            t1=cl.t1,
            k=cl.k,
            knots=cl.knots,
            n_knots=cl.n_knots)
        param_samples = np.linspace(pt.t0, pt.t1, num=cl.n_samples)

        tg = cl.get_tangent(cl.t0)
        V = []
        for t in param_samples:
            tg_next = cl.get_tangent(t)
            v0 = ParallelTransport.parallel_rotation(t0=tg, t1=tg_next, v=v0)
            V.append(v0)
            tg = tg_next

        # Build
        V = np.array(V)
        pt.set_parameters(
            build=True,
            coeffs=make_lsq_spline(
                x=param_samples,
                y=V,
                t=pt.knots,
                k=pt.k).c)

        return pt
    #

    @staticmethod
    def parallel_rotation(t0, t1, v) -> np.ndarray:
        t0dott1 = np.clip(t0.dot(t1), -1.0, 1.0)
        rot_vec = normalize(np.cross(t0, t1)) * np.arccos(t0dott1)
        R = Rotation.from_rotvec(rot_vec)
        return R.apply(v)
    #
#


class Centerline(UniSpline, Node):
    """
    The centerline class contains the main attributes and methods of a Bspline
    curve that models the centerline of a branch.
    """

    def __init__(self):

        Node.__init__(self=self)
        self.joint_t: float = None  # The parameter of the joint at parent centerline

        UniSpline.__init__(self=self)

        # Object reference frame
        self.center: np.array = None
        self.e1: np.array = None
        self.e2: np.array = None
        self.e3: np.array = None

        # Spline
        self.tangent: BSpline = None
        self.v1: ParallelTransport = None
        self.v2: ParallelTransport = None

        # k-d tree for distance computation
        self.kdt: KDTree = None
        self.n_samples: int = 100
        self.samples: np.ndarray = None
        self.parameter_samples: np.ndarray = None
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
                tg = (tg.T * 1 / np.linalg.norm(tg, axis=1)).T

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

        self.parameter_samples = np.linspace(
            self.t0, self.t1, num=self.n_samples)
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
              at t0 and used as initial condition for parallel transport.
            - If no point is passed, the mass center of the centerline is used as p.

        if mode == 'as_is':
            - The argument p must be the vector to be parallel transported.

        Arguments
        ---------

            p : np.ndarray (3,)
                The point/vector to use.

            mode : {'project', 'as_is'}
                The chosen mode to use.

        Returns
        -------

            v : ParallelTransport
                The ParallelTransport object representing the transport of the vector along the
                centerline.

        See Also
        --------
        :py:meth:`compute_adapted_frame`
        :py:meth:`ParallelTransport.compute_parallel_transport_on_centerline`
        """

        if mode == 'project':

            if p is None:
                p = self.center

            i2p = normalize(p - self.evaluate(self.t0))
            t_0 = self.get_tangent(self.t0)
            v0 = normalize(i2p - t_0.dot(i2p) * t_0)

        elif mode == 'as_is':
            if p is None:
                error_message(
                    f'Cannot build parallel transport with mode: {mode} and p: {p}')

            else:
                v0 = p
        else:
            error_message(
                f"Wrong mode passed: mode = {mode}. Available options are {'project', 'as_is'}.")
            return False

        v = ParallelTransport.compute_parallel_transport_on_centerline(
            cl=self, v0=v0)
        return v
    #

    def compute_adapted_frame(self, p=None, mode='project'):
        """
        Compute a parallel transported adapted frame. This frame {t, v1, v2} is
        an stable alternative to Frenet frame and has multiple purposes. The
        argument p can be used to provide a preferred direction for v1. In turn,
        v2 is the cross product of t and v1 for orientation and orthonormality
        reasons. This method uses compute_parallel_transport method, you may be
        interested in checking documentation.

        Arguments
        ---------

            p : np.ndarray (3,), optional
                Default None. A reference vector/point used to define the initial v1. This argument
                can be used in the following ways:

                If no initial vector is passed (p=None), the argument mode is ignored and two options are available:

                    - If the attribute self.v1 is None:
                        Then an initial v1 is computed as the cross product of the tangent of the
                        centerline and the axis of least variability on the centerline,
                        i.e. v1_0 = normalize(t(0), e3)

                    - Otherwise, the attribute self.v1 must be a ParallelTransport object with the initial
                    condition set in its attribute v0, i.e. self.v1.__class__ == ParallelTransport and
                    self.v1.v0 is not None. And this vector is used to build the parallel transport.

                If an initial vector is passed p = (p0, p1, p2) the two options available are described
                in compute_parallel_transport method and are controlled by the mode argument.

            mode : {'project', 'as_is'}, optional
                The mode used to built the adapted frame if p is passed. Check compute_parallel_transport.


        See Also
        --------
        :py:meth:`compute_parallel_transport`
        :py:meth:`ParallelTransport.compute_parallel_transport_on_centerline`
        """

        if p is None:
            if self.v1 is None:
                if self.e3 is None:
                    self.compute_local_ref()
                p = normalize(np.cross(self.get_tangent(self.t0), self.e3))
                aux = normalize(self.center - self(self.t0))
                if p.dot(aux) < 0:
                    p *= -1
                self.v1 = self.compute_parallel_transport(mode=mode, p=p)
            elif isinstance(self.v1, ParallelTransport):
                if self.v1.v0 is not None:
                    self.v1 = self.compute_parallel_transport(
                        p=self.v1.v0, mode='as_is')
                else:
                    error_message(
                        f'Wrong usage of compute_adapted_frame. No p {(p)} has been passed but self.v1 is a ParallelTransport object with v0 == None.')
        else:
            self.v1 = self.compute_parallel_transport(mode=mode, p=p)

        v2_0 = normalize(np.cross(self.get_tangent(self.t0), self.v1.v0))
        self.v2 = self.compute_parallel_transport(mode='as_is', p=v2_0)
        #
    #

    def build(self):
        """
        This method builds the splines and sets up other useful attributes.
        """

        super().build()
        self.tangent = self._spl.derivative()

        # Update functions that depend on centerline.
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
            return np.linalg.norm(c - p)

        def deriv(t_):
            c = self.evaluate(t_)
            d = normalize(c - p.reshape(3, 1))
            return d.T.dot(self.get_tangent(t_)).reshape(1)

        if self.kdt is None:
            self.compute_samples()

        d, i = self.kdt.query(p)
        t = self.parameter_samples[i]
        if method.startswith('vec') or method == 'sample':
            if method == 'vec_jac':
                res = minimize(
                    dist_to_centerline_point,
                    t,
                    jac=deriv,
                    method='trust-constr',
                    bounds=[
                        (self.t0,
                         self.t1)])
                d = float(res.fun)
                x = float(res.x)
            elif method == 'vec':
                res = minimize(
                    dist_to_centerline_point,
                    t,
                    method='trust-constr',
                    bounds=[
                        (self.t0,
                         self.t1)])
                d = float(res.fun)
                x = float(res.x)
            else:
                x = t
        else:
            if i == 0:
                s0 = t
            else:
                s0 = self.parameter_samples[i - 1]

            if i == len(self.parameter_samples) - 1:
                s1 = t
            else:
                s1 = self.parameter_samples[i + 1]

            res = minimize_scalar(
                dist_to_centerline_point,
                method='bounded',
                bounds=[
                    s0,
                    s1])
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
        t, d = self.get_projection_parameter(
            p, method=method, full_output=True)

        if full_output:
            return self.evaluate(t), t, d

        return self.evaluate(t)
    #

    def get_adapted_frame(self, t):
        """
        Get the adapted frame at a centerline point of parameter t

        The adapted frame is defined as:

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

        if not attribute_checker(
                self, ['tangent', 'v1', 'v2'], info='Cant compute adapted frame: '):
            return False

        t_ = self.get_tangent(t)
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
                for more info.

        Returns:
        ---------

            p_vcs : np.ndarray(3,)
                The coordinates of the point in the VCS.

        """

        tau, rho = self.get_projection_parameter(
            p, method=method, full_output=True)
        theta = get_theta_coord(p, self(tau), self.v1(tau), self.v2(tau))
        return np.array((tau, theta, rho))
    #

    def vcs_to_cartesian(self, tau, theta, rho, grid=False, full_output=False):
        """
        Given a point expressed in Vessel Coordinate System (VCS), this method
        computes its cartesian coordinates.

        Using numpy broadcasting this method allows working with arrays of vessel
        coordinates.

        Arguments:
        ----------

            tau : float or array-like (N,)
                The longitudinal coordinate of the point

            theta : float or array-like (N,)
                Angular coordinate of the point

            rho : float or array-like (N,)
                The radial coordinate of the point

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

        arraylike = (list, np.ndarray)
        if isinstance(theta, arraylike) or grid:
            theta = np.array([theta]).reshape(-1, 1)

        if isinstance(rho, arraylike) or grid:
            rho = np.array([rho]).reshape(-1, 1)

        if grid:
            gr = np.meshgrid(tau, theta, rho)
            tau = gr[0].ravel()
            theta = gr[1].reshape(-1, 1)
            rho = gr[2].reshape(-1, 1)

        p = self(tau) + rho * (self.v1(tau) *
                               np.cos(theta) + self.v2(tau) * np.sin(theta))

        if full_output:
            return p, tau, theta, rho

        return p
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

        return np.cross(b, t)
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
        b = np.cross(cp, cpp)

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
            velocity = np.linalg.norm(
                self.get_tangent(
                    t, normalized=False), axis=1)

        return velocity
    #

    def get_arc_length(self, b=None, a=None):
        """
        Compute the arc length of the centerline, with the formula:

                    L_c(a,b) = int_a^b ||c'(t)|| dt.

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

        # Compute the length at the segments
        l = 0
        for i in range(len(segments) - 1):
            l += quad(self.get_parametrization_velocity,
                      segments[i], segments[i + 1])[0]

        return l
    #

    def travel_distance_parameter(self, d, a=None):
        """
        Get the parameter resulting from traveling a distance d, from an initial
        parameter a. Note that if d is negative, the distance will be traveled
        in reverse direction to centerline parameterization.

        Arguments:
        ------------

            d : float
                The signed distance to travel.

            a : float
                Default is self.t0. The initial parameter where to start the
                traveling.

        Returns:
        ----------
            t : float
                The parameter at which the distance from a, along the centerline
                has reached d.
        """

        if a is None:
            a = self.t0

        if d == 0:
            return a

        if d > 0:
            bounds = [a, self.t1]
            if abs(d) > self.get_arc_length(self.t1, a):
                return self.t1

            def f(t):
                return np.abs(d - self.get_arc_length(b=t, a=a))

        if d < 0:
            bounds = [self.t0, a]
            if abs(d) > self.get_arc_length(a, self.t0):
                return self.t0

            def f(t):
                return np.abs(d + self.get_arc_length(b=a, a=t))

        res = minimize_scalar(fun=f, bounds=bounds, method='bounded')

        return res.x
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

        assert dt > 0, f"Wrong value for differential step. It must be greater than 0."

        if t == self.t0:
            b_der = (self.get_frenet_binormal(self.t0+dt) -
                     self.get_frenet_binormal(self.t0))/dt
        elif t == self.t1:
            b_der = (self.get_frenet_binormal(self.t0+dt) -
                     self.get_frenet_binormal(self.t0))/dt
        else:
            dt = min(dt, abs(self.t0-dt), abs(self.t1-dt))
            b_der = (self.get_frenet_binormal(t+dt) -
                     self.get_frenet_binormal(t-dt))/dt
        torsion = - np.linalg.norm(b_der)
        return torsion
    #

    def get_mean_curvature(self, a=None, b=None):
        """
        Get the mean curvature of the centerline in the segment
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
            a = self.t0
        if b is None:
            b = self.t1

        if a < 0:
            raise ValueError(f'Value of a {a} is lower than 0')
        if b < 0:
            raise ValueError(f'Value of b {b} is greater than 0')
        if b < a:
            raise ValueError(f'Value of a:{a} is greater than value of b:{b}')

        # Get the segments
        segments = self.get_knot_segments(a=a, b=b)

        # Compute the curvatures at the segments
        k = 0
        for i in range(len(segments) - 1):
            k += quad(self.get_curvature, segments[i], segments[i + 1])[0]

        k /= self.get_arc_length(b=b, a=a)
        return k
    #

    def to_polydata(self, t_res=None, add_attributes=False):
        """
        Transform centerline into a PolyData based on points and lines.

        Arguments
        ---------

            tau_res : int, opt
                The number of points in which to discretize the curve.

            add_attributes : bool, opt
                Default False. If true, all the attributes necessary to build the
                splines and its hierarchical relations are added as field data.

        Returns
        -------
            poly : pv.PolyData
                A PolyData object with polyline topology defined.
        """

        params = self.parameter_samples
        points = self.samples
        if t_res is not None:
            params = np.linspace(self.t0, self.t1, t_res)
            points = self(params)

        poly = polyline_from_points(points)

        poly['params'] = params
        poly['v1'] = self.v1(params)
        poly['v2'] = self.v2(params)

        if add_attributes:
            # Adding Node atts:
            poly.add_field_data(np.array([self.id]), 'id', deep=True)
            poly.add_field_data(np.array([self.parent]), 'parent', deep=True)
            poly.add_field_data(
                np.array(
                    list(
                        self.children)),
                'children',
                deep=True)
            poly.add_field_data(np.array([self.joint_t]), 'joint_t', deep=True)

            # Adding Spline atts:
            poly.add_field_data(
                np.array([self.t0, self.t1]), 'interval', deep=True)
            poly.add_field_data(np.array([self.k]), 'k', deep=True)
            poly.add_field_data(np.array([self.n_knots]), 'n_knots', deep=True)
            poly.add_field_data(np.array(self.coeffs), 'coeffs', deep=True)
            poly.add_field_data(
                np.array([self.extra]), 'extrapolation', deep=True)

        return poly
    #

    def save(self, fname, binary=True):
        """
        Save the centerline object as a vtk PolyData, appending the essential attributes as field
        data entries.

        Arguments:
        -----------

            fname : str
                Filename to write to. If does not end in .vtk, the extension is appended.

            binary : bool, opt
                Default True. Whether to write the file in binary or ASCII format.
        """

        poly = self.to_polydata(add_attributes=True)
        poly.save(filename=fname, binary=binary)
    #

    def from_polydata(self, poly):
        """
        Build a centerline object from a pyvista PolyData object that contains the required
        attributes as field_data. The minimum required data are the parameters involving the spline
        creation, namely, {'interval' 'k' 'knots' 'coeffs' 'extrapolation'}. Additionally, if the
        centerline belong to a tree, it is required to provide the attributes {'id', 'parent',
        'children', 'joint_t'}.

        TODO: Think a bit more about turning if-elif into match-case

        Arguments:
        ----------

            poly : pv.PolyData

        Returns:
        -----------
            self : Centerline
                The centerline object with the attributes already set.
        """

        spl_atts = ['interval', 'k', 'n_knots', 'coeffs', 'extrapolation']
        for att in spl_atts:
            if att not in poly.field_data:
                error_message(
                    f'Could not find attribute: {att} in polydata. Wont build centerline object')
                return None

        for att in spl_atts:
            value = poly.get_array(att, preference='field')

            if att == 'interval':
                self.set_parameters(t0=value[0], t1=value[1])

            elif att == 'k':
                self.set_parameters(k=round(value[0]))

            elif att == 'n_knots':
                self.set_parameters(n_knots=round(value[0]))

            elif att == 'coeffs':
                self.set_parameters(coeffs=np.array(value))

            elif att == 'extrapolation':
                self.set_parameters(extra=str(value[0]))
        self.build()

        if 'v1' in poly.point_data:
            self.compute_adapted_frame(
                p=poly.get_array(
                    'v1',
                    preference='point')[0],
                mode='as_is')

        node_atts = list(Node().__dict__) + ['joint_t']
        for att in node_atts:
            if att in poly.field_data:
                value = poly.get_array(att, preference='field')
                if att == 'id':
                    self.set_data(**{att: str(value[0])})

                elif att == 'parent':
                    if value in [None, 'None']:
                        self.set_data(parent=None)
                    else:
                        self.set_data(parent=str(value[0]))

                elif att == 'joint_t':
                    if value in [None, 'None']:
                        self.set_data(joint_t=None)
                    else:
                        self.set_data(joint_t=float(value[0]))

                elif att == 'children':
                    self.set_data(children=list(value))

        return self
    #

    @staticmethod
    def read(fname) -> Centerline:
        """
        Read centerline object from a vtk file.

        Arguments:
        ------------

            fname : str
                The name of the file storing the centerline.
        """

        poly = pv.read(fname)
        return Centerline().from_polydata(poly)
    #

    def trim(
            self,
            t0_,
            t1_=None,
            trim_knots=False,
            pass_atts=True,
            n_samps=100) -> Centerline:
        """
        This method trims the centerline from t0_ to t1_ and
        returns the new segment as a centerline object. If pass_atts is true
        all the centerline attributes such as the v1, v2 and others are kept.
        The amount of knots for the trimmed centerline will be computed taking
        into account the amount of knot_segments in the interval [t0_, t1_].

        Arguments
        ---------

            t0_, t1_ : float
                The lower and upper extrema to trim. If t1_ is None, self.t1 is assumed.

            trim_knots : bool, optional
                Default False. If true the number of knots is reduced to keep the spacing as it
                was in the untrimmed centerline. Otherwise, the number of knots is kept.

            pass_atts : bool, optional
                Default True. Whether to pass all the attributes of the current centerline
                to the trimmed one.

            n_samps : int, optional
                Default 100. The amount of samples to generate to perform the approximation.

        Returns
        -------

            cl : Centerline
                The trimmed centerline.

        """

        if t1_ is None:
            t1_ = self.t1

        ts = np.linspace(t0_, t1_, n_samps)
        n_knots = self.n_knots
        if trim_knots:
            n_knots = len(self.get_knot_segments(t0_, t1_)) - 2
        spl = uniform_penalized_spline(points=self(ts),
                                       n_knots=n_knots,
                                       k=self.k,
                                       # Normalized domain
                                       param_values=(ts - t0_) / (t1_ - t0_),
                                       force_ini=True,
                                       force_end=True,
                                       curvature_penalty=0.0)

        cl = Centerline()
        cl.set_parameters(build=True,
                          t0=spl.t[0],
                          t1=spl.t[-1],
                          k=spl.k,
                          knots=spl.t,
                          coeffs=spl.c,
                          n_knots=len(spl.t) - 2 * (spl.k + 1),
                          extra='linear')

        if pass_atts:
            cl.set_data_from_other_node(self)
            cl.set_data(joint_t=self.joint_t)
            cl.compute_adapted_frame(mode='as_is', p=self.v1(t0_))

        return cl
    #

    @staticmethod
    def from_points(
            points,
            n_knots,
            k=3,
            curvature_penalty=1.0,
            param_values=None,
            pt_mode='project',
            p=None,
            force_extremes=True,
            cl=None) -> Centerline:
        """
        Function to build a Centerline object from a list of points. The amount
        knots to perform the LSQ approximation must be provided. An optional
        vector p can be passed to build the adapted frame.

        Arguments
        ---------

            points : np.ndarray (N, 3)
                The 3D-point array to be approximated.

            n_knots : int
                The number of uniform internal knots to build the knot vector.

            k : int, optional
                Default 3. The polynomial degree of the splines.

            curvature_penalty : float, optional
                Default 1.0. A penalization factor for the spline approximation.

            param_values : array-like (N,), optional
                Default None. The parameter values of the points provided so the parametrization
                of the centerline is approximated assuming cl(param_values) = points. If None
                provided the normalized cumulative distance among the points is used.

            pt_mode : str
                The mode option to build the adapted frame by parallel transport.
                If p is not passed pt_mode must be 'project'. See compute_parallel_transport
                method for extra documentation.

            p : np.ndarray
                The initial v1. If pt_mode == 'project' it is projected onto inlet plane.

            force_extremes : {False, True, 'ini', 'end'}
                Default True. Whether to force the centerline to interpolate the boundary behavior
                of the approximation. If True the first and last point are interpolated and its
                tangent is approximated by finite differences using the surrounding points. If
                'ini', respectively 'end', only one of both extremes is forced.

            cl : Centerline
                A Centerline object to be used. All the data will be overwritten.

        Returns
        -------
            cl : Centerline
                The centerline object built from the points passed.
        """

        spl = uniform_penalized_spline(
            points=points,
            n_knots=n_knots,
            k=k,
            param_values=param_values,
            force_ini=force_extremes in [
                True,
                'ini'],
            force_end=force_extremes in [
                True,
                'end'],
            curvature_penalty=curvature_penalty)

        if cl is None:
            cl = Centerline()

        cl.set_parameters(
            build=True,
            t0=spl.t[0],
            t1=spl.t[-1],
            k=spl.k,
            knots=spl.t,
            coeffs=spl.c,
            n_knots=n_knots,
            extra='linear')

        cl.compute_adapted_frame(mode=pt_mode, p=p)

        return cl
    #

    def translate(self, t, update=True):
        """
        Translate the Centerline object.

        All the affine transformations are applied to the coefficients and then the rest of
        attributes are a recomputed.

        Arguments
        ---------

            t : np.ndarray (3,)
                The translation vector.

            update : bool, optional
                Default True. Whether to rebuild the splines after the transformation.
        """

        if self.coeffs is not None:
            self.coeffs += t.reshape(3,)
            if update:
                self.build()
    #

    def scale(self, s, update=True):
        """
        Scale the Centerline.

        All the affine transformations are applied to the coefficients and then the rest of
        attributes are a recomputed.

        Arguments
        ---------

            s : float
                The scale factor.

            update : bool, optional
                Default True. Whether to rebuild the splines after the transformation.
        """

        if self.coeffs is not None:
            self.coeffs *= s
            if update:
                self.build()
    #

    def rotate(self, r, update=True):
        """
        Rotate the Centerline object.

        All the affine transformations are applied to the coefficients and then the rest of
        attributes are a recomputed.

        Arguments
        ---------

            r : np.ndarray (3, 3)
                The rotation matrix.

            update : bool, optional
                Default True. Whether to rebuild the splines after the transformation.
        """

        # ensure normality of the rotation matrix columns
        r /= np.linalg.norm(r, axis=0)

        if self.coeffs is not None:
            self.coeffs = (r @ self.coeffs.T).T
            if update:
                self.build()
    #

    def get_metadata(self) -> np.ndarray:
        """
        This method returns a copy of the metadata array.

        As of this code version the
        metadata array is [7, k, n_knots, n_samples, v1(t0)[0], v1(t0)[1], v1(t0)[2]].

        Returns
        -------
            md : np.ndarray

        See Also
        --------
            :py:meth:`set_metadata`
            :py:meth:`to_feature_vector`
            :py:meth:`from_feature_vector`
        """

        v1 = self.v1(self.t0)
        md = np.array([7,
                      self.k,
                      self.n_knots,
                      self.n_samples,
                      v1[0],
                      v1[1],
                      v1[2]
                       ])

        return md
    #

    def set_metadata(self, md):
        """
        This method extracts and sets the attributes from a the metadata array.

        As of this code version the metadata array is expected to be
                [6, k, n_knots, n_samples, v1(t0)[0], v1(t0)[1], v1(t0)[2]].


        Arguments
        ---------

            md : np.ndarray or array-like
                The metadata array.

            build : bool, optional
                Default False. Whether to build the splines after setting the metadata.
                This should be set to true if parameters such as coefficients have been
                previously set.

        See Also
        --------
            :py:meth:`to_feature_vector`
            :py:meth:`from_feature_vector`
        """

        self.set_parameters(build=False,
                            k=round(md[1]),
                            n_knots=round(md[2]),
                            n_samples=round(md[3]),
                            )

        self.v1 = ParallelTransport()
        self.v1.v0 = md[4:]
    #

    def get_feature_vector_length(self) -> int:
        """
        This method returns the length of the feature vector considering the spline parameters.

        If n is the amount of internal knots, and k is the degree of the BSpline polynomials,
        the length of the centerline feature vector is computed as: 3(n+k+1). The multiplication
        by 3 is due to the three components of the coefficients (a.k.a. control points.).


        Returns
        -------

            l : int
                The length of the centerline feature vector.

        """
        if not attribute_checker(
            self, [
                'n_knots', 'k'], info='Cannot compute the Centerline feature vector length.'):
            return None
        l = 3 * (self.n_knots + self.k + 1)
        return l
    #

    def to_feature_vector(self, add_metadata=True) -> np.ndarray:
        """
        Convert the Centerline object to its feature vector representation.

        The feature vector version of a Centerline consist in appending the raveled centerline
        coefficients. If add_metadata is True (which is the default), a metadata vector is appended
        at the beginning of the feature vector. The first entry of the metadata vector is the amount
        of metadata in total, making it look like [n, md0, ..., mdn], read more about it in get.

        Arguments
        ---------

            add_metadata: bool, optional
                Default True. Wether to append metadata at the beginning of the feature vector.

        Return
        ------

            fv : np.ndarray
                The feature vector according to mode. The shape of each feature vector changes
                accordingly.

        Note
        ----
        Note that the feature vector representation does not bear any hierarchical data, not even
        if add_metadata is True. Be sure that hierarchical data is properly stored if will be later
        required. For storage purposes check `to_multiblock` method.


        See Also
        --------
        :py:meth:`get_metadata`
        :py:meth:`from_feature_vector`

        """

        fv = self.coeffs.ravel()

        if add_metadata:
            fv = np.concatenate([self.get_metadata(), fv])

        return fv
    #

    @staticmethod
    def from_feature_vector(fv, md=None) -> Centerline:
        """
        Build a Centerline object from a feature vector.

        Note that in order to build the Centerline, the feature vector must start with the metadata
        array or it must be passed using the md argument. Read more about the metadata array at
        `get_metadata` method docs.


        Arguments
        ---------

            fv : np.ndarray (N,)
                The feature vector with the metadata at the beginning.

            md : np.ndarray (M,)
                The metadata array to use. If passed, it will be assumed that fv does not
                contain it at the beginning.

        Return
        ------
            cl : Centerline
                The Centerline object built from the feature vector.

        See Also
        --------
        :py:meth:`to_feature_vector`
        :py:meth:`get_metadata`
        """

        cl = Centerline()

        if md is None:
            md, fv = split_metadata_and_fv(fv)

        cl.set_metadata(md)

        l = cl.get_feature_vector_length()
        if len(fv) != l:
            error_message(
                f'Cannot build a Centerline object from feature vector. Expected n_knots+(k+1)={l} coefficients and {len(fv)} were provided.')
            return None

        cl.set_parameters(
            build=True,
            coeffs=fv.reshape(-1, 3),
            extra='linear',
        )

        return cl
    #
#


class CenterlineTree(Tree):
    """
    Class for the centerline of branched vascular geometries.
    """

    # No constructor is needed, the super() will be executed.

    def __setitem__(self, __key, cl: Centerline) -> None:
        """
        Setting items as in dictionaries. However, to belong to a CenterlineTree
        requires consistency in the adapted frames.
        """
        # Checking it has parent attribute.
        if not hasattr(cl, 'parent'):
            error_message(
                f'Aborted insertion of branch with id: {__key}. It has no parent attribute. Not even None.')
            return

        if cl.parent is not None:
            cl.set_data(join_t=self[cl.parent].get_projection_parameter(
                cl(cl.t0), method='scalar'))
            v1 = ParallelTransport.parallel_rotation(t0=self[cl.parent].get_tangent(
                cl.join_t), t1=cl.get_tangent(cl.t0), v=self[cl.parent].v1(cl.join_t))
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
        than _thrs_, the next nearest branch will be tested. If non satisfy the
        criteria, a warning message will be output and the point will be assigned
        to the nearest branch.

        Warning: normal is expected to be used as the surface normal of a point.
        However, normals are sensible to high frequency noise in the mesh, try
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
            q, _, d = cl.get_projection_point(
                p, method=method, full_output=True)
            ids.append(cid)
            dists.append(d)
            if n is not None:
                q2p = normalize(p - q)
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

    def get_projection_parameter(
            self,
            p,
            cl_id=None,
            n=None,
            method='scalar',
            thrs=30,
            full_output=False):
        """
        Get the parameter of the projection onto the centerline tree.
        If centerline id (cl_id) argument is not provided it is computed
        using get_centerline_association.

        Arguments:
        -----------

            p : np.ndarray (3,)
                The 3D point.

            cl_id : str, opt
                Default None. The id of the centerline of the tree to project
                the point. If None, it is computed using get_centerline_association
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

    def get_projection_point(
            self,
            p,
            cl_id=None,
            n=None,
            method='scalar',
            thrs=30,
            full_output=False):
        """
        Get the point projection onto the centerline tree.
        If centerline id (cl_id) argument is not provided it is computed
        using get_centerline_association.

        Arguments
        ---------

            p : np.ndarray (3,)
                The 3D point.

            cl_id : str, opt
                Default None. The id of the centerline in the tree to project the point.  If None,
                it is computed using get_centerline_association method.

            n : np.ndarray, opt
                Default None. A normal direction at the point, useful if the point belongs to the
                surface of the vascular domain, its normal can be used.

            method : Literal {'scalar', 'vec', 'jac-vec', 'sample'}
                The method use to compute the projection.

            full_output : bool
                Whether to return the parameter value, distance and the centerline association or
                not. Default is False.

        Returns
        -------

            p : np.ndarray (3,)
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

        p, t, d = self[cl_id].get_projection_point(
            p=p, method=method, full_output=True)

        if full_output:
            return p, t, cl_id, d

        return p
    #

    def cartesian_to_vcs(
            self,
            p,
            cl_id=None,
            n=None,
            method='scalar',
            thrs=30,
            full_output=False):
        """
        Given a 3D point p expressed in cartesian coordinates, this method
        computes its expression in the Vessel Coordinate System (VCS) of the
        centerline it has been associated to.

        Arguments
        ---------

            p : np.ndarray (3,)
                The 3D point.

            cl_id : str, opt
                Default None. The id of the centerline in the tree to project
                the point. If None, it is computed using get_centerline_association
                method.

            n : np.ndarray, opt
                Default None. A normal direction at the point, useful if the point
                belongs to the surface of the vascular domain, its normal can be used.

            method : Literal {'scalar', 'vec', 'jac-vec', 'sample'}
                The method use to compute the projection.

            full_output : bool, opt
                Default False. Whether to add the cl_id to the returns.

        Returns
        -------

            p_vcs : np.ndarray (3,)
                The (tau, theta, rho) coordinates of the given point.

            cl_id : str, opt
                The id of the centerline the point has been associated to.

        """

        if cl_id is None:
            cl_id = self.get_centerline_association(
                p=p, n=n, method=method, thrs=thrs)

        if full_output:
            return self[cl_id].cartesian_to_vcs(p=p), cl_id

        return self[cl_id].cartesian_to_vcs(p=p)
    #

    def to_multiblock(self, add_attributes=False):
        """
        Return a pyvista MultiBlock with the centerline branches as pyvista PolyData objects.

        Arguments
        ---------

            add_attributes : bool, opt
                Default False. Whether to add all the required attributes to built the
                CenterlineTree back.


        Returns
        -------

            mb : pv.MultiBlock
                The multiblock with the polydata paths.

        See Also
        --------
        :py:meth:`Centerline.to_polydata`

        """

        mb = pv.MultiBlock()
        for i, cl in self.items():
            mb[i] = cl.to_polydata(t_res=None, add_attributes=add_attributes)
        return mb
    #

    @staticmethod
    def from_multiblock(mb):
        """
        Make a CenterlineTree object from a pyvista MultiBlock made polydatas.

        As the counterpart of :py:meth:`to_multiblock`, this static method is meant for building
        CenterlineTree objects from a pyvista MultiBlock, where each element of the MultiBlock
        is a PolyData with the information required to build the Tree structure and the Spline
        information.

        Arguments
        ---------

            mb : pv.MultiBlock
                The multiblock containing the required data.

        Returns
        -------

            cl_tree : CenterlineTree
                The centerline tree extracted from the passed MultiBlock.
        """

        if not mb.is_all_polydata:
            error_message(
                "Can't make CenterlineTree. Some elements of the MultiBlock are not PolyData type.")
            return None

        cl_dict = {
            cid: Centerline().from_polydata(
                poly=mb[cid]) for cid in mb.keys()}
        roots = [
            cid for cid,
            cl in cl_dict.items() if cl.parent in [
                None,
                'None']]

        cl_tree = CenterlineTree()

        def add_to_tree(i):
            cl_tree[i] = cl_dict[i]
            for chid in cl_dict[i].children:
                add_to_tree(chid)

        for rid in roots:
            add_to_tree(rid)

        return cl_tree
    #

    @staticmethod
    def from_multiblock_paths(
            paths,
            n_knots=10,
            curvature_penatly=1,
            graft_rate=0.5,
            force_extremes=True,
            **kwargs) -> CenterlineTree:
        """
        Create a CenterlineTree from a pyvista MultiBlock made polydatas with
        points joined by lines, basically like the output of CenterlinePathExtractor.
        Each polydata must have a field_data called 'parent' and has to be a list with
        a single id (present in the multiblock names).

        Arguments
        ---------

            paths : pv.MultiBlock
                The multiblock containing the centerline paths. All the elements in the paths
                have to be of PolyData type. Each of these polydatas must have a field_data
                called 'parent', that has to be a list with a single id (present in the multiblock names).
                The names of the polydatas must be separable in "path_" + "id" as in path_AsAo

            n_knots : dict[str]
                A dictionary with the knots to perform the spline curve least squares fitting of each polydata.
                The id is accessed by the centerline id, and the value can be the list of knots to use, or a int
                in the latter, a uniform spline is built with the number provided.

            graft_rate : float, opt
                Default is 0.5. A parameter to control the grafting insertion. Represent a distance proportional to the radius
                traveled towards the parent branch inlet along the centerline at the junction.

            force_extremes : {False, True, 'ini', 'end'}
                Default True. Whether to force the centerline to interpolate the boundary behavior
                of the approximation. If True the first and last point are interpolated and its
                tangent is approximated by finite differences using the surrounding points. If
                'ini', respectively 'end', only one of both extremes is forced.

            **kwargs : dict
                The above described arguments can be provided per branch using the kwargs. Say there exist a
                path_AUX in the passed multiblock, to set specific parameters for the branch AUX, one can pass
                the dictionary AUX={n_knots:20}, setting the number of knots to 20 and assuming the default
                values for the rest of the parameters.

        Returns
        -------

            cl_tree : CenterlineTree
                The centerline tree extracted from the passed MultiBlock.
        """

        if not paths.is_all_polydata:
            error_message(
                "Can't make CenterlineTree. Some elements of the MultiBlock are not PolyData type ")
            return None

        cl_tree = CenterlineTree()

        cl_ids = [s.replace('path_', '') for s in paths.keys()]
        parents = {i: paths[f'path_{i}'].field_data['parent'][0]
                   for i in cl_ids}

        def add_to_tree(nid):

            nonlocal n_knots, force_extremes, curvature_penatly, graft_rate
            points = paths[f'path_{nid}'].points
            if parents[nid] != 'None':
                pcl = cl_tree[parents[nid]]
                pre_joint = paths[f'path_{nid}'].points[0]
                pre_joint_t = pcl.get_projection_parameter(pre_joint)
                gr = check_specific(kwargs, nid, 'graft_rate', graft_rate)
                if gr:
                    joint_t = pcl.travel_distance_parameter(
                        d=-paths[f'path_{nid}']['radius'][0] * gr, a=pre_joint_t)
                    joint = pcl(joint_t)
                    ids = np.linalg.norm(
                        points - joint, axis=1) > paths[f'path_{nid}']['radius'][0] * gr
                    points = np.concatenate(
                        [[joint, pcl((joint_t + pre_joint_t) / 2)], paths[f'path_{nid}'].points[ids]])
                else:
                    joint_t = pre_joint_t

            cl = Centerline.from_points(
                points, n_knots=check_specific(
                    kwargs, nid, 'n_knots', n_knots), force_extremes=check_specific(
                    kwargs, nid, 'force_extremes', force_extremes), curvature_penalty=check_specific(
                    kwargs, nid, 'curvature_penalty', curvature_penatly), pt_mode=check_specific(
                    kwargs, nid, 'pt_mode', 'project'), p=check_specific(
                        kwargs, nid, 'p', None))

            cl.id = nid
            if parents[nid] != 'None':
                cl.parent = parents[nid]
                cl.joint_t = joint_t
            cl_tree[nid] = cl

            for cid in cl_ids:
                if parents[cid] == nid:
                    add_to_tree(cid)

        for rid in cl_ids:
            if parents[rid] == 'None':
                add_to_tree(rid)

        return cl_tree
    #

    def translate(self, t, update=True):
        """
        Translate the CenterlineTree object, translating all the Centerline objects, with the translation vector t.

        Arguments
        ---------

            t : np.ndarray (3,)
                The translation vector.

            update : bool, optional
                Default True. Whether to rebuild the splines after the transformation.

        See Also
        --------
        :py:meth:`Centerline.translate`
        """

        for _, cl in self.items():
            cl.translate(t, update=update)
    #

    def scale(self, s, update=True):
        """
        Scale the CenterlineTree object, scaling all the Centerline objects, by a scalar factor s.

        Arguments
        ---------

            s : float
                The scale factor.

            update : bool, optional
                Default True. Whether to rebuild the splines after the transformation.

        See Also
        --------
        :py:meth:`Centerline.scale`
        """

        for _, cl in self.items():
            cl.scale(s, update=update)
    #

    def rotate(self, r, update=True):
        """
        Rotate the CenterlineTree, rotating all the Centerline objects, with the provided rotation matrix r.

        Arguments
        ---------

            r : np.ndarray (3, 3)
                The rotation matrix.

            update : bool, optional
                Default True. Whether to rebuild the splines after the transformation.

        See Also
        --------
        :py:meth:`Centerline.rotate`
        """

        for _, cl in self.items():
            cl.rotate(r, update=update)
    #
#


def extract_centerline(
        vmesh,
        params,
        params_domain=None,
        params_path=None,
        debug=False) -> CenterlineTree:
    """
    Provided a VascularMesh object with its boundaries properly defined, this function
    computes the CenterlineTree of it.

    Arguments
    ---------

        vmesh : VascularMesh
            The VascularMesh object where centerline is to be computed.

        params : dict
            The parameters for the spline approximation for each boundary, together with the grafting rate
            and tangent forcing parameters.

        params_domain : dict, opt
            The parameters for the domain extraction algorithm. More information about it in the
            domain_extractors module.

        params_path : dict
            The parameters for the path extraction algorithm. More information about it in the
            path_extractor module.

        debug : bool, opt
            Defaulting to False. Running in debug mode shows some plots at certain steps.


    Returns
    -------

        cl_tree : CenterlineTree
            The computed Centerline
    """

    cl_domain = extract_centerline_domain(
        vmesh=vmesh, params=params_domain, debug=debug)
    cl_paths = extract_centerline_path(
        vmesh=vmesh, cl_domain=cl_domain, params=params_path)
    cl_tree = CenterlineTree.from_multiblock_paths(
        cl_paths,
        knots=params['knots'],
        graft_rate=params['graft_rate'],
        force_extremes=params['force_extremes'])
    return cl_tree
