from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.integrate import quad
from scipy.interpolate import BSpline
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import KDTree

from .._base import SpatialObject, attribute_checker
from ..splines import UniSpline, uniform_penalized_spline
from ..utils.geometry import polyline_from_points
from ..utils.spatial import compute_ref_from_points, normalize
from .parallel_transport import ParallelTransport


class Curve(UniSpline, SpatialObject):
    """The curve class contains the main attributes and methods of a Bspline curve."""

    def __init__(self):
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

    def get_tangent(self, t, normalized=True):
        """
        Get the tangent of the curve at given parameter values.

        Values are clipped to parameter domain, as in constant extrapolation.

        Parameters
        ----------
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

    def compute_samples(self, n_samples=None):
        """
        Build the self.samples attribute.

        The samples are also included in a k-d tree for fast closest point query.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to compute. If passed, the attribute
            self.n_samples_curve is updated. The default is None.
        """
        if n_samples is not None:
            self.n_samples = n_samples

        self.parameter_samples = np.linspace(self.t0, self.t1, num=self.n_samples)
        self.samples = self.evaluate(t=self.parameter_samples)

        self.kdt = KDTree(self.samples)

    def compute_local_ref(self):
        """Compute the object local axes."""

        if self.samples is None:
            self.compute_samples()
        c, e1, e2, e3 = compute_ref_from_points(points=self.samples)
        self.center = c
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3

    def compute_parallel_transport(self, p=None, mode="project"):
        """
        Build the adapted frame.

        If mode == 'project':
            - If a point is passed, the vector p-c(t0) is projected onto the normal plane at t0 and
            used as initial condition for parallel transport.
            - If no point is passed, the mass center of the curve is used as p.

        if mode == 'as_is':
            - The argument p must be the vector to be parallel transported.

        Parameters
        ----------
            p : np.ndarray (3,)
                The point/vector to use.

            mode : {'project', 'as_is'}
                The chosen mode to use.

        Returns
        -------
            v : ParallelTransport
                The ParallelTransport object representing the transport of the vector along the
                curve.

        See Also
        --------
        compute_adapted_frame
        ParallelTransport.compute_parallel_transport_on_curve
        """

        if mode == "project":
            if p is None:
                p = self.center

            i2p = normalize(p - self.evaluate(self.t0))
            t_0 = self.get_tangent(self.t0)
            v0 = normalize(i2p - t_0.dot(i2p) * t_0)

        elif mode == "as_is":
            if p is None:
                raise ValueError(f"Cannot build parallel transport with mode: {mode} and p: {p}")

            else:
                v0 = p
        else:
            raise ValueError(
                f"Wrong mode passed: mode = {mode}. Available options are {'project', 'as_is'}."
            )

        v = ParallelTransport.compute_parallel_transport_along_curve(curve=self, v0=v0)
        return v

    def compute_adapted_frame(self, p=None, mode="project"):
        """
        Compute a parallel transported adapted frame.


        This frame {t, v1, v2} is an stable alternative to Frenet frame and has multiple purposes.
        The argument p can be used to provide a preferred direction for v1. In turn, v2 is the cross
        product of t and v1 for orientation and orthonormality reasons. This method uses
        compute_parallel_transport method, you may be interested in checking documentation.

        Parameters
        ----------
        p : np.ndarray (3,), optional
            Default None. A reference vector/point used to define the initial v1. This argument can
            be used in the following ways:

            If no initial vector is passed (p=None), the argument mode is ignored and two options
            are available:

                - If the attribute self.v1 is None:
                    Then an initial v1 is computed as the cross product of the tangent of the
                    centerline and the axis of least variability on the centerline,
                    i.e. v1_0 = normalize(t(0), e3)

                - Otherwise, the attribute self.v1 must be a ParallelTransport object with the
                initial condition set in its attribute v0, i.e.
                self.v1.__class__ == ParallelTransport and self.v1.v0 is not None.
                And this vector is used to build the parallel transport.

            If an initial vector is passed p = (p0, p1, p2) the two options available are described
            in compute_parallel_transport method and are controlled by the mode argument.
        mode : {'project', 'as_is'}, optional
            The mode used to built the adapted frame if p is passed. Check compute_parallel_transport.

        See Also
        --------
        compute_parallel_transport
        ParallelTransport.compute_parallel_transport_on_centerline
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
                    self.v1 = self.compute_parallel_transport(p=self.v1.v0, mode="as_is")
                else:
                    raise ValueError(
                        f"Wrong usage of compute_adapted_frame. No p {(p)} has been passed but "
                        + "self.v1 is a ParallelTransport object with v0 == None."
                    )
        else:
            self.v1 = self.compute_parallel_transport(mode=mode, p=p)

        v2_0 = normalize(np.cross(self.get_tangent(self.t0), self.v1.v0))
        self.v2 = self.compute_parallel_transport(mode="as_is", p=v2_0)

    def build(self, samples=True, local_ref=True, adapted_frame=True):
        """Build the splines and sets up useful attributes."""

        super().build()
        self.tangent = self._spl.derivative()

        # Update functions that depend on centerline.
        if samples:
            self.compute_samples()
        if local_ref:
            self.compute_local_ref()
        if adapted_frame:
            self.compute_adapted_frame(mode="project", p=None)

    def get_projection_parameter(self, p, method="scalar", full_output=False):
        """
        Compute the value of the parameter for the point in the curve closest to p.

        Parameters
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
            In all cases, constrained minimization is used to force the
            value of the parameter to be in [self.t0, self.t1]. The default is 'scalar'.
            - 'sample' : the optimization is avoided by keeping the closest
            sampled centerline point.
        full_output : bool
            Whether to return the distance and the value of the parameter
            or not. Default is False.

        Returns
        -------
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
        if method.startswith("vec") or method == "sample":
            if method == "vec_jac":
                res = minimize(
                    dist_to_centerline_point,
                    t,
                    jac=deriv,
                    method="trust-constr",
                    bounds=[(self.t0, self.t1)],
                )
                d = float(res.fun)
                x = float(res.x)
            elif method == "vec":
                res = minimize(
                    dist_to_centerline_point, t, method="trust-constr", bounds=[(self.t0, self.t1)]
                )
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

            res = minimize_scalar(dist_to_centerline_point, method="bounded", bounds=[s0, s1])
            d = float(res.fun)
            x = float(res.x)

        if full_output:
            return x, d

        return x

    def get_projection_point(self, p, method="scalar", full_output=False):
        """
        Compute the point in the centerline closest to p.

        Parameters
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

    def get_adapted_frame(self, t):
        """
        Get the adapted frame at a centerline point of parameter t.

        The adapted frame is defined as:

                    {t, v1, v2}

        where v1 and v2 are the parallel transported vectors and t, the tangent.

        Parameters
        ----------
        t : float
            The parameter value for evaluation

        Returns
        -------
        t_  : np.ndarray
            The tangent.
        v1 : numpy.array
            The v1 vector of the adapted frame.
        v2 : numpy.array
            The v2 vector of the adapted frame.
        """

        attribute_checker(self, ["tangent", "v1", "v2"], info="Cant compute adapted frame: ")

        t_ = self.get_tangent(t)
        v1 = self.v1(t)
        v2 = self.v2(t)

        return t_, v1, v2

    def get_frenet_normal(self, t):
        """
        Get the normal vector of the frenet frame at centerline point of parameter t.

        Returns n_ computed as:

                n_ = b_ x t_

        where b and t_ are the binormal and tangent respectively and x is the
        cross product.


        Parameters
        ----------
        t : float
            The parameter where normal is to be computed.

        Returns
        -------
        n_ : numpy.array
            The normal of the centerline curve.

        """

        b = self.get_frenet_binormal(t)
        t = self.get_tangent(t)

        return np.cross(b, t)

    def get_frenet_binormal(self, t):
        """
        Get the binormal vector of the frenet frame at centerline point.

        Returns b computed as:

                    b_ =  C' x C'' /|| C' x C''||,

        where C is the parametrization of the centerline curve and x the
        cross product.


        Parameters
        ----------
        height : float
            The parameter where binormal is to be computed.

        Returns
        -------
        b : numpy.array
            The binormal of the centerline

        """

        cp = self.get_tangent(t, normalized=False)
        cpp = self.tangent(t, nu=1)
        b = np.cross(cp, cpp)

        return normalize(b)

    def get_parametrization_velocity(self, t):
        """
        Compute the velocity of the centerline parametrization, C(t), as ||C'(t)||.

        Parameters
        ----------
        t : float, array-like
            The parameter where velocity is to be computed.

        Returns
        -------
        velocity : float, np.ndarray
        """

        if isinstance(t, (float, int)):
            velocity = np.linalg.norm(self.get_tangent(t, normalized=False))
        else:
            velocity = np.linalg.norm(self.get_tangent(t, normalized=False), axis=1)

        return velocity

    def get_arc_length(self, b=None, a=None):
        """
        Compute the arc length of the centerline.

        It is computed according to:

                    L_c(a,b) = int_a^b ||c'(t)|| dt.

        Since the centerline is a piecewise polynomial (spline curve)
        each integration is carried out in each polynomial segment
        to improve accuracy.

        Parameters
        ----------
        b : float
            Default t1. The upper parameter to compute length
        a : float
            Default t0. The lower parameter to compute length

        Returns
        -------
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
            l += quad(self.get_parametrization_velocity, segments[i], segments[i + 1])[0]

        return l

    def travel_distance_parameter(self, d, a=None):
        """
        Get the parameter resulting from traveling a distance d, from an initial
        parameter a. Note that if d is negative, the distance will be traveled
        in reverse direction to centerline parameterization.

        Parameters
        ----------
        d : float
            The signed distance to travel.
        a : float
            Default is self.t0. The initial parameter where to start the
            traveling.

        Returns
        -------
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

        res = minimize_scalar(fun=f, bounds=bounds, method="bounded")

        return res.x

    def get_curvature(self, t):
        """
        Get the curvature of the centerline at a given parameter value.
        The curvature is computed assuming the general formula,.

                    k = || C' x C''|| / ||C'||^3,

        where C is the parametrization of the centerline.

        Parameters
        ----------
        t : float
            Parameter on the centerline domain.

        Returns
        -------
        k : float
            Curvature of the centerline at given parameter.
        """

        C_p = self.get_tangent(t)
        C_pp = self.tangent(t, nu=1)
        num = np.linalg.norm(np.cross(C_p, C_pp))
        den = np.linalg.norm(C_p) ** 3
        k = num / den
        return k

    def get_torsion(self, t, dt=1e-4):
        """
        Get the torsion of the centerline at a certain distance from the valve.
        The torsion is computed by numerical differentiation of the binormal
        vector of the Frenet frame,.

                    t = ||b'||,

        where b is the binormal vector.

        Parameters
        ----------
        height : float
            Distance from the valve or in other words, point on the centerline domain.

        Returns
        -------
        t : float
            torsion of the centerline at height point.
        """

        assert dt > 0, "Wrong value for differential step. It must be greater than 0."

        if t == self.t0:
            b_der = (
                self.get_frenet_binormal(self.t0 + dt) - self.get_frenet_binormal(self.t0)
            ) / dt
        elif t == self.t1:
            b_der = (
                self.get_frenet_binormal(self.t0 + dt) - self.get_frenet_binormal(self.t0)
            ) / dt
        else:
            dt = min(dt, abs(self.t0 - dt), abs(self.t1 - dt))
            b_der = (self.get_frenet_binormal(t + dt) - self.get_frenet_binormal(t - dt)) / dt
        torsion = -np.linalg.norm(b_der)
        return torsion

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


        Parameters
        ----------
        a : float
            Default t0. The lower bound of the interval.
            Must be greater than or equal to 0 and lower than b.
        b : Default t1. The upper bound of the interval.
            Must be lower than or equal to 1 and greater than a.

        Returns
        -------
        k : float
            The mean curvature estimated.

        """

        if a is None:
            a = self.t0
        if b is None:
            b = self.t1

        if a < 0:
            raise ValueError(f"Value of a {a} is lower than 0")
        if b < 0:
            raise ValueError(f"Value of b {b} is greater than 0")
        if b < a:
            raise ValueError(f"Value of a:{a} is greater than value of b:{b}")

        # Get the segments
        segments = self.get_knot_segments(a=a, b=b)

        # Compute the curvatures at the segments
        k = 0
        for i in range(len(segments) - 1):
            k += quad(self.get_curvature, segments[i], segments[i + 1])[0]

        k /= self.get_arc_length(b=b, a=a)
        return k

    def to_polydata(self, t_res=None, add_attributes=True):
        """
        Transform curve into a PolyData based on points and lines.

        Parameters
        ----------
        t_res : int, opt
            The number of points in which to discretize the curve.
        add_attributes : bool, opt
            Default True. If true, all the attributes necessary to build the splines and its
            hierarchical relations are added in the PolyData's user_dict.

        Returns
        -------
        poly : pv.PolyData
            A PolyData object with polyline topology defined.

        See Also
        --------
        from_polydata
        """

        params = self.parameter_samples
        points = self.samples
        if t_res is not None:
            params = np.linspace(self.t0, self.t1, t_res)
            points = self(params)

        poly = polyline_from_points(points)

        poly["params"] = params
        poly["tangent"] = self.tangent(params)
        poly["v1"] = self.v1(params)
        poly["v2"] = self.v2(params)

        if add_attributes:
            # Adding Spline atts:
            poly.user_dict["t0"] = self.t0
            poly.user_dict["t1"] = self.t1
            poly.user_dict["k"] = self.k
            poly.user_dict["n_knots"] = self.n_knots
            poly.user_dict["coeffs"] = self.coeffs.tolist()
            poly.user_dict["extrapolation"] = self.extrapolation

        return poly

    def save(self, fname, binary=True):
        """
        Save the centerline object as a vtk PolyData, appending the essential attributes as field
        data entries.

        Parameters
        ----------
        fname : str
            Filename to write to. If does not end in .vtk, the extension is appended.
        binary : bool, opt
            Default True. Whether to write the file in binary or ASCII format.
        """

        poly = self.to_polydata(add_attributes=True)
        poly.save(filename=fname, binary=binary)

    def from_polydata(self, poly: pv.PolyData) -> Curve:
        """
        Build a curve object from a pyvista PolyData object that contains the required
        attributes as field_data. The minimum required data are the parameters involving the spline
        creation, namely, {'interval' 'k' 'knots' 'coeffs' 'extrapolation'}.

        Parameters
        ----------
        poly : pv.PolyData

        Returns
        -------
        self : Curve
            The curve object with the attributes already set.
        """

        atts = ["t0", "t1", "k", "n_knots", "coeffs", "extrapolation"]
        for att in atts:
            if att not in poly.user_dict:
                raise AttributeError(
                    f"Could not find attribute: {att} in polydata. Can't build curve object"
                )

        for att in atts:
            value = poly.user_dict[att]

            if att == "coeffs":
                self.set_parameters(coeffs=np.array(value))
            else:
                self.set_parameters(**{att: value})
        self.build()

        if "v1" in poly.user_dict:
            self.compute_adapted_frame(p=poly.get_array("v1", preference="point")[0], mode="as_is")

        return self

    @staticmethod
    def read(fname) -> Curve:
        """
        Read curve object from a vtk file.

        Parameters
        ----------
            fname : str
                The name of the file storing the curve.
        """

        poly = pv.read(fname)
        return Curve().from_polydata(poly)

    def trim(self, t0_, t1_=None, trim_knots=False, n_samps=100) -> Curve:
        """
        Trim the curve from t0_ to t1_ and return the new segment as a curve object.

        If pass_atts is true all the curve attributes such as the v1, v2 and others are kept.
        The amount of knots for the trimmed curve will be computed taking into account the
        amount of knot_segments in the interval [t0_, t1_].

        Parameters
        ----------
        t0_, t1_ : float
            The lower and upper extrema to trim. If t1_ is None, self.t1 is assumed.
        trim_knots : bool, optional
            Default False. If true the number of knots is reduced to keep the spacing as it
            was in the untrimmed curve. Otherwise, the number of knots is kept.
        n_samps : int, optional
            Default 100. The amount of samples to generate to perform the approximation.

        Returns
        -------
        cl : Curve
            The trimmed curve.
        """

        if t1_ is None:
            t1_ = self.t1

        ts = np.linspace(t0_, t1_, n_samps)
        n_knots = self.n_knots
        if trim_knots:
            n_knots = len(self.get_knot_segments(t0_, t1_)) - 2
        spl = uniform_penalized_spline(
            points=self(ts),
            n_knots=n_knots,
            k=self.k,
            # Normalized domain
            param_values=(ts - t0_) / (t1_ - t0_),
            force_ini=True,
            force_end=True,
            curvature_penalty=0.0,
        )

        curve = self.__class__()
        curve.set_parameters(
            build=True,
            t0=spl.t[0],
            t1=spl.t[-1],
            k=spl.k,
            knots=spl.t,
            coeffs=spl.c,
            n_knots=len(spl.t) - 2 * (spl.k + 1),
            extra="linear",
        )

        return curve

    @staticmethod
    def from_points(
        points,
        n_knots,
        k=3,
        curvature_penalty=1.0,
        param_values=None,
        pt_mode="project",
        p=None,
        force_extremes=True,
        curve=None,
    ) -> Curve:
        """
        Build a Curve object from a list of points.

        The amount knots to perform the LSQ approximation must be provided. An optional vector p can
        be passed to build the adapted frame.

        Parameters
        ----------
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
            of the curve is approximated assuming cl(param_values) = points. If None
            provided the normalized cumulative distance among the points is used.
        pt_mode : str
            The mode option to build the adapted frame by parallel transport.
            If p is not passed pt_mode must be 'project'. See compute_parallel_transport
            method for extra documentation.
        p : np.ndarray
            The initial v1. If pt_mode == 'project' it is projected onto inlet plane.
        force_extremes : {False, True, 'ini', 'end'}
            Default True. Whether to force the curve to interpolate the boundary behavior
            of the approximation. If True the first and last point are interpolated and its
            tangent is approximated by finite differences using the surrounding points. If
            'ini', respectively 'end', only one of both extremes is forced.
        curve : Curve
            A Curve object to be used. All the data will be overwritten.

        Returns
        -------
        curve : Curve
            The curve object built from the points passed.
        """

        spl = uniform_penalized_spline(
            points=points,
            n_knots=n_knots,
            k=k,
            param_values=param_values,
            force_ini=force_extremes in [True, "ini"],
            force_end=force_extremes in [True, "end"],
            curvature_penalty=curvature_penalty,
        )

        if curve is None:
            curve = Curve()

        curve.set_parameters(
            build=True,
            t0=spl.t[0],
            t1=spl.t[-1],
            k=spl.k,
            knots=spl.t,
            coeffs=spl.c,
            n_knots=n_knots,
            extra="linear",
        )

        curve.compute_adapted_frame(mode=pt_mode, p=p)

        return curve

    def translate(self, t):
        """
        Translate the curve.

        All the affine transformations are applied to the coefficients and then the rest of
        attributes are a recomputed.

        Parameters
        ----------
        t : np.ndarray (3,)
            The translation vector.
        """

        if self.coeffs is not None:
            self.coeffs += t.reshape(
                3,
            )
            self.build()

    def scale(self, s):
        """
        Scale the curve.

        All the affine transformations are applied to the coefficients and then the rest of
        attributes are a recomputed.

        Parameters
        ----------
        s : float
            The scale factor.
        """

        if self.coeffs is not None:
            self.coeffs *= s
            self.build()

    def rotate(self, r):
        """
        Rotate the curve.

        All the affine transformations are applied to the coefficients and then the rest of
        attributes are a recomputed.

        Parameters
        ----------
        r : np.ndarray (3, 3)
            The rotation matrix.
        """

        # ensure normality of the rotation matrix columns
        r /= np.linalg.norm(r, axis=0)

        if self.coeffs is not None:
            self.coeffs = (r @ self.coeffs.T).T
            self.build(adapted_frame=False)
            self.v1.rotate(r=r)
            self.v2.rotate(r=r)
