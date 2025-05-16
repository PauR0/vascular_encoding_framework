from __future__ import annotations

from typing import Any

import numpy as np
import pyvista as pv

from .._base import Encoding, Node, attribute_checker, broadcast_kwargs
from ..utils.spatial import get_theta_coord
from .curve import Curve
from .parallel_transport import ParallelTransport


class Centerline(Curve, Node, Encoding):
    """
    The centerline class contains the main attributes and methods of a Bspline
    curve that models the centerline of a branch.
    """

    def __init__(self):
        # Hierarchy data
        Node.__init__(self=self)
        # The parameter of the joint at parent centerline
        self.tau_joint: float = None

        # Geometry data
        Curve.__init__(self=self)

        # Encoding data
        self._hyperparameters: list[str] = [
            # Hierarchy
            "id",
            "parent",
            "children",
            "tau_joint",
            # Curve
            "t0",
            "t1",
            "k",
            "n_knots",
            "extrapolation",
            "n_samples",
            "v1_0",
        ]
        Encoding.__init__(self=self)

    def __str__(self):
        """Return the node data of the centerline as string."""
        return Node.__str__(self=self)

    def cartesian_to_vcs(self, p, method="scalar"):
        """
        Given a 3D point p expressed in cartesian coordinates, this method
        computes its expression in the Vessel Coordinate System (VCS).

        Parameters
        ----------
        p : np.ndarray (3,)
            A 3D point in cartesian coordinates.
        method : Literal{'scalar', 'vec', 'vec_jac'}, opt
            The minimization method to use. See get_projection_parameter
            for more info.

        Returns
        -------
        p_vcs : np.ndarray(3,)
            The coordinates of the point in the VCS.

        """

        tau, rho = self.get_projection_parameter(p, method=method, full_output=True)
        theta = get_theta_coord(p, self(tau), self.v1(tau), self.v2(tau))
        return np.array((tau, theta, rho))

    def vcs_to_cartesian(
        self,
        tau: float,
        theta: float,
        rho: float,
        grid=False,
        gridded=False,
        full_output=False,
    ):
        """
        Given a point expressed in Vessel Coordinate System (VCS), this method
        computes its cartesian coordinates.

        Using numpy broadcasting this method allows working with arrays of vessel
        coordinates.

        Parameters
        ----------
        tau : float or array-like (N,)
            The longitudinal coordinate of the point
        theta : float or array-like (N,)
            Angular coordinate of the point
        rho : float or array-like (N,)
            The radial coordinate of the point
        grid : bool, optional
            Default False. If true, the method returns the cartesian representation of the
            grid tau x theta x rho.
        gridded: bool, optional
            Whether the input comes in a gridded way i.e. tau, theta, and rho have been generated
            by a function like numpy meshgrid.
        full_output : bool, false
            Default False. Whether to return the as well the vcs. Useful in combination with grid.

        Returns
        -------
        p : np.ndarray (N, 3)
            The point in cartesian coordinates.
        tau, theta, rho : np.ndarray (N, ), opt.
            If full_output is True, the vessel coordinates of the points are returned.
        """

        if not gridded:
            tau, theta, rho = broadcast_kwargs(tau=tau, theta=theta, rho=rho).values()

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

        p = self(tau) + rho * (self.v1(tau) * np.cos(theta) + self.v2(tau) * np.sin(theta))

        if full_output:
            return p, tau, theta, rho

        return p

    def plot_adapted_frame(
        self,
        vmesh: pv.PolyData = None,
        plotter: pv.Plotter = None,
        scale: float = 1,
        show: bool = True,
    ):
        """
        Plot the parallel transported frame.

        Parameters
        ----------
        vmesh : pv.PolyData
            The vascular mesh used to compute the centerline.
        plotter : pv.Plotter
            Default None. If passed, parallel_transport is displayed there.
        scale : float, opt
            By default no scale is applied. The scale of the arrows used to plot the adapted frame
            vectors.
        show : bool, opt
            Default True. Whether to show the plot or not.
        """

        if plotter is None:
            plotter = pv.Plotter()

        if vmesh is not None:
            plotter.add_mesh(vmesh, opacity=0.5, color="w")

        pdt = self.to_polydata()

        if scale is None:
            scale = pdt.length / 50

        tgts = pdt.glyph(orient="tangent", scale=False, factor=scale)
        plotter.add_mesh(tgts, color="r")

        v1 = pdt.glyph(orient="v1", scale=False, factor=scale)
        plotter.add_mesh(v1, color="g")

        v2 = pdt.glyph(orient="v2", scale=False, factor=scale)
        plotter.add_mesh(v2, color="b")

        if show:
            plotter.show()

    def to_polydata(self, tau_res=None, add_attributes=False):
        """
        Transform centerline into a PolyData based on points and lines.

        Parameters
        ----------
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

        poly = super().to_polydata(t_res=tau_res, add_attributes=add_attributes)

        if add_attributes:
            # Adding Node atts:
            poly.user_dict["id"] = self.id
            poly.user_dict["parent"] = self.parent
            poly.user_dict["children"] = list(self.children)
            poly.user_dict["tau_joint"] = self.tau_joint

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

    def from_polydata(self, poly) -> Centerline:
        """
        Build a centerline object from a pyvista PolyData.

        It must contain the hyperparameters in user_dict.

        Parameters
        ----------
        poly : pv.PolyData

        Returns
        -------
        self : Centerline
            The centerline object with the attributes already set.
        """

        super().from_polydata(poly=poly)

        # Add Node attributes if present
        node_atts = list(Node().__dict__) + ["tau_joint"]
        for att in node_atts:
            if att in poly.user_dict:
                value = poly.user_dict[att]
                self.set_data(**{att: value})

        return self

    @staticmethod
    def read(fname) -> Centerline:
        """
        Read centerline object from a vtk file.

        Parameters
        ----------
            fname : str
                The name of the file storing the centerline.
        """

        poly = pv.read(fname)
        return Centerline().from_polydata(poly)

    def trim(
        self,
        tau_0: float,
        tau_1: float = None,
        trim_knots: bool = False,
        n_samps: int = 100,
    ) -> Centerline:
        """
        Trim the centerline from tau_0 to tau_1 and return the new segment as a Centerline object.

        The amount of knots for the trimmed curve will be computed taking into account the amount of
        knot_segments in the interval [tau_0, tau_1].

        Parameters
        ----------
        tau_0, tau_1 : float
            The lower and upper extrema to trim. If t1_ is None, self.t1 is assumed.
        trim_knots : bool, optional
            Default False. If true the number of knots is adapted to keep the spacing as it was in
            the untrimmed curve. Otherwise, the number of knots is kept but the spacing is modified.
        n_samps : int, optional
            Default 100. The amount of samples to generate to perform the approximation.

        Returns
        -------
        cl : Centerline
            The trimmed Centerline.
        """

        cl: Centerline = super().trim(t0_=tau_0, t1_=tau_1, trim_knots=trim_knots, n_samps=n_samps)

        cl.set_data_from_other_node(nd=self, extra=["tau_joint"])

        return cl

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
        cl=None,
    ) -> Centerline:
        """
        Build a Centerline object from a list of points.

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
            The Centerline object built from the points passed.
        """

        if cl is None:
            cl = Centerline()

        cl = Curve.from_points(
            points=points,
            n_knots=n_knots,
            k=k,
            curvature_penalty=curvature_penalty,
            param_values=param_values,
            pt_mode=pt_mode,
            p=p,
            force_extremes=force_extremes,
            curve=cl,
        )

        return cl

    def get_hyperparameters(self) -> dict[str, Any]:
        """
        Get the hyperparameter dictionary of the object.

        Returns
        -------
        hp : dict[str, Any]
            The json-serializable hyperparameter dictionary.

        See Also
        --------
        set_hyperparameters
        """

        return super().get_hyperparameters(v1_0=self.v1(self.t0))

    def set_hyperparameters(self, hp: dict[str, Any]):
        """
        Set the attributes from a hyperparameter dictionary.

        Parameters
        ----------
        hp : dict[str, Any]
            The hyperparameter dict.

        See Also
        --------
        to_feature_vector
        from_feature_vector
        """

        self.set_parameters(**{p: v for p, v in hp.items() if p != "v1_0"})

        self.v1 = ParallelTransport()
        self.v1.v0 = np.array(hp["v1_0"])

    def get_feature_vector_length(self) -> int:
        """
        Return the feature vector's length of a Centerline object.

        If n is the amount of internal knots, and k is the degree of the BSpline polynomials,
        the length of the centerline feature vector is computed as: 3(n+k+1). The multiplication
        by 3 is due to the three components of the coefficients (a.k.a. control points).

        Returns
        -------
        l : int
            The length of the centerline feature vector.

        """
        attribute_checker(
            self,
            ["n_knots", "k"],
            info="Can't compute the Centerline feature vector length.",
        )

        l = 3 * (self.n_knots + self.k + 1)
        return l

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert the Centerline object to its feature vector representation.

        The feature vector version of a Centerline consist in appending the raveled centerline
        coefficients.

        > Note that the feature vector itself lacks the hyperparameter data. To be able to re-build
        a centerline object from its representation the hyperparameters are needed as well.

        Returns
        -------
        fv : np.ndarray
            The feature vector representation.


        See Also
        --------
        get_hyperparameters
        from_feature_vector

        """

        return self.coeffs.ravel()

    def from_feature_vector(self, fv, hp=None) -> Centerline:
        """
        Build a Centerline object from a feature vector.

        > Note that while hyperparameters argument is optional it must have been previously set or
        passed.


        Parameters
        ----------
        fv : np.ndarray (N,)
            The feature vector with the metadata at the beginning.
        hp : np.ndarray (M,)
            The hyperparameter dictionary to use.

        Returns
        -------
        cl : Centerline
            The Centerline object built from the feature vector.

        See Also
        --------
        to_feature_vector
        get_hyperparameters
        """

        if hp is not None:
            self.set_hyperparameters(hp)

        l = self.get_feature_vector_length()
        if len(fv) != l:
            raise ValueError(
                "Cannot build a Centerline object from feature vector."
                + f"Expected n_knots+(k+1)={l} coefficients and {len(fv)} were provided."
            )

        self.set_parameters(build=True, coeffs=fv.reshape(-1, 3))

        return self
