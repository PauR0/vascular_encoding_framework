from __future__ import annotations

import numpy as np
import pyvista as pv

from ..messages import error_message
from ..utils._code import Node, attribute_checker, broadcast_kwargs
from ..utils.misc import split_metadata_and_fv
from ..utils.spatial import get_theta_coord
from .curve import Curve
from .parallel_transport import ParallelTransport


class Centerline(Curve, Node):
    """
    The centerline class contains the main attributes and methods of a Bspline
    curve that models the centerline of a branch.
    """

    def __init__(self):
        # Hierarchy data
        Node.__init__(self=self)
        # The parameter of the joint at parent centerline
        self.joint_t: float = None

        # Geometry data
        Curve.__init__(self=self)

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
        self, tau: float, theta: float, rho: float, grid=False, gridded=False, full_output=False
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
            poly.user_dict["joint_t"] = self.joint_t

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

        It must contain the required in user_dict. The minimum required data are the parameters
        involving the spline building, namely, {'interval' 'k' 'knots' 'coeffs' 'extrapolation'}.
        Additionally, if the centerline belong to a tree, it is advised to also include the
        attributes {'id', 'parent', 'children', 'joint_t'} in the PolyData user_dict.

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
        node_atts = list(Node().__dict__) + ["joint_t"]
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

    def get_metadata(self) -> np.ndarray:
        """
        Return a copy of the metadata array.

        As of this code version the
        metadata array is [7, k, n_knots, n_samples, v1(t0)[0], v1(t0)[1], v1(t0)[2]].

        Returns
        -------
        md : np.ndarray

        See Also
        --------
        set_metadata
        to_feature_vector
        from_feature_vector
        """

        v1 = self.v1(self.t0)
        md = np.array([7, self.k, self.n_knots, self.n_samples, v1[0], v1[1], v1[2]])

        return md

    def set_metadata(self, md):
        """
        Extract and set the attributes from a the metadata array.

        As of this code version the metadata array is expected to be
                [6, k, n_knots, n_samples, v1(t0)[0], v1(t0)[1], v1(t0)[2]].


        Parameters
        ----------
        md : np.ndarray or array-like
            The metadata array.
        build : bool, optional
            Default False. Whether to build the splines after setting the metadata.
            This should be set to true if parameters such as coefficients have been
            previously set.

        See Also
        --------
        to_feature_vector
        from_feature_vector
        """

        self.set_parameters(
            build=False,
            k=round(md[1]),
            n_knots=round(md[2]),
            n_samples=round(md[3]),
        )

        self.v1 = ParallelTransport()
        self.v1.v0 = md[4:]

    def get_feature_vector_length(self) -> int:
        """
        Return the length of the feature vector considering the spline parameters.

        If n is the amount of internal knots, and k is the degree of the BSpline polynomials,
        the length of the centerline feature vector is computed as: 3(n+k+1). The multiplication
        by 3 is due to the three components of the coefficients (a.k.a. control points.).


        Returns
        -------
        l : int
            The length of the centerline feature vector.

        """
        if not attribute_checker(
            self, ["n_knots", "k"], info="Cannot compute the Centerline feature vector length."
        ):
            return None
        l = 3 * (self.n_knots + self.k + 1)
        return l

    def to_feature_vector(self, add_metadata=True) -> np.ndarray:
        """
        Convert the Centerline object to its feature vector representation.

        The feature vector version of a Centerline consist in appending the raveled centerline
        coefficients. If add_metadata is True (which is the default), a metadata vector is appended
        at the beginning of the feature vector. The first entry of the metadata vector is the amount
        of metadata in total, making it look like [n, md0, ..., mdn], read more about it in get.

        Parameters
        ----------
        add_metadata: bool, optional
            Default True. Wether to append metadata at the beginning of the feature vector.

        Returns
        -------
        fv : np.ndarray
            The feature vector according to mode. The shape of each feature vector changes
            accordingly.

        Notes
        -----
        Note that the feature vector representation does not bear any hierarchical data, not even
        if add_metadata is True. Be sure that hierarchical data is properly stored if will be later
        required. For storage purposes check to_multiblock method.


        See Also
        --------
        get_metadata
        from_feature_vector

        """

        fv = self.coeffs.ravel()

        if add_metadata:
            fv = np.concatenate([self.get_metadata(), fv])

        return fv

    @staticmethod
    def from_feature_vector(fv, md=None) -> Centerline:
        """
        Build a Centerline object from a feature vector.

        Note that in order to build the Centerline, the feature vector must start with the metadata
        array or it must be passed using the md argument. Read more about the metadata array at
        get_metadata method docs.


        Parameters
        ----------
        fv : np.ndarray (N,)
            The feature vector with the metadata at the beginning.
        md : np.ndarray (M,)
            The metadata array to use. If passed, it will be assumed that fv does not
            contain it at the beginning.

        Returns
        -------
        cl : Centerline
            The Centerline object built from the feature vector.

        See Also
        --------
        to_feature_vector
        get_metadata
        """

        cl = Centerline()

        if md is None:
            md, fv = split_metadata_and_fv(fv)

        cl.set_metadata(md)

        l = cl.get_feature_vector_length()
        if len(fv) != l:
            error_message(
                f"Cannot build a Centerline object from feature vector. Expected n_knots+(k+1)={l} coefficients and {len(fv)} were provided."
            )
            return None

        cl.set_parameters(
            build=True,
            coeffs=fv.reshape(-1, 3),
            extra="linear",
        )

        return cl
