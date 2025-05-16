from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._base import Encoding, attribute_checker
from ..splines import BiSpline, uniform_penalized_bivariate_spline

if TYPE_CHECKING:
    from ..centerline import Centerline


class Radius(BiSpline, Encoding):
    """Radius or Wall function class."""

    def __init__(self):
        BiSpline.__init__(self=self)

        self.x0 = 0
        self.x1 = 1
        self.extra_x = "constant"

        self.y0 = 0
        self.y1 = 2 * np.pi
        self.extra_y = "periodic"

        self._hyperparameters = [
            "x0",
            "x1",
            "kx",
            "n_knots_x",
            "extra_x",
            "y0",
            "y1",
            "ky",
            "n_knots_y",
            "extra_y",
        ]
        Encoding.__init__(self=self)

    def set_parameters_from_centerline(self, cl: Centerline):
        """
        Set the radius bounds equal to the passed Centerline object.

        Parameters
        ----------
        cl : Centerline
            The Centerline of the vessel
        """

        self.x0 = cl.t0
        self.x1 = cl.t1

    def get_hyperparameters(self) -> dict[str, Any]:
        """
        Get the hyperparameter dictionary of the Radius object.

        Returns
        -------
        hp : dict[str, Any]

        See Also
        --------
        set_hyperparameters

        """

        return super().get_hyperparameters()

    def set_hyperparameters(self, hp: dict[str, Any]):
        """
        Set hyperparameters from a dictionary.

        See Also
        --------
        get_metadata
        """

        self.set_parameters(build=False, **hp)

    def get_feature_vector_length(self):
        """
        Return the length of the feature vector considering the spline parameters.

        If nx, ny are the amount of internal knots in each component and kx, ky are the degrees of
        the polynomial BSplines of each component, the length of the radius feature vector is
        (nx+kx+1)*(ny*ky+1).

        Returns
        -------
        rk : int
            The length of the radius feature vector.

        """

        attribute_checker(
            self,
            ["n_knots_x", "n_knots_y", "kx", "ky"],
            info="Cannot compute the Radius feature vector length.",
        )

        rk = (self.n_knots_x + self.kx + 1) * (self.n_knots_y + self.ky + 1)
        return rk

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert the Radius object to its feature vector representation.

        The feature vector version of a Radius object consist in the raveled radius coefficients.

        Returns
        -------
        : np.ndarray
            The feature vector of the Radius object.

        See Also
        --------
        get_metadata
        from_feature_vector
        """

        return self.coeffs.ravel()

    def from_feature_vector(self, fv: np.ndarray, hp: dict[str, Any] = None):
        """
        Build a Radius object from a feature vector.

        > Note that while hyperparameters argument is optional it must have been previously set or
        passed.

        Parameters
        ----------
        fv : np.ndarray (N,)
            The feature vector with the metadata at the beginning.
        hp : dict[str, Any], optional
            The hyperparameter dictionary.

        Returns
        -------
        rd : Radius
            The Radius object built from the feature vector.

        See Also
        --------
        to_feature_vector
        get_metadata
        """

        if hp is not None:
            self.set_hyperparameters(hp)

        r, k = (self.n_knots_x + self.kx + 1), (self.n_knots_y + self.ky + 1)
        rk = r * k
        if len(fv) != rk:
            raise ValueError(
                f"Cannot build a Radius object from feature vector. Expected {rk} knots "
                + f"((tx+kx+1) * (ty+ky+1)) coefficients and {len(fv)} were provided."
            )

        self.set_parameters(build=True, coeffs=np.array(fv))

        return self

    @staticmethod
    def from_points(points, tau_knots, theta_knots, laplacian_penalty=1.0, cl=None, debug=False):
        """
        Build a Radius object from an array of points in the Vessel Coordinate System.


        Radius object are a specialized Bivariate Splines. This function allow to build such objects
        by performing a least squares approximation using the longitudinal and angular coordinates
        to model the radius.

        Parameters
        ----------
        points : np.ndarray (N, 3)
            The vessel coordinates point array to be approximated.
        tau_knots, theta_knots : int
            The number of internal knots in longitudinal and angular dimensions respectively.
            TODO: Allow building non-uniform BSplines.
        laplacian_penalty : float, optional
            Default 1.0. A penalty factor to apply on the laplacian for spline approximation
            optimization.
        cl : Centerline, optional
            Default None. The centerline associated to the radius.
        debug : bool, optional
            Default False. Whether to show plots during the fitting process.

        Returns
        -------
        rd : Radius
            The radius object built based on the passed points.
        """

        rd = Radius()
        if cl is not None:
            rd.set_parameters_from_centerline(cl)

        bispl = uniform_penalized_bivariate_spline(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            nx=tau_knots,
            ny=theta_knots,
            laplacian_penalty=laplacian_penalty,
            y_periodic=True,
            kx=rd.kx,
            ky=rd.ky,
            bounds=(rd.x0, rd.x1, rd.y0, rd.y1),
            debug=debug,
        )
        rd.set_parameters(
            build=True, n_knots_x=tau_knots, n_knots_y=theta_knots, coeffs=bispl.get_coeffs()
        )
        return rd
