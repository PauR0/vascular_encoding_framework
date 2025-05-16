from typing import Literal

import numpy as np
from scipy.interpolate import BSpline

from .._base import Spline, attribute_checker
from .splines import get_uniform_knot_vector


class UniSpline(Spline):
    """Univariate Spline Class."""

    def __init__(self) -> None:
        super().__init__()

        # Extrema of the parameter domain.
        self.t0: float = 0
        self.t1: float = 1

        # Spline params
        self.k: int = 3  # Defaulting to cubic splines.
        self.knots: np.ndarray = None
        self.coeffs: np.ndarray = None  # Shape (3, n_knots+k+1)
        self.n_knots: int = None
        self.extrapolation: Literal["linear", "constant"] = "linear"

        self._spl: BSpline = None

    def __call__(self, t):
        """
        Evaluate the spline at given parameter values. Values are clipped
        to parameter domain, as in constant extrapolation.

        Parameters
        ----------
        t : float or array-like
        """

        return self.evaluate(t)

    def evaluate(self, t, extra=None):
        """
        Evaluate the spline at values provided in t. Values are clipped to
        parameter domain, as in constant extrapolation.

        Parameters
        ----------
        t : float, array-like
            The parameter values to be evaluated.

        Returns
        -------
        p : float or np.ndarray
            The evaluation of t. If coeffs are N-dimensional, the output so will.
        """

        if self._spl is None:
            raise AttributeError("Can't evaluate spline object. It has not been built...")

        if extra is None:
            extra = self.extrapolation

        if extra == "constant":
            tt = np.clip(t, a_min=self.t0, a_max=self.t1)
            p = np.array(self._spl(tt))

        elif extra == "linear":
            # Sorry for the lambda mess...
            def lower_extr(x):
                return self._spl(self.t0) - self._spl.derivative(self.t0) * x

            def upper_extr(x):
                return self._spl(self.t1) + self._spl.derivative(self.t1) * (x - self.t1)

            def middl_intr(x):
                return self._spl(x)

            if self.coeffs.ndim > 1:

                def lower_extr(x):
                    return (
                        self._spl(self.t0).reshape(3, 1)
                        - self._spl.derivative(self.t0).reshape(3, 1) * x
                    ).T

                def upper_extr(x):
                    return (
                        self._spl(self.t1).reshape(3, 1)
                        + self._spl.derivative(self.t1).reshape(3, 1) * (x - 1)
                    ).T

                def middl_intr(x):
                    return self._spl(x).reshape(-1, 3)

            if isinstance(t, (float, int)):
                if t < self.t0:
                    p = lower_extr(t)
                elif t > self.t1:
                    p = upper_extr(t)
                else:
                    p = middl_intr(t)
                p.reshape(
                    3,
                )

            elif isinstance(t, (np.ndarray, list)):
                p = np.empty((len(t), 3))

                low_ids = t < self.t0
                upp_ids = t > self.t1
                mid_ids = np.logical_not(low_ids | upp_ids)

                if low_ids.any():
                    p[low_ids] = lower_extr(t[low_ids])

                if mid_ids.any():
                    p[mid_ids] = middl_intr(t[mid_ids])

                if upp_ids.any():
                    p[upp_ids] = upper_extr(t[upp_ids])

        else:
            raise ValueError(f"Wrong value for extra argument ({extra}).")

        if p.shape[0] == 1:
            return p.ravel()

        return p

    def build(self):
        """Build the spline object internal attributes."""
        attribute_checker(self, ["k", "n_knots", "coeffs"], info="cant build splines.")

        if self.knots is None:
            self.knots = get_uniform_knot_vector(self.t0, self.t1, self.n_knots, mode="complete")

        self._spl = BSpline(t=self.knots, c=self.coeffs, k=self.k)

    def get_knot_segments(self, a, b):
        """
        Given the interval [a, b], this function returns a partition
        P = {p_i}_i=0^N where p_0 = a, p_N = b and p_i = t_i for 0<i<N,
        where t_i are knots of the centerline splines.

        Parameters
        ----------
        a : float
            inferior limit

        b : float
            superior limit

        Returns
        -------
        segments : np.ndarray
            The partition of the interval with a and b as inferior and superior limits.
        """

        # Compute the polynomial segments
        min_id = np.argmax(self._spl.t > a)
        max_id = np.argmax(self._spl.t > b)
        if max_id == 0:
            max_id = -1

        segments = np.concatenate(([a], self._spl.t[min_id:max_id], [b]))

        return segments
