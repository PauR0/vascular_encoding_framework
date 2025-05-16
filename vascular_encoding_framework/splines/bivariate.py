import numpy as np
from scipy.interpolate import BivariateSpline

from .._base import Spline, attribute_checker
from .splines import get_uniform_knot_vector


class BiSpline(Spline):
    """Bivariate Spline Class."""

    def __init__(self) -> None:
        super().__init__()

        # Extrema of the first parameter domain.
        self.x0: np.ndarray = None
        self.x1: np.ndarray = None

        # Extrema of the second parameter domain.
        self.y0: np.ndarray = None
        self.y1: np.ndarray = None

        # First parameter spline params
        self.kx: int = 3  # Defaulting to cubic splines.
        self.knots_x: np.ndarray = None
        self.n_knots_x: int = None
        self.extra_x: str = "constant"  # {'constant', 'periodic'}

        # Second parameter spline params
        self.ky: int = 3  # Defaulting to cubic splines.
        self.knots_y: np.ndarray = None
        self.n_knots_y: int = None
        self.extra_y: str = "constant"  # {'constant', 'periodic'}

        # Coefficient Matrix
        self.coeffs: np.ndarray = None  # Shape (3, n_knots+k+1)

        self._bspl: BivariateSpline = None

    def __call__(self, x, y, grid=False):
        """Evaluate the spline. Equivalent to evaluate method."""
        return self.evaluate(x=x, y=y, grid=grid)

    def build(self):
        """Build the spline object internal attributes."""
        attribute_checker(self, ["kx", "ky", "coeffs"], info="cant build splines.")

        if self.knots_x is None and self.n_knots_x is None:
            raise AttributeError(
                "Cant build bivariate splines. The knots and amount of knots for the first (x)"
                + " parameter is None"
            )
        elif self.knots_x is None and self.n_knots_x is not None:
            mode = "complete"
            if self.extra_x == "periodic":
                mode = "periodic"
            self.knots_x = get_uniform_knot_vector(self.x0, self.x1, self.n_knots_x, mode=mode)

        if self.knots_y is None and self.n_knots_y is None:
            raise AttributeError(
                "Cant build bivariate splines. The knots and amount of knots for the second "
                + " parameter (y) is None"
            )
        elif self.knots_y is None and self.n_knots_y is not None:
            mode = "complete"
            if self.extra_y == "periodic":
                mode = "periodic"
            self.knots_y = get_uniform_knot_vector(self.y0, self.y1, self.n_knots_y, mode=mode)

        self._bispl = BivariateSpline()
        self._bispl.tck = self.knots_x, self.knots_y, self.coeffs.ravel()
        self._bispl.degrees = self.kx, self.ky

    def evaluate(self, x, y, grid=False):
        """
        Evaluate the Bivariate splines at x and y.

        Parameters
        ----------
        x : float or np.ndarray
            The first parameter values
        y : float or np.ndarray
            The second parameter values
        grid : bool, opt
            Default False. Whether to evaluate the spline at the
            grid built by the Cartesian product of x and y.

        Returns
        -------
        z : float or np.ndarray
            The values of spl(x, y)

        """

        def clip_periodic(a, T=2 * np.pi):
            p = a // T
            if isinstance(a, (int, float)):
                a -= T * p
            else:
                a = a.copy()
                ids = (p < 0) | (1 < p)
                a[ids] -= T * p[ids]

            return a

        if self.extra_x == "constant":
            x = np.clip(x, self.x0, self.x1)
        elif self.extra_x == "periodic":
            T = self.x1 - self.x0
            x = clip_periodic(x, T)

        if self.extra_y == "constant":
            y = np.clip(y, self.y0, self.y1)
        elif self.extra_y == "periodic":
            T = self.y1 - self.y0
            y = clip_periodic(y, T)

        return self._bispl(x, y, grid=grid)
