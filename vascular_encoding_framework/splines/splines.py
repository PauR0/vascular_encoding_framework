import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BivariateSpline, BSpline, splev, splrep
from scipy.optimize import minimize

from .psplines import (
    bivariate_optimization_loss,
    get_bivariate_semiperiodic_constraint,
    get_unispline_constraint,
    univariate_optimization_loss,
)


def get_uniform_knot_vector(xb, xe, n, mode="complete", k=3, ext=None):
    """
    Generate a B-Spline uniform knot vector.

    Given the interval [xb, xe], this function returns the even partition in n internal k-nots.
    The mode argument allows the knot vector to account for the different boundary conditions.
    In 'internal' mode only internal knots are returned, leaving the boundaries undefined.
    In 'complete', the extreme of the interval are repeated k+1 times to make the spline interpolate
    the last control point/coefficient. In 'periodic', the extrema of the interval is extended k+1
    times, preserving the spacing between knots. Additionally, an extra 'extended' method allows to
    perform a similar extension, but the amount extensions is controlled by the ext argument, that
    is ignored in any other mode.


    Parameters
    ----------
    xb, xe : float
        The begin and end of the definition interval.
    n : int
        Number of internal knots.
    k : int, optional
        Default is 3. The degree of the spline.
    mode : {'internal', 'complete', 'extended', 'periodic'} , optional
        Default is 'internal'.

        If mode == 'internal' then t is the even spaced partition of [xb, xe]
        without the extrema of the interval.

        If mode == 'complete' t contains [xb]*(k+1) at the beginning and
        [xe]*(k+1) at the end.

        If mode = 'extended' (ext must be passed), it extends ext times the
        knot vector from both ends preserving the spacing.

        mode 'periodic', is the equivalent to setting mode='extended' and ext=k.
        It is useful when combined with scipy B-Splines functions.

    ext : int
        Default is k. Ignored if mode != 'extended'. The times to extend the knot vector from
        both ends preserving the separation between nodes.


    Returns
    -------
    t : np.ndarray
        The knot vector.
    """

    t = np.linspace(xb, xe, n + 2)
    d = (xe - xb) / (n + 1)

    if mode == "periodic":
        mode = "extended"
        ext = k

    if mode == "internal":
        t = t[1:-1]

    elif mode == "complete":
        t = np.concatenate([[t[0]] * k, t, [t[-1]] * k])

    elif mode == "extended":
        if ext is None:
            ext = k

        t = np.concatenate([t[0] + np.arange(-ext, 0) * d, t, t[-1] + np.arange(ext + 1)[1:] * d])

    else:
        raise ValueError(
            f"Wrong value ({mode}) for mode argument. "
            + "The options are {'internal', 'complete', 'extended', 'periodic'}. "
        )

    return t


def get_coefficients_lenght(n_internal_knots, k):
    """
    Get the number of coefficients required to build a spline.

    Parameters
    ----------
    n_internal_knots : int or list[int]
        The amount of internal knots.
    k : int or list[int], 1<=k<=5
        The polynomial degree

    Returns
    -------
    nc : int
        The amount of coefficients required.
    """

    if isinstance(n_internal_knots, int):
        n_internal_knots = [n_internal_knots]
    if isinstance(k, int):
        k = [k]

    nc = np.array(n_internal_knots) + np.array(k) + 1
    nc = np.prod(nc)
    return nc


def compute_normalized_params(points):
    """
    Compute the parametrization parameter as a normalized cumulative distance.

    Parameters
    ----------
    points : np.ndarray (N, d)
        The point array.

    Returns
    -------
    param_values : np.ndarray (N,)
        The computed parameters array.
    """

    param_values = [0.0]
    for i in range(1, points.shape[0]):
        dist = np.linalg.norm(points[i] - points[i - 1])
        param_values.append(param_values[-1] + dist)
    param_values = np.array(param_values)
    param_values = (param_values - param_values[0]) / (param_values[-1] - param_values[0])

    return param_values


def uniform_penalized_spline(
    points, n_knots, k=3, param_values=None, force_ini=False, force_end=False, curvature_penalty=1.0
):
    """
    Compute the curvature-penalized approximation spline curve of a list of d-dimensional points.

    Points must be a numpy array of dimension Nxd for a list of N d-dimensional points.
    The parametrization of the curve can be controlled by the param_values argument, if is None,
    The parameter is computed as the distance traveled from the first point in a poly-line way, and
    then normalized from 0 to L.

    The argument curvature_penalty is the penalization factor for the curvature. If set to 0, a regular LSQ
    approximation is performed.

    Additionally, the argument force_ini and force_end allow to force the optimization to
    force a specific behavior at curve extremes. These arguments force the interpolation of
    the first and last point provided and its tangents. The tangents are approximated by finite
    differences and added as optimization constraints as well.

    Parameters
    ----------
    points : np.ndarray (N, d)
        The array of points.
    n_knots : int
        The amount of internal knots.
    k : int, opt
        Default is 3. The spline polynomial degree.
    param_values : array-like (N,), opt
        The parameter values for each point. Must be a increasing sequence. If not passed it is
        computed as the normalized distance traveled.
    force_ini : bool, optional
        Default False. Whether to impose interpolation and tangent at the beginning of the
        curve.
    force_end : bool, optional
        Default False. Whether to impose interpolation and tangent at the end of the curve.
    curvature_penalty : float, optional
        Default 1.0. The penalization factor for the curvature.

    Returns
    -------
    spl : BSpline
        The approximating spline object of scipy.interpolate.
    """

    d = points.shape[1]

    if param_values is None:
        param_values = compute_normalized_params(points)

    t = get_uniform_knot_vector(param_values[0], param_values[-1], n_knots, mode="complete")

    cons = []
    if force_ini:
        tg = (points[1] - points[0]) / (param_values[1] - param_values[0])
        cons.append(get_unispline_constraint(t, k, param_values[0], points[0]))
        cons.append(get_unispline_constraint(t, k, param_values[0], tg, nu=1))

    if force_end:
        tg = (points[-1] - points[-2]) / (param_values[-1] - param_values[-2])
        cons.append(get_unispline_constraint(t, k, param_values[-1], points[-1], nu=0))
        cons.append(get_unispline_constraint(t, k, param_values[-1], tg, nu=1))
    cons = cons if cons else None

    x0 = np.array(
        [points.mean(axis=0)] * get_coefficients_lenght(n_internal_knots=n_knots, k=k)
    ).ravel()
    res = minimize(
        fun=univariate_optimization_loss,
        x0=x0,
        args=(param_values, points, t, k, curvature_penalty),
        method="SLSQP",
        constraints=cons,
    )

    spl = BSpline(t=t, c=res.x.reshape(-1, d), k=k)

    return spl


def fix_discontinuity(polar_points, n_first=10, n_last=10, degree=3, logger=None):
    """
    Smooth points in the angular discontinuity.

    This function expects a 2D point cloud expressed in polar coordinates
    contained in an array of shape (2,N). This point cloud have to be sorted
    in theta wise order from 0 to 2pi. If these conditions are fulfilled this method
    returns a list of points, where the points close to 0 or 2pi have been smoothed by means
    of a bspline.

    Parameters
    ----------
    polar_points : numpy.array
        array of two rows containing the theta and rho coordinates of the point
        cloud of the form [[theta1, theta2,..., thetaN], [rho1,rho2, ..., rhoN]]
    n_first : int
        number of points after the beginning to be used
    n_last : int
        number of points before the end to be used
    degree : int
        degree of the polynomial to use
    logger: logging.Logger
        output logger

    Returns
    -------
    cont_polar_points : numpy.array
        A copy of the array with the discontinuity reduced

    Notes
    -----
    The last n_last points are placed before the first n_first points. Then,
    a polynomial of degree n is used to approximate a point at theta = 0.
    This value is added at the beginning and at the end of the vector.

    """

    if polar_points.shape[1] < max(n_first, n_last):
        n_first = min(n_first, polar_points.shape[1])
        n_last = min(n_last, polar_points.shape[1])
        if logger is not None:
            logger.debug(f"Not enough points. Reducing to  n_first = {n_first};  n_last = {n_last}")

    # Values of theta before and after the cut
    th_last = polar_points[0, -n_last:] - 2 * np.pi
    th_first = polar_points[0, :n_first]

    # Values of r before and after the cut
    r_last = polar_points[1, -n_last:]
    r_first = polar_points[1, :n_first]

    # Vectors with th and r around the cut, in order
    th = np.concatenate((th_last, th_first))
    r = np.concatenate((r_last, r_first))

    tck = splrep(th, r, k=degree, s=0.1)

    cut = splev(0, tck)

    cont_points = np.concatenate(([[0], [cut]], polar_points, [[2 * np.pi], [cut]]), axis=1)

    return cont_points


def compute_rho_spline(polar_points, n_knots, k=3, logger=None):
    """Compute the coefficients of the approximating spline."""

    # Adding a value at 0 and 2pi
    cont_polar_points = fix_discontinuity(polar_points)

    knots = get_uniform_knot_vector(0, 2 * np.pi, n_knots, mode="internal")

    coeff_r, rmse = None, None
    if polar_points.shape[1] < n_knots + k + 1:
        err_msg = f"compute_slice_coeffs: amount of points ({polar_points.shape[0]}) is less than n_knots_slice + 1 ({n_knots}+1)"
        if logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    else:
        (_, coeff_r, _), _, ier, msg = splrep(
            x=cont_polar_points[0], y=cont_polar_points[1], k=k, t=knots, per=True, full_output=True
        )
        if ier > 0:
            err_msg = f"splrep failed to fit the slice, saying: ier: {ier}, msg: '{msg}'"
            if logger is not None:
                logger.warning(err_msg)
            else:
                print(err_msg)

            coeff_r = None
        else:
            raise NotImplementedError("Root mean squared error not implemented yet!")
            # rmse = compute_slice_rmse(
            #    polar_points=polar_points, n_knots_slice=n_knots, coeff=coeff_r
            # )

    return coeff_r, rmse


def uniform_penalized_bivariate_spline(
    x, y, z, nx, ny, laplacian_penalty=1.0, y_periodic=False, kx=3, ky=3, bounds=None, debug=False
):
    """
    Curvature-penalized LSQ approximation of a bivariate function f(x,y).

    Parameters
    ----------
    x,y,z : np.ndarray
        The samples f(x_i,y_i) = z_i. The three arrays must have the same length.
    nx,ny : int
        The number of subdivisions for each dimension.
    laplacian_penalty : float, optional
        Default 1. The penalization factor applied the laplacian of the coefficients.
    y_periodic : bool, optional
        Default True. Whether to impose periodicity on y coefficients.
    kx,ky : int, optional
        Default 3. The degree of the spline for each dimension.
    bounds : tuple(float)
        The interval extrema of the parameters domain in the form
        (xmin, xmax, ymin, ymax).
    debug : bool, opt
        Display a plot with the extension of the data and the fitting result.

    Returns
    -------
    bispl : BivariateSpline
        The bivariate spline object.
    """

    if bounds is None:
        xb, xe = x.min(), x.max()
        yb, ye = y.min(), y.max()
    else:
        xb, xe, yb, ye = bounds

    tx = get_uniform_knot_vector(xb=xb, xe=xe, n=nx, mode="complete", k=kx, ext=None)
    ty = get_uniform_knot_vector(
        xb=yb, xe=ye, n=ny, mode="periodic" if y_periodic else "extended", k=ky, ext=None
    )

    cons = None
    if y_periodic:
        cons = get_bivariate_semiperiodic_constraint(nx=nx, ny=ny, kx=kx, ky=ky)

    x0 = np.array([z.mean()] * get_coefficients_lenght(n_internal_knots=[nx, ny], k=[kx, ky]))
    res = minimize(
        fun=bivariate_optimization_loss,
        x0=x0,
        args=(x, y, z, tx, ty, kx, ky, laplacian_penalty),
        method="SLSQP",
        constraints=cons,
        options={"disp": debug},
    )  # "maxiter":25, can be set as an option

    bispl = BivariateSpline()
    bispl.tck = tx, ty, res.x
    bispl.degrees = kx, ky

    if debug:
        xx = np.linspace(xb, xe, 50)
        yy = np.linspace(yb, ye, 50)
        X, Y = np.meshgrid(xx, yy)

        fg = plt.figure()
        ax1 = fg.add_subplot(121, projection="3d")
        ax1.scatter(x, y, z, c=z, label="Data set")

        ax2 = fg.add_subplot(122, projection="3d")
        Z = bispl(X, Y, grid=False)
        ax2.plot_surface(X, Y, Z, linewidth=0, color="r", label="bispl", alpha=0.9)
        ax2.scatter(x, y, z, c=z, label="Data set")
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0)
        plt.show()

    return bispl
