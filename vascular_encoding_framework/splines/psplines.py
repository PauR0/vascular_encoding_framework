

import numpy as np

from scipy.ndimage import laplace, convolve1d
from scipy.interpolate import BSpline, BivariateSpline


def univariate_optimization_loss(c, x, y, t, k, l):
    """
    Function to compute the expression to be optimized for univariate spline approximation.

    Parameter l allows to penalize the curvature of the approximation by adding the 2 order finite diference of the coefficients.

    Arguments
    ---------

        c : np.ndarray (m,)
            The coefficients of the splines to use.

        x: np.ndarray (N,)
            The parameter values of the function in the domain.

        y : np.ndarray (N, d)
            The observations of the function at the parameter samples defined in x.

        t : np.ndarray (n, )
            The knot vector.

        k : int, 1<=k<=5
            The polynomials degree.

        l : float
            The penalization factor on curvature.

    Returns
    -------
        err : float
            The evaluation of the error expression to be minimized.
    """

    if len(y.shape) > 1:
        c = c.reshape(-1, y.shape[1])

    spl = BSpline.construct_fast(t=t, c=c, k=k)
    err = (np.linalg.norm(y - spl(x), axis=1)**2).sum()

    if l:
        err += l*(convolve1d(c, [1, -2, 1], axis=0, mode='mirror')**2).sum()
    return err
#

def bivariate_optimization_loss(c, x, y, z, tx, ty, kx, ky, l):
    """
    Function to compute the expression to be optimized for bivariate spline approximation.

    Parameter l allows to penalize the curvature of the approximation by adding the 2 order finite diference of the coefficients.

    Arguments
    ---------

        c : np.ndarray (m,)
            The coefficients of the splines to use.

        Bx : np.ndarray (N, m)
            The matrix with the evaluation of the domain points by the basis splines.

        y : np.ndarray (N,)
            The observations of the function at the domain points.

    Returns
    -------
        err : float
            The evaluation of the error expression to be minimized.
    """

    bspl = BivariateSpline()
    bspl.tck     = tx, ty, c
    bspl.degrees = kx, ky

    err = ((z - bspl(x, y, grid=False))**2).sum()
    if l:
        err += l*(laplace(c)**2).sum()

    return err
#

def get_unispline_constraint(t, k, a, v, nu=0):
    """
    Function to get a constrain for univariate spline approximation. The constrained is expressed
    by means of a dictionary as SLSQP optimization algorithm expects.

    The function can be used to fastly build equation-like constrains to impose the spline fulfill
    equations such as: spl^{nu}(a) = v.
    Where a is a value in the definition domain, v a certain real value and nu the derivative order.
    Recall that nu is at most k (the degree of the polynomial).

    Arguments
    ---------

        t : np.ndarray
            The knot vector that will be used for the optimization.

        k : int, 1<=k<=5
            The polynomials degree.

        a : float
            A parameter value.

        v : float - array-like
            The value to force at the provided parameter value, i.e. f(a) = v.

        nu : int, nu<k
            The derivative order to evaluate. If 0, the function itself is forced to reach
            provided value, if 1, then the first derivative is (f'(a)=v), if 2 then the second
            derivative (f''(a)=v) and so on.

    Returns
    -------
        : dict
            The constraint dictionary as described in scipy.optimize docs.

    """

    d = len(v)
    return {'type':'eq', 'fun': lambda c: np.linalg.norm(BSpline.construct_fast(t, c.reshape(-1, d), k)(a, nu=nu) - v)}
#

def get_bivariate_semiperiodic_constraint(c, nx, ny, k):
    """
    Function to get a constrain dictionary as SLSQP optimization algorithm expects.

    The function can be used to fastly build equation-like constrains to impose the spline fulfill
    equations such as: spl^{nu}(a) = v.
    Where a is a value in the definition domain, v a certain real value and nu the derivative order.
    Recall that nu is at most k (the degree of the polynomial).
    """

    column = lambda coeff, col: coeff.reshape(nx+(kx+1), ny+(ky+1))[:,col]
    cons = [
        {'type':'eq', 'fun': lambda c: column(c, 0) - column(c, -3)},
        {'type':'eq', 'fun': lambda c: column(c, 1) - column(c, -2)},
        {'type':'eq', 'fun': lambda c: column(c, 2) - column(c, -1)},
    ]

    return cons