

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import (splrep, splev, make_lsq_spline,
                               BSpline, BivariateSpline, LSQBivariateSpline,
                               RBFInterpolator)

from scipy.optimize import minimize

from skimage.morphology import dilation
from skimage.measure import label, regionprops

from ..utils._code import attribute_checker, attribute_setter
from ..messages import *

from .psplines import get_unispline_constraint, univariate_optimization_loss

class Spline(ABC):

    def __init__(self):
        ...
    #

    @abstractmethod
    def evaluate(self, **kwargs):
        ...
    #

    @abstractmethod
    def __call__(self, **kwargs):
        """
        Calling a spline object is expected to return the same as the evaluate
        method does.
        """
        ...
    #

    def set_parameters(self, build=False, **kwargs):
        """
        Set parameters and attributes by kwargs.

        Arguments:
        -------------

            build : bool, opt
                Default False. If run build setting the params.
        """

        attribute_setter(self, **kwargs)

        if build:
            self.build()
    #

    @abstractmethod
    def build(self):
        ...
    #
#

class UniSpline(Spline):

    def __init__(self) -> None:

        super().__init__()

        #Extrema of the parameter domain.
        self.t0 : float = 0
        self.t1 : float = 1

        #Spline params
        self.k       : int        = 3    #Defaulting to cubic splines.
        self.knots   : np.ndarray = None
        self.coeffs  : np.ndarray = None #Shape (3, n_knots+k+1)
        self.n_knots : int        = None
        self.extra   : Literal['linear', 'constant'] = 'linear'

        self._spl : BSpline
    #

    def __call__(self, t):
        """
        Evaluate the spline at given parameter values. Values are clipped
        to parameter domain, as in constant extrapolation.
        Arguments:
        -----------

            t : float or array-like
        """

        return self.evaluate(t)
    #

    def evaluate(self, t, extra=None):
        """
        Evaluate the spline at values provided in t. Values are clipped to
        parameter domain, as in constant extrapolation.

        Arguments:
        --------------

            t : float, array-like
                The parameter values to be evaluated.

        Returns:
        ---------
            p : float or np.ndarray
                The evaluation of t. If coeffs are N-dimensional, the output so will.
        """

        if not attribute_checker(self, ['_spl'], info="can't evaluate spline, it has not been built..."):
            return False

        if extra is None:
            extra = self.extra

        if extra == 'constant':
            tt = np.clip(t, a_min=self.t0, a_max=self.t1)
            p  = np.array(self._spl(tt))

        elif extra == 'linear':
            #Sorry for the lambda mess...
            lower_extr = lambda x: self._spl(self.t0) - self._spl.derivative(self.t0) * x
            upper_extr = lambda x: self._spl(self.t1) + self._spl.derivative(self.t1) * (x-self.t1)
            middl_intr = lambda x: self._spl(x)
            if self.coeffs.ndim > 1:
                lower_extr = lambda x: (self._spl(self.t0).reshape(3,1) - self._spl.derivative(self.t0).reshape(3,1) * x).T
                upper_extr = lambda x: (self._spl(self.t1).reshape(3,1) + self._spl.derivative(self.t1).reshape(3,1) * (x-1)).T
                middl_intr = lambda x: self._spl(x).reshape(-1, 3)


            if isinstance(t, (float, int)):
                if t < self.t0:
                    p = lower_extr(t)
                elif t > self.t1:
                    p = upper_extr(t)
                else:
                    p = middl_intr(t)
                p.reshape(3,)

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

        if p.shape[0] == 1:
            return p.ravel()

        return p
    #

    def build(self):

        if not attribute_checker(self, ['k', 'n_knots', 'coeffs'], info="cant build splines."):
            return False

        if self.knots is None:
            self.knots = get_uniform_knot_vector(self.t0, self.t1, self.n_knots, mode='complete')

        self._spl = BSpline(t=self.knots,
                               c=self.coeffs,
                               k=self.k)
    #

    def get_knot_segments(self, a, b):
        """
        Given the interval [a, b], this function returns a partition
        P = {p_i}_i=0^N where p_0 = a, p_N = b and p_i = t_i for 0<i<N,
        where t_i are knots of the centerline splines.

        Parameters:
        -----------
            a : float
                inferior limit

            b : float
                superior limit

        Returns:
        ---------
            segments : np.ndarray
                The partition of the interval with a and b as inferior and superior limits.
        """

        #Compute the polynomial segments
        min_id = np.argmax(self._spl.t > a)
        max_id = np.argmax(self._spl.t > b)
        if max_id == 0:
            max_id = -1

        segments = np.concatenate(([a], self._spl.t[min_id:max_id], [b]))

        return segments
    #
#

class BiSpline(Spline):

    def __init__(self) -> None:

        super().__init__()

        #Extrema of the first parameter domain.
        self.x0 : np.ndarray = None
        self.x1 : np.ndarray = None

        #Extrema of the second parameter domain.
        self.y0 : np.ndarray = None
        self.y1 : np.ndarray = None

        #First parameter spline params
        self.kx       : int        = 3 #Defaulting to cubic splines.
        self.knots_x   : np.ndarray = None
        self.n_knots_x : int        = None
        self.extra_x   : str        = 'constant' #{'constant', 'periodic'}

        #Second parameter spline params
        self.ky       : int        = 3 #Defaulting to cubic splines.
        self.knots_y   : np.ndarray = None
        self.n_knots_y : int        = None
        self.extra_y   : str        = 'constant' #{'constant', 'periodic'}

        #Coefficient Matrix
        self.coeffs  : np.ndarray = None #Shape (3, n_knots+k+1)

        self._bspl : BivariateSpline = None
    #

    def __call__(self, x, y, grid=False):
        """
        Evaluate the spline. Equivalent to evaluate method.
        """
        return self.evaluate(x=x, y=y, grid=grid)
    #

    def build(self):

        if not attribute_checker(self, ['kx', 'ky', 'coeffs'], info="cant build splines."):
            return False

        if self.knots_x is None and self.n_knots_x is None:
            error_message("cant build bivariate splines. The knots and amount of knots for the first (x) parameter is None")
        elif self.knots_x is None and self.n_knots_x is not None:
            mode='complete'
            if self.extra_x == 'periodic':
                mode='periodic'
            self.knots_x = get_uniform_knot_vector(self.x0, self.x1, self.n_knots_x, mode=mode)

        if self.knots_y is None and self.n_knots_y is None:
            error_message("cant build bivariate splines. The knots and amount of knots for the second parameter (y) is None")
        elif self.knots_y is None and self.n_knots_y is not None:
            mode='complete'
            if self.extra_y == 'periodic':
                mode='periodic'
            self.knots_y = get_uniform_knot_vector(self.y0, self.y1, self.n_knots_y, mode=mode)

        self._bispl         = BivariateSpline()
        self._bispl.tck     = self.knots_x, self.knots_y, self.coeffs.ravel()
        self._bispl.degrees = self.kx, self.ky
    #

    def evaluate(self, x, y, grid=False):
        """
        Evaluate the Bivariate splines at x and y.

        Arguments
        ---------

            x : float or np.ndarray
                The first parameter values

            y : float or np.ndarray
                The second parameter values

            grid : bool, opt
                Default False. Whether to evaluate the spline at the
                grid built by the carthesian product of x and y.

        Returns
        -------

            z : float or np.ndarray
                The values of spl(x, y)

        """

        def clip_periodic(a, T=2*np.pi):
            a = a.copy()
            p = a//T
            if isinstance(a, (int, float)):
                a -= T*p
            else:
                ids = (p < 0) | (1 < p)
                a[ids] -= T*p[ids]
            return a

        if self.extra_x == 'constant':
            x = np.clip(x, self.x0, self.x1)
        elif self.extra_x == 'periodic':
            T = self.x1-self.x0
            x = clip_periodic(x, T)

        if self.extra_y == 'constant':
            y = np.clip(y, self.y0, self.y1)
        elif self.extra_y == 'periodic':
            T = self.y1-self.y0
            y = clip_periodic(y, T)

        return self._bispl(x, y, grid=grid)
    #
#

def get_uniform_knot_vector(xb, xe, n, mode='complete', k=3, ext=None):
    """
    Generates a B-Spline uniform knot vector.

    Given the interval [xb, xe], this function returns the even partition in n internal k-nots.
    The mode argument allows the knot vector to account for the different boundary conditions.
    In 'internal' mode only internal knots are returned, leaving the boundarys undefined.
    In 'complete', the extreme of the interval are repeated k+1 times to make the spline interpolate
    the last control point/coefficient. In 'periodic', the extrema of the interval is extended k+1
    times, preserving the spacing between knots. Additionally, an extra 'extended' metod allows to
    perform a similar extension, but the amount extensions is controlled by the ext argument, that
    is ignored in any other mode.


    Arguments
    ---------

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
        Default is None. Ignored if mode != 'extended'. The times to extend the knot vector from
        both ends preserving the separation between nodes.


    Returns
    -------
        t : np.ndarray
            The knot vector.

    """

    t = np.linspace(xb, xe, n+2)
    d = (xe-xb)/(n+1)

    if mode == 'periodic':
        mode = 'extended'
        ext = k

    if mode == 'internal':
        t = t[1:-1]

    elif mode == 'complete':
        t = np.concatenate([[t[0]]*k, t, [t[-1]]*k])

    elif mode == 'extended':
        if ext is None:
            raise ValueError(f"Wrong value ({ext}) for ext argument using extended mode.")

        t = np.concatenate([t[0]+np.arange(-ext, 0)*d, t, t[-1]+np.arange(ext+1)[1:]*d])

    else:
        raise ValueError(f"Wrong value ({mode}) for mode argument. The options are {{'internal', 'complete', 'extended', 'periodic'}}. ")

    return t
#

def get_coefficients_lenght(n_internal_knots, k):
    """
    Get the number of coefficients required to build a spline.

    Arguments
    ---------

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

    nc = np.array(n_internal_knots)+np.array(k)+1
    nc = np.prod(nc)
    return nc
#

def compute_normalized_params(points):
    """
    Compute the parametrization parameter as a normalized cummulative distance.

    Arguments
    ---------

        points : np.ndarray (N, d)
            The point array.

    Returns
    -------
        param_values : np.ndarray (N,)
            The computed parameters array.
    """

    param_values = [0.0]
    for i in range(1, points.shape[0]):
        dist = np.linalg.norm(points[i] - points[i-1])
        param_values.append(param_values[-1]+dist)
    param_values = np.array(param_values)
    param_values = (param_values-param_values[0]) / (param_values[-1] - param_values[0])

    return param_values
#

def uniform_penalized_spline(points,
                             n_knots,
                             k=3,
                             param_values=None,
                             force_ini=False,
                             force_end=False,
                             curvature_penalty=1.0):
    """
    Compute the curvature-penalized approximation spline curve of a list of d-dimensional points.

    Points must be a numpy array of dimension Nxd for a list of N d-dimensional points.
    The parametrization of the curve can be controlled by the param_values argument, if is None,
    The parameter is computed as the distance traveled from the first point in a poly-line way, and
    then normalized from 0 to L.

    The argument curvature_penalty is the penalization factor for the curvature. If set to 0, a regular LSQ
    approximation is performed.

    Additionally, the argument force_ini and force_end allow to force the optimization to
    force a specific behaviour at curve extremes. These arguments force the interpolation of
    the first and last point provided and its tangents. The tangents are approximated by finite
    differences and added as optimization constraints as well.

    Arguments
    ---------

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
            Default False. Whether to impose interpolation and tangent at the begginnig of the
            curve.

        force_end : bool, optional
            Default False. Whether to impose interpolation and tangent at the end of the curve.

        curvature_penalty : float, optional
            Default 1.0. The penalization factor for the curvature.

    Returns:
    ----------
        spl : BSpline
            The approximating spline object of scipy.interpolate.
    """

    d = points.shape[1]

    if param_values is None:
        param_values = compute_normalized_params(points)

    t = get_uniform_knot_vector(param_values[0], param_values[-1], n_knots, mode='complete')

    cons = []
    if force_ini:
        tg = (points[1]-points[0])/(param_values[1] - param_values[0])
        cons.append(get_unispline_constraint(t, k, param_values[0], points[0]))
        cons.append(get_unispline_constraint(t, k , param_values[0], tg, nu=1))

    if force_end:
        tg = (points[-1]-points[-2])/(param_values[-1] - param_values[-2])
        cons.append(get_unispline_constraint(t, k , param_values[-1], points[-1], nu=0))
        cons.append(get_unispline_constraint(t, k , param_values[-1], tg, nu=1))
    cons = cons if cons else None

    x0 = np.array([points.mean(axis=0)] * get_coefficients_lenght(n_internal_knots=n_knots, k=k)).ravel()
    res = minimize(fun=univariate_optimization_loss, x0=x0,
                   args=(param_values, points, t, k, curvature_penalty),
                   method='SLSQP', constraints=cons)

    spl = BSpline(t=t, c=res.x.reshape(-1, d), k=k)

    return spl
#

def fix_discontinuity(polar_points, n_first = 10, n_last  = 10, degree = 3, logger=None):
    """
    This function expects a 2D point cloud expressed in polar coortinates
    contained in an array of shape (2,N). This point cloud have to be sorted
    in theta wise order from 0 to 2pi. If these conditions are fulfilled this method
    returns a list of points, where the points close to 0 or 2pi have been smoothed by means
    of a bspline.

    Arguments
    ----
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

    Explanation
    -----------
    The last n_last points are placed before the firsr n_first points. Then,
    a polynomial of degree n is used to approximate a point at theta = 0.
    This value is added at the beginning and at the end of the vector.

    """


    if polar_points.shape[1] < max(n_first,n_last):
        n_first = min(n_first,polar_points.shape[1])
        n_last = min(n_last,polar_points.shape[1])
        if logger is not None:
            logger.debug(f"Not enough points. Reducing to  n_first = {n_first};  n_last = {n_last}")
    #

    # Values of theta before and after the cut
    th_last = polar_points[0,-n_last:] - 2*np.pi
    th_first = polar_points[0,:n_first]

    # Values of r before and after the cut
    r_last = polar_points[1,-n_last:]
    r_first = polar_points[1,:n_first]

    # Vectors with th and r around the cut, in order
    th = np.concatenate((th_last,th_first))
    r = np.concatenate((r_last,r_first))

    tck = splrep(th, r, k=degree, s = 0.1)

    cut = splev(0,tck)

    cont_points = np.concatenate(([[0],[cut]],
                         polar_points,
                         [[2*np.pi],[cut]]), axis = 1)

    return cont_points
#

def compute_rho_spline(polar_points, n_knots, k=3, logger=None):
    """
    Compute the coefficients of the approximating spline
    """

    # Adding a value at 0 and 2pi
    cont_polar_points = fix_discontinuity(polar_points)

    knots = get_uniform_knot_vector(0,2*np.pi, n_knots, mode='internal')

    coeff_r, rmse = None, None
    if polar_points.shape[1] < n_knots + k +1:
        err_msg = f"compute_slice_coeffs: amount of points ({polar_points.shape[0]}) is less than n_knots_slice + 1 ({n_knots}+1)"
        if logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    else:
        (_, coeff_r, _), _, ier, msg = splrep(x=cont_polar_points[0],
                            y=cont_polar_points[1],
                            k=k,
                            t=knots,
                            per=True,
                            full_output=True)
        if ier > 0:
            err_msg = f"splrep failed to fit the slice, saying: ier: {ier}, msg: '{msg}'"
            if logger is not None:
                logger.warning(err_msg)
            else:
                print(err_msg)

            coeff_r = None
        else:
            rmse = compute_slice_rmse(polar_points=polar_points, n_knots_slice = n_knots, coeff=coeff_r)

    return coeff_r, rmse
#

def compute_point_weights(points, weighting=None, normalize=True):
    """
    Function to compute weights for a list of points. The weighting of the points
    is carried by means of the weighting argument. It should be a list of tuples
    (v, ax, w) specifying the value v, at axis ax, to be weighted by w.
    If normalize is True (which is the default) The weights are normalized to add
    up to 1 for optimization's sake.

    Arguments:
    -----------

        points : np.ndarray, NxD
            The array of D-dimensional points.

        weighting : array-like, Mx3.
            The list of tuples with the value, the axis and weights
            E.g. [(0, 0, 2), (np.pi, 1, 0.5)],
            with this selection, the points with first coordinate
            equal to 0 will weight double the base weight, and those
            whose second coordinate is equal to np.pi will weight half
            the base weight.

        normalize : bool
            Default True. Whether to normalize the weights to add up to 1.

    Returns:
    ---------

        weights : np.ndarray
            The array of weights

    """
    weights = np.ones((points.shape[0],))

    if weighting is not None:
        for v, ax, w in weighting:
            i = np.isclose(points[:,ax], v, rtol=1e-05, atol=1e-08)
            weights[i] = w

    if normalize:
        weights /= weights.sum()

    return weights
#

def extend_periodically_point_cloud(pts, col=1, T=None, d_max=None):
    """

    Args:
    ------
        pts : np.array Nx3
            Array of points

        col : int,
            The periodic variable. By default it is 1, i.e. y axis.

        T : float, optional
            The period of the function. By default it is pts[:,col].max()-pts[:,col].min()

        d_max : float
            The maximum distance to extend the period in both sides. By default it is T.

    """

    m = pts[:,col].min()
    M = pts[:,col].max()

    if T is None:
        T = M-m

    if d_max is None:
        d_max = T

    n_copies = int(1 + (T // d_max))

    pts_out = pts.copy()
    for _ in range(n_copies):
        L = pts.copy()
        L[:,col] -= T
        U = pts.copy()
        U[:,col] += T
        pts_out = np.concatenate( (L, pts_out, U))
    pts_out = pts_out[ (pts_out[:,col] > m - d_max) & (pts_out[:,col] < M + d_max) ]

    return np.unique(pts_out,axis=0)
#

def semiperiodic_LSQ_bivariate_approximation(x, y, z, nx, ny, weighting=None, ext=None, kx=3, ky=3, bounds=None, filling='mean', debug=False):
    """
    A function to perform a LSQ approximation of a bivariate function, f(x,y),
    that is periodic wrt the y axis, by means of uniform bivariate splines. To emulate periodicity
    the provide points are periodically extended in the y-axis, and a bivariate spline with
    its corresponding extended knots is fitted in that space.

    TODO: Allow passing the knot vectors to support non-uniform splines and an int to build the
    uniform knot.

    Arguments:
    ------------

        x,y,z : np.ndarray
            The samples f(x_i,y_i) = z_i. The three arrays must have the same length.

        weighting : list[tuples], [(v, ax, w)]
            The list of values v, at axis ax, to be weighted by w. For example, to
            weight those points with x coordinate equal to 1, with a weight double to
            the rest, weighting=[(1,0,2)] meaning, the points equals to 1 in the ax 0 weight 2.

        nx,ny : int
            The number of subdivisions for each dimension.

        ext : int
            The amount of partitions to add for the periodic extension.

        kx,ky : int
            The degree of the spline for each dimension.

        bounds : tuple(float)
            The interval extrema of the parameters domain in the form
            (xmin, xmax, ymin, ymax).

        filling : {False, 'mean', 'rbf'}, optional
            Default 'mean'. Whether to add interpolated points at detected gaps in the x-y point cloud.
            If False, the hole filling is skipped.

        degbug : bool, opt
            Display a plot with the extension of the data and the fitting result.

    Returns:
    ---------
        Mcoeff : np.ndarray (nx+kx+1, ny+ky+1)
            The coefficients of the bivariate spline.
    """

    if ext is None:
        ext = ky+1

    if bounds is None:
        xb, xe = x.min(), x.max()
        yb, ye = y.min(), y.max()
    else:
        xb, xe, yb, ye = bounds

    tx = get_uniform_knot_vector(xb=xb, xe=xe, n=nx, mode='internal')
    ty = get_uniform_knot_vector(xb=yb, xe=ye, n=ny, mode='extended', ext=ext-1)
    d = ty[1] - ty[0]

    pts = np.concatenate((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)), axis=1)

    if filling:
        #A previous extension is required for if the hole lies in the extrema of the periodic interval
        pts_ext = extend_periodically_point_cloud(pts, col=1, T=ye-yb)

        rbf_interp = True if filling == 'rbf' else False
        fill_xy, fill_z = fill_gaps(pts_ext[:, [0, 1]], pts_ext[:, 2], rbf_interp=rbf_interp, debug=debug)

        if fill_xy is not None and fill_z is not None:
            fill_pts = np.hstack([fill_xy, fill_z[:,None]])
            pts_ext = np.concatenate([pts_ext, fill_pts])
            ids = (xb <= pts_ext[:,0]) & (pts_ext[:,0] <= xe) & (yb <= pts_ext[:,1]) & (pts_ext[:,1] <= ye)
            pts = pts_ext[ids]

    pts_ext = extend_periodically_point_cloud(pts, col=1, T=ye-yb, d_max=ext*d)

    x_ext, y_ext, z_ext = pts_ext[:,0], pts_ext[:,1], pts_ext[:,2]
    yb_ext, ye_ext = y_ext.min(), y_ext.max()

    weights = compute_point_weights(pts_ext, weighting=weighting)
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_ext, y_ext, z_ext, c=z_ext, label='ext pts', s=5*(weights/weights.min()), alpha=0.5) #Extended
        ax.scatter(x, y, z, color='k', s=5, label='orig pts') #Original
        plt.title('Weights and extension')
        plt.legend()
        plt.show()


    ty_ext = get_uniform_knot_vector(xb=yb, xe=ye, n=ny, mode='extended', ext=ext-2)

    bspl_ext = LSQBivariateSpline(x_ext, y_ext, z_ext, tx=tx, ty=ty_ext, w=weights, bbox=[xb, xe, yb_ext, ye_ext], kx=kx, ky=ky)

    nyy = ny + 2*(ext-1)
    Mcoeff_ext=bspl_ext.get_coeffs().reshape(nx+kx+1,nyy+ky+1)

    Mcoeff = Mcoeff_ext[:,ext-1:1-ext]


    if debug:
        #Build the spline
        bspl_rest = BivariateSpline()
        tyy_rest = get_uniform_knot_vector(xb=yb, xe=ye, n=ny, mode='periodic')
        txx = get_uniform_knot_vector(xb=xb, xe=xe, n=nx, mode='complete')
        bspl_rest.tck = txx, tyy_rest, Mcoeff.ravel()
        bspl_rest.degrees = kx, ky

        #Make the analytic data.
        ngrid = 100
        xx = np.linspace(xb, xe, ngrid+1)
        yy = np.linspace(yb, ye, ngrid+1)
        yy_ext = np.linspace(yb_ext, ye_ext, ngrid+1)
        X_ext, Y_ext = np.meshgrid(xx, yy_ext)
        X, Y = np.meshgrid(xx, yy)
        Z_ext = bspl_ext(xx, yy_ext).T
        Z_rest = bspl_rest(xx,yy).T
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        #Plot the surfaces.
        surf = ax.plot_surface(X_ext, Y_ext, Z_ext, color='g', alpha=0.5,
                            linewidth=0, antialiased=True, label='ext bvspl')

        surf = ax.plot_surface(X, Y, Z_rest, color='r', alpha=0.9,
                            linewidth=0, antialiased=True, label='rest bvspl')

        #Plot the points
        ax.scatter(pts_ext[:,0], pts_ext[:,1], pts_ext[:,2], color='b', alpha=0.5, s=2, label='ext pts') #Extended
        ax.scatter(x, y, z, color='k', label='ext pts') #Original
        #plt.legend() Due to a bug in matplotlib (3.5.1), this cant be uncomented
        plt.show()

    return Mcoeff
#

def fill_gaps(points, f, N=None, M=None, d=5, rbf_interp=False, debug=False):
    """
    Provided a 2D point cloud, with a field defined on it. This function
    detects the holes in the image and add new points on it interpolating the map
    at the new points.


    Arguments:
    -----------

        points : np.ndarray (n, 2)
            The cartesian coordinates of the points.

        f : np.ndarray (n,)
            The field values

        N, M : int, opt
            Default None. The resolution of the grid for searching the gap.
            If not provided, the average point density is used.

        d : float, opt
            Default 1.2. If N or M are None, d_ can be used as a factor to
            increase the estimated average point density.

        rbf_interp : bool, opt
            Default False. Whether to use radial basis functions to interpolate
            the field at the gap new points. If false, the average value of the
            gap neighboring points is used.

        debug : bool,
            Plot the process.
    Returns:
    ---------
        new_points, new_f : np.ndarray (n,)
            The new points with the interpolated field
    """

    pm, pM = points.min(axis=0), points.max(axis=0)
    if None in [M, N]:
        d *= points.shape[0] / np.product(pM-pm)
        N = M = np.round(np.sqrt(d)).astype(int)

    grid = np.ones((N,M), dtype=int)
    points.min(axis=0)

    v_tr = np.array([N-1, M-1]) / (pM-pm)
    points_tr = (points-pm)*v_tr
    ids = np.round(points_tr).astype(int)
    ids = (ids[None,:,0], ids[None,:,1])
    grid[ids] = 0
    grid = label(dilation(grid, footprint=np.ones((2,2))))

    new_points, new_f = [], []
    for reg in regionprops(grid):
        rm, cm, rM, cM = reg.bbox
        ids = (rm <= points_tr[:, 0]) & (points_tr[:, 0] <= rM) & (cm <= points_tr[:, 1]) & (points_tr[:, 1] <= cM)
        pts_, f_ = points[ids], f[ids]
        new_pts = reg.coords / v_tr + pm
        new_points.append(new_pts)
        if rbf_interp:
            interp = RBFInterpolator(y=pts_, d=f_, kernel='thin_plate_spline')
            fs = interp(new_pts)
        else:
            fs = np.full((new_pts.shape[0]), fill_value=f_.mean())
        new_f.append(fs)

    if not new_points or not new_f:
        return None, None

    new_points = np.vstack(new_points)
    new_f = np.concatenate(new_f)

    if debug:
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(grid.T, origin='lower')
        ax[0].scatter(points_tr[:,0], points_tr[:,1], s=.5)

        ax[1].scatter(points[:,0],     points[:,1],     c=f,     s=.5)
        ax[1].scatter(new_points[:,0], new_points[:,1], c=new_f, s=1)
        plt.show()

    return new_points, new_f
#
