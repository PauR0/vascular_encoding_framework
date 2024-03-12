
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import (splrep, splev, make_lsq_spline,
                               BSpline, BivariateSpline, LSQBivariateSpline,
                               RBFInterpolator)

from skimage.morphology import dilation
from skimage.measure import label, regionprops

import messages as msg
from utils._code import attribute_checker, attribute_setter


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
        self.extra   : str        = 'linear' #{'linear', 'constant'}

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

    def evaluate(self, t):
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

        if not attribute_checker(self, ['_spl'], extra_info="can't evaluate spline, it has not been built..."):
            return False


        if self.extra == 'constant':
            tt = np.clip(t, a_min=self.t0, a_max=self.t1)
            p  = np.array(self._spl(tt))

        elif self.extra == 'linear':
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

        if not attribute_checker(self, ['k', 'n_knots', 'coeffs'], extra_info="cant build splines."):
            return False

        if self.knots is None:
            self.knots = knots_list(self.t0, self.t1, self.n_knots, mode='complete')

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
        self.extra_y   : str        = 'periodic' #{'constant', 'periodic'}

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

        if not attribute_checker(self, ['kx', 'ky', 'coeffs'], extra_info="cant build splines."):
            return False

        if self.knots_x is None and self.n_knots_x is None:
            msg.error_message("cant build bivariate splines. The knots and amount of knots for the first (x) parameter is None")
        elif self.knots_x is None and self.n_knots_x is not None:
            mode='complete'
            if self.extra_x == 'periodic':
                mode='periodic'
            self.knots_x = knots_list(self.x0, self.x1, self.n_knots_x, mode=mode)

        if self.knots_y is None and self.n_knots_y is None:
            msg.error_message("cant build bivariate splines. The knots and amount of knots for the second parameter (y) is None")
        elif self.knots_y is None and self.n_knots_y is not None:
            mode='complete'
            if self.extra_y == 'periodic':
                mode='periodic'
            self.knots_y = knots_list(self.y0, self.y1, self.n_knots_y, mode=mode)

        self._bispl         = BivariateSpline()
        self._bispl.tck     = self.knots_x, self.knots_y, self.coeffs.ravel()
        self._bispl.degrees = self.kx, self.ky
    #

    def evaluate(self, x, y, grid=False):
        """
        Evaluate the Bivariate splines at x and y.

        Arguments:
        -------------

            x : float or array-like
                The first parameter values

            y : float or array-like
                The second parameter values

            grid : bool, opt
                Default False. Whether to evaluate the spline at the
                grid built by the carthesian product of x and y.

        Returns:
        ---------

            z : float or array-like
                The values of spl(x, y)

        """

        def clip_periodic(a, T=2*np.pi):
            return a - (a//T)*T

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

def knots_list(s0, s1, n, mode='complete', ext=None, k=3):
    """ Generates a B-Spline uniform list of knots

        Given s0 and s1 it returns a numpy array of nodes for a B-Spline

        Parameters
        ----------

        s0 : float
            Starting parameter value

        s1 : float
            Ending parameter value

        n : int
            Number of B-Spline knots from s0 to s1.

        ext : int
            Defaulting to k+1. The times to extend the knot vector from
            both ends preserving the separation between nodes.

        k : int
            The degree of the spline.

        mode : {'simple','complete', 'extended', 'periodic'} , optional
            If mode='simple' then l is a numpy linspace from s0 to s1.

            If mode='complete' l contains [s0,s0,s0] at the beginning and
            [s1,s1,s1] at the end. Needed for scipy.interpolation.BSpline.

            If mode = 'extended' and ext is not none, it extends ext times the
            knot vector from both ends preserving the spacing.

            If mode = 'periodic' then the list of knots is ready for periodic
            B-Splines fitted with splrep. It is the same as setting mode='extended'
            and ext=k+1.

        Output
        ------
            l: the list of knots
    """
    l = np.linspace(s0,s1,n+2)
    d = (s1-s0)/(n+1)

    if mode == 'periodic':
        mode = 'extended'
        ext = k+1

    if mode == 'simple':
        l = l[1:-1]
    elif mode == 'extended':
        if ext is None:
            ext = k+1
        l = np.concatenate([l[0] + np.arange(-ext, 0)*d, l, l[-1] + np.arange(ext+1)[1:]*d])[1:-1]

    else:
        initial  = np.array( [s0]*3 )
        terminal = np.array( [s1]*3 )
        l = np.concatenate((initial,l,terminal))

    return l
#

def lsq_spline_smoothing(points,
                         knots,
                         k=3,
                         param_values=None,
                         norm_param=False,
                         n_weighted_ini=1,
                         n_weighted_end=1,
                         weight_ratio=1.0):
    """
    Compute the smoothing spline of a list of d-dimensional points.

    Points must be a numpy array of dimension Nxd for a list of N
    d-dimensional points.

    Arguments:
    -----------

        points : np.ndarray (N, d)
            The array of points.

        knots : int or array-like
            The knots where the lsq will be performed.

        k : int, opt
            Default is 3. The spline polynomial degree.

        param_values : array-like (N,), opt
            The parameter values for each point. Must be a increasing sequence.
            If not passed it is computed as the normalized distance traveled.

        norm_param : bool, opt
            Whether to normalize the domain to interval [0, 1].

        n_weighted_ini : int opt,
            Default 1. The amount of points weighted at the begining of the list.
            Useful to "force" the normal at the begining.

        n_weighted_end  : int opt,
            Default 1. The amount of points weighted at the end of the list.
            Useful to "force" the normal at the begining.

        weight_ratio : float, opt.
            Default 1.0. The ratio of rate of weights for weighted points.

    Returns:
    ----------
        spl : BSpline
            The approximating spline object of scipy.interpolate.
    """

    N = points.shape[0]

    if param_values is None:
        param_values = [0.0]
        for i in range(1, N):
            d = np.linalg.norm(points[i] - points[i-1])
            param_values.append(param_values[-1]+d)
    param_values = np.array(param_values)

    if norm_param:
        param_values /= param_values[-1]

    if isinstance(knots, int):
        #Computing knots
        knots = knots_list(param_values[0], param_values[-1], knots, mode='complete')

    #Computing weights taking into account fixed nodes
    w = compute_n_weights(n=N, n_weighted_ini=n_weighted_ini, n_weighted_end=n_weighted_end, weight_ratio=weight_ratio)

    #Compute spline
    spl = make_lsq_spline(x=param_values, y=points, t=knots, k=k, w=w)

    return spl
#

def compute_n_weights(n, n_weighted_ini=1, n_weighted_end=1, weight_ratio=None, normalized=True):
    """
    Compute weights for univariate splines. Extra weighting can be assigned to
    initial and ending points to improve normals at the extrema.
    """
    if weight_ratio is None:
        weight_ratio = 2

    w = np.ones((n,))
    w[:n_weighted_ini]  = weight_ratio
    w[-n_weighted_end:] = weight_ratio
    if normalized:
        w /= np.linalg.norm(w)

    return w
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

    knots = knots_list(0,2*np.pi, n_knots, mode='simple')

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

def semiperiodic_LSQ_bivariate_approximation(x, y, z, nx, ny, weighting=None, ext=None, kx=3, ky=3, bounds=None, fill=True, debug=False):
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

        fill : bool, opt
            Whether to add interpolated points at detected gaps in the x-y point cloud.

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

    tx = knots_list(s0=xb, s1=xe, n=nx, mode='simple')
    ty = knots_list(s0=yb, s1=ye, n=ny, mode='extended', ext=ext-1)
    d = ty[1] - ty[0]

    pts = np.concatenate((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)), axis=1)
    pts_ext = extend_periodically_point_cloud(pts, col=1, T=ye-yb, d_max=ext*d)

    if fill:
        fill_xy, fill_z = fill_gaps(pts_ext[:, [0, 1]], pts_ext[:, 2], debug=debug)
        print(fill_xy.shape)
        print(fill_z.shape)
        fill_pts = np.hstack([fill_xy, fill_z[:,None]])
        pts_ext = np.concatenate([pts_ext, fill_pts])

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


    ty_ext = knots_list(s0=yb, s1=ye, n=ny, mode='extended', ext=ext-1)

    bspl_ext = LSQBivariateSpline(x_ext, y_ext, z_ext, tx=tx, ty=ty_ext, w=weights, bbox=[xb, xe, yb_ext, ye_ext], kx=kx, ky=ky)

    nyy = ny + 2*(ext-1)
    Mcoeff_ext=bspl_ext.get_coeffs().reshape(nx+kx+1,nyy+ky+1)

    Mcoeff = Mcoeff_ext[:,ext-1:1-ext]


    if debug:
        #Build the spline
        bspl_rest = BivariateSpline()
        tyy_rest = knots_list(s0=yb, s1=ye, n=ny, mode='periodic')
        txx = knots_list(s0=xb, s1=xe, n=nx, mode='complete')
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

def fill_gaps(points, f, N=None, M=None, d=2, debug=False):
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
    grid = label(dilation(grid, footprint=np.ones((5,5))))

    new_points, new_f = [], []
    for reg in regionprops(grid):
        rm, cm, rM, cM = reg.bbox
        ids = (rm <= points_tr[:, 0]) & (points_tr[:, 0] <= rM) & (cm <= points_tr[:, 1]) & (points_tr[:, 1] <= cM)
        pts_, f_ = points[ids], f[ids]
        interp = RBFInterpolator(y=pts_, d=f_, kernel='thin_plate_spline')
        new_pts = reg.coords / v_tr + pm
        new_points.append(new_pts)
        new_f.append(interp(new_pts))

    print(new_points)
    print(new_f)

    new_points = np.vstack(new_points)
    new_f = np.concatenate(new_f)
    print(new_points.shape)
    print(new_f.shape)

    if debug:
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(grid.T, origin='lower')
        ax[0].scatter(points_tr[:,0], points_tr[:,1], s=.5)

        ax[1].scatter(points[:,0],     points[:,1],     c=f,     s=.5)
        ax[1].scatter(new_points[:,0], new_points[:,1], c=new_f, s=1)
        plt.show()

    return new_points, new_f