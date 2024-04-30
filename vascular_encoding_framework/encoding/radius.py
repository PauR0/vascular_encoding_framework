

import numpy as np

from ..messages import error_message
from ..utils.splines import BiSpline, semiperiodic_LSQ_bivariate_approximation
from ..utils._code import attribute_checker
from ..utils.misc import split_metadata_and_fv

class Radius(BiSpline):

    def __init__(self):

        super().__init__()

        self.x0 = 0
        self.x1 = 1
        self.extra_x = 'constant'

        self.y0 = 0
        self.y1 = 2*np.pi
        self.extra_y = 'periodic'
    #

    def set_parameters_from_centerline(self, cl):
        """
        This method set the radius bounds equal to the Centerline object
        passed.

        Arguments:
        -----------

            cl : Centerline
                The Centerline of the vessel
        """

        self.x0 = cl.t0
        self.x1 = cl.t1
    #

    def get_metadata(self):
        """
        This method returns a copy of the metadata array.

        As of this code version the
        metadata array is [4, k_tau, k_theta, n_knots_tau, n_knots_theta].

        Returns
        -------
            md : np.ndarray

        See Also
        --------
        :py:meth:`get_metadata`

        """

        md = np.array([5,
                      self.kx,
                      self.ky,
                      self.n_knots_x,
                      self.n_knots_y,
                      ])

        return md
    #

    def set_metadata(self, md):
        """
        This method extracts and sets the attributes from a metadata array.

        As of this code version the
        metadata array is [5, k_tau, k_theta, n_knots_tau, n_knots_theta].

        Returns
        -------

            md : np.ndarray
                The metadata array.

        See Also
        --------
        :py:meth:`get_metadata`

        """

        self.set_parameters(
            build = False,
            kx        = round(md[1]),
            ky        = round(md[2]),
            n_knots_x = round(md[3]),
            n_knots_y = round(md[4]),
        )
    #

    def get_feature_vector_length(self):
        """
        This method returns the length of the feature vector considering the spline parameters.

        If nx, ny are the amount of internal knots in each component and kx, ky are the degrees of
        the polynomial BSplines of each component, the length of the radius feature vector is
        (nx+kx+1)*(ny*ky+1).


        Returns
        -------

            rk : int
                The length of the radius feature vector.

        """

        if not attribute_checker(self, ['n_knots_x', 'n_knots_y', 'kx', 'ky'], info="Cannot compute the Radius feature vector length."):
            return None
        rk = (self.n_knots_x+self.kx+1)*(self.n_knots_y+self.ky+1)
        return rk
    #

    def to_feature_vector(self, add_metadata=True):
        """
        Convert the Radius object to its feature vector repressentation.

        The feature vector version of a Radius object consist in the raveled radius coefficients.
        If add_metada is True (which is the default), a metadata array is appended at the beggining
        of the feature vector. The first entry of the metadata vector is the total number of
        metadata, making it look like [n, md0, ..., mdn], read more about it in get.

        Arguments
        ---------

            add_metadata: bool, optional
                Default True. Wether to append metadata at the beggining of the feature vector.

        Return
        ------

            fv : np.ndarray
                The feature vector according to mode. The shape of each feature vector changes acoordingly.


        See Also
        --------
        :py:meth:`get_metadata`
        :py:meth:`from_feature_vector`
        """


        fv = self.coeffs.ravel()

        if add_metadata:
            fv = np.concatenate([self.get_metadata(), fv])

        return fv
    #

    @staticmethod
    def from_feature_vector(fv):
        """
        Build a Centerline object from a feature vector.

        Note that in order to build the Centerline, the feature vector must start with the metadata
        array. Read more about the metadata array at `get_metadata` method docs.


        Arguments
        ---------

            fv : np.ndarray (N,)
                The feature vector with the metadata at the beggining.


        Return
        ------
            rd : Radius
                The Radius object built from the feature vector.

        See Also
        --------
        :py:meth:`to_feature_vector`
        :py:meth:`get_metadata`
        """


        md, fv = split_metadata_and_fv(fv)

        rd = Radius()
        rd.set_metadata(md)

        r , k = (rd.n_knots_x + rd.kx + 1), (rd.n_knots_y + rd.ky + 1)
        rk = r * k
        if len(fv) != rk:
            error_message(f"Cannot build a Radius object from feature vector. Expected rk knots ((tx+kx+1) * (ty+ky+1)) coefficients and {len(fv)} were provided.")
            return

        rd.set_parameters(
            build=True,
            coeffs = fv,
        )

        return rd
    #

    @staticmethod
    def from_points(points, tau_knots, theta_knots, filling='mean', cl=None, debug=False):
        """
        Function to build a Radius object from an array of points in the Vessel Coordinate System.
        Radius object are a specialized Bivarate Splines. This function allow to build such objects
        by performing a least squares approximation using the longitudinal and angular coordinates
        to model the radius.

        Arguments
        ---------

            points : np.ndarray (N, 3)
                The vessel coordinates point array to be approximated.

            tau_knots, theta_knots : int
                The number of internal knots in longitudinal and angular dimensions respectively.
                TODO: Allow building non-uniform BSplines.

            filling : {'mean', 'rbf'}, optional
                The method used to fill detected holes.

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
        rd.set_parameters(build=True,
                          n_knots_x = tau_knots,
                          n_knots_y = theta_knots,
                          coeffs    = semiperiodic_LSQ_bivariate_approximation(x=points[:,0],
                                                                               y=points[:,1],
                                                                               z=points[:,2],
                                                                               nx=tau_knots,
                                                                               ny=theta_knots,
                                                                               kx=rd.kx,
                                                                               ky=rd.ky,
                                                                               bounds=(rd.x0,
                                                                                       rd.x1,
                                                                                       rd.y0,
                                                                                       rd.y1),
                                                                               filling=filling,
                                                                               debug=debug))
        return rd
    #
#
