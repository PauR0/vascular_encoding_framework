

import numpy as np

from ..utils.splines import BiSpline, semiperiodic_LSQ_bivariate_approximation

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
        metadata array is [2, n_knots_tau, n_knots_theta].

        Returns
        -------
            md : np.ndarray
        """

        md = np.array([4,
                      self.kx,
                      self.ky,
                      self.n_knots_x,
                      self.n_knots_y,
                      ])

        return md
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
