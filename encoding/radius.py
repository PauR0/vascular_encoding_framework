
import numpy as np

from utils.splines import BiSpline, semiperiodic_LSQ_bivariate_approximation

class Radius(BiSpline):

    def __init__(self):

        super().__init__()
        self.x0 = 0
        self.x1 = 1
        self.y0 = 0
        self.y1 = 2*np.pi
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

    @staticmethod
    def from_points(points, tau_knots, theta_knots, cl=None, debug=False):
        """
        Function to build a Radius object from an array of points in the Vessel Coordinate System.
        Radius object are a specialized Bivarate Splines. This function allow to build such objects
        by performing a least squares approximation using the longitudinal and angular coordinates
        to model the radius.

        Arguments:
        ------------

            points : np.ndarray (N, 3)
                The vessel coordinates point array to be approximated.

            tau_knots, theta_knots : int
                The number of internal knots in longitudinal and angular dimensions respectively.
                TODO: Allow building non-uniform BSplines.

            cl : Centerline, opt
                Default None. The centerline associated to the radius.

            debug : bool, opt
                Default False. Whether to show plots during the fitting process.

        Returns:
        ----------
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
                                                                               debug=debug))
        return rd
    #
#
