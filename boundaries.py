
import numpy as np

from scipy.interpolate import BSpline

from utils.spatial import planar_coordinates, cart_to_polar, normalize
from utils.splines import knots_list, compute_rho_spline
from utils._code   import attribute_setter, attribute_checker


class Boundary:

    """
    Class to represent the open boundaries of a vascular mesh.
    """

    def __init__(self) -> None:

        #Metadata
        self.id : str = None

        #Local reference frame
        self.center : np.ndarray = None # Shape = (3,)
        self.normal : np.ndarray = None # Shape = (3,)
        self.v1     : np.ndarray = None # Shape = (3,)
        self.v2     : np.ndarray = None # Shape = (3,)

        #Locus
        self.points         : np.ndarray  = None # Shape = (N,3)
        self.points2D_cart  : np.ndarray  = None # Shape = (N,2)
        self.points2D_polar : np.ndarray  = None # Shape = (N,2)

        #To cast into a polydata
        self.faces : np.ndarray = None #Recomended shape is (N, 4) being the first column == 3. (Triangular faces)

        #Boundary curve
        self.rho_spl     : BSpline = None
        self.rho_coef    : np.ndarray = None # Shape = (n_knots_rho + k+1,)
        self.n_knots_rho : int = None
        self.k           : int = 3

        self.area : float = None
    #

    def __str__(self) -> str:
        if self.points is not None:
            npts = self.points.shape[0]
        else:
            npts = 0

        return "Boundary info: \n" \
               +f"\t id        : {self.id} \n"\
               +f"\t center    : {self.center}\n"\
               +f"\t normal    : {self.normal}\n"\
               +f"\t v1        : {self.v1}\n"\
               +f"\t v2        : {self.v2}\n"\
               +f"\t n points  : {npts}\n"\
               +f"\t rho_coeff : {self.rho_coef}\n"\
               +f"\t n knots   : {self.n_knots_rho}\n"\
               +f"\t k         : {self.k}"\
    #

    def save(self, fname):
        """
        Method to save a boundary dict.

        Arguments:
        -----------

            fname : str
                Filename for the json file containing the boundary data.
        """

        pass
        """
        boundary_dict = read_boundary_json()
        for k in boundary_dict['data']:
            v = getattr(self, k, None)
            if isinstance(v, np.ndarray):
                v=v.tolist()
            boundary_dict['data'][k] = v

        write_boundary_json(path=fname, data=boundary_dict, abs_path=True)
        """
    #

    def load(self, fname):
        """
        Method to load a boundary dict.

        Arguments:
        -----------

            fname : str
                Filename for the json file containing the boundary data.
        """

        pass
        """
        boundary_dict = read_boundary_json(path=fname, abs_path=True)['data']
        for k in ["center", "normal", "v1", "v2", "rho_coef"]:
            if boundary_dict[k] is not None:
                boundary_dict[k] = np.array(boundary_dict[k])

        self.set_data(**boundary_dict)
        if self.rho_coef is not None and self.n_knots_rho is not None:
            self.build_rho_spline()
        """
    #

    def set_data(self, update=False, build_splines=False, **kwargs):
        """
        Method to set attributes by means of kwargs.
        E.g.
            a = Boundary()
            a.set_data(center=np.zeros((3,)))

        """


        attribute_setter(self, **kwargs)

        if "points" in kwargs and update:
            self.from_3D_to_polar()

        if build_splines:
            self.build_rho_spline()
    #

    def from_3D_to_2D(self, pts=None):

        """
        Tansform 3D Cartesian points to local plannar coordinates
        of the local reference system and return them.

        I pts argument is None, the attribute points will be used and
        the result will be also stored in points2D_cart.
        points2D_cart will be set

        Arguments:
        -----------

            pts : np.ndarray (N,3)
                The array of 3D points to transform. If None, self.points will be used.
                Defaulting to None.

        Returns:
        ---------
            pts2d np.ndarray (N,2)
                The transformed points

        """

        attribute_checker(self, ['points', 'center', 'v1', 'v2'], extra_info='Cannot compute plannar coordinates.'+\
                                                                            f'Boundary with id {self.id} has no v1 and v2....')

        if pts is None:
            self.points2D_cart = planar_coordinates(self.points.T, c0=self.center, v1=self.v1, v2=self.v2).T
            return self.points2D_cart.copy()

        return planar_coordinates(points=pts.T, c0=self.center, v1=self.v1, v2=self.v2).T
    #

    def cartesian_2D_to_polar(self, pts, sort=True):
        """
        Tansform 2D Cartesian points to polar coordinates
        and return them.

        I pts argument is None, the attribute points2D_cart will be used and
        the result will be also stored in points2D_polar.

        Arguments:
        -----------

            pts : np.ndarray (N,2)
                The array of 2D points to transform. If None, self.point2D_polar will be used.
                Defaulting to None.

            sort : bool
                Default True. Whether to sort the returned list by angular coord.

        Returns:
        ---------
            pts2d np.ndarray (N,2)
                The transformed points

        """

        if pts is None:
            attribute_checker(self, atts=['points2d_cart'], extra_info=f'No points available to transform in polar coordinates at boundary {self.id}')
            self.points2D_polar = cart_to_polar(self.points2D_cart.T, sort=sort).T
            return self.points2D_polar.copy()

        return cart_to_polar(pts.T, sort=sort).T
    #

    def from_3D_to_polar(self, pts=None, sort=True):
        """
        Tansform 3D Cartesian points to plannar polar coordinates
        and return them.

        I pts argument is None, the attribute points will be used and
        the result will be also stored in points2D_cart and points2D_polar.

        Arguments:
        -----------

            pts : np.ndarray (N,3)
                The array of 3D points to transform. If None, self.points will be used.
                Defaulting to None.

            sort : bool
                Default True. Whether to sort the returned list by angular coord.

        Returns:
        ---------
            pts_polar np.ndarray (N,2)
                The transformed points

        """

        pts2D = None
        if pts is None:
            self.from_3D_to_2D(pts=pts)
        else:
            pts2D = self.from_3D_to_2D(pts=pts)

        pts_polar = self.cartesian_2D_to_polar(pts=pts2D, sort=sort)

        return pts_polar
    #

    def build_rho_spline(self):
        """
        Method to build rho function spline.
        """
        if self.rho_coef is None:
            if self.points2D_polar is not None:
                self.rho_coef = compute_rho_spline(polar_points=self.points2D_polar.T, n_knots=self.n_knots_rho, k=self.k)[0][:-self.k-1]
                self.compute_area()
            else:
                print("ERROR: Unable to build rho spline. Both points2D_polar and rho_coeff are None....")

        t = knots_list(0, 2*np.pi, self.n_knots_rho, mode='periodic')

        self.rho_spl = BSpline(t=t, c=self.rho_coef, k=self.k, extrapolate='periodic')
    #

    def compute_area(self, th_ini=0, th_end=None):
        """
        Compute the area of the boundary by integration of the
        rho spline. Optional th_ini and th_end parameters allow to compute
        the area of a section. Defaulting to the total area

        Arguments:
        -----------

            th_ini : float [0, 2pi]
                The begining of the interval to compute the area.

            th_end : float [0, 2pi]
                The end of the interval to compute the area.



        Returns:
        ---------

            area : float
                The computed area.

        """


        if self.rho_spl is None:
            self.build_rho_spline()

        if th_end is None:
            th_end = 2*np.pi

        area = self.rho_spl.integrate(th_ini, th_end, extrapolate='periodic')
        if th_ini == 0 and th_end == 2*np.pi:
            self.area = area

        return area

    @staticmethod
    def from_polydata(pdt):
        """
        Get a new Boundary object built from data stored in a pyvista PolyData.

        Arguments:
        ------------
            pdt : pv.PolyData
                The polydata with the points and faces fields.

        Returns:
        ---------
            b : Boundary
                The boundary object with data derived from the polydata.
        """

        if 'Normals' not in pdt.cell_data:
            pdt = pdt.compute_normals(cell_normals=True)


        b = Boundary()
        b.set_data(center = np.array(pdt.center),
                   normal = normalize(pdt.get_array('Normals', preference='cell').mean(axis=0)),
                   points = pdt.points,
                   faces  = pdt.faces
                  )
        return b
    #
#
