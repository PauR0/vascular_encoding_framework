
import numpy as np

from scipy.interpolate import BSpline

from ..utils.spatial import planar_coordinates, cart_to_polar, normalize
from ..utils.splines import knots_list, compute_rho_spline
from ..utils._code   import Tree, Node, attribute_checker


class Boundary(Node):

    """
    Class to represent the open boundaries of a vascular mesh.
    """

    def __init__(self, nd=None) -> None:

        #Added Node parent.
        super().__init__(nd=None)

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

        #Inherit node data
        if nd is not None:
            self.set_data(**nd.__dict__)
    #

    def to_dict(self, compact=True, serialize=True):
        """
        Make a dictionary with the Boundary object atributes. If compact == True, only main Node
        attributes with the main Boundary attributes are added to the dictionary, otherwise, each
        one is added.

        Arguments:
        ----------

            compact : bool, opt
                Default True. Whether to exclude non essential attributes in the outdict..

            serialize : bool, opt
                Default True. Whether to serialize objects such as numpy array, to be
                json-writable.

        Returns:
        --------

            outdict : dict
                The boundary attributes stored in a dictionary.
        """

        outdict = {}
        if compact:
            atts = list(Node().__dict__.keys()) + ['center', 'normal', 'v1', 'v2']
        else:
            atts = self.__dict__.keys()

        for k in atts:
            v = self.__dict__[k]
            if serialize and isinstance(v, (set, np.ndarray)):
                if isinstance(v, np.ndarray):
                    if v.dtype == 'float32': v=v.astype(float)
                    v = v.astype(float)
                v = list(v)
            outdict[k] = v

        return outdict
    #

    def set_data(self, to_numpy=True, update=False, build_splines=False, **kwargs):
        """
        Method to set attributes by means of kwargs.
        E.g.
            a = Boundary()
            a.set_data(center=np.zeros((3,)))

        Arguments:
        -------------

            to_numpy : bool, opt
                Default True. Whether to cast numeric array-like sequences to numpy ndarray.

            update : bool, opt
                Default False. Whether to update points2D* attributes after setting passing points att.

            build_splines : bool, opt
                Default False. Whether to build the rho spline attribute for the boundary object.
        """

        super().set_data(to_numpy=to_numpy, **kwargs)

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

        attribute_checker(self, ['points', 'center', 'v1', 'v2'], info='Cannot compute plannar coordinates.'+\
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
            attribute_checker(self, atts=['points2d_cart'], info=f'No points available to transform in polar coordinates at boundary {self.id}')
            self.points2D_polar = cart_to_polar(self.points2D_cart.T, sort=sort).T
            return self.points2D_polar.copy()

        return cart_to_polar(pts.T, sort=sort).T
    #

    def from_3D_to_polar(self, pts=None, sort=False):
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
                Default False. Whether to sort the returned list by angular coord.

        Returns:
        ---------
            pts_polar np.ndarray (N,2)
                The transformed points

        """

        pts2D = None
        if pts is None:
            self.from_3D_to_2D(pts=pts, sort=True)
        else:
            pts2D = self.from_3D_to_2D(pts=pts)

        pts_polar = self.cartesian_2D_to_polar(pts=pts2D, sort=sort)

        return pts_polar
    #

    def build_rho_spline(self):
        """
        Method to build rho function spline.
        TODO: Sorting the points by theta coord should be exclusively done here, there will be explosions if I don't sort this out.
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
    #

    def extract_from_polydata(self, pdt):
        """
        Extract main data from a pyvista PolyData.

        Arguments:
        ------------
            pdt : pv.PolyData
                The polydata with the points and faces attributes.

        Returns:
        ---------
            b : Boundary
                The boundary object with data derived from the polydata.
        """

        if 'Normals' not in pdt.cell_data:
            pdt = pdt.compute_normals(cell_normals=True, inplace=False)

        self.set_data(center = np.array(pdt.center),
                      normal = normalize(pdt.get_array('Normals', preference='cell').mean(axis=0)),
                      points = pdt.points,
                      faces  = pdt.faces
                     )
    #

    def to_polydata(self):
        """
        If points attribute is not None, build a pyvista PolyData object with them. If
        faces are not None, they are also added to PolyData.

        Returns:
        ---------
            poly : pv.PolyData
                The polydata containing the Boundaries.
        """

        if not attribute_checker(self, atts=['points', 'faces']):
            return None

        poly = pv.PolyData()
        if self.points is not None:
            poly.point = self.points
        if self.faces is not None:
            poly.faces = self.faces

        return poly
    #
#

class Boundaries(Tree):
    """
    A class containing the boundaries inheriting structure from Tree class.
    """

    #No init required since parent init suffice.

    def __init__(self, hierarchy=None) -> None:

        super().__init__()

        if hierarchy is not None:
            self.graft(Tree.from_hierarchy_dict(hierarchy=hierarchy))
            for i, n in self.items():
                self[i] = Boundary(nd=n)
    #

    #

    @staticmethod
    def read(filename):
        pass
    #

    def translate(self):
        #TODO
        pass
    #

    def rotate(self):
        #TODO
        pass
    #

    def scale(self):
        #TODO
        pass
    #
#
