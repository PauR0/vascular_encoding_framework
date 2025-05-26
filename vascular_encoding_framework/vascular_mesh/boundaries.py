from __future__ import annotations

import os

import numpy as np
import pyvista as pv
from scipy.interpolate import BSpline

from .._base import Node, Tree, attribute_checker
from ..messages import error_message
from ..splines import compute_rho_spline, get_uniform_knot_vector
from ..utils._io import read_json, write_json
from ..utils.spatial import cart_to_polar, normalize, planar_coordinates


class Boundary(Node):
    """Class to represent the open boundaries of a vascular mesh."""

    def __init__(self, nd=None) -> None:
        # Added Node parent.
        super().__init__(nd=None)

        # Local reference frame
        self.center: np.ndarray = None  # Shape = (3,)
        self.normal: np.ndarray = None  # Shape = (3,)
        self.v1: np.ndarray = None  # Shape = (3,)
        self.v2: np.ndarray = None  # Shape = (3,)

        # Locus
        self.points: np.ndarray = None  # Shape = (N,3)
        self.points2D_cart: np.ndarray = None  # Shape = (N,2)
        self.points2D_polar: np.ndarray = None  # Shape = (N,2)

        # To cast into a polydata
        # Recommended shape is (N, 4) being the first column == 3. (Triangular
        # faces)
        self.faces: np.ndarray = None

        # Boundary curve
        self.rho_spl: BSpline = None
        self.rho_coef: np.ndarray = None  # Shape = (n_knots_rho + k+1,)
        self.n_knots_rho: int = None
        self.k: int = 3

        # Inherit node data
        if nd is not None:
            self.set_data(**nd.__dict__)

    def to_dict(self, compact=True, serialize=True):
        """
        Make a dictionary with the Boundary object attributes. If compact == True, only main Node
        attributes with the main Boundary attributes are added to the dictionary, otherwise, each
        one is added.

        Parameters
        ----------
        compact : bool, opt
            Default True. Whether to exclude non essential attributes in the outdict.
        serialize : bool, opt
            Default True. Whether to serialize objects such as numpy array, to be
            json-writable.

        Returns
        -------
        outdict : dict
            The boundary attributes stored in a dictionary.
        """

        outdict = {}
        if compact:
            atts = list(Node().__dict__.keys()) + ["center", "normal", "v1", "v2"]
        else:
            atts = self.__dict__.keys()

        for k in atts:
            v = self.__dict__[k]
            if serialize and isinstance(v, (set, np.ndarray)):
                if isinstance(v, np.ndarray):
                    if v.dtype == "float32":
                        v = v.astype(float)
                    v = v.astype(float)
                v = list(v)
            outdict[k] = v

        return outdict

    def set_data(self, to_numpy=True, update=False, build_splines=False, **kwargs):
        """
        Set attributes by means of kwargs.

        E.g.
            a = Boundary()
            a.set_data(center=np.zeros((3,))).

        Parameters
        ----------
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

    def from_3D_to_2D(self, pts=None):
        """
        Transform 3D Cartesian points to local planar coordinates
        of the local reference system and return them.

        I pts argument is None, the attribute points will be used and
        the result will be also stored in points2D_cart.
        points2D_cart will be set

        Parameters
        ----------
        pts : np.ndarray (N,3)
            The array of 3D points to transform. If None, self.points will be used.
            Defaulting to None.

        Returns
        -------
        pts2d np.ndarray (N,2)
            The transformed points

        """

        attribute_checker(
            self,
            ["points", "center", "v1", "v2"],
            info="Cannot compute planar coordinates."
            + f"Boundary with id {self.id} has no v1 and v2....",
        )

        if pts is None:
            self.points2D_cart = planar_coordinates(
                self.points.T, c0=self.center, v1=self.v1, v2=self.v2
            ).T
            return self.points2D_cart.copy()

        return planar_coordinates(points=pts.T, c0=self.center, v1=self.v1, v2=self.v2).T

    def cartesian_2D_to_polar(self, pts, sort=True):
        """
        Transform 2D Cartesian points to polar coordinates
        and return them.

        I pts argument is None, the attribute points2D_cart will be used and
        the result will be also stored in points2D_polar.

        Parameters
        ----------
        pts : np.ndarray (N,2)
            The array of 2D points to transform. If None, self.point2D_polar will be used.
            Defaulting to None.
        sort : bool
            Default True. Whether to sort the returned list by angular coord.

        Returns
        -------
        pts2d np.ndarray (N,2)
            The transformed points

        """

        if pts is None:
            attribute_checker(
                self,
                atts=["points2d_cart"],
                info=f"No points available to transform in polar coordinates at boundary {self.id}",
            )
            self.points2D_polar = cart_to_polar(self.points2D_cart.T, sort=sort).T
            return self.points2D_polar.copy()

        return cart_to_polar(pts.T, sort=sort).T

    def from_3D_to_polar(self, pts=None, sort=False):
        """
        Transform 3D Cartesian points to planar polar coordinates
        and return them.

        I pts argument is None, the attribute points will be used and
        the result will be also stored in points2D_cart and points2D_polar.

        Parameters
        ----------
        pts : np.ndarray (N,3)
            The array of 3D points to transform. If None, self.points will be used.
            Defaulting to None.

        sort : bool
            Default False. Whether to sort the returned list by angular coord.

        Returns
        -------
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

    def build_rho_spline(self):
        """
        Build rho function spline.
        TODO: Sorting the points by theta coord should be exclusively done here, there will be explosions if I don't sort this out.
        """
        if self.rho_coef is None:
            if self.points2D_polar is not None:
                self.rho_coef = compute_rho_spline(
                    polar_points=self.points2D_polar.T, n_knots=self.n_knots_rho, k=self.k
                )[0][: -self.k - 1]
                self.compute_area()
            else:
                print(
                    "ERROR: Unable to build rho spline. Both points2D_polar and rho_coeff are None...."
                )

        t = get_uniform_knot_vector(0, 2 * np.pi, self.n_knots_rho, mode="periodic")

        self.rho_spl = BSpline(t=t, c=self.rho_coef, k=self.k, extrapolate="periodic")

    def compute_area(self, th_ini=0, th_end=None):
        """
        Compute the area of the boundary by integration of the
        rho spline. Optional th_ini and th_end parameters allow to compute
        the area of a section. Defaulting to the total area.

        Parameters
        ----------
        th_ini : float [0, 2pi]
            The beginning of the interval to compute the area.
        th_end : float [0, 2pi]
            The end of the interval to compute the area.

        Returns
        -------
        area : float
            The computed area.

        """

        if self.rho_spl is None:
            self.build_rho_spline()

        if th_end is None:
            th_end = 2 * np.pi

        area = self.rho_spl.integrate(th_ini, th_end, extrapolate="periodic")

        return area

    def extract_from_polydata(self, pdt):
        """
        Extract main data from a pyvista PolyData.

        Parameters
        ----------
        pdt : pv.PolyData
            The polydata with the points and faces attributes.

        Returns
        -------
        b : Boundary
            The boundary object with data derived from the polydata.
        """

        if "Normals" not in pdt.cell_data:
            pdt = pdt.compute_normals(cell_normals=True, inplace=False)

        self.set_data(
            center=np.array(pdt.center),
            normal=normalize(pdt.get_array("Normals", preference="cell").mean(axis=0)),
            points=pdt.points,
            faces=pdt.faces,
        )

    def to_polydata(self):
        """
        If points attribute is not None, build a pyvista PolyData object with them. If
        faces are not None, they are also added to PolyData.

        TODO: Add node attributes to user_dict.

        Returns
        -------
        poly : pv.PolyData
            The polydata containing the Boundaries.
        """

        attribute_checker(self, atts=["points", "faces"])

        poly = pv.PolyData()
        if self.points is not None:
            poly.points = self.points
        if self.faces is not None:
            poly.faces = self.faces

        return poly

    def translate(self, t):
        """
        Translate the Boundary object.

        Parameters
        ----------
        t : np.ndarray (3,)
            The translation vector.
        update : bool, optional
            Default True. Whether to rebuild the splines after the transformation.
        """

        if self.center is not None:
            self.center += t.reshape((3,))

        if self.points is not None:
            self.points += t.reshape((3,))

        if self.rho_coef is not None:
            self.rho_coeffs += t.reshape((3,))
            self.build_rho_spline()

    def scale(self, s):
        """
        Scale the Boundary object.

        Parameters
        ----------
        s : float
            The scale factor.
        """

        if self.center is not None:
            self.center *= s

        if self.points is not None:
            self.points *= s

        if self.rho_coef is not None:
            self.rho_coef *= s
            self.build_rho_spline()

    def rotate(self, r):
        """
        Rotate the Boundary object.

        Parameters
        ----------
        r : np.ndarray (3, 3)
            The rotation matrix.

        See Also
        --------
        :py:meth:`Centerline.rotate`
        """

        # ensure normality of the rotation matrix columns
        r /= np.linalg.norm(r, axis=0)

        if self.center is not None:
            self.center = (r @ self.center.reshape(3, 1)).reshape((3,))

        if self.normal is not None:
            self.normal = (r @ self.normal.reshape(3, 1)).reshape((3,))

        if self.v1 is not None:
            self.v1 = (r @ self.v1.reshape(3, 1)).reshape((3,))

        if self.v2 is not None:
            self.v2 = (r @ self.v2.reshape(3, 1)).reshape((3,))

        if self.points is not None:
            self.points = (r @ self.points.T).T

        if self.rho_coef is not None:
            self.rho_coeffs = (r @ self.rho_coeffs.T).T
            self.build_rho_spline()


class Boundaries(Tree):
    """A class containing the boundaries inheriting structure from Tree class."""

    def __init__(self, hierarchy=None) -> None:
        Tree.__init__(self=self, _node_type=Boundary)

        if hierarchy is not None:
            self.graft(Boundaries.from_dict(bds=hierarchy))

    def to_dict(self, compact=True, serialize=True):
        """
        Convert the Boundaries object into a python dictionary. If the serialize argument is True,
        numpy arrays will be casted to python lists (for json compatibility).

        Parameters
        ----------
        compact : bool, opt
            Default True. Whether to exclude non-essential attribute of the boundary objects,
            or include them all. To see which attributes are essential see Boundary.to_dict docs.
        serialize : bool, opt
            Default True. Whether to turn numpy arrays to lists.

        Returns
        -------
            outdict : dict
                The output python dictionary.

        See Also
        --------
        :py:meth:`from_dict`
        """

        outdict = {
            i: node.to_dict(compact=compact, serialize=serialize) for i, node in self.items()
        }

        return outdict

    def save(self, fname, binary=True):
        """
        Write Boundaries objects. Given an fname, this method writes the essential
        data from each boundary in a json file. If possible it builds a pv.MultiBlock with
        a PolyData of the boundaries.

        Parameters
        ----------
        fname : str
            The boundaries json file name.
        binary : bool, opt
            Default True. Whether to write boundary multiblock in binary or ascii.

        See Also
        --------
        :py:meth:`read`
        """

        fname, _ = os.path.splitext(fname)

        outdict = self.to_dict(compact=True, serialize=True)

        write_json(f"{fname}.json", outdict, overwrite=True)

        mbfname = f"{fname}.vtm"
        outmultiblock = self.to_multiblock()
        if outmultiblock.n_blocks:
            outmultiblock.save(filename=mbfname, binary=binary)

    def to_multiblock(self):
        """
        Cast Boundaries object in a pyvista MultiBlock object.

        All the Boundary objects stored in it will be converted to polydata objects.

        Returns
        -------
        mb : pv.MultiBlock
            The multiblock object with the polydatas.

        See Also
        --------
            :py:meth:`Boundary.to_polydata`
        """

        mb = pv.MultiBlock()

        for i, bd in self.items():
            polybd = bd.to_polydata()
            if polybd is not None:
                mb[i] = polybd

        return mb

    @staticmethod
    def from_dict(bds: dict[str, dict]) -> Boundaries:
        """
        Create a Boundaries object from a dictionary.

        The dictionary must contain the Boundary objects as dictionaries themselves. Each
        boundary-dict must have the entries id, parent, and children.

        > Note that children must be an iterable of 'ids' that will be turned into a set,
        duplications of children ids are disregarded.

        In the following example, a Boundaries object is created with a root node with id '1', with
        a child node '2', and whose center is at (x1,y1,z1). The node '2', has a child '0', its
        parent is '1', and its center is (x2,y2,z2). Finally, node '0', has no children, its parent
        is '2' and its center is (x0,y0,z0).

        hierarchy = {"1" : {"id"       : "1"
                            "parent"   : None,
                            "center"   : [ x1, y1, z1],
                            "children" : {"2"}
                           }
                     "2" : {"id"       : "2"
                            "parent"   : '1',
                            "center"   : [ x2, y2, z2],
                            "children" : {"0"}
                           }
                     "0" : {"id"       : "0",
                            "parent"   : '2',
                            "center"   : [ x0, y0, z0],
                            "children" : {}
                           }
                    }

        Parameters
        ----------
        bds : dict
            A python dictionary composed with Boundary dicts as
            "k" : boundary_dict.

        See Also
        --------
        :py:meth:`to_dict`
        """

        boundaries = Boundaries()

        roots = [nid for nid, node in bds.items() if node["parent"] in [None, "None"]]

        def add_boundary(nid):
            # Node attributes are required
            for k in Node().__dict__:
                if k not in bds[nid]:
                    raise ValueError(
                        f"Can't build Boundaries object from dictionary. Boundary {nid} has no {k}"
                    )

            n = boundaries._node_type()
            n.id = nid
            n.set_data(**bds[nid])
            boundaries[n.id] = n
            for cid in n.children:
                add_boundary(nid=cid)
            return True

        for rid in roots:
            add_boundary(nid=rid)

        return boundaries

    @staticmethod
    def read(fname):
        """
        Read boundaries from a json file.

        The expected format is a json file containing the essential geometrical and topological
        information. Additionally, if there exist a vtk MultiBlock (with same name but with ext
        .vtm) it is also loaded.


        Parameters
        ----------
        fname : str
            The file name of the boundaries. Should have json extension (.json).

        Returns
        -------
        boundaries : Boundaries
            The Loaded boundaries object.

        See Also
        --------
        save
        """

        fname_, ext = os.path.splitext(fname)
        if ext != ".json":
            error_message(f"Can't read boundaries from {fname}. Only .json files are supported.")
            return None

        bds_dict = read_json(fname)
        boundaries = Boundaries.from_dict(bds_dict=bds_dict)
        mb_name = fname_ + ".vtm"

        if os.path.exists(mb_name):
            mb = pv.read(mb_name).as_polydata_blocks()
            block_names = [mb.get_block_name(i) for i in range(mb.n_blocks)]
            for i, bd in boundaries.items():
                if i in block_names:
                    bd.extract_from_polydata(pdt=mb[i])

        return boundaries

    def translate(self, t):
        """
        Translate the Boundaries object, translating all the Boundary objects, with the translation vector t.

        Parameters
        ----------
        t : np.ndarray (3,)
            The translation vector.

        See Also
        --------
        :py:meth:`Boundary.translate`
        """

        for _, bd in self.items():
            bd.translate(t)

    def scale(self, s):
        """
        Scale the Boundaries object, scaling all the Boundary objects, by a scalar factor s.

        Parameters
        ----------
        s : float
            The scale factor.

        See Also
        --------
        :py:meth:`Boundary.scale`
        """

        for _, bd in self.items():
            bd.scale(s)

    def rotate(self, r):
        """
        Rotate the Boundaries object, by rotating all its Boundary objects, with the provided
        rotation matrix r.

        Parameters
        ----------
        r : np.ndarray (3, 3)
            The rotation matrix.

        See Also
        --------
        :py:meth:`Boundary.rotate`
        """

        for _, bd in self.items():
            bd.rotate(r)
