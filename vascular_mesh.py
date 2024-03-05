#! /usr/bin/env python3

import os

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

from boundaries import Boundaries, Boundary
import messages as msg
from utils.geometry import triangulate_cross_section
from utils.spatial  import compute_ref_from_points, normalize
from utils._code    import attribute_setter

class VascularMesh(pv.PolyData):

    """
    The class to contain the triangle mesh representation of a Vascular
    structure with its attributes such as boundary data. The mesh is expected
    to be open only at the inlet/outlet boundaries. This is a child class
    of pyvista (vtk) PolyData. Note that in contrast with pyvista where the
    standard is that __inplace__ argument is generally False, here is otherwise.
    Furthermore, it is usually not in the method's signature.
    """

    def __init__(self, p:pv.PolyData=None) -> None:


        self.boundaries   : Boundaries = None
        self.n_boundaries : int = None
        self.closed       : pv.PolyData = None

        #To query distances
        self.kdt : KDTree = None

        #Spatial alignment
        self.mass_center : np.ndarray = None
        self.e1 : np.ndarray     = None
        self.e2 : np.ndarray     = None
        self.e3 : np.ndarray     = None

        super().__init__(p)
        if isinstance(p, pv.PolyData):
            self.triangulate()
            self.compute_kdt()
            self.compute_local_ref()
            self.compute_normals()
    #

    def compute_kdt(self):
        """ Compute the KDTree for the points in the wall mesh """
        msg.computing_message(info="KDTree")
        self.kdt = KDTree(self.points)
        msg.done_message(info="KDTree")
    #

    def compute_local_ref(self):
        """
        Compute the object oriented frame by means of a PCA.
        """

        c, e1, e2, e3 = compute_ref_from_points(self.points)
        self.set_local_ref(c, e1, e2, e3)
        return c, e1, e2, e3
    #

    def set_local_ref(self, center, e1, e2, e3):
        """
        Set the objet oriented frame.
        """

        self.mass_center = center
        self.e1          = e1
        self.e2          = e2
        self.e3          = e3
    #

    def set_data(self, **kwargs):
        """
        Set the data of the vascular mesh. Usefull to set a bunch at once.

        Parameters
        ----------
            **kwargs : Any
                keyboard arguments to be set as attributes.

        """
        attribute_setter(self, **kwargs)
    #

    def save(self, filename, binary=True, boundaries_fname=None, **kwargs):
        """
        Save the vascular mesh. kwargs are passed to PolyData.save method.

        Arguments:
        ------------
            filename : str
                The name of the file to store the polydata.

            binary : bool, opt
                Default is True. Whether to save in binary from.

            boundaries_fname : str, opt.
                Default is None. If passed boundaries are saved with at the given path.

        """

        if self.n_points is not None:
            super().save(filename=filename, binary=binary, **kwargs)

        if self.boundaries is not None and boundaries_fname is not None:
            self.boundaries.save(boundaries_fname)

        if self.boundaries is None and self.mesh is None:
            msg.error_message("There is no data to be saved....")
    #

    @staticmethod
    def read(filename=None, boundaries_fname=None):
        """
        Load a vascular mesh with all the available data at a given
        case path, with the given suffix.

        Parameters
        ----------
        filename : string
            The path to the wall mesh.

        boundaries_fname : string, opt
            Default is None. If passed boundaries are loaded from given path.

        """

        p = pv.read(filename)
        vmesh = VascularMesh(p=p)
        if boundaries_fname is not None:
            vmesh.boundaries = Boundaries.read(boundaries_fname)

        return vmesh
    #

    def triangulate(self,inplace=True, **kwargs):
        """
        Triangulate the mesh. This is better performed after instantiation to
        prevent possible crashes with other methods and modules. Non triangular
        meshes are not supported in this library. Although pyvista does support
        them.
        """
        msg.computing_message("mesh triangulation")
        m = super().triangulate(inplace=inplace, **kwargs)
        msg.done_message("mesh triangulation")
        return m
    #

    def compute_normals(self, inplace=True, **kwargs):
        """
        Compute the normals of the mesh. Note that contrarily to pyvista, in
        this library inplace is set to True.
        """

        msg.computing_message("mesh normals")
        m = super().compute_normals(cell_normals=True, point_normals=True, inplace=True, **kwargs)
        msg.done_message("mesh normals")
        return m
    #

    def compute_closed_mesh(self):

        """
        Method to get a polydata with the boundaries closed. It is also set in the closed
        attribute.

        Returns:
        -----------
            self.closed : pv.PolyData
                The closed mesh.
        """


        meshes = []
        if self.boundaries is None:
            self.compute_boundaries()

        for _, b in self.boundaries.items():
            p = pv.PolyData(b.points)
            method = 'unconnected'
            if b.faces.size > 0:
                p.faces = b.faces
                p = p.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=True)
                method = 'connected'
            p = triangulate_cross_section(p, method=method, n=b.normal)
            meshes.append(p)

        self.closed = pv.PolyData(self.append_polydata(*meshes, inplace=False))
        self.closed.clean(inplace=True)

        return self.closed
    #

    def compute_boundaries(self, hierarchy=None, by_center=True, by_id=False):
        """
        Method to build the boundary tree based on the boundary edges
        on the mesh attribute. Attribute can be set through a hierarchy dict,
        however, if none is passed this method leaves the hierarcy undefined.

        Example of a valid hierarcy using by_centers=True:
        Ex. hierarchy = {"1" : {"parent"     : None,
                            "center"   : [ x1, y1, z1],
                            "children" : {"2"}
                           }
                     "2" : {"parent"     : '1',
                            "center"   : [ x2, y2, z2],
                            "children" : {"0"}
                           }
                     "0" : {"parent"     : '2',
                            "center"   : [ x0, y0, z0],
                            "children" : {}
                           }
                    }

        Arguments:
        ------------

            hierarchy : dict, opt
                Default None. A hierarchical dict.

            by_center : bool, opt
                Default True. If True, the hierarchy dictionary must be passed.
                In addition, each node-dict must have a center attribute with
                the coordinates of the center. The closest centroid of boundaris
                will be used to match each computed boundary.

            by_id : bool, opt
                Default False. If True, the hierarchy dictionary must be passed.
                In contrast with by_center, the id must be the id inferred by
                pyvista while computing boundary_edges. This should only be used
                if the ids have been already inspected and the ids are known
                beforehand.

        """

        msg.computing_message("mesh boundaries")
        if hierarchy is None:
            msg.warning_message("No hierarchy defined. No relation will be assumed among boundaries.")

        bnds = self.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)
        bnds = bnds.connectivity()

        if hierarchy is not None:
            self.boundaries = Boundaries(hierarchy=hierarchy)
            msg.info_message(f"Assuming the following hierarchy: \n{self.boundaries}")
            bids = self.boundaries.enumerate()
            centers = np.array([self.boundaries[bid].center for bid in bids])

        else:
            self.boundaries = Boundaries()

        for i in np.unique(bnds['RegionId']):
            ii = str(i)
            b = bnds.extract_cells(bnds['RegionId'] == i).extract_surface(pass_pointid=False, pass_cellid=False)
            b = triangulate_cross_section(b)

            if hierarchy is None:
                bd = Boundary()
                bid = ii

            elif by_id:
                bd = Boundary(self.boundaries[ii])
                bid = ii

            elif by_center:
                aux = np.argmin(np.linalg.norm(centers - np.array(b.center), axis=1))
                bid = bids[aux]
                bd = Boundary(self.boundaries[bid])

            bd.extract_from_polydata(b)
            self.boundaries[ii] = bd

        self.n_boundaries = len(self.boundaries)
        msg.done_message("mesh boundaries")
    #

    def translate(self, t, update_kdt=True):
        """
        Apply a translation to the mesh and boundaries.

        Arguments:
        ------------
            t : vector-like, (3,)
                The translation vector.

            update_kdt : bool, optional.
                Default True. Whether to update the kdt for query distances on
                mesh points.
        """

        msg.computing_message('vascular mesh translation')
        super().translate(t, inplace=True)
        self.closed.translate(t)
        self.boundaries.translate(t)
        msg.done_message('vascular mesh translation')

        if update_kdt:
            self.compute_kdt()
    #

    def rotate(self, r, inverse=False, update_kdt=True):
        """
        Apply a rotation to the mesh and boundaries.

        Input:
        --------
            r: scipy.spatial.Rotation object
                A scipy Rotation object, containing the desired rotation.

            inverse : bool, optional
                Whether to apply the inverse of the passed rotation.

            update_kdt : bool, optional.
                Default True. Whether to update the kdt for query distances on
                mesh points
        """

        msg.computing_message('vascular mesh rotation')
        if inverse:
            r = r.inv()
        v, d = normalize(r.as_rotvec()), np.linalg.norm(r.as_rotvec(degrees=True))
        self.rotate_vector(vector=v, angle=d, inplace=True)
        self.closed.rotate_vector(vector=v, angle=d, inplace=True)
        self.boundaries.rotate(r)

        if update_kdt:
            self.compute_kdt()
        msg.done_message('vascular mesh rotation')
    #

    def scale(self, s, update_kdt=True):
        """
        Function to scale vascular mesh. kwargs can be passed to pyvista
        scaling method.

        Arguments:
        ------------

            s : float
                The scaling factor.

            update_kdt : bool, optional.
                Default True. Whether to update the kdt for query distances on
                mesh points
        """
        msg.computing_message('vascular mesh scaling.')
        super().scale(s, inplace=True)
        self.closed.scale(s, inplace=True)
        self.boundaries.scale(s)
        if update_kdt:
            self.compute_kdt()
        msg.done_message('vascular mesh scaling.')
    #
#


#
