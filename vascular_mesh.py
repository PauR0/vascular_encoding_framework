#! /usr/bin/env python3

import os

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

import messages as msg
from utils.geometry import triangulate_cross_section
from utils.spatial  import compute_ref_from_points, normalize
from utils._code    import attribute_setter


class Boundaries(dict):
    """
    A class containing the boundaries inheriting structure from python's
    dictionary.
    """

    #No init required since parent init suffice.

    def __setitem__(self, __key, __value):
        if __key in self:
            msg.warning_message(f"id {__key} already exists, overwritting boundary. To update a boundary, use update method.")

        if __value.id is None:
            __value.id = __key
        elif __value.id != __key:
            msg.warning_message(f"id {__key} in Boundaries dict is different from the id in the Boundary object ")

        return super().__setitem__(__key, __value)
    #

    def get_ids(self):
        return list(super().keys())
    #

    def save(self, filename):
        pass
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


class VascularMesh(pv.PolyData):

    """
    The class to contain the triangle mesh representation of a Vascular
    structure with its attributes such as boundary data.
    """

    def __init__(self) -> None:

        #For saving and loading
        self.path   : str  = None
        self.f_type : str  = 'vtk'
        self.binary : bool = False

        self.mesh : pv.PolyData = None
        self.boundaries : dict = None
        self.closed_mesh : pv.PolyData = None

        #To query distances
        self.kdt : KDTree = None

        #Spatial alignment
        self.center : np.ndarray = None
        self.e1 : np.ndarray     = None
        self.e2 : np.ndarray     = None
        self.e3 : np.ndarray     = None
    #

    def compute_kdt(self):
        """ Compute the KDTree for the points in the wall mesh """
        msg.computing_message(task="KDTree")
        self.kdt = KDTree(self.mesh.points)
        msg.done_message(task="KDTree")
    #

    def compute_local_ref(self):
        """
        Compute the object oriented frame by means of a PCA.
        """

        c, e1, e2, e3 = compute_ref_from_points(self.mesh.points)
        self.set_local_ref(c, e1, e2, e3)
        return c, e1, e2, e3
    #

    def set_local_ref(self, center, e1, e2, e3):
        """
        Set the objet oriented frame.
        """

        self.center = center
        self.e1     = e1
        self.e2     = e2
        self.e3     = e3
    #

    def set_data(self, mesh=None, kdt=None, boundaries=None,
                       local_ref=None, triangulate_wall=False):
        """
        Set the data of the vascular mesh.

        Set the polydata and geometric information of the aorta mesh.

        Parameters
        ----------
        mesh  : vtkPolyData, optional
            Polydata representing the wall of the vascular mesh.
        boundaries : dict, optional
            The dictionary containing the boundary information.
        triangulate : bool, optional
            Whether to triangulate the wall polydata.
        local_ref : tuple of 3 floats, optional
            Local reference point for the aorta mesh.
        """

        if mesh:
            self.mesh = mesh

            if triangulate_wall:
                self.triangulate_mesh()

            if kdt:
                self.kdt = kdt
            else:
                self.compute_kdt()

            if not local_ref:
                self.compute_local_ref()

            self.compute_wall_normals()

        if boundaries:
            self.boundaries = boundaries

        if local_ref:
            self.set_local_ref(local_ref[0],local_ref[1],local_ref[2], local_ref[3])
    #

    def save(self, suffix=""):
        """
        Save the vascular mesh.

        TODO: Save inlet/outlet data
        """

        if self.mesh is not None:
            fname = os.path.join(self.path, 'meshes', f'mesh{suffix}.{self.f_type}')
            self.mesh.save(filename=fname, binary=self.binary)

        if self.boundaries is None and self.mesh is None:
            print("There is no data to be saved....")
    #

    def load(self, path=None, suffix="", abs_path=False):
        """
        Load a vascular mesh with all the available data at a given
        case path, with the given suffix.

        Parameters
        ----------
        path : string
            The path to the wall mesh. TODO: Work about boundary format (inlet/outlet)

        suffix : string
            A string indicating a suffix in the mesh name. E.g. suffix="_orig"
            means wall_orig.stl

        abs_path : bool, optional
            If true, the path passed must be the path to the file containing
            the vascular mesh wall. If true, no inlet/outlet information will
            be written. If True, suffix is ignored.

        """

        wall_fname=None
        if abs_path:
            wall_fname = path

        elif os.path.isdir(path):
            self.path = path
            wall_fname = os.path.join(self.path, 'meshes', f'mesh{suffix}.{self.f_type}')
            #inlets_fname  = os.path.join(self.path, 'meshes', f'inlets{suffix}.json')
            #outlets_fname = os.path.join(self.path, 'meshes', f'outlets{suffix}.json')

        self.mesh = pv.read(wall_fname)
        self.triangulate_mesh()
        self.compute_kdt()
        self.compute_local_ref()
        self.compute_wall_normals()

        #TODO: self.inlets  !!!!!!
        #TODO: self.outlets !!!!!!

        self.compute_bounds()

        return True
    #

    def save_boundaries(self):
        """
        #TODO
        Saves the boundaries geometric information
        """

        if self.boundaries is None:
            msg.error_message(info=f"Can't save VascularMesh boundaries. The boundaries attributes is {self.boundaries}")
    #

    def compute_bounds(self):
        """
        Returns the bounding box of the mesh.
        The form is: (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        # bounding box
        self.bounds = self.mesh.bounds
        return self.bounds
    #

    def triangulate_mesh(self):
        """
        Triangulate the wall mesh.
        """
        msg.computing_message("mesh triangulation")
        self.mesh.triangulate(inplace=True)
        msg.done_message("mesh triangulation")
        #
    #

    def compute_wall_normals(self, **kwargs):
        """
        Compute the normals of the mesh. Following pyvista's default
        where normals point "outwards".

        """

        msg.computing_message("mesh normals")
        self.mesh.compute_normals(cell_normals=True, point_normals=True, inplace=True, **kwargs)
        msg.done_message("mesh normals")
    #

    def compute_closed_mesh(self):

        """
        Method to get a polydata with the bounds closed.

        Arguments:
        -----------
        """


        if self.boundaries is not None:
            #TODO:Implement the alternative with more info available.
            pass

        else:
            self.closed_mesh = self.mesh.copy(deep=True)
            bnds = self.mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)
            bnds = bnds.connectivity()
            meshes = []
            for i in np.unique(bnds['RegionId']):
                b = bnds.extract_cells(bnds['RegionId'] == i).extract_surface(pass_pointid=False, pass_cellid=False)
                meshes.append(triangulate_cross_section(b))
            self.closed_mesh = self.mesh.append_polydata(*meshes, inplace=False)
            self.closed_mesh.clean(inplace=True)
    #

    def compute_inlet_normal(self, inlet_pca):
        """
        This method computes the inlet normal fitting a plane to a set of points
        then, for ensuring the normal direction it computes a auxiliar centroid
        which is assumed to be inside the aorta.
        """
        inlet_normal = normalize(inlet_pca.components_[2])
        aux_radius,_ = self.kdt.query(inlet_pca.mean_)
        aux_ids = self.kdt.query_ball_point(inlet_pca.mean_, 4*aux_radius)
        aux_ball = self.wall_points[aux_ids]
        aux_centroid = np.mean(aux_ball, axis = 0)
        right_dir = normalize(aux_centroid - inlet_pca.mean_)
        if inlet_normal.dot(right_dir) < 0:
            inlet_normal *=-1

        return normalize(inlet_normal)
    #

    def compute_outlet_normal(self, outlet_pca):
        """
        This method computes the outlet normal fitting a plane to a set of points
        and, then, ensures the normal direction consistency
        """
        outlet_normal = normalize(outlet_pca.components_[2])

        # the normal must be pointing outwards
        if np.dot(outlet_normal, self.center - outlet_pca.mean_ ) > 0:
            outlet_normal *= -1

        return outlet_normal
    #

    def compute_inlet_geometry(self, normal = None,center = None):
        """ Compute the inlet geometry from the mesh """

        print("Computing inlet geometry...")
        inlet_pca = PCA(n_components=3)
        inlet_pca.fit(self.inlet_points)

        if normal is not None:
            self.inlet_normal = normal
        else:
            self.inlet_normal = self.compute_inlet_normal(inlet_pca)

        v1 = np.cross(self.inlet_normal, self.e3)
        if v1.dot(inlet_pca.mean_- self.center) > 0:
            v1*=-1

        self.inlet_v1 = normalize(v1)
        v2 = np.cross(self.inlet_normal, self.inlet_v1)
        self.inlet_v2 = normalize(v2)

        c, _, _, a, b, phi, err = fit_ellipse3d(self.inlet_points.T,
                                                inlet_pca.mean_,
                                                self.inlet_normal,
                                                self.inlet_v1,
                                                self.inlet_v2 )
        self.inlet_ellipse_center = c
        if center is not None:
            self.inlet_center = center
        else:
            self.inlet_center = c
        #
        self.inlet_a = a
        self.inlet_b = b
        self.inlet_phi = phi
        self.inlet_err = err

        print("\t\t\t\t ...........done")
        print("   Valve center: {}".format(self.inlet_center))
        print("   Valve normal direction: {}".format(self.inlet_normal))
        print("   v1={}".format(self.inlet_v1) + "\n   v2={}".format(self.inlet_v2))
    #

    def compute_outlet_geometry(self, normal = None,center = None):
        """ Compute the oulet geometry from the mesh """

        print("Computing outlet geometry...")
        outlet_pca = PCA(n_components=3)
        outlet_pca.fit(self.outlet_points)

        if normal is not None:
            self.outlet_normal = normal
        else:
            self.outlet_normal = self.compute_outlet_normal(outlet_pca)

        v1 = np.cross(self.outlet_normal, self.e3)
        if v1.dot(outlet_pca.mean_- self.center) > 0:
            v1*=-1
        self.outlet_v1 = normalize(v1)

        v2 = np.cross(self.outlet_normal, self.outlet_v1)
        self.outlet_v2 = normalize(v2)

        c, _, _, a, b, phi, err = fit_ellipse3d(self.outlet_points.T,
                                                outlet_pca.mean_,
                                                self.outlet_normal,
                                                self.outlet_v1,
                                                self.outlet_v2)

        self.outlet_ellipse_center = c
        if center is not None:
            self.outlet_center = center
        else:
            self.outlet_center = c
        #
        self.outlet_a = a
        self.outlet_b = b
        self.outlet_phi = phi
        self.outlet_err = err

        print("\t\t\t\t ...........done")
        print("   Outlet center: {}".format(self.outlet_center))
        print("   Outlet normal direction: {}".format(self.outlet_normal))
        print("   v1={}".format(self.outlet_v1) + "\n   v2={}".format(self.outlet_v2))
    #

    def translate(self, t, update_kdt=True, **kwargs):
        """
        Apply a translation to the mesh and boundaries.

        Arguments:
        ------------
            t : vector-like, (3,)
                The translation vector.

            update_kdt : bool, optional.
                Default True. Whether to update the kdt for query distances on
                mesh points.

            **kwargs : any
                Arguments to be passed to pyvista's translate method.
        """

        msg.computing_message('vascular mesh translation')
        self.mesh.translate(t)
        self.center = np.array(self.mesh.center)
        #TODO: Apply translation to boundaries.
        #TODO: Apply translation to boundaries.
        msg.done_message('vascular mesh translation')

        if update_kdt:
            self.compute_kdt()
    #

    def rotate(self, r, inverse=False, update_kdt=True, **kwargs):
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

            **kwargs : any
                Arguments to be passed to pyvista's scale method.
        """

        msg.computing_message('vascular mesh rotation')
        if inverse:
            r = r.inv()
        self.mesh.rotate_vector(normalize(r.as_rotvec()), np.linalg.norm(r.as_rotvec(degrees=True)), **kwargs)
        self.center = np.array(self.mesh.center)
        #TODO: Apply rotation to boundaries.
        #TODO: Apply rotation to boundaries.

        if update_kdt:
            self.compute_kdt()
        msg.done_message('vascular mesh rotation')
    #

    def scale(self, s, update_kdt=True, **kwargs):
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

            **kwargs : any
                Arguments to be passed to pyvista's translate method.
        """
        msg.computing_message('vascular mesh scaling.')
        self.mesh.scale(s, inplace=True, **kwargs)
        self.center = np.array(self.mesh.center)
        #TODO: Apply scale to boundaries.
        #TODO: Apply scale to boundaries.
        if update_kdt:
            self.compute_kdt()
        msg.done_message('vascular mesh scaling.')
    #

#
