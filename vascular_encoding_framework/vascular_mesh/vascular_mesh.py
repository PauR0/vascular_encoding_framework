#! /usr/bin/env python3

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

from ..messages import *
from .boundaries import Boundaries, Boundary
from ..utils.geometry import triangulate_cross_section, approximate_cross_section, extract_section
from ..utils.spatial  import compute_ref_from_points, normalize
from ..utils._code    import attribute_setter

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
        if p.is_manifold:
            self.closed = p
        else:
            self.compute_open_boundaries()
    #

    def compute_kdt(self):
        """ Compute the KDTree for the points in the wall mesh """
        computing_message(info="KDTree")
        self.kdt = KDTree(self.points)
        done_message(info="KDTree")
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
            error_message("There is no data to be saved....")
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
        computing_message("mesh triangulation")
        m = super().triangulate(inplace=inplace, **kwargs)
        done_message("mesh triangulation")
        return m
    #

    def compute_normals(self, inplace=True, **kwargs):
        """
        Compute the normals of the mesh. Note that contrarily to pyvista, in
        this library inplace is set to True.
        """

        computing_message("mesh normals")
        m = super().compute_normals(cell_normals=True, point_normals=True, inplace=True, **kwargs)
        done_message("mesh normals")
        return m
    #

    def compute_closed_mesh(self, w=False):

        """
        Method to get a polydata with the boundaries closed. It is also set in the closed
        attribute.

        Arguments:
        -------------

            w : bool, opt
                Default False. Whether to rewritte existing self.closed attribute.

        Returns:
        -----------
            self.closed : pv.PolyData
                The closed mesh.
        """

        if self.closed is None or w:

            if self.is_manifold:
                self.closed = self.copy()

            else:
                if self.boundaries is None:
                    self.compute_open_boundaries()
                polys=[]
                for _, b in self.boundaries.items():
                    p = pv.PolyData(b.points)
                    if hasattr(b, 'faces'):
                        p.faces = b.faces
                    else:
                        p = triangulate_cross_section(p, method='unconnected', n=b.normal)
                    polys.append(p)

                self.closed = pv.PolyData(self.append_polydata(*polys)).clean().triangulate()

        return self.closed
    #

    def compute_open_boundaries(self, overwrite=False):
        """
        Method to compute the open boundary edges and build a Boundaries object with no hierarchy.
        If boundaries attribute is None or overwrite is True, boundaries attribute is set as the
        computed boundaries.

        Arguments:
        -------------

            overwrite : bool, opt
                Default False. Whether to overwrite the boundaries attribute.

        Returns:
        ---------
            boundaries : Boundaries
                The computed boundaries object.
        """

        computing_message("mesh boundaries")
        bnds = self.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)
        bnds = bnds.connectivity()
        boundaries = Boundaries()

        for i in np.unique(bnds['RegionId']):
            ii = str(int(i))

            b = bnds.extract_cells(bnds['RegionId'] == i).extract_surface(pass_pointid=False, pass_cellid=False)
            b = triangulate_cross_section(b)

            bd = Boundary()
            bd.id = ii
            bd.extract_from_polydata(b)

            boundaries[ii] = bd

        if self.boundaries is None or overwrite:
            self.boundaries   = boundaries
            self.n_boundaries = len(self.boundaries)

        done_message("mesh boundaries")
        return boundaries
    #

    def set_boundary_data(self, data):
        """
        This method allows setting new attributes to the boundaries by means of the set_data
        node method. Argument data is expected to be a dictionary of dictionaries with the desired
        new data as follows:

        data = {
                 'id1' : {'center' : [x,y,z], 'normal' :[x1, y1, z1] }
                 'id2' : {'normal' :[x2, y2, z2] }
                 'id3' : {'center' : [x3,y3,z3]}
        }
        """

        self.boundaries.set_data_to_nodes(data=data)
    #

    def plot_boundary_ids(self, print_data=False, edge_color="red", line_width=None):
        """
        If boundaries attribute is not None. This method shows a plot of the highlighted boundaries
        with the id at the center.

        Arguments:
        -------------

            print_data : bool

            edge_color : str

            line_width : int

        """

        if self.boundaries is None:
            error_message(f"can't plot boundary ids, boundaries attribute is {self.boundaries}")
            return

        p = pv.Plotter()
        p.add_mesh(self, color='w')
        p.add_point_labels(np.array([b.center for _, b in self.boundaries.items()]), self.boundaries.enumerate())

        for _, b in self.boundaries.items():
            poly = pv.PolyData()
            if hasattr(b, 'points'):
                poly.points=b.points
            if hasattr(b, 'faces'):
                poly.faces=b.faces
            p.add_mesh(poly, style='wireframe', color=edge_color, line_width=line_width)

        if print_data:
            print(self.boundaries)

        p.show()

        return
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

        computing_message('vascular mesh translation')
        super().translate(t, inplace=True)
        self.closed.translate(t)
        self.boundaries.translate(t)
        done_message('vascular mesh translation')

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

        computing_message('vascular mesh rotation')
        if inverse:
            r = r.inv()
        v, d = normalize(r.as_rotvec()), np.linalg.norm(r.as_rotvec(degrees=True))
        self.rotate_vector(vector=v, angle=d, inplace=True)
        self.closed.rotate_vector(vector=v, angle=d, inplace=True)
        self.boundaries.rotate(r)

        if update_kdt:
            self.compute_kdt()
        done_message('vascular mesh rotation')
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
        computing_message('vascular mesh scaling.')
        super().scale(s, inplace=True)
        self.closed.scale(s, inplace=True)
        self.boundaries.scale(s)
        if update_kdt:
            self.compute_kdt()
        done_message('vascular mesh scaling.')
    #

    @staticmethod
    def from_closed_mesh_and_boundaries(cmesh, boundaries, debug=False):
        """
        Given a closed vascular mesh, and a boundaries object where each boundary has
        a center attribute. This function approximate the boundary cross section of each
        boundary and computes the open vascular mesh.

        Arguments:
        ------------

            vmesh : pv.PolyData
                The vascular mesh.

            boundaries : Boundaries or dict
                The boundaries object already built or the dictionary to built them.
                Note that each boundary (object or dict) must have a center attribute.

            debug : bool, opt
                Default False. Show some plots of the process.

        Returns:
        ----------
            vmesh : VascularMesh
                The VascularMesh object with open boundaries. The passed closed mesh is stored in
                closed_mesh attribute of the VascularMesh.
        """

        if not cmesh.is_all_triangles:
            cmesh = cmesh.triangulate()

        if not isinstance(boundaries, Boundaries):
            boundaries = Boundaries(boundaries)

        cs_bounds = pv.PolyData()
        kdt = KDTree(cmesh.points)
        for _, bound in boundaries.items():
            d = kdt.query(bound.center)[0]
            cs = approximate_cross_section(point=bound.center,
                                           mesh=cmesh,
                                           max_d=d*1.5,
                                           min_perim=2 * np.pi * d * 0.75,
                                           debug=debug)
            bound.extract_from_polydata(cs)
            c = cs.center
            cs.points = (cs.points - c) * 1.1 + c #Scaling from center
            cs_bounds += cs

        col_mesh, _ = cmesh.collision(cs_bounds)
        colls = np.ones(cmesh.n_cells, dtype=bool)
        colls[col_mesh.field_data['ContactCells']] = False
        open_vmesh = col_mesh.extract_cells(colls).extract_largest().extract_surface()

        vmesh            = VascularMesh(p=open_vmesh)
        vmesh.closed     = cmesh
        vmesh.boundaries = boundaries

        if debug:
            p = pv.Plotter()
            p.add_mesh(cmesh, color='w')
            p.add_mesh(vmesh, color='b')
            p.add_mesh(cs_bounds, color='r')
            p.show()

        return vmesh
    #

    @staticmethod
    def from_closed_mesh_and_centerline(cmesh, cl_net, debug=False):
        """
        Given a closed vascular mesh, and a CenterlineNetwork object. This function approximate
        the cross section of each boundary using the tangent of the centerline at the extrema.

        Arguments:
        ------------

            vmesh : pv.PolyData
                The vascular mesh.

            cl_net : CenterlineNetwork
                The centerline network of the vasculare mesh already computed.

            debug : bool, opt
                Default False. Show some plots of the process.

        Returns:
        ----------
            vmesh : VascularMesh
                The VascularMesh object with open boundaries. The passed closed mesh is stored in
                closed_mesh attribute of the VascularMesh.
        """

        if not cmesh.is_all_triangles:
            cmesh = cmesh.triangulate()

        boundaries = Boundaries()

        def scale_from_center(cs, s=1.1):
            scs = cs.copy(deep=True)
            c = scs.center
            scs.points = ((scs.points - c) * s) + c #Scaling from center
            return scs

        def compute_boundary(p, n):
            b = Boundary()
            cs = extract_section(mesh=cmesh, normal=n, origin=p, triangulate=True)
            b.extract_from_polydata(cs)
            return b

        def add_centerline_boundary(cid, root=False):
            cl = cl_net[cid]
            if root:
                inlet = compute_boundary(p=cl(cl.t0), n=cl.get_tangent(cl.t0))
                iid = f"root_{len(boundaries.roots)}"
                if cl.parent not in [None, 'None']:
                    iid = cl.parent
                inlet.set_data(id       = iid,
                               parent   = None)
                boundaries[inlet.id] = inlet

            outlet = compute_boundary(p=cl(cl.t1), n=cl.get_tangent(cl.t1))
            outlet.set_data_from_other_node(cl)
            if root:
                outlet.parent = inlet.id
            boundaries[outlet.id] = outlet

            for chid in cl.children:
                add_centerline_boundary(cid=chid)


        for rid in cl_net.roots:
            add_centerline_boundary(rid, root=True)

        cs_bounds = pv.PolyData()
        for _, b in boundaries.items():
            cs_bounds += scale_from_center(pv.PolyData(b.points, b.faces))

        col_mesh, _ = cmesh.collision(cs_bounds)
        colls = np.ones(cmesh.n_cells, dtype=bool)
        colls[col_mesh.field_data['ContactCells']] = False
        open_vmesh = col_mesh.extract_cells(colls).extract_largest().extract_surface()

        vmesh            = VascularMesh(p=open_vmesh)
        vmesh.closed     = cmesh
        vmesh.boundaries = boundaries

        if debug:
            p = pv.Plotter()
            p.add_mesh(cmesh, color='w', opacity=0.8)
            p.add_mesh(vmesh, color='b')
            p.add_mesh(cs_bounds, color='r')
            p.show()

        return vmesh
    #
#

def load_vascular_mesh(path, suffix, abs_path=False):
    """
    Load a vascular mesh with all the available data at a given
    case path, with the given suffix.

    Parameters
    ----------
    path : string
        The path to the wall mesh.

    suffix : string
        A string indicating a suffix in the mesh name. E.g. suffix="_orig"
        means wall_orig.stl

    abs_path : bool, optional
        If true, the path passed must be the path to the file containing
        the vascular mesh wall. If true, no inlet/outlet information will
        be written. If True, suffix is ignored.
    """
    pass
#
