import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

from .._base import SpatialObject, attribute_setter
from ..utils.geometry import approximate_cross_section, extract_section, triangulate_cross_section
from ..utils.spatial import compose_transformation_matrix, compute_ref_from_points
from .boundaries import Boundaries, Boundary


class VascularMesh(pv.PolyData, SpatialObject):
    """
    The class to contain the triangle mesh representation of a Vascular
    structure with its attributes such as boundary data. The mesh is expected
    to be open only at the inlet/outlet boundaries. This is a child class
    of pyvista (vtk) PolyData. Note that in contrast with pyvista where the
    standard is that __inplace__ argument is generally False, here is otherwise.
    Furthermore, it is usually not in the method's signature.
    """

    def __init__(self, p: pv.PolyData = None, compute_boundaries=True) -> None:
        """VascularMesh constructor."""

        self.boundaries: Boundaries = None
        self.n_boundaries: int = None
        self.closed: pv.PolyData = None

        # To query distances
        self.kdt: KDTree = None

        # Spatial alignment
        self.mass_center: np.ndarray = None
        self.e1: np.ndarray = None
        self.e2: np.ndarray = None
        self.e3: np.ndarray = None

        super().__init__(p)
        if isinstance(p, pv.PolyData):
            self.triangulate()
            self.compute_kdt()
            self.compute_local_ref()
            self.compute_normals()

        if p.is_manifold:
            self.closed = p

        if compute_boundaries:
            self.compute_open_boundaries()

    def compute_kdt(self):
        """Compute the KDTree for the points in the wall mesh."""
        self.kdt = KDTree(self.points)

    def compute_local_ref(self):
        """Compute the object oriented frame by means of a PCA."""

        c, e1, e2, e3 = compute_ref_from_points(self.points)
        self.set_local_ref(c, e1, e2, e3)
        return c, e1, e2, e3

    def set_local_ref(self, center, e1, e2, e3):
        """Set the objet oriented frame."""

        self.mass_center = center
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3

    def set_data(self, **kwargs):
        """
        Set the data of the vascular mesh. Useful to set a bunch at once.

        Parameters
        ----------
        **kwargs : Any
            keyboard arguments to be set as attributes.

        """
        attribute_setter(self, **kwargs)

    def save(self, fname, binary=True, boundaries_fname=None, **kwargs):
        """
        Save the vascular mesh. kwargs are passed to PolyData.save method.

        Parameters
        ----------
        fname : str
            The name of the file to store the polydata.
        binary : bool, opt
            Default is True. Whether to save in binary from.
        boundaries_fname : str, opt.
            Default is None. If passed boundaries are saved with at the given path.

        """

        if self.n_points:
            super().save(filename=fname, binary=binary, **kwargs)

        if self.boundaries is not None and boundaries_fname is not None:
            self.boundaries.save(boundaries_fname)

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
        comp_bounds = True if boundaries_fname is None else False

        vmesh = VascularMesh(p=p, compute_boundaries=comp_bounds)
        if boundaries_fname is not None:
            vmesh.boundaries = Boundaries.read(boundaries_fname)

        return vmesh

    def triangulate(self, inplace=True, **kwargs):
        """
        Triangulate the mesh. This is better performed after instantiation to
        prevent possible crashes with other methods and modules. Non triangular
        meshes are not supported in this library. Although pyvista does support
        them.
        """
        m = super().triangulate(inplace=inplace, **kwargs)
        return m

    def compute_normals(self, inplace=True, **kwargs):
        """
        Compute the normals of the mesh. Note that contrarily to pyvista, in
        this library inplace is set to True.
        """

        m = super().compute_normals(inplace=inplace, **kwargs)
        return m

    def compute_closed_mesh(self, overwrite=False):
        """
        Get a polydata with the boundaries closed. It is also set in the closed
        attribute.

        Parameters
        ----------
        w : bool, opt
            Default False. Whether to rewrite existing self.closed attribute.

        Returns
        -------
        self.closed : pv.PolyData
            The closed mesh.
        """

        if self.closed is None or overwrite:
            if self.is_manifold:
                self.closed = self.copy()

            else:
                if self.boundaries is None:
                    self.compute_open_boundaries()
                polys = []
                for _, b in self.boundaries.items():
                    p = pv.PolyData(b.points)
                    if hasattr(b, "faces"):
                        p.faces = b.faces
                    else:
                        p = triangulate_cross_section(p, method="unconnected", n=b.normal)
                    polys.append(p)

                self.closed = pv.PolyData(self.append_polydata(*polys)).clean().triangulate()

        return self.closed

    def compute_open_boundaries(self, overwrite=False):
        """
        Compute the open boundary edges and build a Boundaries object with no hierarchy. If
        boundaries attribute is None or overwrite is True, boundaries attribute is set as the
        computed boundaries.

        Parameters
        ----------
        overwrite : bool, opt
            Default False. Whether to overwrite the boundaries attribute.

        Returns
        -------
        boundaries : Boundaries
            The computed boundaries object.
        """

        bnds = self.extract_feature_edges(
            boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False
        )
        bnds = bnds.connectivity(extraction_mode="all", label_regions=True)
        boundaries = Boundaries()

        rid = bnds.get_array("RegionId", preference="point")
        for i in np.unique(rid):
            ii = f"B{i:d}"

            b = bnds.extract_points(rid == i).extract_surface(pass_pointid=False, pass_cellid=False)
            b = triangulate_cross_section(b)

            bd = Boundary()
            bd.id = ii
            bd.extract_from_polydata(b)

            boundaries[ii] = bd

        if self.boundaries is None or overwrite:
            self.boundaries = boundaries
            self.n_boundaries = len(self.boundaries)

        return boundaries

    def set_boundary_data(self, data):
        """
        Set new attributes for each boundary.

        Argument data is expected to be a dictionary of dictionaries with the desired
        new data as follows:

        data = {
                 'id1' : {'center' : [x,y,z], 'normal' :[x1, y1, z1] }
                 'id2' : {'normal' :[x2, y2, z2] }
                 'id3' : {'center' : [x3,y3,z3]}
        }

        Parameters
        ----------
        data : dict
            The data to be added to the boundaries.
        """

        self.boundaries.set_data_to_nodes(data=data)

    def plot_boundary_ids(self, print_data=False, edge_color="red", line_width=None):
        """
        If boundaries attribute is not None. This method shows a plot of the highlighted boundaries
        with the id at the center.

        Parameters
        ----------
        print_data : bool, optional
            Default False. Whether to print boundary data in the terminal.
        edge_color : str, optional
            Default red. A pyvista-compatible color.
        line_width : int
            Default None. Defaulting to pyvista's default.

        """

        if self.boundaries is None:
            raise AttributeError("Can't plot boundary ids, boundaries attribute is None")

        p = pv.Plotter()
        p.add_mesh(self, color="w", opacity=0.9)
        p.add_point_labels(
            np.array([b.center for _, b in self.boundaries.items()]), self.boundaries.enumerate()
        )

        for _, b in self.boundaries.items():
            poly = pv.PolyData()
            if hasattr(b, "points"):
                poly.points = b.points
            if hasattr(b, "faces"):
                poly.faces = b.faces
            p.add_mesh(poly, style="wireframe", color=edge_color, line_width=line_width)

        if print_data:
            print(self.boundaries)

        p.add_axes()
        p.show()

        return

    def translate(self, t, update_kdt=True):
        """
        Apply a translation to the mesh and boundaries.

        Parameters
        ----------
        t : vector-like, (3,)
            The translation vector.
        update_kdt : bool, optional.
            Default True. Whether to update the kdt for query distances on
            mesh points.

        See Also
        --------
        :py:meth:`Boundaries.translate`

        """

        super().translate(t, inplace=True)

        if self.closed is not None:
            self.closed.translate(t)

        if self.boundaries is not None:
            self.boundaries.translate(t)

        if update_kdt:
            self.compute_kdt()

    def rotate(self, r, update_kdt=True, transform_all_input_vectors=False):
        """
        Apply a rotation to the mesh and boundaries.

        Parameters
        ----------
        r: np.ndarray (3,3)
            A rotation matrix.
        update_kdt : bool, optional.
            Default True. Whether to update the kdt for query distances on mesh points.
        """

        R = compose_transformation_matrix(r=r)
        self.transform(
            trans=R, transform_all_input_vectors=transform_all_input_vectors, inplace=True
        )

        if self.closed is not None:
            self.closed.transform(
                trans=R, transform_all_input_vectors=transform_all_input_vectors, inplace=True
            )

        if self.boundaries is not None:
            self.boundaries.rotate(r)

        if update_kdt:
            self.compute_kdt()

    def scale(self, s, update_kdt=True):
        """
        Scale vascular mesh. kwargs can be passed to pyvista scaling method.

        Parameters
        ----------
        s : float
            The scaling factor.
        update_kdt : bool, optional.
            Default True. Whether to update the kdt for query distances on mesh points
        """

        super().scale(s, inplace=True)

        if self.closed is not None:
            self.closed.scale(s, inplace=True)

        if self.boundaries is not None:
            self.boundaries.scale(s)

        if update_kdt:
            self.compute_kdt()

    @staticmethod
    def from_closed_mesh_and_boundaries(cmesh, boundaries, debug=False):
        """
        Given a closed vascular mesh, and a boundaries object where each boundary has
        a center attribute. This function approximate the boundary cross section of each
        boundary and computes the open vascular mesh.

        Parameters
        ----------
        vmesh : pv.PolyData
            The vascular mesh.
        boundaries : Boundaries or dict
            The boundaries object already built or the dictionary to built them.
            Note that each boundary (object or dict) must have a center attribute.
        debug : bool, opt
            Default False. Show some plots of the process.

        Returns
        -------
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
            cs = approximate_cross_section(
                point=bound.center,
                mesh=cmesh,
                max_d=d * 1.5,
                min_perim=2 * np.pi * d * 0.75,
                debug=debug,
            )
            bound.extract_from_polydata(cs)
            c = cs.center
            cs.points = (cs.points - c) * 1.1 + c  # Scaling from center
            cs_bounds += cs

        col_mesh, _ = cmesh.collision(cs_bounds)
        colls = np.ones(cmesh.n_cells, dtype=bool)
        colls[col_mesh.field_data["ContactCells"]] = False
        open_vmesh = col_mesh.extract_cells(colls).extract_largest().extract_surface()

        vmesh = VascularMesh(p=open_vmesh, compute_boundaries=False)
        vmesh.closed = cmesh
        vmesh.boundaries = boundaries

        if debug:
            p = pv.Plotter()
            p.add_mesh(cmesh, color="w")
            p.add_mesh(vmesh, color="b")
            p.add_mesh(cs_bounds, color="r")
            p.show()

        return vmesh

    @staticmethod
    def from_closed_mesh_and_centerline(cmesh, cl_tree, debug=False):
        """
        Given a closed vascular mesh, and a CenterlineTree object. This function approximate the
        cross section of each boundary using the tangent of the centerline at the extrema.

        Parameters
        ----------
        vmesh : pv.PolyData
            The vascular mesh.
        cl_tree : CenterlineTree
            The centerline tree of the vascular mesh already computed.
        debug : bool, opt
            Default False. Show some plots of the process.

        Returns
        -------
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
            scs.points = ((scs.points - c) * s) + c  # Scaling from center
            return scs

        def compute_boundary(p, n):
            b = Boundary()
            cs = extract_section(mesh=cmesh, normal=n, origin=p, triangulate=True)
            b.extract_from_polydata(cs)
            return b

        def add_centerline_boundary(cid, root=False):
            cl = cl_tree[cid]
            if root:
                inlet = compute_boundary(p=cl(cl.t0), n=cl.get_tangent(cl.t0))
                iid = f"root_{len(boundaries.roots)}"
                if cl.parent not in [None, "None"]:
                    iid = cl.parent
                inlet.set_data(id=iid, parent=None)
                boundaries[inlet.id] = inlet

            outlet = compute_boundary(p=cl(cl.t1), n=cl.get_tangent(cl.t1))
            outlet.set_data_from_other_node(cl)
            if root:
                outlet.parent = inlet.id
            boundaries[outlet.id] = outlet

            for chid in cl.children:
                add_centerline_boundary(cid=chid)

        for rid in cl_tree.roots:
            add_centerline_boundary(rid, root=True)

        cs_bounds = pv.PolyData()
        for _, b in boundaries.items():
            cs_bounds += scale_from_center(pv.PolyData(b.points, b.faces))

        col_mesh, _ = cmesh.collision(cs_bounds)
        colls = np.ones(cmesh.n_cells, dtype=bool)
        colls[col_mesh.field_data["ContactCells"]] = False
        open_vmesh = col_mesh.extract_cells(colls).extract_largest().extract_surface()

        vmesh = VascularMesh(p=open_vmesh)
        vmesh.closed = cmesh
        vmesh.boundaries = boundaries

        if debug:
            p = pv.Plotter()
            p.add_mesh(cmesh, color="w", opacity=0.8)
            p.add_mesh(vmesh, color="b")
            p.add_mesh(cs_bounds, color="r")
            p.show()

        return vmesh
