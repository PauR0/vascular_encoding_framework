from abc import ABC, abstractmethod

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

from .._base import attribute_checker, attribute_setter
from ..messages import computing_message, done_message
from ..vascular_mesh import VascularMesh


class CenterlineDomainExtractor(ABC):
    """
    Base class for centerline domain extractors. This method has the minimum
    requirements for centerline domain extractors. Note that since it inherits
    Abstract Base Class it forces run() method exist on subclasses.
    """

    def __init__(self):
        self.mesh: pv.PolyData = None
        self.output_domain: pv.PolyData = None

    def set_mesh(self, m: pv.PolyData):
        """Set the boundary surface mesh."""
        self.mesh = m

    def set_parameters(self, **kwargs):
        """Set parameters as attributes of the objects."""
        cl = self.__class__()
        params = {k: v for k, v in kwargs.items() if k in cl.__dict__}
        attribute_setter(self, **params)

    def get_output(self):
        """Get the extracted domain."""
        return self.output_domain

    @abstractmethod
    def run(self):
        """
        Run the implemented algorithm.

        Must be overwritten by subclasses.
        """
        ...


class Seekers(CenterlineDomainExtractor):
    """
    Centerline domain extractor based on the seekers approach.

    According to the grass fire equation the surface inward normal is a good estimation for finding
    the centerline. Experimentally we found that the midpoint between any given point in the wall
    and its projection by ray trace using the inward normal is a robust, and fast approximation of
    the centerline locus. The benefit of this method against others is that it does not require a
    closed surface, as long as the normals are pointing inward.

    Caveats: optimality of the centerline is not warranted.
    """

    def __init__(self):
        super().__init__()

        self.seekers: pv.PolyData = None

        # Parameters
        self.reduction_rate: float = 0.75
        self.smooth_iters: int = 100
        self.eps: float = 1e-3
        self.check_dirs: bool = (True,)
        self.check_inside: bool = (False,)
        self.multi_ray_trace: bool = False

        self.debug: bool = False

    def set_mesh(self, m, update=True):
        """Set the boundary mesh."""
        super().set_mesh(m)

        if update:
            self.compute_seekers_initial_positions()

    def compute_seekers_initial_positions(self) -> pv.PolyData:
        """
        Compute the initial position of the seekers by computing a decimation of the original mesh.
        The mesh is then smoothed to enhance the normal direction. This method requires mesh and
        target_reduction attributes to be already set.

        Returns
        -------
            self.seekers : pv.PolyData
                The seekers initial position.
        """

        attribute_checker(
            obj=self,
            atts=["mesh", "reduction_rate"],
            info="Can't compute initial seekers position.",
        )

        self.seekers = (
            self.mesh.decimate(target_reduction=self.reduction_rate, attribute_error=False)
            .smooth(n_iter=self.smooth_iters)
            .compute_normals(cell_normals=False, point_normals=True)
        )
        return self.seekers

    def flip_seekers_directions(self):
        """Flip the seekers directions to point inwards."""

        computing_message("flipped normals")
        self.seekers = self.seekers.compute_normals(flip_normals=True)
        if self.debug:
            p = pv.Plotter()
            p.add_mesh(self.mesh, opacity=0.5)
            p.add_mesh(self.seekers, render_points_as_spheres=True, style="points")
            p.add_mesh(self.seekers.glyph(orient="Normals", scale="Normals"))
            p.show()

        done_message("flipped normals")

    def check_seekers_direction(self, n_tests=50):
        """
        Ensure the seekers directions point inwards.

        This method requires mesh to be a closed surface to work properly. It works by performing a
        step of the seekers algorithm with a reduced number of points (controlled by n_tests). Then
        if the number of points falling outside the mesh surface is greater than those inside, the
        normals are flipped.
        """

        eps = self.mesh.length * self.eps
        ids = np.random.randint(low=0, high=self.seekers.n_points - 1, size=n_tests)
        dirs = self.seekers.get_array("Normals", preference="point")[ids]
        # The initial position is moved an epsilon inwards to prevent capturing
        # the initial intersection.
        start = self.seekers.points[ids] + dirs * eps
        stop = self.seekers.points[ids] + dirs * self.mesh.length
        intersection = []
        for stt, stp in zip(start, stop):
            p = np.array([self.seekers.ray_trace(origin=stt, end_point=stp, first_point=True)[0]])
            if p.size == 0:
                p = stp
            intersection.append(p.ravel())
        intersection = np.vstack(intersection)

        pts = pv.PolyData((start + intersection) / 2)
        pts = pts.select_enclosed_points(self.mesh, check_surface=False)

        if self.debug:
            p = pv.Plotter()
            p.add_mesh(self.mesh, opacity=0.5)
            p.add_mesh(start, color="b", render_points_as_spheres=True, label="start")
            p.add_mesh(stop, color="r", render_points_as_spheres=True, label="stop")
            p.add_mesh(
                intersection, color="pink", render_points_as_spheres=True, label="intersection"
            )
            p.add_mesh(pts, color="orange", render_points_as_spheres=True, label="midpoint")
            arrows = pv.PolyData().append_polydata(
                *[pv.Line(pointa=s, pointb=b) for s, b in zip(start, intersection)]
            )
            p.add_mesh(arrows, color="g", label="Normals")
            p.add_legend()
            p.show()

        if pts["SelectedPoints"].sum() < n_tests * (2 / 3):
            self.flip_seekers_directions()

    def run(self) -> pv.PolyData:
        """
        Run the algorithm and move seekers positions to its sought position.

        Arguments:
        ------------
            check_normals : bool, optional
                Default False. Whether to check for seekers direction before running the algorithm.
                Caveats: for this check to work well, the mesh must be a closed surface.

            check_inside : bool, optional
                Default False. Whether to remove seekers out of the mesh surface.
                Caveats: for this check to work well, the mesh must be a closed surface.

            multi_ray_trace : bool, optional
                Default False. If true, instead of using a for loop in python, ray tracing is
                performed by means of pyvista code which may be faster.
                Caveats: for this check to work well, the mesh must be a closed surface.

        Returns
        -------
            self.seekers : pv.Polydata.
                The polydata containing the seekers final positions.

        """

        attribute_checker(self, atts=["mesh"], info="Can't run seekers.")

        if self.seekers is None:
            self.compute_seekers_initial_positions()

        if self.check_dirs:
            self.check_seekers_direction()

        self.seekers.active_vectors_name = "Normals"

        dirs = self.seekers.active_normals

        d = self.mesh.length
        eps = d * 1e-4
        # The initial position is moved an epsilon inwards to prevent capturing
        # the initial intersection.
        start = self.seekers.points + dirs * eps

        if self.multi_ray_trace:
            intersection, _, _ = self.mesh.multi_ray_trace(
                origins=start, directions=dirs, first_point=True, retry=True
            )

        else:
            stop = self.seekers.points + dirs * d
            intersection = []
            for stt, stp in zip(start, stop):
                p = np.array([self.mesh.ray_trace(origin=stt, end_point=stp, first_point=True)[0]])
                if p.size == 0:
                    p = stp
                intersection.append(p.ravel())

        intersection = np.vstack(intersection)

        self.output_domain = pv.PolyData((start + intersection) / 2)

        if self.check_inside:
            self.output_domain = self.output_domain.select_enclosed_points(
                self.mesh, check_surface=False
            ).threshold(value=0.5, scalars="SelectedPoints", all_scalars=False, method="upper")

        if self.debug:
            p = pv.Plotter(shape=(1, 3))

            p.subplot(0, 0)
            p.add_mesh(self.mesh)

            p.subplot(0, 1)
            p.add_mesh(self.mesh, opacity=0.5)
            p.add_mesh(
                start, style="points", render_points_as_spheres=True, point_size=5, color="r"
            )
            p.add_arrows(start, dirs)

            p.subplot(0, 2)
            p.add_mesh(self.mesh, opacity=0.3)
            p.add_mesh(
                self.output_domain,
                style="points",
                render_points_as_spheres=True,
                point_size=5,
                color="b",
            )
            p.link_views()
            p.show()

        return self.output_domain


class Flux(CenterlineDomainExtractor):
    """
    Centerline domain extractor based on the flux approach. Theoretically the centerline geometric
    locus corresponds with the divergence null region of the gradient of the function distance to
    boundary or wall. Which is approximated by a flux estimation following:
        http://www.cim.mcgill.ca/~shape/publications/cvpr00.pdf.

    Caveats: The surface must be a closed surface!

    """

    def __init__(self) -> None:
        super().__init__()

        self.volume: pv.UnstructuredGrid = None
        self.mesh_kdt: KDTree = None
        self.volume_kdt: KDTree = None

        # Parameters
        self.dx: float = None
        self.dy: float = None
        self.dz: float = None
        self.thrs: float = 0.0
        # Setting this to true may help in bad connectivity scenarios.
        self.relax: bool = False

        self.debug: bool = False

    def set_mesh(self, m, update=True):
        """
        Set the surface mesh to be discretized.

        Arguments:
        -----------
            m : pv.PolyData,
                The mesh to be used.

            update : bool,
                Default True. If true, the KDTree is computed using the new
                mesh.
        """
        super().set_mesh(m=m)

        if update:
            self.compute_mesh_kdt()

    def compute_mesh_kdt(self):
        """Compute the KDTree of the mesh for fast distance query."""
        computing_message("KDTree")
        self.mesh_kdt = KDTree(self.mesh.points)
        computing_message("KDTree")

    def compute_voxelization(self, update=True) -> pv.UnstructuredGrid:
        """
        Compute the discretization of the inner volume of a closed surface by sampling the bounding
        box with sx, sy and sz spacing and rejecting outside points.
        """

        attribute_checker(self, atts=["mesh"], info="Can't voxelize mesh.")

        s = [self.dx, self.dy, self.dz]
        if None not in s:
            d = np.array(s)
        else:
            d = None
            self.dx, self.dy, self.dz = [self.mesh.length / 100] * 3

        computing_message("mesh voxelization")
        self.volume = pv.voxelize(self.mesh, density=d, check_surface=False)
        done_message("mesh voxelization")

        if update:
            self.compute_volume_kdt()

        return self.volume

    def compute_volume_kdt(self):
        """Build the volume KDTree attribute for fast distance query."""
        computing_message("volume KDTree")
        self.volume_kdt = KDTree(self.volume.points)
        computing_message("volume KDTree")

    def run(self) -> pv.UnstructuredGrid:
        """
        Extract the centerline domain with flux method.

        This method runs the centerline domain extraction based on a threshold over the flux
        (http://www.cim.mcgill.ca/~shape/publications/cvpr00.pdf) computed over a voxelization of
        the vascular volume.

        Returns
        -------
        self.output_domain : pv.UnstructuredGrid
            The extracted domain.
        """

        computing_message("centerline domain extraction using the flux...")
        attribute_checker(self, atts=["mesh"], info="Can't run seekers.")

        if self.mesh_kdt is None:
            self.compute_mesh_kdt()

        if self.volume is None:
            self.compute_voxelization()

        if self.volume_kdt is None:
            self.compute_volume_kdt()

        computing_message("flux field")
        self.volume["radius"] = self.mesh_kdt.query(self.volume.points)[0]
        self.volume = self.volume.compute_derivative(scalars="radius")

        r = np.max((self.dx, self.dy, self.dz)) * 1.2

        def net_flux(p):
            neighs = self.volume_kdt.query_ball_point(p, r=r)

            def flux(i):
                return (self.volume.points[i] - p).dot(self.volume["gradient"][i])

            fluxes = list(map(flux, neighs))
            return np.sum(fluxes)

        self.volume["flux"] = list(map(net_flux, self.volume.points))
        done_message("flux field")

        # Normalize and make it positive
        def normalize_field(arr):
            return arr / np.abs(arr).min() + 1

        self.volume["flux"] = normalize_field(self.volume["flux"])
        self.output_domain = self.volume.threshold(
            value=self.thrs, scalars="flux", all_scalars=self.relax, method="lower"
        ).connectivity(extraction_mode="largest")

        done_message("centerline domain extraction using the flux...")

        if self.debug:
            p = pv.Plotter()
            p.add_mesh(self.mesh, opacity=0.5)
            p.add_mesh(self.output_domain, scalars="flux")
            p.show()

        return self.output_domain


def extract_centerline_domain(
    vmesh: pv.PolyData, params: dict = None, debug: bool = False
) -> pv.DataObject:
    """
    Extract the centerline domain.

    The centerline domain extraction is the first step to compute the centerline. Here an unordered
    discrete representation of the centerline locus is computed. It can be tougher or finer, the
    optimum path is computed by means of the centerline path extractor.

    Parameters
    ----------
    vmesh : VascularMesh | pv.PolyData
        The surface defining the vascular domain where centerline is to be
        computed.
    method : {'seekers', 'flux'}, opt
        Defaulting to seekers. The algorithm used for the domain extraction.
    method_params : dict, opt
        A dictionary containing parameters of the chosen method.
    debug : bool, opt
        Defaulting to False. Running in debug mode shows some plots at certain steps.

    """

    if params is None:
        params = {}

    if "method" not in params:
        params["method"] = "seekers"

    if params["method"] not in ["seekers", "flux"]:
        raise ValueError(
            "Wrong value for method argument must be in {{'seekers', 'flux'}}, "
            + f"given is {params['method']}."
        )

    if params["method"] == "seekers":
        alg = Seekers()
    elif params["method"] == "flux":
        alg = Flux()
    else:
        raise Exception("How the hell have we arrived here?")

    alg.debug = debug

    update = True
    input_mesh = vmesh
    if isinstance(vmesh, VascularMesh):
        if vmesh.closed is None:
            vmesh.compute_closed_mesh()
        input_mesh = vmesh.closed.compute_normals(cell_normals=False, point_normals=True)
        if vmesh.kdt is not None and params["method"] == "flux":
            alg.mesh_kdt = vmesh.kdt
            update = False

    alg.set_mesh(m=input_mesh, update=update)

    alg.set_parameters(**params)
    alg.run()

    return alg.output_domain
