#!/usr/bin/env python3


from scipy.spatial import KDTree
import numpy as np
import pyvista as pv

from utils import attribute_checker
import messages as msg

class Seekers:
    """
    Centerline domain extractor based on the seekers approach.
    According to the grass fire equation the surface inward normal
    is a good estimation for finding the centerline. Experimentally
    we found that the midpoint between any given point in the wall
    and its projection by ray trace using the inward normal is a robust,
    and fast approximation of the centerline locus. The benefit of this
    method against others is that it does not require a closed surface,
    as long as the normals are pointing inward.

    Caveats: optimality of the centerline is not warranted.
    """

    def __init__(self):

        self.mesh : pv.PolyData = None
        self.reduction_rate     = 0.66
        self.smooth_iters       = 100
        self.seekers : pv.Polydata = None
        self.eps = 1e-3

        self.debug : bool = False
    #

    def set_mesh(self, m, update=True):
        self.mesh = m

        if update:
            self.compute_seekers_initial_positions()
    #

    def compute_seekers_initial_positions(self):
        """
        Compute the initial position of the seekers by computing a decimation of the original mesh.
        The mesh is then smoothed to enhance the normal direction. This method requires mesh and
        target_reduction attributes to be already set.

        Returns:
        ----------
            self.seekers : pv.PolyData
                The seekers initial position.
        """

        if not attribute_checker(obj=self, atts=['mesh', 'reduction_rate'], extra_info='Cannot compute initial seekers position.'):
            return False

        self.seekers = self.mesh.decimate(target_reduction=self.reduction_rate, attribute_error=False).smooth(n_iter=self.smooth_iters).compute_normals(cell_normals=False, point_normals=True)
        return self.seekers
    #

    def flip_seekers_directions(self):
        """
        Flip the seekers directions to point inwards.
        """

        msg.computing_message("flipped normals")
        self.seekers = self.seekers.compute_normals(flip_normals=True)
        msg.done_message("flipped normals")
    #

    def check_seekers_direction(self, n_tests=21):
        """
        Ensure the seekers directions point inwards. This method requires mesh to
        be a closed surface to work properly. It works by performing a step of the
        seekers algorithm with a reduced number of points (controled by n_tests).
        Then if the number of points falling outside the mesh surface is greater than
        those inside, the normals are flipped.
        """

        eps = self.mesh.length * self.eps
        ids = np.random.randint(low=0, high=self.seekers.n_points-1, size=n_tests)
        dirs = self.seekers['Normals'][ids]
        #The initial position is moved an epsilon inwards to prevent capturing the initial intersection.
        start = self.seekers.points[ids] + dirs * eps
        stop  = self.seekers.points[ids] + dirs * self.mesh.length
        intersection = []
        for (stt, stp) in zip(start, stop):
            p = np.array([self.mesh.ray_trace(origin=stt, end_point=stp, first_point=True)[0]])
            if p.size == 0:
                p = stp
            intersection.append(p.ravel())
        intersection = np.vstack(intersection)

        pts = pv.PolyData((start+intersection)/2)
        pts = pts.select_enclosed_points(self.mesh)
        if not pts["SelectedPoints"].sum() > pts.n_points * 0.5:
            self.flip_seekers_directions()
    #

    def run(self, check_dirs=False, check_inside=False, multi_ray_trace=False):
        """
        Run the algorithm and move seekers positions to its seeked position.

        Arguments:
        ------------

            check_normals : bool, optional
                Default False. Whether to check for seekers direction before runnig the algorithm.
                Caveats: for this check to work well, the mesh must be a closed surface.

            check_inside : bool, optional
                Default False. Whether to remove seekers out of the mesh surface.
                Caveats: for this check to work well, the mesh must be a closed surface.

            multi_ray_trace : bool, optional
                Default False. If true, instead of using a for loop in python, ray tracing is
                performed by means of pyvista code which may be faster.
                Caveats: for this check to work well, the mesh must be a closed surface.

        Returns:
        --------
            self.seekers : pv.Polydata.
                The polydata containing the seekers final positions.

        """

        if not attribute_checker(self, atts=['mesh'], extra_info="Cannot run seekers."):
            return False

        if self.seekers is None:
            self.compute_seekers_initial_positions()

        if check_dirs:
            self.check_seekers_direction()


        self.seekers.active_vectors_name = 'Normals'

        dirs = self.seekers.active_normals

        d = self.mesh.length
        eps = d * 1e-4
        start = self.seekers.points + dirs*eps #The initial position is moved an epsilon inwards to prevent capturing the initial intersection.

        if multi_ray_trace:
            intersection, _, _ = self.mesh.multi_ray_trace(origins=start, directions=dirs, first_point=True, retry=True)

        else:
            stop = self.seekers.points + dirs * d
            intersection = []
            for (stt, stp) in zip(start, stop):
                p = np.array([self.mesh.ray_trace(origin=stt, end_point=stp, first_point=True)[0] ])
                if p.size == 0:
                    p = stp
                intersection.append(p.ravel())

        intersection = np.vstack(intersection)

        self.seekers = pv.PolyData((start+intersection) / 2)


        if check_inside:
            self.seekers = self.seekers.select_enclosed_points(self.mesh)
            self.seekers = self.seekers.extract_points(self.seekers['SelectedPoints'], adjacent_cells=True)


        if self.debug:
            p = pv.Plotter(shape=(1, 3))

            p.subplot(0,0)
            p.add_mesh(self.mesh)

            p.subplot(0, 1)
            p.add_mesh(self.mesh, opacity=0.5)
            p.add_mesh(start, style='points', render_points_as_spheres=True, point_size=5, color='r')
            p.add_arrows(start, dirs, mag=1)

            p.subplot(0, 2)
            p.add_mesh(self.mesh, opacity=0.3)
            p.add_mesh(self.seekers, style='points', render_points_as_spheres=True, point_size=5, color='b')
            p.link_views()
            p.show()

        return self.seekers
    #
#


class Flux:

    """
    Centerline domain extractor based on the flux approach. Theoretically
    the centerline geometric locus corresponds with the divergence null
    region of the gradient of the function distance to boundary or wall.
    Which is approximated by a flux estimation following:
    http://www.cim.mcgill.ca/~shape/publications/cvpr00.pdf

    Caveats: The surface must be a closed surface!

    """

    def __init__(self) -> None:

        self.mesh     : pv.PolyData = None
        self.mesh_kdt : KDTree
        self.volume   : pv.UnstructuredGrid = None
        self.volume_kdt : KDTree

        # Discretization deltas
        self.dx = None
        self.dy = None
        self.dz = None

        self.debug  : bool = False
    #

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
        self.mesh = m

        if update:
            self.compute_mesh_kdt()
    #

    def compute_mesh_kdt(self):
        msg.computing_message("KDTree")
        self.mesh_kdt = KDTree(self.mesh.points)
        msg.computing_message("KDTree")
    #

    def compute_voxelization(self, update=True):

        """
        Compute the discretization of the inner volume of a closed surface by
        sampling the bounding box with sx, sy and sz spacing and rejecting outside points.
        """

        if not attribute_checker(self, atts=['mesh'], extra_info="Cannot voxelize mesh."):
            return False

        s = [self.dx, self.dy, self.dz]
        if not None in s:
            d = np.array(s)
        else:
            d = None
            self.dx, self.dy, self.dz = [self.mesh.length/100]*3

        msg.computing_message("mesh voxelization")
        self.volume = pv.voxelize(self.mesh,
                                  density=d,
                                  check_surface=True)
        msg.done_message("mesh voxelization")

        if update:
            self.compute_volume_kdt()

        return self.volume
    #

    def compute_volume_kdt(self):
        msg.computing_message("volume KDTree")
        self.volume_kdt = KDTree(self.volume.points)
        msg.computing_message("volume KDTree")
    #

    def run(self, thrs=0.0):
        """
        This method runs the centerline domain extraction based on a threshold over
        the flux (http://www.cim.mcgill.ca/~shape/publications/cvpr00.pdf) computed
        over a voxelization of the vascular volume.

        Arguments:
        ------------
            thrs : float, optional.
                Default 0.0. The higher bound for the flux thresholding.

        """

        msg.computing_message("centerline domain extraction using the flux...")
        if not attribute_checker(self, atts=['mesh'], extra_info="Cannot run seekers."):
            return False

        if self.mesh_kdt is None:
            self.compute_mesh_kdt()

        if self.volume is None:
            self.compute_voxelization()

        if self.volume_kdt is None:
            self.compute_volume_kdt()

        msg.computing_message("flux field")
        self.volume['distance'] = self.mesh_kdt.query(self.volume.points)[0]
        self.volume = self.volume.compute_derivative(scalars='distance')

        r = np.max((self.dx, self.dy, self.dz))*1.2
        def net_flux(p):
            neighs = self.volume_kdt.query_ball_point(p, r=r)
            flux = lambda i: (self.volume.points[i]-p).dot(self.volume["gradient"][i])
            fluxes = list(map(flux,neighs))
            return np.sum(fluxes)

        self.volume['flux'] = list(map(net_flux, self.volume.points))
        msg.done_message("flux field")

        # Normalize and make it positive
        normalize_field = lambda arr: arr / np.abs(arr).min() + 1
        self.volume['flux'] = normalize_field(self.volume['flux'])
        self.volume = self.volume.extract_points(self.volume['flux'] < thrs, adjacent_cells=True)
        self.volume = self.volume.connectivity(extraction_mode='largest')

        msg.done_message("centerline domain extraction using the flux...")

        if self.debug:
            p = pv.Plotter()
            p.add_mesh(self.mesh, opacity=0.5)
            p.add_mesh(self.volume, scalars='flux')
            p.show()
        return self.volume
    #
#