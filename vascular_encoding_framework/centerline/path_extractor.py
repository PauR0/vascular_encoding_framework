

import heapq as hp

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

from ..messages import *
from ..vascular_mesh import VascularMesh, Boundaries
from ..utils._code import attribute_checker, attribute_setter


def minimum_cost_path(heuristic, cost, adjacency, initial, ends):
    """
    Function to compute the minimum cost path by means of the A* algorithm.

    Args:
    ------

        heuristic : callable,
            A one arg callable heuristic object.

        cost : callable,
            The two args callable object that returns the
            numeric cost of the edge between two connected nodes.

        adjacency : callable,
            A one arg callable object that returns the neighbours of a node.

        initial : int
            The id of the initial node

        ends : int
            The ids of the terminal states

    Returns:
    ---------
        path : list
            List containing the ids of the computed path.
            If the algorithm fails, it returns an empty list

    """
    g = {} # Cummulative distance from the origin
    f = {} # Cost of each node
    h = heuristic

    # Step 1: Algorithm start
    g[initial] = 0
    f[initial] = g[initial]+h(initial)

    OpenList = [ (f[initial], initial) ]
    OpenSet = set([initial])

    Closed = set([])

    current_node = initial

    pointers={}
    pointers[initial] = None
    n_expl = 0
    while OpenList:

        _, current_node = hp.heappop(OpenList)
        OpenSet.remove(current_node)

        Closed.add(current_node)

        if current_node in ends:
            break

        successors = adjacency(current_node)

        for s in successors:
            if s == pointers[current_node] or s == current_node:
                continue

            g_s = g[current_node] + cost(current_node, s)

            old = None

            s_in_open = s in OpenSet
            s_in_closed = s in Closed

            if s_in_open:
                old = s
                if g[old] > g_s:
                    OpenList.remove((f[old], old))
                    g[old] = g_s
                    pointers[old] = current_node
                    f[old] = g[old] + h(old)
                    hp.heappush(OpenList, (f[old], old))
            if s_in_closed:
                old = s
                if g[old] > g_s:
                    g[old] = g_s
                    pointers[old] = current_node
                    f[old] = g[old] + h(old)
                    Closed.remove(old)
                    hp.heappush(OpenList, (f[old], old))
                    OpenSet.add(old)

            if not s_in_open and not s_in_closed:
                pointers[s] = current_node
                g[s] = g_s
                f[s] = g[s] + h(s)
                hp.heappush(OpenList, (f[s], s))
                OpenSet.add(s)
        n_expl += 1

    if current_node in ends:
        return build_path(current_node, pointers)

    return []
#

def build_path(current_node, pointers, reverse_path=False):
    path = []
    previous = current_node

    while previous is not None:
        path.append(previous)
        previous = pointers[previous]
    if reverse_path:
        path.reverse()

    return path
#

class CenterlinePathExtractor:
    """
    The class to compute the centerline paths in the centerline domain of a
    vascular mesh.

    """

    def __init__(self):

        self.vmesh              : VascularMesh = None
        self.centerline_domain  : np.ndarray   = None
        self.domain_kdt         : KDTree       = None
        self.radius             : np.ndarray   = None
        self.inverse_radius     : np.ndarray   = None
        self.boundaries         : Boundaries   = None


        self.mode             : str   = 'j2o'
        self.reverse          : bool  = False
        self.adjacency_factor : float = 0.33
        self.pass_pointid     : bool  = True
        self.debug            : bool = False


        self.id_paths : list[list[int]] = None
        self.paths    : pv.MultiBlock   = None
    #

    def set_parameters(self, **kwargs):
        """
        Method to set parameters of the path extractor.
        """
        cl = self.__class__()
        params = {k:v for k,v in kwargs.items() if k in cl.__dict__}
        attribute_setter(self, **params)
    #

    def set_vascular_mesh(self, vm, check_radius=True, update_boundaries=True):
        """
        Set the vascular domain. If check_radius is True, self.centerline_domain
        is not None, and self.radius is None, then self.radius is computed.

        Arguments:
        -------------

            vm : VascularMesh
                The input vascular mesh

            check_radius : bool, opt
                Whether to check for radius existence and comput it if required.

            update_boundaries : bool, opt
                Whether to update the boundaries dict with the ones in vm.

        """

        self.vmesh = vm
        if check_radius and self.radius is None and self.centerline_domain is not None:
            self.compute_radius_fields()

        if update_boundaries:
            self.boundaries_from_vascular_mesh()
    #

    def set_centerline_domain(self, cntrln_dmn, check_fields=True, check_radius=True):
        """
        Set the centerline domain. If cd is a pv.PolyData, the radius field
        if searched in the fields defined on the point_data.

        Arguments:
        -----------

            cntrln_dmn : np.ndarray | pv.DataSet
                The centerline domain. It can be a numpy array in shape (N,3)
                or a pyvista object (PolyData, UnstructuredGrid...) with
                the points attribute.

            check_fields : bool, opt.
                If cntrln_dmn is a pyvista object and check_fields is True,
                the point_data attribute of cntrln_dmn checked looking for the
                radius field.

            check_radius : bool
                Whether to check for radius existence and comput it if required.

        """

        if isinstance(cntrln_dmn, np.ndarray):
            if cntrln_dmn.shape[0] != 3:
                error_message(f"Unable to set an array with shape {cntrln_dmn.shape} as centerline domain. " +\
                                  " Centerline domain must be a list of points wiht shape must be (3, N)")
                return False
            self.centerline_domain = cntrln_dmn

        elif isinstance(cntrln_dmn, pv.DataSet):
            self.centerline_domain = cntrln_dmn.points
            if check_fields:
                if 'radius' in cntrln_dmn.point_data:
                    self.set_radius(cntrln_dmn.get_array('radius', preference='point'))

        if check_radius and self.radius is None and self.vmesh is not None:
            self.compute_radius_fields()

        #Computing centerline_domain_kdt
        self.compute_kdt()

        return True
    #

    def compute_radius_fields(self):
        """
        Use KDTree attribute in VascularMesh attribute to compute the radius
        field.
        """

        if not attribute_checker(self, ['vmesh'], info='Cant compute radius field.'):
            return False

        if not attribute_checker(self.vmesh, ['kdt'], info='Cant compute radius field. The vascular mesh provided has no kdt'):
            return False

        self.radius = self.vmesh.kdt.query(self.centerline_domain)[0]
        self.inverse_radius = 1/self.radius

        return self.radius
    #

    def set_radius(self, r, update_inv=True):
        """
        Set the radius attribute and if update_inv is True, update the inverse_radi
        """

        self.radius = r

        if update_inv:
            self.inverse_radius = 1 / self.radius
    #

    def remove_from_centerline_domain_by_id(self, ids, update_kdt=True):
        """
        Remove points from centerline domain according to their index. Then linked
        attributes such as the radius or the domain kdt are updated.

        Arguments:
        ------------

            ids : int or array-like
                The indexes of the points to remove.

            update_kdt : bool, opt
                Default True. Whether to update the domain_kdt. This argument should only be false
                if domain_kdt is going to be updated later.
        """

        mask = np.ones((self.centerline_domain.shape[0],), dtype=bool)
        mask[ids] = False
        self.centerline_domain = self.centerline_domain[mask]
        self.radius = self.radius[mask]
        self.inverse_radius = self.inverse_radius[mask]

        if update_kdt:
            self.compute_kdt()
    #

    def add_point_to_centerline_domain(self, p, where='end', update_kdt=True):
        """
        Append a point to the centerline domain updating the radius and inverse radius arrays.
        If an array with dimensions (N, 3), the N points are appended. The argument where can
        be used to decide if inserting the point(s) in the beggining of the lists (ini) or at
        the end (end).

        WARNING: Using this method changes the indices of inlets and outlets in the arrays.
        Hence, it should be used with caution!!


        Arguments:
        ------------

            p : np.ndarray, either (3,) or (N, 3)
                The point(s) to be appended.

            where : str {'ini', 'end'}
                Default to 'end'
                Where to locate the point in the arrays.

            update_kdt : bool
                Default True. Whether to recompute kdt or not. This should only
                be set to False if multiple calls are going to be made, and the
                last call shoult let it to True.

        Returns:
        ---------
            ind : [int]
                A list with point indices
        """

        if (p.ndim == 1 and p.shape == (3,)) or (p.ndim == 2 and p.shape[1] == 3):

            if where == 'ini':
                stack_p = lambda x, y : [x.reshape(-1,3), y]
                stack_r = lambda x, y : [x, y]
                if p.ndim == 1:
                    ind = 0
                else:
                    ind = range(len(p))

            elif where == 'end':
                stack_p = lambda x, y : [y, x.reshape(-1,3)]
                stack_r = lambda x, y : [y, x]
                n_old = self.centerline_domain.shape[0]
                if p.ndim == 1:
                    ind = n_old
                else:
                    ind = list(range(n_old, n_old+len(p)))
            else:
                error_message("Wrong argument for where in add_point_to_centerline_domain method. Available options are {'ini', 'end'}.")
                return False

            #We update the centerline domain list and the ones in correspondence (radius and inverse_radius)
            self.centerline_domain = np.vstack(stack_p(p, self.centerline_domain))

            r = self.vmesh.kdt.query(p)[0]
            self.radius         = np.hstack(stack_r(r, self.radius))
            self.inverse_radius = np.hstack(stack_r(1/r, self.inverse_radius))

            if update_kdt:
                self.compute_kdt()

            return ind

        error_message("Trying to insert a point into centerline's domain with bad shape." + \
                              "Accepted shapes are (3,) or (N,3)")
        return False
    #

    def compute_kdt(self):
        """
        Compute the KDTree using the available points.
        """
        self.domain_kdt = KDTree(self.centerline_domain)
    #

    def boundaries_from_vascular_mesh(self, vm=None, force_tangent=True, copy=True):
        """
        Assume the hierarchy defined by the boundaries of a vascular mesh.

        If force_tangent is true, and "node" key/atribute is present,
        points at distance radius from center are removed, and eight points
        are inserted as center +t*radius*normal with t in linspace(-1,1,8).



        Arguments:
        -------------

            vm : VascularMesh, opt.
                Default self.vmesh. The vascular mesh to use.

            force_tangent : bool, opt
                Default True. Whether to force centerline tangent using boundary normals.

            copy : bool, opt.
                Default True. Whether to make a deep copy of the hierarchy or
                use it by reference.

        """

        if vm is None:
            if not attribute_checker(self, ['vmesh'], info="no VascularMesh has been passed and..."):
                return
            vm = self.vmesh

        if not attribute_checker(vm, ['boundaries'], info="can't compute hirarchy from vmesh."):
            return

        self.set_boundaries(bndrs=vm.boundaries, force_tangent=force_tangent, copy=copy)
    #

    def set_boundaries(self, bndrs, force_tangent=False, copy=True):
        """
        Set the boundary hierarchy to compute the centerline paths.
        If hierarchy is set paths are first computed from children to
        parents. Then, the id_paths are arranged according to mode and
        reverse.

        Ex. Ex. hierarchy = {"1" : {"parent"     : None,
                                    "center"   : [ x1, y1, z1],
                                    "children" : {"2"}},
                             "2" : {"parent"     : '1',
                                     "center"   : [ x2, y2, z2],
                                     "children" : {"0"}},
                             "0" : {"parent"     : '2',
                                     "center"   : [ x0, y0, z0],
                                     "children" : {}}
                            }

        With this hierarchy the algorithm would extract the paths:
        "1" - "2"
               |
              "0"
        And then rearrange them according to self.mode and self.reverse to set directions propperly.

        Arguments:
        ------------

            bndr : Boundaries or dict
                A Boundaries object where each Boundary object has its center attribute, or a hierarchy
                dictionary to parse.

            force_tangent : bool, opt
                Default True. Whether to force centerline tangent using boundary normals.

            copy : bool, opt.
                Default True. Whether to make a deep copy of the hierarchy or
                use it by reference.

        """

        if isinstance(bndrs, Boundaries):
            if copy:
                self.boundaries = bndrs.copy()
            else:
                self.boundaries = bndrs
        elif isinstance(bndrs, dict):
            self.boundaries = Boundaries(bndrs)

        if force_tangent:
            def ensure_boundary_normal(bid):
                c, n = self.boundaries[bid].center.reshape(-1, 1), self.boundaries[bid].normal.reshape(-1, 1)
                r    = self.vmesh.kdt.query(c.T)[0] * self.adjacency_factor * 1.5
                ids = self.domain_kdt.query_ball_point(x=c.ravel(), r=r)
                self.remove_from_centerline_domain_by_id(ids=ids)

                points = (c + n*r*np.linspace(-0.75, 0, round(2/self.adjacency_factor), endpoint=False)).T
                self.add_point_to_centerline_domain(p=points, where='end', update_kdt=False)

                points = (c + n*r*np.linspace(0.75, 0, round(2/self.adjacency_factor), endpoint=False)).T
                self.add_point_to_centerline_domain(p=points, where='end')

                for cid in self.boundaries[bid].children:
                    ensure_boundary_normal(bid=cid)

            for rid in self.boundaries.roots:
                ensure_boundary_normal(bid=rid)

        def add_bound_point(bid):
            self.boundaries[bid].set_data(cl_domain_id=self.add_point_to_centerline_domain(p=self.boundaries[bid].center, where='end', update_kdt=False))
            for cid in self.boundaries[bid].children:
                add_bound_point(bid=cid)

        for rid in self.boundaries.roots:
            add_bound_point(bid=rid)

        self.compute_kdt()
    #

    def _compose_id_paths(self):
        """
        Arrange the id_path attribute according to the policy established in the
        mode attribute.
        """

        if not attribute_checker(self, ['mode'], info="wrong mode chosen to extract centerline paths...", opts=[['i2o', 'j2o']]):
            return False

        def arrange_path(bid):

            if self.boundaries[bid].parent is not None:

                pid = self.boundaries[bid].parent
                joint = self.boundaries[bid].id_path[0] #Junction id in cl_domain
                if joint not in self.boundaries[pid].id_path:
                    error_message(f"At node {bid} cant find joint id in parent's path (parent id {pid}). Something has crashed during path extraction...")
                    return
                jid = self.boundaries[pid].id_path.index(joint)

                if self.mode == 'i2o':
                        self.boundaries[bid].id_path = self.boundaries[pid].id_path[:jid] + self.boundaries[bid].id_path
                if self.reverse:
                    self.boundaries[bid].id_path.reverse()

            for cid in self.boundaries[bid].children:
                arrange_path(cid)

        for rid in self.boundaries.roots:
            arrange_path(rid)
            self.boundaries[rid].id_path = None
    #

    def compute_paths(self):
        """
        Compute the paths from each outlet to the inlet. The path computation
        is based on an implementation of the minimum cost path A* algorithm,
        however, since the heuristic is set to 0 the algorithm is efectively
        Dijkstra's. The stopping criteria is the reach of the inlet, or a
        previously transited path.
        """

        if self.debug:
            p = pv.Plotter()
            p.add_mesh(self.vmesh, opacity=0.4)
            p.add_mesh(self.centerline_domain, scalars=self.radius, render_points_as_spheres=True)
            p.add_point_labels(points=np.array([b.center for _, b in self.boundaries.items()]),
                               labels=self.boundaries.enumerate(), font_size=20, point_color='red',
                               point_size=20, render_points_as_spheres=True, always_visible=True)
            p.show()

        def path_to_parent(bid):
            if self.boundaries[bid].parent is None:
                self.boundaries[bid].set_data(to_numpy=False, id_path=[self.boundaries[bid].cl_domain_id])
            else:
                pid = self.boundaries[bid].parent
                parent_path = self.boundaries[pid].id_path
                id_path=minimum_cost_path(heuristic = self._heuristic,
                                        cost      = self._cost,
                                        adjacency = self._adjacency,
                                        initial=self.boundaries[bid].cl_domain_id,
                                        ends=parent_path)
                if not id_path:
                    error_message("Could not found path between boundaries {bid} and {pid}. This may happen due to a too tiny adjacency_ratio or a too sparse centerline domain...")
                self.boundaries[bid].set_data(to_numpy=False, id_path=id_path)
            for cid in self.boundaries[bid].children:
                path_to_parent(cid)

        for rid in self.boundaries.roots:
            path_to_parent(rid)

        self._compose_id_paths()
        self.make_paths_multiblock()

        return
    #

    def make_paths_multiblock(self):
        """
        Build a pyvista(vtk) MultiBlock, by converting the id_paths into a PolyData.

        Returns:
        ----------
            self.paths : pyvista.MultiBlock
                The MutliBlock with the centerline paths
        """

        self.paths = pv.MultiBlock()

        def make_polydata_path(bid):
            if self.boundaries[bid].id_path is not None:
                pdt = pv.PolyData()
                pdt.points = self.centerline_domain[self.boundaries[bid].id_path]
                pdt.lines  = np.array([[2, j, j+1] for j in range(len(self.boundaries[bid].id_path)-1)], dtype=int)
                pdt['cl_domain_id'] = np.array(self.boundaries[bid].id_path, dtype=int)
                pdt['radius']       = self.radius[self.boundaries[bid].id_path]
                pdt.field_data['parent'] = [self.boundaries[bid].parent]
                if self.boundaries[bid].parent in self.boundaries.roots:
                    pdt.field_data['parent'] = ['None']
                self.paths.append(pdt, name=f"path_{bid}")
            for cid in self.boundaries[bid].children:
                make_polydata_path(bid=cid)

        for rid in self.boundaries.roots:
            make_polydata_path(rid)
    #

    def _heuristic(self, n):
        """At this moment, we use no heuristics."""
        return 0.0
    #

    def _cost(self, current, neigh):
        """
        We defined the cost as the inverse of the radius field.
        Only the arriving node radius is taking into account.
        """
        return self.inverse_radius[neigh]
    #

    def _adjacency(self, n):
        """
        The neighbours are points at a certain distance proportional to the radius of the point.
        As long as the adjacency factor is below 1, it preserves the topology of the vascular segment.
        """
        return self.domain_kdt.query_ball_point(self.centerline_domain[n], r=self.radius[n]*self.adjacency_factor)
    #
#


def extract_centerline_path(vmesh, cl_domain, params, debug=False):
    """
    Compute the discrete centerline path of a vascular mesh based on a discretization of the lumen.
    It is computed as the polylines definig the minimum cost path between the boundary centers of a
    vascular mesh. If the boundary normal is provided, extra points are added to the domain to
    ensure appropriate boundary conditions.

    Warning: The minimum cost path is computed using A* algorithm without heuristic function (what
    makes it Dikjsta's algorithm), for which the adjacency (neighborhood) is built using the radius
    field scalated by an adjacency_factor (default .33). If this function fails to extract the path
    try providing a more dense discretizatión of the lumen or increasing the adjacency_factor.

    Arguments:
    ----------

        vmesh : VascularMesh
            The VascularMesh object where centerline path is to be computed.

        cntrln_dmn : np.ndarray | pv.DataSet
            The lumen discretization. It can be a numpy array in shape (N,3)
            or a pyvista object (PolyData, UnstructuredGrid...) with
            the points attribute. Additionally if the point_data 'radius' exists
            the computation is skipped.

        params : dict
            A dictionary with the parameters for the path extraction. The keys of
            the dictionaries have to be as the attributes defined in the constructor
            of the CenterlinePathExtractor object. Other dictionary entries are ignored.

        debug : bool, optional
            Default False. Whether to run path extraction in debug mode.

    Returns:
    --------
        outpaths : pv.MultiBlock

    """

    cp_xtractor = CenterlinePathExtractor()
    cp_xtractor.set_centerline_domain(cl_domain)
    cp_xtractor.set_vascular_mesh(vmesh, update_boundaries=True)
    cp_xtractor.debug = debug

    if params is not None:
        cp_xtractor.set_parameters(**params)

    cp_xtractor.compute_paths()
    outpaths = cp_xtractor.paths

    return outpaths