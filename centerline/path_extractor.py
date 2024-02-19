#! /usr/bin/env python3

import heapq as hp

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

from vascular_mesh import VascularMesh
import messages as msg
from utils._code import attribute_checker

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
        self.inlet              : list         = None
        self.outlets            : list         = None

        self.mode             : str   = 'i2o' #TODO: Implement j2o (junction to outlets, hierarchical tree)
        self.adjacency_factor : float = 0.33
        self.pass_pointid     : bool   = True

        self.id_paths : list[list[int]] = None
        self.paths    : pv.MultiBlock   = None
    #

    def set_vascular_mesh(self, vm, check_radius=True):
        """
        Set the vascular domain. If check_radius is True, self.centerline_domain
        is not None, and self.radius is None, then self.radius is computed.

        Arguments:
        -------------

            vm : VascularMesh
                The input vascular mesh

            check_radius : bool
                Whether to check for radius existence and comput it if required.
        """

        self.vmesh = vm
        if check_radius and self.radius is None and self.centerline_domain is not None:
            self.compute_radius_fields()
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
                msg.error_message(f"Unable to set an array with shape {cntrln_dmn.shape} as centerline domain. " +\
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
        self.domain_kdt = KDTree(self.centerline_domain)

        return True
    #

    def compute_radius_fields(self):
        """
        Use KDTree attribute in VascularMesh attribute to compute the radius
        field.
        """

        if not attribute_checker(self, ['vmesh'], extra_info='Cant compute radius field.'):
            return False

        if not attribute_checker(self.vmesh, ['kdt'], extra_info='Cant compute radius field. The vascular mesh provided has no kdt'):
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
                msg.error_message("Wrong argument for where in add_point_to_centerline_domain method. Available options are {'ini', 'end'}.")
                return False

            #We update the centerline domain list and the ones in correspondence (radius and inverse_radius)
            self.centerline_domain = np.vstack(stack_p(p, self.centerline_domain))
            self.set_radius(np.hstack(stack_r(self.vmesh.kdt.query(p)[0], self.radius)), update_inv=True)

            if update_kdt:
                self.domain_kdt = KDTree(self.centerline_domain)

            return ind

        msg.error_message("Trying to insert a point into centerline's domain with bad shape." + \
                              "Accepted shapes are (3,) or (N,3)")
        return False
    #

    def set_inlet(self, inlt):
        """
        Set the inlet of the vascular structure. It can be both, the id
        of a point in the centerline domain, or a numpy array containing
        the coordinates of new 3D point.

        Arguments:
        ----------

            inlt : int or np.ndarray (3,)
                The id of the point in the centerline_domain array
                or a 3D point to use as the inlet.
        """

        if isinstance(inlt, int):
            self.inlet = inlt
        elif isinstance(inlt, np.ndarray):
            if inlt in self.centerline_domain:
                self.inlet = np.where(self.centerline_domain)[0][0]
            else:
                self.inlet = self.add_point_to_centerline_domain(p=inlt, where='ini')
    #

    def set_outlets(self, outlt):
        """
        Set the outlets of the vascular structure. It can be both, a list
        with the ids of the outlet points in the centerline domain, or a
        numpy array containing the coordinates them.

        Arguments:
        ----------

            outlt : list[int] or array-like (N, 3)
                The id of the point in the centerline_domain array
                or a 3D point to use as the inlet.
        """
        if isinstance(outlt, int):
            self.outlets = outlt
        elif isinstance(outlt, np.ndarray):
            self.outlets = self.add_point_to_centerline_domain(p=outlt, where='end')
    #

    def compose_id_paths(self, raw_paths):
        """
        Compose the path attribute according to the policy established in the
        mode attribute.

        Arguments:
        ----------

            raw_paths : list[[ids]]
                The raw paths extracted using the A* algorithm.

        Returns:
        --------

            paths : list[[ids]]
                The list of paths meeting the mode criteria.
        """

        if not attribute_checker(self, ['mode'], extra_info="wrong mode chosen to extract centerline paths...", opts=['i2o', 'o2i', 'j2o', 'o2j']):
            return False

        paths = []

        if self.mode in ['i2o', 'o2i']:
            for i, path in enumerate(raw_paths):
                if i == 0 and self.inlet not in path:
                    msg.error_message("Can't find inlet in the first path provided...")
                    return False

                if i == 0:
                    paths.append(path)

                else:
                    j, joint = 0, path[0] #id of the path, and id of the joint.
                    while joint not in paths[j] and j < len(paths)-1:
                        j += 1
                    if joint not in paths[j]:
                        msg.error_message(f"Unable to find the joint for the branch at outlet {path[0]}")
                        paths.append(path)
                    else:
                        jid = paths[j].index(joint)
                        paths.append(paths[j][:jid] + path)

            if self.mode == 'o2i':
                for p in paths: p.reverse()

            return paths

        if self.mode in ['j2o', 'o2j']:
            msg.error_message("Not implemented yet....") #TODO: Implement! (requires hierarchy)
            return None
    #

    def compute_paths(self):
        """
        Compute the paths from each outlet to the inlet. The path computation
        is based on an implementation of the minimum cost path A* algorithm,
        however, since the heuristic is set to 0 the algorithm is efectively
        Dijkstra's. The stopping criteria is the reach of the inlet, or a
        previously transited path.
        """

        msg.computing_message("centerline paths")
        raw_paths = []
        end_points = [self.inlet]
        for out in self.outlets:
            new_path = minimum_cost_path(heuristic = self._heuristic,
                                         cost      = self._cost,
                                         adjacency = self._adjacency,
                                         initial=out,
                                         ends=end_points)
            end_points += new_path
            raw_paths.append(new_path)

        self.id_paths = self.compose_id_paths(raw_paths)
        self.make_paths_multiblock()

        msg.done_message("centerline paths")

        return
    #

    def make_paths_multiblock(self):
        """
        Build a pyvista(vtk) MultiBlock, by converting each id_path into a PolyData.

        Returns:
        ----------
            self.path : pyvista.MultiBlock
                The MutliBlock with the centerline paths
        """

        if not attribute_checker(self, ['id_paths'], extra_info="Cant make polyldata path."):
            return

        self.paths = pv.MultiBlock()
        for i, idp in enumerate(self.id_paths):
            pdt = pv.PolyData()
            pdt.points = self.centerline_domain[idp]
            pdt.lines  = [[2, j, j+1] for j in range(len(idp)-1)]
            if self.pass_pointid:
                pdt['cl_domain_id'] = np.array(idp, dtype=int)
            self.paths.append(pdt, name=f"Branch_{i}")
        #
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