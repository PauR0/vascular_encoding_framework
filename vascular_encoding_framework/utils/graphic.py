

import numpy as np
import pyvista as pv

from ..centerline import Centerline, CenterlineNetwork
from ..messages import *


def plot_adapted_frame(cntrln, vmesh=None, plotter=None, scale=1, show=True):
    """
    Plot the parallel transport defined on a centerline.

    Arguments:
    -----------

        cl : Centerline or CenterlineNetwork
            The centerline to plot

        vmesh : VascularMesh or pv.PolyData
            The vascular mesh used to compute the centerline.

        plotter : pv.Plotter
            Default None. If passed, parallel_transport is displayed there.

        scale : float, opt
            By default no scale is applied. The scale of the arrows used to plot the adapted frame vectors.

        show : bool, opt
            Default True. Whether to show the plot or not.

    """

    if plotter is None:
        plotter = pv.Plotter()

    def plot_cl_pl(cl):

        pdt = pv.PolyData()
        pdt.points = cl.samples
        pdt.lines = np.array([[2, i, i + 1]
                             for i in range(cl.n_samples - 1)], dtype=int)
        plotter.add_mesh(
            pdt,
            color='k',
            render_lines_as_tubes=True,
            line_width=5)

        pdt['tangent'] = np.array([cl.get_tangent(t)
                                  for t in cl.parameter_samples])
        tgts = pdt.glyph(orient='tangent', factor=scale)
        plotter.add_mesh(tgts, color='r')

        pdt['v1'] = np.array([cl.v1(t) for t in cl.parameter_samples])
        v1 = pdt.glyph(orient='v1', factor=scale)
        plotter.add_mesh(v1, color='g')

        pdt['v2'] = np.array([cl.v2(t) for t in cl.parameter_samples])
        v2 = pdt.glyph(orient='v2', factor=scale)
        plotter.add_mesh(v2, color='b')

        if isinstance(cntrln, CenterlineNetwork) and hasattr(cl, 'children'):
            for cid in cl.children:
                plot_cl_pl(cntrln[cid])

    if isinstance(cntrln, Centerline):
        plot_cl_pl(cl=cntrln)

    elif isinstance(cntrln, CenterlineNetwork):
        for rid in cntrln.roots:
            plot_cl_pl(cntrln[rid])

    else:
        error_message(
            'The argument cntrln must be an instance of Centerline or CenterlineNetwork.')
        return

    if vmesh is not None:
        plotter.add_mesh(vmesh, opacity=0.5, color='w')

    if show:
        plotter.show()
#


def plot_open_boundary_ids(poly, plotter=None, show=True):
    """
    Detect open boundaries at a polydata and display them with the assigned ids.

    Arguments:
    -------------

        poly : pv.PolyData
            The polydata

        plotter : pv.Plotter
            A plotter object to add the plots on. Otherwise it is created.

        show : bool, opt
            Whether to plot or not at the end of the function.

    Returns:
    ----------
        plotter : pv.Plotter
            The potter object.
    """

    if plotter is None:
        plotter = pv.Plotter()

    if poly.is_manifold:
        warning_message('No boundary edges found.')
        return plotter

    boundaries = poly.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False)
    boundaries = boundaries.connectivity()

    plotter.add_mesh(poly, opacity=0.5)
    print('.' * 20)
    for i in np.unique(boundaries['RegionId']):
        boundary = boundaries.extract_points(boundaries['RegionId'] == i)
        plotter.add_point_labels(np.array(boundary.center), [str(i)])
        print(f'{i} : {boundary.center}')
    print('.' * 20)

    if show:
        plotter.show()
#
