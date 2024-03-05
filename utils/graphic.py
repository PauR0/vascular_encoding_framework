
import numpy as np
import pyvista as pv

import messages as msg
from centerline import Centerline, CenterlineNetwork



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
        pdt.lines = np.array([[2, i, i+1] for i in range(cl.n_samples-1)], dtype=int)
        plotter.add_mesh(pdt, color='k', render_lines_as_tubes=True, line_width=5)

        pdt['tangent'] = np.array([cl.get_tangent(t) for t in cl.parameter_samples])
        tgts = pdt.glyph(orient="tangent", factor=scale)
        plotter.add_mesh(tgts, color='r')

        pdt['v1']      = np.array([cl.v1(t) for t in cl.parameter_samples])
        v1 = pdt.glyph(orient="v1", factor=scale)
        plotter.add_mesh(v1, color='g')

        pdt['v2']      = np.array([cl.v2(t) for t in cl.parameter_samples])
        v2 = pdt.glyph(orient="v2", factor=scale)
        plotter.add_mesh(v2, color='b')

        if hasattr(cl, 'children'):
            for cid in cl.children:
                plot_cl_pl(cntrln[cid])

    if isinstance(cntrln, Centerline):
        plot_cl_pl(cl=cntrln)

    elif isinstance(cntrln, CenterlineNetwork):
        for rid in cntrln.roots:
            plot_cl_pl(cntrln[rid])

    else:
        msg.error_message("The argument cntrln must be an instance of Centerline or CenterlineNetwork.")

    if vmesh is not None:
        plotter.add_mesh(vmesh, opacity=0.5, color='w')

    if show:
        plotter.show()
#
