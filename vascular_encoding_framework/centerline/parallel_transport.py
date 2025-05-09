from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import make_lsq_spline
from scipy.spatial.transform import Rotation

from .._base import SpatialObject
from ..splines import UniSpline
from ..utils.spatial import normalize

if TYPE_CHECKING:
    from .curve import Curve


class ParallelTransport(UniSpline, SpatialObject):
    """Parallel Transport class."""

    def __init__(self) -> None:
        super().__init__()

        # The initial vector to be transported.
        self.v0: np.ndarray = None

    @staticmethod
    def compute_parallel_transport_along_curve(curve: Curve, v0: np.ndarray) -> ParallelTransport:
        """
        Build the parallel transport of a given vector v0 along a curve object.

        The parallel transported vector is interpolated using the parameters of the curve.

        It is build according to the algorithm from:
            https://legacy.cs.indiana.edu/ftp/techreports/TR425.pdf

        Briefly described, given a initial vector, orthogonal to the tangent of a curve. A parallel
        transport of given vector can be obtained by applying the rotation required by the curvature
        to remain normal.

        Parameters
        ----------
        curve : Curve,
            The input curve along which the parallel transport will be computed.
        v0 : np.ndarray (3,)
            The initial vector to be transported.

        """

        # Build the Parallel and inherit spline curve parameters.
        pt = ParallelTransport()
        pt.set_parameters(
            v0=v0, t0=curve.t0, t1=curve.t1, k=curve.k, knots=curve.knots, n_knots=curve.n_knots
        )
        param_samples = np.linspace(pt.t0, pt.t1, num=curve.n_samples)

        tg = curve.get_tangent(curve.t0)
        V = []
        for t in param_samples:
            tg_next = curve.get_tangent(t)
            v0 = ParallelTransport.parallel_rotation(t0=tg, t1=tg_next, v=v0)
            V.append(v0)
            tg = tg_next

        # Build
        V = np.array(V)
        pt.set_parameters(
            build=True, coeffs=make_lsq_spline(x=param_samples, y=V, t=pt.knots, k=pt.k).c
        )

        return pt

    @staticmethod
    def parallel_rotation(t0: np.ndarray, t1: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Apply to v the rotation that takes t0 to t1.

        Parameters
        ----------
        t0, t1, v : np.ndarray (3,)

        Returns
        -------
        : np.ndarray
        """
        t0dott1 = np.clip(t0.dot(t1), -1.0, 1.0)
        rot_vec = normalize(np.cross(t0, t1)) * np.arccos(t0dott1)
        R = Rotation.from_rotvec(rot_vec)
        return R.apply(v)

    def translate(self, t):
        """
        Has no effect on parallel transport.

        Parameters
        ----------
        t : np.ndarray (3,)
            Ignored.
        """
        ...

    def scale(self, s):
        """
        Has no effect on parallel transport.

        Parameters
        ----------
        s : float
            Ignored.
        """
        ...

    def rotate(self, r):
        """
        Rotate the Parallel Transport.

        Rotation is applied to the coefficients of the transport splines.

        Parameters
        ----------
        r : np.ndarray (3, 3)
            The rotation matrix.
        update : bool, optional
            Default True. Whether to rebuild the splines after the rotation.
        """

        # ensure normality of the rotation matrix columns
        r /= np.linalg.norm(r, axis=0)

        if self.coeffs is not None:
            self.coeffs = (r @ self.coeffs.T).T
            self.build()
