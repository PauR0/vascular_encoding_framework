from __future__ import annotations

from typing import Literal

import numpy as np
import pyvista as pv

from ...messages import error_message, info_message
from ...utils.spatial import get_theta_coord, sort_glob_ids_by_angle


def compute_rho_discretization(
    rho_res: int,
    r0: float = 0.0,
    r1: float = 1.0,
    n_layers: int | None = None,
    growth_rate: float | None = None,
    min_percentage: float | None = None,
):
    """
    Compute the radial discretization of a cross section including uniform and prismatic layers.

    The argument rho_res is used to determine equally spaced values, it has to be greater than 1.
    Instead of the [0,1] default values a different radial domain can be passed with r0 and r1.
    If n_layers is not None, n_layers prismatic layers are added inwards, growing at the provided
    growth_rate and from the provided minimum percentage.

    Parameters
    ----------
    rho_res : int,
        The amount of equally levels of discretization

    r0,r1 : float, optional
        Default 0.0 and 1.0 respectively. The radial domain to be discretized.

    n_layers : int,
        Defaulting to None. The amount of layers.

    growth_rate : float,
        Defaulting to None. The rate between two consecutive layers.

    min_percentage : float,
        Defaulting to None. The percentage of radius covered by the thinnest layer expressed
        from 0 to 1.

    Returns
    -------
    rho_discr : np.ndarray
        The radial discretization expressed in normalized values.

    """

    assert rho_res > 0, f"Wrong Value for rho_res: {rho_res}. It must be a positive integer"

    prism_layers = []
    if n_layers is not None:
        aux = r1
        for n in range(n_layers):
            prism_layers.append(aux - min_percentage * growth_rate**n)
            aux = prism_layers[-1]
        prism_layers.reverse()

        if prism_layers[0] < 0:
            error_message(
                "Too many layers to fit for the given growth rate. The last surpasses the "
                + f"centerline with a percentage of {prism_layers[0]}...."
            )
            return None

        max_layer_thickness = prism_layers[1] - prism_layers[0]
        transition_ratio = max_layer_thickness / (prism_layers[0] / rho_res)
        if transition_ratio < 0.8 or transition_ratio > 1.5:
            info_message(
                "The transition between the equally spaced and prismatic layers might be too "
                f"uneven. The ratio between thickness is {transition_ratio} "
            )

    prism_layers += [r1]

    if rho_res == 1:
        return np.array(prism_layers)

    rho_discr = np.concatenate([np.linspace(r0, prism_layers[0], rho_res)[:-1], prism_layers])

    return rho_discr


class CrossSectionScheme(pv.PolyData):
    """
    Base class for cross section discretization schemes.

    Although it is possible, this class is not meant to be instantiated, it ignores rho_res and
    prismatic layer parameters.
    """

    def __init__(
        self,
        theta_res: int,
        rho_res: int,
        radius: float = 1.0,
        n_layers: int = 0,
        growth_rate: float | None = None,
        min_percentage: float | None = None,
        origin: np.ndarray = None,
        v1: np.ndarray = None,
        v2: np.ndarray = None,
        **kwargs,
    ):
        """
        Construct CrossSectionScheme object.

        Base class of the cross section discretization schemes.

        Parameters
        ----------
        theta_res : int
            Amount of divisions in the angular axis.
        rho_res : int
            Amount of divisions in the radial axis.
        radius : float, optional
            Default 1. Radius of the cross section.
        n_layers : int,
            Defaulting to 0. The amount of layers.
        growth_rate : float,
            Defaulting to None. The rate between two consecutive layers.
        min_percentage : float,
            Defaulting to None. The percentage of radius covered by the thinnest layer expressed
            from 0 to 1.
        origin : np.ndarray, optional
            Default (0,0,0). The origin of the cross section.
        v1 : np.ndarray, optional
            Default (1,0,0). The first axis of the plane containing the cross section. Must be
            orthonormal to v2
        v2 : np.ndarray, optional
            Default (0,1,0). The second axis of the plane containing the cross section. Must be
            orthonormal to v2
        **kwargs
            Each children classes are expected to have some extra parameters.
        """

        super().__init__()

        # Angular and radial resolution
        self.theta_res: int = theta_res

        if rho_res <= 0:
            raise ValueError(f"Radial resolution must be strictly positive 0. Given {rho_res}")
        self.rho_res: int = rho_res

        # Prism layer parameters
        self.n_layers: int | None = n_layers
        self.growth_rate: float | None = growth_rate
        self.min_percentage: float | None = min_percentage

        # Geometric parameter for the cross section and its plane.
        self.radius: float = radius
        self.origin: np.ndarray = self._validate_array(origin, (3,), np.array([0, 0, 0]), "origin")
        self.v1: np.ndarray = self._validate_array(v1, (3,), np.array([1, 0, 0]), "v1")
        self.v2: np.ndarray = self._validate_array(v2, (3,), np.array([0, 1, 0]), "v2")

        self.build()
        self.compute_polar_coordinates()

    def _validate_array(self, in_array: np.ndarray, shape: tuple, default: np.ndarray, name: str):
        """
        Validate the shape of a provided array.

        Array reshaping is attempted.

        Parameters
        ----------
        in_array: np.ndarray
            The input array.

        shape : tuple
            The expected shape

        default : np.ndarray
            A default value in case the array passed is None.

        name: str, optional
            The name of the variable attempted to be set.

        Returns
        -------
        out_array : np.ndarray
            The proper array, either the original one in the right shape or the default.

        Raises
        ------
        ValueError
            Whenever the input array is not in the right format or if in_array is not an array.

        """

        if in_array is None:
            return default

        if not isinstance(in_array, np.ndarray):
            raise ValueError(f"Wrong type for {name} attribute. Numpy array {shape} was expected.")

        if in_array.shape != shape:
            try:
                return in_array.reshape(shape)
            except ValueError:
                raise ValueError(
                    f"Provided array for {name} attribute has shape {shape}. Expected is {shape}"
                )
        return in_array

    def sample_circumference(
        self, n: int | np.ndarray, r: float = 1.0, aoff: float = 0.0
    ) -> np.ndarray:
        """
        Sample n points along a circumference of given radius r at the cross section.

        Parameters
        ----------
        n : int or np.ndarray (N,)
            If integer it is used as the number of points to be generated. If a np.ndarray it is
            directly used as the radians partition.

        r: float, optional,
            Default 1. The radius of the circumference.

        aoff : float, optional
            Default 0.0. An angle offset to start the generation. The parametrization employed goes
            from 0 to 2pi, the offset is added at both extremes.

        Returns
        -------
        points : numpy.array (theta_res, 3)
            An array of 3D points with the slice samples

        """

        if isinstance(n, int):
            theta = np.linspace(0 + aoff, 2 * np.pi + aoff, n + 1)[:-1]
        elif isinstance(n, np.ndarray) and n.ndim == 1:
            theta = n
        else:
            error_message(f"Wrong value for theta_res argument {n}. Expecting int or 1D array.")
            return

        points = (
            self.origin.reshape(3, 1)
            + self.v1.reshape(3, 1) * r * np.cos(theta)
            + self.v2.reshape(3, 1) * r * np.sin(theta)
        ).T

        return points

    def get_angular_coordinates(self, points: np.ndarray, deg: bool = False) -> np.ndarray:
        """
        Compute the angular coordinates of a list of points in the cross section reference system.

        The angular domain is [0, 2pi]

        Parameters
        ----------
        point : np.ndarray (3,) or (N,3)
            The point(s) to compute the angular coordinates

        deg : bool, optional
            Default False. Whether to return the result in degrees rather than radians.

        Returns
        -------
            ang: float or np.ndarray (N)
        """

        return get_theta_coord(points=points, c=self.origin, v1=self.v1, v2=self.v2, deg=deg)

    def compute_polar_coordinates(self):
        """
        Compute the polar coordinates of the points in the cross section.

        The angular domain is [0, 2pi]
        """

        self["theta"] = self.get_angular_coordinates(points=self.points)
        self["rho"] = np.linalg.norm(self.points - self.origin, axis=1)

    def build(self) -> CrossSectionScheme:
        """
        Build the cross section discretization scheme.

        This method is intended to be overwritten by child classes.
        """

        self.points = self.sample_circumference(n=self.theta_res, r=self.radius, aoff=False)

        self.faces = np.array([self.theta_res] + list(range(self.theta_res)))
        self.triangulate(inplace=True)

        return self


class CylindricalCrossSection(CrossSectionScheme):
    """A cylindrical cross section discretization scheme."""

    def __init__(
        self,
        theta_res: int,
        rho_res: int,
        radius: float = 1.0,
        n_layers: int = 0,
        growth_rate: float | None = None,
        min_percentage: float | None = None,
        twist: bool = False,
        origin=None,
        v1=None,
        v2=None,
    ):
        """
        Construct a cylindrical cross section discretization scheme.

        Parameters
        ----------
        theta_res : int
            Amount of divisions in the angular axis.
        rho_res : int
            Amount of divisions in the radial axis.
        radius : float, optional
            Default 1. Radius of the cross section.
        n_layers : int,
            Defaulting to 0. The amount of layers.
        growth_rate : float,
            Defaulting to None. The rate between two consecutive layers.
        min_percentage : float,
            Defaulting to None. The percentage of radius covered by the thinnest layer expressed
            from 0 to 1.
        twist: bool, optional
            Default False. Whether to add an angular offset of half the angular distance every
            second concentric circumference. If rho_res is smaller than 3, it has no effect.
        origin : np.ndarray, optional
            Default (0,0,0). The origin of the cross section.
        v1 : np.ndarray, optional
            Default (1,0,0). The first axis of the plane containing the cross section. Must be
            orthonormal to v2.
        v2 : np.ndarray, optional
            Default (0,1,0). The second axis of the plane containing the cross section. Must be
            orthonormal to v1.
        """

        self.twist: bool = twist

        super().__init__(
            theta_res=theta_res,
            rho_res=rho_res,
            radius=radius,
            n_layers=n_layers,
            growth_rate=growth_rate,
            min_percentage=min_percentage,
            origin=origin,
            v1=v1,
            v2=v2,
        )

    def build(self) -> CylindricalCrossSection:
        """
        Make a cross section with cylindrical discretization scheme.

        Returns
        -------
            self : CylindricalCrossSection
                A triangulated mesh with the generated cross section. A point scalar field with the vcs
                coordinate is can be accessed by cs['tau'], cs['theta'], cs['rho'], cs['rho_n'].
        """

        rhos = compute_rho_discretization(
            self.rho_res,
            r0=0.0,
            r1=self.radius,
            n_layers=self.n_layers,
            growth_rate=self.growth_rate,
            min_percentage=self.min_percentage,
        )

        pts = []
        faces = []

        for i, rho in enumerate(rhos):
            if i == 0:
                pp = self.origin.copy()[None, :]

            elif i == 1:
                pp = self.sample_circumference(n=self.theta_res, r=rho)
                for j in range(self.theta_res):
                    if j < self.theta_res - 1:
                        faces.append([3, 1 + j, 0, 1 + j + 1])
                    else:
                        faces.append([3, 1 + j, 0, 1])
            else:
                aoff = 2 * np.pi / (2 * self.theta_res) if self.twist and i % 2 else 0
                pp = self.sample_circumference(n=self.theta_res, r=rho, aoff=aoff)

                for j in range(self.theta_res):
                    if j < self.theta_res - 1:
                        faces.append(
                            [
                                4,
                                1 + (i - 1) * self.theta_res + j,
                                1 + (i - 2) * self.theta_res + j,
                                1 + (i - 2) * self.theta_res + j + 1,
                                1 + (i - 1) * self.theta_res + j + 1,
                            ]
                        )
                    else:
                        faces.append(
                            [
                                4,
                                1 + (i - 1) * self.theta_res + j,
                                1 + (i - 2) * self.theta_res + j,
                                1 + (i - 2) * self.theta_res,
                                1 + (i - 1) * self.theta_res,
                            ]
                        )

            pts.append(pp)

        self.points = np.concatenate(pts)
        self.faces = np.hstack(faces)

        return self


class OGridCrossSection(CrossSectionScheme):
    """O-Grid cross section discretization scheme."""

    def __init__(
        self,
        theta_res: int,
        rho_res: int,
        r: float,
        radius: float = 1.0,
        n_layers: int = 0,
        growth_rate: float | None = None,
        min_percentage: float | None = None,
        origin: np.ndarray = None,
        v1: np.ndarray = None,
        v2: np.ndarray = None,
    ):
        """
        Construct o-grid CrossSectionScheme.

        Base class of the cross section discretization schemes.

        Parameters
        ----------
        theta_res : int
            Amount of divisions in the angular axis.
        rho_res : int
            Amount of divisions in the radial axis.
        r : float
            The proportion of radius covered by half diagonal of the o-grid inner square.
        radius : float, optional
            Default 1. Radius of the cross section.
        n_layers : int,
            Defaulting to 0. The amount of layers.
        growth_rate : float,
            Defaulting to None. The rate between two consecutive layers.
        min_percentage : float,
            Defaulting to None. The percentage of radius covered by the thinnest layer expressed
            from 0 to 1.
        origin : np.ndarray, optional
            Default (0,0,0). The origin of the cross section.
        v1 : np.ndarray, optional
            Default (1,0,0). The first axis of the plane containing the cross section. Must be
            orthonormal to v2.
        v2 : np.ndarray, optional
            Default (0,1,0). The second axis of the plane containing the cross section. Must be
            orthonormal to v1.
        """

        if theta_res % 4 != 0:
            raise ValueError(f"Angular resolution has to be a multiple of 4. Given {theta_res}")

        if rho_res <= 0:
            raise ValueError(f"Radial resolution must be strictly positive 0. Given {rho_res}")

        if r <= 0 or r >= 1:
            raise ValueError(f"The argument r must be in ]0,1[. Given is {r}.")

        self.r: float = r

        super().__init__(
            theta_res=theta_res,
            rho_res=rho_res,
            radius=radius,
            n_layers=n_layers,
            growth_rate=growth_rate,
            min_percentage=min_percentage,
            origin=origin,
            v1=v1,
            v2=v2,
        )

    @staticmethod
    def make_rectangle_grid(vertex: np.ndarray, n: int) -> pv.PolyData:
        """
        Build a rectangle polydata discretized in n points per size.

        Let c and v1, v2 be a reference system for a plane in the space. Then, given 4 vertex a number
        of points per side

        Parameters
        ----------
        vertex : np.ndarray (4,3)
            The corners of the rectangle ordered by angle.
        n : int
            The amount of points per size for the discretization.

        Returns
        -------
        rect : pv.PolyData
            The inner rectangle grid.
        """

        d = np.linspace(0, 1, n)
        u1 = vertex[1] - vertex[0]
        u2 = vertex[3] - vertex[0]

        bound_ids = []
        pts, faces = [], []
        for i in range(n):
            for j in range(n):
                pts.append(vertex[0] + d[i] * u1 + d[j] * u2)
                if j <= n - 2 and i <= n - 2:
                    faces.append(
                        [4, i * n + j, i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j]
                    )
                if i * j == 0 or (i - n + 1) * (j - n + 1) == 0:
                    bound_ids.append(i * n + j)
        pts, faces = np.array(pts), np.concatenate(faces)

        rect = pv.PolyData(pts, faces)
        rect["outer points"] = np.zeros((rect.n_points))
        rect["outer points"][bound_ids] = 1

        return rect

    def build(self) -> OGridCrossSection:
        """Generate the cross section with the o-grid discretization scheme."""

        in_r = self.r * self.radius

        sq_corners = self.sample_circumference(n=4, r=in_r, aoff=np.pi / 4)

        inner_rect = self.make_rectangle_grid(sq_corners, n=(self.theta_res // 4) + 1)

        bound_ids = inner_rect["outer points"].nonzero()[0]

        bound_ang_gids = sort_glob_ids_by_angle(
            bound_ids, inner_rect.points[bound_ids], c=self.origin, v1=self.v1, v2=self.v2
        )

        glob_pts = inner_rect.points.tolist()
        gi = inner_rect.n_points
        faces = []

        rhos = compute_rho_discretization(
            self.rho_res + 1,  # +1 Due to the outer edge of the inner square
            r0=in_r,
            r1=self.radius,
            n_layers=self.n_layers,
            growth_rate=self.growth_rate,
            min_percentage=self.min_percentage,
        )

        # The outermost points of the inner rectangle
        out_rect_pts = inner_rect.points[bound_ang_gids]
        angs = self.get_angular_coordinates(out_rect_pts)
        # Their projection to the outermost rectangle
        out_circ_pts = self.sample_circumference(n=angs, r=self.radius)

        for i, rho in enumerate(rhos):
            # Connect the sq with the first circ.
            if i == 0:
                for j in range(self.theta_res):
                    if j < self.theta_res - 1:
                        faces.append(
                            [4, bound_ang_gids[j], bound_ang_gids[j + 1], gi + j + 1, gi + j]
                        )
                    else:
                        faces.append([4, bound_ang_gids[-1], bound_ang_gids[0], gi, gi + j])
            else:
                # Parameterization from square to outer circ -> [0,1]
                p = (rho - in_r) / (self.radius * (1 - self.r))
                pts = out_rect_pts * (1 - p) + out_circ_pts * p
                glob_pts += pts.tolist()
                if i < rhos.size - 1:
                    for j in range(self.theta_res):
                        if j <= self.theta_res - 2 and i <= rhos.size - 2:
                            faces.append(
                                [
                                    4,
                                    gi + (i - 1) * self.theta_res + j,
                                    gi + (i - 1) * self.theta_res + j + 1,
                                    gi + i * self.theta_res + j + 1,
                                    gi + i * self.theta_res + j,
                                ]
                            )

                        elif j == self.theta_res - 1:
                            faces.append(
                                [
                                    4,
                                    gi + (i - 1) * self.theta_res + j,
                                    gi + (i - 1) * self.theta_res,
                                    gi + i * self.theta_res,
                                    gi + i * self.theta_res + j,
                                ]
                            )

        self.points = np.array(glob_pts)
        self.faces = np.concatenate([inner_rect.faces, np.concatenate(faces)])

        return self


def get_cross_section(
    scheme: Literal["base", "ogrid", "cylindrical"], theta_res: int, rho_res: int, **kwargs
) -> CrossSectionScheme:
    """
    Generate a cross section with given parameters.

    Scheme-specific and prismatic layers parameters can be provided via kwargs.

    Parameters
    ----------
    scheme : {'base', 'ogrid', 'cylindrical'}, optional
        The discretization scheme to use
    theta_res : int
        Angular resolution.
    rho_res : int
        Radiaul resolution.
    **kwargs
        Scheme specific arguments such as the 'r' in ogrid, or the prismatic layers
        parameters can be passed as kewyword arguments.

    Returns
    -------
    cs : CrossSectionScheme
        The generated cross section.
    """

    if scheme == "base":
        cs = CrossSectionScheme(
            theta_res=theta_res,
            rho_res=rho_res,
        )

    elif scheme == "ogrid":
        cs = OGridCrossSection(
            theta_res=theta_res,
            rho_res=rho_res,
            r=kwargs.get("r", None),
            n_layers=kwargs.get("n_layers", None),
            growth_rate=kwargs.get("growth_rate", None),
            min_percentage=kwargs.get("min_percentage", None),
        )

    elif scheme == "cylindrical":
        cs = CylindricalCrossSection(
            theta_res=theta_res,
            rho_res=rho_res,
            n_layers=kwargs.get("n_layers", None),
            growth_rate=kwargs.get("growth_rate", None),
            min_percentage=kwargs.get("min_percentage", None),
            twist=kwargs.get("twist", None),
        )

    else:
        raise ValueError(
            f"Wrong value for scheme argument ({scheme})."
            "Available options are {'ogrid', 'cylindrical'}"
        )

    return cs
