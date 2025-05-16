from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyvista as pv

from ..._base import check_specific, filter_specific
from .cross_sections import CrossSectionScheme, get_cross_section

if TYPE_CHECKING:
    from ..vascular_encoding import VascularAnatomyEncoding
    from ..vessel_encoding import VesselAnatomyEncoding


class VesselMeshing(ABC):
    """
    An abstract class containing vessel meshing methods available by class inheritance
    inherited in VesselAnatomyEncoding.

    """

    def make_points_along_radius(
        self: VesselAnatomyEncoding, tau: int, theta: float, rho_res: int, diameter=False
    ) -> np.ndarray:
        """
        Get a sample of a diameter at a given slice.

        Returns a sample of n points at even values of the rho axis
        starting from given theta to theta+pi at distance height
        from the valve.

        Parameters
        ----------
        tau : float
            The distance from the valve for the slice to be returned.
            It must be in [0,1]
        theta : float
            The angle at which generate the diameter.
        rho_res : int
            Number of points to discretize the radius/diameter segment
        diameter : bool
            Default False. Whether to return a diameter instead of a radius segment.
            The diameter is taken from theta to theta+pi


        Returns
        -------
        slice : numpy.ndarray
            An array of 3D points with the diameter samples

        """

        vc_extremes = [[tau, theta, 1.0]]
        if diameter:
            if theta < np.pi:
                ang = 2 * np.pi - theta
            else:
                ang = theta - np.pi
            vc_extremes.append([tau, ang, 1.0])
        else:
            vc_extremes.append([tau, 0.0, 0.0])

        p0, p1 = [self.vcs_to_cartesian(t, th, r, rho_norm=True) for (t, th, r) in vc_extremes]

        return np.array([p0 * (1 - t) + p1 * t for t in np.linspace(0, 1, rho_res)]).reshape(
            rho_res, 3
        )

    def make_cross_section(
        self: VesselAnatomyEncoding,
        scheme: Literal["base", "ogrid", "cylindrical"],
        tau: float,
        theta_res: int,
        rho_res: int,
        **kwargs,
    ) -> CrossSectionScheme:
        """
        Generate a cross section of the vessel with the provided scheme and parameters.

        Parameters
        ----------
        scheme : {'base', 'ogrid', 'cylindrical'}, optional
            The discretization scheme to use
        tau: float,
            The centerline parameter of the cross section.
        theta_res : int
            Angular resolution.
        rho_res : int
            Radial resolution.
        **kwargs
            Scheme specific arguments such as the 'r' in ogrid, or the prismatic layers
            parameters can be passed as keyword arguments.

        Returns
        -------
        cs : CrossSectionScheme
            The generated cross section.
        """

        cs = get_cross_section(scheme=scheme, theta_res=theta_res, rho_res=rho_res, **kwargs)

        # Grid Points
        points, tau, theta, rho, rho_norm = self.vcs_to_cartesian(
            tau=tau, theta=cs["theta"], rho=cs["rho"], rho_norm=True, full_output=True
        )

        cs.points = points
        cs["tau"] = tau
        cs["theta"] = theta
        cs["rho"] = rho
        cs["rho_n"] = rho_norm

        return cs

    def make_ribbon(
        self: VesselAnatomyEncoding, theta: float, tau_res: int, rho_res: int
    ) -> pv.PolyData:
        """
        Make a triangulated ribbon section along the centerline.

        Parameters
        ----------
        theta : float
            The angle to start the ribbon it must be in [0, pi].
        tau_res : int
            The number of points to discretize the centerline.
        rho_res : int
            The number of discretizations in the radius axis.

        Returns
        -------
        poly : pyvista.PolyData
            A triangulated mesh with the generated ribbon section.

        """

        pts, triangles = [], []
        tau, rho, rho_n = [], [], []

        taus = np.linspace(self.centerline.t0, self.centerline.t1, tau_res)
        for i, t in enumerate(taus):
            pts.append(self.points_along_radius(tau=t, theta=theta, rho_res=rho_res))

            tau.append([t] * pts[-1].shape[0])
            c = self.centerline(t)
            rho.append(np.linalg.norm(pts[-1] - c, axis=1))
            rho_n.append(rho[-1] / self.radius(t, theta))

            if i > 0:
                for j in range(rho_res):
                    if j != rho_res - 1:
                        triangles.append(
                            [3, i * rho_res + j, (i - 1) * rho_res + j, (i - 1) * rho_res + j + 1]
                        )
                        triangles.append(
                            [3, i * rho_res + j, i * rho_res + j + 1, (i - 1) * rho_res + j + 1]
                        )

        pts = np.concatenate(pts)
        ribb = pv.PolyData(pts, triangles)
        ribb["tau"] = np.ravel(tau)
        ribb["theta"] = np.full((ribb.n_points,), fill_value=theta)
        ribb["rho"] = np.ravel(rho)
        ribb["rho_n"] = np.ravel(rho_n)

        return ribb

    def make_tube(
        self: VesselAnatomyEncoding,
        tau_res: int,
        theta_res: int,
        radius: float = 1,
        normalized=True,
    ) -> pv.PolyData:
        """
        Make a surface along centerline somehow parallel to the wall.

        Parameters
        ----------
        tau_res : int
            The number of points along the longitudinal axis
        theta_res : int
            The number of points along the radial axis.
        radius : float
            Either the absolute or a relative value.
        normalized : bool
            Defaulting to False. How to interpret radius.

        Returns
        -------
        tube : pv.PolyData
        """

        pts = []
        faces = []
        vessel_coord = {"tau": [], "theta": [], "rho": [], "rho_n": []}

        faces_block = []
        for j in range(theta_res):
            if j < theta_res - 1:
                faces_block.append([j, j + 1, theta_res + j + 1, theta_res + j])
            else:
                faces_block.append([j, 0, theta_res, theta_res + j])
        faces_block = np.array(faces_block, dtype=int)

        taus = np.linspace(self.centerline.t0, self.centerline.t1, tau_res)

        cs = CrossSectionScheme(
            theta_res=theta_res,
            rho_res=1,
        )

        for i, tau in enumerate(taus):
            points, ta, th, rh, rh_n = self.vcs_to_cartesian(
                tau=tau,
                theta=cs["theta"],
                rho=cs["rho"] * radius if normalized else radius,
                rho_norm=normalized,
                full_output=True,
            )

            pts.append(points)
            vessel_coord["tau"].append(ta)
            vessel_coord["theta"].append(th)
            vessel_coord["rho"].append(rh)
            vessel_coord["rho_n"].append(rh_n)

            if i < tau_res - 1:
                faces.append(faces_block + i * theta_res)

        pts = np.concatenate(pts)
        faces = np.vstack(faces)
        off = np.full(shape=(faces.shape[0], 1), fill_value=4)
        faces = np.hstack((off, faces), dtype=int)
        faces = np.ravel(faces)
        tube = pv.PolyData(pts, faces=faces.ravel())
        for name, vals in vessel_coord.items():
            tube[name] = np.array(vals).ravel()

        return tube

    def make_volume_mesh(
        self: VesselAnatomyEncoding,
        tau_res: int,
        theta_res: int = None,
        rho_res: int = None,
        scheme: Literal["ogrid", "cylindrical"] = None,
        cs: CrossSectionScheme = None,
        **kwargs,
    ) -> pv.UnstructuredGrid:
        """
        Make a volumetric mesh.

        Either the parameters to build a cross section scheme or a cross section scheme objects must
        be provided. If a cross section is provided, its radius must have been set to 1.

        Parameters
        ----------
        tau_res : int
            The number of longitudinal divisions or cross sections
        theta_res : int
            The number of angular divisions. The meaning changes depending on the cross section
            scheme chosen.
        rho_res : int
            The number of radial divisions. The meaning changes depending on the cross section
            scheme chosen.
        scheme : Literal['ogrid', 'cylindrical']
            The cross section scheme to use
        cs : CrossSectionScheme, optional
            A cross section scheme object already built with radius 1, by default None. If passed
            previous arguments, such as theta_res or rho_res are ignored.

        **kwargs
            Cross section scheme parameters can be passed as keyword arguments, for example, if the
            argument scheme == 'ogrid', the parameter r=0.9 can be passed. Prismatic layers can also
            be passed using kwargs.

        Returns
        -------
        grid : pv.UnstructuredGrid
            The volumetric grid built.

        """

        taus = np.linspace(self.centerline.t0, self.centerline.t1, tau_res)

        points = []
        cells = {}
        cell_types = {}
        vessel_coord = {"tau": [], "theta": [], "rho": [], "rho_n": []}

        if cs is None:
            cs = get_cross_section(theta_res=theta_res, rho_res=rho_res, scheme=scheme, **kwargs)

        # The block of cells between slices
        cells_block = {}
        cs_n_points = cs.n_points
        for face in cs.irregular_faces:
            n = 2 * len(face)
            ncell = face.tolist() + (cs_n_points + face).tolist()
            if n not in cells_block:
                cells_block[n] = []
            cells_block[n].append(ncell)

        # Initialize and validate the global cells
        for n, ncells in cells_block.items():
            cells_block[n] = np.array(ncells, dtype=int)
            cells[n] = []
            if n == 6:
                cell_types[n] = pv.CellType.WEDGE
            elif n == 8:
                cell_types[n] = pv.CellType.HEXAHEDRON
            else:
                print(
                    f"WARNING: The provided CrossSectionScheme contains unsupported face with {n // 2} faces."
                    + "Currently supporting {3, 4} vertex per face."
                )

        for i, tau in enumerate(taus):
            # Grid Points
            pts, tau, theta, rho, rho_norm = self.vcs_to_cartesian(
                tau=tau, theta=cs["theta"], rho=cs["rho"], rho_norm=True, full_output=True
            )
            points.append(pts)

            vessel_coord["tau"].append(tau)
            # vessel_coord["tau"].append(np.full(shape=(cs_n_points), fill_value=tau))
            vessel_coord["theta"].append(theta)
            # vessel_coord["theta"].append(theta)
            vessel_coord["rho"].append(rho)
            # vessel_coord["rho"].append(rho)
            vessel_coord["rho_n"].append(rho_norm)
            # vessel_coord["rho_norm"].append(rho_norm)

            # Grid Topology
            if i < tau_res - 1:
                for n, ncells in cells_block.items():
                    ncells_i = i * cs_n_points + ncells
                    cells[n].append(ncells_i)

        # Globals
        gcells = []
        gcell_types = []
        for n, ncells in cells.items():
            cells[n] = np.vstack(ncells)
            aux = np.full(shape=(cells[n].shape[0], 1), fill_value=1, dtype=int)
            ct = (aux * cell_types[n]).astype("uint")
            off = aux * n
            cells[n] = np.hstack((off, cells[n]))

            gcells.append(cells[n].ravel())
            gcell_types.append(ct)

        cells = np.concatenate(gcells)
        celltypes = np.concatenate(gcell_types, dtype="uint")
        points = np.vstack(points)
        grid = pv.UnstructuredGrid(cells, celltypes, points)

        for name, field in vessel_coord.items():
            grid[name] = np.ravel(field)

        return grid


class VascularMeshing(ABC):
    """
    An abstract class containing vascular meshing methods available by class inheritance
    inherited in VesselAnatomyEncoding.

    """

    def make_cross_section(
        self: VascularAnatomyEncoding,
        scheme: Literal["base", "ogrid", "cylindrical"],
        tau: float,
        theta_res: int,
        rho_res: int,
        bid: str | None = None,
        **kwargs,
    ) -> CrossSectionScheme | pv.MultiBlock:
        """
        Generate a cross sections of the vessels with the provided scheme and parameters.

        Parameters
        ----------
        scheme : {'base', 'ogrid', 'cylindrical'}, optional
            The discretization scheme to use
        tau: float,
            The centerline parameter of the cross section.
        theta_res : int
            Angular resolution.
        rho_res : int
            Radial resolution.
        bid : str | None
            Default None. If passed, the cross section is generated for the given branch, otherwise
            the cross section is generated at each branch.
        **kwargs
            Keyword arguments can be used for two purposes:
            - Set cross section scheme parameters. For example, if the argument scheme == 'ogrid',
            the parameter r=0.9 can be passed as well as prismatic layers parameters.
            - Set branch specific parameters. For example, if you have a vascular domain with two
            branches "B0" and "B1", whose radial function notably differ, you may pass the
            arguments: B0={theta_res=60}, B1={theta_res=20}.

            Both this approaches can be combined to obtain completely different branch specific
            grids

        Returns
        -------
         : CrossSectionScheme | pv.MultiBlock
            The generated cross section.
        """

        if bid is not None and bid in self:
            return self[bid].make_cross_section(
                scheme=scheme, tau=tau, theta_res=theta_res, rho_res=rho_res, **kwargs
            )

        return pv.MultiBlock(
            {
                bid: self[bid].make_cross_section(
                    scheme=check_specific(kwargs, bid, "scheme", scheme),
                    tau=check_specific(kwargs, bid, "tau", tau),
                    theta_res=check_specific(kwargs, bid, "theta_res", theta_res),
                    rho_res=check_specific(kwargs, bid, "rho_res", rho_res),
                    **filter_specific(kwargs, bid),
                )
                for bid in self
            }
        )

    def make_ribbon(
        self: VascularAnatomyEncoding,
        theta: float,
        tau_res: int,
        rho_res: int,
        bid: str | None = None,
        **kwargs,
    ) -> pv.PolyData | pv.MultiBlock:
        """
        Make a triangulated ribbon section along the centerline.

        Parameters
        ----------
        theta : float
            The angle to start the ribbon it must be in [0, pi].
        tau_res : int
            The number of points to discretize the centerline.
        rho_res : int
            The number of discretizations in the radius axis.
        bid : str | None
            Default None. If passed, the cross section is generated for the given branch, otherwise
            the cross section is generated at each branch.
        **kwargs
            Keyword arguments can be used to set branch specific parameters. For example, if you
            have a vascular domain with two branches "B0" and "B1", whose length notably differ, you
            may pass the arguments: B0={tau_res=100}, B1={tau_res=20}.

        Returns
        -------
        : pyvista.PolyData | pv.MultiBlock
            A triangulated mesh with the generated ribbon section(s).

        """

        if bid is not None and bid in self:
            return self[bid].make_ribbon(theta=theta, tau_res=tau_res, rho_res=rho_res)

        return pv.MultiBlock(
            {
                bid: self[bid].make_ribbon(
                    theta=check_specific(kwargs, bid, "theta", theta),
                    tau_res=check_specific(kwargs, bid, "tau_res", tau_res),
                    rho_res=check_specific(kwargs, bid, "rho_res", rho_res),
                )
                for bid in self
            }
        )

    def make_tube(
        self: VascularAnatomyEncoding,
        tau_res: int,
        theta_res: int,
        radius: float = 1,
        normalized=True,
        bid: str | None = None,
        **kwargs,
    ) -> pv.PolyData | pv.MultiBlock:
        """
        Make a surface along centerline somehow parallel to the wall.

        Parameters
        ----------
        tau_res : int
            The number of points along the longitudinal axis
        theta_res : int
            The number of points along the radial axis.
        radius : float
            Either the absolute or a relative value.
        normalized : bool
            Defaulting to False. How to interpret radius.
        bid : str | None
            Default None. If passed, the cross section is generated for the given branch, otherwise
            the cross section is generated at each branch.
        **kwargs
            Keyword arguments can be used to set branch specific parameters. For example, if you
            have a vascular domain with two branches "B0" and "B1", whose length notably differ,
            you may pass the arguments: B0={tau_res=100}, B1={tau_res=20}.

        Returns
        -------
        : pv.PolyData | pv.MultiBlock
        """

        if bid is not None and bid in self:
            return self[bid].make_tube(
                tau_res=tau_res, theta_res=theta_res, radius=radius, normalized=normalized
            )

        return pv.MultiBlock(
            {
                bid: self[bid].make_tube(
                    tau_res=check_specific(kwargs, bid, "tau_res", tau_res),
                    theta_res=check_specific(kwargs, bid, "theta_res", theta_res),
                    radius=check_specific(kwargs, bid, "radius", radius),
                    normalized=normalized,
                )
                for bid in self
            }
        )

    def make_volume_mesh(
        self: VascularAnatomyEncoding,
        tau_res: int,
        theta_res: int = None,
        rho_res: int = None,
        scheme: Literal["ogrid", "cylindrical"] = None,
        cs: CrossSectionScheme = None,
        bid: str | None = None,
        **kwargs,
    ) -> pv.UnstructuredGrid | pv.MultiBlock:
        """
        Make a volumetric mesh.

        Either the parameters to build a cross section scheme or a cross section scheme objects must
        be provided. If a cross section is provided, its radius must have been set to 1.

        Parameters
        ----------
        tau_res : int
            The number of longitudinal divisions or cross sections
        theta_res : int
            The number of angular divisions. The meaning changes depending on the cross section
            scheme chosen.
        rho_res : int
            The number of radial divisions. The meaning changes depending on the cross section
            scheme chosen.
        scheme : Literal['ogrid', 'cylindrical']
            The cross section scheme to use
        cs : CrossSectionScheme, optional
            A cross section scheme object already built with radius 1, by default None. If passed
            previous arguments, such as theta_res or rho_res are ignored.
        bid : str | None
            Default None. If passed, the cross section is generated for the given branch, otherwise
            the cross section is generated at each branch.

        **kwargs
            Keyword arguments can be used for two purposes:
            - Set cross section scheme parameters. For example, if the argument scheme == 'ogrid',
            the parameter r=0.9 can be passed as well as prismatic layers parameters.
            - Set branch specific parameters. For example, if you have a vascular domain with two
            branches "B0" and "B1", whose length notably differ, you may pass the arguments:
            B0={tau_res=100}, B1={tau_res=20}.

            Both this approaches can be combined to obtain completely different branch specific
            grids.

        Returns
        -------
        : pv.UnstructuredGrid | pv.MultiBlock
            The volumetric grid built.

        """

        if bid is not None and bid in self:
            return self[bid].make_volume_mesh(
                bid=bid,
                tau_res=tau_res,
                theta_res=theta_res,
                rho_res=rho_res,
                scheme=scheme,
                cs=cs,
                **kwargs,
            )

        return pv.MultiBlock(
            {
                bid: self[bid].make_volume_mesh(
                    bid=bid,
                    tau_res=check_specific(kwargs, bid, "tau_res", tau_res),
                    theta_res=check_specific(kwargs, bid, "theta_res", theta_res),
                    rho_res=check_specific(kwargs, bid, "rho_res", rho_res),
                    scheme=check_specific(kwargs, bid, "scheme", scheme),
                    cs=cs,
                    **filter_specific(
                        kwargs,
                        bid,
                        exclude=["tau_res", "theta_res", "rho_res", "scheme"],
                    ),
                )
                for bid in self
            }
        )
