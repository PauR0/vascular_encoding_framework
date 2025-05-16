from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pyvista as pv
from scipy.optimize import minimize_scalar

from .._base import Encoding, Node, SpatialObject, attribute_checker, broadcast_kwargs, is_numeric
from ..centerline import Centerline
from ..messages import error_message
from ..utils.spatial import normalize, radians_to_degrees
from .radius import Radius
from .remesh import VesselMeshing


class VesselAnatomyEncoding(Node, Encoding, VesselMeshing, SpatialObject):
    """The class for encoding a single branch vessel."""

    def __init__(self):
        Node.__init__(self=self)

        self.centerline: Centerline = None
        self.radius: Radius = None

        self._hyperparameters = ["centerline", "radius"]

    def set_data(self, **kwargs):
        """Set attributes using kwargs and the setattr function."""

        if "centerline" in kwargs:
            self.set_centerline(cl=kwargs["centerline"])
            kwargs.pop("centerline")

        return super().set_data(**kwargs)

    def set_centerline(self, cl: Centerline):
        """
        Set the centerline attribute.

        > Note that the VesselAnatomyEncoding object inherits the node
        attributes from the centerline, in addition if tau_joint is defined, it is also inherited.

        Parameters
        ----------
        cl : Centerline
            The centerline object to be set.
        """
        self.centerline = cl
        extra = ["tau_joint"] if hasattr(cl, "tau_joint") else None
        self.set_data_from_other_node(cl, extra=extra)

    def build(self):
        """Build internal spline objects."""
        self.centerline.build()
        self.radius.build()

    def cartesian_to_vcs(self, p, rho_norm=False, method="scalar"):
        """
        Given a 3D point p expressed in cartesian coordinates, this method
        computes its expression in the Vessel Coordinate System (VCS). The method
        requires the attribute centerline to be set, additionally if rho normalization
        is desired, the radius spline attributes must have been built.

        Parameters
        ----------
        p : np.ndarray (3,)
            A 3D point in cartesian coordinates.
        rho_norm : bool, opt
            Default False. If radius attribute is built, and rho_norm
            is True, the radial coordinate is normalized by the expression:
            rho_n = rho /rho_w(tau, theta)
        method : Literal{'scalar', 'vec', 'vec_jac'}, opt
            The minimization method to use. See get_projection_parameter
            for more info.

        Returns
        -------
        p_vcs : np.ndarray(3,)
            The coordinates of the point in the VCS.

        """

        attribute_checker(self, atts=["centerline"], info="Cant compute VCS.")

        tau, theta, rho = self.centerline.cartesian_to_vcs(p=p, method=method)
        if rho_norm:
            attribute_checker(self, atts=["radius"], info="cant compute normalized VCS.")
            rho /= self.radius(tau, theta)

        return np.array((tau, theta, rho))

    def vcs_to_cartesian(
        self,
        tau: float | np.ndarray,
        theta: float | np.ndarray,
        rho: float | np.ndarray,
        rho_norm=True,
        grid=False,
        gridded=False,
        full_output=False,
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """
        Given a point expressed in Vessel Coordinate System (VCS), this method
        computes its cartesian coordinates.

        Using numpy broadcasting this method allows working with arrays of vessel
        coordinates.

        Parameters
        ----------
        tau : float or array-like (N,)
            The longitudinal coordinate of the point
        theta : float or array-like (N,)
            Angular coordinate of the point
        rho : float or array-like (N,)
            The radial coordinate of the point
        rho_norm : bool, opt
            Default False. Whether the rho passed is normalized or not.
        grid : bool
            Default False. If true, the method returns the cartesian representation of the
            grid tau x theta x rho.
        gridded: bool, optional
            Whether the input comes in a gridded way i.e. tau, theta, and rho have been generated
            by a function like numpy meshgrid.
        full_output : bool, false
            Default False. Whether to return the as well the vcs. Useful in combination with grid.

        Returns
        -------
        p : np.ndarray (N, 3)
            The point in cartesian coordinates.
        tau, theta, rho, rho_norm : np.ndarray (N, ), opt.
            If full_output is True, the vessel coordinates of the points are returned.
        """

        if not gridded:
            tau, theta, rho = broadcast_kwargs(tau=tau, theta=theta, rho=rho).values()

        if grid:
            gr = np.meshgrid(tau, theta, rho, indexing="ij")
            tau = gr[0].ravel()
            theta = gr[1].reshape(-1, 1)
            rho = gr[2].reshape(-1, 1)

        if rho_norm:
            if is_numeric(rho):
                rho = np.array([rho])
            rho_norm = deepcopy(rho)
            rho *= self.radius(tau, np.ravel(theta)).reshape(rho.shape)
        else:
            rho_norm = rho / self.radius(tau, np.ravel(theta)).reshape(rho.shape)

        p = self.centerline.vcs_to_cartesian(
            tau, theta, rho, gridded=True if grid or gridded else False
        )

        if full_output:
            return p, tau, theta, rho, rho_norm

        return p

    def extract_vessel_from_network(
        self, vmesh, thrs=5, use_normal=True, normal_thrs=30, cl=None, debug=False
    ):
        r"""
        Extract the vessel mesh from a vascular structure based on the centerline.

        The approach is similar to the centerline association method of the CenterlineTree class,
        however, in this other method each point is associated to a single branch, and this method
        does not care for other branches, allowing points to belong to different vessels.

        The vessel is extracted as follows:
        For each point, p, in the the mesh, its projection, q, is computed. Assuming
        wall normals are pointing outwards,
        Then, the deviation of the point to the cross section it belongs is measured by
        the angle that makes the tangent t, with vector q2p. If a points belong to a cross
        section, the angle between t and q2p should be 90. Then, points whose deviation
        is over thrs argument are rejected. Once points have been identified, they are
        extracted from the mesh and the largest connected component is considered as
        the vessel of interest.

        If use_normal is True, instead of considering the angle between t and q2p,
        the angle considered is t and the surface normal of p, :math:`\hatN(p)`.

        This method requires self.centerline. Warning: If argument cl is passed, the
        centerline object is set as self.centerline.

        Parameters
        ----------
        vmesh : pv.PolyData
            The vascular surface mesh.
        thrs : list[float],opt
            Defaulting to 10. The angle allowed between the tangent and q2p, can be seen
            as the longitudinal deviation from a cross section, for instance, thrs = 20
            allows points whose fulfilling that 70<angle(t, q2p)<110.
        cl : Centerline
            The Vessel centerline.

        Returns
        -------
        vsl_mesh : pv.PolyData
            The vessel polydata extracted.
        """

        if cl is not None:
            self.set_centerline(cl=cl)

        attribute_checker(self, ["centerline"], info="Can't extract Vessel from tree.")

        if "Normals" not in vmesh.point_data:
            vmesh.compute_normals(inplace=True)
        normals = vmesh.get_array("Normals", preference="point")

        vcs = []
        ids = np.zeros((vmesh.n_points,))
        for i in range(vmesh.n_points):
            p = vmesh.points[i]

            p_vcs = self.centerline.cartesian_to_vcs(p, method="scalar")
            vcs.append(p_vcs)

            q, t = self.centerline(p_vcs[0]), p_vcs[0]
            int_pts, _ = vmesh.ray_trace(q, p, first_point=False)
            if int_pts.shape[0] < 2:
                q2p = normalize(p - q)
                if q2p.dot(normals[i]) > 0:
                    tg = self.centerline.get_tangent(t)
                    angle = radians_to_degrees(np.arccos(q2p.dot(tg)))
                    if abs(angle - 90) < thrs:
                        if use_normal:
                            angle = radians_to_degrees(
                                np.arccos(np.clip(normals[i].dot(q2p), -1, 1))
                            )
                            if angle < normal_thrs:
                                ids[i] = 1
                        else:
                            ids[i] = 1

        vmesh["vcs"] = np.array(vcs)
        vsl_mesh = vmesh.extract_points(
            ids.astype(bool), adjacent_cells=True, include_cells=True
        ).connectivity(extraction_mode="largest")
        if debug:
            p = pv.Plotter()
            p.add_mesh(vmesh, scalars=ids, n_colors=2, opacity=0.4)
            p.add_mesh(vsl_mesh, color="g", opacity=0.7)
            p.add_mesh(
                self.centerline.to_polydata(), render_lines_as_tubes=True, color="g", line_width=10
            )
            p.show()

        return vsl_mesh

    def encode_vessel_mesh(
        self, vsl_mesh, tau_knots, theta_knots, laplacian_penalty=1.0, cl=None, debug=False
    ):
        """
        Encode a vessel using the centerline and the anisotropic radius.

        If the centerline have hierarchical data like its parent or tau_joint
        it is also set as a parameter for the branch.

        This method requires self.centerline to be set or passed.
        Warning: If argument cl is passed, the centerline object is set as
        self.centerline what may overwrite possible existing data.


        Parameters
        ----------
        vsl_mesh : pv.PolyData
            The mesh representing the vessel.
        knots_tau, knots_theta : int
            The amount of divisions to build the uniform knot vector.
        laplacian_penalty : float, optional
            Default 1. The penalization factor on radius laplacian.
        cl : Centerline, opt
            Default None. The centerline of said vessel. If passed is stored
            at self.centerline and node data is copied from it.

        Returns
        -------
        self : VesselAnatomyEncoding
            The VesselAnatomyEncoding object.
        """

        if cl is not None:
            self.set_centerline(cl=cl)

        if "vcs" not in vsl_mesh.point_data:
            points_vcs = np.array([self.centerline.cartesian_to_vcs(p) for p in vsl_mesh.points])
        else:
            points_vcs = vsl_mesh["vcs"]

        self.radius = Radius.from_points(
            points=points_vcs,
            tau_knots=tau_knots,
            theta_knots=theta_knots,
            laplacian_penalty=laplacian_penalty,
            cl=cl,
            debug=debug,
        )

    def compute_residual(self, p):
        """
        Compute the residual error of the encoding approximation.

        The method requires the VesselAnatomyEncoding splines to have been built.

        Given a point p, located in the vessel wall, let (ta, th) be the two first coordinates of
        the point in the VCS computed using the centerline, then the residual of the approximation
        at the point is defined as:
                || p - cl(ta) + rho_w(ta, th)(v1(ta)cos(th) + v2(ta)sin(th)) ||

        Parameters
        ----------
        p : np.ndarray (N,)
            The point on the vessel wall.

        Returns
        -------
        res : float
            The residual.
        """

        p_vcs = self.cartesian_to_vcs(p, rho_norm=True)
        res = np.linalg.norm(
            p - self.vcs_to_cartesian(tau=p_vcs[0], theta=p_vcs[1], rho=p_vcs[2], rho_norm=True)
        )
        return res

    def compute_centerline_intersection(self, cl, mode="point"):
        """
        Given a centerline that intersects the vessel wall, this method computes the location of
        said intersection. Depending on the mode selected it can return either the intersection
        point or the parameter value of the intersection in the provided centerline.

        Warning: If the passed centerline intersects more than one time, only the first found will
        be returned.

        Parameters
        ----------
        cl : Centerline
            The intersecting centerline.
        mode : {'point', 'parameter'}, opt
            Default 'point'. What to return.

        Returns
        -------
        : np.ndarray or float
            The intersection (parameter or point).

        """

        mode_opts = ["point", "parameter"]
        if mode not in mode_opts:
            error_message(f"Wrong value for mode argument. It must be in {mode_opts} ")

        def intersect(t):
            vcs = self.cartesian_to_vcs(cl(t), rho_norm=True)
            return abs(1 - vcs[2])

        res = minimize_scalar(
            intersect, bounds=(cl.t0, cl.t1), method="bounded"
        )  # Parameter at intersection

        if mode == "parameter":
            return res.x

        return cl(res.x)

    def to_multiblock(self, add_attributes=True, tau_res=100, theta_res=50):
        """
        Make a multiblock with two PolyData objects, one for the centerline and one for the radius.

        Parameters
        ----------
        add_attributes : bool, opt
            Default True. Whether to add all the attributes required to convert the multiblock
            back to a VesselAnatomyEncoding object.
        tau_res, theta_res : int, opt
            The resolution to build the vessel wall.

        Returns
        -------
        vsl_mb : pv.MultiBlock
            The multiblock with the required data to restore the vessel anatomy encoding.

        See Also
        --------
        from_multiblock

        """

        attribute_checker(
            self,
            ["centerline", "radius"],
            info=f"Can't convert vessel anatomy encoding {self.id} to multiblock.",
        )

        vsl_mb = pv.MultiBlock()
        vsl_mb["centerline"] = self.centerline.to_polydata(
            add_attributes=add_attributes, tau_res=tau_res
        )

        wall = self.make_tube(tau_res=tau_res, theta_res=theta_res)
        if add_attributes:
            rhp = self.radius.get_hyperparameters()
            rhp["feature vector"] = self.radius.to_feature_vector().tolist()
            wall.user_dict["radius"] = rhp

        vsl_mb["wall"] = wall

        return vsl_mb

    @staticmethod
    def from_multiblock(vsl_mb):
        """
        Make a VesselAnatomyEncoding object from a multiblock containing two PolyData objects,
        one for the centerline and one for the radius.

        This static method is the counterpart of to_multiblock. To properly work, this method
        requires the passed MultiBlock entries to contain the essential attributes as though
        returned by to_multiblock with add_attributes argument set to True.


        Parameters
        ----------
        vsl_mb : pv.MultiBlock
            A pyvista Multiblock with both wall and centerline PolyDatas and the required
            parameters.

        Returns
        -------
        vsl_enc : VesselAnatomyEncoding
            The VesselAnatomyEncoding object built from the MultiBlock.

        See Also
        --------
        to_multiblock

        """

        block_names = vsl_mb.keys()
        for name in ["centerline", "wall"]:
            if name not in block_names:
                raise ValueError(
                    "Cannot build vessel anatomy encoding from multiblock."
                    + f"{name} is not in {block_names}."
                )

        vsl_enc = VesselAnatomyEncoding()

        cl = Centerline().from_polydata(poly=vsl_mb["centerline"])
        vsl_enc.set_centerline(cl)

        wall = vsl_mb["wall"]
        radius = Radius().from_feature_vector(
            fv=np.array(wall.user_dict["radius"]["feature vector"]),
            hp={p: v for p, v in wall.user_dict["radius"].items() if p != "feature vector"},
        )
        vsl_enc.set_data(radius=radius)

        return vsl_enc

    def get_hyperparameters(self) -> dict[str, Any]:
        """
        Get the hyperparameter dictionary of the VesselAnatomyEncoding object.

        Returns
        -------
        hp : dict[str, Any]
            The hyperparameter dictionary.

        See Also
        --------
        set_hyperparameters
        Centerline.get_hyperparameters
        Radius.get_hyperparameters
        from_feature_vector
        """

        return super().get_hyperparameters()

    def set_hyperparameters(self, hp: dict[str, Any]):
        """
        Set the hyperparameters.

        Parameters
        ----------
        hp : dict[str, Any]
            The hyperparameters dictionary.

        See Also
        --------
        get_hyperparameters
        Centerline.set_hyperparameters
        Radius.set_hyperparameters
        from_feature_vector
        """

        # Centerline
        centerline = self.centerline
        if centerline is None:
            centerline = Centerline()
        centerline.set_hyperparameters(hp=hp["centerline"])
        self.set_centerline(centerline)

        # Radius
        if self.radius is None:
            self.radius = Radius()
        self.radius.set_parameters_from_centerline(self.centerline)
        self.radius.set_hyperparameters(hp=hp["radius"])

    def to_feature_vector(self, mode="full"):
        """
        Convert the VesselAnatomyEncoding to a feature vector.

        The feature vector version of a VesselAnatomyEncoding consist in appending the flattened
        centerline and radius coefficients.


        Parameters
        ----------
        mode : {'full', 'centerline', 'radius'}, optional
            Default 'full'. Argument to select the components to compose the  feature vector. Each
            of the modes works as follows:

            - 'full': Concatenation of both centerline and radius feature vectors. This option is
            required for posterior rebuilding of the VesselAnatomyEncoding.

            - 'centerline' : This mode only returns the centerline coefficients. It is equivalent to
            self.centerline.to_feature_vector()

            - 'radius' : This mode only returns the radius coefficients. It is equivalent to
            self.centerline.to_feature_vector()

        Returns
        -------
        fv : np.ndarray
            The feature vector according to mode. The shape of each feature vector changes
            accordingly.

        See Also
        --------
        from_feature_vector
        Centerline.to_feature_vector
        Radius.from_feature_vector
        """

        if mode not in {"full", "centerline", "radius"}:
            raise ValueError(
                "Wrong value for mode argument."
                + f"Provided is: {mode}, must be in {{'full', 'centerline','radius', 'image'}}."
            )

        cfv = []
        if mode in ["full", "centerline"]:
            cfv = self.centerline.to_feature_vector()

        rfv = []
        if mode in ["full", "radius"]:
            rfv = self.radius.to_feature_vector()

        fv = np.concatenate([cfv, rfv])

        return fv

    def split_feature_vector(self, fv):
        """
        Split the centerline component from the radius component of a feature vector.

        This function requires the previous setting of the hyperparameters of both centerline and
        radius objects.

        Parameters
        ----------
        fv : np.ndarray or array-like (N,)

        Returns
        -------
            cfv, rfv : np.ndarray
                The centerline and radius feature vectors respectively.

        See Also
        --------
        to_feature_vector
        Centerline.to_feature_vector
        Radius.to_feature_vector
        """

        l = self.centerline.get_feature_vector_length()
        rk = self.radius.get_feature_vector_length()
        if len(fv) != l + rk:
            raise ValueError(
                f"Cant split feature vector with length {len(fv)} into a centerline fv of length"
                + f" {l} and a radius fv of length {rk}"
            )

        cfv, rfv = fv[:l], fv[l:]
        return cfv, rfv

    def get_feature_vector_length(self):
        """
        Return the length of the feature vector considering only the spline parameters.

        The length of a VesselAnatomyEncoding feature vector is the sum of the length of the
        centerline and radius feature vectors.

        Returns
        -------
        n : int
            The length of the centerline feature vector.
        """

        attribute_checker(
            self,
            ["centerline", "radius"],
            info="Cannot compute the VesselAnatomyEncoding feature vector length.",
        )

        n = self.centerline.get_feature_vector_length() + self.radius.get_feature_vector_length()
        return n

    def from_feature_vector(
        self, fv: np.ndarray, hp: dict[str, Any] = None
    ) -> VesselAnatomyEncoding:
        """
        Build a VesselAnatomyEncoding object from a full feature vector.

        > Note that while hyperparameters argument is optional it must have been previously set or
        passed.


        Parameters
        ----------
        fv : np.ndarray (N,)
            The feature vector.
        hp : dict[str, Any], optional
            The hyperparameter dictionary.

        Returns
        -------
        vsl_enc : VesselAnatomyEncoding
            The vessel anatomy encoding built from the fv.

        See Also
        --------
        get_hyperparameters
        to_feature_vector
        """

        if hp is not None:
            self.set_hyperparameters(hp)

        cfv, rfv = self.split_feature_vector(fv)
        self.centerline.set_parameters(build=True, coeffs=cfv.reshape(-1, 3))
        self.radius.set_parameters(build=True, coeffs=rfv)
        return self

    def translate(self, t):
        """
        Translate the VesselAnatomyEncoding object.

        The translation only requires translating the centerline coefficients, since the radius is
        is expressed with respect to the centerline.

        Parameters
        ----------
        t : np.ndarray (3,)
            The translation vector.

        See Also
        --------
        Centerline.translate
        """

        if self.centerline is not None:
            self.centerline.translate(t)

    def scale(self, s):
        """
        Scale the VesselAnatomyEncoding object.

        The scale is applied to both centerline and radius coefficients. No anisotropic scaling is
        allowed, and a single scalar is required.

        Parameters
        ----------
        s : float
            The scale factor.
        update : bool, optional
            Default True. Whether to rebuild the splines after the transformation.

        See Also
        --------
        Centerline.scale
        """

        if not isinstance(s, (int, float)):
            error_message(
                f"Wrong value for radius object scaling. Expected a float|int, provided is {s}."
            )

        if self.centerline is not None:
            self.centerline.scale(s=s)
        if self.radius is not None:
            self.radius.scale()

    def rotate(self, r):
        """
        Rotate the VesselAnatomyEncoding object.

        The rotation only requires translating the centerline coefficients, since the radius is
        expressed with respect to the centerline.

        Parameters
        ----------
        r : np.ndarray (3, 3)
            The rotation matrix.

        See Also
        --------
        Centerline.rotate
        """
        self.centerline.rotate(r=r)
