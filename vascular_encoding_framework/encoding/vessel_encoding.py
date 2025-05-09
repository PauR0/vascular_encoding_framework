from __future__ import annotations

from copy import deepcopy

import numpy as np
import pyvista as pv
from scipy.optimize import minimize_scalar

from .._base.misc import attribute_checker, broadcast_kwargs, is_numeric
from .._base.node import Node
from ..centerline import Centerline
from ..messages import error_message
from ..utils.misc import split_metadata_and_fv
from ..utils.spatial import normalize, radians_to_degrees
from .encoding import Encoding
from .radius import Radius
from .remesh import VesselMeshing


class VesselAnatomyEncoding(Node, Encoding, VesselMeshing):
    """The class for encoding a single branch vessel."""

    def __init__(self):
        Node.__init__(self=self)

        self.centerline: Centerline = None
        self.radius: Radius = None

    def set_data(self, **kwargs):
        """Set attributes using kwargs and the setattr function."""

        if "centerline" in kwargs:
            self.set_centerline(cl=kwargs["centerline"])
            kwargs.pop("centerline")

        return super().set_data(**kwargs)

    def set_centerline(self, cl):
        """
        Set the centerline attribute. Note that the VesselAnatomyEncoding object inherits
        the node attributes from the centerline, in addition if tau_joint is defined,
        it is also inherited.
        """

        self.centerline = cl
        self.set_data_from_other_node(cl)
        if hasattr(cl, "tau_joint"):
            self.set_data(tau_joint=cl.tau_joint)

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

        if not attribute_checker(self, atts=["centerline"], info="cant compute VCS."):
            return False

        tau, theta, rho = self.centerline.cartesian_to_vcs(p=p, method=method)
        if rho_norm:
            if not attribute_checker(self, atts=["radius"], info="cant compute normalized VCS."):
                return False
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

        if not attribute_checker(self, ["centerline"], info="cannot extract Vessel from tree."):
            return False

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

    def to_multiblock(self, add_attributes=True, tau_res=None, theta_res=None):
        """
        Make a multiblock with two PolyData objects, one for the centerline and one for the radius.

        Parameters
        ----------
        add_attributes : bool, opt
            Default True. Whether to add all the attributes required to convert the multiblock
            back to a VesselAnatomyEncoding object.
        tau_res, theta_res : int, opt
            The resolution to build the vessel wall. Defaulting to make_surface_mesh default
            values.

        Returns
        -------
        vsl_mb : pv.MultiBlock
            The multiblock with the required data to restore the vessel anatomy encoding.

        See Also
        --------
        :py:meth:`from_multiblock`

        """

        if not attribute_checker(
            self,
            ["centerline", "radius"],
            info=f"Cannot convert vessel anatomy encoding {self.id} multiblock.",
        ):
            return None

        vsl_mb = pv.MultiBlock()
        vsl_mb["centerline"] = self.centerline.to_polydata(
            add_attributes=add_attributes, tau_res=tau_res
        )

        wall = self.make_tube(tau_res=tau_res, theta_res=theta_res)
        if add_attributes:
            # Adding tau atts
            wall.add_field_data(
                np.array([self.radius.x0, self.radius.x1]), "tau_interval", deep=True
            )
            wall.add_field_data(np.array([self.radius.kx]), "tau_k", deep=True)
            wall.add_field_data(np.array([self.radius.n_knots_x]), "n_tau_knots", deep=True)
            wall.add_field_data(np.array([self.radius.extra_x]), "tau_extrapolation", deep=True)

            # Adding theta atts
            wall.add_field_data(
                np.array([self.radius.y0, self.radius.y1]), "theta_interval", deep=True
            )
            wall.add_field_data(np.array([self.radius.ky]), "theta_k", deep=True)
            wall.add_field_data(np.array([self.radius.n_knots_y]), "n_theta_knots", deep=True)
            wall.add_field_data(np.array([self.radius.extra_y]), "theta_extrapolation", deep=True)

            wall.add_field_data(np.array(self.radius.coeffs), "coeffs", deep=True)
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
                Default True. Whether to add all the attributes required to convert the multiblock
                back to a VesselAnatomyEncoding object.

        Returns
        -------
            vsl_enc : VesselAnatomyEncoding
                The VesselAnatomyEncoding object built with the attributes stored as field data.

        See Also
        --------
        to_multiblock

        """

        block_names = vsl_mb.keys()
        for name in ["centerline", "wall"]:
            if name not in block_names:
                error_message(
                    info=f"Cannot build vessel anatomy encoding from multiblock. {name} is not in {block_names}. "
                )
                return None

        vsl_enc = VesselAnatomyEncoding()

        cl = Centerline().from_polydata(poly=vsl_mb["centerline"])
        vsl_enc.set_centerline(cl)

        radius = Radius()
        wall = vsl_mb["wall"]
        # Setting tau params
        radius.set_parameters(
            x0=wall.get_array("tau_interval", preference="field")[0],
            x1=wall.get_array("tau_interval", preference="field")[1],
            kx=wall.get_array("tau_k", preference="field")[0],
            n_knots_x=wall.get_array("n_tau_knots", preference="field")[0],
            extra_x=wall.get_array("tau_extrapolation", preference="field")[0],
        )

        # Setting theta params
        radius.set_parameters(
            y0=wall.get_array("theta_interval", preference="field")[0],
            y1=wall.get_array("theta_interval", preference="field")[1],
            ky=wall.get_array("theta_k", preference="field")[0],
            n_knots_y=wall.get_array("n_theta_knots", preference="field")[0],
            extra_y=wall.get_array("theta_extrapolation", preference="field")[0],
        )

        radius.set_parameters(build=True, coeffs=wall.get_array("coeffs", preference="field"))

        vsl_enc.set_data(radius=radius)

        return vsl_enc

    def get_metadata(self):
        """
        Return a copy of the metadata array.

        The metadata array of a VesselAnatomyEncoding object is composed by the centerline and radius
        metadata arrays as follows:
                [nc+nr+1, nc, cmd_0,...cmd_nc-1, nr, rmd_0,...,rmd_nr-1]

        Returns
        -------
        md : np.ndarray
            The metadata array.

        See Also
        --------
        set_metadata
        Centerline.get_metadata
        Radius.get_metadata
        to_feature_vector
        from_feature_vector

        """

        cmd = self.centerline.get_metadata()
        rmd = self.radius.get_metadata()
        md = np.concatenate([[cmd[0] + rmd[0] + 1], cmd, rmd])
        return md

    def set_metadata(self, md):
        """
        Extract and set the attributes from a the metadata array.

        See get_metadata method's documentation for further information on the expected format.

        Parameters
        ----------
        md : np.ndarray
            The metadata array.

        See Also
        --------
        get_metadata
        Centerline.set_metadata
        Radius.set_metadata
        to_feature_vector
        from_feature_vector
        """

        # Centerline
        nc = round(md[1])
        ini = 1
        end = ini + nc
        cmd = md[ini:end]
        if self.centerline is None:
            self.centerline = Centerline()
        self.centerline.set_metadata(md=cmd)

        # Radius
        nr = round(md[end])
        ini = end
        end = ini + nr
        rmd = md[ini:end]
        if self.radius is None:
            self.radius = Radius()
        self.radius.set_parameters_from_centerline(self.centerline)
        self.radius.set_metadata(md=rmd)

    def to_feature_vector(self, mode="full", add_metadata=True):
        """
        Convert the VesselAnatomyEncoding to a feature vector.

        The feature vector version of a VesselAnatomyEncoding consist in appending the flattened
        centerline and radius coefficients.


        Parameters
        ----------
        mode : {'full', 'centerline', 'radius', 'image'}, optional
            Default 'full'. Argument to control the way the VesselObject is converted in a
            feature vector. Each of the modes works as follows:

            - 'full': This mode stores all the information required to convert the feature
                vector back to a VesselAnatomyEncoding. The feature vector built by this mode starts
                with some metadata, followed by the raveled centerline spline coefficients and
                finishes with the raveled radius coefficients.
                It should look like:
                fv = (clx_0,...,clx_l, cly_0,...,cly_l, clz_0,...,clz_l, r_00,...,r_kr)
                Where: - l = n_knots_centerline + k       + 1
                        - k = n_knots_tau        + k_tau   + 1
                        - r = n_knots_theta      + k_theta + 1

            - 'centerline' : This mode only returns the centerline coefficients.
                It should look like:
                fv = (clx_0,...,clx_L, cly_0,...,cly_L, clz_0,...,clz_L)

            - 'radius' : This mode only returns the radius coefficients.
                It should look like:
                fv = (r_00,...,r_0R,r_10,...,r_KR)

            - 'image' : return the feature vector arranged as an image.
                NOT IMPLEMENTED YET
        add_metadata : bool, optional
            Default True. If True, a metadata array is append at the beginning of the feature vector.
            The first element of it corresponds with the number of metadata elements.
                md = (nmd, md_0,...,m_nmd-1)

        Returns
        -------
        fv : np.ndarray
            The feature vector according to mode. The shape of each feature vector changes
            accordingly.

        Notes
        -----
        Note that the feature vector representation does not bear any hierarchical data, not even
        if add_metadata is True. Be sure that hierarchical data is properly stored if will be later
        required. For storage purposes check `to_multiblock` method.


        See Also
        --------
        :py:meth:`from_feature_vector`
        :py:meth:`VesselAnatomyEncoding.to_feature_vector`
        :py:meth:`VesselAnatomyEncoding.from_feature_vector`
        """

        md = []

        if mode not in {"full", "centerline", "radius", "image"}:
            error_message(
                "Wrong value for mode argument, cannot make a feature vector."
                + f"Provided is: {mode}, must be in ['full', 'centerline','radius', 'image']."
            )
            return None

        if mode == "centerline":
            fv = self.centerline.to_feature_vector(add_metadata=add_metadata)

        if mode == "radius":
            fv = self.radius.to_feature_vector(add_metadata=add_metadata)

        if mode == "full":
            cfv = self.centerline.to_feature_vector(add_metadata=False)
            rfv = self.radius.to_feature_vector(add_metadata=False)
            if add_metadata:
                md = self.get_metadata()
            fv = np.concatenate([md, cfv, rfv])

        if mode == "image":
            # TODO
            raise NotImplementedError(
                "The implementation is not yet developed, this mode will be available in future versions."
            )

        return fv

    def split_feature_vector(self, fv, has_metadata=False):
        """
        Split the centerline component from the radius component of a feature vector.

        This function requires the metadata of the centerline and radius objects exist.

        Parameters
        ----------
        fv : np.ndarray or array-like (N,)
        has_metadata : bool, optional
            Default False. If true, the first element is assumed to be the amount of metadata,
            then the first elements are removed according to this number.

        Returns
        -------
            cfv, rfv : np.ndarray
                The centerline and radius feature vectors respectively, both with no metadata.

        See Also
        --------
        to_feature_vector
        Centerline.to_feature_vector
        Radius.to_feature_vector
        """

        if has_metadata:
            nmd = fv[0]
            fv = fv[nmd:]

        l = self.centerline.get_feature_vector_length()
        rk = self.radius.get_feature_vector_length()
        if len(fv) != l + rk:
            error_message(
                f"Cant split feature vector with length {len(fv)} in a centerline fv of length {l} and a radius fv of length {rk}"
            )
            return None, None

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

        if not attribute_checker(
            self,
            ["centerline", "radius"],
            info="Cannot compute the VesselAnatomyEncoding feature vector length.",
        ):
            return None

        n = self.centerline.get_feature_vector_length() + self.radius.get_feature_vector_length()
        return n

    def extract_from_feature_vector(self, fv, md=None):
        """
        Update the attributes of the VesselAnatomyEncoding object with those provided in a feature
        vector.

        To create a new VesselAnatomyEncoding object see `from_feature_vector` static method.


        Parameters
        ----------
        fv : np.ndarray or array-like (N,)
            The feature vector with the metadata array at the beginning.
        md : np.ndarray, optional
            Default None. If fv does not contain the metadata array at the beginning it can be
            passed through this argument.


        See Also
        --------
        from_feature_vector
        to_feature_vector
        """

        if md is not None:
            self.set_metadata(md=md)

        n = self.get_feature_vector_length()
        if len(fv) != n:
            error_message(
                f"Cannot extract attributes from feature vector. Expected a feature vector of length {n} and the one provided has {len(fv)} elements."
            )
            return None

        cfv, rfv = self.split_feature_vector(fv)
        self.centerline.set_parameters(build=True, coeffs=cfv.reshape(-1, 3))
        self.radius.set_parameters(build=True, coeffs=rfv)

    @staticmethod
    def from_feature_vector(fv, md=None):
        """
        Build a VesselAnatomyEncoding object from a full feature vector.

        Warning: This method only works if the feature vector has the metadata at the beginning or it
        is passed using the md argument.

        Warning: The returned VesselAnatomyEncoding wont have any hierarchical properties nor id since that
        information is not stored on the feature vector.


        Parameters
        ----------
        fv : np.ndarray or array-like (N,)
            The feature vector with the metadata array at the beginning.
        md : np.ndarray, optional
            Default None. If fv does not contain the metadata array at the beginning it can be
            passed through this argument.

        Returns
        -------
        vsl_enc : VesselAnatomyEncoding
            The vessel anatomy encoding built from the fv.

        See Also
        --------
        get_metadata
        set_metadata
        to_feature_vector

        """

        if md is None:
            md, fv = split_metadata_and_fv(fv)

        vsl_enc = VesselAnatomyEncoding()
        vsl_enc.extract_from_feature_vector(fv=fv, md=md)
        return vsl_enc

    def translate(self, t, update=True):
        """
        Translate the VesselAnatomyEncoding object.

        The translation only requires translating the centerline coefficients, since the radius is
        is expressed with respect to the centerline.

        Parameters
        ----------
        t : np.ndarray (3,)
            The translation vector.
        update : bool, optional
            Default True. Whether to rebuild the splines after the transformation.

        See Also
        --------
        Centerline.translate
        """

        if self.centerline is not None:
            self.centerline.coeffs += t.reshape(
                3,
            )
            if update:
                self.centerline.build()

    def scale(self, s, update=True):
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
            self.centerline.coeffs *= s
            if update:
                self.centerline.build()

        if self.radius is not None:
            self.radius.coeffs *= s
            if update:
                self.radius.build()

    def rotate(self, r, update=True):
        """
        Rotate the VesselAnatomyEncoding object.

        The rotation only requires translating the centerline coefficients, since the radius is
        is expressed with respect to the centerline.

        Parameters
        ----------
        r : np.ndarray (3, 3)
            The rotation matrix.
        update : bool, optional
            Default True. Whether to rebuild the splines after the transformation.

        See Also
        --------
        Centerline.rotate
        """

        # ensure normality of the rotation matrix columns
        r /= np.linalg.norm(r, axis=0)

        if self.centerline is not None:
            self.centerline.coeffs = (r @ self.centerline.coeffs.T).T
            if update:
                self.centerline.build()
