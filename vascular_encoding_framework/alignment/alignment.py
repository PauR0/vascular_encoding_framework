

from abc import ABC, abstractmethod

import numpy as np
import pyvista as pv

from ..messages import *
from ..encoding import VascularEncoding
from ..utils.spatial import decompose_transformation_matrix, transform_point_array
from ..utils._code import attribute_checker, attribute_setter


def OrthogonalProcrustes(A, B):
    """
    Compute the Orthogonal Procrustes [1]_ alignment between to point sets.

    In other words rigid alignment between a pair of point sets in correspondence [2]_.

    Credits to: https://github.com/nghiaho12/rigid_transform_3D for the implementation of [3]

    Arguments
    ---------

        A, B : np.ndarray (3, N)
            The source and target point arrays.


    Returns
    -------

        R : np.ndarray (3,3)
            Rotation matrix of the rigid alignment.

        t : np.ndarray (3,1)
            translation vector of the rigid alignment.

    Notes
    -----

        [1] https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

        [2] "A generalized solution of the orthogonal procrustes problem." Schönemann, P.H.
        Psychometrika 31, 1-10 (1966). https://doi.org/10.1007/BF02289451

        [3] "Least-Squares Fitting of Two 3-D Point Sets", Arun, K. S. and Huang, T. S. and
        Blostein, S. D, IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume
        9 Issue 5, May 1987

    """

    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t
#


class Alignment(ABC):
    """
    Abstract base class for alignment methods.
    """

    def __init__(self):

        self.target : np.ndarray | pv.DataObject = None
        self.source : np.ndarray | pv.DataObject = None

        self.translation : np.ndarray = None
        self.scale       : np.ndarray = None
        self.rotation    : np.ndarray = None
    #

    def set_parameters(self, **kwargs):
        """
        Method to set parameters as attributes of the object.
        """
        clss = self.__class__()
        params = {k:v for k,v in kwargs.items() if k in clss.__dict__}
        attribute_setter(self, **params)
    #

    @abstractmethod
    def run(self, apply:bool=False):
        """
        If apply is True, this method should return self.transform_source()
        """
    #

    def transform_source(self):
        """
        Transform the source geometry to be aligned with target.

        Returns
        -------

            trans_source : np.ndarray | pv.DataObject | None
                The transformed source geometry. If self.source has not been set, returns None.
        """

        trans_source = None

        if isinstance(self.source, np.ndarray):
            trans_source = self.apply_transformation(self.source)

        elif isinstance(self.source, pv.DataObject):
            trans_source = self.source.copy(deep=True)
            trans_source = self.apply_transformation(trans_source)

        return trans_source
    #

    def apply_transformation(self, points : np.ndarray | pv.DataObject):
        """
        Apply the alignment transformation to a given set of points.

        If any of the transform attributes is None, that aspect of the tranform will be ignored.
        For instance if sacle is None, the transformation wont include scaling.

        Arguments
        ---------

            points : np.ndarray | pv.DataObject
                The point array to be transformed with shape (N, 3). If a pyvista object is passed,
                its attribute points will be modified.

        Returns
        -------
            points : np.ndarray | pv.DataObject
        """

        if not isinstance(points, (np.ndarray, pv.DataObject, VascularEncoding)):
            error_message(f"Wrong type for points argument. Available types are {np.ndarray, pv.DataObjects, VascularEncoding} and passed was: {type(points)}")
            return None

        if isinstance(points, np.ndarray) and points.shape[1] != 3:
            error_message("Wrong shape passed can't apply transformation. Point arrays muy be in shape (N, 3).")
            return None

        if isinstance(points, np.ndarray):
            points = transform_point_array(points, t=self.translation, s=self.scale, r=self.rotation)

        if isinstance(points, pv.DataObject):
            points.points = transform_point_array(points.points, t=self.translation, s=self.scale, r=self.rotation)

        return points
    #
#

class IterativeClosestPoint(Alignment):
    """
    Class to perform iterative closest point algorithm between two PolyData meshes.
    """

    def __init__(self):
        """
        ICP constructor
        """

        super().__init__()

        self.max_iterations : int = 100
        self.max_landmarks  : int = 100
        self.rigid          : bool = True
    #

    def run(self, apply=True):
        """
        Compute the transformation to align source to target.

        The decomposed transformation matrix is stored as atributes.

        Arguments
        ---------

            apply : bool, optional
                Default True. Whether to apply the computed transformation to source and return it.

        Returns
        -------

            trans_source : np.ndarray | pv.DataObject, optional
                If apply is True (which is the default) this method
                returns the source object transformed to be aligned with target.

        """

        if not attribute_checker(obj=self, atts=['source', 'target'], info="Can't compute ICP alignment..."):
            return

        _source = self.source
        if isinstance(_source, np.ndarray):
            _source = pv.PolyData(_source)

        _target = self.target
        if isinstance(_target, np.ndarray):
            _target = pv.PolyData(_target)

        trans_source, trans_matrix = _source.align(target=_target,
                                                   max_landmarks=self.max_landmarks,
                                                   max_iterations=self.max_iterations,
                                                   return_matrix=True)

        self.translation, self.scale, self.rotation = decompose_transformation_matrix(matrix=trans_matrix)

        if apply:
            return trans_source
    #
#

class RigidProcrustesAlignment(Alignment):
    """
    Class to compute the rigid alignment of two point arrays assuming index based correspondence.
    This means, the first element of the source array is expected to be in correspondence with the
    first element of the target array.
    """

    def run(self, apply=True):
        """
        Method to run rigid procrustes alignment computation using source and target attributes.

        Arguments
        ---------

            apply : bool, optional
                Default True. Whether to apply the computed transformation to source and return it.

        Returns
        -------

            apply : bool
                Default True. Whether to apply the computed transformation to source and return it.

        """

        if not attribute_checker(obj=self, atts=['source', 'target'], info="Can't compute RigidProcrustes alignment..."):
            return

        _source = self.source
        if isinstance(self.source, pv.DataObject):
            _source = self.source.points

        _target = self.target
        if isinstance(self.target, pv.DataObject):
            _target = self.target.points

        r, t = OrthogonalProcrustes(A=_source.T,
                                    B=_target.T)

        self.translation = t.flatten()
        self.rotation    = r

        if apply:
            return self.transform_source()
    #
#