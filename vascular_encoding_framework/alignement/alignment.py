

from abc import ABC, abstractmethod
from typing import Literal

import vtk
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from ..messages import *
from ..utils.spatial import decompose_transformation_matrix
from ..utils._code import attribute_checker, attribute_setter


def rigid_alignment(A, B):
    """
    Compute the rigid alignment between a pair of point sets in correspondence.

    Implementation of:
    "Least-Squares Fitting of Two 3-D Point Sets", Arun, K. S. and Huang, T. S. and Blostein, S. D,
    IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume 9 Issue 5, May 1987

    Credits to: https://github.com/nghiaho12/rigid_transform_3D

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
    def run(self, **kwargs):
        ...
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
            trans_source.points = self.apply_transformation(trans_source.points)

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
            _points : np.ndarray | pv.DataObject
        """

        _points = points
        if isinstance(points, pv.DataObject):
            _points = points.points

        if self.rotation is not None:
            _points = Rotation.from_matrix(self.rotation).apply(_points)

        if self.scale is not None:
            _points = _points * self.scale.reshape(3,)

        if self.translation is not None:
            _points = _points + self.translation.reshape(3,)

        if isinstance(points, pv.DataObject):
            return points

        return _points
    #
#

class ICP(Alignment):
    """
    Class to perform iterative closest point algorithm between two PolyData meshes.
    """

    def __init__(self):
        """
        ICP constructor
        """

        super().__init__()

        self.max_iter      : int = 100
        self.metric        : Literal['rms', 'abs'] = 'rms'
        self.max_landmarks : int = 100
        self.rigid         : bool = True
    #

    def run(self):
        """
        Compute the transformation to align source to target.

        The decomposed transformation matrix is stored as atributes.

        Returns
        -------

            trans_matrix : np.ndarray (4,4)

        """

        if not attribute_checker(obj=self, atts=['source', 'target'], info="Can't compute ICP alignment...")
            return

        icp = vtk.vtkIterativeClosestPointTransform()

        _source = self.source
        if isinstance(_source, np.ndarray):
            source_ = pv.PolyData(_source)
        icp.SetSource(source=_source)

        _target = self.target
        if isinstance(_target, np.ndarray):
            _target = pv.PolyData(_target)
        icp.SetTarget(_target)

        icp.GetLandmarkTransform().SetModeToRigidBody()
        if not self.rigid:
            icp.GetLandmarkTransform().SetModeToAffine()

        icp.SetMaximumNumberOfIterations(self.max_iter)
        icp.SetMaximumNumberOfLandmarks(self.max_landmarks)

        if self.metric == 'rms':
            icp.SetMeanDistanceModeToRMS()
        elif self.metric == 'abs':
            icp.SetMeanDistanceModeToAbsoluteValue()
        else:
            error_message(info="Wrong value for metric attribute {self.metric}. Defaulting to rms")
            icp.SetMeanDistanceModeToRMS()

        icp.StartByMatchingCentroidsOn()
        icp.CheckMeanDistanceOn()
        icp.Modified()
        icp.Update()

        #From vtk matrix to numpy matrix
        vtk_matrix = icp.GetMatrix()
        trans_matrix = np.array([vtk_matrix.GetElement(i, j) for i in range(4) for j in range(4)]).reshape(4,4)

        self.translation_vector, self.scale, self.rotation = decompose_transformation_matrix(matrix=trans_matrix)

        return trans_matrix
    #
#

class RigidCorrespondenceAlignment:
    """
    Class to compute the rigid alignment of two point arrays assuming index based correspondence.
    This means, the first element of the source array is expected to be in correspondence with the
    first element of the target array.
    """

    def run(self):
        """
        This method perform a rigid registration using the centerline in
        the feature vector provided (source_fv), and the target aorta feature vector.

        """

        _source = self.source
        if isinstance(self.source, pv.DataObject):
            _source = self.source.points

        _target = self.target
        if isinstance(self.target, pv.DataObject):
            _target = self.target.points

        r, t = rigid_alignment(A=_source.reshape(3, -1),
                                  B=_target.reshape(3, -1))

        self.set_translation_vector(t.flatten())
        self.set_rotation(Rotation.from_matrix(r))
