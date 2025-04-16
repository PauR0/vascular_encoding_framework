from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Encoding(ABC):
    """
    Base class for encoding object. This class contains the method required to exist in an encoding.
    """

    def __init__(self):
        ...
    #

    @abstractmethod
    def get_metadata(self, **kwargs) -> np.ndarray:
        """
        This method returns a copy of the metadata array.

        The format of the metadata array of an Encoding is to be specified in the documentation
        of the subclass.

        Returns
        -------
            md : np.ndarray
                The metadata array.
        """
        ...
    #

    @abstractmethod
    def set_metadata(self, md, **kwargs) -> np.ndarray:
        """
        This method extracts and sets the attributes from a the metadata array.

        See get_metadata method's documentation for further information on the expected format.

        Arguments
        ---------
            md : np.ndarray
                The metadata array.
        """
        ...
    #

    @abstractmethod
    def get_feature_vector_length(self, **kwargs) -> int:
        """
        This method returns the length of the feature vector.

        Returns
        -------
            n : int
                The length of the centerline feature vector.
        """
        ...
    #

    @abstractmethod
    def to_feature_vector(self, add_metadata=True, **kwargs) -> np.ndarray:
        """
        Convert the Encoding to a feature vector.

        Return
        ------
            fv : np.ndarray (N,)
                The feature vector with the selected data.

        """
        ...
    #

    @staticmethod
    def from_feature_vector(fv, md=None) -> Encoding:
        """
        Build an Encoding object from a feature vector.

        Warning: This method only works if the feature vector has the metadata at the beginning or it
        is passed using the md argument.

        Warning: Due to the lack of hierarchical data of the feature vector mode the returned
        Encoding object will only have root nodes whose ids correspond to the its order in the
        feature vector.


        Arguments
        ---------

            fv : np.ndarray or array-like (N,)
                The feature vector with the metadata array at the beginning.

            md : np.ndarray, optional
                Default None. If fv does not contain the metadata array at the beginning it can be
                passed through this argument.
        """
        ...
    #

    @abstractmethod
    def translate(self, t, update=True):
        """
        Translate the Encoding object and update (if True) the underlying splines.

        Arguments
        ---------

            t : np.ndarray (3,)
                The translation vector.

            update : bool, optional
                Default True. Whether to rebuild the splines after the transformation.

        """
        ...
    #

    @abstractmethod
    def scale(self, s, update=True):
        """
        Scale the Encoding object by a scalar factor s. Then update (if True) the underlying splines.

        Arguments
        ---------

            s : float
                The scale factor.

            update : bool, optional
                Default True. Whether to rebuild the splines after the transformation.
        """
        ...
    #

    @abstractmethod
    def rotate(self, r, update=True):
        """
        Rotate the Encoding with the provided rotation matrix r. Then update (if True) the underlying splines.

        Arguments
        ---------

            r : np.ndarray (3, 3)
                The rotation matrix.

            update : bool, optional
                Default True. Whether to rebuild the splines after the transformation.
        """
        ...
    #
#
