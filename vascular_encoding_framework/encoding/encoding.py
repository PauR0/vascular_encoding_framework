from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Encoding(ABC):
    """
    Base class for encoding object. This class contains the method required to
    exist in an encoding.
    """

    @abstractmethod
    def get_metadata(self, **kwargs) -> np.ndarray:
        """
        Get a copy of the metadata array.

        The format of the metadata array of an Encoding is to be specified in the documentation
        of the subclass.

        Returns
        -------
        md : np.ndarray
            The metadata array.
        """
        ...

    @abstractmethod
    def set_metadata(self, md, **kwargs) -> np.ndarray:
        """
        Extract and set the attributes from a the metadata array.

        See get_metadata method's documentation for further information on the expected format.

        Parameters
        ----------
        md : np.ndarray
            The metadata array.
        kwargs:
            Specific keyword arguments of the subclass implementation
        """
        ...

    @abstractmethod
    def get_feature_vector_length(self, **kwargs) -> int:
        """
        Return the length of the feature vector.

        Returns
        -------
        n : int
            The length of the centerline feature vector.
        """
        ...

    @abstractmethod
    def to_feature_vector(self, add_metadata=True, **kwargs) -> np.ndarray:
        """
        Convert the Encoding to a feature vector.

        Return:
        ------
        fv : np.ndarray (N,)
            The feature vector with the selected data.

        """
        ...

    @staticmethod
    def from_feature_vector(fv, md=None) -> Encoding:
        """
        Build an Encoding object from a feature vector.

        Warning: This method only works if the feature vector has the metadata at the beginning or it
        is passed using the md argument.

        Warning: Due to the lack of hierarchical data of the feature vector mode the returned
        Encoding object will only have root nodes whose ids correspond to the its order in the
        feature vector.


        Parameters
        ----------
        fv : np.ndarray or array-like (N,)
            The feature vector with the metadata array at the beginning.
        md : np.ndarray, optional
            Default None. If fv does not contain the metadata array at the beginning it can be
            passed through this argument.
        """
        ...
