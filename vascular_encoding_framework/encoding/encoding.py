from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Encoding(ABC):
    """
    Base class for encoding object. This class contains the method required to
    exist in an encoding.
    """

    def __init__(self):
        required_attributes = ["_hyperparameters"]
        for attr in required_attributes:
            if not hasattr(self, attr):
                raise NotImplementedError(
                    f"{self.__class__.__name__} must define the attribute: '{attr}'"
                )

    @abstractmethod
    def get_hyperparameters(self) -> dict[str, Any]:
        """
        Get dict containing the hyperparameters of the encoding object.

        Returns
        -------
        hp : dict[str, Any]
            The hyperparameter dict.

        See Also
        --------
        set_hyperparameters
        """
        ...

    @abstractmethod
    def set_hyperparameters(self, hp: dict[str, Any], **kwargs):
        """
        Set the hyperparameters.

        Parameters
        ----------
        hp : dict[str, Any]
            The hyperparameter dictionary.
        kwargs:
            Specific keyword arguments of the subclass implementation

        See Also
        --------
        get_hyperparameters
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
    def to_feature_vector(self) -> np.ndarray:
        """
        Convert the Encoding to a feature vector.

        Return:
        ------
        fv : np.ndarray (N,)
            The feature vector with the selected data.

        """
        ...

    @staticmethod
    def from_feature_vector(fv: np.ndarray, hp: dict[str, Any] = None) -> Encoding:
        """
        Build an Encoding object from a feature vector.

        Warning: The hyperparameters must either be passed or set.


        Parameters
        ----------
        fv : np.ndarray or array-like (N,)
            The feature vector.
        hp : dict[str, Any], optional
            Default None. If not passed, the hyperparameter dict must have been set previously.

        Returns
        -------
        enc : Encoding
            The built encoding with all the attributes appropriately set.

        See Also
        --------
        get_hyperparameters
        set_hyperparameters
        """
        ...
