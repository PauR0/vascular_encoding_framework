from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np


class Encoding(ABC):
    """Base class for encoding classes."""

    def __init__(self):
        if TYPE_CHECKING:
            self._hyperparameters: list[str]

        required_attributes = ["_hyperparameters"]
        for attr in required_attributes:
            if not hasattr(self, attr):
                raise NotImplementedError(
                    f"{self.__class__.__name__} must define the attribute: '{attr}'"
                )

    @abstractmethod
    def get_hyperparameters(self, **kwargs) -> dict[str, Any]:
        """
        Get dict containing the hyperparameters of the encoding object.

        Parameters
        ----------
        **kwargs
            Child subclasses may pass those hyperparameters that are not object attributes.

        Returns
        -------
        hp : dict[str, Any]
            The hyperparameter dict.

        See Also
        --------
        set_hyperparameters
        """

        hp = {}
        for p in self._hyperparameters:
            if p in kwargs:
                value = kwargs[p]
            elif hasattr(self, p):
                value = getattr(self, p)
            else:
                raise AttributeError(
                    f"Unable to build hyperparameter dict for class {self.__class__.__name__}."
                    + f"Parameter {p} is not an attribute nor has been passed."
                )

            if isinstance(value, Encoding):
                hp[p] = value.get_hyperparameters()
            else:
                hp[p] = value

        return hp

    @abstractmethod
    def set_hyperparameters(self, hp: dict[str, Any], **kwargs):
        """
        Set the hyperparameters.

        Parameters
        ----------
        hp : dict[str, Any]
            The hyperparameter dictionary.
        kwargs:
            Specific keyword arguments of the subclass implementation.

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

    @abstractmethod
    def from_feature_vector(self, fv: np.ndarray, hp: dict[str, Any] = None) -> Encoding:
        """
        Build an Encoding object from a feature vector.

        Warning: The hyperparameters must either be passed or set previously.

        Parameters
        ----------
        fv : np.ndarray or array-like (N,)
            The feature vector.
        hp : dict[str, Any], optional
            The hyperparameter dictionary.

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
