__all__ = [
    "SpatialObject",
]

from abc import ABC, abstractmethod

import numpy as np


class SpatialObject(ABC):
    @abstractmethod
    def translate(self, t: np.ndarray, **kwargs):
        """
        Translate the spatial object.

        Method to be overwritten by subclasses.

        Parameters
        ----------
        t : np.ndarray (3,)
            The translation vector.
        **kwargs
            Subclass specific arguments.
        """
        ...

    @abstractmethod
    def scale(self, s: float):
        """
        Scale the curve.

        Method to be overwritten by subclasses.

        Parameters
        ----------
        s : float
            The scale factor.
        **kwargs
            Subclass specific arguments.
        """
        ...

    @abstractmethod
    def rotate(self, r):
        """
        Rotate the curve.

        Method to be overwritten by subclasses.

        Parameters
        ----------
        r : np.ndarray (3, 3)
            The rotation matrix.
        **kwargs
            Subclass specific arguments.
        """
        ...
