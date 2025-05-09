from abc import ABC, abstractmethod

from vascular_encoding_framework._base import attribute_setter


class Spline(ABC):
    """Abstract base class for splines."""

    def __init__(self): ...

    @abstractmethod
    def evaluate(self, **kwargs):
        """Evaluate the spline object at provided domain points."""
        ...

    @abstractmethod
    def __call__(self, **kwargs):
        """Equivalent to evaluate."""
        ...

    def set_parameters(self, build=False, **kwargs):
        """
        Set parameters and attributes by kwargs.

        Parameters
        ----------
        build : bool, opt
            Default False. If run build setting the params.
        """

        attribute_setter(self, **kwargs)

        if build:
            self.build()

    @abstractmethod
    def build(self):
        """Build the spline objects."""
        ...
