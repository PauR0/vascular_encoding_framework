__all__ = [
    "Node",
]

from copy import deepcopy

import numpy as np

from .misc import attribute_setter, is_arrayable


class Node:
    """Abstract class for tree node."""

    def __init__(self, nd=None) -> None:
        self.id = None
        self.parent = None
        self.children: set = set()

        if nd is not None:
            self.set_data_from_other_node(nd=nd)

    def __str__(self):
        long_atts = ["points", "faces"]
        strout = "\n".join(
            [
                f"{k}".ljust(20, ".") + f": {v}"
                for k, v in self.__dict__.items()
                if k not in long_atts
            ]
        )
        for att in long_atts:
            if att in self.__dict__:
                val = getattr(self, att)
                if val is not None:
                    n = len(val)
                    if att == "faces":
                        n /= 4
                    strout += f"\nn_{att}".ljust(20, ".") + f": {n}"

        return strout

    def set_data(self, to_numpy=True, **kwargs):
        """
        Set attributes by means of kwargs.

        If to_numpy lists containing floats or lists of floats
        will be tried to turn into numpy arrays.

        E.g.
            a = Node()
            a.set_data(center=np.zeros((3,)))

        Parameters
        ----------
        to_numpy : bool, opt
            Default True. Whether to try to cast numerical sequences to
            numpy arrays.
        """

        if "children" in kwargs:
            self.children = set(kwargs["children"])
            kwargs.pop("children")

        if to_numpy:
            kwargs_np = deepcopy(kwargs)
            for k, v in kwargs.items():
                if v is not None:
                    if is_arrayable(v):
                        kwargs_np[k] = k = np.array(v)
            kwargs = kwargs_np

        attribute_setter(self, **kwargs)

    def set_data_from_other_node(self, nd, extra: list[str] = None):
        """
        Copy the Node attribute from other Node object into this.

        Note that only the default Node attributes defined in the
        class constructor will be copied.

        Parameters
        ----------
        nd : Node
            The node from which attributes will be copied.
        extra: list[str]
            Extra attributes to be set from the node object.
        """

        atts = set(Node().__dict__.keys())
        if extra is not None:
            atts.update(extra)

        self.set_data(**{k: getattr(nd, k) for k in atts})

    def add_child(self, c):
        """Add a child to this branch."""
        self.children.add(c)

    def remove_child(self, c):
        """Remove child. If does not exists, nothing happens."""
        self.children.discard(c)
