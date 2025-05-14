from __future__ import annotations

from typing import Any, Generic, Type, TypeVar

import numpy as np

from .encoding import Encoding
from .tree import Tree

_ET = TypeVar("_ET")


class EncodingTree(Tree[_ET], Generic[_ET], Encoding):
    """Abstract class for Encoding trees."""

    def __init__(self, _node_type: Type[_ET]) -> None:
        """Initialize EncodingTree class."""

        # The elements of the tree must be a subtype of Encoding themselves.
        if not issubclass(_node_type, Encoding):
            raise ValueError(
                f"Expecting a subclass of Encoding. Provided {_node_type.__name__} is not."
            )

        Tree.__init__(self=self, _node_type=_node_type)

        self._hyperparameters = []
        Encoding.__init__(self=self)

    def get_hyperparameters(self) -> dict[str, Any]:
        """
        Get the dict containing the hyperparameters of the Encoding object.

        Returns
        -------
        hp : dict[str, Any]
            The json serializable dictionary with the hyperparameters of the encoding.
        """

        self._hyperparameters = list(self.keys())
        return super().get_hyperparameters(**self)

    def set_hyperparameters(self, hp: dict[str, Any], roots: list[str] = None):
        """
        Set the hyperparameters of an EncodingTree object.

        Note that this method initalizes the -potentially missing- Encoding elements present in the
        hyperparameters dictionary.

        Parameters
        ----------
        hp : dict[str, Any]
            The hyperparameter's dictionary.
        roots: list[str], optional
            Default None. A list containing the roots of th
        """

        if roots is None:
            roots = [rid for rid, _hp in hp.items() if _hp["parent"] is None]

        def set_enc_hp(bid):
            enc = self[bid] if bid in self else self._node_type()
            enc.set_hyperparameters(hp=hp[bid])
            self[bid] = enc

            for cid in enc.children:
                set_enc_hp(cid)

        for rid in roots:
            set_enc_hp(rid)

    def get_feature_vector_length(self):
        """
        Return the length of the feature vector.

        The length of a EncodingTree feature vector is the sum of the length of all
        the feature vectors of the encodings contained in it.

        Returns
        -------
        n : int
            The length of the centerline feature vector.
        """
        n = 0
        for enc in self.values():
            n += enc.get_feature_vector_length()

        return n

    def to_feature_vector(self, **kwargs) -> np.ndarray:
        """
        Convert the EncodingTree to a feature vector.

        The feature vector version of an EncodingTree consist in the appending of its Encoding
        objects in a alphabetic-inductive order. This is, the first root branch is picked in
        alphabetic order, then its first children in alphabetic order, and so on, and so on.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments that are passed to Encoding elements.

        Returns
        -------
        fv : np.ndarray (N,)
            The feature vector with the selected data.

        See Also
        --------
        from_feature_vector
        VesselAnatomyEncoding.to_feature_vector
        VesselAnatomyEncoding.from_feature_vector
        """

        fv = []

        def append_fv(vid):
            fv.append(self[vid].to_feature_vector(**kwargs))
            for cid in sorted(self[vid].children):
                append_fv(cid)

        for rid in sorted(self.roots):
            append_fv(rid)

        fv = np.concatenate(fv)
        return fv

    def from_feature_vector(self, fv: np.ndarray, hp: dict[str, Any] = None) -> EncodingTree:
        """
        Build an EncodingTree object from a feature vector.

        > Note that while hyperparameters argument is optional it must have been previously set or
        passed.

        Parameters
        ----------
        fv : np.ndarray
            The feature vector.
        hp : dict[str, Any], optional
            The hyperparameter dictionary for the EncodingTree object.

        Returns
        -------
        enc : EncodingTree
            The object itself with the elements built from the fv.

        See Also
        --------
        get_hyperparameters
        set_hyperparameters
        to_feature_vector
        """

        if hp is not None:
            self.set_hyperparameters(hp=hp)

        n = self.get_feature_vector_length()
        if len(fv) != n:
            raise ValueError(
                f"Cannot build a {self.__class__.__name__} object from feature vector. Expected a"
                + f"feature vector of length {n} and the one provided has {len(fv)} elements."
            )

        ini = 0

        def extract_encoding_fv(vid):
            nonlocal ini
            enc = self[vid]
            end = ini + enc.get_feature_vector_length()
            enc.from_feature_vector(fv=fv[ini:end])
            ini = end

            for cid in sorted(enc.children):
                extract_encoding_fv(cid)

        for rid in sorted(self.roots):
            extract_encoding_fv(rid)

        return self
