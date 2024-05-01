#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Literal, Dict, Any


import numpy as np
import pyvista as pv


from ..messages import *
from ..encoding.encoding import Encoding
from ..utils._code import attribute_checker, attribute_setter


from .alignment import Alignment, IterativeClosestPoint, RigidProcrustesAlignment, as_an_array


class GeneralizedProcrustesAlignment:
    """
    Class with the GPA algorithm.

    TODO: The implementation aligns the centerline network as a whole using the same transformation
    for all the branches. Since the objective of the GPA is to remove spatial artifacts, it would
    be interesting to implement a decoupled version of the GPA. This version could compute a
    per-branch alignment storing previously the actual origin and tangent of the centerline using
    paren'ts VCS. This would be compatible with PCA and pretty much every other algorithm that uses
    the encoding...
    """

    def __init__(self):

        self.data_set : Dict[str : np.ndarray | pv.DataObject] = None

        self.alignment_method : Literal['procrustes', 'ICP'] = 'procrustes'
        self.alignment_params : Dict[str:Any]                = None
        self.alignment        : Alignment                    = None

        self.n_iters        : int = None
        self.data_set       : Dict[str : np.ndarray | pv.DataObject | Encoding] = None
        self.reference_id   : int | str = 0 #The key (or its index) of the shape to use in the
                                            #first iteration as the mean shape.
    #

    def set_parameters(self, build=True, **kwargs):
        """
        Method to set parameters as attributes of the object.
        """
        gpa = self.__class__()
        params = {k:v for k,v in kwargs.items() if k in gpa.__dict__}
        attribute_setter(self, **params)

        if build:
            self.build_alignment()
    #

    def build_alignment(self):
        """
        Set the alignment attribute and its parameters using alignment_method and alignment_params atts.

        Warning: This method overwrites the existing alignment attribute and its current parameter.
        """

        if not attribute_checker(self, ['alignment_method'], info="Cannot set alignment.", opts=[['procrustes', 'ICP']]):
            return

        if self.alignment_method == 'ICP':
            self.alignment = IterativeClosestPoint()
        elif self.alignment_method == 'procrustes':
            self.alignment = RigidProcrustesAlignment()
        else:
            error_message("Something must've gone wrong before getting to this point.")

        if self.alignment_params is not None:
            self.alignment.set_parameters(**self.alignment_params)
    #

    def compute_mean_shape(self):
        """
        Compute the mean shape of the data set.

        This is called at the beginning of every iteration after the first, to compute the
        reference shape to which align the whole data set.

        Returns
        -------
            : np.ndarray
                The mean point set.
        """

        if not attribute_checker(self, atts=['data_set'], info="Can't compute mean shape."):
            return None

        return np.mean([as_an_array(v) for _, v in self.data_set.items()], axis=0)
    #

    def run(self):
        """
        Compute the GPA over the data set.
        """

        if not attribute_checker(self, ['alignment', 'data_set'], info="Cant run Generalized Procrustes Alignment."):
            return

        if isinstance(self.reference_id, int):
            if self.reference_id >= len(self.data_set):
                error_message(f"The reference_id ({self.reference_id}) is greater than the amount \
                              of elements in data set ({len(self.data_set)}). Using the id at first \
                              position.")
                self.reference_id = 0

            self.reference_id = list(self.data_set.keys())[self.reference_id]

        self.alignment.target = self.data_set[self.reference_id]
        n_iter = 0
        while n_iter < self.n_iters:

            for sid in self.data_set:
                self.alignment.source = self.data_set[sid]
                self.data_set[sid] = self.alignment.run(apply=True)

            self.alignment.target = self.compute_mean_shape()
            n_iter += 1
    #