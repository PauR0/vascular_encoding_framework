#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Literal, Dict, Any


import numpy as np
import pyvista as pv


from ..messages import *
from ..encoding import VascularEncoding
from ..utils._code import attribute_checker, attribute_setter


from .alignment import Alignment, IterativeClosestPoint, RigidProcrustesAlignment


def as_an_array(obj):
    """
    Function to extract the array expression of different objects.
    Supported objects are: [np.ndarray, pv.DataObject, VascularEncoding]

    This function has no effect on numpy arrays.

    Arguments
    ---------

        obj : np.ndarray, pv.DataObject, VascularEncoding
            The object of which the array will be extracted.

    Returns
    -------

        arr : np.ndarray
            The extracted array
    """

    if not isinstance(obj, [np.ndarray, pv.DataObject, VascularEncoding]):
        return None

    if isinstance(obj, np.ndarray):
        return obj

    if isinstance(obj, pv.DataObject):
        return obj.points

    if isinstance(obj, VascularEncoding):
        return obj.to_feature_vector(centerline=True, radius=False)

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

        self.data_set       : Dict[str : np.ndarray | pv.DataObject | VascularEncoding] = None
        self.n_iters        : int = None
        self.reference_id   : int | str = 0 #The key (or its index) of the shape to use in the
                                            #first iteration as the mean shape.
    #

    def set_parameters(self, build=True, **kwargs):
        """
        Method to set parameters as attributes of the object.
        """
        cl = self.__class__()
        params = {k:v for k,v in kwargs.items() if k in cl.__dict__}
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
        """

        if not attribute_checker(self, atts=['data_set'], info="Can't compute mean shape."):
            return None

        return np.mean([as_an_array(v) for _, v in self.data_set.items()], axis=0)

    def run(self):
        """
        Compute the GPA over the data set.
        """

        if not attribute_checker(self, ['alignment', 'data_set'], info="Cant run Generalized Procrustes Alignment.")
            return

        if isinstance(self.reference_id, int):
            if self.reference_id >= len(self.data_set):
                error_message(f"The reference_id ({self.reference_id}) is greater than the amount \
                              of elements in data set ({len(self.data_set)}). Using the id at first \
                              position.")
                self.reference_id = 0

            self.reference_id = list(self.data_set.keys())[self.reference_id]

        mean_shape = self.data_set[self.reference_id]
        self.alignment.target = mean_shape
        n_iter = 0
        while n_iter < self.n_iters:

            for sid, shape in self.data_set:
                self.data_set[sid] = self.alignment.run(apply=True)

            mean_shape = self.compute_mean_shape()

            n_iter += 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generalized Procrustes analysis of an aorta family.")

    parser.add_argument('pop_path',
                        nargs='?',
                        type=str,
                        help = "Path to the aorta family directory.")

    parser.add_argument('-s',
                        '--save',
                        nargs='?',
                        type=str,
                        default='',
                        const='all',
                        dest ='s',
                        help = """If used with no args, it saves registered mean mesh case, meshes,
                        registered feature vectors. A list of options can be used as follows:
                            '-s mean' to save only mean case,
                            '-s mesh' to save the meshes or,
                            '-s fv' to save the feature vectors,
                            '-s fv,mean' to save both the fv and the mean.""")

    parser.add_argument('--reference-shape',
                        type=check_nonnegative,
                        default=0,
                        dest='rs',
                        help = "The id number of the chosen initial reference shape.")

    parser.add_argument('-m',
                        '--max-iter',
                        type=check_positive,
                        default=1,
                        dest='max_iter',
                        help = "The maximum number of iteration for the GPA.")

    parser.add_argument('--ignore',
                        type=str,
                        nargs='?',
                        default='',
                        dest='ig',
                        help = "Directories in pop_path to ignore. Functions default include Log and mean_case dir")

    parser.add_argument('-r',
                        '--params-reg',
                        type=str,
                        nargs='?',
                        default='',
                        dest='r',
                        help = "Absolute path to a json registrate parameter file.")

    parser.add_argument('-c',
                        '--params-ch',
                        type=str,
                        nargs='?',
                        default='',
                        dest='c',
                        help = "Absolute path to a json characterize parameter file.")

    parser.add_argument('--params-rec',
                        type=str,
                        nargs='?',
                        default='',
                        dest='rec',
                        help = "Absolute path to a json reconstruct parameter file.")



    args = parser.parse_args()
    pop_path = args.pop_path

    if pop_path is None or not os.path.exists(pop_path):
        print("Wrong path given")
        exit(1)


    generalized_procrustes_alignment(pop_path = pop_path,
                                     save=args.s,
                                     rs0=args.rs,
                                     max_iter=args.max_iter,
                                     ignore_dirs=args.ig)
