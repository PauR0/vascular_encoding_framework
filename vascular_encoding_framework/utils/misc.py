"""
General purpose functions
"""


def split_metadata_and_fv(fv):
    """
    This function splits the metadata array from the feature vector.

    This function assumes that the length of the metadata array is the first element
    of the feature vector passed.

    Arguments
    ---------

        fv : np.ndarray
            The feature vector with the metadata at the beggining.

    Returns
    -------

        : np.ndarray
            The metada and feature vector respectively.
    """

    n_md = round(fv[0])
    return fv[:n_md+1], fv[n_md+1:]
#
