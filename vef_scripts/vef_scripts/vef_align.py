import vascular_encoding_framework as vef

from .config.readers import read_alignment_config
from .config.writers import write_alignment_config
from .vef_cohort import load_cohort_object, save_cohort_object


def align_encodings(cohort_dir, params=None, exclude=None, overwrite=False):
    """
    Use the Generalized Procrustes Alignment to align VascularAnatomyEncodings from a given cohort.

    Arguments:
    ---------
        cohort_dir : str
            Path to a cohort directory with case directories in it.

        exclude : list[str]
            A list of directories to exclude from alignment.

        params : dict
            The parameters to be passed to GeneralizedProcrustesAlignment


    Returns
    -------
        gpa : vef.GeneralizedProcrustesAlignment
            The gpa object used.

    """

    if params is None:
        params = read_alignment_config(path=cohort_dir)

    encodings = load_cohort_object(
        cohort_dir=cohort_dir, which="encoding", exclude=exclude, keys_from_dirs=True
    )

    gpa = vef.GeneralizedProcrustesAlignment()
    gpa.set_parameters(build=True, **params)
    gpa.data_set = encodings
    gpa.run()

    save_cohort_object(
        cohort_dir=cohort_dir, cohort=gpa.data_set, suffix="_aligned", overwrite=overwrite
    )
    write_alignment_config(path=cohort_dir, data=params)


#
