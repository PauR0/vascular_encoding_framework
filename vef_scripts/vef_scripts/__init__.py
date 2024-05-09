"""
The scripts module out of the vascular_encoding_framework is designed to provide a commandline
interface to the module allowing to perform basic operations such as centerline computation or
encoding. The module is designed to assume a directory structure, which will be called case
directory, as follows:

case_dir/
        Meshes/
            mesh_input.vtk           #The input mesh
            boundaries_input.json    #The input boundaries, (computed or user-defined).
            config.json              #A json with extra configuration parameters.
        Centerline/
            domain.vtk        #A vtk object containing the extracted centerline domain.
            path.vtk          #A vtk MultiBlock containing the centerline with topology information.
            centerline.tbd    #Yet to be determined....
            config.json       #A json with centerline computation configuration parameters.
        Encoding/
            vascular_encode.tbd    #The vascular encoding computed
            config.json            #A json with encoding configuration parameters.

Not all the scripts require the existence of all the files and directories, for instance,
vef_compute_centerline does not require the Encoding directory to exist or to contain anything.

TODO: Study the option of emedding this package onto vef __main__.py at top level.
PEP-338: https://peps.python.org/pep-0338/
"""