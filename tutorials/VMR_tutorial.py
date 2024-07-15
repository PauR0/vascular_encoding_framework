"""
Basic tutorial of the Vascular Encoding Framework using a model publicly available on the
Vascular Model Repository (https://www.vascularmodel.com/index.html).

To run this tutorial the user need to donwload the case 0010_H_AO_H It can be found
using the search functionality on the filters tab of the repository web.

After downloading and unziping it, the user can either move the directory inside the tutorials
directory or modify this file to set the path to the unziped directory.

- Works with conda final_vascular_env
"""


import os
import sys

import numpy as np
import pyvista as pv
import vascular_encoding_framework as vef
from vascular_encoding_framework.jose.casepath import CasePath
from vascular_encoding_framework.utils.graphic import plot_adapted_frame

case_path_obj = CasePath()
case_path = case_path_obj.get_case_path()
print(f"Final case directory: {case_path}")

mesh_path = os.path.join(case_path, 'Meshes', '0093_0001.vtp')

mesh = pv.read(mesh_path)
mesh = mesh.threshold(value=0.1, scalars='CapID',
                      method='lower').extract_surface()

# Smooth the mesh to reduce potential segmentation artifacts
# mesh = mesh.smooth(n_iter=0)  # Increased smoothing iterations

mesh = mesh.compute_normals(
    auto_orient_normals=True, flip_normals=False, consistent_normals=True
)

mesh = mesh.subdivide(1)  # Increase the level of subdivision if necessary

# Initialize vef.VascularMesh with the opened mesh
vmesh = vef.VascularMesh(mesh)

vmesh.plot_boundary_ids()
# Define the hierarchy
hierarchy = {
    "5": {"id": "5", "parent": None, "children": {"0"}},
    "0": {"id": "0", "parent": "5", "children": {"3", "4", "1"}},
    "3": {"id": "3", "parent": "0", "children": {"2"}},
    "4": {"id": "4", "parent": "0", "children": {}},
    "1": {"id": "1", "parent": "0", "children": {}},
    "2": {"id": "2", "parent": "3", "children": {}},
}
vmesh.set_boundary_data(hierarchy)

print("Initial boundaries:")

# Check boundary connectivity
for boundary_id, boundary in vmesh.boundaries.items():
    print(f"Boundary {boundary_id} connected to: {boundary.children}")

# Function to extract centerline domain
# Attempt to extract the centerline domain with adjusted parameters
c_domain = vef.centerline.extract_centerline_domain(
    vmesh=vmesh,
    params={'method': 'seekers', 'reduction_rate': 0, 'eps': 1e-3},
    debug=True
)

# Compute the path tree
cp_xtractor = vef.centerline.CenterlinePathExtractor()
cp_xtractor.debug = False
cp_xtractor.set_centerline_domain(c_domain)
cp_xtractor.set_vascular_mesh(vmesh, update_boundaries=True)
cp_xtractor.compute_paths()

# Check if paths were computed successfully
if cp_xtractor.paths is None or cp_xtractor.paths.n_blocks == 0:
    print("Failed to compute paths. Unable to create CenterlineNetwork.")
    sys.exit()

# Function to log path computation


def log_path_computation(start, end):
    print(f"Attempting to compute path from boundary {start} to {end}")
    try:
        path_ids = cp_xtractor.boundaries[start].id_path
        if path_ids is None or len(path_ids) == 0:
            print(f"No valid path found from boundary {start} to {end}")
            return None
        path = c_domain.points[path_ids]
        print(f"Path found: {len(path)} points")
        return path
    except Exception as e:
        print(f"Error computing path: {str(e)}")
        return None


# Compute paths manually
paths = []
for start, data in hierarchy.items():
    for end in data['children']:
        path = log_path_computation(start, end)
        if path is not None and len(path) > 0:
            paths.append(path)

# Visualize the paths
for i, path in enumerate(paths):
    plotter = pv.Plotter(off_screen=False)
    plotter.add_mesh(mesh, opacity=0.5)  # Use the original mesh loaded
    plotter.add_mesh(c_domain, color='red', point_size=2)
    plotter.add_mesh(pv.PolyData(path), color='green', line_width=5)
    plotter.show(screenshot=f'path_{i}.png')

print(f"Number of centerline domain points: {len(c_domain.points)}")
print(f"Centerline domain bounding box: {c_domain.bounds}")

# Define knots for each branch
knot_params = {
    "5": {"cl_knots": None, "tau_knots": None, "theta_knots": None},
    "0": {"cl_knots": 15, "tau_knots": 19, "theta_knots": 19},
    "3": {"cl_knots": 15, "tau_knots": 10, "theta_knots": 10},
    "4": {"cl_knots": 15, "tau_knots": 10, "theta_knots": 10},
    "1": {"cl_knots": 15, "tau_knots": 10, "theta_knots": 10},
    "2": {"cl_knots": 15, "tau_knots": 10, "theta_knots": 10},
}

try:
    cl_net = vef.CenterlineNetwork.from_multiblock_paths(
        cp_xtractor.paths, knots={k: v['cl_knots']
                                  for k, v in knot_params.items()}
    )

    # Plot the adapted frame
    plot_adapted_frame(cl_net, vmesh, scale=0.5)

    # Compute centerline association and vessel coordinates
    bid = [
        cl_net.get_centerline_association(
            p=vmesh.points[i],
            n=vmesh.get_array(name='Normals', preference='point')[i],
            method='scalars',
            thrs=60,
        )
        for i in range(vmesh.n_points)
    ]
    vcs = np.array(
        [
            cl_net.cartesian_to_vcs(p=vmesh.points[i], cl_id=bid[i])
            for i in range(vmesh.n_points)
        ]
    )
    vmesh['cl_association'] = bid
    vmesh['tau'] = vcs[:, 0]
    vmesh['theta'] = vcs[:, 1]
    vmesh['rho'] = vcs[:, 2]

    # Print data statistics
    print("Tau values - Mean:",
          np.mean(vmesh['tau']), "Std:", np.std(vmesh['tau']))
    print(
        "Theta values - Mean:", np.mean(vmesh['theta']
                                        ), "Std:", np.std(vmesh['theta'])
    )
    print("Rho values - Mean:",
          np.mean(vmesh['rho']), "Std:", np.std(vmesh['rho']))

except Exception as e:
    print(f"Error in creating CenterlineNetwork or processing data: {str(e)}")

print("Script completed.")
