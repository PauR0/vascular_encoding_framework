
# Vascular Encoding Framework

## Introduction

The Vascular Encoding Framework (VEF) is a Python-based tool designed to analyze and compare vascular structures. This framework provides a systematic approach to process, visualize, and extract meaningful data from 3D vascular meshes. By leveraging advanced computational techniques, VEF can generate centerlines, define boundary hierarchies, and convert Cartesian coordinates into a Vessel Coordinate System (VCS).

Furthermore, the Vascular Encoding Framework assists in obtaining a precise and low-dimensional geometric representation of vascular structures. This geometric data is particularly useful for subsequent fluid simulations, providing a stable and accurate foundation for computational fluid dynamics (CFD) models.

This tutorial guides users through the essential steps to utilize VEF effectively. It includes instructions on installing the necessary dependencies and loading and preparing vascular meshes for the usage. Whether you are a researcher working on vascular modeling, a bioengineer developing computational models of blood flow, or a student learning about biomedical data analysis, this tutorial will help you understand and apply the VEF to your projects.




## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Program](#running-the-program)
- [Usage](#usage)
  - [Library Import](#library-import)
  - [Case Study Configuration](#case-study-configuration)
  - [Mesh Loading and Preparation](#mesh-loading-and-preparation)
  - [VascularMesh Initialization](#vascularmesh-initialization)
  - [Boundary Hierarchy Definition](#boundary-hierarchy-definition)
  - [Centerline Extraction](#centerline-extraction)
  - [Centerline Path Calculation](#centerline-path-calculation)
  - [Branch Node Definition](#branch-node-definition)
  - [Centerline Network Creation](#centerline-network-creation)
  - [Adapted Frame Visualization](#adapted-frame-visualization)
  - [Vessel Coordinates Calculation](#vessel-coordinates-calculation)
  - [Results Visualization](#results-visualization)
- [Contributions](#contributions)
- [License](#license)

## Requirements

- Python 3.x
- Conda (recommended for environment management)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/PauR0/vascular_encoding_framework.git
    ```

2. Create a virtual environment and install dependencies:

    ```bash
    conda create -n vef_env python=3.11
    conda activate vef_env
    ```

## Running the Program

Basic tutorial of the Vascular Encoding Framework using a model publicly available on the Vascular Model Repository (https://www.vascularmodel.com/index.html).

To run this tutorial, the user needs to download the case 0010_H_AO_H. It can be found using the search functionality on the filters tab of the repository web.

After downloading and unzipping it, the user can either move the directory inside the tutorials directory or modify this file to set the path to the unzipped directory.




1. To ensure that the code correctly references your working directory, you need to modify the case_path variable. If your directory structure is /home/user/desktop/vascular_encoding_framework/tutorials/0010_H_AO_H, you can set it as follows:

    ```bash
    case_path = f"{os.path.expanduser('~')}/desktop/vascular_encoding_framework/tutorials/0010_H_AO_H"
    ```

1. Run the script:

    ```bash
    python VMR_tutorial.py
    ```

## Usage

### Library Import

```python
import os
import numpy as np
import pyvista as pv
import vascular_encoding_framework as vef
from vascular_encoding_framework.jose.casepath import CasePath
```

### Case Study Configuration

```python
case_path_obj = CasePath()
case_path = case_path_obj.get_case_path()
mesh_path = os.path.join(case_path, 'Meshes', '0093_0001.vtp')
```

### Mesh Loading and Preparation

```python
mesh = pv.read(mesh_path)
mesh = mesh.threshold(value=0.1, scalars='CapID', method='lower').extract_surface()
mesh = mesh.smooth(n_iter=100)
mesh = mesh.subdivide(1)
```

### VascularMesh Initialization

```python
vmesh = vef.VascularMesh(mesh)
```

### Boundary Hierarchy Definition

```python
vmesh.plot_boundary_ids()
hierarchy = {
    "5": {"id": "5", "parent": None, "children": {"0"}},
    "0": {"id": "0", "parent": "5", "children": {"3", "4", "1"}},
    "3": {"id": "3", "parent": "0", "children": {"2"}},
    "4": {"id": "4", "parent": "0", "children": {}},
    "1": {"id": "1", "parent": "0", "children": {}},
    "2": {"id": "2", "parent": "3", "children": {}},
}
vmesh.set_boundary_data(hierarchy)
```

### Centerline Extraction

```python
c_domain = vef.centerline.extract_centerline_domain(
    vmesh=vmesh,
    method='seekers',
    method_params={'reduction_rate': 0.75, 'eps': 1e-3},
    debug=False
)
```

### Centerline Path Calculation

```python
cp_xtractor = vef.centerline.CenterlinePathExtractor()
cp_xtractor.debug = True
cp_xtractor.set_centerline_domain(c_domain)
cp_xtractor.set_vascular_mesh(vmesh, update_boundaries=True)
cp_xtractor.compute_paths()
```

### Branch Node Definition

```python
knot_params = {
    "5": {"cl_knots": None, "tau_knots": None, "theta_knots": None},
    "0": {"cl_knots": 15, "tau_knots": 19, "theta_knots": 19},
    "3": {"cl_knots": 15, "tau_knots": 10, "theta_knots": 10},
    "4": {"cl_knots": 15, "tau_knots": 10, "theta_knots": 10},
    "1": {"cl_knots": 15, "tau_knots": 10, "theta_knots": 10},
    "2": {"cl_knots": 15, "tau_knots": 10, "theta_knots": 10},
}
```

### Centerline Network Creation

```python
cl_net = vef.CenterlineNetwork.from_multiblock_paths(
    cp_xtractor.paths, 
    knots={k: v['cl_knots'] for k, v in knot_params.items()}
)
```

### Adapted Frame Visualization

```python
plot_adapted_frame(cl_net, vmesh, scale=.5)
```

### Vessel Coordinates Calculation

```python
bid = [
    cl_net.get_centerline_association(
        p=vmesh.points[i], 
        n=vmesh.get_array(name='Normals', preference='point')[i], 
        method='scalars', 
        thrs=60
    ) 
    for i in range(vmesh.n_points)
]
vcs = np.array([
    cl_net.cartesian_to_vcs(p=vmesh.points[i], cl_id=bid[i]) 
    for i in range(vmesh.n_points)
])
vmesh['cl_association'] = bid
vmesh['tau'] = vcs[:, 0]
vmesh['theta'] = vcs[:, 1]
vmesh['rho'] = vcs[:, 2]
```

### Results Visualization

```python
vmesh.plot(scalars='cl_association')
vmesh.plot(scalars='tau')
vmesh.plot(scalars='theta')
vmesh.plot(scalars='rho')
```


## Contributions

Contributions are welcome. Please open an issue or send a pull request.

## License

This project is licensed under the MIT License.

