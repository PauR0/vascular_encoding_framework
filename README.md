# Vascular Encoding Framework
This project contains the python implementation of the Vascular Encoding Framework to analyze and compare vascular structures. It can be installed as a python library. Additionally a CLI external module is also available under the name of vef_scripts.

## Getting Started
1. First clone the repo locally:
    - With https:
        ```
        git clone https://github.com/PauR0/vascular_encoding_framework.git
        ```
    - With ssh:
        ```
        git clone git@github.com:PauR0/vascular_encoding_framework.git
        ```

2. Install it with pip.
    > :warning: We highly recommend to install the package inside a virtual environment and not at the user-level python interpreter.

    ```
    cd vascular_encoding_framework
    ```
    and then,
    ```
    pip install .
    ```
3. To check if the installation succeeded it may be worth trying to run a tutorial.

## Citing the project.
Right now, the two main ways to cite this project are citing the repo:

The preferred one, is citing the [paper](https://doi.org/10.1016/j.amc.2024.129078) where the Vessel Coordinate System was introduced:
```bibtex
@article{romero2025,
  author   = {Pau Romero and Abel Pedrós and Rafael Sebastian and Miguel Lozano and Ignacio García-Fernández},
  journal  = {Applied Mathematics and Computation},
  title    = {A robust shape model for blood vessels analysis},
  year     = {2025},
  issn     = {0096-3003},
  pages    = {129078},
  volume   = {487},
  doi      = {https://doi.org/10.1016/j.amc.2024.129078},
  url      = {https://www.sciencedirect.com/science/article/pii/S0096300324005393},
}
```

Although, there also exists the possibility of citing the repo as is:
```bibtex
@software{VEF,
  author = {Romero, Pau},
  title = {Vascular Encoding Framework},
  url = {https://github.com/PauR0/vascular_encoding_framework},
  year = {2024}
}
```
