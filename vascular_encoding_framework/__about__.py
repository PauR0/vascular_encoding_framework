import os.path

__all__ = [
    "__title__",
    "__summary__",
    "__url__",
    "__version__",
    "__author__",
    "__email__",
    #"__license__",
]


try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = None


__title__ = "vascular_encoding_framework"
__summary__ = "A python implementation of the vascular encoding framework and vessel coordinate system."
__url__ = "https://github.com/PauR0/vascular_encoding_framework"

__version__ = "0.0.1"

if base_dir is not None and os.path.exists(os.path.join(base_dir, ".commit")):
    with open(os.path.join(base_dir, ".commit")) as fp:
        __commit__ = fp.read().strip()
else:
    __commit__ = None

__author__ = "Pau Romero"
__email__ = "pau.romero@uv.es"

#__license__ = ""
