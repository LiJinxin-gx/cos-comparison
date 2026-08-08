"""
The module is a project to explore AGI.
"""

try:
    import os.path as _osp

    _script_dir = _osp.dirname(_osp.abspath(__file__))
    _file_path = _osp.join(_script_dir, "VERSION.txt")

    with open(_file_path, "r") as _file:
        __version__ = _file.read()

    del _file
    del _file_path
    del _script_dir
except Exception:
    __version__ = "0.4"

version = __version__
version_tuple = tuple(__version__.split("."))