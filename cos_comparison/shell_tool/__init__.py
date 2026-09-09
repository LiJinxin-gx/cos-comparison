"""
shell_tool - plugin command directory.

Every .py file is a standalone command script (file name = command name),
executed by __main__ via runpy with no registration.  load_plugin()
imports a plugin on demand; list_plugins() scans the directory.
"""

import importlib
import os

__all__ = ("list_plugins", "load_plugin")

_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))


def list_plugins():
    """List plugin command names by scanning the directory files."""
    return sorted(f[:-3] for f in os.listdir(_PLUGIN_DIR)
                  if f.endswith(".py") and not f.startswith("_"))


def load_plugin(name):
    """Lazily import a plugin module by command name."""
    return importlib.import_module(__package__ + "." + name)
