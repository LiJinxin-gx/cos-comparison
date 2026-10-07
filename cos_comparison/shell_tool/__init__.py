"""
shell_tool - plugin command directory.

Command scripts are standalone .py files under this directory carrying a
``__main__`` entry guard (file name = command name); they are executed by
__main__ via runpy with no registration.  Helper modules without the
guard (shared libraries such as value.py) live alongside and are not
commands.  load_plugin() imports a plugin module on demand; list_plugins()
scans the directory.
"""

import importlib
import os

__all__ = ("list_plugins", "load_plugin")

_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))


def _is_command(path):
    """A command script carries a top-level ``__main__`` entry guard."""
    try:
        with open(path, encoding="utf-8-sig") as fh:
            for line in fh:
                if line.lstrip().startswith("if __name__"):
                    return True
    except (OSError, UnicodeDecodeError):
        return False
    return False


def list_plugins():
    """List plugin command names by scanning the directory files."""
    return sorted(f[:-3] for f in os.listdir(_PLUGIN_DIR)
                  if f.endswith(".py") and not f.startswith("_")
                  and _is_command(os.path.join(_PLUGIN_DIR, f)))


def load_plugin(name):
    """Lazily import a plugin module by command name."""
    return importlib.import_module(__package__ + "." + name)
