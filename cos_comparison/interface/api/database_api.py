"""
database_api.py - database driver module abstraction (distributed).

The ONLY place where database drivers may be introduced; layered modules
consume database capabilities through this abstraction instead of opening
connections directly.
"""

import sqlite3

DATABASE_DRIVER = sqlite3

__all__ = (
    "DATABASE_DRIVER",
    "DatabaseToolWrap",
)


class DatabaseToolWrap:
    """Wrap a database driver MODULE with a unified extensible interface.

    Distributed: each instance carries its own tool and injectable
    delegation slots (default: the tool's method of the same name).
    Does not abstract connections or cursors.
    """

    __slots__ = ("connect_func", "tool")

    def __init__(self, tool=None, connect_func=None):
        self.tool = DATABASE_DRIVER if tool is None else tool
        if connect_func is None:
            # default: call the tool's own method of the same name directly
            connect_func = lambda *args, **kwargs: getattr(  # noqa: B009
                self.tool, "connect")(*args, **kwargs)
        self.connect_func = connect_func

    def connect(self, *args, **kwargs):
        """Delegated call of the tool's ``connect`` method."""
        return self.connect_func(*args, **kwargs)

    def call(self, name, *args, **kwargs):
        """Call any tool module method through one uniform entry point."""
        return getattr(self.tool, name)(*args, **kwargs)
