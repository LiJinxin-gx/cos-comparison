"""
database_api.py - external database access abstraction.

This module is the ONLY place where database drivers may be introduced.
Layered modules must consume database capabilities through this abstraction
instead of opening external connections directly (Modular Architecture R1).
"""

import sqlite3

DATABASE_DRIVER = sqlite3

__all__ = ("DatabaseToolWrap", "DATABASE_DRIVER")


def default_connect(tool, database):
    """Open a connection using the driver module (e.g. sqlite3.connect)."""
    return tool.connect(database)


def default_cursor(connection, *args, **kwargs):
    """Create a cursor from an existing connection."""
    return connection.cursor(*args, **kwargs)


class DatabaseToolWrap:
    """Wrap a DB driver with pluggable connect / cursor operations."""

    __slots__ = ("tool", "conn_func", "cursor_func")

    def __init__(self, tool=None, conn_func=None, cursor_func=None):
        self.tool = tool
        self.conn_func = conn_func if conn_func is not None else default_connect
        self.cursor_func = cursor_func if cursor_func is not None else default_cursor

    def connect(self, database):
        """Establish a connection to the given database."""
        return self.conn_func(self.tool, database)

    def cursor(self, connection, *args, **kwargs):
        """Create a cursor on the given connection."""
        return self.cursor_func(connection, *args, **kwargs)