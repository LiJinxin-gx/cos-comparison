#database tools

from ...core import no_done
from ...interface.api import DATABASE_DRIVER, DatabaseToolWrap
from .basememory import *


def _conn_method(name):
    """Build a delegation function for a Connection object method:
    func(conn, *args, **kwargs) == getattr(conn, name)(*args, **kwargs),
    so Connection type methods fit directly (e.g. sqlite3.Connection.cursor).
    """
    def func(conn, *args, **kwargs):
        return getattr(conn, name)(*args, **kwargs)
    return func

class DatabaseMemory(Memory):
    """Wrap the database storage procedure around one connector.

    * ``connect`` establishes the connector; the other methods transcribe
      the corresponding Connection methods (cursor / commit / rollback /
      close / execute / executemany).
    * Every transcription is a delegation function injected at ``__init__``
      (connect_func / cursor_func / ...), receiving the connector as its
      FIRST argument, so Connection type methods can be passed directly
      (e.g. cursor_func=sqlite3.Connection.cursor).
    * ``connect_func`` defaults to DatabaseToolWrap.connect; ``execute`` /
      ``executemany`` remember the returned cursor (``last_cursor``) and
      expose a small PEP 249 convenience surface on it.
    """
    __slots__ = (
        "close_func",
        "commit_func",
        "connect_func",
        "connector",
        "cursor_func",
        "execute_func",
        "executemany_func",
        "last_cursor",
        "rollback_func",
        "wrap",
    )
    def __init__(self, database_tool=None, database=":memory:", refer_func=None,
                 connect_func=None, cursor_func=None, commit_func=None,
                 rollback_func=None, close_func=None, execute_func=None,
                 executemany_func=None):
        tool = DATABASE_DRIVER if database_tool is None else database_tool
        if tool is None:
            raise RuntimeError("DatabaseMemory: no database driver available, provide one via database_tool=")
        self.wrap = DatabaseToolWrap(tool)
        self.connect_func = connect_func if connect_func is not None else self.wrap.connect
        self.connector = None
        self.last_cursor = None
        super().__init__(None, refer_func=refer_func if refer_func else no_done)
        # base Memory.__init__ resets the same-named delegation slots to
        # no_done; re-establish the connection-method transcriptions here
        self.cursor_func = cursor_func if cursor_func is not None else _conn_method("cursor")
        self.commit_func = commit_func if commit_func is not None else _conn_method("commit")
        self.rollback_func = rollback_func if rollback_func is not None else _conn_method("rollback")
        self.close_func = close_func if close_func is not None else _conn_method("close")
        self.execute_func = execute_func if execute_func is not None else _conn_method("execute")
        self.executemany_func = executemany_func if executemany_func is not None else _conn_method("executemany")
        self.connect(database)
    def __enter__(self):
        return self
    def __exit__(self,exc_type,exc_val,exc_tb):
        self.close()
    def connect(self, database=":memory:"):
        """Connection operation: establish the connector (delegated)."""
        self.connector = self.connect_func(database)
        self.memory = self.connector
        self.last_cursor = None
        return self.connector
    def cursor(self, *args, **kwargs):
        """Transcribe Connection.cursor (connector as first argument)."""
        return self.cursor_func(self.connector, *args, **kwargs)
    def commit(self, *args, **kwargs):
        """Transcribe Connection.commit (connector as first argument)."""
        return self.commit_func(self.connector, *args, **kwargs)
    def rollback(self, *args, **kwargs):
        """Transcribe Connection.rollback (connector as first argument)."""
        return self.rollback_func(self.connector, *args, **kwargs)
    def close(self, *args, **kwargs):
        """Transcribe Connection.close (connector as first argument)."""
        result = self.close_func(self.connector, *args, **kwargs)
        self.last_cursor = None
        return result
    def execute(self, command, arg=()):
        """Transcribe Connection.execute; the returned cursor is remembered
        as ``last_cursor`` and returned."""
        self.last_cursor = self.execute_func(self.connector, command, arg)
        return self.last_cursor
    def executemany(self, command, args=()):
        """Transcribe Connection.executemany; the returned cursor is
        remembered as ``last_cursor`` and returned."""
        self.last_cursor = self.executemany_func(self.connector, command, args)
        return self.last_cursor
    # ------------------------------------------------------------------
    # PEP 249 convenience surface (on the last executed cursor)
    # ------------------------------------------------------------------
    def fetchone(self):
        if self.last_cursor is None:
            return None
        return self.last_cursor.fetchone()
    def fetchmany(self, size=None):
        if self.last_cursor is None:
            return []
        if size is None:
            size = getattr(self.last_cursor, "arraysize", 1)
        return self.last_cursor.fetchmany(size)
    def fetchall(self):
        if self.last_cursor is None:
            return []
        return self.last_cursor.fetchall()
    @property
    def rowcount(self):
        if self.last_cursor is None:
            return -1
        return self.last_cursor.rowcount
    @property
    def description(self):
        if self.last_cursor is None:
            return None
        return self.last_cursor.description
    @property
    def arraysize(self):
        if self.last_cursor is None:
            return 1
        return getattr(self.last_cursor, "arraysize", 1)
    @arraysize.setter
    def arraysize(self,value):
        if self.last_cursor is not None:
            self.last_cursor.arraysize = value
    @property
    def lastrowid(self):
        if self.last_cursor is None:
            return None
        return getattr(self.last_cursor, "lastrowid", None)
