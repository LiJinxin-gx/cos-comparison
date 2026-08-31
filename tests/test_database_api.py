"""Tests for the split design: DatabaseToolWrap is a database MODULE tool
abstraction (delegation, default getattr(tool, name)); DatabaseMemory
wraps the storage procedure around one connector and transcribes
Connection object methods through delegation functions passed in at
__init__ (connector as first argument)."""

import sqlite3
import unittest

from cos_comparison.interface.api.database_api import (
    DATABASE_DRIVER,
    DatabaseToolWrap,
)
from cos_comparison.memory_layer.memory.database_memory import DatabaseMemory


class TestToolWrapModuleAbstraction(unittest.TestCase):
    def test_default_tool(self):
        wrap = DatabaseToolWrap()
        self.assertIs(wrap.tool, DATABASE_DRIVER)

    def test_connect_defaults_to_tool_method(self):
        wrap = DatabaseToolWrap()
        conn = wrap.connect(":memory:")
        self.assertIsInstance(conn, sqlite3.Connection)
        conn.close()

    def test_connect_default_calls_getattr(self):
        wrap = DatabaseToolWrap()
        self.assertEqual(wrap.connect_func(":memory:").total_changes, 0)
        wrap.connect(":memory:").close()

    def test_custom_connect_func(self):
        calls = []

        def my_connect(*args, **kwargs):
            calls.append((args, kwargs))
            return sqlite3.connect(*args, **kwargs)

        wrap = DatabaseToolWrap(connect_func=my_connect)
        conn = wrap.connect(":memory:")
        self.assertEqual(calls, [((":memory:",), {})])
        conn.close()

    def test_call_uniform_entry(self):
        wrap = DatabaseToolWrap()
        conn = wrap.call("connect", ":memory:")
        self.assertIsInstance(conn, sqlite3.Connection)
        conn.close()

    def test_custom_tool(self):
        class FakeTool:
            def connect(self, db):
                return ("connected", db)

        wrap = DatabaseToolWrap(tool=FakeTool())
        self.assertEqual(wrap.connect("x"), ("connected", "x"))


class TestDatabaseMemoryConnector(unittest.TestCase):
    def setUp(self):
        self.mem = DatabaseMemory()
        self.mem.execute("CREATE TABLE t (a INTEGER, b TEXT)")
        self.mem.executemany(
            "INSERT INTO t (a, b) VALUES (?, ?)",
            [(1, "one"), (2, "two")])
        self.mem.commit()

    def tearDown(self):
        self.mem.close()

    def test_connector_established(self):
        self.assertIsNotNone(self.mem.connector)
        self.assertIsInstance(self.mem.connector, sqlite3.Connection)
        self.assertIs(self.mem.memory, self.mem.connector)

    def test_connect_method_switches_connector(self):
        conn = self.mem.connect(":memory:")
        self.assertIs(conn, self.mem.connector)

    def test_cursor_transcription(self):
        cur = self.mem.cursor()
        self.assertIsInstance(cur, sqlite3.Cursor)
        self.assertEqual(
            cur.execute("SELECT * FROM t ORDER BY a").fetchall(),
            [(1, "one"), (2, "two")])

    def test_execute_transcribes_connection(self):
        rows = self.mem.execute("SELECT * FROM t ORDER BY a").fetchall()
        self.assertEqual(rows, [(1, "one"), (2, "two")])

    def test_commit_rollback(self):
        mem = DatabaseMemory()
        mem.execute("CREATE TABLE t (a INTEGER)")
        mem.execute("INSERT INTO t (a) VALUES (1)")
        mem.rollback()
        mem.execute("INSERT INTO t (a) VALUES (2)")
        mem.commit()
        self.assertEqual(mem.execute("SELECT a FROM t").fetchall(), [(2,)])
        mem.close()

    def test_context_manager(self):
        with DatabaseMemory() as mem:
            mem.execute("CREATE TABLE t (a INTEGER)")
            mem.execute("INSERT INTO t (a) VALUES (5)")
            mem.commit()
            self.assertEqual(mem.execute("SELECT a FROM t").fetchall(),
                             [(5,)])

    def test_connector_first_argument_type_methods(self):
        mem = DatabaseMemory(
            connect_func=sqlite3.connect,
            cursor_func=sqlite3.Connection.cursor,
            execute_func=sqlite3.Connection.execute,
        )
        mem.execute("CREATE TABLE t (a INTEGER)")
        mem.execute("INSERT INTO t (a) VALUES (7)")
        self.assertEqual(
            mem.cursor().execute("SELECT a FROM t").fetchall(), [(7,)])
        mem.close()

    def test_execute_returns_cursor_and_last_cursor(self):
        cur = self.mem.execute("SELECT * FROM t")
        self.assertIs(cur, self.mem.last_cursor)
        self.assertIsInstance(cur, sqlite3.Cursor)

    def test_rollback_delegation(self):
        mem = DatabaseMemory()
        mem.execute("CREATE TABLE t (a INTEGER)")
        self.assertIsNone(mem.rollback())


class TestDatabaseMemoryPEP249Surface(unittest.TestCase):
    def setUp(self):
        self.mem = DatabaseMemory()
        self.mem.execute("CREATE TABLE t (a INTEGER, b TEXT)")
        self.mem.executemany(
            "INSERT INTO t (a, b) VALUES (?, ?)",
            [(1, "one"), (2, "two"), (3, "three")])
        self.mem.commit()

    def tearDown(self):
        self.mem.close()

    def test_fetchall(self):
        self.mem.execute("SELECT * FROM t ORDER BY a")
        self.assertEqual(self.mem.fetchall(),
                         [(1, "one"), (2, "two"), (3, "three")])

    def test_fetchmany_arraysize(self):
        self.mem.execute("SELECT * FROM t ORDER BY a")
        self.mem.arraysize = 2
        self.assertEqual(len(self.mem.fetchmany()), 2)
        self.assertEqual(len(self.mem.fetchmany()), 1)
        self.assertIsNone(self.mem.fetchone())

    def test_rowcount_description_lastrowid(self):
        self.mem.execute("SELECT a, b FROM t")
        self.assertEqual(self.mem.description[0][0], "a")
        self.mem.execute("UPDATE t SET b = b WHERE a >= 0")
        self.assertEqual(self.mem.rowcount, 3)
        self.assertIsNotNone(self.mem.lastrowid)

    def test_empty_states(self):
        mem = DatabaseMemory()
        self.assertIsNone(mem.fetchone())
        self.assertEqual(mem.fetchall(), [])
        self.assertEqual(mem.fetchmany(), [])
        self.assertEqual(mem.rowcount, -1)
        self.assertIsNone(mem.description)
        mem.close()


if __name__ == "__main__":
    unittest.main()
