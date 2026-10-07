"""io_api tests: file/memory streams and delegation slots."""
import os
import tempfile
import unittest

from cos_comparison.interface.api import IOFile, IOMemory, IOStream


def _close_quietly(stream):
    if stream is None:
        return
    try:
        stream.close()
    except Exception:
        pass


def _remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


class TestIOFile(unittest.TestCase):
    """File stream: lazy open, read/write/seek/tell/readline/close."""

    def test_write_read_roundtrip(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        f = r = None
        try:
            f = IOFile(path, "w")
            f.open()
            f.write("hello")
            f.write(" world")
            f.flush()
            f.close()
            f = None
            r = IOFile(path, "r")
            r.open()
            self.assertEqual(r.read(), "hello world")
            r.close()
            r = None
        finally:
            _close_quietly(f)
            _close_quietly(r)
            _remove_quietly(path)

    def test_seek_tell_readline(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        r = None
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("line1\nline2\n")
            r = IOFile(path, "r")
            r.open()
            self.assertEqual(r.readline(), "line1\n")
            self.assertGreater(r.tell(), 0)   # bytes offset incl. CRLF
            r.seek(0)
            self.assertEqual(r.read(4), "line")
            r.close()
            r = None
        finally:
            _close_quietly(r)
            _remove_quietly(path)

    def test_binary_mode(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        f = r = None
        try:
            f = IOFile(path, "wb")
            f.open()
            f.write(b"\x00\x01")
            f.close()
            f = None
            r = IOFile(path, "rb")
            r.open()
            self.assertEqual(r.read(), b"\x00\x01")
            r.close()
            r = None
        finally:
            _close_quietly(f)
            _close_quietly(r)
            _remove_quietly(path)


class TestIOMemory(unittest.TestCase):
    """In-memory stream: bytes and text."""

    def test_bytes_memory(self):
        m = IOMemory(b"abc")
        m.open()
        self.assertEqual(m.read(2), b"ab")
        self.assertEqual(m.read(), b"c")
        m.write(b"d")
        m.seek(0)
        self.assertEqual(m.read(), b"abcd")
        self.assertEqual(m.getvalue(), b"abcd")

    def test_text_memory(self):
        m = IOMemory("hi", binary=False)
        m.open()
        self.assertEqual(m.readline(), "hi")
        m.write("!")
        m.seek(0)
        self.assertEqual(m.getvalue(), "hi!")


class TestIOStreamDelegation(unittest.TestCase):
    """Delegation slots: injected funcs receive the container."""

    def test_delegated_slots(self):
        log = []
        s = IOStream(obj="o",
                     read_func=lambda self, n: log.append(("r", self.obj, n)),
                     write_func=lambda self, d: log.append(("w", self.obj, d)),
                     close_func=lambda self: log.append(("c", self.obj)))
        s.read(3)
        s.write("x")
        s.close()
        self.assertEqual(log, [("r", "o", 3), ("w", "o", "x"), ("c", "o")])

    def test_no_done_default(self):
        s = IOStream(obj=None)
        self.assertIsNone(s.read())
        self.assertIsNone(s.close())

    def test_falsy_slot_kept(self):
        class FalsyRead:
            def __bool__(self):
                return False
            def __call__(self, container, n):
                return "kept"

        s = IOStream(obj="o", read_func=FalsyRead())
        self.assertEqual(s.read(2), "kept")


if __name__ == "__main__":
    unittest.main()
