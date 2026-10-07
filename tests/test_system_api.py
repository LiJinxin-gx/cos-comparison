"""system_api tests: file management (explicit-stack, no recursion)."""
import hashlib
import os
import tempfile
import unittest

from cos_comparison.interface.api.system_api import (
    FileManager,
    copy_path,
    file_hash,
    file_info,
    file_match,
    find_files,
    list_dir,
    make_dirs,
    move_path,
    read_file,
    remove_path,
    write_file,
)


def _make_tree(root, depth=3, files_per_dir=2):
    """Build a nested directory tree; returns the leaf file path."""
    os.makedirs(root, exist_ok=True)
    leaf = None
    for d in range(depth):
        current = os.path.join(root, "d" + str(d))
        os.makedirs(current, exist_ok=True)
        for f in range(files_per_dir):
            p = os.path.join(current, "f" + str(f) + ".txt")
            with open(p, "wb") as fh:
                fh.write(b"x" * 10)
            leaf = p
        root = current
    return leaf


class TestFileBasics(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sys_")
        self.addCleanup(remove_path, self.dir)

    def test_write_read_roundtrip(self):
        p = os.path.join(self.dir, "a.txt")
        n = write_file(p, "hello", encoding="utf-8")
        self.assertEqual(n, 5)
        self.assertEqual(read_file(p, encoding="utf-8"), "hello")

    def test_write_read_bytes(self):
        p = os.path.join(self.dir, "b.bin")
        write_file(p, b"\x00\x01\xff")
        self.assertEqual(read_file(p), b"\x00\x01\xff")

    def test_file_info(self):
        p = os.path.join(self.dir, "c.txt")
        write_file(p, b"12345")
        info = file_info(p)
        self.assertEqual(info["size"], 5)
        self.assertTrue(info["is_file"])
        self.assertFalse(info["is_dir"])

    def test_file_hash(self):
        p = os.path.join(self.dir, "d.txt")
        write_file(p, b"hashme")
        self.assertEqual(file_hash(p),
                         hashlib.sha256(b"hashme").hexdigest())
        self.assertEqual(len(file_hash(p, algo="md5")), 32)


class TestDirOps(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sys_")
        self.addCleanup(remove_path, self.dir)

    def test_list_dir_flat_and_sorted(self):
        write_file(os.path.join(self.dir, "b.txt"), b"1")
        write_file(os.path.join(self.dir, "a.txt"), b"1")
        names = [os.path.basename(p) for p in
                 list_dir(self.dir, sort=True)]
        self.assertEqual(names, ["a.txt", "b.txt"])

    def test_list_dir_pattern(self):
        write_file(os.path.join(self.dir, "x.py"), b"1")
        write_file(os.path.join(self.dir, "x.txt"), b"1")
        py = [os.path.basename(p) for p in
              list_dir(self.dir, pattern="*.py")]
        self.assertEqual(py, ["x.py"])

    def test_list_dir_recursive(self):
        leaf = _make_tree(self.dir, depth=3)
        self.assertTrue(os.path.exists(leaf))
        all_files = [p for p in list_dir(self.dir, recursive=True)
                     if os.path.isfile(p)]
        self.assertEqual(len(all_files), 6)

    def test_make_dirs(self):
        p = os.path.join(self.dir, "a", "b", "c")
        make_dirs(p)
        self.assertTrue(os.path.isdir(p))

    def test_remove_tree(self):
        _make_tree(self.dir, depth=4)
        remove_path(self.dir)
        self.assertFalse(os.path.exists(self.dir))

    def test_copy_tree(self):
        dst = self.dir + "_copy"
        remove_path(dst)
        try:
            _make_tree(self.dir, depth=3)
            copy_path(self.dir, dst)
            self.assertTrue(os.path.isdir(dst))
            src_files = {os.path.relpath(p, self.dir) for p in
                         list_dir(self.dir, recursive=True)
                         if os.path.isfile(p)}
            dst_files = {os.path.relpath(p, dst) for p in
                         list_dir(dst, recursive=True)
                         if os.path.isfile(p)}
            self.assertEqual(src_files, dst_files)
        finally:
            remove_path(dst)

    def test_move_path(self):
        src = os.path.join(self.dir, "m.txt")
        dst = os.path.join(self.dir, "n.txt")
        write_file(src, b"mv")
        move_path(src, dst)
        self.assertTrue(os.path.exists(dst))
        self.assertFalse(os.path.exists(src))

    def test_find_files(self):
        _make_tree(self.dir, depth=3)
        found = [os.path.basename(p) for p in
                 find_files(self.dir, "*.txt")]
        self.assertEqual(len(found), 6)


class TestDeepNoRecursion(unittest.TestCase):
    """Deep nesting must work without recursion (explicit stacks)."""

    def test_deep_tree_remove(self):
        root = tempfile.mkdtemp(prefix="sysdeep_")
        # POSIX keeps going past the default recursion limit (Windows
        # paths stay short because of MAX_PATH)
        depth = 1200 if os.name != "nt" else 40
        current = root
        for i in range(depth):
            current = os.path.join(current, "a")
            os.makedirs(current, exist_ok=True)
            write_file(os.path.join(current, "f.txt"), b"x")
        remove_path(root)
        self.assertFalse(os.path.exists(root))


class TestFileManager(unittest.TestCase):
    """Delegated container: default impls and injection."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sys_fm_")
        self.addCleanup(remove_path, self.dir)

    def test_default_ops(self):
        fm = FileManager()
        p = os.path.join(self.dir, "fm.txt")
        remove_path(p)
        try:
            fm.write(p, b"data")
            self.assertEqual(fm.read(p), b"data")
            self.assertEqual(fm.info(p)["size"], 4)
            self.assertEqual(len(fm.hash(p)), 64)
            fm.remove(p)
            self.assertFalse(os.path.exists(p))
        finally:
            remove_path(p)

    def test_injected_slot(self):
        calls = []
        fm = FileManager(read_func=lambda self, path, encoding=None:
                         calls.append(path) or "injected")
        self.assertEqual(fm.read("/x"), "injected")
        self.assertEqual(calls, ["/x"])

    def test_falsy_injected_slot_kept(self):
        class FalsyRead:
            def __bool__(self):
                return False
            def __call__(self, manager, path, encoding=None):
                return "kept"

        fm = FileManager(read_func=FalsyRead())
        self.assertEqual(fm.read("/x"), "kept")


class TestFileMatch(unittest.TestCase):
    """file_match: 0-based first matching line index, -1 when absent."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sys_")
        self.addCleanup(remove_path, self.dir)
        self.path = os.path.join(self.dir, "m.txt")
        write_file(self.path, b"alpha\nbeta\ngamma\nalpha\n")

    def test_match_index(self):
        self.assertEqual(file_match(self.path, b"beta"), 1)

    def test_match_first_occurrence(self):
        self.assertEqual(file_match(self.path, b"alpha"), 0)

    def test_match_start_offset(self):
        self.assertEqual(file_match(self.path, b"alpha", start=1), 3)

    def test_no_match(self):
        self.assertEqual(file_match(self.path, b"zzz"), -1)

    def test_callable_pattern(self):
        self.assertEqual(file_match(self.path, lambda line: b"ga" in line), 2)


class TestFileDescriptor(unittest.TestCase):
    """File ops accept raw file descriptors (original fd untouched)."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sys_")
        self.addCleanup(remove_path, self.dir)
        self.path = os.path.join(self.dir, "fd.txt")

    def test_read_write_fd(self):
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            n = write_file(fd=fd, data=b"hello fd")
            self.assertEqual(n, 8)
            os.lseek(fd, 0, 0)
            self.assertEqual(read_file(fd=fd), b"hello fd")
        finally:
            os.close(fd)

    def test_info_hash_match_fd(self):
        write_file(self.path, b"a\nb\nc\n")
        fd = os.open(self.path, os.O_RDONLY)
        try:
            info = file_info(fd=fd)
            self.assertEqual(info["size"], 6)
            self.assertTrue(info["is_file"])
            self.assertEqual(file_hash(fd=fd), file_hash(self.path))
            os.lseek(fd, 0, 0)
            self.assertEqual(file_match(fd=fd, pattern=b"b"), 1)
        finally:
            os.close(fd)

    @unittest.skipIf(os.name == "nt", "os.listdir(fd) unsupported on Windows")
    def test_list_dir_fd(self):
        write_file(self.path, b"1")
        write_file(os.path.join(self.dir, "x.txt"), b"1")
        fd = os.open(self.dir, os.O_RDONLY)
        try:
            names = list_dir(fd=fd, sort=True)
            self.assertEqual(names, ["fd.txt", "x.txt"])
        finally:
            os.close(fd)


if __name__ == "__main__":
    unittest.main()
