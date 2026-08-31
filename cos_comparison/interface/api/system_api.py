# It gives some APIs to interact with the system.

# -------- import --------
import os
import subprocess
import sys
import threading

__all__ = (
    "BaseProcess",
    "FileManager",
    "Process",
    "command",
    "copy_path",
    "file_hash",
    "file_info",
    "file_match",
    "find_files",
    "getpid",
    "getppid",
    "home_executable",
    "kill",
    "list_dir",
    "make_dirs",
    "move_path",
    "read_file",
    "remove_path",
    "write_file",
)

# ------- system -------
def command(commands, input=None, timeout=None):
    obj = subprocess.Popen(commands,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        out,err = obj.communicate(input=input, timeout=timeout)
        return out,err,obj.returncode
    except subprocess.TimeoutExpired:
        # Kill and reap the timed-out child so callers do not leak a live
        # process; the OSError guard covers the exit-at-timeout race.
        try:
            obj.kill()
        except OSError:
            pass
        obj.communicate()
        raise
    finally:
        if obj.stdin is not None and not obj.stdin.closed:
            obj.stdin.close()

getpid = os.getpid
getppid = os.getppid
home_executable = sys.executable
kill = os.kill

# ------- file management -------
def list_dir(path=None, fd=None, pattern=None, recursive=False, sort=False):
    """List paths under a directory; optional fnmatch filter; explicit stack.

    fd lists names only (no recursion; POSIX only for dirs).
    """
    import fnmatch
    out = []
    if fd is not None:
        names = os.listdir(fd)
        for name in names:
            if pattern is None or fnmatch.fnmatch(name, pattern):
                out.append(name)
        if sort:
            out.sort()
        return out
    if not recursive:
        try:
            names = os.listdir(path)
        except OSError:
            return out
        for name in names:
            if pattern is None or fnmatch.fnmatch(name, pattern):
                out.append(os.path.join(path, name))
    else:
        stack = [path]
        while stack:
            current = stack.pop()
            try:
                names = os.listdir(current)
            except OSError:
                continue
            for name in names:
                child = os.path.join(current, name)
                if os.path.isdir(child) and not os.path.islink(child):
                    stack.append(child)
                if pattern is None or fnmatch.fnmatch(name, pattern):
                    out.append(child)
    if sort:
        out.sort()
    return out


def make_dirs(path):
    """Create directory chain (mkdir -p); returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def remove_path(path):
    """Remove a file or directory tree; explicit stack, no recursion."""
    if not os.path.lexists(path):
        return
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
        return
    stack = [path]
    while stack:
        current = stack[-1]
        try:
            names = os.listdir(current)
        except OSError:
            os.rmdir(current)
            stack.pop()
            continue
        if not names:
            os.rmdir(current)
            stack.pop()
            continue
        child = os.path.join(current, names.pop())
        if os.path.isdir(child) and not os.path.islink(child):
            stack.append(child)
        else:
            os.remove(child)


def copy_path(src, dst):
    """Copy a file or directory tree; explicit stack, no recursion."""
    import shutil
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        return
    stack = [(src, dst)]
    while stack:
        s, d = stack.pop()
        os.makedirs(d, exist_ok=True)
        try:
            names = os.listdir(s)
        except OSError:
            continue
        for name in names:
            child_src = os.path.join(s, name)
            child_dst = os.path.join(d, name)
            if os.path.isdir(child_src) and not os.path.islink(child_src):
                stack.append((child_src, child_dst))
            else:
                shutil.copy2(child_src, child_dst)


def move_path(src, dst):
    """Move a file or directory (shutil.move, non-recursive)."""
    import shutil
    return shutil.move(src, dst)


def file_info(path=None, fd=None):
    """Return basic file/directory metadata (path or fd keyword)."""
    import stat as _stat
    if fd is not None:
        st = os.fstat(fd)
        return {"path": fd, "size": st.st_size, "mtime": st.st_mtime,
                "is_dir": _stat.S_ISDIR(st.st_mode),
                "is_file": _stat.S_ISREG(st.st_mode)}
    st = os.stat(path)
    return {"path": path, "size": st.st_size, "mtime": st.st_mtime,
            "is_dir": os.path.isdir(path), "is_file": os.path.isfile(path)}


def file_hash(path=None, fd=None, algo="sha256", chunk=65536):
    """Compute a file hash by chunked reading (path or fd, no recursion)."""
    import hashlib
    h = hashlib.new(algo)
    if fd is not None:
        while True:
            block = os.read(fd, chunk)
            if not block:
                break
            h.update(block)
        return h.hexdigest()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_file(path=None, fd=None, encoding=None):
    """Read a whole file (path or fd keyword); text mode when encoding given."""
    if fd is not None:
        if encoding is not None:
            with os.fdopen(os.dup(fd), "r", encoding=encoding) as fh:
                return fh.read()
        chunks = []
        while True:
            block = os.read(fd, 65536)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    mode = "r" if encoding is not None else "rb"
    with open(path, mode, encoding=encoding) as fh:
        return fh.read()


def write_file(path=None, data=None, fd=None, encoding=None):
    """Write a whole file (path or fd keyword); text mode when encoding given."""
    if fd is not None:
        if encoding is not None:
            with os.fdopen(os.dup(fd), "w", encoding=encoding) as fh:
                fh.write(data)
            return len(data)
        if isinstance(data, str):
            data = data.encode()
        total = 0
        view = memoryview(data)
        while total < len(view):
            total += os.write(fd, view[total:])
        return total
    mode = "w" if encoding is not None else "wb"
    with open(path, mode, encoding=encoding) as fh:
        fh.write(data)
    return len(data)


def file_match(path=None, pattern=None, fd=None, start=0, encoding=None):
    """Return the 0-based index of the first matching line, else -1.

    pattern: substring (str/bytes) or callable(line); path or fd keyword.
    """
    if fd is not None:
        with os.fdopen(os.dup(fd), "r" if encoding else "rb",
                       encoding=encoding) as fh:
            return _first_match(fh, pattern, start)
    with open(path, "r" if encoding else "rb", encoding=encoding) as fh:
        return _first_match(fh, pattern, start)


def _first_match(fh, pattern, start):
    for idx, line in enumerate(fh):
        if idx < start:
            continue
        if pattern(line) if callable(pattern) else pattern in line:
            return idx
    return -1


def find_files(root, pattern, recursive=True):
    """Find files matching a pattern under root (delegates to list_dir)."""
    return [p for p in list_dir(root, pattern=pattern, recursive=recursive)
            if os.path.isfile(p)]


class FileManager:
    """Delegated file-operation container; every op is an injectable slot.

    Default slots wrap module-level functions with FuncWrap, so injected
    funcs receive (self, *args).
    """
    __slots__ = (
        "copy_func",
        "hash_func",
        "info_func",
        "list_func",
        "move_func",
        "read_func",
        "remove_func",
        "write_func",
    )
    def __init__(self, list_func=None, copy_func=None, move_func=None,
                 remove_func=None, info_func=None, hash_func=None,
                 read_func=None, write_func=None):
        from ..tools.func_tool import FuncWrap
        self.list_func = list_func if list_func else FuncWrap(list_dir)
        self.copy_func = copy_func if copy_func else FuncWrap(copy_path)
        self.move_func = move_func if move_func else FuncWrap(move_path)
        self.remove_func = remove_func if remove_func else FuncWrap(remove_path)
        self.info_func = info_func if info_func else FuncWrap(file_info)
        self.hash_func = hash_func if hash_func else FuncWrap(file_hash)
        self.read_func = read_func if read_func else FuncWrap(read_file)
        self.write_func = write_func if write_func else FuncWrap(write_file)
    def list(self, path, pattern=None, recursive=False, sort=False):
        return self.list_func(self, path, pattern=pattern,
                              recursive=recursive, sort=sort)
    def copy(self, src, dst):
        return self.copy_func(self, src, dst)
    def move(self, src, dst):
        return self.move_func(self, src, dst)
    def remove(self, path):
        return self.remove_func(self, path)
    def info(self, path):
        return self.info_func(self, path)
    def hash(self, path, algo="sha256"):
        return self.hash_func(self, path, algo=algo)
    def read(self, path, encoding=None):
        return self.read_func(self, path, encoding=encoding)
    def write(self, path, data, encoding=None):
        return self.write_func(self, path, data, encoding=encoding)

# ------ process interaction ------
class BaseProcess(subprocess.Popen):
    def __init__(self, args, text=False, **kwargs):
        kwargs.setdefault("stdin", subprocess.PIPE)
        kwargs.setdefault("stdout", subprocess.PIPE)
        kwargs.setdefault("stderr", subprocess.PIPE)
        kwargs.setdefault("text", text)
        super().__init__(args, **kwargs)


class Process(BaseProcess):
    __slots__ = (
        "_stderr_thread",
        "_stdout_thread",
        "_stop_event",
        "stderr_line",
        "stderr_lock",
        "stdin_line",
        "stdin_lock",
        "stdout_line",
        "stdout_lock"
    )

    def __init__(self, executable, arg_list, **kwarg):
        if isinstance(arg_list, str):
            args = [executable, arg_list]
        elif isinstance(arg_list, (tuple, list)):
            args = [executable, *arg_list]
        else:
            raise TypeError("arg_list is unsupported type.")

        # Accumulated I/O data (bytearray) and one RLock per buffer
        self.stdin_line = bytearray()
        self.stdout_line = bytearray()
        self.stderr_line = bytearray()

        self.stdin_lock = threading.RLock()
        self.stdout_lock = threading.RLock()
        self.stderr_lock = threading.RLock()

        self._stop_event = threading.Event()
        self._stdout_thread = None
        self._stderr_thread = None

        kwarg["text"] = False          # force bytes mode
        super().__init__(args, **kwarg)

        # Start reader threads synchronously so stop() never observes
        # unassigned handles (no data race).
        self._stdout_thread = threading.Thread(target=self._read_stdout_loop, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr_loop, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def execute(self, command):
        """Write a command to the subprocess's stdin."""
        if isinstance(command, str):
            command = command.encode()
        with self.stdin_lock:
            self.stdin.write(command)
            self.stdin.flush()
            self.stdin_line.extend(command)   # FIX: use extend instead of append
        return len(command)

    def _readline_stdout(self):
        """Read one line from stdout (blocking) and store it."""
        line = self.stdout.readline()
        if line:
            with self.stdout_lock:
                self.stdout_line.extend(line)  # FIX: extend the whole line

    def _readline_stderr(self):
        """Read one line from stderr and store it."""
        line = self.stderr.readline()
        if line:
            with self.stderr_lock:
                self.stderr_line.extend(line)  # FIX: extend

    def _read_stdout_loop(self):
        """Continuously read lines from stdout until pipe closes or stop event is set."""
        while not self._stop_event.is_set():
            line = self.stdout.readline()
            if not line:               # EOF → child process closed pipe
                break
            with self.stdout_lock:
                self.stdout_line.extend(line)

    def _read_stderr_loop(self):
        """Continuously read lines from stderr until pipe closes or stop event is set."""
        while not self._stop_event.is_set():
            line = self.stderr.readline()
            if not line:
                break
            with self.stderr_lock:
                self.stderr_line.extend(line)

    def stop(self, timeout=0.5, terminate=True):
        """
        Signal reader threads to stop, close stdin, and (by default)
        terminate and reap the child. terminate=False keeps it alive.
        """
        self._stop_event.set()
        # Optionally close stdin to trigger child exit (if it reads from stdin)
        if self.stdin and not self.stdin.closed:
            self.stdin.close()
        if terminate and self.poll() is None:
            try:
                self.terminate()
            except OSError:
                pass
            try:
                self.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        if self._stdout_thread and self._stdout_thread.is_alive():
            self._stdout_thread.join(timeout=timeout)
        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=timeout)

    def get_stdout(self, clear=False):
        """Return accumulated stdout data (bytes), optionally clear the buffer."""
        with self.stdout_lock:
            data = bytes(self.stdout_line)
            if clear:
                self.stdout_line.clear()
        return data

    def get_stderr(self, clear=False):
        with self.stderr_lock:
            data = bytes(self.stderr_line)
            if clear:
                self.stderr_line.clear()
        return data

    def get_stdin(self, clear=False):
        with self.stdin_lock:
            data = bytes(self.stdin_line)
            if clear:
                self.stdin_line.clear()
        return data
