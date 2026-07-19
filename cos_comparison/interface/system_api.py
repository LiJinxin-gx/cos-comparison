# It gives some APIs to interact with the system.

# -------- import --------
import os
import sys
import threading
import subprocess

# ------- system -------
command = os.system
getpid = os.getpid
getppid = os.getppid
home_executable = sys.executable


# ------ process interaction ------
class BaseProcess(subprocess.Popen):
    def __init__(self, args, text=False, **kwargs):
        kwargs.setdefault("stdin", subprocess.PIPE)
        kwargs.setdefault("stdout", subprocess.PIPE)
        kwargs.setdefault("stderr", subprocess.PIPE)
        kwargs.setdefault("text", text)
        super().__init__(args, text=text, **kwargs)


class Process(BaseProcess):
    __slots__ = (
        "stdin_line", "stdout_line", "stderr_line",
        "stdin_lock", "stdout_lock", "stderr_lock",
        "_stop_event", "_stdout_thread", "_stderr_thread"
    )

    def __init__(self, executable, arg_list, **kwarg):
        if isinstance(arg_list, str):
            args = [executable, arg_list]
        elif isinstance(arg_list, (tuple, list)):
            args = [executable, *arg_list]
        else:
            raise TypeError("arg_list is unsupported type.")

        # Buffers for accumulated I/O data (bytearray)
        self.stdin_line = bytearray()
        self.stdout_line = bytearray()
        self.stderr_line = bytearray()

        # Three separate RLock objects to protect each buffer
        self.stdin_lock = threading.RLock()
        self.stdout_lock = threading.RLock()
        self.stderr_lock = threading.RLock()

        # Event to signal reader threads to stop
        self._stop_event = threading.Event()
        self._stdout_thread = None
        self._stderr_thread = None

        kwarg["text"] = False          # force bytes mode
        super().__init__(args, **kwarg)

        # Start background reader threads
        threading.Thread(target=self._run, daemon=True).start()

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

    def _run(self):
        """Launch background threads that continuously read stdout/stderr until stopped."""
        # Create reader threads
        self._stdout_thread = threading.Thread(target=self._read_stdout_loop, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr_loop, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

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

    def stop(self, timeout=0.5):
        """Signal the reader threads to stop and wait for them."""
        self._stop_event.set()
        # Optionally close stdin to trigger child exit (if it reads from stdin)
        if self.stdin and not self.stdin.closed:
            self.stdin.close()
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
