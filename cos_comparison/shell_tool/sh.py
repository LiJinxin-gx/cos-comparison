"""sh command: execute external commands (standalone plugin script).

Runs the command through the system shell by default; ``executable``
selects a specific shell / executor (None = the system default).
Implemented with subprocess.Popen.

With no arguments it enters an interactive system command line (like
cmd): the prompt shows ``user@host:working-directory >`` (e.g.
``alice@DESKTOP:C:\\py >``) and each line is executed through the
terminal until exit/quit/q/EOF.
With arguments the whole line is one command executed directly.

Run via: python -m <package> sh [<command words...>]
"""
import getpass
import os
import socket
import subprocess
import sys

__all__ = ("run",)


def _prompt():
    """Live prompt: user and host are resolved on every call."""
    return (f"{getpass.getuser()}@{socket.gethostname()}:"
            f"{os.getcwd()} > ")


def run(*, command=None, executable=None, interactive=False):
    """Execute an external command via subprocess.Popen (system shell by
    default; executable selects a specific shell / executor, None = the
    system default).  ``command`` is passed by keyword.

    interactive=True inherits the current terminal (stdin/stdout/stderr
    pass through - entry into REPL-style commands); otherwise output is
    captured as raw bytes.  Returns (stdout, stderr, returncode);
    interactive returns (None, None, returncode)."""
    if command is None:
        raise ValueError("command is required")
    if interactive:
        proc = subprocess.Popen(command, shell=True, executable=executable)
        return None, None, proc.wait()
    proc = subprocess.Popen(command, shell=True, executable=executable,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    return out, err, proc.returncode


if __name__ == "__main__":
    args = list(sys.argv[2:])   # strip the command name 'sh'
    if not args:
        # interactive system command line (like cmd): user@host:cwd >
        while True:
            try:
                line = input(_prompt())
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = line.strip()
            if not line:
                continue
            if line in ("exit", "quit", "q"):
                break
            run(command=line, interactive=True)
    else:
        _out, _err, rc = run(command=" ".join(args), interactive=True)
        sys.exit(rc)
