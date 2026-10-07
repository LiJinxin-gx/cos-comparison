"""
Command-line entry: directory-searched plugins executed via runpy.

No registration: a root command names a command script under shell_tool/
(a .py file with a __main__ entry guard; the file name is the command
name); plugins run via runpy with a shared mapping injected as __ns__.
KeyboardInterrupt breaks the command; other errors are intercepted.

Interactive call page (python -m cos_comparison): >>> executes commands,
... shows nested execution (e.g. the shell command line); help / exit.
"""
import os
import runpy
import sys

from .shell_tool import _is_command

_PLUGIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "shell_tool")
_NAMESPACE = {}

_PROMPT = ">>> "


def plugin_path(name):
    """Command-script path for ``name``; None when unknown.  The name must
    be a plain file name (no path components), so a command can never
    escape the plugin directory; helper modules without a ``__main__``
    entry guard are not commands."""
    if not name or os.path.basename(name) != name or name.startswith("."):
        return None
    path = os.path.join(_PLUGIN_DIR, name + ".py")
    if os.path.isfile(path) and _is_command(path):
        return path
    return None


def list_commands():
    return sorted(f[:-3] for f in os.listdir(_PLUGIN_DIR)
                  if f.endswith(".py") and not f.startswith("_")
                  and _is_command(os.path.join(_PLUGIN_DIR, f)))


def usage(stream=None):
    stream = sys.stdout if stream is None else stream
    print("usage: python -m cos_comparison [command] [args...]", file=stream)
    print("commands: " + ", ".join(list_commands()), file=stream)
    print("options: -h, --help, help", file=stream)


def help_text(stream=None):
    stream = sys.stdout if stream is None else stream
    usage(stream)
    print(file=stream)
    print("interactive call page (python -m cos_comparison):", file=stream)
    print("  >>> command [args...]   execute a command", file=stream)
    print("  ... input ...           nested execution prompt (the shell", file=stream)
    print("                          command falls into its own command", file=stream)
    print("                          line operation, shown with ...)", file=stream)
    print("  help                    show this help", file=stream)
    print("  exit, quit              leave the call page", file=stream)


def run_command(name, args=None):
    """Run a plugin directly via runpy; return the exit code.  The plugin
    runs with run_name="__main__" and a shared __ns__ mapping; argv is the
    CLI form (argv[1] = command name, plugins read argv[2:]).  The page is
    never killed: KeyboardInterrupt breaks the command, SystemExit is
    translated to its exit code.  Other exceptions are intercepted."""
    path = plugin_path(name)
    if path is None:
        print("unknown command: " + name)
        usage()
        return 1
    # Import the plugin package first so fallback imports derive the
    # package root from sys.modules (no hard-coded package name needed).
    try:
        from . import shell_tool as _plugin_pkg  # noqa: F401
    except Exception:  # noqa: BLE001, S110 - best effort, non-fatal
        pass
    old_argv = sys.argv
    try:
        # always expose the CLI argv shape, bare commands included
        sys.argv = [path, name] + (list(args) if args else [])
        try:
            runpy.run_path(path, init_globals={"__ns__": _NAMESPACE},
                           run_name="__main__")
        except KeyboardInterrupt:
            print("interrupted")
            return 130
        except SystemExit as se:
            # mirror Python exit semantics without killing the page:
            # None -> 0; int -> that code; message -> stderr + 1
            code = se.code
            if code is None:
                return 0
            if isinstance(code, int):
                return code
            print(code, file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001 - intentional interception
            print("error: " + type(exc).__name__ + ": " + str(exc))
            return 1
    finally:
        sys.argv = old_argv
    return 0


def _interactive():
    """Interactive call page: >>> executes commands, ... shows nesting."""
    print("cos_comparison call page - enter 'help' for help, 'exit' to quit")
    while True:
        try:
            line = input(_PROMPT)
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        if line in ("exit", "quit", "q"):
            return 0
        if line in ("help", "h", "-h", "--help"):
            help_text()
            continue
        parts = line.split(None, 1)
        name = parts[0]
        if len(parts) > 1:
            from .shell_tool.shell import _split_words
            args = _split_words(parts[1])
        else:
            args = None
        run_command(name, args)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return _interactive()
    if argv[0] in ("-h", "--help", "help"):
        help_text()
        return 0
    return run_command(argv[0], argv[1:])


if __name__ == "__main__":
    sys.exit(main())
