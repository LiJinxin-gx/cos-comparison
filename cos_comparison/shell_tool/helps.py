"""helps command: view a module/function's documentation (plugin).

Uses standard pydoc to render docs of any module / callable reachable in
the shell namespace.

Run via: python -m <package> helps <module|func> [name]
"""
import pydoc
import sys

__all__ = ()


def run(args):
    if not args:
        return "usage: helps <module|func>"
    name = args[0]
    try:
        return pydoc.render_doc(name, renderer=pydoc.plaintext)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        return "error: " + type(exc).__name__ + ": " + str(exc)


if __name__ == "__main__":
    print(run(sys.argv[2:]))   # strip the command name 'helps'
    sys.exit(0)
