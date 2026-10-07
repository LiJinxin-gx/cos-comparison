"""helps command: view a module/function's documentation (plugin).

Uses standard pydoc to render docs of any module / callable reachable in
the shell namespace.  Names resolve in this order: the shared shell
namespace (imported modules / functions / variables registered by
``__main__`` as ``__ns__``), then plain pydoc name resolution.  A second
argument selects a member of the first (``helps math sqrt``).

Run via: python -m <package> helps <module|func> [name]
"""
import pydoc
import sys

__all__ = ()


def _namespace_lookup(name):
    """Look up a shell-namespace object (module / callable / variable)."""
    ns = globals().get("__ns__")
    if ns is None:
        return None
    for table in ("modules", "funcs", "vars"):
        value = ns.get(table, {}).get(name)
        if value is not None:
            return value
    return None


def run(args):
    if not args:
        return "usage: helps <module|func>"
    name = args[0]
    if len(args) > 1:
        name = name + "." + args[1]
    target = _namespace_lookup(name)
    try:
        return pydoc.render_doc(target if target is not None else name,
                                renderer=pydoc.plaintext)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        return "error: " + type(exc).__name__ + ": " + str(exc)


if __name__ == "__main__":
    print(run(sys.argv[2:]))   # strip the command name 'helps'
    sys.exit(0)
