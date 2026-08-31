"""batch command: batch execute commands (standalone plugin script).

Independent of app modules; uses the shell namespace and the unified
instruction file protocol (same format as the app reader and the shell
run command):

    func arg1 arg2 -kw value             execute a command
    func arg1 arg2 -> pos                store the result at batch data[pos]
    data<N>                              variable: batch data[N] value
    IF <cond-call> ... ELSE ... END      conditional jump block
    WHILE <cond-call> ... END            loop jump block
    # comment / blank line               skipped

The batch keeps its own data region (plain dict - no shared stack).
Interrupt handling: a first KeyboardInterrupt requests a graceful stop
(after the current command); a second consecutive one forces termination.
"""
import sys

__all__ = ()

try:
    from .shell import (
        execute_instruction_items,
        note_interrupt,
        parse_instruction_file,
        resolve_callable,
    )
except ImportError:
    # runpy direct execution: derive the package root dynamically (never
    # hard-code the package name).
    import importlib as _il
    _pkg_root = None
    if __package__ and "." in __package__:
        _pkg_root = __package__.rpartition(".")[0]
    else:
        _pkg_root = next(
            (m.rsplit(".", 1)[0] for m in list(sys.modules)
             if m.endswith(".shell_tool") and "." in m), None)
    if _pkg_root is None:
        raise
    _shell = _il.import_module(_pkg_root + ".shell_tool.shell")
    execute_instruction_items = _shell.execute_instruction_items
    note_interrupt = _shell.note_interrupt
    parse_instruction_file = _shell.parse_instruction_file
    resolve_callable = _shell.resolve_callable


def run_batch(lines, data=None, interrupt_window=1.0):
    """Execute batch commands; returns (data, stats).  The data dict is
    the batch's own region (no shared stack).  A first KeyboardInterrupt
    does not terminate - a running subcommand may capture it; a second
    CONSECUTIVE one (within interrupt_window) forces termination."""
    data = {} if data is None else data
    text_lines = []
    for line in lines:
        try:
            text_lines.append(line)
        except KeyboardInterrupt:
            note_interrupt(interrupt_window)   # idle: two interrupts exit
    text = "\n".join(text_lines)
    items = parse_instruction_file(text, resolve_callable)
    _, stats = execute_instruction_items(
        items, data, print, interrupt_window=interrupt_window)
    return data, stats


if __name__ == "__main__":
    args = sys.argv[2:]   # strip the command name 'batch'
    if args:
        # the instruction file's own directory joins sys.path so the
        # file can import helper modules next to it (e.g. import_module
        # input for setting the training data location)
        import os as _os
        sys.path.insert(0, _os.path.dirname(_os.path.abspath(args[0]))
                        or ".")
        with open(args[0], encoding="utf-8-sig") as fh:
            lines = fh.read().splitlines()
    else:
        lines = sys.stdin.read().splitlines()
    data, stats = run_batch(lines)
    sys.exit(0)
