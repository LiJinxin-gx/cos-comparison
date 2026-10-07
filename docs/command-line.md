# Command Line & Instruction Files

Unified function-style instruction control shared by the **shell command line**, the **batch executor** and the **app (Docker) reader**.

## Running

```bash
python -m cos_comparison              # interactive call page (>>>)
python -m cos_comparison shell        # shell command line
python -m cos_comparison batch file.module_bat
python -m cos_comparison sh           # interactive system command line
python -m cos_comparison sh echo hi   # run one external command
```

## Command Plugin Conventions

- A command is a `shell_tool/*.py` file carrying a top-level
  `if __name__ == "__main__"` guard; the file name is the command name
  (no registration).  Helper modules without the guard (e.g. `value.py`)
  are libraries - they are never listed or executed as commands.
- Command names are plain file names: path components (`../x`, `a/b`)
  are rejected before any file is touched.
- Plugins receive the CLI argv shape (`argv[1]` = command name, plugins
  read `argv[2:]`) and share the injected `__ns__` namespace across runs
  in one process, so imports / variables persist between commands and the
  nested command line.
- `SystemExit` never kills the call page: `None` -> 0, an integer -> that
  code, a message -> printed to stderr and code 1 (Python semantics).
- A command that ends with `KeyboardInterrupt` returns 130; other errors
  are intercepted and reported as `error: <Type>: <message>`.

## Instruction Syntax

```
func arg1 arg2 -kw value        function call (one instruction)
func arg1 arg2 -> pos           store the result at data[pos]
data[index]                     data region reference (dataN legacy ok)
%expr                           unpack a sequence into arguments
%%expr                          unpack a mapping into keyword arguments
let name value                  variable assignment (function style)
IF <cond-call> ... ELSE ... END conditional block
WHILE <cond-call> ... END       loop block
# comment / blank line          skipped
```

Literals follow Python conventions: `(a b)` tuple, `[a b]` list, quoted
strings, numbers, `True`/`False`/`None`, nested groups.  Built-in
functions (`len`, `sum`, `int`, `str`, ...) are pre-imported; `&name`
gives a variable index and `*index` dereferences it (pointer style).

## Namespace

- Project modules resolve by dotted names (`core.add_chain`,
  `core.cos_comparison_passive`), with dot-path attribute extraction.
- Management functions: `import_module <mod> [-name alias]`,
  `import_all_module <mod> [-namespace target]` (`from module import *`; default namespace = the current shell function table),
  `list_modules`, `let`, `get`, `delete`.
- Interrupts (command-line style): a first KeyboardInterrupt is only
  recorded (a running sub-program may capture it); a second consecutive
  one (within the window) forces termination.  Loops check this at the
  end of every iteration.

## Executors

| Executor | Behaviour |
|----------|-----------|
| shell | one instruction per line; `run <file>` executes an instruction file |
| batch | executes an instruction file; the file's directory joins `sys.path` (helper modules next to the file are importable) |
| sh | runs external commands through the system shell (`subprocess.Popen`, `shell=True`); with arguments it executes one command, without arguments it enters an interactive system command line (prompt = `user@host:working-directory > ` — user and host resolved live on every prompt); `run(command=..., executable=None, interactive=False)` selects a specific executor or captures output |
| app | Docker reads the same format; `subdocker(docker, flow, ...)` delegates a task to a child Docker (pools are copied, the child runs synchronously, its data_pool is returned); `suspend()` stops the run at the next step boundary and returns a `SuspendSnapshot`, `resume(snapshot)` continues from the recorded cursor on the next `run` |

## Linear Algebra (math_tool.linear_algebra)

`dot`, `norm`, `normalize`, `scale`, `add`, `multiply`, `power`,
`clip`, `flatten`, `tensor_sum`, `tensor_mean`.  Dimension-generic,
duck typed (any sequence-protocol container).  Every function passes
the tensor out through the keyword `output` (never returned directly)
and returns an integer status: `0` success / `1` mismatch / `2` no
output / `3` conversion failure.  Identical behaviour in the pure-Python
fallback and the C extension.
