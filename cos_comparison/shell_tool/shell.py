"""shell command: imperative function calls (standalone plugin script).

Run via: python -m <package> shell func arg1 -kw value

Syntax (no outer parentheses): func arg1 arg2 -kw value; quoted strings
kept verbatim.  Values use the unified literal grammar (see value.py):
Python-style containers, infix expressions, nested calls and built-in
operator symbols as function names.  The shared mapping injected by
__main__ as __ns__ is used as plugin, otherwise a fresh one is created.
Namespace operations: import_module, let, get, delete.
"""
import importlib
import os
import sys

__all__ = ()

try:
    from .. import core as _core
    from ..interface.api import CallDict
    from ..interface.tools.math_tool import fourier as _fourier
    from ..interface.tools.math_tool import topology as _topology
    from .value import (
        DataRef,
        SYMBOL_FUNCS,
        ValueCode,
        ValueEnv,
        compile_value,
        execute_code,
    )
except ImportError:
    # runpy direct execution: derive the package root dynamically
    # (never hard-code the package name).
    import importlib as _il
    import sys as _sys
    _pkg_root = None
    if __package__ and "." in __package__:
        _pkg_root = __package__.rpartition(".")[0]
    else:
        _pkg_root = next(
            (m.rsplit(".", 1)[0] for m in list(_sys.modules)
             if m.endswith(".shell_tool") and "." in m), None)
    if _pkg_root is None:
        raise
    _core = _il.import_module(_pkg_root + ".core")
    _CallDict_mod = _il.import_module(_pkg_root + ".interface.api")
    CallDict = _CallDict_mod.CallDict
    _fourier = _il.import_module(
        _pkg_root + ".interface.tools.math_tool.fourier")
    _topology = _il.import_module(
        _pkg_root + ".interface.tools.math_tool.topology")
    _value_mod = _il.import_module(_pkg_root + ".shell_tool.value")
    DataRef = _value_mod.DataRef
    SYMBOL_FUNCS = _value_mod.SYMBOL_FUNCS
    ValueCode = _value_mod.ValueCode
    ValueEnv = _value_mod.ValueEnv
    compile_value = _value_mod.compile_value
    execute_code = _value_mod.execute_code


def _pkg_root():
    """The project package root name (cos_comparison directory), never
    hard-coded."""
    if __package__ and "." in __package__:
        return __package__.rpartition(".")[0]
    return next(
        (m.rsplit(".", 1)[0] for m in list(sys.modules)
         if m.endswith(".shell_tool") and "." in m), None)


def _collect_project_module_names():
    """All importable module names under the project package directory
    (e.g. "core", "core.cos_comparison", "interface.api.database_api"),
    collected lazily via os.walk (no import side effects)."""
    root = _pkg_root()
    if root is None:
        return ()
    try:
        pkg = importlib.import_module(root)
        base = os.path.dirname(pkg.__file__)
    except (ImportError, AttributeError):
        return ()
    names = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".ruff_cache")]
        rel = os.path.relpath(dirpath, base)
        prefix = "" if rel == "." else rel.replace(os.sep, ".")
        for f in filenames:
            if f == "__init__.py":
                if prefix:
                    names.append(prefix)
            elif f.endswith(".py"):
                name = f[:-3]
                if name in ("__main__", "conftest"):
                    continue
                names.append(prefix + "." + name if prefix else name)
    return tuple(sorted(set(names)))

_PROJECT_FUNCS = (
    "cos_comparison_passive", "cos_comparison_active", "cos",
    "threshold_map", "threshold_judge", "data_filter", "data_mapping",
    "mean_local", "local_variance", "load_as_default_data",
    "infer_shape", "get_item", "set_item",
)

# Namespace: the shared __ns__ injected by __main__ takes priority,
# otherwise a fresh one is built
_ns = globals().get("__ns__") or {}
_ns.setdefault("vars", {})
_ns.setdefault("funcs", {})
_ns.setdefault("modules", {})
_ns.setdefault("_project_modules", None)
_ns.setdefault("_var_index", {})
_ns.setdefault("_var_seq", 0)
_DISPATCH = CallDict()


def _ensure_project_modules():
    """Lazily collect the project package module names (once)."""
    if _ns["_project_modules"] is None:
        _ns["_project_modules"] = _collect_project_module_names()
    return _ns["_project_modules"]


def _resolve_root(prefix):
    """Resolve a dotted prefix to an object: namespace vars -> registered
    modules -> project modules -> importlib as-is -> project package root
    prefix.  Modules are registered back into the namespace."""
    obj = _ns["vars"].get(prefix)
    if obj is not None:
        return obj
    mod = _ns["modules"].get(prefix)
    if mod is not None:
        return mod
    root = _pkg_root()
    if prefix in _ensure_project_modules() and root is not None:
        try:
            mod = importlib.import_module(root + "." + prefix)
        except ImportError:
            mod = None
        if mod is not None:
            _ns["modules"][prefix] = mod
            return mod
    try:
        mod = importlib.import_module(prefix)
    except ImportError:
        mod = None
    if mod is None and root is not None:
        try:
            mod = importlib.import_module(root + "." + prefix)
        except ImportError:
            mod = None
    if mod is not None:
        _ns["modules"][prefix] = mod
        return mod
    return None


def _extract_dotted(name):
    """Dot-path extraction: walk the longest registered prefix (vars ->
    modules -> project modules -> imports), then getattr the remaining
    parts; callables cached in the function table, other values in vars."""
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        root = _resolve_root(".".join(parts[:i]))
        if root is None:
            continue
        obj = root
        try:
            for part in parts[i:]:
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if callable(obj):
            _ns["funcs"][name] = obj
        else:
            _ns["vars"][name] = obj
        return obj
    return None


def _register_module(module, names, prefix=""):
    for n in names:
        obj = getattr(module, n, None)
        if callable(obj):
            _DISPATCH.add(prefix + n, obj)
            _ns["funcs"][prefix + n] = obj


_register_module(_core, _PROJECT_FUNCS)
_register_module(_fourier, ("dft", "idft", "dft_kernel_real",
                            "dft_kernel_imag", "power_spectrum"))
_register_module(_topology, ("shortest_path_between",
                             "Euler_characteristic_compute_by_cell"))


def _register_builtins():
    """Pre-import the built-in functions (the __builtins__ module: len /
    sum / int / str / list ...) into the function table, so the command
    line can call them directly.  Existing registrations are kept
    (setdefault - the project's own names win)."""
    import builtins as _builtins
    for _name in dir(_builtins):
        if _name.startswith("__"):
            continue
        _obj = getattr(_builtins, _name)
        if callable(_obj):
            _ns["funcs"].setdefault(_name, _obj)


_register_builtins()

# the built-in operator symbols (+ - * / // % ** & | ^ ~ << >> == !=
# < <= > >=) are first-class names in the function table; the word "not"
# stays expression-only (a bare `not x` line parses as the value form)
for _sym_name, _sym_fn in SYMBOL_FUNCS.items():
    if _sym_name != "not":
        _ns["funcs"].setdefault(_sym_name, _sym_fn)


def register_callable(name, func):
    """Inject a callable into the namespace function table."""
    _ns["funcs"][name] = func


def resolve_callable(name):
    fn = _ns["funcs"].get(name)
    if fn is not None:
        return fn
    obj = _extract_dotted(name)
    if obj is not None and callable(obj):
        return obj
    raise KeyError("unknown function: " + str(name))


def _parse_group(tok):
    """Parse a tuple/list literal iteratively (explicit stack, no
    recursion); quoted strings keep their content verbatim (commas/spaces
    inside quotes never separate)."""
    stack = []          # [(kind, items), ...]; kind is '(' or '['
    buf = ""
    quote = None
    for ch in tok:
        if quote is not None:
            buf += ch
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            buf += ch
        elif ch in "([":
            stack.append((ch, []))
            buf = ""
        elif ch in ")]":
            if buf.strip():
                stack[-1][1].append(_atom_leaf(buf))
                buf = ""
            kind, items = stack.pop()
            value = tuple(items) if kind == "(" else list(items)
            if stack:
                stack[-1][1].append(value)
            else:
                return value
        elif ch in ", \t\n":
            if buf.strip():
                stack[-1][1].append(_atom_leaf(buf))
                buf = ""
        else:
            buf += ch
    raise ValueError("unbalanced literal: " + tok)


def _var_deref(index):
    """Dereference a variable index (like a pointer): return the value of
    the variable stored at that index in the vars table."""
    for name, i in _ns["_var_index"].items():
        if i == index:
            return _ns["vars"].get(name)
    raise ValueError("no variable at index " + str(index))


def _atom_leaf(tok):
    """Convert a non-group literal token to a Python value: quoted
    strings, numbers, True/False/None, and the pointer forms (&name /
    *expr).  The group parser calls this for its leaf tokens (so the
    group parser never calls back into the group parser)."""
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
        return tok[1:-1]
    if tok == "True":
        return True
    if tok == "False":
        return False
    if tok == "None":
        return None
    if tok.startswith("&") and len(tok) > 1:
        name = tok[1:]
        try:
            return _ns["_var_index"][name]
        except KeyError:
            raise ValueError("unknown variable: " + name)
    if tok.startswith("*") and len(tok) > 1:
        rest = tok[1:]
        if rest.isdigit():
            return _var_deref(int(rest))
        if rest in _ns["vars"] and isinstance(_ns["vars"][rest], int):
            return _var_deref(_ns["vars"][rest])
        if rest.startswith("&") and len(rest) > 1:
            name = rest[1:]
            try:
                return _var_deref(_ns["_var_index"][name])
            except KeyError:
                raise ValueError("unknown variable: " + name)
        raise ValueError("invalid dereference: " + tok)
    try:
        return int(tok)
    except ValueError:
        try:
            return float(tok)
        except ValueError:
            return tok


def _atom(tok):
    """Convert a literal token to a Python value: quoted strings, numbers,
    True/False/None, tuples/lists, and nesting (iterative)."""
    if tok.startswith(("(", "[")):
        if not ((tok.startswith("(") and tok.endswith(")")) or
                (tok.startswith("[") and tok.endswith("]"))):
            raise ValueError("unbalanced literal: " + tok)
        return _parse_group(tok)
    return _atom_leaf(tok)


def _split_words(text):
    words, cur, quote, depth = [], "", None, 0
    for ch in text:
        if quote is not None:
            cur += ch
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            cur += ch
        elif ch in "([{":
            depth += 1
            cur += ch
        elif ch in ")]}":
            depth -= 1
            cur += ch
        elif ch in " \t\n" and depth == 0:
            if cur:
                words.append(cur)
                cur = ""
        else:
            cur += ch
    if cur:
        words.append(cur)
    return words


def parse_call(text):
    """Parse a shell call: func arg1 arg2 -kw value -> (func, args,
    kwargs).  Literals follow the instruction conventions (data[index],
    % / %% unpack markers)."""
    words = _split_words(text)
    if not words:
        raise ValueError("empty shell call")
    func = words[0]
    args, kwargs = [], {}
    i = 1
    while i < len(words):
        w = words[i]
        if w.startswith("-") and len(w) > 1 and i + 1 < len(words):
            kwargs[w[1:]] = _call_value(words[i + 1])
            i += 2
        else:
            args.append(_call_value(w))
            i += 1
    return func, tuple(args), kwargs


def execute_call(func, args, kwargs, data=None):
    """Call func by name with prepared args (unpack markers expanded
    against the optional data region; KeyboardInterrupt breaks)."""
    try:
        flat_args, flat_kwargs = _prepare_call_args(args, kwargs, data)
        return resolve_callable(func)(*flat_args, **flat_kwargs)
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 - intentional error interception
        return {"error": type(exc).__name__ + ": " + str(exc)}


def ns_import(module_name, name=None):
    """Import a module into the namespace (injected into the function
    table).  Dotted paths like ``core.cos_comparison`` work: the project
    package root prefix is applied when the bare name does not import;
    ``name`` controls the registered name (alias)."""
    module = _resolve_root(module_name)
    if module is None:
        return "cannot import " + module_name
    alias = name if name is not None else module_name
    _ns["modules"][alias] = module
    names = [n for n in dir(module) if not n.startswith("_")]
    for n in names:
        obj = getattr(module, n)
        if callable(obj):
            _ns["funcs"][alias + "." + n] = obj
    return "imported " + alias + " (" + str(len(names)) + " names)"


def ns_list_modules():
    """List the project package modules and the modules imported into the
    namespace so far (injected into the function table)."""
    project = ", ".join(_ensure_project_modules()) or "(none)"
    imported = ", ".join(sorted(_ns["modules"])) or "(none)"
    return ("project modules: " + project + "\n"
            "imported: " + imported)


def import_all_module(module_name, namespace=None):
    """Import a module and attach ALL of its public objects into the
    given namespace (the effect of ``from module import *``).

    module_name is the module to import; namespace (keyword) is the
    target mapping and defaults to the CURRENT shell namespace (the
    function table - not necessarily the global namespace).  Returns a
    summary string."""
    if namespace is None:
        namespace = _ns["funcs"]
    module = _resolve_root(module_name)
    if module is None:
        return "cannot import " + module_name
    names = [n for n in dir(module) if not n.startswith("_")]
    for n in names:
        namespace[n] = getattr(module, n)
    return "imported " + module_name + " (" + str(len(names)) + " names)"


# namespace management functions are injected into the function table
_ns["funcs"]["import_module"] = ns_import
_ns["funcs"]["list_modules"] = ns_list_modules
_ns["funcs"]["import_all_module"] = import_all_module


# ---------------------------------------------------------------------------
# Instruction-style control flow (assembly-like jumps): each shell call is one
# instruction; the condition is evaluated and the jump lands on the true/false
# target via the name mapping (callables pass through; a (fn, args, kwargs)
# tuple is a deferred call, executed on the jump).  Loop interrupt handling: a
# first KeyboardInterrupt is only counted (a running sub-program may capture
# it); a second consecutive one (within the window) forces termination.  The
# pending check runs at the END of every iteration, so long loops stay
# interruptible.
# ---------------------------------------------------------------------------
_interrupt = {"count": 0, "last": 0.0}
INTERRUPT_WINDOW = 1.0


def note_interrupt(window=INTERRUPT_WINDOW):
    """Record a KeyboardInterrupt (first one yields to a running
    sub-program); a second consecutive interrupt raises to force
    termination.  The count resets outside the window."""
    import time as _time
    now = _time.monotonic()
    if now - _interrupt["last"] > window:
        _interrupt["count"] = 0
    _interrupt["count"] += 1
    _interrupt["last"] = now
    if _interrupt["count"] >= 2:
        _interrupt["count"] = 0
        raise KeyboardInterrupt()


def _interrupt_pending():
    """Loop-end pending check: a second consecutive interrupt breaks the
    loop (without raising)."""
    if _interrupt["count"] >= 2:
        _interrupt["count"] = 0
        return True
    return False


def _jump_eval(item):
    """Name-mapping abstraction for jumps: a callable is invoked, a
    (fn, args, kwargs) tuple is a deferred call, a string name is
    resolved through the function table and invoked, else as-is."""
    if callable(item):
        return item()
    if isinstance(item, tuple) and len(item) == 3:
        fn, args, kwargs = item
        return fn(*args, **kwargs)
    if isinstance(item, str):
        return resolve_callable(item)()
    return item


def IF(condition, true_target=None, false_target=None):
    """Conditional jump: evaluate the condition (callable / deferred call /
    name / truthiness), invoke the true/false target (resolved through the
    name mapping); returns the target result (None when no target)."""
    cond = bool(_jump_eval(condition))
    target = true_target if cond else false_target
    if target is None:
        return None
    return _jump_eval(target)


def WHILE(condition, body_target=None):
    """Loop jump: repeat the body target (name mapping) while the condition
    holds; returns the iteration count.  A first KeyboardInterrupt inside
    the body is only counted; a second consecutive one breaks the loop
    (pending check also runs at the end of every iteration)."""
    count = 0
    while bool(_jump_eval(condition)):
        if body_target is not None:
            try:
                _jump_eval(body_target)
            except KeyboardInterrupt:
                note_interrupt()          # first: yield; second: raise
                if _interrupt["count"] == 1:
                    continue
                break
        count += 1
        if _interrupt_pending():
            break
    return count


_ns["funcs"]["IF"] = IF
_ns["funcs"]["WHILE"] = WHILE


# ---------------------------------------------------------------------------
# Unified instruction file protocol (shared by app / shell / batch;
# parser fully iterative, no recursion):
#
#     func arg1 arg2 -kw value            one instruction (function call)
#     func arg1 arg2 -> pos               store the result at data[pos]
#     data[index]                         variable: data[index] value
#     %expr                               unpack a sequence into args
#     %%expr                              unpack a mapping into kwargs
#     IF <cond-call> ... ELSE ... END      conditional jump block
#     WHILE <cond-call> ... END            loop jump block
#     # comment / blank line               skipped
#
# Values use the unified literal grammar of value.py: Python-style
# containers ((1, 2), [1, 2], {1: "a"}, "()" is the empty tuple),
# infix expressions with the built-in operators (data[0]+1, comparisons,
# and/or/not), and nested functional calls ((func arg1 arg2)) - every
# parser stage is iterative.  Variable assignment uses let (function
# style).  The parser emits a uniform item list: plain items are
# (fn, args, kwargs, pos) with fn already resolved; control-flow items
# are (IF/WHILE, (cond, branch_items...), {}, pos) where cond is a
# deferred call tuple.  Each executor (app / shell / batch) wraps items
# into its own environment.
# ---------------------------------------------------------------------------
class UnpackRef:
    """Sequence unpack marker (``%expr``): the value is unpacked into
    positional arguments at call time."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"UnpackRef({self.value!r})"


class UnpackMapRef:
    """Mapping unpack marker (``%%expr``): the value is unpacked into
    keyword arguments at call time."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"UnpackMapRef({self.value!r})"


def _instruction_value(tok):
    """Instruction literal value: compiled with the unified grammar
    (Python-style containers, infix expressions, nested calls,
    ``data[index]`` / ``data<N>`` references, ``%`` / ``%%`` unpack
    markers)."""
    return compile_value(tok)


def _call_value(tok):
    """Shell command-line literal: same instruction conventions."""
    return _instruction_value(tok)


_NAME_ARG_FUNCS = None     # filled at module import end (fn -> arg count,
                           # -1 = all positional args are raw names)


def _name_code(word):
    """Compile a raw-name argument: the plain word stays a literal
    string - names are never resolved as variables."""
    return ValueCode((("push", word),), word)


def _argument_value(fn, index, word, value_func):
    """Value of one argument word: instructions that take raw names
    (let / get / delete / import_module / import_all_module) keep their
    leading name arguments as literal words."""
    if _NAME_ARG_FUNCS is not None:
        count = _NAME_ARG_FUNCS.get(fn)
        if count is not None and (count < 0 or index < count):
            return _name_code(word)
    return value_func(word)


def _parse_instruction_call(body, resolve_func, value_func):
    """Parse one instruction call: func arg1 arg2 -kw value ->
    (fn, args, kwargs) with fn resolved by resolve_func and values by
    value_func."""
    words = _split_words(body)
    if not words:
        raise ValueError("empty instruction")
    fn = resolve_func(words[0])
    args, kwargs = [], {}
    i = 1
    while i < len(words):
        w = words[i]
        if w.startswith("-") and len(w) > 1 and i + 1 < len(words):
            kwargs[w[1:]] = value_func(words[i + 1])
            i += 2
        else:
            args.append(_argument_value(fn, i - 1, w, value_func))
            i += 1
    return fn, args, kwargs


def parse_instruction_file(text, resolve_func, value_func=None):
    """Parse an instruction file into a uniform item list (iterative,
    explicit block stack, no recursion).  resolve_func(name) resolves
    instruction names; value_func(tok) converts literals (default: shell
    conventions with ``data<N>`` -> DataRef)."""
    value_func = value_func if value_func is not None else _instruction_value
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    items = []
    stack = [("root", items, None, 0)]
    index = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("IF "):
            cond = _parse_cond_call(line[3:], resolve_func, value_func)
            stack.append(("IF", [], (cond, None, index), index))
            index = 0
        elif line.startswith("WHILE "):
            cond = _parse_cond_call(line[5:], resolve_func, value_func)
            stack.append(("WHILE", [], (cond, index), index))
            index = 0
        elif line == "ELSE":
            kind, collected, info, parent = stack[-1]
            if kind != "IF":
                raise ValueError("unexpected ELSE")
            stack[-1] = ("ELSE", [], (collected, info[0], None, info[2]),
                         parent)
        elif line == "END":
            if len(stack) == 1:
                raise ValueError("unexpected END")
            kind, collected, info, parent = stack.pop()
            if kind == "ELSE":
                true_items, cond, _unused, pos = info
                item = (IF, (cond, true_items, collected), {}, pos)
            elif kind == "IF":
                cond, _unused, pos = info
                item = (IF, (cond, collected, None), {}, pos)
            else:
                cond, pos = info
                item = (WHILE, (cond, collected), {}, pos)
            stack[-1][1].append(item)
            index = parent + 1
        else:
            result_pos = index
            body = line
            if "->" in line:
                lhs, _, rhs = line.partition("->")
                result_pos = _atom(rhs.strip())
                body = lhs.strip()
            try:
                fn, args, kwargs = _parse_instruction_call(
                    body, resolve_func, value_func)
            except KeyError:
                # a bare value line (infix expression / nested call /
                # single word): evaluate it and store the result
                vcode = _try_value_line(body, value_func)
                if vcode is None:
                    raise
                fn = _self_value
                args = (vcode,)
                kwargs = {}
            # import_all_module (default namespace) registers at parse
            # time so following short-name lines resolve against it
            # (its arguments are executed against the shell environment).
            if fn is import_all_module and "namespace" not in kwargs:
                names, _kw = _prepare_call_args(args, {}, None)
                import_all_module(*names, **_kw)
            stack[-1][1].append((fn, args, kwargs, result_pos))
            index += 1
        i += 1
    if len(stack) != 1:
        raise ValueError("unterminated control block")
    return stack[0][1]


def _resolve_instruction(value, data):
    """Resolve a DataRef variable reference against a data region (the
    classic object path - programmatic operate pools)."""
    if isinstance(value, DataRef):
        if data is None:
            raise ValueError("data reference outside a data region")
        return value.resolve(data)
    return value


def _shell_env(data):
    """The shell execution environment for compiled values: the data
    region, the namespace variables, the function table resolver and the
    variable-index table."""
    return ValueEnv(data=data, vars=_ns["vars"],
                    resolve=resolve_callable,
                    var_index=_ns["_var_index"])


def _arg_value(value, env, data):
    """Resolve one argument: compiled ValueCode executes against the
    environment; classic objects (DataRef / plain values) resolve through
    the data region."""
    if isinstance(value, ValueCode):
        return value.execute(env)
    return _resolve_instruction(value, data)


def _prepare_call_args(args, kwargs, data):
    """Prepare a call: execute compiled values and unpack the markers
    (``%`` sequence unpack into positional arguments, ``%%`` mapping
    unpack into keyword arguments)."""
    env = _shell_env(data)
    flat = []
    for a in args:
        if isinstance(a, ValueCode):
            value = a.execute(env)
            if a.kind == "unpack":
                flat.extend(value)
            elif a.kind == "unpack_map":
                kwargs.update(value)
            else:
                flat.append(value)
        elif isinstance(a, UnpackRef):
            value = _resolve_instruction(a.value, data)
            flat.extend(value)
        elif isinstance(a, UnpackMapRef):
            kwargs.update(_resolve_instruction(a.value, data))
        else:
            flat.append(_resolve_instruction(a, data))
    kwargs = {k: _arg_value(v, env, data) for k, v in kwargs.items()}
    return flat, kwargs


def _self_value(value):
    return value


def _try_value_line(body, value_func):
    """Compile a bare value line (an instruction line whose first word is
    not a function name): infix expression / nested call / single word.
    Returns the ValueCode or None when the line is not a value form."""
    words = _split_words(body)
    if not words:
        return None
    if len(words) > 1:
        if not any(w in SYMBOL_FUNCS for w in words[1:]):
            return None
        return value_func(" ".join(words))
    return value_func(words[0])


def _parse_cond_call(text, resolve_func, value_func):
    """Parse an IF/WHILE condition: either the classic cond-call form
    (``func arg1 ...``, resolved through resolve_func) or a value
    condition compiled with the unified grammar - a nested call
    (``(f a)``), an infix expression with the built-in operators
    (``data[0] > 5``), a data reference or a plain word; the condition
    then evaluates to its truthiness at run time."""
    words = _split_words(text)
    if not words:
        raise ValueError("empty condition")
    if len(words) > 1 and any(w in SYMBOL_FUNCS for w in words[1:]):
        return value_func(" ".join(words))
    if len(words) == 1:
        try:
            return _parse_instruction_call(text, resolve_func, value_func)
        except KeyError:
            return value_func(words[0])
    return _parse_instruction_call(text, resolve_func, value_func)


def _cond_closure(cond, data):
    """Wrap a condition into a runtime closure bound to the data region:
    classic (fn, args, kwargs) deferred calls and compiled ValueCode
    conditions both re-evaluate on every call (loop conditions see fresh
    data); returns the truthiness."""
    if isinstance(cond, ValueCode):
        env = _shell_env(data)

        def run():
            return bool(cond.execute(env))

        return run
    fn, args, kwargs = cond

    def run():
        rargs, rkwargs = _prepare_call_args(args, kwargs, data)
        return bool(fn(*rargs, **rkwargs))

    return run


def _nearest_while(frames):
    """Depth of the innermost while frame in the execution stack."""
    for depth in range(len(frames) - 1, -1, -1):
        if frames[depth][0] == "while":
            return depth
    return None


def _handle_interrupt(frames, stats, interrupt_window):
    """KeyboardInterrupt policy: the innermost loop yields (condition
    restarts, count unchanged); a second consecutive interrupt terminates
    that loop and unwinds one level.  Returns "retry" or "raise" (the top
    level may re-raise to terminate)."""
    depth = _nearest_while(frames)
    if depth is None:
        note_interrupt(interrupt_window)
        stats["interrupt"] = True
        return "retry"
    frame = frames[depth]
    try:
        note_interrupt()
    except KeyboardInterrupt:
        del frames[depth:]
        return "raise"
    # first interrupt: yield - restart the condition, drop the body frames
    frame[3] = None
    del frames[depth + 1:]
    return "retry"


def execute_instruction_items(items, data=None, print_result=None,
                              interrupt_window=INTERRUPT_WINDOW):
    """Execute a uniform instruction item list (function-style control
    flow) on an explicit frame stack.  DataRef variables resolve against
    the data region (a plain dict by default).  Returns (data, stats).
    First KeyboardInterrupt: recorded, execution continues; second
    consecutive one (within the window): raises to terminate."""
    data = {} if data is None else data
    stats = {"run": 0, "interrupt": False}

    def wrap_cond(cond):
        if isinstance(cond, tuple) and len(cond) == 3 \
                or isinstance(cond, ValueCode):
            return _cond_closure(cond, data)
        return cond

    def finish_pending(parent, result):
        """Complete a suspended IF/WHILE item with the child result
        (write the result position, print hook, stats, frame last)."""
        pending = parent[-1]
        if pending is None:
            return
        parent[-1] = None
        pos = pending[3]
        if pos is not None:
            try:
                data[pos] = result
            except (TypeError, IndexError, KeyError):
                pass
        if print_result is not None:
            print_result(result)
        stats["run"] += 1
        parent[-2] = result

    def handle(item):
        """Process one item; returns a frame to push or None.  A plain
        call updates the current frame's last slot directly."""
        if item is None:
            return None
        fn, args, kwargs, pos = item
        frame = frames[-1]
        if fn is IF:
            cond = wrap_cond(args[0])
            _cond, true_items, false_items = args
            branch = true_items if bool(cond()) else false_items
            if not branch:
                if pos is not None:
                    try:
                        data[pos] = None
                    except (TypeError, IndexError, KeyError):
                        pass
                if print_result is not None:
                    print_result(None)
                stats["run"] += 1
                frame[-2] = None
                return None
            frame[-1] = item
            return ["items", iter(branch), None, None]
        if fn is WHILE:
            cond = wrap_cond(args[0])
            _cond, body_items = args
            frame[-1] = item
            return ["while", cond, body_items, None, 0, None, None]
        args, kwargs = _prepare_call_args(args, kwargs, data)
        result = fn(*args, **kwargs)
        if pos is not None:
            try:
                data[pos] = result
            except (TypeError, IndexError, KeyError):
                pass
        if print_result is not None:
            print_result(result)
        stats["run"] += 1
        frame[-2] = result
        return None

    frames = [["items", iter(items), None, None]]
    while frames:
        try:
            frame = frames[-1]
            if frame[0] == "while":
                if frame[3] is None:
                    if not bool(frame[1]()):
                        frames.pop()
                        if frames:
                            finish_pending(frames[-1], frame[4])
                        continue
                    frame[3] = iter(frame[2])
                try:
                    item = next(frame[3])
                except StopIteration:
                    frame[3] = None
                    frame[4] += 1
                    if _interrupt_pending():
                        frames.pop()
                        if frames:
                            finish_pending(frames[-1], frame[4])
                        continue
                    continue
                new_frame = handle(item)
                if new_frame is not None:
                    frames.append(new_frame)
            else:
                try:
                    item = next(frame[1])
                except StopIteration:
                    frames.pop()
                    if frames:
                        finish_pending(frames[-1], frame[2])
                    continue
                new_frame = handle(item)
                if new_frame is not None:
                    frames.append(new_frame)
        except KeyboardInterrupt:
            while True:
                if _handle_interrupt(frames, stats,
                                     interrupt_window) != "raise":
                    break
            continue
    return data, stats


def namespace():
    """The shared default execution namespace (function table) used by
    the shell command line and the app default executor."""
    return _ns["funcs"]


def variables():
    """The shared namespace variable table (let / set bindings)."""
    return _ns["vars"]


def variable_indexes():
    """The shared variable-index table for the pointer forms."""
    return _ns["_var_index"]


def ns_set(name, value):
    """Set a namespace variable to a compiled literal value (evaluated
    immediately against the shell environment)."""
    if isinstance(value, str):
        value = compile_value(value)
    env = _shell_env(None)
    _ns["vars"][name] = _arg_value(value, env, None)
    if name not in _ns["_var_index"]:
        _ns["_var_index"][name] = _ns["_var_seq"]
        _ns["_var_seq"] += 1
    return str(_ns["vars"][name])


def ns_get(name):
    val = _ns["vars"].get(name)
    if val is None and "." in name:
        val = _extract_dotted(name)
    return "undefined" if val is None else str(val)


def ns_delete(names):
    removed = []
    for name in names:
        if name in _ns["vars"]:
            del _ns["vars"][name]
            _ns["_var_index"].pop(name, None)
            removed.append(name)
        elif name in _ns["funcs"]:
            del _ns["funcs"][name]
            removed.append(name)
    return "deleted: " + ", ".join(removed) if removed else "nothing to delete"


def _ns_let(name, value):
    """Namespace let (instruction form: the value is already parsed);
    registered so instruction files use let/get/delete uniformly."""
    _ns["vars"][name] = value
    if name not in _ns["_var_index"]:
        _ns["_var_index"][name] = _ns["_var_seq"]
        _ns["_var_seq"] += 1
    return str(value)


def _ns_get(name):
    return ns_get(name)


def _ns_delete(*names):
    return ns_delete(list(names))


_ns["funcs"]["let"] = _ns_let
_ns["funcs"]["get"] = _ns_get
_ns["funcs"]["delete"] = _ns_delete

# instructions whose leading positional arguments are raw names (never
# resolved as variables)
_NAME_ARG_FUNCS = {
    _ns_let: 1,
    _ns_get: 1,
    _ns_delete: -1,
    ns_import: 2,
    import_all_module: 1,
}


def run_file(path, data=None, print_result=None):
    """Read and execute an instruction file (unified protocol: imperative
    calls, result positions, ``data<N>``, IF/WHILE blocks).  Names
    resolve through the shell namespace.  Returns (data, stats)."""
    with open(path, encoding="utf-8-sig") as fh:
        text = fh.read()
    items = parse_instruction_file(text, resolve_callable)
    return execute_instruction_items(items, data, print_result)


def run_shell(args):
    """Execute one shell call; return the result as text."""
    if not args:
        return "usage: shell func arg1 -kw value"
    if args[0] == "let":
        return ns_set(args[1], args[2])
    if args[0] == "get":
        return ns_get(args[1])
    if args[0] == "delete":
        return ns_delete(args[1:])
    if args[0] == "run":
        if len(args) < 2:
            return "usage: run <instruction-file>"
        _data, stats = run_file(args[1])
        return "run: " + str(stats["run"]) + " instructions"
    result = execute_call(*parse_call(" ".join(args)))
    if isinstance(result, dict) and "error" in result:
        return "error: " + result["error"]
    return result


def repl(prompt="... "):
    """Interactive command line operation (prompt ...).  Called by the
    call page when the shell command runs without arguments; exit/quit/q
    returns to the outer call page (>>>).  A first KeyboardInterrupt is
    only recorded; a second consecutive one exits the command line."""
    print("shell command line - enter 'exit' to return")
    while True:
        try:
            line = input(prompt)
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            try:
                note_interrupt()
            except KeyboardInterrupt:
                print()
                return 0
            print("(interrupt: press again to exit)")
            continue
        line = line.strip()
        if not line:
            continue
        if line in ("exit", "quit", "q"):
            return 0
        try:
            result = run_shell(_split_words(line))
        except KeyboardInterrupt:
            try:
                note_interrupt()
            except KeyboardInterrupt:
                print()
                return 0
            continue
        except Exception as exc:  # noqa: BLE001 - intercepted here
            print("error: " + type(exc).__name__ + ": " + str(exc))
            continue
        if result is not None:
            print(result)


if __name__ == "__main__":
    print(run_shell(sys.argv[2:]))   # strip the command name 'shell'
    sys.exit(0)
