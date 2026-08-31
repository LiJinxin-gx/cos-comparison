"""
Default function interfaces for the Docker container (protocol style).

Provides a complete set of default functions so a bare Docker is runnable:
reading and running are decoupled (read_* parses an arrangement,
submit_* hands it to the docker, run() executes it); the default reader
uses the unified instruction file protocol shared with shell/batch; the
default run is driven by the brain-layer control flow and the
action-layer operate flow.  Every method is a *default interface*: the
Docker instance binds its corresponding attribute to this method;
replacing an attribute (or the whole manager) replaces that interface.
"""

import inspect
from collections.abc import Mapping
from collections.abc import Sequence as _abc_Sequence

from ..action_layer.executer import ExecuterDriver
from ..brain_layer.control import Control, ControlFlatten, Sequence
from ..core import no_done
from ..shell_tool.shell import (
    IF,
    WHILE,
    DataRef,
    UnpackMapRef,
    UnpackRef,
    parse_instruction_file,
)

__all__ = ("DataRef", "DockerProtocol", "subdocker")


class DockerProtocol:
    """Complete set of default functions for the Docker container."""

    def manage(self, docker, *args, **kwargs):
        """Manage the docker's internal state (e.g. terminate / send data
        to the outside).  Default: do nothing (core.no_done semantics)."""
        return no_done(*args, **kwargs)

    def store_error(self, docker, exc, *args, **kwargs):
        """Store an exception instance for later inspection (default: keep
        the latest one on docker.error; does not swallow the exception)."""
        docker.error = exc
        return 0

    # ---------- reading (parses an arrangement from a source) ----------

    def read_operate_flow(self, docker, source=None, *args, **kwargs):
        """Read the operate flow.  With *source* given (text, or a file
        path) the default reader parses the imperative shell format:

            func arg1 arg2 -kw value -> result_pos

        one operation per line; ``data<N>`` becomes DataRef(N) (data_pool
        index references).  Without *source* it reads the arrangement
        already submitted (iterate docker.operate_pool)."""
        if source is None:
            return list(docker.operate_pool)
        if hasattr(source, "read"):
            text = source.read()
        else:
            text = source
        return _parse_shell_flow(text, docker)

    def read_control_flow(self, docker, source=None, *args, **kwargs):
        """Read the control flow (brain_layer Control containers) from the
        arrangement - an extension point for inspecting / wrapping /
        injecting control structures."""
        return [op for op in docker.operate_pool
                if isinstance(op, Control)]

    # ---------- submit (hand an arrangement to the docker) ----------

    def submit_operate_flow(self, docker, items, *args, **kwargs):
        """Submit operate items to the docker (the generic interface used
        by reading logic to commit an operate arrangement; duck: any
        iterable of items)."""
        for item in items:
            docker.operate_pool.append(item)
        return 0

    def submit_control_flow(self, docker, flow, *args, **kwargs):
        """Submit a control flow container to the docker (generic interface
        used by reading logic to commit a control arrangement)."""
        docker.operate_pool.append(flow)
        return 0

    # ---------- running (brain layer control + action layer operate) ----------

    def run(self, docker, *args, **kwargs):
        """Run: drive the arrangement.  Control containers are flattened /
        expanded into steps; steps run non-blocking through ExecuterDriver
        on a background worker, each storing its result at
        data_pool[result_pos].  Exceptions terminate by default (the
        maintainer decides whether to catch); the stop flag
        (docker.terminated) is checked before each step."""
        steps = self._build_steps(docker)
        driver = ExecuterDriver(
            caller_list=steps,
            worker_func=getattr(docker, "worker", None))
        driver.call_all()
        return driver

    def _build_steps(self, docker):
        """Expand the arrangement into executable step callables.
        Control-flow items may yield plain callables or operate items
        (4-tuples), unpacked uniformly; IF/WHILE conditions become
        docker-bound closures and branches become runners."""
        steps = []
        index = 0
        for op in docker.operate_pool:
            items = ControlFlatten()(op) if isinstance(op, Control) \
                else (op,)
            for item in items:
                if item is None:
                    continue
                fn, args, kwargs, result_pos = _unpack(item)
                pos = index if result_pos is None else result_pos
                args = _prepare_control(fn, args, docker)
                steps.append(_step(fn, docker, (args, kwargs), pos))
                index += 1
        return steps

    def start(self, docker, run):
        """Start the docker to run.  Default: launch run non-blocking
        (background thread); inject a different starter to change the
        concurrency model."""
        from ..interface.api.parallel_api import Thread
        thread = Thread(target=run)
        thread.start()
        return thread


def _step(fn, docker, packed, pos):
    """One executable step: run fn (with data injection decided by the
    function itself), resolve DataRef args, store the result at the
    target position of data_pool.  An exception terminates the run by
    default (stored via store_error, docker.terminated set)."""
    def execute():
        if getattr(docker, "terminated", False):
            return None
        try:
            if packed is None:
                result = _apply(fn, docker)
            else:
                args, kwargs = packed
                args, kwargs = _prepare_args(
                    args, kwargs, docker.data_pool)
                result = _apply_unpacked(fn, args, kwargs, docker)
        except Exception as exc:  # default termination (maintainer decides)
            docker.store_error(exc)
            docker.terminated = True
            raise
        _set_data(docker.data_pool, pos, result)
        return result
    return execute

def _resolve_arg(value, data):
    """Resolve DataRef variable references against the data pool."""
    if isinstance(value, DataRef):
        return value.resolve(data)
    return value


def _prepare_args(args, kwargs, data):
    """Prepare a call (app layer, self-contained): resolve DataRef
    variables and expand the unpack markers (``%`` sequence unpack into
    positional arguments, ``%%`` mapping unpack into keyword
    arguments)."""
    flat = []
    for a in args:
        if isinstance(a, UnpackRef):
            flat.extend(_resolve_arg(a.value, data))
        elif isinstance(a, UnpackMapRef):
            kwargs.update(_resolve_arg(a.value, data))
        else:
            flat.append(_resolve_arg(a, data))
    kwargs = {k: _resolve_arg(v, data) for k, v in kwargs.items()}
    return flat, kwargs


def _parse_shell_flow(text, docker):
    """Default reader: parse the unified instruction file protocol (the
    format shared with shell/batch) into operate items; ``data<N>``
    becomes DataRef(N).  Function names resolve through the docker
    namespace (default {}) then the shell namespace; IF/ELSE/END and
    WHILE/END blocks become control-flow items calling IF / WHILE."""
    return parse_instruction_file(
        text, lambda name: _resolve_func(name, docker))


def subdocker(docker, flow, data=None, timeout=None):
    """External delegation: create a child Docker (copies the parent's
    pools, reuses its manager / namespace / worker), read ``flow``
    (instruction text or an operate item list), run synchronously
    (waited), and return its data_pool.  A child error raises."""
    from .docker import Docker
    sub = Docker(
        data_pool=dict(docker.data_pool) if data is None else data,
        interface_pool=list(docker.interface_pool),
        extension_pool=list(docker.extension_pool),
        manager=docker.manager,
        namespace=docker.namespace,
        worker=docker.worker)
    if isinstance(flow, str):
        sub.operate_pool = sub.read_operate_flow(flow)
    else:
        sub.operate_pool = flow
    driver = sub.run()
    driver.wait(timeout=timeout)
    if sub.error is not None:
        raise sub.error
    return sub.data_pool


def _control_closure(docker, cond):
    """Wrap a deferred condition call (fn, args, kwargs) into a runtime
    closure bound to the docker: DataRef variables resolve against the
    data pool on every call (loop conditions see fresh data); returns
    truthiness."""
    fn, args, kwargs = cond

    def run():
        rargs, rkwargs = _prepare_args(args, kwargs, docker.data_pool)
        return bool(_apply_unpacked(fn, rargs, rkwargs, docker))

    return run


def _prepare_control(fn, args, docker):
    """Wrap a control-flow item (IF/WHILE): the deferred condition becomes
    a docker-bound closure, nested branch lists become runners (each
    branch its own sequential loop - no recursion).  Plain items pass
    through unchanged."""
    if fn not in (IF, WHILE):
        return args
    cond = args[0]
    if isinstance(cond, tuple) and len(cond) == 3:
        cond = _control_closure(docker, cond)
    if fn is IF:
        true_items, false_items = args[1], args[2]
        true_runner = _items_runner(docker, true_items)
        false_runner = (_items_runner(docker, false_items)
                        if false_items else None)
        return (cond, true_runner, false_runner)
    body_items = args[1]
    return (cond, _items_runner(docker, body_items))


def _items_runner(docker, items):
    """Wrap a branch's item list as a runnable closure: a brain-layer
    Sequence flattened at run time, each step executed against the docker
    (nested control-flow items wrapped the same way); returns the last
    step result."""
    flow = Sequence(items)

    def run():
        last = None
        for item in ControlFlatten()(flow):
            if item is None:
                continue
            fn, args, kwargs, pos = _unpack(item)
            args = _prepare_control(fn, args, docker)
            args, kwargs = _prepare_args(args, kwargs, docker.data_pool)
            result = _apply_unpacked(fn, args, kwargs, docker)
            _set_data(docker.data_pool, pos, result)
            last = result
        return last

    return run


def _resolve_func(name, docker):
    """Resolve a function name: docker namespace first, then registered
    shell namespace, then module reflection (module.func)."""
    namespace = getattr(docker, "namespace", None)
    if isinstance(namespace, Mapping) and name in namespace:
        return namespace[name]
    from ..shell_tool.shell import resolve_callable
    try:
        return resolve_callable(name)
    except KeyError:
        raise ValueError("unknown function: " + name)


def _unpack(op):
    """Unpack an operate item: default form is the 4-tuple (callable,
    args, kwargs, result_pos); bare callable / 2-tuple / 3-tuple are
    tolerated.  Generic types: any Sequence (not str/bytes), any
    Mapping."""
    if isinstance(op, _abc_Sequence) and not isinstance(op, (str, bytes)):
        fn = op[0]
        args = op[1] if len(op) > 1 else ()
        if not isinstance(args, _abc_Sequence):
            args = (args,)
        kwargs = op[2] if len(op) > 2 and isinstance(op[2], Mapping) else {}
        result_pos = op[3] if len(op) > 3 else None
        return fn, args, kwargs, result_pos
    return op, (), {}, None


def _apply_unpacked(fn, args, kwargs, docker):
    """Execute one unpacked operate item.  Data injection is decided by
    the function itself: a first positional parameter named ``data``
    receives the data_pool, one named ``docker`` receives the docker."""
    name = _first_param_name(fn)
    if name == "data":
        args = (docker.data_pool,) + tuple(args)
    elif name == "docker":
        args = (docker,) + tuple(args)
    return fn(*args, **kwargs)


def _apply(fn, docker):
    """Execute one bare control-flow function (data injection by name)."""
    name = _first_param_name(fn)
    if name == "data":
        return fn(docker.data_pool)
    if name == "docker":
        return fn(docker)
    return fn()


def _set_data(data, index, result):
    """Write a step result into the data container (duck typing):
    data[index] = result when supported, otherwise left untouched."""
    try:
        data[index] = result
    except (TypeError, IndexError, KeyError):
        pass


def _first_param_name(fn):
    """Name of the first positional parameter of the callable, or None."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    for param in sig.parameters.values():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            return param.name
        if param.kind == param.VAR_POSITIONAL:
            return None
    return None
