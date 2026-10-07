"""
Default function interfaces for the Docker container (protocol style).

Provides the complete default function set so a bare Docker is runnable:
reading and running are decoupled (read_* parses an arrangement, submit_*
hands it to the docker, run() executes it); the default reader uses the
unified instruction-file protocol shared with shell/batch; the default run
is driven by the brain-layer control flow and the action-layer operate
flow.  Every method is a *default interface*: the Docker instance binds
its corresponding attribute to this method; replacing an attribute (or the
whole manager) replaces that interface.
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
from ..shell_tool.value import (
    ValueCode,
    ValueEnv,
)

__all__ = ("DataRef", "DockerProtocol", "SuspendSnapshot", "subdocker")


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

    # ---------- suspend / resume (step-boundary snapshot) ----------

    def suspend(self, docker, *args, **kwargs):
        """Suspend: request a stop at the next step boundary, wait for the
        running execution to finish, and return a snapshot (SuspendSnapshot
        export)."""
        docker.suspended = True
        handle = getattr(docker, "_run_handle", None)
        if handle is not None:
            handle.wait(*args, **kwargs)
        return SuspendSnapshot(
            data_pool=_copy_pool(docker.data_pool),
            operate_pool=_copy_pool(docker.operate_pool),
            interface_pool=_copy_pool(docker.interface_pool),
            extension_pool=_copy_pool(docker.extension_pool),
            function_pool=_copy_pool(docker.function_pool),
            done_steps=docker.done_steps,
            error=docker.error,
            terminated=docker.terminated,
            suspended=docker.suspended,
            manager=docker.manager,
            namespace=docker.namespace,
            worker=docker.worker)

    def resume(self, docker, snapshot=None, *args, **kwargs):
        """Resume: load a snapshot back onto the docker (pools, cursor and
        flags); the next run/start continues after the recorded cursor
        (already-done steps are skipped, their side effects kept)."""
        if snapshot is None:
            return 0
        for name in SuspendSnapshot.__slots__:
            if name != "suspended":
                setattr(docker, name, getattr(snapshot, name))
        docker.suspended = False
        docker._resume_from = snapshot.done_steps
        return 1

    # ---------- reading (parses an arrangement from a source) ----------

    def read_operate_flow(self, docker, source=None, *args, **kwargs):
        """Read the operate flow.  With *source* given (a readable object
        exposing ``read()``, or the text itself) the default reader parses
        the imperative shell format:

            func arg1 arg2 -kw value -> result_pos

        one operation per line; ``data<N>`` becomes DataRef(N) (a data_pool
        index reference).  Without *source* it reads the arrangement
        already submitted (docker.operate_pool)."""
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
        """Submit operate items to the docker (generic interface used by
        reading logic to commit an operate arrangement; duck: any iterable
        of items)."""
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
        (docker.terminated) is checked before each step.  After resume()
        the already-done steps (docker._resume_from) are skipped."""
        resume_from = getattr(docker, "_resume_from", 0) or 0
        docker._resume_from = 0
        if not resume_from:
            docker.done_steps = 0
        steps = self._build_steps(docker)
        if resume_from:
            steps = steps[resume_from:]
        driver = ExecuterDriver(
            caller_list=steps,
            worker_func=getattr(docker, "worker", None))
        docker._run_handle = driver
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


class SuspendSnapshot:
    """Docker suspend state: shallow pool copies, the run cursor
    (done_steps), flags, and configuration references (manager / namespace
    / worker) for same-process resume."""
    __slots__ = ("data_pool", "done_steps", "error", "extension_pool",
                 "function_pool", "interface_pool", "manager", "namespace",
                 "operate_pool", "suspended", "terminated", "worker")
    def __init__(self, **state):
        for name in self.__slots__:
            setattr(self, name, state.get(name))


def _copy_pool(pool):
    """Shallow-copy a duck pool (dict / sequence); other carriers are kept
    as-is."""
    if isinstance(pool, dict):
        return pool.copy()
    try:
        return pool[:]
    except TypeError:
        return pool


def _step(fn, docker, packed, pos):
    """One executable step: run fn (data injection decided by the function
    itself), resolve DataRef args, store the result at the target data_pool
    position.  An exception terminates the run by default (store_error +
    docker.terminated); a suspend request (docker.suspended) stops the run
    cleanly at this boundary."""
    def execute():
        if getattr(docker, "terminated", False) or \
                getattr(docker, "suspended", False):
            return None
        try:
            if packed is None:
                result = _apply(fn, docker)
            else:
                args, kwargs = packed
                args, kwargs = _prepare_args(
                    args, kwargs, docker.data_pool, _docker_env(docker))
                result = _apply_unpacked(fn, args, kwargs, docker)
        except Exception as exc:  # default termination (maintainer decides)
            docker.store_error(exc)
            docker.terminated = True
            raise
        _set_data(docker.data_pool, pos, result)
        docker.done_steps += 1
        return result
    return execute

def _resolve_arg(value, data, env=None):
    """Resolve one argument value: compiled ValueCode executes against
    the docker environment; DataRef variable references resolve against
    the data pool; anything else passes through."""
    if isinstance(value, ValueCode):
        if env is None:
            raise ValueError("compiled value needs an environment")
        return value.execute(env)
    if isinstance(value, DataRef):
        return value.resolve(data)
    return value


def _docker_env(docker):
    """The execution environment for compiled values: the data pool as
    the data region, the docker namespace as the name resolver, and the
    shared shell variable tables for the pointer/word forms."""
    from ..shell_tool.shell import variable_indexes, variables
    return ValueEnv(data=docker.data_pool,
                    vars=variables(),
                    resolve=lambda name: _resolve_func(name, docker),
                    var_index=variable_indexes())


def _prepare_args(args, kwargs, data, env=None):
    """Prepare a call (app layer, self-contained): execute compiled
    values and resolve DataRef variables and expand the unpack markers
    (``%`` sequence unpack into positional arguments, ``%%`` mapping
    unpack into keyword arguments)."""
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
            value = _resolve_arg(a.value, data, env)
            flat.extend(value)
        elif isinstance(a, UnpackMapRef):
            kwargs.update(_resolve_arg(a.value, data, env))
        else:
            flat.append(_resolve_arg(a, data, env))
    kwargs = {k: _resolve_arg(v, data, env) for k, v in kwargs.items()}
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
    (instruction text or an operate item list), run synchronously, and
    return its data_pool.  A child error raises."""
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
    """Wrap a deferred condition into a runtime closure bound to the
    docker: a compiled ValueCode condition or a classic (fn, args,
    kwargs) deferred call - DataRef variables resolve against the data
    pool on every call (loop conditions see fresh data); returns
    truthiness."""
    if isinstance(cond, ValueCode):
        env = _docker_env(docker)

        def run():
            return bool(cond.execute(env))

        return run
    fn, args, kwargs = cond

    def run():
        env = _docker_env(docker)
        rargs, rkwargs = _prepare_args(args, kwargs, docker.data_pool, env)
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
    if isinstance(cond, tuple) and len(cond) == 3 \
            or isinstance(cond, ValueCode):
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
    """Wrap a branch's item list as a runnable closure on an explicit
    frame stack: the brain-layer Sequence flattens at run time, nested
    IF/WHILE items expand onto the stack, each step runs against the
    docker; returns the last step result."""
    flow = Sequence(items)
    env = _docker_env(docker)
    frames = []

    def wrap_cond(cond):
        if isinstance(cond, tuple) and len(cond) == 3 \
                or isinstance(cond, ValueCode):
            return _control_closure(docker, cond)
        return cond

    def finish_pending(parent, result):
        pending = parent[-1]
        if pending is None:
            return
        parent[-1] = None
        _set_data(docker.data_pool, pending[0], result)
        parent[-2] = result

    def handle(item):
        fn, args, kwargs, pos = _unpack(item)
        frame = frames[-1]
        if fn is IF:
            cond = wrap_cond(args[0])
            _cond, true_items, false_items = args
            branch = true_items if bool(cond()) else false_items
            if not branch:
                _set_data(docker.data_pool, pos, None)
                frame[-2] = None
                return None
            frame[-1] = (pos,)
            return ["items", ControlFlatten()(Sequence(branch)), None, None]
        if fn is WHILE:
            cond = wrap_cond(args[0])
            _cond, body_items = args
            frame[-1] = (pos,)
            return ["while", cond, body_items, None, 0, None, None]
        rargs, rkwargs = _prepare_args(args, kwargs, docker.data_pool, env)
        result = _apply_unpacked(fn, rargs, rkwargs, docker)
        _set_data(docker.data_pool, pos, result)
        frame[-2] = result
        return None

    def run():
        frames[:] = [["items", ControlFlatten()(flow), None, None]]
        last = None
        while frames:
            frame = frames[-1]
            if frame[0] == "while":
                if frame[3] is None:
                    if not bool(frame[1]()):
                        frames.pop()
                        if frames:
                            finish_pending(frames[-1], frame[4])
                        continue
                    frame[3] = ControlFlatten()(Sequence(frame[2]))
                try:
                    item = next(frame[3])
                except StopIteration:
                    frame[3] = None
                    frame[4] += 1
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
                    else:
                        last = frame[2]
                    continue
                new_frame = handle(item)
                if new_frame is not None:
                    frames.append(new_frame)
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
    """Execute one unpacked operate item.  Data injection is by name: a
    first positional parameter named ``data`` receives the data_pool, one
    named ``docker`` receives the docker."""
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
