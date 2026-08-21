"""
Function helper tools: compose functions via a shared stack
(ComposalFunction), standard-module helper slots (FuncHelper), and the
function-wrapping FuncWrap. Standard library only.
"""
#It provides function helper tools.

import functools
import operator


class ComposalFunctionManage:
    """Default management of a ComposalFunction: stack sizing, argument
    lookup and result placement. Subclass or inject to customize."""

    __slots__ = ()

    def make_stack(self, first_res_index, steps_res_start, step_count,
                   last_args_index, last_kwargs_index):
        """Size a fresh shared stack to cover every referenced slot."""
        size = max(first_res_index,
                   steps_res_start + step_count - 1,
                   *(last_args_index or (0,)),
                   *(last_kwargs_index.values() or (0,))) + 1
        return [None] * max(size, 1)

    def place_result(self, stack, index, value):
        stack[index] = value

    def collect_args(self, stack, args_index, kwargs_index):
        return (tuple(stack[i] for i in args_index),
                {name: stack[i] for name, i in kwargs_index.items()})


class ComposalFunction:
    """Callable composed from multiple functions sharing a stack.

    first_func receives the call arguments directly and its result is
    placed into first_res_index. Middle steps are (func, args_index,
    kwargs_index) triples reading arguments from stack slots; step i
    result goes to steps_res_start + i. last_func reads its arguments
    from last_args_index / last_kwargs_index and its result is returned
    to the caller.

    Execution and stack maintenance are delegated to a manager (default:
    ComposalFunctionManage); inject or subclass to customize.
    """

    __slots__ = ("manager", "stack", "first_func", "first_res_index",
                 "steps", "steps_res_start", "last_func",
                 "last_args_index", "last_kwargs_index", "_stack_owned")

    def __init__(self, first_func, last_func, steps=(), stack=None,
                 first_res_index=0, steps_res_start=1,
                 last_args_index=(), last_kwargs_index=None, manager=None):
        self.manager = manager if manager is not None else ComposalFunctionManage()
        self.first_func = first_func
        self.last_func = last_func
        self.steps = list(steps)
        self.first_res_index = first_res_index
        self.steps_res_start = steps_res_start
        self.last_args_index = tuple(last_args_index)
        self.last_kwargs_index = dict(last_kwargs_index) if last_kwargs_index else {}
        self._stack_owned = stack is None
        self.stack = (stack if stack is not None
                      else self.manager.make_stack(first_res_index, steps_res_start,
                                                   len(self.steps),
                                                   self.last_args_index,
                                                   self.last_kwargs_index))

    def __call__(self, *args, **kwargs):
        stack = self.stack
        if self._stack_owned:
            for i in range(len(stack)):
                stack[i] = None        # fresh state per call
        self.manager.place_result(stack, self.first_res_index,
                                  self.first_func(*args, **kwargs))
        for step_index, (func, args_index, kwargs_index) in enumerate(self.steps):
            args, kwargs = self.manager.collect_args(stack, args_index, kwargs_index)
            self.manager.place_result(stack, self.steps_res_start + step_index,
                                      func(*args, **kwargs))
        args, kwargs = self.manager.collect_args(stack, self.last_args_index,
                                                 self.last_kwargs_index)
        return self.last_func(*args, **kwargs)


def _default_compose(*funcs):
    """Compose funcs right-to-left: compose(f, g)(x) equals f(g(x))."""
    if not funcs:
        raise ValueError("compose requires at least one function")
    def composed(arg):
        value = arg
        for func in reversed(funcs):
            value = func(value)
        return value
    return composed


class FuncHelper:
    """Standard-module function helpers with delegated slots.

    Each slot defaults to the matching stdlib implementation
    (functools / operator); inject a replacement to override.
    """

    __slots__ = ("partial_func", "reduce_func", "compose_func",
                 "wrap_func", "itemgetter_func", "attrgetter_func")

    def __init__(self, partial_func=None, reduce_func=None, compose_func=None,
                 wrap_func=None, itemgetter_func=None, attrgetter_func=None):
        self.partial_func = partial_func if partial_func is not None else functools.partial
        self.reduce_func = reduce_func if reduce_func is not None else functools.reduce
        self.compose_func = compose_func if compose_func is not None else _default_compose
        self.wrap_func = wrap_func if wrap_func is not None else functools.wraps
        self.itemgetter_func = itemgetter_func if itemgetter_func is not None else operator.itemgetter
        self.attrgetter_func = attrgetter_func if attrgetter_func is not None else operator.attrgetter


class FuncWrap:
    """Wrap a function invoked with the first argument discarded.
    Copied from brain_layer.mapper.base_map (FuncWrap) for reuse."""

    __slots__ = ("func",)

    def __init__(self, func):
        self.func = func

    def __call__(self, first_arg, *args, **kwargs):
        return self.func(*args, **kwargs)

"""
Explicit public exports (prevents import-star namespace pollution).
"""
__all__ = (
    "ComposalFunctionManage",
    "ComposalFunction",
    "FuncHelper",
    "FuncWrap",
)
