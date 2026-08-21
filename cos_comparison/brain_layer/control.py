"""
Control-flow containers (brain layer): sequence / conditional branch / loop.

Flat control flow simulation: iterating a container yields the FUNCTIONS of
the simulated flow — the caller executes them (e.g. `for func in ctrl:
func()`). The containers themselves never invoke the yielded functions
(no delegated execution).

Nested control flow is expanded iteratively by ControlFlatten (explicit
stack, no recursion); ControlFlowDriver composes flows dynamically via
the sequence protocol (driver[i] = flow) and multi-index access
(driver[i, "if_func"] = func).
"""


class Control:
    """Base marker for control-flow containers. Placeholder only: no
    behaviour; each concrete container implements its own protocols."""

    __slots__ = ()


class Sequence(Control):
    """Chain of functions; the sequence protocol maps to func_chain."""

    __slots__ = ("func_chain",)

    def __init__(self, func_chain=None):
        self.func_chain = list(func_chain) if func_chain is not None else []

    def __len__(self):
        return len(self.func_chain)

    def __getitem__(self, index):
        return self.func_chain[index]

    def __setitem__(self, index, value):
        self.func_chain[index] = value

    def __iter__(self):
        return iter(self.func_chain)


class Branch(Control):
    """Yield the chosen branch function once: the first trigger that returns
    truthy wins (its paired func), otherwise the else_func (None when no
    branch matches and no else is configured). Trigger errors are silently
    skipped (branch attempts)."""

    __slots__ = ("if_chain", "else_func", "flag")

    def __init__(self, if_chain=None, else_func=None):
        self.if_chain = if_chain if if_chain is not None else ()
        self.else_func = else_func
        self.flag = 1

    @classmethod
    def _keys(cls):
        return [name for name in cls.__slots__ if not name.startswith("_")]

    def __getitem__(self, key):
        if key not in self._keys():
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key, value):
        if key not in self._keys():
            raise KeyError(key)
        setattr(self, key, value)

    def keys(self):
        return self._keys()

    def __iter__(self):
        self.flag = 1
        return self

    def __next__(self):
        if not self.flag:
            raise StopIteration
        self.flag = 0
        for trigger, func in self.if_chain:
            try:
                if trigger():
                    return func
            except Exception:
                continue
        return self.else_func


class Loop(Control):
    """Yield the loop body functions expanded over the iterations:

        while do_func():
            [continue_func() truthy -> skip this iteration's body]
            yield exec_func
            [yield update_func when configured]
            [break_func() truthy -> stop yielding]

    Condition callbacks (do/continue/break) are evaluated by the iterator
    (control logic); the yielded functions (exec/update) are never invoked
    here — the caller executes them in order.
    """

    __slots__ = ("do_func", "exec_func", "continue_func", "update_func",
                 "break_func", "_pending_update")

    def __init__(self, do_func, exec_func, continue_func=None, update_func=None,
                 break_func=None):
        self.do_func = do_func
        self.exec_func = exec_func
        self.continue_func = continue_func
        self.update_func = update_func
        self.break_func = break_func
        self._pending_update = None

    @classmethod
    def _keys(cls):
        return [name for name in cls.__slots__ if not name.startswith("_")]

    def __getitem__(self, key):
        if key not in self._keys():
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key, value):
        if key not in self._keys():
            raise KeyError(key)
        setattr(self, key, value)

    def keys(self):
        return self._keys()

    def __iter__(self):
        self._pending_update = None
        return self

    def __next__(self):
        if self._pending_update is not None:
            update, self._pending_update = self._pending_update, None
            return update
        while True:
            if not self.do_func():
                raise StopIteration
            if self.continue_func is not None and self.continue_func():
                continue
            if self.break_func is not None and self.break_func():
                raise StopIteration
            if self.update_func is not None:
                self._pending_update = self.update_func
            return self.exec_func


class ControlFlatten:
    """Flatten nested Control containers iteratively (explicit stack, no
    recursion). Owns the expansion state; subclass to customize."""

    __slots__ = ()

    @staticmethod
    def _content(control):
        """Structural items: sequence protocol first, then iteration,
        then mapping keys."""
        length = getattr(control, "__len__", None)
        getitem = getattr(control, "__getitem__", None)
        if callable(length) and callable(getitem):
            return (control[i] for i in range(length()))
        iterator = getattr(control, "__iter__", None)
        if callable(iterator):
            return iter(control)
        keys = getattr(control, "keys", None)
        if callable(keys):
            return (control[k] for k in keys())
        raise TypeError("not a control-flow container: %r" % (control,))

    def flatten(self, control):
        stack = [self._content(control)]
        while stack:
            try:
                item = next(stack[-1])
            except StopIteration:
                stack.pop()
                continue
            if isinstance(item, Control):
                stack.append(self._content(item))
            else:
                yield item

    def __call__(self, control):
        return self.flatten(control)


default_flattener = ControlFlatten()


class ControlFlowDriver(Control):
    """Composable, mutable control-flow container.

    init -> runtime adjustment -> flattened iteration: sibling flows are
    managed via the sequence protocol (driver[i] = flow), nested slots via
    multi-index access (driver[i, "if_func"] = func).
    """

    __slots__ = ("slots", "flattener")

    def __init__(self, slots=None, flattener=None):
        self.slots = list(slots) if slots is not None else []
        self.flattener = flattener if flattener is not None else default_flattener

    def __len__(self):
        return len(self.slots)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            value = self.slots[index[0]]
            for key in index[1:]:
                value = value[key]
            return value
        return self.slots[index]

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            target = self.slots[index[0]]
            for key in index[1:-1]:
                target = target[key]
            target[index[-1]] = value
        else:
            self.slots[index] = value

    def append(self, value):
        self.slots.append(value)

    def keys(self):
        return range(len(self.slots))

    def __iter__(self):
        return self.flattener.flatten(self)
