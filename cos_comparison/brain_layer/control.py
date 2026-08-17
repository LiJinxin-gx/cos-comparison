"""
Control-flow containers (brain layer): sequence / conditional branch / loop.

Flat control flow simulation: iterating a container yields the FUNCTIONS of
the simulated flow — the caller executes them (e.g. `for func in ctrl:
func()`). The containers themselves never invoke the yielded functions
(no delegated execution); nested control flow will be composed later by a
dedicated mechanism, not here.
"""


class Control:
    """Base marker for control-flow containers."""


class Sequence(Control):
    """Yield the chain functions in order."""

    __slots__ = ("func_chain",)

    def __init__(self, func_chain=None):
        self.func_chain = func_chain if func_chain is not None else ()

    def __iter__(self):
        for func in self.func_chain:
            yield func


class Brance(Control):
    """Yield the chosen branch function once: the first trigger that returns
    truthy wins (its paired func), otherwise the else_func (None when no
    branch matches and no else is configured). Trigger errors are silently
    skipped (branch attempts)."""

    __slots__ = ("if_chain", "else_func", "flag")

    def __init__(self, if_chain=None, else_func=None):
        self.if_chain = if_chain if if_chain is not None else ()
        self.else_func = else_func
        self.flag = 1

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
