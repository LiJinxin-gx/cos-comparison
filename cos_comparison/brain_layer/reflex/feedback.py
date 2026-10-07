#Trigger-style reflex hub: paired feedback objects and feedback functions.

from ...interface.api.parallel_api import Thread

__all__ = (
    "Feedback",
    "default_dispatch", "default_manage", "default_receive",
)


def default_receive(hub):
    """Hub-receive (default): trigger non-blocking - dispatch on a
    background thread and return the thread handle."""
    thread = Thread(target=hub.dispatch)
    thread.start()
    return thread


def default_dispatch(hub):
    """Hub-feedback (default): hand each feedback function its paired
    object; errors are caught (kept on hub.last_error) and the dispatch
    continues.  Returns 0 when every call succeeded, else 1."""
    status = 0
    for obj, func in list(hub.registry):
        try:
            func(obj)
        except Exception as exc:
            hub.last_error = exc
            status = 1
    return status


def default_manage(hub, operation, *args):
    """Hub management (default): op 'add' (obj, func) / 'remove' (obj) /
    'has' (obj) / 'len'."""
    if operation == "add":
        obj, func = args
        hub.registry.append((obj, func))
        return 0
    if operation == "remove":
        (obj,) = args
        for i, (o, _f) in enumerate(hub.registry):
            if o is obj or o == obj:
                del hub.registry[i]
                return 0
        return 1
    if operation == "has":
        (obj,) = args
        return any(o is obj or o == obj for o, _f in hub.registry)
    if operation == "len":
        return len(hub.registry)
    raise ValueError(f"unknown manage operation: {operation!r}")


class Feedback:
    """Reflex feedback hub: paired (feedback object, feedback function)
    registrations; the hub-receive function triggers (default: non-
    blocking), the hub-feedback function hands every feedback function
    its own paired object (Trigger-style - the object is the shared
    carrier the function operates on).  All hub functions are delegating
    slots with working defaults."""
    __slots__ = ("dispatch_func", "last_error", "manage_func",
                 "receive_func", "registry")

    def __init__(self, receive_func=None, dispatch_func=None,
                 manage_func=None):
        self.registry = []  # [(obj, func), ...]
        self.last_error = None
        self.receive_func = (default_receive
                             if receive_func is None else receive_func)
        self.dispatch_func = (default_dispatch
                              if dispatch_func is None else dispatch_func)
        self.manage_func = (default_manage
                            if manage_func is None else manage_func)

    # -- single-line delegations -----------------------------------------
    def register(self, obj, func):
        """Pair a feedback object with its feedback function."""
        return self.manage_func(self, "add", obj, func)

    def remove(self, obj):
        """Remove the pair of a feedback object (1 when absent)."""
        return self.manage_func(self, "remove", obj)

    def has(self, obj):
        """Whether a feedback object is registered."""
        return self.manage_func(self, "has", obj)

    def __len__(self):
        return self.manage_func(self, "len")

    def receive(self, *args, **kwargs):
        """Hub-receive: trigger the paired feedback functions (delegated -
        default non-blocking, returns the background thread)."""
        return self.receive_func(self, *args, **kwargs)

    def dispatch(self, *args, **kwargs):
        """Hub-feedback: hand each feedback function its paired object
        (delegated - default synchronous, returns 0/1)."""
        return self.dispatch_func(self, *args, **kwargs)

    def manage(self, operation, *args, **kwargs):
        """Hub management entry (delegated)."""
        return self.manage_func(self, operation, *args, **kwargs)
