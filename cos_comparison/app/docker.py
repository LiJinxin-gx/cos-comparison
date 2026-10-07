"""
It provides docker to run.
"""

import threading
from abc import ABC, abstractmethod

from ..core import no_done
from ..shell_tool.shell import namespace as _shell_namespace
from .protocol import DockerProtocol, SuspendSnapshot


class BaseDocker(ABC):
    @abstractmethod
    def run(self):
        pass

class Docker(BaseDocker):
    __slots__ = ("_resume_from","_run_handle","controller","data_pool",
                 "done_steps","error","extension_pool","function_pool",
                 "interface_pool","maintainer","manager","namespace",
                 "operate_pool","starter","suspended","terminated","worker")
    def __init__(self,
                 data_pool=None,operate_pool=None,controller=None,
                 starter=None,maintainer=None,
                 interface_pool=None,extension_pool=None,
                 function_pool=None,manager=None,namespace=None,
                 worker=None):
        # Pools kept as given (duck typed - dict, list, ...); None -> the
        # default empty dict / list.
        self.data_pool = {} if data_pool is None else data_pool
        self.operate_pool = [] if operate_pool is None else operate_pool
        self.controller = no_done if controller is None else controller
        # starter/maintainer: None -> default execution (start / run); an
        # explicit no_done keeps the historical no-op.
        self.starter = self.default_start if starter is None else starter
        self.maintainer = self.default_run if maintainer is None else maintainer
        self.interface_pool = [] if interface_pool is None else interface_pool
        # Exposes the public interface of the instance to external calls.
        self.extension_pool = [] if extension_pool is None else extension_pool
        # Function pool (duck typed - dict by default, sequences accepted):
        # pooled functions receive self (this docker) as their first
        # argument; submit/call/remove operate by key (index marker).
        self.function_pool = {} if function_pool is None else function_pool
        # Manager: protocol instance providing the complete set of default
        # function interfaces (for later extension).
        self.manager = manager if manager is not None else DockerProtocol()
        # Namespace for the default reader (function name resolution);
        # worker drives the operate flow (default: background thread).
        # The default namespace is the shared shell namespace (project
        # modules, dot extraction, import_module / list_modules).
        self.namespace = _shell_namespace() if namespace is None else namespace
        self.worker = worker
        self.error = None      # exception instance storage (store_error)
        self.terminated = False
        self.suspended = False    # cooperative suspend request
        self.done_steps = 0       # completed top-level steps (run cursor)
        self._resume_from = 0     # one-shot continuation cursor (resume)
        self._run_handle = None   # current driver (suspend waits on it)
    def run(self,*args,**kwargs):
        """Run the docker (maintainer entry)."""
        return self.maintainer(self,*args,**kwargs)
    def start(self):
        """Start the docker (starter entry)."""
        return self.starter(self.run)
    def manage(self,*args,**kwargs):
        """Manage the docker's internal state (e.g. terminate / send data
        to the outside).  Default: do nothing."""
        return self.manager.manage(self,*args,**kwargs)
    def suspend(self,*args,**kwargs):
        """Suspend: stop at the next step boundary and return a snapshot
        (SuspendSnapshot - pool copies, cursor and flags)."""
        return self.manager.suspend(self,*args,**kwargs)
    def resume(self,snapshot=None,*args,**kwargs):
        """Resume: load a snapshot back; the next run/start continues
        after the recorded cursor (already-done steps are skipped)."""
        return self.manager.resume(self,snapshot,*args,**kwargs)
    def submit_function(self, func, key=None):
        """Submit a function into the pool; every invocation receives self
        (this docker) as its first argument.  key defaults to the next free
        integer (mapping pools) or to append (sequence pools); returns the
        key (index marker)."""
        pool = self.function_pool
        if key is None and not isinstance(pool, dict):
            pool.append(func)
            return len(pool) - 1
        if key is None:
            key = 0
            while key in pool:
                key += 1
        pool[key] = func
        return key

    def call_function(self, key, *args, **kwargs):
        """Call pool[key] with self injected as the first argument."""
        return self.function_pool[key](self, *args, **kwargs)

    def remove_function(self, key):
        """Remove pool[key] (KeyError/IndexError when absent)."""
        del self.function_pool[key]

    def start_functions(self, *args, **kwargs):
        """Start each pooled function on its own background thread (each
        invocation receives self); returns the pool size."""
        pool = self.function_pool
        funcs = list(pool.values()) if isinstance(pool, dict) \
            else list(pool)
        for func in funcs:
            threading.Thread(target=func, args=(self,) + args,
                            kwargs=kwargs).start()
        return len(funcs)

    def stop_functions(self):
        """Stop the pool: clear the pooled functions."""
        pool = self.function_pool
        if isinstance(pool, dict):
            pool.clear()
        else:
            del pool[:]
    def store_error(self,exc,*args,**kwargs):
        """Store an exception instance for later inspection (default: keep
        the latest one on docker.error; does not swallow the exception)."""
        return self.manager.store_error(self,exc,*args,**kwargs)
    def read_operate_flow(self,source=None,*args,**kwargs):
        """Read the operate flow.  With *source* (text or file) the default
        reader parses the imperative shell format (func arg1 arg2 -kw value
        -> result_pos; data<N> becomes a data_pool index reference)."""
        return self.manager.read_operate_flow(self,source,*args,**kwargs)
    def read_control_flow(self,*args,**kwargs):
        """Read the control flow (brain_layer Control containers) from the
        arrangement - an extension point."""
        return self.manager.read_control_flow(self,*args,**kwargs)
    def submit_operate_flow(self,items,*args,**kwargs):
        """Submit operate items to the docker (generic interface used by
        reading logic to commit an operate arrangement)."""
        return self.manager.submit_operate_flow(self,items,*args,**kwargs)
    def submit_control_flow(self,flow,*args,**kwargs):
        """Submit a control flow container to the docker (generic interface
        used by reading logic to commit a control arrangement)."""
        return self.manager.submit_control_flow(self,flow,*args,**kwargs)
    def default_run(self,*args,**kwargs):
        """Default execution (maintainer default): brain-layer control flow
        drives the arrangement; the action-layer operate flow runs the
        steps via ExecuterDriver (non-blocking background worker)."""
        return self.manager.run(self,*args,**kwargs)
    def default_start(self,run):
        """Default start: launch run on a background thread (non-blocking)."""
        return self.manager.start(self,run)

"""
Explicit public exports (prevents import-star namespace pollution).
"""
__all__ = (
    "BaseDocker",
    "Docker",
    "DockerProtocol",
    "SuspendSnapshot",
)
