"""
It provides docker to run.
"""

from abc import ABC, abstractmethod

from ..core import no_done
from ..shell_tool.shell import namespace as _shell_namespace
from .protocol import DockerProtocol


class BaseDocker(ABC):
    @abstractmethod
    def run(self):
        pass

class Docker(BaseDocker):
    __slots__ = ("controller","data_pool","error","extension_pool",
                 "interface_pool","maintainer","manager","namespace",
                 "operate_pool","starter","terminated","worker")
    def __init__(self,
                 data_pool=None,operate_pool=None,controller=None,
                 starter=None,maintainer=None,
                 interface_pool=None,extension_pool=None,
                 manager=None,namespace=None,worker=None):
        # Data flow kept as given (duck typing - dict, list, ...); None
        # gives the default empty dict.
        self.data_pool = {} if data_pool is None else data_pool
        self.operate_pool = [] if operate_pool is None else operate_pool
        self.controller = no_done if controller is None else controller
        # Defaults: None selects the default execution (protocol run / non-
        # blocking start); an explicit no_done keeps the historical no-op.
        self.starter = self.default_start if starter is None else starter
        self.maintainer = self.default_run if maintainer is None else maintainer
        self.interface_pool = [] if interface_pool is None else interface_pool
        # Exposes the public interface of the instance to external calls.
        self.extension_pool = [] if extension_pool is None else extension_pool
        # Manager: stores the protocol-class instance (complete set of
        # default function interfaces) for later extension.
        self.manager = manager if manager is not None else DockerProtocol()
        # Namespace for the default reader (function name resolution);
        # worker drives the operate flow (default: background thread).
        # The default namespace is the shared shell namespace (project
        # modules, dot extraction, import_module / list_modules).
        self.namespace = _shell_namespace() if namespace is None else namespace
        self.worker = worker
        self.error = None      # exception instance storage (store_error)
        self.terminated = False
    def run(self,*args,**kwargs):
        """
        Really run.
        """
        return self.maintainer(self,*args,**kwargs)
    def start(self):
        """
        To start docker to run.
        """
        return self.starter(self.run)
    def manage(self,*args,**kwargs):
        """Manage the docker's internal state (e.g. terminate / send data
        to the outside).  Default: do nothing."""
        return self.manager.manage(self,*args,**kwargs)
    def store_error(self,exc,*args,**kwargs):
        """Store an exception instance for later inspection (default: keep
        the latest one on docker.error; does not swallow the exception)."""
        return self.manager.store_error(self,exc,*args,**kwargs)
    def read_operate_flow(self,source=None,*args,**kwargs):
        """Read the operate flow.  With *source* (text or file) the default
        reader parses the imperative shell format (func arg1 arg2 -kw value
        -> result_pos; data<N> variables become data_pool index refs)."""
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
        drives the arrangement, action-layer operate flow (ExecuterDriver,
        non-blocking background worker) runs the steps."""
        return self.manager.run(self,*args,**kwargs)
    def default_start(self,run):
        """Default start: non-blocking launch of run on a background thread."""
        return self.manager.start(self,run)

"""
Explicit public exports (prevents import-star namespace pollution).
"""
__all__ = (
    "BaseDocker",
    "Docker",
    "DockerProtocol",
)
