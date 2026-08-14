"""
It provides docker to run.
"""

from abc import ABC, abstractmethod
from ..core import no_done

class BaseDocker(ABC):
    @abstractmethod
    def run(self):
        pass

class Docker(BaseDocker):
    __slots__ = ("data_pool","operate_pool","controller","starter","interface_pool","extension_pool")
    def __init__(self,
                 data_pool=None,operate_pool=None,controller=None,
                 starter=None,maintainer=None,
                 interface_pool=None,extension_pool=None):
        self.data_pool = [] if data_pool is None else data_pool #Data flow
        self.operate_pool = [] if operate_pool is None else operate_pool #Operation flow
        self.controller = no_done if controller is None else controller #Control flow
        self.starter = no_done if starter is None else starter
        self.maintainer = no_done if maintainer is None else maintainer
        self.interface_pool = [] if interface_pool is None else interface_pool
        #To expose the public interface of the instance to external calls
        self.extension_pool = [] if extension_pool is None else extension_pool
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
