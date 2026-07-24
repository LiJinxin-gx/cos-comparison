#It allows to use inner or external modules.

from abc import ABC

import importlib
import ctypes

class BaseCallContainer(ABC):
    __slots__ = ("container",)
    def __init__(self,obj):
        self.container = obj
    def call(self,name,args=(),kwargs=None):
        # It does not use "__getattr__",because it will cause bugs while get inner attribute like "container".
        kwargs= kwargs if kwargs else {}
        return getattr(self.container,name)(*args,**kwargs)
    def get(self,name):
        return getattr(self.container,name)

class Module_CallContain(BaseCallContainer):
    def __init__(self,module_name,*args,package=None,**kwarg):
        super().__init__(importlib.import_module(module_name,package=package))

class C_CallContainer(BaseCallContainer):
    def __init__(self, library_path, loader=ctypes.CDLL, *args, **kwargs):
        lib = loader(library_path, *args, **kwargs)
        super().__init__(lib)

class CDLL_CallContainer(C_CallContainer):
    def __init__(self, library_path, *args, **kwargs):
        super().__init__(library_path, ctypes.CDLL, *args, **kwargs)

class WinDLL_CallContainer(C_CallContainer):
    def __init__(self, library_path, *args, **kwargs):
        super().__init__(library_path, ctypes.WinDLL, *args, **kwargs)

class CallDict:
    __slots__ = ("dict",)
    def __init__(self,init_dict=None):
        self.dict = init_dict if init_dict else {}
    def add(self,tag,func):
        self.dict[tag] = func
    def call(self,tag,args=(),kwargs=None):
        #it support hot path acceleration.
        #eg.
        #CallDict.call(4)
        kwargs = kwargs if kwargs else {}
        return self.dict[tag](*args,**kwargs)
