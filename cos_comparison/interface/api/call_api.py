#It allows to use inner or external modules.

from abc import ABC

import importlib
import ctypes

class BaseCallContainer(ABC):
    __slots__ = ("container",)
    def __init__(self,obj):
        self.container = obj
    def call(self,name,args=(),kwargs=None,init_func=None):
        # It does not use "__getattr__",because it will cause bugs while get inner attribute such as "container".
        kwargs= kwargs if kwargs else {}
        return self.get_call(name,init_func=init_func)(*args,**kwargs)
    def get_call(self,name,init_func=None):
        if init_func:
            return init_func(getattr(self.container,name))
        else:
            return getattr(self.container,name)
    def get(self,name):
        return getattr(self.container,name)
    def set(self,name,value):
        return setattr(self.container,name,value)

class Module_CallContain(BaseCallContainer):
    def __init__(self,module_name,*args,package=None,**kwarg):
        super().__init__(importlib.import_module(module_name,package=package))

class C_CallContainer(BaseCallContainer):
    def __init__(self, library_path, loader=ctypes.CDLL, *args, **kwargs):
        lib = loader(library_path, *args, **kwargs)
        super().__init__(lib)
    def get_call(self,name,argstypes=(),restype=None,init_func=None):
        caller = super().get_call(name,init_func=init_func)
        caller.argtypes = argstypes
        caller.restype = restype
        return caller
    def call(self,name,args=()):
        return super().call(name,args=args)

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
