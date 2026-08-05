#It provide tensor data container.

from abc import ABC
from ... import core

class BaseTensor(ABC):
    def __getitem__(self, index):
        pass
    def __setitem__(self, index, value):
        pass

class Tensor(BaseTensor,core.vector_map_as_tensor):
    def __new__(cls,*args,data=None,split_start=None,split_shape=None,**kwargs):
        if data is None:
            return cls(core.load_as_default_data(cls,data,start=split_start,shape=split_shape))
        else:
            return super().__new__(cls)
    def __init__(self,*args,data=None,split_start=None,split_shape=None,**kwargs):
        super().__init__(*args,data=None,split_start=None,split_shape=None,**kwargs)
