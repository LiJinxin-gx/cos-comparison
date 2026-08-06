#It provide tensor data container.

from abc import ABC, abstractmethod
from ... import core

class BaseTensor(ABC):
    @abstractmethod
    def __getitem__(self, index):
        pass
    @abstractmethod
    def __setitem__(self, index, value):
        pass
    @abstractmethod
    def __len__(self):
        pass

class Tensor(core.vector_map_as_tensor,BaseTensor):
    # core.vector_map_as_tensor is FIRST in the MRO, so its concrete
    # __getitem__/__setitem__/etc. provide the BaseTensor abstract interface
    # and nothing is shadowed. All parent flexible settings (vector, shape,
    # start, strides, offset, start_offset, step_offset) are passed through
    # completely via *args/**kwargs.
    def __init__(self,*args,data=None,split_start=None,split_shape=None,**kwargs):
        if data is None:
            super().__init__(*args,**kwargs)
        else:
            loaded = core.load_as_default_data(data, start=split_start, shape=split_shape)
            # explicit vector/shape come from the loaded data source; every
            # other optional parent setting in kwargs is still forwarded.
            super().__init__(*args,vector=loaded.vector,shape=loaded.shape,**kwargs)
