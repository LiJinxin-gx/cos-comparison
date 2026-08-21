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
        if data is not None:
            loaded = core.load_as_default_data(data, start=split_start, shape=split_shape)
            shape = kwargs.get("shape")
            if shape is not None and tuple(shape) != loaded.shape:
                # an explicit `shape` reshapes the loaded flat data instead
                # of being silently dropped; length mismatch is a clear error
                size = int(core.multiple_chain(tuple(shape)))
                if size != len(loaded.vector):
                    raise ValueError(
                        "data size %d does not match shape %s"
                        % (len(loaded.vector), tuple(shape)))
                loaded = core.vector_map_as_tensor(
                    vector=list(loaded.vector), shape=tuple(shape))
            # explicit vector/shape come from the loaded data source; every
            # other optional parent setting in kwargs is still forwarded.
            kwargs["vector"] = loaded.vector
            kwargs["shape"] = loaded.shape
        # core's `start` is a scalar flat offset; upper layers express
        # per-dimension starts as tuples, which map to core's start_offset
        start = kwargs.pop("start", None)
        if start is not None and not isinstance(start, int):
            kwargs["start_offset"] = tuple(start)
        elif start is not None:
            kwargs["start"] = start
        super().__init__(*args,**kwargs)

"""
Explicit public exports (prevents import-star namespace pollution).
"""
__all__ = (
    "BaseTensor",
    "Tensor",
)
