"""
It provide some tensor data struct to ensure parallel security.
To support extension,it adopt context injection to introduct parallel synchronization mechanism.
"""

from .base_tensor import Tensor

from ... import core
from ...interface.api import parallel_rlock as inner_lock
from ...interface.api import share_array, load_array

class Task:
    __slots__ = ("caller","args","kwargs")
    def __init__(self,caller,args=(),kwargs=None):
        self.caller = caller
        self.args = args
        self.kwargs = {} if kwargs is None else kwargs
    def __call__(self):
        self.caller(*self.args,**self.kwargs)

class SafeTensor(Tensor):
    """
    Write-atomic tensor for concurrent use.

    The lock protects WRITE atomicity only (each write is all-or-nothing) to
    keep data integrity. Reads are lock-free so multiple threads may read
    simultaneously without contention; the caller is responsible for
    synchronizing reads against concurrent writers.
    """
    __slots__ = ("lock","flag")
    def __init__(self,*args,data=None,lock=None,split_start=None,split_shape=None,**kwargs):
        self.lock = inner_lock if lock is None else lock
        # forward everything to the parent, including data + all parent settings
        super().__init__(*args,data=data,split_start=split_start,split_shape=split_shape,**kwargs)
    # reads inherit from Tensor: lock-free, no contention
    def __setitem__(self,index,value):
        with self.lock:
            super().__setitem__(index,value)
    def __set_item__(self,index,value):
        with self.lock:
            super().__set_item__(index,value)

class ParallelTensor(Tensor):
    """
    A tensor backed by a shared array('d') buffer so elements can be read and
    written from different threads or processes.

    Branch dispatch (data decides the loading path):
      - data is not None : load ONCE (shape + contents), fill the supplied
        `vector` (length-checked) or allocate a shared buffer from the loaded
        shape; an explicit `shape`, when given, must match the data.
      - data is None     : `shape` is required; allocate a shared buffer when
        `vector` is not supplied, otherwise wrap the supplied buffer.
    All validation happens before any allocation or construction.
    """
    def __init__(self,*args,vector=None,shape=None,data=None,split_start=None,split_shape=None,**kwargs):
        if data is not None:
            loaded = core.load_as_default_data(data, start=split_start, shape=split_shape)
            if shape is not None and loaded.shape != tuple(shape):
                raise ValueError(f"data shape {loaded.shape} does not match shape {tuple(shape)}")
            shape = loaded.shape
            if vector is None:
                vector = share_array("d", int(core.multiple_chain(shape)))
            elif len(vector) != int(core.multiple_chain(shape)):
                raise ValueError(f"vector length {len(vector)} does not match shape {tuple(shape)}")
            vector[:] = load_array("d",loaded.vector)
        else:
            if shape is None:
                raise TypeError("shape is required when neither data nor shape is given")
            if vector is None:
                vector = share_array("d", int(core.multiple_chain(shape)))
        # vector/shape are concrete here; every other parent flexible setting
        # (start, strides, offset, ...) in kwargs is still forwarded completely.
        super().__init__(vector=vector,shape=shape,**kwargs)
        
