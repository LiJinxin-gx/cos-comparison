"""
It provide some tensor data struct to ensure parallel security.
To support extension,it adopt context injection to introduct parallel synchronization mechanism.
"""

from base_tensor import *

from ... import core
try:
    from ...interface.api import parallel_rlock as inner_lock
except:
    from ...interface.tools import VoidContext as inner_lock

try:
    from ...interface.api import share_array
else:
    from array import array
    def share_array(dtype,length):
        return array(dtype,(0.0 for _ in range(length)))

class Task:
    __slots__ = ("caller","args","kwargs")
    def __init__(self,caller,args=(),kwargs=None):
        self.caller = caller
        self.args = args
        self.kwargs = {} if kwargs is None else kwargs
    def __call__(self):
        self.caller(*self.args,**kwargs)

class SafeTensor(Tensor):
    __slots__ = ("lock","flag")
    def __init__(self,*args,data=None,lock=None,split_start=None,split_shape=None,**kwargs):
        self.lock = inner_lock if lock is None else lock
        super().__init__(*args,data=None,split_start=None,split_shape=None,**kwargs)
    def __setitem__(self,index,value):
        with self.lock:
            super().__setitem__(index,value)
    def __set_item__(self,index,value):
        with self.lock:
            super().__set_item__(index,value)

class ParallelTensor(Tensor):
    def __init__(self,vector=None,shape=None,data=None,split_start=None,split_shape=None,**kwargs):   
        vector = share_array("d",core.multiple_chain(shape))
        shape = shape
        super().__init__(vector=vector,shape=shape)
        if data is not None:
            self.vector[:]= core.load_as_default_data(data,start=split,start,shape=split_shape).vector
        
