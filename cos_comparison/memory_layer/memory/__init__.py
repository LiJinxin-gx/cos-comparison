"""
module 'memory' aim to provide effective memory vector to shield lower-level implementation differences.
It can define the use of various storage media for general normalized memory functionality through flexible rule definitions.
"""

from .basememory import *
from .database_memory import *
from .inner_memory import *

class MemoryWrap:
    """
    It wrap kinds of memory to provide Unified Operation Interfaces.
    It also support hierarchical tagging of memory carrier information such as level,
    also supporting short-term and long-term memory.
    """
    __slots__ = ("memory_body","memory_type","name","level","id")
    def __init__(self,memory_body=None,memory_type=None,args=(),kwargs=None,name="",level=0):
        self.name = name
        self.level = level
        self.id = None
        if memory_body is None:
            kwargs={} if kwargs is None else kwargs
            self.memory_body = memory_type(*args,**kwargs)
            self.memory_type = memory_type
        else:
            self.memory_body = memory_body
            self.memory_type = type(memory_body)
    def process(self,caller,args=(),kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        return caller(self.memory_body,*args,**kwargs)
    def call(self,name,args=(),kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        return getattr(self.memory_body,name)(*args,**kwargs)
    def get(self,key):
        return self.memory_body[key]
    def set(self,key,value):
        self.memory_body[key] = value
    def get_body_attr(self,name):
        return getattr(self.memory_body,name)
    def set_body_attr(self,name,value):
        return setattr(self.memory_body,name,value)

class MemoryWrapPool:
    __slots__ = ("pool","name","level","id")
    def __init__(self,pool=None,name="",level=0):
        self.name = name
        self.level =level
        self.id = None
        self.pool = [] if pool is None else pool
    def __len__(self):
        return len(self.pool)
    def __contains__(self,element):
        return element in self.pool
    def add(self,memorywrap):
        self.pool.append(memorywrap)
    def set(self,index,memorywrap):
        self.pool[index] = memorywrap
    def operate(self,index,call_name,args=(),kwargs=None):
        return getattr(self.pool[index],call_name)(*args,**kwargs)
    def get_memory_attr(self,index,name):
        return getattr(self.pool[index],name)

class MemoryWrapMap(MemoryWrapPool):
    def __init__(self,map_pool=None,name="",level=0):
        map_pool = {} if map_pool is None else map_pool
        super().__init__(pool=map_pool,name=name,level=level)
    def __contains__(self,element):
        pool = self.pool
        return any( (pool[k]==element for k in pool) )
    def add(self,index,memorywrap):
        super().set(index,memorywrap)
    def get_by_name(self,name):
        return self.pool[name]
