"""
module 'memory' aim to provide effective memory vector to shield lower-level implementation differences.
It can define the use of various storage media for general normalized memory functionality through flexible rule definitions.
"""

from .basememory import *
from .database_memory import *
from .inner_memory import *

class MappingBase:
    """Mapping-protocol base: keys() via an injected delegate (keys_func
    slot); __iter__ / items derive from it."""
    __slots__ = ()
    def __init__(self, keys_func=None):
        self.keys_func = keys_func if keys_func is not None else self.default_keys
    def keys(self):
        return self.keys_func(self)
    def __iter__(self):
        return iter(self.keys())
    def items(self):
        for key in self.keys():
            yield (key, self[key])


class MemoryWrap(MappingBase):
    """
    It wrap kinds of memory to provide Unified Operation Interfaces.
    It also support hierarchical tagging of memory carrier information such as level,
    also supporting short-term and long-term memory.
    """
    __slots__ = ("memory_body","memory_type","name","level","id","keys_func")
    def __init__(self,memory_body=None,memory_type=None,args=(),kwargs=None,name="",level=0,
                 keys_func=None):
        super().__init__(keys_func)
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
    @staticmethod
    def default_keys(self):
        """Default keys (Python mapping protocol): carrier keys() when
        supported; index keys for sized sequences; else empty."""
        source = getattr(self.memory_body, "keys", None)
        if source is not None:
            return source()
        try:
            return range(len(self.memory_body))
        except TypeError:
            return ()
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
    def __getitem__(self,key):
        return self.get(key)

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

class MemoryWrapMap(MemoryWrapPool, MappingBase):
    __slots__ = ("keys_func",)
    def __init__(self,map_pool=None,name="",level=0,keys_func=None):
        MappingBase.__init__(self, keys_func)
        map_pool = {} if map_pool is None else map_pool
        MemoryWrapPool.__init__(self, pool=map_pool, name=name, level=level)
    @staticmethod
    def default_keys(self):
        """Default keys (Python mapping protocol): carrier keys() when
        supported; index keys for sized sequences; else empty."""
        source = getattr(self.pool, "keys", None)
        if source is not None:
            return source()
        try:
            return range(len(self.pool))
        except TypeError:
            return ()
    def __contains__(self,element):
        pool = self.pool
        return any( (pool[k]==element for k in pool) )
    def add(self,index,memorywrap):
        super().set(index,memorywrap)
    def get_by_name(self,name):
        return self.pool[name]
    def __getitem__(self,key):
        return self.pool[key]



