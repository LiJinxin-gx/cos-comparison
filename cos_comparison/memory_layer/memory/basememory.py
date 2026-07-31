#It provide a base memory class and tool with a rule.

from abc import ABC
from enum import Flag,auto

def no_done(*args,**kwargs):
    pass

class Status(Flag):
    READ = auto()
    WRITE = auto()
    EXECUTE = auto()

ST_READ = Status.READ
ST_WRITE = Status.WRITE
ST_EXECUTE = Status.EXECUTE

class BaseMemory(ABC):
    def initialize(self):
        pass
    def save(self):
        pass
    def commit(self):
        pass
    def refer(self):
        pass
    def close(self):
        pass

class Memory(BaseMemory):
    __slots__ = ("memory","init_func","save_func","commit_func", "refer_func","close_func")
    def __init__(self,memory,
                 init_func=None,
                 save_func=None,
                 commit_func=None,
                 refer_func=None,
                 close_func=None):
        self.memory =  memory
        self.init_func = init_func if init_func else no_done
        self.save_func = save_func if save_func else no_done
        self.commit_func = commit_func if commit_func else no_done
        self.refer_func = refer_func if refer_func  else no_done
        self.close_func = close_func if close_func  else no_done
    def initialize(self,*args,**kwargs):
        return self.init_func(self,*args,**kwargs)
    def save(self,*args,**kwargs):
        return self.save_func(self,*args,**kwargs)
    def commit(self,*args,**kwargs):
        return self.commit_func(self,*args,**kwargs)
    def refer(self,*args,**kwargs):
        return self.refer_func(self,*args,**kwargs)
    def close(self,*args,**kwargs):
        return self.close_func(self,*args,**kwargs)
    def process(self,caller=None,args=(),kwargs=None):
        return caller(self.memory,*args,**kwargs)
    def call(self,name,args=(),kwargs=None):
        return getattr(self,name)(*args,**kwargs)
