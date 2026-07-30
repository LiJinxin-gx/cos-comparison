#It provide a base memory class and tool with a rule.

from abc import ABC

def no_done(*args,**kwargs):
    pass

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
    __slots__ = ("memory","save_func","commit_func", "refer_func","close_func")
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
    def save(self,*args,**kwarg):
        return self.save_func(*args,**kwargs)
    def commit(self,*args,**kwargs):
        return self.commit_func(*args,**kwargs)
    def refer(self,*args,**kwargs):
        return self.refer_func(*args,**kwargs)
    def refer(self,*args,**kwargs):
        return self.refer_func(*args,**kwargs)
