# It provides symbol logic ability.

#-------- import --------
from enum import Flag,auto

#-------- constant supports -------
class Logic(Flag):
    TRUE = auto()
    SURE = auto()
    
Logic_true = Logic.TRUE
Logic_sure = Logic.SURE

sure_true = Logic.TRUE | Logic.SURE

class No_limit:
    def __contains__(self,other):
        return True

#------- logical error -------
class LogicError(Exception):
    pass

#-------- logical supports ----------
class Atomic_proposition:
    __slots__=("subject","verb","objects","adv","limit","status")
    
    def __init__(self,subject=None,verb=None,objects=None,adv=sure_true,limit=None,status=sure_true):
        self.subject=subject
        self.verb=verb
        self.objects=objects
        self.adv=adv
        self.limit=limit if limit is None else No_limit()
        self.status=status
    def __bool__(self)->bool:
        return bool(self.status & Logic_true)
    def __iter__(self):
        for name in self.__slots__:
            yield getattr(self,name)
            
    def __getitem__(self,key):
        return getattr(self,key)
    def __setitem__(self,key,value):
        setattr(self,key,value)
    def __delitem__(self,key):
        setattr(self,key,None)

class Logic_bind:
    __slots__=("reason","result","limit","status")

    def __init__(self,reason,result,limit=None,status=sure_true):
        self.reason=reason
        self.result=result
        self.limit=limit if limit is None else No_limit()
        self.status=status
    def __bool__(self)->bool:
        return bool(self.status & Logic_true)
    def __iter__(self):
        for name in self.__slots__:
            yield getattr(self,name)
            
    def __getitem__(self,key):
        return getattr(self,key)
    def __setitem__(self,key,value):
        setattr(self,key,value)
    def __delitem__(self,key):
        setattr(self,key,None)

class Logic_context:
    __slots__ = ("name","binds")
    def __init__(self,name="",binds=None):
        self.name = name
        self.binds = binds if binds else []
    def add(self,logic_bind):
        logic=tuple(logic_bind)
        if any(logic[0:2] == sub[0:2] for sub in self.binds):
            raise ValueError("it has occupied in the context")
        binds.append(logic)
    def pop(self,logic_bind):
        logic_bind = tuple(logic_bind)
        n = 0
        for bind in self.binds:
            if binds[0:2] == tuple(logic_bind):
                return self.binds.pop(n)
            n +=1
        return None
        
        
