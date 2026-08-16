"""
Symbolic logic system for cognitive reasoning.
Implements three-valued logic, atomic propositions, variables and logical inference primitives.
"""
# It provides symbol logic ability.

#-------- import --------
from enum import Flag,auto
from ...core import no_done
from ...interface.tools.math_tool.topology import DirectedGraph, shortest_path_between

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

#------- type supports -------
class Variable:
    __slots__ = ("name","value")
    def __init__(self,name,value=None):
        self.name = name
        self.value = value
    def __eq__(self,other):
        if type(self) == type(other):
            return self.value==other.value
        else:
            return False
    def __hash__(self):
        return hash(self.value)

#------- logical error -------
class UnsupportedError(Exception):
    pass

class LogicError(Exception):
    pass

#-------- logical supports ----------
class Atomic_proposition:
    __slots__=("logic_args","arg_names","status") 
    def __init__(self,*logic_args,arg_names=None,status=sure_true):
        #a=Atomic_proposition(...)
        #If arg_names is None,eg: a_list = *a
        #else,eg: a_dict = **a
        self.logic_args = list(logic_args)
        self.arg_names = arg_names
        self.status = status
    def __bool__(self)->bool:
        return bool(self.status & Logic_true)
    def __iter__(self):
        for logic_arg in self.logic_args:
            yield logic_arg
    def __len__(self):
        return len(self.logic_args)
    def keys(self):
        return tuple(self.arg_names)
    def __getitem__(self,key):
        if self.arg_names is None:
            raise UnsupportedError("It does not support the operation because itdid not use 'arg_names'.")
        else:
            if key in self.arg_names:
                for i,name in enumerate(self.arg_names):
                    if name==key:
                        return self.logic_args[i]
            else:
                raise ValueError(f"It does not have the arg '{key}'.")
    def __setitem__(self,key,value):
        if self.arg_names is None:
            raise UnsupportedError("It does not support the operation because itdid not use 'arg_names'.")
        else:
            if key in self.arg_names:
                for i,name in enumerate(self.arg_names):
                    if name==key:
                        self.logic_args[i] = value
            else:
                raise ValueError(f"It does not have the arg '{key}'.")

class Logic_bind:
    __slots__=("reason","result","limit","status")
    def __init__(self,reason,result,limit=None,status=sure_true):
        self.reason=reason
        self.result=result
        self.limit=limit if limit is not None else No_limit()
        self.status=status
    def __bool__(self)->bool:
        return bool(self.status & Logic_true)
    def __iter__(self):
        for logic in self.__slots__:
            yield getattr(self,logic)
    def keys(self):
        return self.__slots__
    def __len__(self):
        return len(self.__slots__)
    def __getitem__(self,key):
        return getattr(self,key)
    def __setitem__(self,key,value):
        return setattr(self,key,value)

class Logic_context:
    __slots__ = ("name","binds","extension","init_func","add_func","pop_func","judge_func")
    def __init__(self,name="",binds=None,
                 init_func=None,
                 add_func=None,
                 pop_func=None,
                 judge_func=None):
        self.name = name
        self.extension = None #It provides extension to use by callback functions.
        self.binds = binds if binds is not None else []
        self.init_func = init_func if init_func else no_done
        self.add_func = add_func if add_func else no_done
        self.pop_func = pop_func if pop_func else no_done
        self.judge_func = judge_func if judge_func else default_judge_func
    def initialize(self,*args,**kwargs):
        return self.init_func(self.binds,*args,**kwargs)
    def add(self,logic_bind,**kwargs):
        return self.add_func(self.binds,logic_bind,**kwargs)
    def pop(self,logic_bind,**kwargs):
        return self.pop_func(self.binds,logic_bind,**kwargs)
    def logic_judge(self,a,b,**kwargs):
        return self.judge_func(self.binds,a,b,**kwargs)


def _as_binds(context):
    """Duck protocol: a context exposing .binds, or a bare binds container."""
    return getattr(context, "binds", context)


def default_judge_func(context, a, b, return_path=False, graph_factory=DirectedGraph):
    """Default rule-chain judge: is b derivable from a?
    binds elements expose reason/result (arc reason -> result); returns bool,
    or the rule list [bind1, ...] proving a -> b when return_path=True
    ([] if a == b, None when unreachable)."""
    binds = _as_binds(context)
    g = graph_factory()
    for bind in binds:
        g.add_edge(getattr(bind, "reason"), getattr(bind, "result"))
    if a == b:
        return [] if return_path else True
    path = shortest_path_between(g, a, b)
    if path is None:
        return None if return_path else False
    if not return_path:
        return True
    rules = []
    for i in range(1, len(path)):
        for bind in binds:
            if getattr(bind, "reason") == path[i - 1] and getattr(bind, "result") == path[i]:
                rules.append(bind)
                break
    return rules



