"""
It provides basic way to achieve mapping function.
"""

from abc import ABC, abstractmethod

from ...core import no_done

class BaseMap(ABC):
    #Note : It performs the functional operation of mapping rather than data carriage.
    @abstractmethod
    def __getitem__(self,input_obj):
        pass
    @abstractmethod
    def __setitem__(self,old_obj,input_obj):
        pass
    @abstractmethod
    def __contains__(self,input_obj):
        pass

def default_map_func(map_obj, input_obj):
    """dict protocol: map_obj[input_obj]."""
    return map_obj[input_obj]


def default_set_func(map_obj, old_obj, input_obj):
    """dict protocol: map_obj[old_obj] = input_obj."""
    map_obj[old_obj] = input_obj


def default_contain_judge_func(map_obj, input_obj):
    """dict protocol: input_obj in map_obj."""
    return input_obj in map_obj


class Map(BaseMap):
    __slots__ = ("map_obj","map_func","set_func","contain_judge_func")
    def __init__(self,map_obj=None,map_func=None,set_func=None,contain_judge_func=None):
        self.map_obj = map_obj
        self.map_func = map_func if map_func is not None else default_map_func
        self.set_func = set_func if set_func is not None else default_set_func
        self.contain_judge_func = contain_judge_func if contain_judge_func is not None else default_contain_judge_func
    def __getitem__(self,input_obj):
        return self.map_func(self.map_obj,input_obj)
    def __setitem__(self,old_obj,input_obj):
        return self.set_func(self.map_obj,old_obj,input_obj)
    def __contains__(self,input_obj):
        return self.contain_judge_func(self.map_obj,input_obj)

class FuncWrap:
    __slots__ = ("func",)
    def __init__(self,func):
        self.func = func
    def __call__(self,first_arg,*args,**kwargs):
        return self.func(*args,**kwargs)
