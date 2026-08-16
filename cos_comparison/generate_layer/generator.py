"""
It provides some tools to generate data by matching data struct.
"""

from .. import core

class Generator:
    __slots__ = ("data",)
    def __init__(self,data):
        self.data = data
    def fix(self,call,args=(),kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        return call(self.data,*args,**kwargs)

class TensorGenerator(Generator):
    def __init__(self,data):
        super().__init__(data)
    def generate(self,func,args=(),kwargs=None):
        """Uniform delegation entry: func(self.data, *args, **kwargs).
        Generation and modification logic lives in external functions which
        are executed through this entry (duck protocol)."""
        kwargs = {} if kwargs is None else kwargs
        return func(self.data,*args,**kwargs)
    def set_point(self,index,value):
        """Write one element via the core set_item protocol
        (__set_item__ authoritative, plain nested assignment fallback)."""
        return core.set_item(self.data,index,value)

#generate functions.
def copy_region(target, source, *, shape=None, source_start=None, source_step=None,
                target_start=None, target_step=None):
    """Direct region fill: copy a sub-region of source into target's
    corresponding positions (core.load_data wrapper; each side has its own
    start/step, out-of-bounds silently clipped). Returns elements copied.
    Target comes first so generate() fills the owned data naturally:
    generate(copy_region, args=(template,))."""
    return core.load_data(source, target, shape=shape, source_start=source_start,
                          source_step=source_step, target_start=target_start,
                          target_step=target_step)
        
