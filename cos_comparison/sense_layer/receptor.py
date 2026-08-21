from .. import core

class Receptor:
    __slots__ = ("data",)
    def __init__(self,data):
        self.data = data
    def initialize(self,caller,*args,**kwargs):
        return caller(self.data,*args,**kwargs)
    def receptor(self,caller,args=(),kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        return caller(self.data,*args,**kwargs)

class TensorReceptor(Receptor):
    def __init__(self,data):
        super().__init__(data)
    def point(self,index):
        return core.get_item(self.data,index)
    def comparison_passive(self,output=None,**kwargs):
        return core.cos_comparison_passive(self.data,output=output,**kwargs)
    def comparison_active(self,kernel=None,output=None,**kwargs):
        return core.cos_comparison_active(self.data,kernel=kernel,output=output,**kwargs)


def data_match(data, template, start=None, end=None, step=None, algorithm=None,
               low=None, high=None, inclusive=(True, True)):
    """Match data against a template via the core active comparison and
    iterate the matching positions: yields the output positions whose match
    value lies in [low, high] (interval optional — omitted bounds match
    everything; inclusive=(lo_in, hi_in) controls endpoint membership).
    None region parameters are omitted so every backend treats them
    uniformly; algorithm=None uses the core default (resolved inside the
    backend, valid on the C extension too)."""
    kwargs = {}
    if start is not None:
        kwargs["start"] = start
    if end is not None:
        kwargs["end"] = end
    if step is not None:
        kwargs["step"] = step
    if algorithm is not None:
        kwargs["algorithm"] = algorithm
    out = core.cos_comparison_active(data, kernel=template, **kwargs)
    if low is None and high is None:
        return core.data_filter(out, lambda value: True)
    return core.threshold_filter(out, low, high, inclusive=inclusive)

"""
Explicit public exports (prevents import-star namespace pollution).
"""
__all__ = (
    "Receptor",
    "TensorReceptor",
    "data_match",
)
