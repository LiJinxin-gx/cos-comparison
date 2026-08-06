"""
It provide unified data carrier interfaces to carry data struct.
the aim to the module is to decouple data carrying from data operations.
"""

from . import tensor

class DataWrap:
    __slots__ = ("data","id","dtype")
    def __init__(self,data_body=None):
        self.data = data_body
        self.id = None
        self.dtype = None
    def __getitem__(self,index):
        return self.data.__getitem__(index)
    def __setitem__(self,index,value):
        self.data[index] = value
    def __get_item__(self,*index):
        return self.data.__get_item__(*index)
    def __set_item__(self,index,value):
        self[index] = value
    def process(self,caller,args=(),kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        return caller(self.data,*args,**kwargs)
    def call(self,name,args=(),kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        return getattr(self.data,name)(*args,**kwargs)
    def getattr(self,name):
        return getattr(self.data,name)
    def setattr(self,name,value):
        return setattr(self.data,name,value)
