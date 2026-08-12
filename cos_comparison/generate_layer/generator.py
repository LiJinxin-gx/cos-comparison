"""
It provides some tools to generate data by matching data struct.
"""

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
    def set_point(self,index,data):
        self.data[index] = data 
        
