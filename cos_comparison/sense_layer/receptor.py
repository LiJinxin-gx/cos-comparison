from .. import core

class Receptor:
    __slots__ = ("data",)
    def __init__(self,data):
        self.data = data
    def initialize(self,caller,*args,**kwargs):
        return caller(self.data,*args,**kwargs)
    def receptor(self,caller,args=(),kwargs=None):
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
