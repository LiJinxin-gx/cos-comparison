from .. import core

class Receptor:
    __slots__ = ("data",)
    def __init__(self,data):
        self.data = data
    def initialize(self,caller,*args,**kwargs):
        return caller(self.data,*args,**kwargs)
    def point(self,index):
        return self.data.__getitem__(*index)

class TensorReceptor(Receptor):
    def __init__(self,data):
        try:
            data = memoryview(data)
        except:
            pass
        super().__init__(data)
    def comparison_passive(self,output=None,**kwargs):
        core.cos_comparison_passive(self.data,output=output,**kwargs)
    def comparison_active(self,output=None,**kwargs):
        core.cos_comparison_active(self.data,output=output,**kwargs)
