#the file aim to provide a data base to carry data generated.

from .. import core

class DataBase(core.vector_map_as_tensor):
    def __init__(self,*arg,data=None,start=None,shape=None,**kwarg):
        if data is None:
            super().__init__(*arg,**kwarg)
        else:
            origin = core.load_as_default_data(data,start,shape)
            super().__init__(origin.vector,origin.tensor_size)
    
    def filter_s(self,func):
        for p in range(self.start,self.end):
            temp = self.vector[p]
            if func(temp):
                yield temp
    def filter(self,func):
        return tuple(self.filter_s())
    
    def point(self,index,vaule):
        self.__set_item__(index,value)
        
    def comparison_passive(self,*arg,**kwarg):
        core.cos_comparison_passive(self,*arg,**kwarg)
    def comparison_active(self,*arg,**kwarg):
        core.cos_comparison_active(self,*arg,**kwarg)
