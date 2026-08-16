#a executer tool set.

class ExecuterDriver:
    __slots__ = ("call_list",)
    def __init__(self,caller_list=()):
        self.call_list = caller_list
    def call(self,index,args=(),kwargs=None):
        kwargs = kwargs if kwargs is not None else {}
        return self.call_list[index](*args,**kwargs)
        

