#a executer tool set.

class ExecuterDriver:
    __slots__ = ("call_list",)
    def __init__(self,caller_list=()):
        self.call_list = call_list
    def call(index,args=(),kwargs=None):
        kwargs = {} if kwargs else kwargs
        self.call_list[index](*args,**kwargs)
        

