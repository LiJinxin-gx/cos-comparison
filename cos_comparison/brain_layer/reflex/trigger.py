#It is a module to achieve triggers.

class Empty:
    pass

DEVNULL=0 #default vaule to discard some value

class Trigger:
    __slots__ = ("trigger","callback","stack","a_res_index","b_res_index","args_index","kwargs_index")
    def __init__(self,
                 trigger,callback,
                 stack=None,
                 a_res_index : int=DEVNULL , b_res_index : int=DEVNULL ,
                 args_index=() , kwargs_index=() ):
        self.trigger = trigger #a function trigger event
        self.callback = callback #a function execute while event is triggered
        self.stack = stack if stack is not None else [None]* ( max(max(args_index) if args_index else 0 , max((i[1] for i in kwargs_index)) if kwargs_index else 0 , a_res_index , b_res_index) + 1 )
        self.a_res_index = a_res_index #eg. 1
        self.b_res_index = b_res_index #eg. 1
        self.args_index = args_index #eg. (1,2,3)
        self.kwargs_index = kwargs_index #eg. (("a",1),("b",2),("c",3))
    def exec(self):
        try:
            s = self.stack
            self.stack[self.b_res_index] = self.callback(*(s[i] for i in self.args_index),**{a:s[b] for a,b in self.kwargs_index})
            return 0
        except:
            return 1
    def target(self,*args,**kwargs):
        try:
            self.stack[self.a_res_index] = self.trigger(*args,**kwargs)
            if self.exec():
                return 1
            return 0
        except:
            return 2
    def stack_operate(self,caller,*args,**kwargs):
        return caller(self,self.stack,*args,**kwargs)


"""
Explicit public exports (prevents import-star namespace pollution).
"""
__all__ = (
    "Empty",
    "DEVNULL",
    "Trigger",
)
