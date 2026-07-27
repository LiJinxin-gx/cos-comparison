#It is a module to achieve reflex.

import asyncio
import threading

class Empty:
    pass

DEVNULL=0 #default vaule to discard some value

class Trigger:
    __slots__ = ("trigger","callback","stack","res_index","args_index")
    def __init__(self,
                 trigger,callback,
                 stack=None,
                 a_res_index : int=DEVNULL , b_res_index : int=DEVNULL ,
                 args_index=() , kwargs_index=() ):
        self.trigger = trigger #a function trigger event
        self.callback = callback #a function execute while event is triggered
        self.stack = stack if stack is not None else [None]* ( max(max(arg_index) if args_index else 0 , max((i[1] for i in kwargs_index)) if kwargs_index else 0 , a_res_index , b_res_index) + 1 )
        self.a_res_index = a_res_index #eg. 1
        self.b_res_index = b_res_index #eg. 1
        self.args_index = args_index #eg. (1,2,3)
        self.kwargs_index = kwargs_index #eg. (("a",1),("b",2),("c",3))
    def exec(self):
        try:
            s = self.stack
            self.stack[self.b_res_index] = self.callback(*(s[i] for i in self.args_index),**{a:s[b] for a,b in selfkwargs_index})
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

async def _monitor(obj,target,trigger:str,callback,sleep=0.01,times=-1):
    if type(times) != int:
        raise TypeError("arg 'times' must be a int.")
    while 1:
        if target(getattr(obj,trigger)):
            if times !=0:
                callback()
                times -= 1
            else:
                break
        await asyncio.sleep(sleep)

class Monitor:
    __slots__ = ("target","callbacks")
    def __init__(self,callbacks=None,target=None):
        self.target = Empty() if target is  None else target
        self.callbacks = callbacks if callbacks else []
    def add_target(self,target,trigger,callback,times=-1):
        self.callbacks.append((target,trigger,callback,times))
    async def _run(self):
        await asyncio.gather(*[_monitor(self.target,target,trigger,callback,t) for target,trigger,callback,t in self.callbacks])
    def run(self):
        threading.Thread(target=asyncio.run(self._run),daemon=True).start()
