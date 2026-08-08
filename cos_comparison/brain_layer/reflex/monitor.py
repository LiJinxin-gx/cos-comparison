#It is a module to achieve monitors.

import asyncio
import threading

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
        threading.Thread(target=lambda: asyncio.run(self._run()),daemon=True).start()
