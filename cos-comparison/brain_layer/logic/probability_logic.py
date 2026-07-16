#It gives some tools to deal with probability.

class event_bind:
    __slots__=("name","event","binds","id")
    def __init__(self,name="",event=None):
        self.name=str(event)
        self.event=event
        self.binds=[]  #It stores Conditional Probability , like P(A | B)
    def __iter__(self):
        for bind in self.binds:
            yield (self.event,*bind)
    def bind(self,event,relative_p):
        if 0<=relative_p<=1 :
            if True not  in set((event==bind[0] for bind in self.binds)):
                self.binds.append((event,relative_p))
                return 0
            else:
                return 1
        else:
            raise ValueError("effectless args")
    def unbind(self,event):
        try:
            n=0
            for bind in self.binds:
                if bind[0]==event:
                    del self.binds[n]
                    return 0
                n += 1
            return 1
        except:
            return 2

class UnionEvent(event_bind): #P(A+B)
    def __init__(self,name,*events):
        super().__init__(name=name,event=events)

class IntersectionEvent(event_bind): #P(AB)
    def __init__(self,name,*events):
        super().__init__(name=name,event=events)


class event_context:
    __slots__ = ("name","binds")
    def __init__(self,name="",binds=None):
        self.name = name
        self.binds = binds if binds else []
    def __iter__(self):
        for bind in self.binds:
            yield bind
            
    def add_bind(self,binds): 
        for bind in binds:
            self.binds.append(bind)
