#It gives some tools to deal with probability.

#----------- event class -----------
class UnionEvent(frozenset): #A+B+...
    __slots__ = ("name",)
    def __init__(self,*event):
        self.name = "UnionEvent"
    def __new__(cls,*event):
        return super().__new__(event)
    def __hash__(self):
        return hash(("Union",super().__hash__()))

class IntersectionEvent(frozenset): #AB...
    __slots__ = ("name",)
    def __init__(self,*event):
        self.name = "IntersectionEvent"
    def __new__(cls,*event):
        return super().__new__(event)
    def __hash__(self):
        return hash(("Intersection",super().__hash__()))

class GlobalEvent: 
    # In face,there is not any absolute global event,the class make a relative benchmark in a context.
    def __repr__(self):
        return f"<GlobalEvent : id={id(self)} , hash={hash(self)}>"

global_event = GlobalEvent()
#It provide a Singleton to make a relative benchmark in module.

#------------ event_bind ---------------
class event_bind:
    #We should know that proibability of all event is relative.
    __slots__=("name","event","binds","id")
    def __init__(self,name="",event=None):
        self.name=name if name else  str(event)
        self.event=event
        self.binds = {} #It stores Conditional Probability , like P(A | B)
        self.id=id(self)
    def __iter__(self):
        for bind in self.binds:
            yield (self.event,bind,self.binds[bind])
    def get_bind(self,event):
        return self.binds[event]
    def bind_exist(self,event):
        if event in self.binds:
            return True
        else:
            return False
    def bind(self,event,relative_p):
        if 0<=relative_p<=1 :
            self.binds[event] = relative_p #if it has existed,overwrite it.
            return 0
        else:
            raise ValueError("effectless args")
    def unbind(self,event):
        try:
            del self.binds[event]
            return 0
        except:
            return 1

#--------- context ----------
class event_context:
    # core formal  is Bayes' formal : P(B|A) * P(A|C) = P(A|B) * P(B|C) = P(AB|C)
    __slots__ = ("name","binds")
    def __init__(self,name="",binds=None):
        self.name = name
        self.binds = binds if binds else {}
    def __iter__(self):
        for bind in self.binds:
            yield (*bind,bind[bind])
            
    def add_bind(self,binds): 
        for bind in binds:
            key,p=(bind[0],bind[1]),bind[2]
            self.binds[key] = p

    def bind_probability(self,A,B=global_event):
        """
        A : event or tuple ,B : event or tuple (default value : global_event) , return value : P(A|B)
        """
        try:
            return self.binds[(A,B)] #search directly.
        except KeyError:
            #formal base.
            # P(AB|C) = P(A|B) * P(B|C) = P(B|A) * P(A|C)  (Bayes' formal)
            # P(A+B | C) = P(A|C) + P(B|C)  - P(AB|C)
            pass
        except:
            raise
