from .basememory import *

def create_map(types):
    return types()

class Transaction:
    __slots__ = ("key","value","nesting","create")
    def __init__(self,key,value,nesting=True,create=True):
        self.key , self.value = key , value
        self.nesting , self.create = nesting , create
    def __iter__(self):
        for attr in self.__class__.__slots__:
            yield getattr(self,attr)
    def __hash__(self):
        return hash(tuple(self))

class MapMemory(Memory):
    __slots__ = ("cache","close_commit","closer")
    def __init__(self,map_obj=None,close_commit=False,closer=None):
        self.cache = set()
        self.close_commit = close_commit
        self.closer = closer
        super().__init__(memory=map_obj)
    def refer(self,key,nesting=False):
        if nesting:
            obj = self.memory
            for k in key:
                obj = obj[k]
            return obj
        else:
            return self.memory[key]
    def save(self,key,value,nesting=False,create=True):
        self.cache.add( Transaction(key,value,nesting=nesting,create=create) )
    def real_save(self,key,value,nesting=False,create=True):
        """
        the save function to save data in cache to support atomic event,you should commit it.
        """
        if nesting:
            obj = self.memory
            *keys,end = key
            for k in keys:
                try:
                    obj = obj[k]
                except:
                    if create:
                        obj[k] = create_map(type(self.memory))
                        obj = obj[k]
                    else:
                        raise
                obj[end] =value
        else:
            self.memory[key] = value
    def commit(self):
        for k in self.cache:
            # key , value , nesting , create = k
            self.real_save(*k)
    def rollback(self):
        self.cache = set()
    def close(self):
        if self.close_commit:
            self.commit()
        self.memory = None
        if self.closer:
            self.closer()

class TableMemory(MapMemory):
    def __init__(self,table_obj=None,close_commit=False,closer=None):
        super().__init__(table_obj,close_commit=close_commit,closer=closer)
    def save(self,keys,value):
        super().save(keys,value,nesting=True,create=True)
    def refer(self,keys):
        super().refer(keys,nesting=True)
