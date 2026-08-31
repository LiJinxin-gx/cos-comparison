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
        self.cache = []
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
        self.cache.append( Transaction(key,value,nesting=nesting,create=create) )
    def real_save(self,key,value,nesting=False,create=True,create_hook=None):
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
                        if create_hook is not None:
                            obj[k] = create_hook(type(obj))
                        else:
                            obj[k] = create_map(type(obj))
                        obj = obj[k]
                    else:
                        raise
            obj[end] =value
        else:
            self.memory[key] = value
    def commit(self):
        applied = 0
        try:
            while applied < len(self.cache):
                self.real_save(*self.cache[applied])
                applied += 1
        finally:
            # always drop the applied entries so a retry never re-applies
            # them; the failing entry (and everything after it) stays queued.
            self.cache = self.cache[applied:]
    def rollback(self):
        self.cache = []
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
        return super().save(keys,value,nesting=True,create=True)
    def refer(self,keys):
        return super().refer(keys,nesting=True)

"""
Explicit public exports (prevents import-star namespace pollution).
"""
__all__ = (
    "create_map",
    "Transaction",
    "MapMemory",
    "TableMemory",
)
