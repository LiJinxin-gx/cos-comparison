from .basememory import *

def create_map(types):
    return types()

class MapMemory(Memory):
    __slots__ = ("cache","close_commit","closer")
    def __init__(self,map_obj=None,close_commit=False,closer=None):
        self.cache = {}
        self.close_commit = close_commit
        self.closer = closer
        super.__init__(memory=memory)
    def refer(self,key,nesting=False):
        if nesting:
            obj = self.memory
            for k in key:
                obj = obj[k]
            return obj
        else:
            return self.memory[key]
    def save(self,key,value,nesting=False,create=True):
        """
        the save function to save data in cache to support atomic event,you should commit it.
        """
        if nesting:
            obj = self.cache
            *key,end = key
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
            self.cache[key] = value
    def commit(self):
        for k in self.cache:
            self.memory[k] = self.cache[k]
    def close(self):
        if self.close_commit:
            self.commit()
        self.memory = {}
        if closer:
            closer()

class TableMemory(MapMemory):
    def __init__(self,table_obj=None,close_commit=False,closer=None):
        super().__init__(table_obj,close_commit=close_commit,closer=closer)
    def save(self,keys,value):
        super().save(keys,value,nesting=True,create=True)
    def refer(self,keys):
        super().refer(keys,nesting=True)
