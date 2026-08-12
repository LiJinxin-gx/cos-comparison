#database tools

from .basememory import *
from ...interface.api import DatabaseToolWrap, DATABASE_DRIVER

class DatabaseMemory(Memory):
    __slots__ = ("cursor", "wrap")
    def __init__(self, database_tool=None, database=":memory:", refer_func=None):
        tool = DATABASE_DRIVER if database_tool is None else database_tool
        if tool is None:
            raise RuntimeError("DatabaseMemory: no database driver available, provide one via database_tool=")
        self.wrap = DatabaseToolWrap(tool)
        self.cursor = None
        super().__init__(self.wrap.connect(database))
        self.refer_func = refer_func if refer_func else no_done
        self.initialize()
    def __enter__(self):
        return self
    def __exit__(self,exc_type,exc_val,exc_tb):
        self.close()
    def initialize(self):
        try:
            self.cursor = self.wrap.cursor(self.memory)
            return 0
        except:
            return 1
    def commit(self):
        self.memory.commit()
    def rollback(self):
        self.memory.rollback()
    def close(self):
        if self.cursor is not None:
            self.cursor.close()
        self.memory.close()
    def execute(self,command,arg=()):
        try:
            self.cursor.execute(command,arg)
        except:
            self.memory.execute(command,arg)
    def executemany(self,command,args=()):
        try:
            self.cursor.executemany(command,args)
            return 0
        except:
            try:
                self.memory.executemany(command,args)
                return 1
            except:
                return 2