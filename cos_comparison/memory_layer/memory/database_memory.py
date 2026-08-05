#database tools

from .basememory import *
from ...core import no_done

class DatabaseToolWrap:
    __slots__ = ("tool","conn_func","exec_func","execmany_func","cursor_func")
    def __init__(self,tool,conn_func=None,cursor_func=None):
        self.tool = tool
        self.conn_func = conn_func if conn_func else no_done
    def connect(self,database):
        self.conn_func(self.tool,datavase)

try:
    import sqlite3 as default_tool
except:
    pass

class DatabaseMemory(Memory):
    __slots__ = ("cursor",)
    def __init__(self,database_tool=None,database=":memory:",refer_func=None):
        database_tool = default_tool if database is None else database_tool
        memory = database_tool.connect(database)
        super().__init__(memory)
        self.refer_func = refer_func
        self.initialize()
    def __enter__(self):
        return self
    def __exit__(self,exc_type,exc_val,exc_tb):
        self.close()
    def initialize(self):
        try:
            self.cursor = self.memory.cursor()
            return 0
        except:
            return 1
    def commit(self):
        self.memory.commit()
    def rollback(self):
        self.memory.rollback()
    def close(self):
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
        except:
            self.memory.executemany(command,args)
