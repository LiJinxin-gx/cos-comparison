#database tools

from basememory import *
import sqlite3

class DatabaseMemory(Memory):
    __slots__ = ("cursor",)
    def __init__(self,database=":memory:"):
        memory = sqlite3.connect(database)
        super().__init__(memory)
        self.initialize()
    def __enter__(self):
        return self
    def __exit__(self,exc_type,exc_val,exc_tb):
        self.close()
    def initialize(self):
        self.cursor = self.memory.cursor()
    def commit(self):
        self.memory.commit()
    def close(self):
        self.cursor.close()
        self.memory.close()
