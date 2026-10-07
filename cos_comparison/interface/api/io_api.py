#io stream abstraction (file / memory / delegated).

from abc import ABC, abstractmethod

from ...core import no_done

#class chain
#BaseIO(ABC)
#    IOStream(delegated container)
#        IOFile(file)
#        IOMemory(in-memory bytes/text)

class BaseIO(ABC):
    @abstractmethod
    def open(self, *args, **kwargs): pass
    @abstractmethod
    def read(self, n=-1): pass
    @abstractmethod
    def write(self, data): pass
    @abstractmethod
    def close(self): pass
    @abstractmethod
    def seek(self, offset, whence=0): pass
    @abstractmethod
    def tell(self): pass
    @abstractmethod
    def flush(self): pass
    @abstractmethod
    def readline(self, size=-1): pass

class IOStream(BaseIO):
    """Delegated stream container: every operation is a slot."""
    __slots__ = (
        "close_func",
        "flush_func",
        "obj",
        "open_func",
        "read_func",
        "readline_func",
        "seek_func",
        "tell_func",
        "write_func",
    )
    def __init__(self, obj=None, open_func=None, read_func=None,
                 write_func=None, close_func=None, seek_func=None,
                 tell_func=None, flush_func=None, readline_func=None):
        self.obj = obj
        self.open_func = open_func if open_func is not None else no_done
        self.read_func = read_func if read_func is not None else no_done
        self.write_func = write_func if write_func is not None else no_done
        self.close_func = close_func if close_func is not None else no_done
        self.seek_func = seek_func if seek_func is not None else no_done
        self.tell_func = tell_func if tell_func is not None else no_done
        self.flush_func = flush_func if flush_func is not None else no_done
        self.readline_func = readline_func if readline_func is not None else no_done
    def open(self, *args, **kwargs):
        return self.open_func(self, *args, **kwargs)
    def read(self, n=-1):
        return self.read_func(self, n)
    def write(self, data):
        return self.write_func(self, data)
    def close(self):
        return self.close_func(self)
    def seek(self, offset, whence=0):
        return self.seek_func(self, offset, whence)
    def tell(self):
        return self.tell_func(self)
    def flush(self):
        return self.flush_func(self)
    def readline(self, size=-1):
        return self.readline_func(self, size)

def _io_file_open(self, *args, **kwargs):
    import builtins
    # persistent stream handle; lifetime managed by close()
    f = builtins.open(self.path, self.mode, encoding=self.encoding)  # noqa: SIM115
    self.obj = f
    return f

def _io_file_read(self, n=-1):
    return self.obj.read(n)

def _io_file_write(self, data):
    return self.obj.write(data)

def _io_file_close(self):
    if self.obj is not None:
        self.obj.close()

def _io_file_seek(self, offset, whence=0):
    return self.obj.seek(offset, whence)

def _io_file_tell(self):
    return self.obj.tell()

def _io_file_flush(self):
    if self.obj is not None:
        self.obj.flush()

def _io_file_readline(self, size=-1):
    return self.obj.readline(size)

class IOFile(IOStream):
    """File stream factory: lazy open(path, mode)."""
    __slots__ = ("encoding", "mode", "path")
    def __init__(self, path, mode="r", encoding=None):
        self.path = path
        self.mode = mode
        self.encoding = encoding
        super().__init__(obj=None,
                         open_func=_io_file_open,
                         read_func=_io_file_read,
                         write_func=_io_file_write,
                         close_func=_io_file_close,
                         seek_func=_io_file_seek,
                         tell_func=_io_file_tell,
                         flush_func=_io_file_flush,
                         readline_func=_io_file_readline)

def _io_mem_open(self, *args, **kwargs):
    self.obj.seek(0)
    return self.obj

def _io_mem_read(self, n=-1):
    return self.obj.read(n)

def _io_mem_write(self, data):
    return self.obj.write(data)

def _io_mem_close(self):
    pass

def _io_mem_seek(self, offset, whence=0):
    return self.obj.seek(offset, whence)

def _io_mem_tell(self):
    return self.obj.tell()

def _io_mem_flush(self):
    pass

def _io_mem_readline(self, size=-1):
    return self.obj.readline(size)

class IOMemory(IOStream):
    """In-memory stream (BytesIO / StringIO)."""
    __slots__ = ("binary",)
    def __init__(self, data=b"", binary=True):
        import io as _io_mod
        self.binary = binary
        stream = _io_mod.BytesIO(data) if binary else _io_mod.StringIO(data)
        super().__init__(obj=stream,
                         open_func=_io_mem_open,
                         read_func=_io_mem_read,
                         write_func=_io_mem_write,
                         close_func=_io_mem_close,
                         seek_func=_io_mem_seek,
                         tell_func=_io_mem_tell,
                         flush_func=_io_mem_flush,
                         readline_func=_io_mem_readline)
    def getvalue(self):
        return self.obj.getvalue()

"""
Explicit public exports (prevents import-star namespace pollution).
"""
__all__ = (
    "BaseIO",
    "IOFile",
    "IOMemory",
    "IOStream",
)
