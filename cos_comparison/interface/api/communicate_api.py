#communicate tools

import os
import socket

from abc import ABC, abstractmethod

from ...core import no_done

#class chain
#BaseCommunicate(ABC)
#    Communicate
#        IOCommunicate
#            SocketCommunicate
#            FileCommunicate
#            FdCommunicate
#                PIPECommunicate

class BaseCommunicate(ABC):
    @abstractmethod
    def send(self):pass
    @abstractmethod
    def sendto(self):pass
    @abstractmethod
    def recv(self):pass
    @abstractmethod
    def recvfrom(self):pass
    @abstractmethod
    def close(self):pass

class Communicate(BaseCommunicate):
    __slots__ = ("obj","target","send_func","sendto_func","recv_func","recvfrom_func","close_func")
    def __init__(self,obj=None,target=None,
                 send_func=None,sendto_func=None,
                 recv_func=None,recvfrom_func=None,
                 close_func=None):
        self.obj = obj
        self.target = target
        self.send_func = send_func if send_func else no_done
        self.sendto_func = sendto_func if sendto_func else no_done
        self.recv_func = recv_func if recv_func else no_done
        self.recvfrom_func = recvfrom_func if recvfrom_func else no_done
        self.close_func = close_func if close_func else no_done
    def send(self,*args,**kwargs):
        return self.send_func(self,*args,**kwargs)
    def sendto(self,*args,**kwargs):
        return self.sendto_func(self,*args,**kwargs)
    def recv(self,*args,**kwargs):
        return self.recv_func(self,*args,**kwargs)
    def recvfrom(self,*args,**kwargs):
        return self.recvfrom_func(self,*args,**kwargs)
    def close(self,*args,**kwargs):
        return self.close_func(self,*args,**kwargs)

class IOCommunicate(Communicate):
    __slots__ = ("mode","reader","writer","closer")
    def __init__(self,obj=None,target=None,mode="rb",reader=None,writer=None,closer=None):
        self.mode = mode
        self.reader = reader if reader else no_done
        self.writer = writer if writer else no_done
        self.closer = closer if closer else no_done
        
        send_func = lambda self,data : self.writer(self.target,data)
        sendto_func = lambda self,data,target_fd : self.writer(target_fd,data)
        recv_func = lambda self,size : self.reader(self.obj,size)
        recvfrom_func = lambda self,target,size : self.reader(target,size)
        close_func = lambda self : self.closer(self.obj)

        super().__init__(obj=obj,target=target,
                         send_func=send_func,sendto_func=sendto_func,
                         recv_func=recv_func,recvfrom_func=recvfrom_func,
                         close_func=close_func)
    def connect(self,target):
        self.target = target
    def bind(self,fd):
        self.obj = fd

class FdCommunicate(IOCommunicate):
    __slots__ = ()
    def __init__(self,fd=None,target_fd=None,mode="rb",
                 reader=None,writer=None,closer=None):
        super().__init__(obj=fd,target=target_fd,mode=mode,
                         reader = reader if reader else os.read,
                         writer = writer if writer else os.write,
                         closer = closer if closer else os.close)

class PIPECommunicate(FdCommunicate):
    __slots__ = ()
    def __init__(self,read_fd=None,write_fd=None,auto=False):
        if auto:
            # explicit fds win (including fd 0, a valid descriptor that is
            # falsy); missing ones are filled from a fresh os.pipe()
            if read_fd is not None and write_fd is not None:
                pass
            else:
                r,w=os.pipe()
                read_fd = read_fd if read_fd is not None else r
                write_fd = write_fd if write_fd is not None else w
        super().__init__(fd=read_fd,target_fd=write_fd)

class SocketCommunicate(IOCommunicate):
    __slots__ = ()
    def __init__(self,addr_family=-1,socket_type=-1,proto=0,fileno=None,obj=None):
        obj = socket.socket(addr_family,socket_type,proto,fileno) if obj is None else obj
        super().__init__( obj=obj,target=obj,
                         reader=socket.socket.recv,
                         writer=socket.socket.sendall,
                         closer=socket.socket.close )
    def connect(self,addr):
        self.obj.connect(addr)
    def bind(self,addr):
        self.target.bind(addr)
    def listen(self,num=socket.SOMAXCONN):
        self.obj.listen(num)
    def accept(self):
        sub,addr = self.obj.accept()
        return self.__class__(obj=sub),addr

class FileCommunicate(IOCommunicate):
    __slots__ = ()
    def __init__(self,file_path,mode="rb",buffering=-1):
        obj = open(file_path,mode,buffering=buffering)
        io_type = type(obj)
        super().__init__(obj=obj,target = obj,
                         reader=io_type.read,
                         writer=io_type.write,
                         closer=io_type.close)

"""
Explicit public exports (prevents import-star namespace pollution).
"""
__all__ = (
    "BaseCommunicate",
    "Communicate",
    "IOCommunicate",
    "FdCommunicate",
    "PIPECommunicate",
    "SocketCommunicate",
    "FileCommunicate",
)
