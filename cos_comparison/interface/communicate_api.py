#communicate tools

import os
import socket

from abc import ABC

def no_done(*args,**kwargs):
    pass

class BaseCommunicate(ABC):
    def send(self):
        pass
    def sendto(self):
        pass
    def recv(self):
        pass
    def recvfrom(self):
        pass
    def close(self):
        pass

class Communicate(BaseCommunicate):
    __slots__("obj","send_func","sendto_func","recv_func","recvfrom_func","close_func")
    def __init__(self,obj,
                 send_func=None,sendto_func=None,
                 recv_func=None,recvfrom_func=None,
                 close_func=None):
        self.obj = obj
        self.send_func = send_func if send_func else no_done
        self.sendto_func = sendto_func if sendto_func else no_done
        self.recv_func = recv_func if recv_func else no_done
        self.recvfrom_func = recvfrom_func if recvfrom_func else no_done
        self.close_func = close_func if close_func else no_done
    def send(self,*args,**kwargs):
        return self.send_func(*args,**kwargs)
    def sendto(self,*args,**kwargs):
        return self.sendto_func(*args,**kwargs)
    def recv(self,*args,**kwargs):
        return self.recv_func(*args,**kwargs)
    def recvfrom(self,*args,**kwargs):
        return self.recvfrom_func(*args,**kwargs)
    def close(self,*args,**kwargs):
        return self.close_func(*args,**kwargs)

class PIPECommunicate(Communicate):
    def __init__(self):
        pass
