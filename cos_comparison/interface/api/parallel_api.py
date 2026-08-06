"""
It provides abstraction of underlying parallel tools.
"""

import threading
import multiprocessing
import asyncio

from ..tools.context_tool import *

thread_lock = threading.Lock
process_lock = multiprocessing.Lock
parallel_lock = IntegrateContext(process_lock(),thread_lock())

thread_rlock = threading.RLock
process_rlock = multiprocessing.RLock
parallel_rlock = IntegrateContext(process_rlock(),thread_rlock())

Thread = threading.Thread
Process = multiprocessing.Process

def share_array(dtype,length):
    """
    It provides an array to share in different process.
    """
    return multiprocessing.Array(dtype,length)

def load_array(dtypes,sequence,length=None):
    length = len(sequence) if length is None else length
    container = share_array(dtypes,length)
    for i in range(length):
        container[i] = sequence[i]
    return container
