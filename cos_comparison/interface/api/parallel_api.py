"""
It provides abstraction of underlying parallel tools.
"""

import threading
import multiprocessing
import asyncio

from context_tool import *

thread_lock = threading.Lock
process_lock = multiprocessing.Lock
parallel_lock = IntegrateContext(process_lock(),thread_lock())

thread_rlock = threading.RLock
process_rlock = multiprocessing.RLock
parallel_rlock = IntegrateContext(process_rlock(),thread_rlock())

Thread = threading.Thread
Process = multiprocessing.Process
