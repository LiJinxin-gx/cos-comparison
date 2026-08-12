"""
It provides docker to run.
"""

from abc import ABC, abstractmethod

class BaseDocker(ABC):
    @abstractmethod
    def fork(self):
        pass
