#It gives basic tools to debug.

#--------- import ----------
import time

#-------- time tools ---------
perf_count = time.perf_counter

class Timer:
    __slots__ = ("timer","total_time","last_time")
    def __init__( self , start = 0.0 , timer = perf_count ):
        self.total_time = start 
        self.timer = timer
        self.last_time = self.timer()
        
    def mark(self):
        a = self.last_time
        self.last_time = self.timer()
        self.total_time += ( self.last_time - a )
        
    def get_time(self):
        return self.total_time + self.timer() - self.last_time

    def reset(self):
        self.total_time = 0.0
        self.last_time = self.timer()
