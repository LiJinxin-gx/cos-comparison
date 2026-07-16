#it gives some api to interact with system.

#-------- import --------
import os
import sys
import threading
import subprocess

#------- system -------
command = os.system
getpid = os.getpid
getppid = os.getppid
home_executable = sys.executable

#------ process interaction ------
class BaseProcess(subprocess.Popen):
    def __init__(self, args, text=False, **kwargs):
        kwargs.setdefault("stdin", subprocess.PIPE)
        kwargs.setdefault("stdout", subprocess.PIPE)
        kwargs.setdefault("stderr", subprocess.PIPE)
        kwargs.setdefault("text", text)
        super().__init__(args, text=text,**kwargs)

class Process(BaseProcess):
    __slots__=("stdin_line","stdout_line","stderr_line","lock")
    def __init__(self,executable,arg_list,**kwarg):
        if type(arg_list) == str:
            args=[executable,arg_list]
        elif type(arg_list) in (tuple,list):
            args=[executable,*arg_list]
        else:
            raise TypeError("arg_list is unsupposed type.")
        self.stdin_line = []
        self.stdout_line =[]
        self.stderr_line = []
        self.lock = threading.RLock()
        kwarg["text"] = False #It force to use bytes.
        super().__init__(args,**kwarg) 
        threading.Thread(target=self._run,daemon=True).start()
    def execute(self,command):
        if isinstance(command, str):
            command = command.encode()
        w=self.stdin.write(command)
        self.stdin.flush()
        self.stdin_line.append(command)
        return w
    def _readline_stdout(self):
        line = self.stdout.readline()
        for message in line:
            self.stdout_line.append(message)
    def _readline_stderr(self):
        line = self.stderr.readline()
        for message in line:
            self.stderr_line.append(message)
    def _run(self):
        with self.lock:
            #the function can only run in one thread at the same time.
            out = threading.Thread(target=self._readline_stdout,daemon=True)
            err = threading.Thread(target=self._readline_stderr,daemon=True)
            out.start()
            err.start()
