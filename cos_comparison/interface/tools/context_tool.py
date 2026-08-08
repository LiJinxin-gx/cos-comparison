"""
Context tools
"""

class VoidContext:
    def __enter__(self):
        return self
    def __exit__(self,err_type,err_val,err_tb):
        pass

class IntegrateContext:
    __slots__ = ("contexts","flag")
    def __init__(self,*context):
        self.contexts = tuple(context)
        self.flag = 0
    def __enter__(self):
        try:
            n=0
            for c in self.contexts:
                c.__enter__()
                n+=1
        except Exception as e:
            self.flag = 0
            for i in range(n+1):
                c.__exit__(type(e),e,e.__traceback__ )
            raise
        else:
            self.flag = 1
            return self
    def __exit__(self,err_type,err_val,err_tb):
        if self.flag:
            for c in reversed(self.contexts):
                c.__exit__(err_type,err_val,err_tb)

class AsyncIntegrateContext(IntegrateContext):
    __slots__ = ("a_contexts",)
    def __init__(self,*a_context):
        self.a_contexts = tuple(a_context)
    async def __aenter__(self):
        for c in self.a_contexts:
            await c.__aenter__()
        return self
    async def __aexit__(self,err_type,err_val,err_tb):
        for c in reversed(self.a_contexts):
            await c.__aexit__(err_type,err_val,err_tb)
