"""
Context tools
"""

class VoidContext:
    def __enter__(self):
        return self
    def __exit__(self,err_type,err_val,err_tb):
        pass

class IntegrateContext:
    __slots__ = ("contexts",)
    def __init__(self,*context):
        self.contexts = tuple(context)
    def __enter__(self):
        for c in self.contexts:
            c.__enter__()
        return self
    def __exit__(self,err_type,err_val,err_tb):
        for c in reversed(self.contexts):
            c.__exit__(err_type,err_val,err_tb)

class AsyncIntegrateContext(IntegrateContext):
    __slots__ = ("a_contexts",)
    def __init__(self,*a_context):
        self.a_contexts = tuple(a_context)
    async def __enter__(self):
        for c in self.a_contexts:
            c.__aenter__()
        return self
    async def __exit__(self,err_type,err_val,err_tb):
        for c in reversed(self.a_contexts):
            c.__aexit__(err_type,err_val,err_tb)
