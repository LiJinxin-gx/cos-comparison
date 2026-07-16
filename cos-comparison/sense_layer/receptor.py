# It give a tool to recept data.

import os
import sys
from abc import ABC
from itertools import islice, product, tee, repeat

home_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, home_path)

import core

import threading
import multiprocessing

def thread_map(func, arg_list=(), kwarg_list=(), wait=False):
    tasks = []
    for arg, kwarg in zip(arg_list, kwarg_list):
        t = threading.Thread(target=func, args=arg, kwargs=kwarg)
        t.start()
        tasks.append(t)
    if wait:
        for t in tasks:
            t.join()

def process_map(func, arg_list=(), kwarg_list=(), count=None):
    count = count if count else os.cpu_count()
    with multiprocessing.Pool(count) as p:
        for arg, kwarg in zip(arg_list, kwarg_list):
            p.apply_async(func, args=arg, kwds=kwarg)

class BaseData(ABC):
    def __getitem__(self, index):
        pass
    def __setitem__(self, index, value):
        pass

class Data(core.vector_map_as_tensor, BaseData):
    def __init__(self,*arg,data=None,spilt_start=None,tensor_size=None,**kwarg):
        if data is None:
            super().__init__(*arg,**kwarg)
        else:
            origin=core.load_as_default_data(data,start=spilt_start,shape=tensor_size)
            super().__init__(origin.vector,origin.tensor_size)
            
    def _comparison_passive(self,*arg,**kwarg):
        origin=core.cos_comparison_passive(self,*arg,**kwarg)
        origin.__class__=self.__class__
        return origin
    def _comparison_active(self,*arg,**kwarg):
        origin=core.cos_comparison_passive(self,*arg,**kwarg)
        origin.__class__=self.__class__
        return origin

    compasrison_passive = _comparison_passive
    compasrison_active = _comparison_active

class Auto_Data(Data):
    __slots__ = ("task",)

    def __init__(self, data=None, task=None, **kwargs):
        self.task = task if task else thread_map
        # it do not advice to use process_map directly,because Process isolates context.
        #If you want to use process_map,you can pass the process-level shared data to the parameter "output".
        
        # Pass data to parent via keyword; other parent params are handled by **kwargs
        super().__init__(data=data, **kwargs)

    def _task_split(self, start=None, step=None, end=None, d=None, n=1):
        """
        Return a tuple of range objects, one per dimension.
        Each range yields valid starting indices for blocks.
        The length of each range determines the output shape.
        """
        shape = self.tensor_size
        ndim = len(shape)

        start = (start or (0,) * ndim)[:ndim] + (0,) * max(0, ndim - len(start or ()))
        step = (step or (1,) * ndim)[:ndim] + (1,) * max(0, ndim - len(step or ()))
        end = (end or shape)[:ndim] + shape[len(end or ()):] if end is not None else shape
        d = (d or (0,) * ndim)[:ndim] + (0,) * max(0, ndim - len(d or ()))

        # Block stride = step * n (original logic)
        block_stride = tuple(st * n for st in step)

        dim_ranges = []
        for dim in range(ndim):
            s = start[dim]
            e = end[dim]
            st = block_stride[dim]
            if s < e:
                dim_ranges.append(range(s, e, st))
            else:
                dim_ranges.append(range(0, 0))
        return tuple(dim_ranges)

    def _task(self, dimension_iters):
        return product(*dimension_iters)

    def _output_shape_from_ranges(self, dim_ranges):
        return tuple(len(r) for r in dim_ranges)

    def comparison_passive(self, *arg, start=None, step=None, end=None,
                           d=None, output=None, n=1, **kwarg):
        dim_ranges = self._task_split(start=start, step=step, end=end, d=d, n=n)
        out_shape = self._output_shape_from_ranges(dim_ranges)

        if not any(out_shape):
            return None

        if output is None:
            output = core.create_void_list(length_list=out_shape)

        start_iter = self._task(dim_ranges)
        ndim = len(self.tensor_size)
        if step is None:
            step = (1,) * ndim
        block_size = tuple(st * n for st in step)
        d_vals = d if d is not None else (0,) * ndim

        def kwarg_generator():
            for s in start_iter:
                end_block = tuple(min(s[i] + block_size[i], self.tensor_size[i]) for i in range(ndim))
                kw_dict = {
                    "start": s,
                    "end": end_block,
                    "step": step,
                    "d": d_vals,
                    "output": output,
                }
                kw_dict.update(kwarg)
                yield kw_dict

        arg_iter = repeat((self,) + arg)
        self.task(core.cos_comparison_passive,
                  arg_list=arg_iter,
                  kwarg_list=kwarg_generator())

        # Return new instance using the fast path (share vector)
        return type(self)(output.vector, output.tensor_size)

    def comparison_active(self, *arg, kernel=None, start=None, step=None,
                          end=None, output=None, n=1, **kwarg):
        if kernel is None:
            raise ValueError("kernel must be provided for active mode")

        dim_ranges = self._task_split(start=start, step=step, end=end, d=None, n=n)
        out_shape = self._output_shape_from_ranges(dim_ranges)

        if not any(out_shape):
            return None

        if output is None:
            output = core.create_void_list(length_list=out_shape)

        ndim = len(self.tensor_size)
        if step is None:
            step = (1,) * ndim
        block_size = tuple(st * n for st in step)

        start_iter = self._task(dim_ranges)

        def kwarg_generator():
            for s in start_iter:
                end_block = tuple(min(s[i] + block_size[i], self.tensor_size[i]) for i in range(ndim))
                kw_dict = {
                    "start": s,
                    "end": end_block,
                    "step": step,
                    "kernel": kernel,
                    "output": output,
                }
                kw_dict.update(kwarg)
                yield kw_dict

        arg_iter = repeat((self,) + arg)
        self.task(core.cos_comparison_active,
                  arg_list=arg_iter,
                  kwarg_list=kwarg_generator())

        return type(self)(output.vector, output.tensor_size)
