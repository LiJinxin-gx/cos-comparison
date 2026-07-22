"""
cos_comparison_c.py - ctypes-based backend for cos_comparison.

This module provides the same API as the pure Python core but delegates
the heavy computation to a compiled C shared library (libcos_core.so/.dylib/.dll).
All functions accept the same parameters and return the same result types.

The C library must be compiled from the provided C sources (core.c, type_data.h, etc.)
and placed in a location where ctypes can find it.
"""

import ctypes
import os
import sys
from ctypes import c_int, c_double, c_void_p, POINTER, Structure, CFUNCTYPE
from math import sqrt

# ---------- platform-specific library loading ----------
_lib_path = None
_lib = None

_candidates = [
    "libcos_core.so",
    "libcos_core.dylib",
    "cos_core.dll",
    "libcos_core.dll",
    "core.dll",
]

_dir = os.path.dirname(os.path.abspath(__file__))
for name in _candidates:
    path = os.path.join(_dir, name)
    if os.path.exists(path):
        _lib_path = path
        break
if _lib_path is None:
    for name in _candidates:
        try:
            _lib = ctypes.CDLL(name)
            _lib_path = name
            break
        except OSError:
            continue
else:
    try:
        _lib = ctypes.CDLL(_lib_path)
    except OSError:
        _lib = None

if _lib is None:
    raise ImportError(
        "C library (libcos_core) not found. Please compile the C sources and place "
        "the shared library in the same directory as this file, or add it to LD_LIBRARY_PATH."
    )

# ---------- ctypes type definitions ----------
SIM_FUNC = CFUNCTYPE(c_double, c_double, c_double, c_double)

class Data(Structure):
    _fields_ = [
        ("dimension", c_int),
        ("shape", POINTER(c_int)),
        ("strides", POINTER(c_int)),
        ("data", POINTER(c_double)),
        ("owns_data", c_int),
    ]

class Linear(Structure):
    _fields_ = [
        ("w1", c_double),
        ("b1", c_double),
        ("w2", c_double),
        ("b2", c_double),
    ]

class Control(Structure):
    _fields_ = [
        ("li", Linear),
        ("start", POINTER(c_int)),
        ("end", POINTER(c_int)),
        ("step", POINTER(c_int)),
        ("d", POINTER(c_int)),
    ]

class Callback(Structure):
    _fields_ = [
        ("start", c_void_p),
        ("end", c_void_p),
        ("iter", c_void_p),
        ("local_error", c_void_p),
        ("global_error", c_void_p),
        ("return_cb", c_void_p),
    ]

# ---------- set up C function signatures (with ctx) ----------
_lib.cos_comparison_passive.argtypes = [
    POINTER(Data),          # data
    POINTER(c_int),         # window_size
    POINTER(Control),       # ctrl
    POINTER(Callback),      # cb
    SIM_FUNC,               # sim_func
    POINTER(Data),          # out_param (pre-allocated output)
    POINTER(c_int),         # output_start
    POINTER(c_int),         # output_step
    c_void_p,               # ctx
]
_lib.cos_comparison_passive.restype = POINTER(Data)

_lib.cos_comparison_active.argtypes = [
    POINTER(Data),          # data
    POINTER(Data),          # kernel
    POINTER(Control),       # ctrl
    POINTER(Callback),      # cb
    SIM_FUNC,               # sim_func
    POINTER(Data),          # out_param (pre-allocated output)
    POINTER(c_int),         # output_start
    POINTER(c_int),         # output_step
    c_void_p,               # ctx
]
_lib.cos_comparison_active.restype = POINTER(Data)

_lib.cos_full.argtypes = [
    POINTER(Data),          # a
    POINTER(Data),          # b
    SIM_FUNC,               # sim_func
]
_lib.cos_full.restype = c_double

# ---------- internal utilities ----------
def _to_c_int_array(seq):
    if seq is None:
        return None
    arr = (c_int * len(seq))(*seq)
    return arr

def _to_c_double_array(seq):
    if seq is None:
        return None
    arr = (c_double * len(seq))(*seq)
    return arr

def _create_data_from_list(lst):
    # Support vector_map_as_tensor input directly
    if isinstance(lst, vector_map_as_tensor):
        dim = lst.dimension - lst.p
        shape = tuple(lst.tensor_size[lst.p:])
        total = lst.end - lst.start
        data = Data()
        data.dimension = dim
        shape_arr = (c_int * dim)(*shape)
        data.shape = shape_arr
        strides = [1] * dim
        for i in range(dim-2, -1, -1):
            strides[i] = strides[i+1] * shape[i+1]
        stride_arr = (c_int * dim)(*strides)
        data.strides = stride_arr
        # Copy data from tensor's underlying vector
        data_arr = (c_double * total)()
        for i in range(total):
            data_arr[i] = lst.vector[lst.start + i]
        data.data = data_arr
        data.owns_data = 1
        data._shape_ref = shape_arr
        data._stride_ref = stride_arr
        data._data_ref = data_arr
        return data
    shape = []
    tmp = lst
    while isinstance(tmp, (list, tuple)):
        shape.append(len(tmp))
        if len(tmp) == 0:
            break
        tmp = tmp[0]
    dim = len(shape)
    if dim == 0:
        raise ValueError("empty tensor not supported")

    flat = []
    # Iterative flatten using stack to avoid recursion depth issues
    stack = [iter(lst)]
    while stack:
        try:
            node = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if isinstance(node, (list, tuple)):
            stack.append(iter(node))
        else:
            flat.append(float(node))
    total = len(flat)
    if total == 0:
        raise ValueError("empty tensor not supported")

    data = Data()
    data.dimension = dim
    shape_arr = (c_int * dim)(*shape)
    data.shape = shape_arr
    strides = [1] * dim
    for i in range(dim-2, -1, -1):
        strides[i] = strides[i+1] * shape[i+1]
    stride_arr = (c_int * dim)(*strides)
    data.strides = stride_arr
    data_arr = (c_double * total)(*flat)
    data.data = data_arr
    data.owns_data = 1
    # Keep references to avoid GC freeing the arrays.
    data._shape_ref = shape_arr
    data._stride_ref = stride_arr
    data._data_ref = data_arr
    return data

def _data_to_list(data):
    if not data:
        return None
    d = data.contents
    dim = d.dimension
    shape = [d.shape[i] for i in range(dim)]
    strides = [d.strides[i] for i in range(dim)]
    total = 1
    for s in shape:
        total *= s
    flat = [d.data[i] for i in range(total)]

    # Build nested list iteratively using carry mechanism - no recursion
    if dim == 0:
        return flat[0] if flat else None

    num_list = [None] + [1] * dim
    flag = dim
    result = None
    current_levels = []  # stack of current list references at each depth

    while flag:
        if flag == dim:
            # Calculate flat index
            idx = 0
            for i in range(dim):
                idx += (num_list[i + 1] - 1) * strides[i]
            val = flat[idx]

            # Navigate/create structure down to dim-1 level
            current = result
            for depth in range(dim - 1):
                pos = num_list[depth + 1] - 1
                if current is None:
                    current = []
                    result = current
                if pos >= len(current):
                    current.append([])
                current = current[pos]
            current.append(val)

        if num_list[flag] < shape[flag - 1]:
            num_list[flag] += 1
            flag = dim
        else:
            num_list[flag] = 1
            flag -= 1

    return result

def _copy_nested(src, dst):
    """Iteratively copy all elements from src to dst (same shape).
    Works with both nested lists and vector_map_as_tensor instances.
    """
    # Infer shape
    shape = []
    cur = src
    while True:
        try:
            n = len(cur)
        except (TypeError, AttributeError):
            break
        shape.append(n)
        if n == 0:
            break
        try:
            cur = cur[0]
        except (TypeError, IndexError):
            break
    dim = len(shape)
    if dim == 0:
        try:
            dst[0] = src
        except (TypeError, IndexError):
            pass
        return

    # Iterate using carry mechanism
    num_list = [None] + [1] * dim
    flag = dim
    while flag:
        if flag == dim:
            idx = tuple(num_list[i + 1] - 1 for i in range(dim))
            # Navigate to leaf for both src and dst using index access
            s_val = src
            d_val = dst
            for i in range(dim - 1):
                s_val = s_val[idx[i]]
                d_val = d_val[idx[i]]
            d_val[idx[-1]] = s_val[idx[-1]]

        if num_list[flag] < shape[flag - 1]:
            num_list[flag] += 1
            flag = dim
        else:
            num_list[flag] = 1
            flag -= 1


def _copy_nested_with_step(src, dst, start=None, step=None, dim=0):
    """Iteratively copy src into dst with output start offset and step.
    Works with both nested lists and vector_map_as_tensor instances.
    """
    # Infer src shape
    src_shape = []
    cur = src
    while True:
        try:
            n = len(cur)
        except (TypeError, AttributeError):
            break
        src_shape.append(n)
        if n == 0:
            break
        try:
            cur = cur[0]
        except (TypeError, IndexError):
            break
    dim_count = len(src_shape)
    if dim_count == 0:
        return

    if start is None:
        start = tuple(0 for _ in range(dim_count))
    if step is None:
        step = tuple(1 for _ in range(dim_count))

    # Iterate using carry mechanism
    num_list = [None] + [1] * dim_count
    flag = dim_count
    while flag:
        if flag == dim_count:
            src_idx = tuple(num_list[i + 1] - 1 for i in range(dim_count))
            dst_idx = tuple(start[i] + step[i] * (num_list[i + 1] - 1) for i in range(dim_count))

            # Navigate to leaf for src
            s_val = src
            for i in range(dim_count - 1):
                s_val = s_val[src_idx[i]]

            # Navigate to leaf for dst
            d_val = dst
            for i in range(dim_count - 1):
                d_val = d_val[dst_idx[i]]

            d_val[dst_idx[-1]] = s_val[src_idx[-1]]

        if num_list[flag] < src_shape[flag - 1]:
            num_list[flag] += 1
            flag = dim_count
        else:
            num_list[flag] = 1
            flag -= 1

def _to_int_array(arr, dim):
    if arr is None:
        return None
    if not isinstance(arr, (list, tuple)):
        arr = [arr] * dim
    if len(arr) != dim:
        if len(arr) < dim:
            arr = list(arr) + [0] * (dim - len(arr))
        else:
            arr = arr[:dim]
    return (c_int * dim)(*arr)

def _is_contiguous_array(obj):
    # Zero-dependency: only check for ctypes arrays, no numpy required
    if isinstance(obj, ctypes.Array):
        if obj._type_ == ctypes.c_double:
            length = len(obj)
            return True, (length,)
    return False, None

def _create_data_view(obj, shape):
    # Zero-dependency: create data view from ctypes array or buffer object
    dim = len(shape)
    shape_arr = (c_int * dim)(*shape)
    strides = [1] * dim
    for i in range(dim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    strides_arr = (c_int * dim)(*strides)
    data_ptr = ctypes.cast(obj, POINTER(c_double))
    data = Data()
    data.dimension = dim
    data.shape = shape_arr
    data.strides = strides_arr
    data.data = data_ptr
    data.owns_data = 0
    data._shape_arr = shape_arr
    data._strides_arr = strides_arr
    data._obj = obj
    return data

def _free_data(data):
    pass

# ---------- default callback (do nothing) ----------
_noop = CFUNCTYPE(None, c_void_p)(lambda x: None)
_null_cb = Callback()
_null_cb.start = c_void_p(0)
_null_cb.end = c_void_p(0)
_null_cb.iter = c_void_p(0)
_null_cb.local_error = c_void_p(0)
_null_cb.global_error = c_void_p(0)
_null_cb.return_cb = c_void_p(0)

# ---------- Python implementations of similarity functions ----------
def _py_cos(a, b, ab):
    return ab / sqrt(a*b) if a*b else (1.0 if a == b else 0.0)

def _py_mod(a, b, ab):
    return 2*sqrt(a*b)/(a+b) if a*b else (1.0 if a == b else 0.0)

def _py_cosmod(a, b, ab):
    return 2*ab/(a+b) if a*b else (1.0 if a == b else 0.0)

_cos_c = SIM_FUNC(_py_cos)
_mod_c = SIM_FUNC(_py_mod)
_cosmod_c = SIM_FUNC(_py_cosmod)

_algo_map = {
    _py_cos: _cos_c,
    _py_mod: _mod_c,
    _py_cosmod: _cosmod_c,
}

def _get_callback(algorithm):
    if algorithm is None:
        return None
    for py_func, c_func in _algo_map.items():
        if algorithm is py_func or algorithm.__code__ == py_func.__code__:
            return c_func
    def wrapper(a, b, ab):
        try:
            return algorithm(a, b, ab)
        except TypeError:
            return algorithm(a, b, ab, None)
    return SIM_FUNC(wrapper)

# ---------- private module (matching Python core) ----------
_cos = lambda a, b, ab, name: ab / sqrt(a * b) if a * b else (1.0 if a == b else 0.0)
_mod = lambda a, b, ab, name: 2 * sqrt(a * b) / (a + b) if a * b else (1.0 if a == b else 0.0)
_cosmod = lambda a, b, ab, name: 2 * ab / (a + b) if a * b else (1.0 if a == b else 0.0)
_default_algorithm = _cosmod
private_dict = {
    "_cos": _cos,
    "_mod": _mod,
    "_cosmod": _cosmod,
    "_default_algorithm": _default_algorithm
}

_algo_map[_cos] = _cos_c
_algo_map[_mod] = _mod_c
_algo_map[_cosmod] = _cosmod_c

# ---------- Callback wrappers (NEW) ----------
def _start_cb(ctx):
    if ctx:
        name_space = ctypes.cast(ctx, ctypes.py_object).value
        if hasattr(name_space, 'start_callback') and name_space.start_callback:
            name_space.start_callback(name_space)

def _end_cb(ctx):
    if ctx:
        name_space = ctypes.cast(ctx, ctypes.py_object).value
        if hasattr(name_space, 'end_callback') and name_space.end_callback:
            name_space.end_callback(name_space)

def _iter_cb(ctx):
    if ctx:
        name_space = ctypes.cast(ctx, ctypes.py_object).value
        # iter_callback is used for iter_a_callback and iter_b_callback in Python
        if hasattr(name_space, 'iter_a_callback') and name_space.iter_a_callback:
            name_space.iter_a_callback(name_space)
        if hasattr(name_space, 'iter_b_callback') and name_space.iter_b_callback:
            name_space.iter_b_callback(name_space)

def _local_error_cb(ctx):
    if ctx:
        name_space = ctypes.cast(ctx, ctypes.py_object).value
        if hasattr(name_space, 'local_error_callback') and name_space.local_error_callback:
            name_space.local_error_callback(name_space)

def _global_error_cb(ctx):
    if ctx:
        name_space = ctypes.cast(ctx, ctypes.py_object).value
        if hasattr(name_space, 'global_error_callback') and name_space.global_error_callback:
            name_space.global_error_callback(name_space)

def _return_cb(output, ctx):
    if ctx:
        name_space = ctypes.cast(ctx, ctypes.py_object).value
        if hasattr(name_space, 'return_callback') and name_space.return_callback:
            return name_space.return_callback(output, name_space)
    return output

# C function pointers for callbacks
START_CB = CFUNCTYPE(None, c_void_p)(_start_cb)
END_CB = CFUNCTYPE(None, c_void_p)(_end_cb)
ITER_CB = CFUNCTYPE(None, c_void_p)(_iter_cb)
LOCAL_ERR_CB = CFUNCTYPE(None, c_void_p)(_local_error_cb)
GLOBAL_ERR_CB = CFUNCTYPE(None, c_void_p)(_global_error_cb)
RETURN_CB = CFUNCTYPE(c_void_p, c_void_p, c_void_p)(_return_cb)

# ---------- public API ----------
def cos_comparison_passive(data, *arg, window_size=None, w1=1.0, w2=1.0, b1=0.0, b2=0.0,
                           start=None, end=None, step=None, d=None,
                           algorithm=_default_algorithm,
                           output=None, output_start=None, output_step=None,
                           start_callback=None, end_callback=None,
                           iter_a_callback=None, iter_b_callback=None,
                           global_error_callback=None, local_error_callback=None,
                           return_callback=None, **kwargs):
    if hasattr(data, "__cos_comparison_passive__"):
        dicts = locals()
        return data.__cos_comparison_passive__(data, *arg, **dicts)

    data_c = _create_data_from_list(data)
    if not data_c:
        raise ValueError("invalid data")
    
    result = None
    try:
        dim = data_c.dimension
        if window_size is None:
            ws = (c_int * dim)(*([1]*dim))
        else:
            if len(window_size) != dim:
                raise ValueError("window_size length must match data dimension")
            ws = (c_int * dim)(*window_size)

        ctrl = Control()
        ctrl.li.w1 = w1
        ctrl.li.b1 = b1
        ctrl.li.w2 = w2
        ctrl.li.b2 = b2

        if start is None:
            st = (c_int * dim)(*([0]*dim))
        else:
            st = (c_int * dim)(*start)
        if end is None:
            en = (c_int * dim)(*([-1]*dim))
        else:
            en = (c_int * dim)(*end)
        if step is None:
            sp = (c_int * dim)(*([1]*dim))
        else:
            sp = (c_int * dim)(*step)
        if d is None:
            dd = (c_int * dim)(*([0]*dim))
            if dim > 0:
                dd[0] = 1
        else:
            dd = (c_int * dim)(*d)

        ctrl.start = st
        ctrl.end = en
        ctrl.step = sp
        ctrl.d = dd

        sim_cb = _get_callback(algorithm)

        name = func_name_space(
            output=output,
            output_start=output_start,
            output_step=output_step,
            window_size=window_size,
            linear=(w1, w2, b1, b2),
            start=start, end=end, step=step, d=d,
            algorithm=algorithm,
            num=None,
            start_callback=start_callback,
            end_callback=end_callback,
            iter_a_callback=iter_a_callback,
            iter_b_callback=iter_b_callback,
            global_error_callback=global_error_callback,
            local_error_callback=local_error_callback,
            return_callback=return_callback,
        )

        # Create callback structure
        cb_struct = Callback()
        cb_struct.start = ctypes.cast(START_CB, c_void_p) if start_callback else None
        cb_struct.end = ctypes.cast(END_CB, c_void_p) if end_callback else None
        cb_struct.iter = ctypes.cast(ITER_CB, c_void_p)
        cb_struct.local_error = ctypes.cast(LOCAL_ERR_CB, c_void_p) if local_error_callback else None
        cb_struct.global_error = ctypes.cast(GLOBAL_ERR_CB, c_void_p) if global_error_callback else None
        cb_struct.return_cb = ctypes.cast(RETURN_CB, c_void_p) if return_callback else None

        # Pass name_space as ctx
        name_obj = ctypes.py_object(name)
        ctx = ctypes.cast(ctypes.pointer(name_obj), c_void_p)

        # output pre-allocation zero-copy
        use_pointer_opt = False
        out_view = None
        output_start_arr = None
        output_step_arr = None
        if output is not None:
            is_contig, shape = _is_contiguous_array(output)
            if is_contig:
                out_view = _create_data_view(output, shape)
                dim = len(shape)
                output_start_arr = _to_int_array(output_start, dim)
                output_step_arr = _to_int_array(output_step, dim)
                use_pointer_opt = True

        result_c = _lib.cos_comparison_passive(
            ctypes.byref(data_c),
            ws,
            ctypes.byref(ctrl),
            ctypes.byref(cb_struct),
            sim_cb,
            ctypes.byref(out_view) if use_pointer_opt else None,
            output_start_arr,
            output_step_arr,
            ctx
        )
        if not result_c:
            raise ValueError("passive mode failed (invalid parameters)")

        if use_pointer_opt:
            result = output
        else:
            # Convert C result to flat list
            d = result_c.contents
            dim = d.dimension
            shape = tuple(d.shape[i] for i in range(dim))
            total = 1
            for s in shape:
                total *= s
            flat = [d.data[i] for i in range(total)]
            # Create result tensor, use same class as input if input is vector_map_as_tensor
            if isinstance(data, vector_map_as_tensor):
                result = data.__class__(flat, shape, start=0, end=total, p=0, cache=None)
            else:
                result = vector_map_as_tensor(flat, shape, start=0, end=total, p=0, cache=None)
            try:
                _lib.Data_free.argtypes = [POINTER(Data)]
                _lib.Data_free.restype = None
                _lib.Data_free(result_c)
            except AttributeError:
                pass

        if output is not None and not use_pointer_opt:
            if output_start or output_step:
                _copy_nested_with_step(result, output, output_start, output_step)
            else:
                _copy_nested(result, output)
            result = output

        if end_callback:
            end_callback(name)

        if return_callback:
            result = return_callback(result, name)
            
    finally:
        # Free temporary input data (C-side copy, not the original Python object data)
        try:
            _lib.Data_free.argtypes = [POINTER(Data)]
            _lib.Data_free.restype = None
            _lib.Data_free(data_c)
        except AttributeError:
            pass

    return result

def cos_comparison_active(data, *arg, kernel=None, w1=1.0, w2=1.0, b1=0.0, b2=0.0,
                          start=None, end=None, step=None,
                          algorithm=_default_algorithm,
                          output=None, output_start=None, output_step=None,
                          start_callback=None, end_callback=None,
                          iter_a_callback=None, iter_b_callback=None,
                          global_error_callback=None, local_error_callback=None,
                          return_callback=None, **kwargs):
    if hasattr(data, "__cos_comparison_active__"):
        dicts = locals()
        return data.__cos_comparison_active__(data, *arg, **dicts)

    if kernel is None:
        raise ValueError("kernel must be provided for active mode")

    data_c = _create_data_from_list(data)
    if not data_c:
        raise ValueError("invalid data")
    kernel_c = _create_data_from_list(kernel)
    if not kernel_c:
        raise ValueError("invalid kernel")
    if data_c.dimension != kernel_c.dimension:
        raise ValueError("data and kernel dimensions must match")
    
    result = None
    try:
        dim = data_c.dimension

        ctrl = Control()
        ctrl.li.w1 = w1; ctrl.li.b1 = b1; ctrl.li.w2 = w2; ctrl.li.b2 = b2
        if start is None:
            st = (c_int * dim)(*([0]*dim))
        else:
            st = (c_int * dim)(*start)
        if end is None:
            en = (c_int * dim)(*([-1]*dim))
        else:
            en = (c_int * dim)(*end)
        if step is None:
            sp = (c_int * dim)(*([1]*dim))
        else:
            sp = (c_int * dim)(*step)
        ctrl.start = st; ctrl.end = en; ctrl.step = sp
        dd = (c_int * dim)(*([0]*dim))
        ctrl.d = dd

        sim_cb = _get_callback(algorithm)

        name = func_name_space(
            output=output,
            output_start=output_start,
            output_step=output_step,
            kernel=kernel,
            linear=(w1, w2, b1, b2),
            start=start, end=end, step=step,
            algorithm=algorithm,
            num=None,
            start_callback=start_callback,
            end_callback=end_callback,
            iter_a_callback=iter_a_callback,
            iter_b_callback=iter_b_callback,
            global_error_callback=global_error_callback,
            local_error_callback=local_error_callback,
            return_callback=return_callback,
        )

        cb_struct = Callback()
        cb_struct.start = ctypes.cast(START_CB, c_void_p) if start_callback else None
        cb_struct.end = ctypes.cast(END_CB, c_void_p) if end_callback else None
        cb_struct.iter = ctypes.cast(ITER_CB, c_void_p)
        cb_struct.local_error = ctypes.cast(LOCAL_ERR_CB, c_void_p) if local_error_callback else None
        cb_struct.global_error = ctypes.cast(GLOBAL_ERR_CB, c_void_p) if global_error_callback else None
        cb_struct.return_cb = ctypes.cast(RETURN_CB, c_void_p) if return_callback else None

        name_obj = ctypes.py_object(name)
        ctx = ctypes.cast(ctypes.pointer(name_obj), c_void_p)

        use_pointer_opt = False
        out_view = None
        output_start_arr = None
        output_step_arr = None
        if output is not None:
            is_contig, shape = _is_contiguous_array(output)
            if is_contig:
                out_view = _create_data_view(output, shape)
                dim = len(shape)
                output_start_arr = _to_int_array(output_start, dim)
                output_step_arr = _to_int_array(output_step, dim)
                use_pointer_opt = True

        result_c = _lib.cos_comparison_active(
            ctypes.byref(data_c),
            ctypes.byref(kernel_c),
            ctypes.byref(ctrl),
            ctypes.byref(cb_struct),
            sim_cb,
            ctypes.byref(out_view) if use_pointer_opt else None,
            output_start_arr,
            output_step_arr,
            ctx
        )
        if not result_c:
            raise ValueError("active mode failed")

        if use_pointer_opt:
            result = output
        else:
            # Convert C result to flat list
            d = result_c.contents
            dim = d.dimension
            shape = tuple(d.shape[i] for i in range(dim))
            total = 1
            for s in shape:
                total *= s
            flat = [d.data[i] for i in range(total)]
            # Create result tensor, use same class as input if input is vector_map_as_tensor
            if isinstance(data, vector_map_as_tensor):
                result = data.__class__(flat, shape, start=0, end=total, p=0, cache=None)
            else:
                result = vector_map_as_tensor(flat, shape, start=0, end=total, p=0, cache=None)
            try:
                _lib.Data_free.argtypes = [POINTER(Data)]
                _lib.Data_free.restype = None
                _lib.Data_free(result_c)
            except AttributeError:
                pass

        if output is not None and not use_pointer_opt:
            if output_start or output_step:
                _copy_nested_with_step(result, output, output_start, output_step)
            else:
                _copy_nested(result, output)
            result = output

        if end_callback:
            end_callback(name)

        if return_callback:
            result = return_callback(result, name)
            
    finally:
        # Free temporary input data (C-side copies, not original Python object data)
        try:
            _lib.Data_free.argtypes = [POINTER(Data)]
            _lib.Data_free.restype = None
            _lib.Data_free(data_c)
            _lib.Data_free(kernel_c)
        except AttributeError:
            pass

    return result

# 1D-4D aliases (same as generic)
cos_comparison_passive_1d = cos_comparison_passive
cos_comparison_passive_2d = cos_comparison_passive
cos_comparison_passive_3d = cos_comparison_passive
cos_comparison_passive_4d = cos_comparison_passive
cos_comparison_active_1d = cos_comparison_active
cos_comparison_active_2d = cos_comparison_active
cos_comparison_active_3d = cos_comparison_active
cos_comparison_active_4d = cos_comparison_active

# ---------- full tensor similarity ----------
def cos(a, b, algorithm=_cos):
    data_a = _create_data_from_list(a)
    data_b = _create_data_from_list(b)
    if not data_a or not data_b:
        raise ValueError("invalid input")
    try:
        sim_cb = _get_callback(algorithm)
        res = _lib.cos_full(ctypes.byref(data_a), ctypes.byref(data_b), sim_cb)
        return res
    finally:
        try:
            _lib.Data_free.argtypes = [POINTER(Data)]
            _lib.Data_free.restype = None
            _lib.Data_free(data_a)
            _lib.Data_free(data_b)
        except AttributeError:
            pass

cos_1d = cos
cos_2d = cos
cos_3d = cos
cos_4d = cos

# ---------- local mean and variance ----------
def mean_local(data, *arg, local_size=None, step=None, output=None, output_start=None, output_step=None, **kwargs):
    def ones(shape):
        if len(shape) == 1:
            return [1.0] * shape[0]
        else:
            return [ones(shape[1:]) for _ in range(shape[0])]
    kernel = ones(local_size)
    N = 1
    for s in local_size:
        N *= s
    def _mean_algo(a, b, ab, name=None):
        return ab / N
    result = cos_comparison_active(data, *arg, kernel=kernel, step=step, algorithm=_mean_algo,
                                   output=output, output_start=output_start, output_step=output_step,
                                   **kwargs)
    return result

def local_variance(data, *arg, local_size=None, step=None, output=None, output_start=None, output_step=None, **kwargs):
    mean = mean_local(data, *arg, local_size=local_size, step=step, **kwargs)
    def square(x):
        if isinstance(x, vector_map_as_tensor):
            return x * x
        if isinstance(x, list):
            return [square(y) for y in x]
        else:
            return x*x
    data_sq = square(data)
    mean_sq = mean_local(data_sq, *arg, local_size=local_size, step=step, **kwargs)
    def variance(msq, m):
        if isinstance(msq, vector_map_as_tensor):
            return msq - m * m
        if isinstance(msq, list):
            return [variance(x, y) for x, y in zip(msq, m)]
        else:
            return msq - m*m
    result = variance(mean_sq, mean)
    if output is not None:
        if output_start or output_step:
            _copy_nested_with_step(result, output, output_start, output_step)
        else:
            _copy_nested(result, output)
        return output
    return result

mean_local_1d = mean_local
mean_local_2d = mean_local
mean_local_3d = mean_local
mean_local_4d = mean_local
local_variance_1d = local_variance
local_variance_2d = local_variance
local_variance_3d = local_variance
local_variance_4d = local_variance

# ---------- utility functions (matching Python core) ----------
def multiple_chain(iterable, base=1):
    for m in iterable:
        base *= m
    return base

def add_chain(iterable, base=0):
    for m in iterable:
        base += m
    return base

def no_done(*arg, **kwarg):
    """Default empty callback"""
    pass

def create_void_list(length_list=(1,), default=None):
    if not length_list:
        return default
    length_list = tuple(length_list)
    total = multiple_chain(length_list)
    return vector_map_as_tensor([default for _ in range(total)], length_list)

def load_as_default_data(data, start=None, shape=None):
    """
    Load data as vector_map_as_tensor, matching pure Python implementation
    start: tuple of int, start coordinates in each dimension
    shape: tuple of int, size of the region to load in each dimension
    Supports loading sub-regions from multi-dimensional data, hides underlying type details
    """
    # First infer full shape of input data
    temp = data
    full_shape = []
    while True:
        try:
            full_shape.append(len(temp))
            temp = temp[0]
        except (TypeError, IndexError, AttributeError):
            break
    if not full_shape:
        raise ValueError("data is empty or scalar; cannot infer shape")
    dimension = len(full_shape)
    full_shape = tuple(full_shape)

    # Default shape is full shape if not provided
    if shape is None:
        shape = full_shape
    else:
        shape = tuple(shape)
        if len(shape) != dimension:
            raise ValueError(f"shape length {len(shape)} does not match data dimension {dimension}")
        for i in range(dimension):
            if shape[i] < 0 or shape[i] > full_shape[i]:
                raise ValueError(f"shape[{i}] = {shape[i]} out of bounds for dimension size {full_shape[i]}")

    # Default start is all zeros if not provided
    if start is None:
        start = tuple(0 for _ in range(dimension))
    else:
        start = tuple(start)
        if len(start) != dimension:
            raise ValueError(f"start length {len(start)} does not match data dimension {dimension}")
        for i in range(dimension):
            if start[i] < 0 or start[i] + shape[i] > full_shape[i]:
                raise ValueError(f"start[{i}] = {start[i]} out of bounds for dimension size {full_shape[i]} with shape {shape[i]}")

    # Calculate strides for each dimension (to convert multi-dim index to 1d offset)
    strides = [1] * dimension
    for i in range(dimension-2, -1, -1):
        strides[i] = strides[i+1] * full_shape[i+1]

    # Flatten the requested region using carry mechanism
    total_elements = 1
    for s in shape:
        total_elements *= s
    vector = [0.0] * total_elements

    num_list = [None] + [1] * dimension
    flag = dimension
    pos = 0
    while flag:
        if flag == dimension:
            # Calculate multi-dim index relative to region start
            idx = tuple(start[i] + num_list[i+1] - 1 for i in range(dimension))
            vector[pos] = get_item(data, idx)
            pos += 1
        if num_list[flag] < shape[flag - 1]:
            num_list[flag] += 1
            flag = dimension
        else:
            num_list[flag] = 1
            flag -= 1

    # Create tensor: note that the underlying vector is a copy of the region,
    # so start is 0 for the new tensor's own vector, shape is the requested region shape
    return vector_map_as_tensor(vector, shape, start=0, end=total_elements, p=0, cache=None)

def get_item(obj, index):
    if hasattr(obj, "__get_item__"):
        return obj.__get_item__(*index)
    temp = obj
    for i in index:
        temp = temp[i]
    return temp

# ---------- custom types (matching Python core) ----------
class func_name_space:
    __slots__ = ("output", "output_start", "output_step", "window_size", "kernel",
                 "linear", "start", "end", "d", "step", "algorithm", "num",
                 "start_callback", "end_callback", "iter_a_callback", "iter_b_callback",
                 "global_error_callback", "local_error_callback", "return_callback")
    def __init__(self, *arg, **kwarg):
        for key, value in kwarg.items():
            setattr(self, key, value)

class default_contain:
    __slots__ = ("default", "deep", "default_dict", "leng")
    def __init__(self, default, default_dict=None):
        self.default, self.default_dict = default, (default_dict if default_dict else {})
    def __len__(self):
        return float('nan')
    def __getitem__(self, index):
        return self.default_dict.get(index, self.default)

class vector_map_as_tensor:
    __slots__ = ("vector", "tensor_size", "dimension", "start", "end", "p", "cache")
    def __init__(self, vector, tensor_size, start=0, end=None, p=0, cache=None):
        self.vector = vector
        self.tensor_size = tensor_size
        self.dimension = len(tensor_size)
        self.start = start
        self.end = len(vector) if end is None else end
        self.p = p
        self.cache = multiple_chain(tensor_size[p+1:]) if cache is None else cache
    def __repr__(self):
        return f"<vector_map_as_tensor: dim={len(self.tensor_size)-self.p}, start={self.start}, end={self.end}, p={self.p}, cache={self.cache}>"

    def __getitem__(self, index):
        # Handle tuple of integers
        if isinstance(index, tuple) and all(isinstance(i, int) for i in index):
            return self.__get_item__(*index)
        # Handle single slice
        if isinstance(index, slice):
            length = self.tensor_size[self.p]
            start, stop, step = index.indices(length)
            if step != 1:
                raise NotImplementedError("step != 1 not supported for slice")
            new_start = self.start + start * self.cache
            new_end = self.start + stop * self.cache
            return self.__class__(self.vector, self.tensor_size, start=new_start, end=new_end, p=self.p, cache=self.cache)
        # Handle single integer
        if isinstance(index, int):
            if index < 0:
                index += self.tensor_size[self.p]
            new_p = self.p + 1
            new_start = self.start + index * self.cache
            if new_p == self.dimension:
                return self.vector[new_start]
            new_cache = multiple_chain(self.tensor_size[new_p+1:]) if new_p < self.dimension -1 else 1
            return self.__class__(self.vector, self.tensor_size, start=new_start, end=new_start + self.cache, p=new_p, cache=new_cache)
        raise TypeError(f"Invalid index type: {type(index)}")

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and all(isinstance(i, int) for i in key):
            idx = self.start
            for i, dim_i in enumerate(key):
                stride = multiple_chain(self.tensor_size[self.p + i + 1:]) if self.p + i < self.dimension -1 else 1
                idx += dim_i * stride
            self.vector[idx] = value
            return
        if isinstance(key, slice):
            raise NotImplementedError("slice assignment not supported")
        if isinstance(key, int):
            if key < 0:
                key += self.tensor_size[self.p]
            if self.p == self.dimension - 1:
                self.vector[self.start + key] = value
                return
            raise NotImplementedError("sub-tensor assignment not supported")
        raise TypeError(f"Invalid index type: {type(key)}")

    def __get_item__(self, *indexs):
        if len(indexs) != self.dimension - self.p:
            raise IndexError(f"expected {self.dimension - self.p} indices, got {len(indexs)}")
        idx = self.start
        for i, dim_i in enumerate(indexs):
            stride = multiple_chain(self.tensor_size[self.p + i + 1:]) if self.p + i < self.dimension -1 else 1
            idx += dim_i * stride
        return self.vector[idx]

    def __set_item__(self, indexs, value):
        p = self.p
        remaining = len(self.tensor_size) - p
        if len(indexs) == remaining:
            ptr = self.start
            cache = self.cache
            for i, idx in enumerate(indexs):
                ptr += idx * cache
                if i < remaining - 1:
                    cache //= self.tensor_size[p + i]
            self.vector[ptr] = value
        elif len(indexs) == 0:
            return
        else:
            raise IndexError("It was given some effectless index.")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return self.tensor_size[self.p]

    def _check_shape(self, other):
        if self.dimension != other.dimension or self.tensor_size != other.tensor_size:
            raise ValueError("the shape of two tensors are not same.")

    def __add__(self, other):
        if not isinstance(other, vector_map_as_tensor):
            raise TypeError("operands must be vector_map_as_tensor")
        self._check_shape(other)
        result = self.__class__([0.0]*(self.end - self.start), self.tensor_size[self.p:], start=0, end=self.end-self.start, p=0, cache=None)
        for i in range(self.end - self.start):
            result.vector[i] = self.vector[self.start + i] + other.vector[other.start + i]
        return result

    def __sub__(self, other):
        if not isinstance(other, vector_map_as_tensor):
            raise TypeError("operands must be vector_map_as_tensor")
        self._check_shape(other)
        result = self.__class__([0.0]*(self.end - self.start), self.tensor_size[self.p:], start=0, end=self.end-self.start, p=0, cache=None)
        for i in range(self.end - self.start):
            result.vector[i] = self.vector[self.start + i] - other.vector[other.start + i]
        return result

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = self.__class__([0.0]*(self.end - self.start), self.tensor_size[self.p:], start=0, end=self.end-self.start, p=0, cache=None)
            for i in range(self.end - self.start):
                result.vector[i] = self.vector[self.start + i] * other
            return result
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            result = self.__class__([0.0]*(self.end - self.start), self.tensor_size[self.p:], start=0, end=self.end-self.start, p=0, cache=None)
            for i in range(self.end - self.start):
                result.vector[i] = self.vector[self.start + i] * other.vector[other.start + i]
            return result
        raise TypeError("unsupported operand type(s) for *")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("division by zero")
            result = self.__class__([0.0]*(self.end - self.start), self.tensor_size[self.p:], start=0, end=self.end-self.start, p=0, cache=None)
            for i in range(self.end - self.start):
                result.vector[i] = self.vector[self.start + i] / other
            return result
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            result = self.__class__([0.0]*(self.end - self.start), self.tensor_size[self.p:], start=0, end=self.end-self.start, p=0, cache=None)
            for i in range(self.end - self.start):
                b = other.vector[other.start + i]
                if b == 0:
                    raise ZeroDivisionError("division by zero")
                result.vector[i] = self.vector[self.start + i] / b
            return result
        raise TypeError("unsupported operand type(s) for /")

    def __iadd__(self, other):
        if not isinstance(other, vector_map_as_tensor):
            raise TypeError("operands must be vector_map_as_tensor")
        self._check_shape(other)
        for i in range(self.end - self.start):
            self.vector[self.start + i] += other.vector[other.start + i]
        return self

    def __isub__(self, other):
        if not isinstance(other, vector_map_as_tensor):
            raise TypeError("operands must be vector_map_as_tensor")
        self._check_shape(other)
        for i in range(self.end - self.start):
            self.vector[self.start + i] -= other.vector[other.start + i]
        return self

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            for i in range(self.end - self.start):
                self.vector[self.start + i] *= other
            return self
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            for i in range(self.end - self.start):
                self.vector[self.start + i] *= other.vector[other.start + i]
            return self
        raise TypeError("unsupported operand type(s) for *=")

    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("division by zero")
            for i in range(self.end - self.start):
                self.vector[self.start + i] /= other
            return self
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            for i in range(self.end - self.start):
                b = other.vector[other.start + i]
                if b == 0:
                    raise ZeroDivisionError("division by zero")
                self.vector[self.start + i] /= b
            return self
        raise TypeError("unsupported operand type(s) for /=")

    def __neg__(self):
        result = self.__class__([0.0]*(self.end - self.start), self.tensor_size[self.p:], start=0, end=self.end-self.start, p=0, cache=None)
        for i in range(self.end - self.start):
            result.vector[i] = -self.vector[self.start + i]
        return result

    def __pos__(self):
        result = self.__class__([0.0]*(self.end - self.start), self.tensor_size[self.p:], start=0, end=self.end-self.start, p=0, cache=None)
        for i in range(self.end - self.start):
            result.vector[i] = self.vector[self.start + i]
        return result

    def __abs__(self):
        total = 0.0
        for i in range(self.end - self.start):
            val = self.vector[self.start + i]
            total += val * val
        return total ** 0.5

    def mean(self):
        total = 0.0
        count = self.end - self.start
        for i in range(count):
            total += self.vector[self.start + i]
        return total / count

    def variance(self):
        m = self.mean()
        total = 0.0
        count = self.end - self.start
        for i in range(count):
            val = self.vector[self.start + i]
            total += val * val
        return total / count - m * m

# ---------- expose the public API ----------
__all__ = [
    'cos_comparison_passive', 'cos_comparison_passive_1d', 'cos_comparison_passive_2d', 'cos_comparison_passive_3d', 'cos_comparison_passive_4d',
    'cos_comparison_active', 'cos_comparison_active_1d', 'cos_comparison_active_2d', 'cos_comparison_active_3d', 'cos_comparison_active_4d',
    'cos', 'cos_1d', 'cos_2d', 'cos_3d', 'cos_4d',
    'mean_local', 'mean_local_1d', 'mean_local_2d', 'mean_local_3d', 'mean_local_4d',
    'local_variance', 'local_variance_1d', 'local_variance_2d', 'local_variance_3d', 'local_variance_4d',
    'multiple_chain', 'add_chain', 'create_void_list', 'load_as_default_data', 'get_item',
    'vector_map_as_tensor', 'func_name_space', 'default_contain',
    '_cos', '_mod', '_cosmod', '_default_algorithm'
]
