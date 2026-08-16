"""
cos_comparison_c.py - ctypes-based backend for cos_comparison.

This module provides the same API as the pure Python core but delegates
the heavy computation to a compiled C shared library (libcos_core.so/.dylib/.dll).
All functions accept the same parameters and return the same result types.

The C library must be compiled from the provided C sources (core.c, type_data.h, etc.)
and placed in a location where ctypes can find it.
"""

import ctypes
import os.path
from ctypes import c_int, c_double, c_void_p, POINTER, Structure, CFUNCTYPE
from math import sqrt

# ---------- platform-specific library loading ----------
_lib_path = None
_lib = None

_candidates = [
    "core.so",
    "core.dylib",
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
        ("dtype", c_int),
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
        shape = tuple(lst.shape)
        dim = len(shape)
        total = 1
        for s in shape:
            total *= s
        data = Data()
        data.dimension = dim
        shape_arr = (c_int * dim)(*shape)
        data.shape = shape_arr
        strides = [1] * dim
        for i in range(dim-2, -1, -1):
            strides[i] = strides[i+1] * shape[i+1]
        stride_arr = (c_int * dim)(*strides)
        data.strides = stride_arr
        # Copy data from tensor using _iter_flat (handles non-contiguous views correctly)
        data_arr = (c_double * total)()
        indices = list(lst._iter_flat())
        for i in range(total):
            data_arr[i] = lst.vector[indices[i]]
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

    data = Data()
    data.dimension = dim
    shape_arr = (c_int * dim)(*shape)
    data.shape = shape_arr
    strides = [1] * dim
    for i in range(dim-2, -1, -1):
        strides[i] = strides[i+1] * shape[i+1]
    stride_arr = (c_int * dim)(*strides)
    data.strides = stride_arr
    # Allocate at least 1 element to avoid zero-length ctypes arrays
    alloc_count = total if total > 0 else 1
    if total > 0:
        data_arr = (c_double * alloc_count)(*flat)
    else:
        data_arr = (c_double * alloc_count)()
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
# ctx transport uses the name object's id() mapped through a registry.
# This avoids ctypes.cast(..., ctypes.py_object), which depends on raw
# memory layout and is unreliable (may crash) under newer Python versions.
_callback_registry = {}

def _get_cb_name(ctx):
    if not ctx:
        return None
    return _callback_registry.get(ctypes.cast(ctx, c_void_p).value)

def _start_cb(ctx):
    name_space = _get_cb_name(ctx)
    if name_space is not None:
        if hasattr(name_space, 'start_callback') and name_space.start_callback:
            name_space.start_callback(name_space)

def _end_cb(ctx):
    name_space = _get_cb_name(ctx)
    if name_space is not None:
        if hasattr(name_space, 'end_callback') and name_space.end_callback:
            name_space.end_callback(name_space)

def _iter_cb(ctx):
    name_space = _get_cb_name(ctx)
    if name_space is not None:
        # iter_callback is used for iter_a_callback and iter_b_callback in Python
        if hasattr(name_space, 'iter_a_callback') and name_space.iter_a_callback:
            name_space.iter_a_callback(name_space)
        if hasattr(name_space, 'iter_b_callback') and name_space.iter_b_callback:
            name_space.iter_b_callback(name_space)

def _local_error_cb(ctx):
    name_space = _get_cb_name(ctx)
    if name_space is not None:
        if hasattr(name_space, 'local_error_callback') and name_space.local_error_callback:
            name_space.local_error_callback(name_space)

def _global_error_cb(ctx):
    name_space = _get_cb_name(ctx)
    if name_space is not None:
        if hasattr(name_space, 'global_error_callback') and name_space.global_error_callback:
            name_space.global_error_callback(name_space)

def _return_cb(output, ctx):
    name_space = _get_cb_name(ctx)
    if name_space is not None:
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
        if iter_a_callback or iter_b_callback:
            cb_struct.iter = ctypes.cast(ITER_CB, c_void_p)
        else:
            cb_struct.iter = None
        cb_struct.local_error = ctypes.cast(LOCAL_ERR_CB, c_void_p) if local_error_callback else None
        cb_struct.global_error = ctypes.cast(GLOBAL_ERR_CB, c_void_p) if global_error_callback else None
        cb_struct.return_cb = ctypes.cast(RETURN_CB, c_void_p) if return_callback else None

        _callback_registry[id(name)] = name
        ctx = id(name)

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
            raise ValueError("effectless args.")

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
                result = data.__class__(vector=flat, shape=shape, start=0)
            else:
                result = vector_map_as_tensor(vector=flat, shape=shape, start=0)
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
        # NOTE: data_c is a Python-managed ctypes Structure (its buffers are
        # Python-owned arrays). It must NOT be released via C Data_free, which
        # only frees structs allocated with Data_create. Python GC handles it.
        if 'name' in locals():
            _callback_registry.pop(id(name), None)
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
        if iter_a_callback or iter_b_callback:
            cb_struct.iter = ctypes.cast(ITER_CB, c_void_p)
        else:
            cb_struct.iter = None
        cb_struct.local_error = ctypes.cast(LOCAL_ERR_CB, c_void_p) if local_error_callback else None
        cb_struct.global_error = ctypes.cast(GLOBAL_ERR_CB, c_void_p) if global_error_callback else None
        cb_struct.return_cb = ctypes.cast(RETURN_CB, c_void_p) if return_callback else None

        _callback_registry[id(name)] = name
        ctx = id(name)

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
            raise ValueError("effectless args.")

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
                result = data.__class__(vector=flat, shape=shape, start=0)
            else:
                result = vector_map_as_tensor(vector=flat, shape=shape, start=0)
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
        # NOTE: data_c/kernel_c are Python-managed ctypes Structures (their
        # buffers are Python-owned arrays). They must NOT be released via C
        # Data_free; Python GC handles them. Only result_c (allocated by C
        # Data_create) is released above.
        if 'name' in locals():
            _callback_registry.pop(id(name), None)
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
        # NOTE: data_a/data_b are Python-managed ctypes Structures (their
        # buffers are Python-owned arrays). They must NOT be released via C
        # Data_free, which would free() non-heap Python buffers. Python GC
        # handles them.
        pass

cos_1d = cos
cos_2d = cos
cos_3d = cos
cos_4d = cos

# ---------- local mean and variance ----------
def _build_ones_iter(shape):
    """Build an all-one kernel with the given shape (iterative, no recursion)."""
    if isinstance(shape, int):
        return [1.0] * shape
    shape = tuple(shape)
    if len(shape) == 0:
        return 1.0
    flat = [1.0] * multiple_chain(shape, 1)
    stack = flat
    for dim in range(len(shape) - 1, -1, -1):
        width = shape[dim]
        stack = [stack[i:i + width] for i in range(0, len(stack), width)]
    return stack[0]

def _flatten_nested(obj):
    """Flatten an arbitrarily nested iterable into a flat list (row-major).
    Iterative only — never recurses."""
    flat = []
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            for sub in reversed(item):
                stack.append(sub)
        else:
            flat.append(item)
    return flat

def _flat_to_window(values, shape):
    """Reshape a flat list into a nested list of the given shape (row-major).
    Iterative only — never recurses.
    The first `product(shape)` values are taken; the rest are ignored,
    matching the C backend's per-window pattern semantics."""
    if not shape:
        return values[0]
    n = 1
    for s in shape:
        n *= s
    if len(values) < n:
        raise ValueError("weight is too small for the local_size window")
    stack = [1.0 * v for v in values[:n]]
    for dim in range(len(shape) - 1, -1, -1):
        width = shape[dim]
        stack = [stack[i:i + width] for i in range(0, len(stack), width)]
    return stack[0]

def mean_local(data, *arg, local_size=None, step=None, weight=None,
               output=None, output_start=None, output_step=None, **kwargs):
    # Match pure-Python defaults: None/builtin int/1-D sequences are normalized
    if local_size is None:
        local_size = (1,) * len(infer_shape(data) or (1,))
    elif isinstance(local_size, int):
        local_size = (local_size,)
    else:
        local_size = tuple(local_size)

    if weight is None:
        kernel = _build_ones_iter(local_size)
    else:
        # Same semantics as the (authoritative) pydll backend: weight is a
        # per-window pattern built from the first product(local_size)
        # flattened values; the window size is always local_size.
        kernel = _flat_to_window(_flatten_nested(weight), local_size)
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
    if local_size is None:
        local_size = (1,) * len(infer_shape(data) or (1,))
    elif isinstance(local_size, int):
        local_size = (local_size,)
    else:
        local_size = tuple(local_size)
    if step is None:
        step = (1,) * len(local_size)
    elif isinstance(step, int):
        step = (step,)
    else:
        step = tuple(step)
    kernel = _build_ones_iter(local_size)
    N = 1
    for s in local_size:
        N *= s
    def _var_algo(a, b, ab, name=None):
        mean = ab / N
        return a / N - mean * mean
    result = cos_comparison_active(data, *arg, kernel=kernel, step=step, algorithm=_var_algo,
                                   output=output, output_start=output_start, output_step=output_step,
                                   **kwargs)
    return result

mean_local_1d = mean_local
mean_local_2d = mean_local
mean_local_3d = mean_local
mean_local_4d = mean_local
local_variance_1d = local_variance
local_variance_2d = local_variance
local_variance_3d = local_variance
local_variance_4d = local_variance

# element-wise filter / mapping over a sampled region (callback-based)
# ---------------------------------------------------------------------------

def _region_spec(data, start, shape, step):
    """Resolve (effective_shape, start, step) for a read region, with the
    same clipping semantics as load_data's source side. Iterative only."""
    data_shape = infer_shape(data)
    if data_shape is None:
        raise ValueError("cannot infer shape of data")
    dimension = len(data_shape)
    if start is None:
        start = (0,) * dimension
    else:
        start = tuple(start)
        if len(start) != dimension:
            raise ValueError("start length does not match data dimension")
        for v in start:
            if v < 0:
                raise ValueError("start entries must be non-negative")
    if step is None:
        step = (1,) * dimension
    else:
        step = tuple(step)
        if len(step) != dimension:
            raise ValueError("step length does not match data dimension")
        for v in step:
            if v <= 0:
                raise ValueError("step entries must be positive")
    if shape is None:
        shape = tuple((data_shape[i] + step[i] - 1) // step[i]
                      for i in range(dimension))
    else:
        shape = tuple(shape)
        if len(shape) != dimension:
            raise ValueError("shape length does not match data dimension")
        for v in shape:
            if v < 0:
                raise ValueError("shape entries cannot be negative")
    effective = []
    for i in range(dimension):
        if start[i] < data_shape[i]:
            avail = (data_shape[i] - start[i] + step[i] - 1) // step[i]
        else:
            avail = 0
        effective.append(min(shape[i], avail))
    return tuple(effective), start, step


def _region_walk(effective):
    """Yield local coordinates in row-major order (iterative odometer)."""
    total = 1
    for n in effective:
        total *= n
    dimension = len(effective)
    idx = [0] * dimension
    for _ in range(total):
        yield tuple(idx)
        for i in range(dimension - 1, -1, -1):
            idx[i] += 1
            if idx[i] < effective[i]:
                break
            idx[i] = 0


def data_filter(data, callback, *, start=None, shape=None, step=None,
                origin=None, basis=None):
    """
    Yield the position (multi-dimensional index) of every element whose
    callback(value) is truthy, over a sampled read region.

    Read region: start / shape / step (same semantics as load_data's source
    side; out-of-bounds silently clipped). Reported position:
    origin + basis * local, where local is the row-major in-region
    coordinate (defaults origin=start, basis=step, i.e. the global read
    position). Callback errors are silently skipped. Iterative, never
    recursive; the callback is stateless (value only).
    """
    effective, r_start, r_step = _region_spec(data, start, shape, step)
    dimension = len(effective)
    if origin is None:
        origin = r_start
    else:
        origin = tuple(origin)
        if len(origin) != dimension:
            raise ValueError("origin length does not match data dimension")
    if basis is None:
        basis = r_step
    else:
        basis = tuple(basis)
        if len(basis) != dimension:
            raise ValueError("basis length does not match data dimension")
    for local in _region_walk(effective):
        read = tuple(r_start[i] + local[i] * r_step[i] for i in range(dimension))
        value = get_item(data, read)
        try:
            hit = callback(value)
        except Exception:
            continue
        if hit:
            yield tuple(origin[i] + basis[i] * local[i] for i in range(dimension))


def data_mapping(data, callback, *, start=None, shape=None, step=None,
                 out=None, out_start=None, out_step=None):
    """
    Map every element of the sampled read region through callback(value) and
    write the result to the output at the corresponding position; callback
    errors are silently skipped (the position stays untouched).

    Output: pre-allocated via `out` (default: a fresh tensor shaped like the
    read region); write position = out_start + out_step * local
    (out-of-bounds silently clipped, same semantics as load_data's target
    side). Returns the output tensor.
    """
    effective, r_start, r_step = _region_spec(data, start, shape, step)
    dimension = len(effective)
    if out is None:
        out = create_void_list(effective)
    out_shape = infer_shape(out)
    if out_shape is None:
        raise ValueError("cannot infer shape of output")
    if len(out_shape) != dimension:
        raise ValueError("output dimension does not match data dimension")
    if out_start is None:
        out_start = (0,) * dimension
    else:
        out_start = tuple(out_start)
        if len(out_start) != dimension:
            raise ValueError("out_start length does not match data dimension")
        for v in out_start:
            if v < 0:
                raise ValueError("out_start entries must be non-negative")
    if out_step is None:
        out_step = (1,) * dimension
    else:
        out_step = tuple(out_step)
        if len(out_step) != dimension:
            raise ValueError("out_step length does not match data dimension")
        for v in out_step:
            if v <= 0:
                raise ValueError("out_step entries must be positive")
    for local in _region_walk(effective):
        read = tuple(r_start[i] + local[i] * r_step[i] for i in range(dimension))
        value = get_item(data, read)
        try:
            mapped = callback(value)
        except Exception:
            continue
        write = tuple(out_start[i] + local[i] * out_step[i]
                      for i in range(dimension))
        if all(write[i] < out_shape[i] for i in range(dimension)):
            set_item(out, write, mapped)
    return out


def _make_threshold_predicate(low, high, inclusive):
    """Stateless predicate for the interval [low, high] (bounds optional);
    inclusive=(lo_in, hi_in) controls endpoint membership."""
    lo_in, hi_in = inclusive
    if low is None and high is None:
        raise ValueError("threshold requires at least one bound")
    if low is not None and high is not None and low > high:
        raise ValueError("low must not exceed high")
    if low is not None and high is None:
        return (lambda v: v >= low) if lo_in else (lambda v: v > low)
    if low is None and high is not None:
        return (lambda v: v <= high) if hi_in else (lambda v: v < high)
    if lo_in and hi_in:
        return lambda v: low <= v <= high
    if not lo_in and hi_in:
        return lambda v: low < v <= high
    if lo_in and not hi_in:
        return lambda v: low <= v < high
    return lambda v: low < v < high


class _Skip(Exception):
    """Internal sentinel: skip this position (mapped to silent skip)."""


def threshold_filter(data, low=None, high=None, *, inclusive=(True, True),
                     **region):
    """data_filter over the interval [low, high]: yields the positions whose
    value lies in the threshold range (endpoints per inclusive)."""
    predicate = _make_threshold_predicate(low, high, inclusive)
    return data_filter(data, predicate, **region)


def threshold_map(data, low=None, high=None, *, inclusive=(True, True),
                  map_func=None, **region):
    """data_mapping restricted to the interval [low, high]: map_func(value)
    is applied only where the value is in range; positions outside the
    interval (and callback errors) are silently skipped."""
    if map_func is None:
        raise ValueError("threshold_map requires map_func")
    predicate = _make_threshold_predicate(low, high, inclusive)

    def masked(value):
        if predicate(value):
            return map_func(value)
        raise _Skip()

    return data_mapping(data, masked, **region)

# ---------- utility functions (matching Python core) ----------
def multiple_chain(iterable, base=1):
    """Multiply all elements in an iterable."""
    for m in iterable:
        base *= m
    return base

def add_chain(iterable, base=0):
    """Add all elements in an iterable."""
    for m in iterable:
        base += m
    return base

def no_done(*arg, **kwarg):
    """Placeholder callback that does nothing."""
    pass

def infer_shape(data):
    """
    Infer the shape of multi-dimensional data.
    Priority: PyBuffer protocol > __shape__() method > recursive length detection.
    Returns shape tuple, or None if cannot infer.
    """
    # Try PyBuffer protocol first
    if hasattr(data, '__buffer__'):
        try:
            mv = memoryview(data)
            if mv.ndim > 0:
                return tuple(mv.shape)
        except (TypeError, ValueError):
            pass
    
    # Try __shape__ method (fast path for our own tensors, overridable by subclasses)
    if hasattr(data, '__shape__') and callable(data.__shape__):
        try:
            result = data.__shape__()
            if result is not None:
                return tuple(result)
        except (TypeError, ValueError):
            pass
    
    # Fallback: iterative length detection (general algorithm, no recursion)
    shape = []
    temp = data
    while True:
        try:
            n = len(temp)
            shape.append(n)
            if n == 0:
                break
            temp = temp[0]
        except (TypeError, IndexError, AttributeError):
            break
    
    return tuple(shape) if shape else None

def create_void_list(length_list=(1,), default=0.0):
    """Create a multi-dimensional nested list filled with default value."""
    if not length_list:
        return default
    length_list = tuple(length_list)
    total = multiple_chain(length_list)
    return vector_map_as_tensor(vector=[default for _ in range(total)], shape=length_list)

def load_as_default_data(data, start=None, shape=None, step=None):
    """
    Load data as vector_map_as_tensor, matching pure Python implementation.
    
    Args:
        data: Input data (nested list, buffer-like, or vector_map_as_tensor)
        start: tuple of int, start coordinates in each dimension (default: all zeros)
        shape: tuple of int, size of the region to load in each dimension (default: full shape)
        step: tuple of int, step size in each dimension (default: all ones)
    
    Supports loading sub-regions with arbitrary step sizes from multi-dimensional data.
    Hides underlying type details - works with lists, arrays, memoryview, and tensors.
    Uses efficient slice operations when input is already a tensor type.
    """
    # Fast path: input is already our tensor type - use native slicing
    if isinstance(data, vector_map_as_tensor):
        slices = []
        dim = data.dimension
        for i in range(dim):
            s = start[i] if start is not None else None
            e = (start[i] + shape[i]) if (start is not None and shape is not None) else None
            st = step[i] if step is not None else None
            slices.append(slice(s, e, st))
        result = data[tuple(slices)]
        # Return a copy (contiguous) to match expected behavior
        total = 1
        for s in result.shape:
            total *= s
        new_vector = [0.0] * total
        idx = 0
        for flat_idx in result._iter_flat():
            new_vector[idx] = result.vector[flat_idx]
            idx += 1
        return vector_map_as_tensor(vector=new_vector, shape=result.shape, start=0)
    
    # Infer full shape of input data
    full_shape = infer_shape(data)
    if full_shape is None:
        raise ValueError("cannot infer shape of input data")
    dimension = len(full_shape)
    full_shape = tuple(full_shape)
    
    # Default step is all ones
    if step is None:
        step = tuple(1 for _ in range(dimension))
    else:
        step = tuple(step)
        if len(step) != dimension:
            raise ValueError(f"step length {len(step)} does not match data dimension {dimension}")
        for i in range(dimension):
            if step[i] <= 0:
                raise ValueError(f"step[{i}] = {step[i]} must be positive")
    
    # Default shape is full shape (adjusted for step) if not provided
    if shape is None:
        shape = tuple((full_shape[i] + step[i] - 1) // step[i] for i in range(dimension))
    else:
        shape = tuple(shape)
        if len(shape) != dimension:
            raise ValueError(f"shape length {len(shape)} does not match data dimension {dimension}")
        for i in range(dimension):
            if shape[i] < 0:
                raise ValueError(f"shape[{i}] = {shape[i]} cannot be negative")
    
    # Default start is all zeros if not provided
    if start is None:
        start = tuple(0 for _ in range(dimension))
    else:
        start = tuple(start)
        if len(start) != dimension:
            raise ValueError(f"start length {len(start)} does not match data dimension {dimension}")
        for i in range(dimension):
            if start[i] < 0:
                raise ValueError(f"start[{i}] = {start[i]} cannot be negative")
            if shape[i] > 0 and start[i] + (shape[i] - 1) * step[i] >= full_shape[i]:
                raise ValueError(f"start[{i}] + (shape[{i}]-1)*step[{i}] = {start[i] + (shape[i] - 1) * step[i]} "
                                 f"out of bounds for dimension size {full_shape[i]}")
    
    # Fast path: PyBuffer protocol with contiguous data and step=1
    if hasattr(data, '__buffer__') and all(s == 1 for s in step):
        try:
            mv = memoryview(data)
            if mv.ndim == dimension and mv.format == 'd':
                total = 1
                for s in shape:
                    total *= s
                vector = [0.0] * total
                
                num_list = [0] * dimension
                pos = 0
                while True:
                    idx = tuple(start[i] + num_list[i] for i in range(dimension))
                    vector[pos] = mv[idx]
                    pos += 1
                    dim = dimension - 1
                    while dim >= 0:
                        num_list[dim] += 1
                        if num_list[dim] < shape[dim]:
                            break
                        num_list[dim] = 0
                        dim -= 1
                    if dim < 0:
                        break
                
                return vector_map_as_tensor(vector=vector, shape=shape, start=0)
        except (TypeError, ValueError):
            pass
    
    # General path: iterate using carry mechanism
    total_elements = 1
    for s in shape:
        total_elements *= s
    vector = [0.0] * total_elements
    
    num_list = [0] * dimension
    pos = 0
    while True:
        # Calculate multi-dim index: start[i] + num_list[i] * step[i]
        idx = tuple(start[i] + num_list[i] * step[i] for i in range(dimension))
        vector[pos] = get_item(data, idx)
        pos += 1
        # Increment with carry
        dim = dimension - 1
        while dim >= 0:
            num_list[dim] += 1
            if num_list[dim] < shape[dim]:
                break
            num_list[dim] = 0
            dim -= 1
        if dim < 0:
            break
    
    # Create tensor
    return vector_map_as_tensor(vector=vector, shape=shape, start=0)

#------------------ load_data (none) --------------------------
_numeric_formats = frozenset(("b","B","h","H","i","I","l","L","q","Q","n","N","f","d","e","g"))

def _load_data_shape(data):
    """infer_shape with a BufferError fallback to pure length detection:
    containers that declare but cannot export the buffer protocol must not
    break shape inference."""
    try:
        return infer_shape(data)
    except BufferError:
        shape = []
        temp = data
        while True:
            try:
                n = len(temp)
                shape.append(n)
                if n == 0:
                    break
                temp = temp[0]
            except (TypeError, IndexError, AttributeError):
                break
        return tuple(shape) if shape else None

def load_data(source, target, *,
              source_start=None, source_step=None, shape=None,
              target_start=None, target_step=None):
    """
    Load a sub-region of source data into target data, each side having
    its own start position and step size.

    Args:
        source: data to load from (nested list, buffer-like, or tensor)
        target: data to load into (nested list or writable buffer-like tensor)
        source_start: tuple of int, first read index per dimension (default: zeros)
        source_step:  tuple of int, read step per dimension (default: ones)
        shape:        tuple of int, sampled region size per dimension,
                      used as the upper bound for both sides
                      (default: full sampled shape of source with given step)
        target_start: tuple of int, first write index per dimension (default: zeros)
        target_step:  tuple of int, write step per dimension (default: ones)

    The effective element count per dimension is
        min(shape[i],
            available source elements from source_start with source_step,
            available target elements from target_start with target_step),
    so out-of-bounds requests are silently clipped on either side.

    Both sides try the PyBuffer protocol first (independently, with a
    single probing access); any failure falls back to the generic
    get_item / set_item loop. Iterative implementation, never recursive;
    single-threaded, no locking.

    Returns the total number of elements actually copied (0 when the
    effective region is empty).
    """
    source_shape = _load_data_shape(source)
    if source_shape is None:
        raise ValueError("cannot infer shape of source data")
    target_shape = _load_data_shape(target)
    if target_shape is None:
        raise ValueError("cannot infer shape of target data")
    dimension = len(source_shape)
    if len(target_shape) != dimension:
        raise ValueError(f"dimension {len(target_shape)} of target does not "
                         f"match dimension {dimension} of source")

    if source_step is None:
        source_step = tuple(1 for _ in range(dimension))
    else:
        source_step = tuple(source_step)
        if len(source_step) != dimension:
            raise ValueError(f"source_step length {len(source_step)} does not match "
                             f"data dimension {dimension}")
        for i in range(dimension):
            if source_step[i] <= 0:
                raise ValueError(f"source_step[{i}] = {source_step[i]} must be positive")

    if shape is None:
        shape = tuple((source_shape[i] + source_step[i] - 1) // source_step[i]
                      for i in range(dimension))
    else:
        shape = tuple(shape)
        if len(shape) != dimension:
            raise ValueError(f"shape length {len(shape)} does not match "
                             f"data dimension {dimension}")
        for i in range(dimension):
            if shape[i] < 0:
                raise ValueError(f"shape[{i}] = {shape[i]} cannot be negative")

    if source_start is None:
        source_start = tuple(0 for _ in range(dimension))
    else:
        source_start = tuple(source_start)
        if len(source_start) != dimension:
            raise ValueError(f"source_start length {len(source_start)} does not match "
                             f"data dimension {dimension}")
        for i in range(dimension):
            if source_start[i] < 0:
                raise ValueError(f"source_start[{i}] = {source_start[i]} cannot be negative")

    if target_start is None:
        target_start = tuple(0 for _ in range(dimension))
    else:
        target_start = tuple(target_start)
        if len(target_start) != dimension:
            raise ValueError(f"target_start length {len(target_start)} does not match "
                             f"data dimension {dimension}")
        for i in range(dimension):
            if target_start[i] < 0:
                raise ValueError(f"target_start[{i}] = {target_start[i]} cannot be negative")

    if target_step is None:
        target_step = tuple(1 for _ in range(dimension))
    else:
        target_step = tuple(target_step)
        if len(target_step) != dimension:
            raise ValueError(f"target_step length {len(target_step)} does not match "
                             f"data dimension {dimension}")
        for i in range(dimension):
            if target_step[i] <= 0:
                raise ValueError(f"target_step[{i}] = {target_step[i]} must be positive")

    effective = []
    total = 1
    for i in range(dimension):
        if source_start[i] < source_shape[i]:
            avail_src = (source_shape[i] - source_start[i] + source_step[i] - 1) // source_step[i]
        else:
            avail_src = 0
        if target_start[i] < target_shape[i]:
            avail_tgt = (target_shape[i] - target_start[i] + target_step[i] - 1) // target_step[i]
        else:
            avail_tgt = 0
        n = min(shape[i], avail_src, avail_tgt)
        effective.append(n)
        total *= n
    if total == 0:
        return 0

    probe_src = source_start
    probe_tgt = target_start

    buffer_read = False
    mv_src = None
    if hasattr(source, "__buffer__") or isinstance(source, (bytes, bytearray, memoryview)):
        try:
            view = memoryview(source)
            if view.ndim == dimension and view.format in _numeric_formats:
                view[probe_src]
                mv_src = view
                buffer_read = True
        except (TypeError, ValueError, IndexError, NotImplementedError, BufferError):
            pass

    buffer_write = False
    mv_tgt = None
    if hasattr(target, "__buffer__") or isinstance(target, (bytearray, memoryview)):
        try:
            view = memoryview(target)
            if (not view.readonly and view.ndim == dimension
                    and view.format in _numeric_formats):
                if buffer_read:
                    first_value = mv_src[probe_src]
                else:
                    first_value = get_item(source, probe_src)
                view[probe_tgt] = first_value
                # Verify the write persisted: re-export the buffer and read
                # back. Containers that export a fresh snapshot per call
                # (e.g. pure-Python tensor __buffer__) must fall back to
                # the generic set_item path.
                view[probe_tgt]
                recheck = memoryview(target)
                if recheck[probe_tgt] == first_value:
                    mv_tgt = view
                    buffer_write = True
        except (TypeError, ValueError, IndexError, OverflowError,
                NotImplementedError, BufferError):
            pass

    if buffer_read:
        read = mv_src.__getitem__
    else:
        read = lambda idx: get_item(source, idx)
    if buffer_write:
        write = mv_tgt.__setitem__
    else:
        write = lambda idx, value: set_item(target, idx, value)

    def copy_loop(read_fn, write_fn):
        num_list = [0] * dimension
        while True:
            src_idx = tuple(source_start[i] + num_list[i] * source_step[i]
                            for i in range(dimension))
            tgt_idx = tuple(target_start[i] + num_list[i] * target_step[i]
                            for i in range(dimension))
            write_fn(tgt_idx, read_fn(src_idx))
            dim = dimension - 1
            while dim >= 0:
                num_list[dim] += 1
                if num_list[dim] < effective[dim]:
                    break
                num_list[dim] = 0
                dim -= 1
            if dim < 0:
                break

    try:
        copy_loop(read, write)
    except Exception:
        copy_loop(lambda idx: get_item(source, idx),
                  lambda idx, value: set_item(target, idx, value))
    return total

def get_item(obj, index):
    """Get item from nested list with multi-dimensional index."""
    if hasattr(obj, "__get_item__"):
        return obj.__get_item__(*index)
    temp = obj
    for i in index:
        temp = temp[i]
    return temp

def set_item(obj, index, value):
    """Set item in nested list with multi-dimensional index."""
    if hasattr(obj, "__set_item__"):
        obj.__set_item__(index, value)
        return
    temp = obj
    *indexp, endp = index
    for p in indexp:
        temp = temp[p]
    temp[endp] = value

# ---------- custom types (matching Python core) ----------
class func_name_space:
    __slots__ = ("output", "output_start", "output_step", "window_size", "kernel",
                 "linear", "start", "end", "d", "step", "algorithm", "num",
                 "start_callback", "end_callback", "iter_a_callback", "iter_b_callback",
                 "global_error_callback", "local_error_callback", "return_callback",
                 "_extra")
    def __init__(self, *arg, **kwarg):
        self._extra = {}
        for key, value in kwarg.items():
            setattr(self, key, value)
    # The compiled pydll backend stores arbitrary attributes in a dict
    # (tp_getattro/tp_setattro); replicate that protocol here: the fixed
    # callback fields stay in __slots__, any other name goes to _extra.
    def __setattr__(self, name, value):
        if name in self.__class__.__slots__:
            object.__setattr__(self, name, value)
        else:
            self._extra[name] = value
    def __getattr__(self, name):
        try:
            return self._extra[name]
        except KeyError:
            raise AttributeError(name) from None
    def __delattr__(self, name):
        if name in self._extra:
            del self._extra[name]
        else:
            object.__delattr__(self, name)

class default_contain:
    __slots__ = ("default", "deep", "default_dict", "leng")
    def __init__(self, default, default_dict=None):
        self.default, self.default_dict = default, (default_dict if default_dict else {})
    def __len__(self):
        return 1
    def __getitem__(self, index):
        return self.default_dict.get(index, self.default)
    def __contains__(self, index):
        # every key is "contained" because lookup always returns a value
        # (same contract as the compiled backend's sq_contains)
        return True
    def __repr__(self):
        return "<default_contain: default=%r>" % (self.default,)

class vector_map_as_tensor:
    __slots__ = ("vector", "shape", "strides", "start", "offset", "start_offset", "step_offset")
    def __init__(self, *, vector=(1,), shape=(1,), start=0, strides=None, offset=0, start_offset=None, step_offset=None):
        self.shape = tuple(shape)
        ndim = len(self.shape)

        # Auto-create the default (zero-filled) flat vector when none is
        # provided: vector=None + shape= yields a list of zeros sized to the
        # shape, matching create_void_list semantics (the C backends use
        # their native zero-filled array instead).
        if vector is None:
            total = 1
            for s in self.shape:
                total *= s
            vector = [0.0] * total
        self.vector = vector
        self.start = start
        self.offset = offset
        
        # Precompute strides if not provided (C-order contiguous)
        if strides is None:
            if ndim == 0:
                strides = ()
            else:
                strides = [1] * ndim
                for i in range(ndim - 2, -1, -1):
                    strides[i] = strides[i+1] * self.shape[i+1]
        self.strides = tuple(strides)
        
        # Initialize per-dimension offsets
        if start_offset is None:
            self.start_offset = tuple(0 for _ in range(ndim))
        else:
            self.start_offset = tuple(start_offset)
        if step_offset is None:
            self.step_offset = tuple(1 for _ in range(ndim))
        else:
            self.step_offset = tuple(step_offset)

    def __shape__(self):
        """Shape protocol method for fast infer_shape. Can be overridden by subclasses."""
        return self.shape

    @property
    def dimension(self):
        return len(self.shape)

    @property
    def tensor_size(self):
        """Backward compatible alias for shape."""
        return self.shape

    def __repr__(self):
        return f"<vector_map_as_tensor: dim={len(self.shape)}, shape={self.shape}, start={self.start}, offset={self.offset}>"

    def __buffer__(self, flags):
        """PEP 688 buffer export: memoryview(self) exposes the tensor as a
        READ-ONLY contiguous C-order double snapshot.  Non-contiguous views
        are materialized into a fresh copy, keeping the export semantics
        simple and portable.  The Python backends are list-backed and cannot
        offer a stable writable C buffer, so the snapshot is exported
        read-only - same contract as numpy's frombuffer of immutable input
        (writes raise TypeError instead of being silently lost).
        """
        total = 1
        for s in self.shape:
            total *= s
        data = bytearray(total * 8)
        view = memoryview(data).cast('d', (total,))
        pos = 0
        for flat in self._iter_flat():
            view[pos] = self.vector[flat]
            pos += 1
        try:
            if self.shape:
                return memoryview(data).cast('d', self.shape).toreadonly()
        except (TypeError, ValueError):
            pass
        return memoryview(data).cast('d', (total,))

    def _flat_index(self, indices):
        """Compute flat index from per-dimension indices.
        
        Handles negative indices properly. Iterative implementation, no recursion.
        """
        idx = self.start + self.offset
        for i, ind in enumerate(indices):
            dim_len = self.shape[i]
            # Handle negative indices
            if ind < 0:
                ind += dim_len
            idx += self.strides[i] * (self.start_offset[i] + ind * self.step_offset[i])
        return idx

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        
        new_shape = []
        new_strides = []
        new_start_offset = []
        new_step_offset = []
        new_offset = self.offset
        
        for i, idx in enumerate(index):
            if isinstance(idx, int):
                # Integer index: dimension collapses, add to offset
                if idx < 0:
                    idx += self.shape[i]
                if idx < 0 or idx >= self.shape[i]:
                    raise IndexError(f"index {idx} out of bounds for axis {i} with size {self.shape[i]}")
                new_offset += self.strides[i] * (self.start_offset[i] + idx * self.step_offset[i])
            elif isinstance(idx, slice):
                # Slice: dimension remains, strides unchanged, update offsets
                length = self.shape[i]
                slice_start, slice_stop, slice_step = idx.indices(length)
                if slice_step == 0:
                    raise ValueError("slice step cannot be zero")
                new_size = max(0, (slice_stop - slice_start + (slice_step - (1 if slice_step > 0 else -1))) // slice_step)
                new_shape.append(new_size)
                # Strides remain original strides, do NOT multiply by step!
                new_strides.append(self.strides[i])
                # New start offset: original start + slice_start * original step
                new_start_offset.append(self.start_offset[i] + slice_start * self.step_offset[i])
                # New step offset: original step * slice step
                new_step_offset.append(self.step_offset[i] * slice_step)
            else:
                raise TypeError(f"Invalid index type: {type(idx)}")
        
        # Add remaining unindexed dimensions
        for i in range(len(index), len(self.shape)):
            new_shape.append(self.shape[i])
            new_strides.append(self.strides[i])
            new_start_offset.append(self.start_offset[i])
            new_step_offset.append(self.step_offset[i])
        
        # All dimensions indexed: return scalar
        if len(new_shape) == 0:
            return self.vector[self.start + new_offset]
        
        # Return new view
        return self.__class__(
            vector=self.vector,
            shape=tuple(new_shape),
            start=self.start,
            offset=new_offset,
            strides=tuple(new_strides),
            start_offset=tuple(new_start_offset),
            step_offset=tuple(new_step_offset)
        )

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            key = (key,)
        
        target = self[key]
        if not isinstance(target, vector_map_as_tensor):
            self.vector[self._flat_index(key)] = value
            return
        
        indices = list(target._iter_flat())
        total = len(indices)
        
        if isinstance(value, (int, float)):
            for idx in indices:
                self.vector[idx] = value
            return
        
        if isinstance(value, (list, tuple)):
            if len(value) != total:
                raise ValueError(f"expected {total} values, got {len(value)}")
            for i, idx in enumerate(indices):
                self.vector[idx] = value[i]
            return
        
        if isinstance(value, vector_map_as_tensor):
            src_indices = list(value._iter_flat())
            if len(src_indices) != total:
                raise ValueError(f"cannot assign {len(src_indices)} values to {total} elements")
            for i in range(total):
                self.vector[indices[i]] = value.vector[src_indices[i]]
            return
        
        if hasattr(value, '__buffer__'):
            mv = memoryview(value)
            try:
                if mv.format in ('d',):
                    buf = mv
                elif mv.format in ('f', 'i', 'l', 'q', 'b', 'B', 'h', 'H'):
                    buf = mv
                else:
                    buf = mv.cast('d')
                if len(buf) != total:
                    raise ValueError(f"buffer length {len(buf)} does not match {total} elements")
                for i, idx in enumerate(indices):
                    self.vector[idx] = buf[i]
                return
            except (TypeError, ValueError):
                pass
        
        raise TypeError("value must be scalar, sequence, Vector, or buffer-like object")

    def __get_item__(self, *indexs):
        if len(indexs) != len(self.shape):
            raise IndexError(f"expected {len(self.shape)} indices, got {len(indexs)}")
        return self.vector[self._flat_index(indexs)]

    def __set_item__(self, indexs, value):
        if len(indexs) != len(self.shape):
            raise IndexError(f"expected {len(self.shape)} indices, got {len(indexs)}")
        self.vector[self._flat_index(indexs)] = value

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        if len(self.shape) == 0:
            return 0
        return self.shape[0]

    def _iter_flat(self):
        """Iterate over all flat indices in this view (handles steps correctly, iterative no recursion)."""
        ndim = len(self.shape)
        if ndim == 0:
            yield self.start + self.offset
            return
        # Use carry-based iteration, no recursion
        num_list = [0] * ndim
        while True:
            # Compute current flat index
            idx = self.start + self.offset
            for i in range(ndim):
                idx += self.strides[i] * (self.start_offset[i] + num_list[i] * self.step_offset[i])
            yield idx
            # Increment with carry
            dim = ndim - 1
            while dim >= 0:
                num_list[dim] += 1
                if num_list[dim] < self.shape[dim]:
                    break
                num_list[dim] = 0
                dim -= 1
            if dim < 0:
                break

    def _check_shape(self, other):
        if not isinstance(other, vector_map_as_tensor):
            raise TypeError("It can not compute with other type.")
        if self.shape != other.shape:
            raise ValueError("the shape of two tensors are not same.")

    def __add__(self, other):
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            indices_a = list(self._iter_flat())
            indices_b = list(other._iter_flat())
            new_vec = [0.0] * len(indices_a)
            for i in range(len(indices_a)):
                new_vec[i] = self.vector[indices_a[i]] + other.vector[indices_b[i]]
            return self.__class__(vector=new_vec, shape=self.shape, start=0)
        elif isinstance(other, (int, float)):
            indices = list(self._iter_flat())
            new_vec = [0.0] * len(indices)
            for i, idx in enumerate(indices):
                new_vec[i] = self.vector[idx] + other
            return self.__class__(vector=new_vec, shape=self.shape, start=0)
        else:
            raise TypeError("It can not compute with other type.")
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            indices_a = list(self._iter_flat())
            indices_b = list(other._iter_flat())
            new_vec = [0.0] * len(indices_a)
            for i in range(len(indices_a)):
                new_vec[i] = self.vector[indices_a[i]] - other.vector[indices_b[i]]
            return self.__class__(vector=new_vec, shape=self.shape, start=0)
        elif isinstance(other, (int, float)):
            indices = list(self._iter_flat())
            new_vec = [0.0] * len(indices)
            for i, idx in enumerate(indices):
                new_vec[i] = self.vector[idx] - other
            return self.__class__(vector=new_vec, shape=self.shape, start=0)
        else:
            raise TypeError("It can not compute with other type.")
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            indices = list(self._iter_flat())
            new_vec = [0.0] * len(indices)
            for i, idx in enumerate(indices):
                new_vec[i] = other - self.vector[idx]
            return self.__class__(vector=new_vec, shape=self.shape, start=0)
        raise TypeError("It can not compute with other type.")

    def __mul__(self, other):
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            indices_a = list(self._iter_flat())
            indices_b = list(other._iter_flat())
            new_vec = [0.0] * len(indices_a)
            for i in range(len(indices_a)):
                new_vec[i] = self.vector[indices_a[i]] * other.vector[indices_b[i]]
            return self.__class__(vector=new_vec, shape=self.shape, start=0)
        elif isinstance(other, (int, float)):
            indices = list(self._iter_flat())
            new_vec = [0.0] * len(indices)
            for i, idx in enumerate(indices):
                new_vec[i] = self.vector[idx] * other
            return self.__class__(vector=new_vec, shape=self.shape, start=0)
        else:
            raise TypeError("It can not compute with other type.")
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            indices_a = list(self._iter_flat())
            indices_b = list(other._iter_flat())
            new_vec = [0.0] * len(indices_a)
            for i in range(len(indices_a)):
                b = other.vector[indices_b[i]]
                if b == 0:
                    raise ZeroDivisionError("division by zero")
                new_vec[i] = self.vector[indices_a[i]] / b
            return self.__class__(vector=new_vec, shape=self.shape, start=0)
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("division by zero")
            indices = list(self._iter_flat())
            new_vec = [0.0] * len(indices)
            for i, idx in enumerate(indices):
                new_vec[i] = self.vector[idx] / other
            return self.__class__(vector=new_vec, shape=self.shape, start=0)
        else:
            raise TypeError("It can not be used with other type.")
    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            indices = list(self._iter_flat())
            new_vec = [0.0] * len(indices)
            for i, idx in enumerate(indices):
                b = self.vector[idx]
                if b == 0:
                    raise ZeroDivisionError("division by zero")
                new_vec[i] = other / b
            return self.__class__(vector=new_vec, shape=self.shape, start=0)
        raise TypeError("It can not be used with other type.")

    def __pow__(self, other):
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            indices_a = list(self._iter_flat())
            indices_b = list(other._iter_flat())
            new_vec = [0.0] * len(indices_a)
            for i in range(len(indices_a)):
                new_vec[i] = self.vector[indices_a[i]] ** other.vector[indices_b[i]]
            return self.__class__(vector=new_vec, shape=self.shape, start=0)
        elif isinstance(other, (int, float)):
            indices = list(self._iter_flat())
            new_vec = [0.0] * len(indices)
            for i, idx in enumerate(indices):
                new_vec[i] = self.vector[idx] ** other
            return self.__class__(vector=new_vec, shape=self.shape, start=0)
        raise TypeError("unsupported operand type(s) for **")

    def __iadd__(self, other):
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            for a_idx, b_idx in zip(self._iter_flat(), other._iter_flat()):
                self.vector[a_idx] += other.vector[b_idx]
        elif isinstance(other, (int, float)):
            for idx in self._iter_flat():
                self.vector[idx] += other
        else:
            raise TypeError("It can not compute with other type.")
        return self

    def __isub__(self, other):
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            for a_idx, b_idx in zip(self._iter_flat(), other._iter_flat()):
                self.vector[a_idx] -= other.vector[b_idx]
        elif isinstance(other, (int, float)):
            for idx in self._iter_flat():
                self.vector[idx] -= other
        else:
            raise TypeError("It can not compute with other type.")
        return self

    def __imul__(self, other):
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            for a_idx, b_idx in zip(self._iter_flat(), other._iter_flat()):
                self.vector[a_idx] *= other.vector[b_idx]
        elif isinstance(other, (int, float)):
            for idx in self._iter_flat():
                self.vector[idx] *= other
        else:
            raise TypeError("It can not be used with other type.")
        return self

    def __itruediv__(self, other):
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            for a_idx, b_idx in zip(self._iter_flat(), other._iter_flat()):
                b = other.vector[b_idx]
                if b == 0:
                    raise ZeroDivisionError("division by zero")
                self.vector[a_idx] /= b
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("division by zero")
            for idx in self._iter_flat():
                self.vector[idx] /= other
        else:
            raise TypeError("It can not be used with other type.")
        return self

    def __ipow__(self, other):
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            for a_idx, b_idx in zip(self._iter_flat(), other._iter_flat()):
                self.vector[a_idx] **= other.vector[b_idx]
        elif isinstance(other, (int, float)):
            for idx in self._iter_flat():
                self.vector[idx] **= other
        else:
            raise TypeError("unsupported operand type(s) for **=")
        return self

    def __neg__(self):
        indices = list(self._iter_flat())
        new_vec = [0.0] * len(indices)
        for i, idx in enumerate(indices):
            new_vec[i] = -self.vector[idx]
        return self.__class__(vector=new_vec, shape=self.shape, start=0)

    def __pos__(self):
        indices = list(self._iter_flat())
        new_vec = [0.0] * len(indices)
        for i, idx in enumerate(indices):
            new_vec[i] = self.vector[idx]
        return self.__class__(vector=new_vec, shape=self.shape, start=0)

    def __abs__(self):
        square_sum = 0.0
        for idx in self._iter_flat():
            val = self.vector[idx]
            square_sum += val * val
        return sqrt(square_sum)

    def mean(self):
        count = 0
        mean = 0.0
        for idx in self._iter_flat():
            val = self.vector[idx]
            count += 1
            delta = val - mean
            mean += delta / count
        if count == 0:
            return None
        return mean

    def variance(self):
        count = 0
        mean = 0.0
        M2 = 0.0
        for idx in self._iter_flat():
            val = self.vector[idx]
            count += 1
            delta = val - mean
            mean += delta / count
            delta2 = val - mean
            M2 += delta * delta2
        if count == 0:
            return None
        return M2 / count


# ---------- constants and utility functions (matching pure Python API) ----------
NaN = float("nan")

def vector_chain_compute(A):
    """
    Chain vector computation utility, matches pure Python implementation.
    Returns (compute, fix, get) closures for chained dot product operations.
    """
    a = A
    def compute(vector):
        nonlocal a
        leng = len(a)
        return tuple(sum(m*n for m,n in zip(vector, a[i])) for i in range(leng))
    def fix(new):
        nonlocal a
        a = new
    def get():
        return a
    return compute, fix, get

# Private algorithm dictionary (matches pure Python backend)
private_dict = {
    "_cos": _cos,
    "_mod": _mod,
    "_cosmod": _cosmod,
    "_default_algorithm": _default_algorithm
}

# ---------- expose the public API ----------
__all__ = [
    'NaN', 'sqrt', '_cos', '_mod', '_cosmod',
    'cos_comparison_passive', 'cos_comparison_passive_1d', 'cos_comparison_passive_2d', 'cos_comparison_passive_3d', 'cos_comparison_passive_4d',
    'cos_comparison_active', 'cos_comparison_active_1d', 'cos_comparison_active_2d', 'cos_comparison_active_3d', 'cos_comparison_active_4d',
    'cos', 'cos_1d', 'cos_2d', 'cos_3d', 'cos_4d',
    'mean_local', 'mean_local_1d', 'mean_local_2d', 'mean_local_3d', 'mean_local_4d',
    'local_variance', 'local_variance_1d', 'local_variance_2d', 'local_variance_3d', 'local_variance_4d',
'data_filter', 'data_mapping', 'threshold_filter', 'threshold_map',
    'multiple_chain', 'add_chain', 'no_done', 'create_void_list', 'load_as_default_data', 'load_data', 'infer_shape', 'get_item', 'set_item',
    'vector_chain_compute',
    'vector_map_as_tensor', 'func_name_space', 'default_contain',
    'private_dict'
]
