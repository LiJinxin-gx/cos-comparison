 # it provides basic functions of cos_comparsion module, an AI algorithm set
#
# core : information is produced by local comparison in raw data

#--------------------- import ------------------------------
from math import sqrt

#-------------------- constant -----------------------------
NaN = float("nan")

#------------------ tool module -----------------       
def multiple_chain(iterable, base=1):
    """
    Use base to multiple element of iterable object in turn.
    """
    for m in iterable:
        base = base * m
    return base

def add_chain(iterable, base=0):
    """
    Use base to add element of iterable object in turn.
    """
    for m in iterable:
        base = base + m
    return base

def vector_chain_compute(A):
    a=A
    def compute(vector):
        nonlocal a
        leng=len(a)
        return tuple( (sum((m*n for m,n in zip(vector,a[i]))) for i in range(leng)) )
    def fix(new):
        nonlocal a
        a=new
    def get():
        return a
    return compute,fix,get

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
    """
    create a default void list.
    """
    length_list = tuple(length_list)
    total = 1
    for s in length_list:
        total *= s
    return vector_map_as_tensor(
        vector=[default for _ in range(total)],
        shape=length_list
    )

def load_as_default_data(data, start=None, shape=None, step=None):
    """
    Load data as a default data type (vector_map_as_tensor).
    
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
        # Build slice tuple
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
            # Check bounds: start + (shape-1)*step < full_shape
            if shape[i] > 0 and start[i] + (shape[i] - 1) * step[i] >= full_shape[i]:
                raise ValueError(f"start[{i}] + (shape[{i}]-1)*step[{i}] = {start[i] + (shape[i] - 1) * step[i]} "
                                 f"out of bounds for dimension size {full_shape[i]}")
    
    # Fast path: PyBuffer protocol with contiguous data and step=1
    if hasattr(data, '__buffer__') and all(s == 1 for s in step):
        try:
            mv = memoryview(data)
            if mv.ndim == dimension and mv.format == 'd':
                # Double precision, contiguous - can directly copy
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

def get_item(object, index):
    if hasattr(object,"__get_item__"):
        return object.__get_item__(*index) #allow to reload
    temp = object
    for p in index:
        temp = temp[p]
    return temp

def set_item(object,index,value):
    if hasattr(object,"__set_item__"):
        object.__set_item__(index,value)
    temp = object
    *indexp,endp=index
    for p in indexp:
        temp = temp[p]
    temp[endp] = value

def no_done(*arg,**kwarg):
    pass

#------------------- class support --------------------------
#     --------------- type support -----------------
class vector_map_as_tensor:
    __slots__ = ("vector", "shape", "strides", "start", "offset", "start_offset", "step_offset")
    def __init__(self, *, vector=(1,), shape=(1,), start=0, strides=None, offset=0, start_offset=None, step_offset=None):
        self.vector = vector
        self.start = start
        self.offset = offset
        
        self.shape = tuple(shape)
        ndim = len(self.shape)
        
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

#    ---------containers----------
class func_name_space:
    __slots__ = ("output", "output_start", "output_step", "window_size", "kernel",
                 "linear", "start", "end", "d", "step", "algorithm", "num")
    def __init__(self, *arg, **kwarg):
        for key, value in kwarg.items():
            setattr(self, key, value)

class default_contain:
    __slots__ = ("default", "deep", "default_dict", "leng")
    def __init__(self, default, default_dict=None):
        self.default, self.default_dict = default, (default_dict if default_dict else {})
    def __len__(self)->int:
        return 1
    def __getitem__(self, index):
        return self.default_dict.get(index, self.default)

#----------------- private module ---------------------------
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

#-------------------- core ----------------------------------
#    ------------------ passive mode ------------------------
def cos_comparison_passive(data,
                           *arg,
                           window_size=None,
                           w1=1, w2=1,
                           b1=0, b2=0,
                           start=None, end=None,
                           step=None, d=None,
                           algorithm=_default_algorithm,
                           output=None,
                           output_start=None, output_step=None,
                           start_callback=None,
                           end_callback=None,
                           global_error_callback=None,
                           local_error_callback=None,
                           return_callback=lambda output, name: output,
                           **kwargs):
    if hasattr(data, "__cos_comparison_passive__"):
        dicts = locals()
        return data.__cos_comparison_passive__(data, *arg, **dicts)  # allow to reload.

    # Use infer_shape for shape detection (unified, multi-priority)
    shape = infer_shape(data)
    if shape is None:
        raise ValueError("cannot infer shape of input data")
    length = list(shape)
    dimension = len(length)

    start = start if start is not None else tuple([0 for _ in range(dimension)])
    end = end if end is not None else tuple(length)
    step = step if step is not None else tuple([1 for _ in range(dimension)])
    d = d if d is not None else tuple([1] + [0 for _ in range(dimension - 1)])
    window_size = window_size if window_size is not None else tuple([1 for _ in range(dimension)])
    num = [0 for _ in range(dimension)]
    for i in range(dimension):
        step_effective = end[i] - start[i] - window_size[i] - d[i]
        if step_effective >= 0:
            num[i] = (step_effective // step[i]) + 1
        else:
            raise ValueError("effectless args.")

    output_start = output_start if output_start is not None else tuple([0 for _ in range(dimension)])
    output_step = output_step if output_step is not None else tuple([1 for _ in range(dimension)])
    output = output if output is not None else create_void_list(
        ((n - 1) * s + 1 for n, s in zip(num, output_step))
    )

    name = func_name_space(
        output=output,
        output_start=output_start, output_step=output_step,
        window_size=window_size,
        linear=(w1, w2, b1, b2),
        start=start, end=end, step=step, d=d,
        algorithm=algorithm,
        num=num
    )

    if start_callback:
        start_callback(name)

    flag = dimension  # start at innermost dimension
    num_list = [None] + [1 for _ in num]  # 1‑based indices
    main, other, mu = 0, 0, 0

    while flag:
        try:
            if flag == dimension:
                output_temp = output
                # Inner loop over window offsets
                inner_list = [None] + [1 for _ in window_size]
                inner_flag = len(window_size)
                main = 0
                other = 0
                mu = 0

                while inner_flag:
                    try:
                        if inner_flag == dimension:
                            main_place = tuple(
                                start[i] + step[i] * (num_list[i + 1] - 1) + (inner_list[i + 1] - 1)
                                for i in range(dimension)
                            )
                            other_place = tuple(
                                main_place[i] + d[i] for i in range(dimension)
                            )
                            a = w1 * get_item(data, main_place) + b1
                            b = w2 * get_item(data, other_place) + b2
                            main += a * a
                            other += b * b
                            mu += a * b

                        # Advance inner position or carry left
                        if inner_list[inner_flag] < window_size[inner_flag - 1]:
                            inner_list[inner_flag] += 1
                            inner_flag = dimension
                        else:
                            inner_list[inner_flag] = 1
                            inner_flag -= 1
                    except Exception as e:
                        if local_error_callback:
                            local_error_callback(e, name)

                # Write result to output
                output_places = tuple( ( output_start[p] + output_step[p] * (num_list[p + 1] - 1) for p in range(dimension) ) )
                set_item(output,output_places,algorithm(main, other, mu, name))

            # Advance output position or carry left
            if num_list[flag] < num[flag - 1]:
                num_list[flag] += 1
                flag = dimension
            else:
                num_list[flag] = 1
                flag -= 1
        except Exception as e:
            if global_error_callback:
                global_error_callback(e, name)

    if end_callback:
        end_callback(name)

    return return_callback(output, name)

cos_comparison_passive_1d=cos_comparison_passive
cos_comparison_passive_2d=cos_comparison_passive
cos_comparison_passive_3d=cos_comparison_passive
cos_comparison_passive_4d=cos_comparison_passive

# -------------------- active mode --------------------
def cos_comparison_active(data,
                          *arg,
                          kernel=None,
                          w1=1, w2=1,
                          b1=0, b2=0,
                          start=None, end=None,
                          step=None,
                          algorithm=_default_algorithm,
                          output=None,
                          output_start=None, output_step=None,
                          start_callback=None,
                          end_callback=None,
                          global_error_callback=None,
                          local_error_callback=None,
                          return_callback=lambda output, name: output,
                          **kwargs):
    if hasattr(data, "__cos_comparison_active__"):
        dicts = locals()
        return data.__cos_comparison_active__(data, *arg, **dicts)

    if kernel is None:
        raise ValueError("kernel must be provided for active mode")

    # Use infer_shape for data shape detection (unified, multi-priority)
    shape = infer_shape(data)
    if shape is None:
        raise ValueError("cannot infer shape of input data")
    length = list(shape)
    dimension = len(length)

    # Infer kernel shape using infer_shape as well
    kshape = infer_shape(kernel)
    if kshape is None:
        raise ValueError("cannot infer shape of kernel")
    kernel_shape = list(kshape)
    if len(kernel_shape) != dimension:
        raise ValueError(f"kernel dimension {len(kernel_shape)} does not match data dimension {dimension}")

    start = start if start is not None else tuple([0 for _ in range(dimension)])
    end = end if end is not None else tuple(length)
    step = step if step is not None else tuple([1 for _ in range(dimension)])
    # In active mode, window size is determined by kernel shape
    window_size = tuple(kernel_shape)

    num = [0 for _ in range(dimension)]
    for i in range(dimension):
        step_effective = end[i] - start[i] - window_size[i]
        if step_effective >= 0:
            num[i] = (step_effective // step[i]) + 1
        else:
            raise ValueError("effectless args.")

    output_start = output_start if output_start is not None else tuple([0 for _ in range(dimension)])
    output_step = output_step if output_step is not None else tuple([1 for _ in range(dimension)])
    output = output if output is not None else create_void_list(
        ((n - 1) * s + 1 for n, s in zip(num, output_step))
    )

    name = func_name_space(
        output=output,
        output_start=output_start, output_step=output_step,
        window_size=window_size,
        kernel=kernel,
        linear=(w1, w2, b1, b2),
        start=start, end=end, step=step,
        algorithm=algorithm,
        num=num
    )

    if start_callback:
        start_callback(name)

    flag = dimension  # start at innermost dimension
    num_list = [None] + [1 for _ in num]
    main, other, mu = 0, 0, 0

    while flag:
        try:
            if flag == dimension:
                output_temp = output
                inner_list = [None] + [1 for _ in window_size]
                inner_flag = len(window_size)
                main = 0
                other = 0
                mu = 0

                while inner_flag:
                    try:
                        if inner_flag == dimension:
                            # Data window position
                            data_place = tuple(
                                start[i] + step[i] * (num_list[i + 1] - 1) + (inner_list[i + 1] - 1)
                                for i in range(dimension)
                            )
                            # Kernel position (always starting at 0 for each dim)
                            kern_place = tuple(
                                inner_list[i + 1] - 1 for i in range(dimension)
                            )
                            a = w1 * get_item(data, data_place) + b1
                            b = w2 * get_item(kernel, kern_place) + b2
                            main += a * a
                            other += b * b
                            mu += a * b

                        # Advance inner position or carry left
                        if inner_list[inner_flag] < window_size[inner_flag - 1]:
                            inner_list[inner_flag] += 1
                            inner_flag = dimension
                        else:
                            inner_list[inner_flag] = 1
                            inner_flag -= 1
                    except Exception as e:
                        if local_error_callback:
                            local_error_callback(e, name)

                # Write result to output
                output_places = tuple( ( output_start[p] + output_step[p] * (num_list[p + 1] - 1) for p in range(dimension) ) )
                set_item(output,output_places,algorithm(main, other, mu, name))

            # Advance output position or carry left
            if num_list[flag] < num[flag - 1]:
                num_list[flag] += 1
                flag = dimension
            else:
                num_list[flag] = 1
                flag -= 1
        except Exception as e:
            if global_error_callback:
                global_error_callback(e, name)

    if end_callback:
        end_callback(name)

    return return_callback(output, name)

cos_comparison_active_1d=cos_comparison_active
cos_comparison_active_2d=cos_comparison_active
cos_comparison_active_3d=cos_comparison_active
cos_comparison_active_4d=cos_comparison_active

#-------------- cos mode -----------
def cos_1d(a,b,algorithm=_cos):
    sum_a = 0.0
    sum_b = 0.0
    sum_ab = 0.0
    try:
        if len(a) != len(b) :
            raise ValueError("the shape of two tensors are not same.")
        shape1= len(a)
    except:
        raise ValueError("the args you gave are not tensors.")
    for p1 in range(shape1):
        a_temp = a[p1]
        b_temp = b[p1]
        sum_a += a_temp*a_temp
        sum_b += b_temp*b_temp
        sum_ab += a_temp*b_temp
    return algorithm(sum_a,sum_b,sum_ab,None)

def cos_2d(a,b,algorithm=_cos):
    sum_a = 0.0
    sum_b = 0.0
    sum_ab = 0.0
    try:
        if len(a) != len(b) or len(a[0]) !=len(b[0]) :
            raise ValueError("the shape of two tensors are not same.")
        shape1= len(a)
        shape2 = len(a[0])
    except:
        raise ValueError("the args you gave are not tensors.")
    for p1 in range(shape1):
        for p2 in range(shape2):
            a_temp = a[p1][p2]
            b_temp = b[p1][p2]
            sum_a += a_temp*a_temp
            sum_b += b_temp*b_temp
            sum_ab += a_temp*b_temp
    return algorithm(sum_a,sum_b,sum_ab,None)

def cos_3d(a,b,algorithm=_cos):
    sum_a = 0.0
    sum_b = 0.0
    sum_ab = 0.0
    try:
        if len(a) != len(b) or len(a[0]) !=len(b[0]) or len(a[0][0]) != len(b[0][0]) :
            raise ValueError("the shape of two tensors are not same.")
        shape1= len(a)
        shape2 = len(a[0])
        shape3 = len(a[0][0])
    except:
        raise ValueError("the args you gave are not tensors.")
    for p1 in range(shape1):
        for p2 in range(shape2):
            for p3 in range(shape3):
                a_temp = a[p1][p2][p3]
                b_temp = b[p1][p2][p3]
                sum_a += a_temp*a_temp
                sum_b += b_temp*b_temp
                sum_ab += a_temp*b_temp
    return algorithm(sum_a,sum_b,sum_ab,None)

def cos_4d(a,b,algorithm=_cos):
    sum_a = 0.0
    sum_b = 0.0
    sum_ab = 0.0
    try:
        if len(a) != len(b) or len(a[0]) !=len(b[0]) or len(a[0][0]) != len(b[0][0]) or len(a[0][0][0]) != len(b[0][0][0]) :
            raise ValueError("the shape of two tensors are not same.")
        shape1= len(a)
        shape2 = len(a[0])
        shape3 = len(a[0][0])
        shape4 = len(a[0][0][0])
    except:
        raise ValueError("the args you gave are not tensors.")
    for p1 in range(shape1):
        for p2 in range(shape2):
            for p3 in range(shape3):
                for p4 in range(shape4):
                    a_temp = a[p1][p2][p3][p4]
                    b_temp = b[p1][p2][p3][p4]
                    sum_a += a_temp*a_temp
                    sum_b += b_temp*b_temp
                    sum_ab += a_temp*b_temp
    return algorithm(sum_a,sum_b,sum_ab,None)

def cos(a, b, algorithm=_cos):
    """
    Compute similarity between two whole tensors (nested structures).
    The tensors are flattened element‑wise, squared sums and dot product
    are accumulated, then passed to the given algorithm.
    """
    # ----- Infer shapes and verify consistency (same as passive mode) -----
    shape = []
    tmp_a, tmp_b = a, b
    dimension = 0
    while True:
        try:
            len_a, len_b = len(tmp_a), len(tmp_b)
            if len_a != len_b:
                raise ValueError("the shape of two tensors are not same.")
            shape.append(len_a)
            dimension += 1
            tmp_a, tmp_b = tmp_a[0], tmp_b[0]
        except:
            shape = tuple(shape)
            break
    if dimension == 0:
        raise ValueError("the args you gave are not tensors.")

    # ----- Traverse all indices (1‑based, same as passive mode) -----
    num_list = [None] + [1] * dimension   # current position in each dim (1‑based)
    flag = dimension                      # start at innermost dimension

    sum_a = 0.0
    sum_b = 0.0
    sum_ab = 0.0

    while flag:
        if flag == dimension:
            # Build index tuple (0‑based) for the current position
            idx = tuple(num_list[i] - 1 for i in range(1, dimension + 1))

            val_a = get_item(a, idx)
            val_b = get_item(b, idx)

            sum_a += val_a * val_a
            sum_b += val_b * val_b
            sum_ab += val_a * val_b

        # Advance position or carry left
        if num_list[flag] < shape[flag - 1]:
            num_list[flag] += 1
            flag = dimension
        else:
            num_list[flag] = 1
            flag -= 1

    return algorithm(sum_a, sum_b, sum_ab, None)


#----------------- important module ------------------
#    ----------------------- statistic mode -----------------------
def mean_local(data, *arg, local_size=None, step=None,weight=None,
               output=None, output_start=None, output_step=None, **kwarg):
    """
    Generic local mean (arbitrary N‑dim).
    Supports external container via `output`, `output_start`, `output_step`.
    """
    if local_size is None:
        local_size = default_contain(1)
    if step is None:
        step = default_contain(1)

    # Build an all‑one kernel according to the given shape
    def _build_ones(shape):
        if isinstance(shape, default_contain):
            return shape               # let active mode handle it
        if isinstance(shape, int):
            return [1.0] * shape
        if isinstance(shape, (list, tuple)):
            if len(shape) == 0:
                return 1.0
            if len(shape) == 1:
                return _build_ones(shape[0])
            # shape is (dim0, dim1, ...): create dim0 copies of the remaining shape
            return [_build_ones(shape[1:]) for _ in range(shape[0])]
        raise TypeError("local_size must be a sequence, int, or default_contain")

    kernel = weight if weight else _build_ones(local_size)

    # Compute total number of elements in the window
    if hasattr(local_size, '__len__') and not isinstance(local_size, default_contain):
        N = multiple_chain(local_size, 1)
    else:
        N = 1   # fallback for default_contain (will be overridden by kernel size)

    return cos_comparison_active(data, *arg,
                                 kernel=kernel,
                                 step=step,
                                 output=output,
                                 output_start=output_start,
                                 output_step=output_step,
                                 algorithm=lambda a, b, ab, name: ab / N,
                                 **kwarg)


def local_variance(data, *arg, local_size=None, step=None,
                   output=None, output_start=None, output_step=None, **kwarg):
    """
    Generic local variance (arbitrary N‑dim).
    Supports external container via `output`, `output_start`, `output_step`.
    """
    if local_size is None:
        local_size = default_contain(1)
    if step is None:
        step = default_contain(1)

    def _build_ones(shape):
        if isinstance(shape, default_contain):
            return shape
        if isinstance(shape, int):
            return [1.0] * shape
        if isinstance(shape, (list, tuple)):
            if len(shape) == 0:
                return 1.0
            if len(shape) == 1:
                return _build_ones(shape[0])
            # shape is (dim0, dim1, ...): create dim0 copies of the remaining shape
            return [_build_ones(shape[1:]) for _ in range(shape[0])]
        raise TypeError("local_size must be a sequence, int, or default_contain")

    kernel = _build_ones(local_size)

    if hasattr(local_size, '__len__') and not isinstance(local_size, default_contain):
        N = multiple_chain(local_size, 1)
    else:
        N = 1

    def var_alg(a, b, ab, name):
        mean = ab / N
        return a / N - mean * mean

    return cos_comparison_active(data, *arg,
                                 kernel=kernel,
                                 step=step,
                                 output=output,
                                 output_start=output_start,
                                 output_step=output_step,
                                 algorithm=var_alg,
                                 **kwarg)

mean_local_1d = mean_local
mean_local_2d = mean_local
mean_local_3d = mean_local
mean_local_4d = mean_local

local_variance_1d = local_variance
local_variance_2d = local_variance
local_variance_3d = local_variance
local_variance_4d = local_variance

# ---------- public API export list (matches other backends) ----------
__all__ = [
    'NaN', 'sqrt',
    'cos_comparison_passive', 'cos_comparison_passive_1d', 'cos_comparison_passive_2d', 'cos_comparison_passive_3d', 'cos_comparison_passive_4d',
    'cos_comparison_active', 'cos_comparison_active_1d', 'cos_comparison_active_2d', 'cos_comparison_active_3d', 'cos_comparison_active_4d',
    'cos', 'cos_1d', 'cos_2d', 'cos_3d', 'cos_4d',
    'mean_local', 'mean_local_1d', 'mean_local_2d', 'mean_local_3d', 'mean_local_4d',
    'local_variance', 'local_variance_1d', 'local_variance_2d', 'local_variance_3d', 'local_variance_4d',
    'multiple_chain', 'add_chain', 'no_done', 'create_void_list', 'load_as_default_data', 'infer_shape', 'get_item', 'set_item', '_cos', '_mod', '_cosmod',
    'vector_chain_compute',
    'vector_map_as_tensor', 'func_name_space', 'default_contain',
    'private_dict'
]

