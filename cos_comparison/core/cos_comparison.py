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
    """Multi-dimensional access with __get_item__ protocol support.

    Falls back to plain nested indexing when the object does not
    implement the optional ``__get_item__`` extension protocol.
    A scalar index is treated as a 1-D index (matches the C backend).
    """
    if not isinstance(index, tuple):
        index = (index,)
    if hasattr(obj, "__get_item__"):
        return obj.__get_item__(*index)
    temp = obj
    for p in index:
        temp = temp[p]
    return temp

def set_item(obj, index, value):
    """Multi-dimensional assignment with __set_item__ protocol support.

    The protocol method (when present) is authoritative and returns
    immediately; otherwise plain nested assignment is used. Iterative,
    never recursive; safe for arbitrary nesting depth.
    """
    if hasattr(obj, "__set_item__"):
        obj.__set_item__(index, value)
        return
    temp = obj
    *indexp, endp = index
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

#    ---------containers----------
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
    def __len__(self)->int:
        return 1
    def __getitem__(self, index):
        return self.default_dict.get(index, self.default)
    def __contains__(self, index):
        # every key is "contained" because lookup always returns a value
        # (same contract as the compiled backend's sq_contains)
        return True
    def __repr__(self):
        return "<default_contain: default=%r>" % (self.default,)

#----------------- private module ---------------------------
_cos = lambda a, b, ab, name: ab / sqrt(a * b) if a * b else (1.0 if a == b else 0.0)
_mod = lambda a, b, ab, name: 2 * sqrt(a * b) / (a + b) if a * b else (1.0 if a == b else 0.0)
_cosmod = lambda a, b, ab, name: 2 * ab / (a + b) if a * b else (1.0 if a == b else 0.0)
_convolution = lambda a, b, ab, name: ab
_default_algorithm = _cosmod
private_dict = {
    "_cos": _cos,
    "_mod": _mod,
    "_cosmod": _cosmod,
    "_convolution": _convolution,
    "_default_algorithm": _default_algorithm
}

#-------------------- core ----------------------------------
#    ------------------ passive mode ------------------------
def cos_comparison_passive(data,
                           *arg,
                           window_size=None,
                           w1=1.0, w2=1.0,
                           b1=0.0, b2=0.0,
                           start=None, end=None,
                           step=None, d=None,
                           algorithm=_default_algorithm,
                           output=None,
                           output_start=None, output_step=None,
                           start_callback=None,
                           end_callback=None,
                           iter_a_callback=None, iter_b_callback=None,
                           global_error_callback=None,
                           local_error_callback=None,
                           return_callback=None,
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
        num=num,
        start_callback=start_callback,
        end_callback=end_callback,
        iter_a_callback=iter_a_callback,
        iter_b_callback=iter_b_callback,
        global_error_callback=global_error_callback,
        local_error_callback=local_error_callback,
        return_callback=return_callback,
    )

    if start_callback:
        start_callback(name)

    flag = dimension  # start at innermost dimension
    num_list = [None] + [1 for _ in num]  # 1-based indices
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

                if iter_a_callback:
                    iter_a_callback(name)
                if iter_b_callback:
                    iter_b_callback(name)

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

    if return_callback:
        return return_callback(output, name)
    return output

cos_comparison_passive_1d=cos_comparison_passive
cos_comparison_passive_2d=cos_comparison_passive
cos_comparison_passive_3d=cos_comparison_passive
cos_comparison_passive_4d=cos_comparison_passive

# -------------------- active mode --------------------
def cos_comparison_active(data,
                          *arg,
                          kernel=None,
                          w1=1.0, w2=1.0,
                          b1=0.0, b2=0.0,
                          start=None, end=None,
                          step=None,
                          algorithm=_default_algorithm,
                          output=None,
                          output_start=None, output_step=None,
                          start_callback=None,
                          end_callback=None,
                          iter_a_callback=None, iter_b_callback=None,
                          global_error_callback=None,
                          local_error_callback=None,
                          return_callback=None,
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
        num=num,
        start_callback=start_callback,
        end_callback=end_callback,
        iter_a_callback=iter_a_callback,
        iter_b_callback=iter_b_callback,
        global_error_callback=global_error_callback,
        local_error_callback=local_error_callback,
        return_callback=return_callback,
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

                if iter_a_callback:
                    iter_a_callback(name)
                if iter_b_callback:
                    iter_b_callback(name)

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

    if return_callback:
        return return_callback(output, name)
    return output

cos_comparison_active_1d=cos_comparison_active
cos_comparison_active_2d=cos_comparison_active
cos_comparison_active_3d=cos_comparison_active
cos_comparison_active_4d=cos_comparison_active

#-------------- cos mode -----------
def cos(a, b, algorithm=_cos):
    """
    Compute similarity between two whole tensors (nested structures).
    The tensors are flattened element-wise, squared sums and dot product
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

    # ----- Traverse all indices (1-based, same as passive mode) -----
    num_list = [None] + [1] * dimension   # current position in each dim (1-based)
    flag = dimension                      # start at innermost dimension

    sum_a = 0.0
    sum_b = 0.0
    sum_ab = 0.0

    while flag:
        if flag == dimension:
            # Build index tuple (0-based) for the current position
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

cos_1d = cos
cos_2d = cos
cos_3d = cos
cos_4d = cos


#----------------- important module ------------------
#    ----------------------- statistic mode -----------------------
# Window pattern helpers for weighted local statistics.
def _build_ones(shape):
    """Build an all-one kernel with the given shape (iterative, no recursion)."""
    if isinstance(shape, default_contain):
        return shape
    if isinstance(shape, int):
        return [1.0] * shape
    shape = tuple(shape)
    if len(shape) == 0:
        return 1.0
    # reshape a flat list of 1.0s, innermost dimension first
    flat = [1.0] * multiple_chain(shape, 1)
    stack = flat
    for dim in range(len(shape) - 1, -1, -1):
        width = shape[dim]
        nxt = [stack[i:i + width] for i in range(0, len(stack), width)]
        stack = nxt
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
    matching the C backend's per-window pattern semantics.
    """
    if not shape:
        return values[0]
    n = 1
    for s in shape:
        n *= s
    if len(values) < n:
        raise ValueError("weight is too small for the local_size window")
    # build from the innermost dimension outwards
    stack = [1.0 * v for v in values[:n]]
    for dim in range(len(shape) - 1, -1, -1):
        width = shape[dim]
        nxt = []
        for start in range(0, len(stack), width):
            nxt.append(stack[start:start + width])
        stack = nxt
    return stack[0]

def mean_local(data, *arg, local_size=None, step=None, weight=None,
               output=None, output_start=None, output_step=None, **kwargs):
    """
    Generic local mean (arbitrary N-dim).
    Supports external container via `output`, `output_start`, `output_step`.
    `weight` supports N-dim nested/iterable data and is interpreted as a
    per-window pattern: the first `product(local_size)` values (row-major
    flatten) are applied to every window, matching the C backend.
    """
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

    if weight is None:
        kernel = _build_ones(local_size)
    else:
        # Same semantics as the C backend: weight is a per-window pattern
        # built from the first product(local_size) flattened values.
        kernel = _flat_to_window(_flatten_nested(weight), local_size)

    # Compute total number of elements in the window
    N = multiple_chain(local_size, 1)

    return cos_comparison_active(data, *arg,
                                 kernel=kernel,
                                 step=step,
                                 output=output,
                                 output_start=output_start,
                                 output_step=output_step,
                                 algorithm=lambda a, b, ab, name: ab / N,
                                 **kwargs)


def local_variance(data, *arg, local_size=None, step=None,
                   output=None, output_start=None, output_step=None, **kwargs):
    """
    Generic local variance (arbitrary N-D).
    Supports external container via `output`, `output_start`, `output_step`.
    """
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

    kernel = _build_ones(local_size)

    N = multiple_chain(local_size, 1)

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
                                 **kwargs)

mean_local_1d = mean_local
mean_local_2d = mean_local
mean_local_3d = mean_local
mean_local_4d = mean_local

local_variance_1d = local_variance
local_variance_2d = local_variance
local_variance_3d = local_variance
local_variance_4d = local_variance


# ---------------------------------------------------------------------------
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




def threshold_filter(data, low=None, high=None, *, inclusive=(True, True),
                     **region):
    """data_filter over the interval [low, high]: yields the positions whose
    value lies in the threshold range (endpoints per inclusive)."""
    predicate = _make_threshold_predicate(low, high, inclusive)
    return data_filter(data, predicate, **region)


def threshold_map(data, pairs, *, default_value=0.0, **region):
    """Map every element of the sampled region by iterating the (func, value)
    pairs: the first func(value) that is truthy selects its paired value;
    when none matches, default_value is used. Callback errors are silently
    skipped. Region/read/write parameters (start/shape/step/out/out_start/
    out_step) keep their usual data_mapping semantics."""
    def matcher(value):
        for func, mapped in pairs:
            try:
                if func(value):
                    return mapped
            except Exception:
                continue
        return default_value

    return data_mapping(data, matcher, **region)


# ---------- public API export list (matches other backends) ----------
__all__ = [
    'NaN', 'sqrt',
    'cos_comparison_passive', 'cos_comparison_passive_1d', 'cos_comparison_passive_2d', 'cos_comparison_passive_3d', 'cos_comparison_passive_4d',
    'cos_comparison_active', 'cos_comparison_active_1d', 'cos_comparison_active_2d', 'cos_comparison_active_3d', 'cos_comparison_active_4d',
    'cos', 'cos_1d', 'cos_2d', 'cos_3d', 'cos_4d',
    'mean_local', 'mean_local_1d', 'mean_local_2d', 'mean_local_3d', 'mean_local_4d',
    'local_variance', 'local_variance_1d', 'local_variance_2d', 'local_variance_3d', 'local_variance_4d',
    'multiple_chain', 'add_chain', 'no_done', 'create_void_list', 'load_as_default_data', 'load_data', 'infer_shape', 'get_item', 'set_item', '_cos', '_mod', '_cosmod', '_convolution',
    'data_filter', 'data_mapping', 'threshold_filter', 'threshold_map', 'threshold_judge',
    'vector_chain_compute',
    'vector_map_as_tensor', 'func_name_space', 'default_contain',
    'private_dict'
]


def threshold_judge(low=None, high=None, *, inclusive=(True, True)):
    """Return a judge function: 1 if the value lies within [low, high],
    else 0 (single-bound thresholds allowed). Companion for threshold_map's
    (func, value) pair iteration, e.g.
        threshold_map(data, [(threshold_judge(low=3, high=7), 2.0),
                             (lambda v: True, 3.0)])
    """
    predicate = _make_threshold_predicate(low, high, inclusive)
    return lambda value: 1 if predicate(value) else 0


