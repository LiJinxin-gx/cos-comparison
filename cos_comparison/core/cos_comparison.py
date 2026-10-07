"""Pure-Python reference implementation of the local-comparison core.

Information is produced by local comparison of raw data.  This module is
the reference backend (stdlib only) and mirrors every C-extension
behaviour one-to-one.  All walks are carry-based odometers (no recursion).
"""


# ---- imports ----
import numbers
import operator
from collections.abc import Mapping
from math import sqrt


# ---- public API ----
__all__ = [
    'NaN', 'sqrt',
    'cos_comparison_passive', 'cos_comparison_passive_1d', 'cos_comparison_passive_2d', 'cos_comparison_passive_3d', 'cos_comparison_passive_4d',
    'cos_comparison_active', 'cos_comparison_active_1d', 'cos_comparison_active_2d', 'cos_comparison_active_3d', 'cos_comparison_active_4d',
    'cos', 'cos_1d', 'cos_2d', 'cos_3d', 'cos_4d',
    'mean_local', 'mean_local_1d', 'mean_local_2d', 'mean_local_3d', 'mean_local_4d',
    'local_variance', 'local_variance_1d', 'local_variance_2d', 'local_variance_3d', 'local_variance_4d',
    'multiple_chain', 'add_chain', 'no_done', 'create_void_list', 'load_as_default_data', 'load_data', 'infer_shape', 'get_item', 'set_item', '_cos', '_mod', '_cosmod', '_convolution',
    'data_filter', 'data_mapping', 'elementwise', 'position_map', 'elementwise_position', 'pool', 'threshold_filter', 'threshold_map', 'threshold_judge',
    'vector_chain_compute',
    'vector_map_as_tensor', 'func_name_space', 'default_contain',
    'private_dict'
]


# ---- constants ----
NaN = float("nan")


# ---- types ----
class vector_map_as_tensor:
    """Tensor view over a flat vector (stride + offset; zero-copy slices)."""

    __slots__ = ("vector", "shape", "strides", "start", "offset", "start_offset", "step_offset")

    def __init__(self, *, vector=(1,), shape=(1,), start=0, strides=None, offset=0, start_offset=None, step_offset=None):
        self.shape = tuple(shape)
        ndim = len(self.shape)

        if vector is None:  # auto-create a zero-filled flat vector
            total = 1
            for s in self.shape:
                total *= s
            vector = [0.0] * total
        self.vector = vector
        self.start = start
        self.offset = offset

        if strides is None:  # C-order contiguous by default
            if ndim == 0:
                strides = ()
            else:
                strides = [1] * ndim
                for i in range(ndim - 2, -1, -1):
                    strides[i] = strides[i+1] * self.shape[i+1]
        self.strides = tuple(strides)

        if start_offset is None:
            self.start_offset = tuple(0 for _ in range(ndim))
        else:
            self.start_offset = tuple(start_offset)
        if step_offset is None:
            self.step_offset = tuple(1 for _ in range(ndim))
        else:
            self.step_offset = tuple(step_offset)

    def __shape__(self):
        """Shape protocol used by infer_shape (overridable by subclasses)."""
        return self.shape

    @property
    def dimension(self):
        return len(self.shape)

    @property
    def tensor_size(self):
        """Backward-compatible alias for shape."""
        return self.shape

    def __repr__(self):
        return f"<vector_map_as_tensor: dim={len(self.shape)}, shape={self.shape}, start={self.start}, offset={self.offset}>"

    def __buffer__(self, flags):
        """PEP 688 export: a read-only contiguous C-order double snapshot
        (non-contiguous views are materialized).  Writes raise TypeError."""
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
        """Flat index of per-dimension indices (negative indices allowed)."""
        idx = self.start + self.offset
        for i, ind in enumerate(indices):
            dim_len = self.shape[i]
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
                # integer index: the dimension collapses into the offset
                if idx < 0:
                    idx += self.shape[i]
                if idx < 0 or idx >= self.shape[i]:
                    raise IndexError(f"index {idx} out of bounds for axis {i} with size {self.shape[i]}")
                new_offset += self.strides[i] * (self.start_offset[i] + idx * self.step_offset[i])
            elif isinstance(idx, slice):
                # slice: keep the stride, compose offsets (strides unchanged)
                length = self.shape[i]
                slice_start, slice_stop, slice_step = idx.indices(length)
                new_size = max(0, (slice_stop - slice_start + (slice_step - (1 if slice_step > 0 else -1))) // slice_step)
                new_shape.append(new_size)
                new_strides.append(self.strides[i])
                new_start_offset.append(self.start_offset[i] + slice_start * self.step_offset[i])
                new_step_offset.append(self.step_offset[i] * slice_step)
            else:
                raise TypeError(f"Invalid index type: {type(idx)}")

        for i in range(len(index), len(self.shape)):  # remaining dims
            new_shape.append(self.shape[i])
            new_strides.append(self.strides[i])
            new_start_offset.append(self.start_offset[i])
            new_step_offset.append(self.step_offset[i])

        if len(new_shape) == 0:  # fully indexed: scalar
            return self.vector[self.start + new_offset]

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
        """Iterate the flat indices of this view (carry-based, steps honoured)."""
        ndim = len(self.shape)
        if ndim == 0:
            yield self.start + self.offset
            return
        for s in self.shape:  # empty tensor yields nothing
            if s == 0:
                return
        num_list = [0] * ndim
        while True:
            idx = self.start + self.offset
            for i in range(ndim):
                idx += self.strides[i] * (self.start_offset[i] + num_list[i] * self.step_offset[i])
            yield idx
            dim = ndim - 1
            while dim >= 0:  # carry
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

    # -- operators (shared helpers; error messages kept per operator) --
    def _binary_op(self, other, op, *, error="It can not compute with other type.", check_zero=False):
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            return self.__class__(
                vector=[op(self.vector[a], other.vector[b])
                        for a, b in zip(self._iter_flat(), other._iter_flat())],
                shape=self.shape, start=0)
        if isinstance(other, (int, float)):
            if check_zero and other == 0:
                raise ZeroDivisionError("division by zero")
            return self.__class__(
                vector=[op(self.vector[i], other) for i in self._iter_flat()],
                shape=self.shape, start=0)
        raise TypeError(error)

    def _inplace_op(self, other, op, *, error="It can not compute with other type.", check_zero=False):
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            for a, b in zip(self._iter_flat(), other._iter_flat()):
                self.vector[a] = op(self.vector[a], other.vector[b])
        elif isinstance(other, (int, float)):
            if check_zero and other == 0:
                raise ZeroDivisionError("division by zero")
            for i in self._iter_flat():
                self.vector[i] = op(self.vector[i], other)
        else:
            raise TypeError(error)
        return self

    def __add__(self, other):
        return self._binary_op(other, operator.add)

    def __radd__(self, other):
        return self._binary_op(other, operator.add)

    def __sub__(self, other):
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(
                vector=[other - self.vector[i] for i in self._iter_flat()],
                shape=self.shape, start=0)
        raise TypeError("It can not compute with other type.")

    def __mul__(self, other):
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other):
        return self._binary_op(other, operator.mul)

    def __truediv__(self, other):
        return self._binary_op(other, operator.truediv,
                               error="It can not be used with other type.", check_zero=True)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(
                vector=[operator.truediv(other, self.vector[i])
                        for i in self._iter_flat()],
                shape=self.shape, start=0)
        raise TypeError("It can not be used with other type.")

    def __pow__(self, other):
        return self._binary_op(other, operator.pow,
                               error="unsupported operand type(s) for **")

    def __iadd__(self, other):
        return self._inplace_op(other, operator.iadd)

    def __isub__(self, other):
        return self._inplace_op(other, operator.isub)

    def __imul__(self, other):
        return self._inplace_op(other, operator.imul,
                                error="It can not be used with other type.")

    def __itruediv__(self, other):
        return self._inplace_op(other, operator.itruediv,
                                error="It can not be used with other type.", check_zero=True)

    def __ipow__(self, other):
        return self._inplace_op(other, operator.ipow,
                                error="unsupported operand type(s) for **=")

    def __neg__(self):
        return self.__class__(
            vector=[-self.vector[idx] for idx in self._iter_flat()],
            shape=self.shape, start=0)

    def __pos__(self):
        return self.__class__(
            vector=[self.vector[idx] for idx in self._iter_flat()],
            shape=self.shape, start=0)

    def __abs__(self):
        square_sum = 0.0
        for idx in self._iter_flat():
            val = self.vector[idx]
            square_sum += val * val
        return sqrt(square_sum)

    def mean(self):
        """Arithmetic mean of the view (Welford); None for an empty view."""
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
        """Population variance of the view (Welford); None for an empty view."""
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


# ---- helper tools ----

# ----- chain tools
def multiple_chain(iterable, base=1):
    """Product of all elements, starting from base."""
    for m in iterable:
        base = base * m
    return base


def add_chain(iterable, base=0):
    """Sum of all elements, starting from base."""
    for m in iterable:
        base = base + m
    return base


def vector_chain_compute(A):
    """Closures over the matrix A: compute(vector) -> tuple of row dot
    products; fix(new) replaces A; get() returns A."""
    a=A
    def compute(vector):
        nonlocal a
        return tuple(sum(m * n for m, n in zip(vector, a[i]))
                     for i in range(len(a)))
    def fix(new):
        nonlocal a
        a=new
    def get():
        return a
    return compute,fix,get


def no_done(*arg,**kwarg):
    """No-op placeholder."""


# ----- shape / access protocol
def infer_shape(data):
    """Infer the shape of nested data.
    Priority: PyBuffer protocol > __shape__() method > iterative length
    detection.  Returns a shape tuple, or None when uninferable."""
    if hasattr(data, '__buffer__'):
        try:
            mv = memoryview(data)
            if mv.ndim > 0:
                return tuple(mv.shape)
        except (TypeError, ValueError):
            pass

    if hasattr(data, '__shape__') and callable(data.__shape__):
        try:
            result = data.__shape__()
            if result is not None:
                return tuple(result)
        except (TypeError, ValueError):
            pass

    shape = []
    temp = data
    while True:
        # text and mappings are values, not tensor dimensions
        if isinstance(temp, (str, bytes)) or hasattr(temp, "keys"):
            break
        try:
            n = len(temp)
        except TypeError:
            break
        shape.append(n)
        if n == 0:
            break
        try:
            temp = temp[0]
        except (IndexError, KeyError, TypeError, AttributeError):
            break

    return tuple(shape) if shape else None


def get_item(obj, index):
    """Multi-dimensional read honouring the __get_item__ protocol; a
    non-tuple index is treated as a 1-D index (matches the C backend)."""
    if not isinstance(index, tuple):
        index = (index,)
    if hasattr(obj, "__get_item__"):
        return obj.__get_item__(*index)
    temp = obj
    for p in index:
        temp = temp[p]
    return temp


def set_item(obj, index, value):
    """Multi-dimensional write honouring the __set_item__ protocol
    (authoritative when present); iterative, never recursive."""
    if hasattr(obj, "__set_item__"):
        obj.__set_item__(index, value)
        return
    temp = obj
    *indexp, endp = index
    for p in indexp:
        temp = temp[p]
    temp[endp] = value


# ----- loading / creation
def create_void_list(length_list=(1,), default=0.0):
    """New tensor of the given shape filled with default."""
    length_list = tuple(length_list)
    total = 1
    for s in length_list:
        total *= s
    return vector_map_as_tensor(
        vector=[default for _ in range(total)],
        shape=length_list
    )


def load_as_default_data(data, start=None, shape=None, step=None):
    """Flatten a (sub-)region of data into a fresh contiguous tensor;
    start/shape/step select the sampled region (defaults: origin, full
    shape, unit step).  Native slicing is used for tensor input."""
    # fast path: tensor input slices natively, then materializes a copy
    if isinstance(data, vector_map_as_tensor):
        slices = []
        dim = data.dimension
        for i in range(dim):
            s = start[i] if start is not None else None
            e = (start[i] + shape[i]) if (start is not None and shape is not None) else None
            st = step[i] if step is not None else None
            slices.append(slice(s, e, st))
        result = data[tuple(slices)]
        total = 1
        for s in result.shape:
            total *= s
        new_vector = [0.0] * total
        idx = 0
        for flat_idx in result._iter_flat():
            new_vector[idx] = result.vector[flat_idx]
            idx += 1
        return vector_map_as_tensor(vector=new_vector, shape=result.shape, start=0)

    full_shape = infer_shape(data)
    if full_shape is None:
        raise ValueError("cannot infer shape of input data")
    dimension = len(full_shape)
    full_shape = tuple(full_shape)

    if step is None:
        step = tuple(1 for _ in range(dimension))
    else:
        step = tuple(step)
        if len(step) != dimension:
            raise ValueError(f"step length {len(step)} does not match data dimension {dimension}")
        for i in range(dimension):
            if step[i] <= 0:
                raise ValueError(f"step[{i}] = {step[i]} must be positive")

    if shape is None:  # full shape adjusted for step
        shape = tuple((full_shape[i] + step[i] - 1) // step[i] for i in range(dimension))
    else:
        shape = tuple(shape)
        if len(shape) != dimension:
            raise ValueError(f"shape length {len(shape)} does not match data dimension {dimension}")
        for i in range(dimension):
            if shape[i] < 0:
                raise ValueError(f"shape[{i}] = {shape[i]} cannot be negative")

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

    # fast path: contiguous double buffer with step 1
    if hasattr(data, '__buffer__') and all(s == 1 for s in step):
        try:
            mv = memoryview(data)
            if mv.ndim == dimension and mv.format == 'd':
                total = 1
                for s in shape:
                    total *= s
                if total == 0:
                    return vector_map_as_tensor(vector=[], shape=shape,
                                                start=0)
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

    # general path: carry-based read through get_item
    total_elements = 1
    for s in shape:
        total_elements *= s
    if total_elements == 0:
        return vector_map_as_tensor(vector=[], shape=shape, start=0)
    vector = [0.0] * total_elements

    num_list = [0] * dimension
    pos = 0
    while True:
        idx = tuple(start[i] + num_list[i] * step[i] for i in range(dimension))
        vector[pos] = get_item(data, idx)
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


_numeric_formats = frozenset(("b","B","h","H","i","I","l","L","q","Q","n","N","f","d","e","g"))


def _load_data_shape(data):
    """infer_shape with a BufferError fallback to pure length detection
    (containers that declare but cannot export the buffer protocol)."""
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
    """Copy a sampled sub-region of source into target; each side has its
    own start and step.  The effective region is clipped on both sides
    (out-of-bounds is silently truncated).  PyBuffer fast path first,
    get_item/set_item fallback.  Iterative.  Returns the number of
    elements actually copied (0 for an empty region)."""
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
                # containers exporting a fresh snapshot per call cannot
                # persist a write - fall back to the generic set_item path
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


# ----- region helpers
def _region_spec(data, start, shape, step):
    """Resolve (effective_shape, start, step) for a read region (clipping
    matches load_data's source side)."""
    data_shape = infer_shape(data)
    if data_shape is None:
        raise ValueError("cannot infer shape of data")
    return _region_spec_by_shape(data_shape, start, shape, step)


def _region_spec_by_shape(data_shape, start, shape, step):
    """_region_spec taking the data shape directly (available to custom
    iterators for resolving the position parameters)."""
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
    """Yield local coordinates of the region in row-major order."""
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


def _position_transform(real, origin, scale):
    """Real coordinates -> logical callback coordinates:
    logical_i = real_i * scale_i - origin_i.  Integral values normalize
    to int, otherwise float."""
    out = []
    for i in range(len(real)):
        value = real[i] * scale[i] - origin[i]
        if isinstance(value, float) and value.is_integer():
            out.append(int(value))
        else:
            out.append(value)
    return tuple(out)


def _position_region(output, start, shape, step, origin, scale):
    """Resolve the real traversal region and the logical transform for
    position-based callbacks."""
    effective, r_start, r_step = _region_spec(output, start, shape, step)
    dimension = len(effective)
    if origin is None:
        origin = (0,) * dimension
    else:
        origin = tuple(origin)
        if len(origin) != dimension:
            raise ValueError("origin length does not match dimension")
    if scale is None:
        scale = (1,) * dimension
    else:
        scale = tuple(scale)
        if len(scale) != dimension:
            raise ValueError("scale length does not match dimension")
    return effective, r_start, r_step, origin, scale


# ---- core types ----
class func_name_space(Mapping):
    """Callback namespace: fixed callback slots plus arbitrary extra fields
    (mirrors the C FuncNameSpaceType, including the mapping protocol)."""

    __slots__ = ("output", "output_start", "output_step", "window_size", "kernel",
                 "start", "end", "d", "step", "algorithm", "num",
                 "start_callback", "end_callback",
                 "global_error_callback", "local_error_callback", "return_callback",
                 "_extra")

    def __init__(self, *arg, **kwarg):
        self._extra = {}
        for key, value in kwarg.items():
            setattr(self, key, value)

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

    # mapping protocol: ns unpacking, dict(ns), len, iteration
    def keys(self):
        names = [n for n in self.__class__.__slots__ if n != "_extra"
                 and hasattr(self, n)]
        return names + list(self._extra)

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())


class default_contain:
    """Default-value container: every key is 'contained' (lookup always
    returns a value; matches the C sq_contains contract)."""

    __slots__ = ("default", "default_dict")

    def __init__(self, default, default_dict=None):
        self.default, self.default_dict = default, (default_dict if default_dict else {})

    def __len__(self)->int:
        return 1

    def __getitem__(self, index):
        return self.default_dict.get(index, self.default)

    def __contains__(self, index):
        return True

    def __repr__(self):
        return "<default_contain: default=%r>" % (self.default,)


# ---- core functions ----

# ----- similarity algorithms
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


# ----- passive / active mode (B-class)
def _passive_kernel(index, *, data, output, output_start, output_step,
                    window_size, window_start, window_step, d, algorithm, name,
                    global_error_callback, local_error_callback,
                    transform1=None, transform2=None):
    """One output element: aggregate the window offsets, apply the
    algorithm, write the output."""
    dimension = len(index)
    try:
        main = other = mu = 0
        inner = [0] * dimension
        inner_flag = dimension
        while inner_flag:
            if inner_flag == dimension:
                main_place = tuple(window_start[i] + window_step[i] * index[i] + inner[i]
                                   for i in range(dimension))
                other_place = tuple(main_place[i] + d[i]
                                    for i in range(dimension))
                try:
                    a = get_item(data, main_place)
                    b = get_item(data, other_place)
                    if transform1 is not None:
                        a = transform1(a)
                    if transform2 is not None:
                        b = transform2(b)
                    main += a * a
                    other += b * b
                    mu += a * b
                except Exception as e:
                    if local_error_callback:
                        local_error_callback(e, name)
            if inner[inner_flag - 1] + 1 < window_size[inner_flag - 1]:
                inner[inner_flag - 1] += 1
                inner_flag = dimension
            else:
                inner[inner_flag - 1] = 0
                inner_flag -= 1
        output_places = tuple(output_start[p] + output_step[p] * index[p]
                              for p in range(dimension))
        set_item(output, output_places, algorithm(main, other, mu, name))
    except Exception as e:
        if global_error_callback:
            global_error_callback(e, name)


def cos_comparison_passive(data,
                           *arg,
                           window_size=None,
                           start=None, end=None,
                           step=None, d=None,
                           algorithm=_default_algorithm,
                           output=None,
                           output_start=None, output_step=None,
                           start_callback=None,
                           end_callback=None,
                           global_error_callback=None,
                           local_error_callback=None,
                           return_callback=None,
                           use_namespace=True,
                           namespace_hook=None,
                           iterate=None,
                           transform1=None, transform2=None,
                           **kwargs):
    """Passive mode: per output position, compare the window at start with
    the window shifted by d and write algorithm(main, other, mu).  The
    optional iterate engine delegates the element work to the injected
    iterator; transform1/transform2 map the two windows' read values
    (default identity).  See docs/api for the full parameter contract."""
    if hasattr(data, "__cos_comparison_passive__"):
        # Bound reload hook: data is already the bound self, so forward
        # only the call's extra arguments plus the effective option values.
        context = {
            "window_size": window_size, "start": start, "end": end,
            "step": step, "d": d, "algorithm": algorithm, "output": output,
            "output_start": output_start, "output_step": output_step,
            "start_callback": start_callback, "end_callback": end_callback,
            "global_error_callback": global_error_callback,
            "local_error_callback": local_error_callback,
            "return_callback": return_callback,
            "use_namespace": use_namespace, "namespace_hook": namespace_hook,
            "iterate": iterate, "transform1": transform1,
            "transform2": transform2,
        }
        context.update(kwargs)
        return data.__cos_comparison_passive__(*arg, **context)

    shape = infer_shape(data)
    if shape is None:
        raise ValueError("cannot infer shape of input data")
    length = list(shape)
    dimension = len(length)

    start = start if start is not None else (0,) * dimension
    end = end if end is not None else tuple(length)
    step = step if step is not None else (1,) * dimension
    d = d if d is not None else (1,) + (0,) * (dimension - 1)
    window_size = window_size if window_size is not None else (1,) * dimension
    for i in range(dimension):
        if start[i] < 0:
            raise ValueError("start entries must be non-negative")
        if step[i] <= 0:
            raise ValueError("step must be positive for all dimensions")
        if window_size[i] <= 0:
            raise ValueError("window_size must be positive for all dimensions")
        if d[i] + start[i] < 0 or end[i] > length[i] \
                or end[i] - d[i] > length[i]:
            raise ValueError(f"region exceeds the data bounds on axis {i}")
    num = [0 for _ in range(dimension)]
    for i in range(dimension):
        step_effective = end[i] - start[i] - window_size[i] - d[i]
        if step_effective >= 0:
            num[i] = (step_effective // step[i]) + 1
        else:
            raise ValueError("effectless args.")

    output_start = output_start if output_start is not None else (0,) * dimension
    output_step = output_step if output_step is not None else (1,) * dimension
    if any(v < 0 for v in output_start):
        raise ValueError("output_start entries must be non-negative")
    if any(v <= 0 for v in output_step):
        raise ValueError("output_step must be positive for all dimensions")
    if output is None and any(v != 0 for v in output_start):
        raise ValueError("output_start requires an explicit output")
    output = output if output is not None else create_void_list(
        ((n - 1) * s + 1 for n, s in zip(num, output_step))
    )

    # namespace_hook creates the callback namespace; use_namespace=False
    # skips it (callbacks then receive None)
    if use_namespace:
        _hook = (func_name_space if namespace_hook is None
                 else namespace_hook)
        name = _hook(
        output=output,
        output_start=output_start, output_step=output_step,
        window_size=window_size,
        start=start, end=end, step=step, d=d,
        algorithm=algorithm,
        num=num,
        start_callback=start_callback,
        end_callback=end_callback,
        global_error_callback=global_error_callback,
        local_error_callback=local_error_callback,
        return_callback=return_callback,
    )
    else:
        name = None

    if start_callback:
        start_callback(name)

    if iterate is not None:
        iterate(_passive_kernel, data_shape=tuple(num), data=data,
                output=output, output_start=output_start,
                output_step=output_step, window_size=window_size,
                window_start=start, window_step=step, d=d, algorithm=algorithm,
                name=name, global_error_callback=global_error_callback,
                local_error_callback=local_error_callback,
                transform1=transform1, transform2=transform2)
        if end_callback:
            end_callback(name)
        if return_callback:
            return return_callback(output, name)
        return output

    # default skeleton: double odometer (output positions x window offsets)
    flag = dimension
    num_list = [None] + [1] * len(num)  # 1-based indices
    main, other, mu = 0, 0, 0

    while flag:
        try:
            if flag == dimension:
                inner_list = [None] + [1] * len(window_size)
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
                            a = get_item(data, main_place)
                            b = get_item(data, other_place)
                            if transform1 is not None:
                                a = transform1(a)
                            if transform2 is not None:
                                b = transform2(b)
                            main += a * a
                            other += b * b
                            mu += a * b

                        if inner_list[inner_flag] < window_size[inner_flag - 1]:
                            inner_list[inner_flag] += 1
                            inner_flag = dimension
                        else:
                            inner_list[inner_flag] = 1
                            inner_flag -= 1
                    except Exception as e:
                        if local_error_callback:
                            local_error_callback(e, name)

                output_places = tuple(output_start[p] + output_step[p] * (num_list[p + 1] - 1) for p in range(dimension))
                set_item(output, output_places, algorithm(main, other, mu, name))

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


def _active_kernel(index, *, data, window_kernel, output, output_start, output_step,
                   window_size, window_start, window_step, algorithm, name,
                   global_error_callback, local_error_callback,
                   transform1=None, transform2=None):
    """One output element: aggregate the data window against the kernel
    template, apply the algorithm, write the output."""
    dimension = len(index)
    try:
        main = other = mu = 0
        inner = [0] * dimension
        inner_flag = dimension
        while inner_flag:
            if inner_flag == dimension:
                data_place = tuple(window_start[i] + window_step[i] * index[i] + inner[i]
                                   for i in range(dimension))
                kern_place = tuple(inner[i] for i in range(dimension))
                try:
                    a = get_item(data, data_place)
                    b = get_item(window_kernel, kern_place)
                    if transform1 is not None:
                        a = transform1(a)
                    if transform2 is not None:
                        b = transform2(b)
                    main += a * a
                    other += b * b
                    mu += a * b
                except Exception as e:
                    if local_error_callback:
                        local_error_callback(e, name)
            if inner[inner_flag - 1] + 1 < window_size[inner_flag - 1]:
                inner[inner_flag - 1] += 1
                inner_flag = dimension
            else:
                inner[inner_flag - 1] = 0
                inner_flag -= 1
        output_places = tuple(output_start[p] + output_step[p] * index[p]
                              for p in range(dimension))
        set_item(output, output_places, algorithm(main, other, mu, name))
    except Exception as e:
        if global_error_callback:
            global_error_callback(e, name)


def cos_comparison_active(data,
                          *arg,
                          kernel=None,
                          start=None, end=None,
                          step=None,
                          algorithm=_default_algorithm,
                          output=None,
                          output_start=None, output_step=None,
                          start_callback=None,
                          end_callback=None,
                          global_error_callback=None,
                          local_error_callback=None,
                          return_callback=None,
                          use_namespace=True,
                          namespace_hook=None,
                          iterate=None,
                          transform1=None, transform2=None,
                          **kwargs):
    """Active mode: slide the kernel template over the data and write
    algorithm(main, other, mu) per position; the window size is the kernel
    shape.  The optional iterate engine delegates the element work to the
    injected iterator; transform1/transform2 map the data window and the
    kernel values (default identity)."""
    if hasattr(data, "__cos_comparison_active__"):
        # Bound reload hook: data is already the bound self, so forward
        # only the call's extra arguments plus the effective option values.
        context = {
            "kernel": kernel, "algorithm": algorithm, "output": output,
            "output_start": output_start, "output_step": output_step,
            "start": start, "end": end, "step": step,
            "start_callback": start_callback, "end_callback": end_callback,
            "global_error_callback": global_error_callback,
            "local_error_callback": local_error_callback,
            "return_callback": return_callback,
            "use_namespace": use_namespace, "namespace_hook": namespace_hook,
            "iterate": iterate, "transform1": transform1,
            "transform2": transform2,
        }
        context.update(kwargs)
        return data.__cos_comparison_active__(*arg, **context)

    if kernel is None:
        raise ValueError("kernel must be provided for active mode")

    shape = infer_shape(data)
    if shape is None:
        raise ValueError("cannot infer shape of input data")
    length = list(shape)
    dimension = len(length)

    kshape = infer_shape(kernel)
    if kshape is None:
        raise ValueError("cannot infer shape of kernel")
    kernel_shape = list(kshape)
    if len(kernel_shape) != dimension:
        raise ValueError(f"kernel dimension {len(kernel_shape)} does not match data dimension {dimension}")

    start = start if start is not None else (0,) * dimension
    end = end if end is not None else tuple(length)
    step = step if step is not None else (1,) * dimension
    window_size = tuple(kernel_shape)  # active window = kernel shape

    for i in range(dimension):
        if start[i] < 0:
            raise ValueError("start entries must be non-negative")
        if step[i] <= 0:
            raise ValueError("step must be positive for all dimensions")
        if window_size[i] <= 0:
            raise ValueError("window_size must be positive for all dimensions")
        if end[i] > length[i]:
            raise ValueError(f"region exceeds the data bounds on axis {i}")
    num = [0 for _ in range(dimension)]
    for i in range(dimension):
        step_effective = end[i] - start[i] - window_size[i]
        if step_effective >= 0:
            num[i] = (step_effective // step[i]) + 1
        else:
            raise ValueError("effectless args.")

    output_start = output_start if output_start is not None else (0,) * dimension
    output_step = output_step if output_step is not None else (1,) * dimension
    if any(v < 0 for v in output_start):
        raise ValueError("output_start entries must be non-negative")
    if any(v <= 0 for v in output_step):
        raise ValueError("output_step must be positive for all dimensions")
    if output is None and any(v != 0 for v in output_start):
        raise ValueError("output_start requires an explicit output")
    output = output if output is not None else create_void_list(
        ((n - 1) * s + 1 for n, s in zip(num, output_step))
    )

    if use_namespace:
        _hook = (func_name_space if namespace_hook is None
                 else namespace_hook)
        name = _hook(
        output=output,
        output_start=output_start, output_step=output_step,
        window_size=window_size,
        kernel=kernel,
        start=start, end=end, step=step,
        algorithm=algorithm,
        num=num,
        start_callback=start_callback,
        end_callback=end_callback,
        global_error_callback=global_error_callback,
        local_error_callback=local_error_callback,
        return_callback=return_callback,
    )
    else:
        name = None

    if start_callback:
        start_callback(name)

    if iterate is not None:
        iterate(_active_kernel, data_shape=tuple(num), data=data,
                window_kernel=kernel, output=output, output_start=output_start,
                output_step=output_step, window_size=window_size,
                window_start=start, window_step=step, algorithm=algorithm,
                name=name, global_error_callback=global_error_callback,
                local_error_callback=local_error_callback,
                transform1=transform1, transform2=transform2)
        if end_callback:
            end_callback(name)
        if return_callback:
            return return_callback(output, name)
        return output

    flag = dimension
    num_list = [None] + [1 for _ in num]
    main, other, mu = 0, 0, 0

    while flag:
        try:
            if flag == dimension:
                inner_list = [None] + [1] * len(window_size)
                inner_flag = len(window_size)
                main = 0
                other = 0
                mu = 0

                while inner_flag:
                    try:
                        if inner_flag == dimension:
                            data_place = tuple(
                                start[i] + step[i] * (num_list[i + 1] - 1) + (inner_list[i + 1] - 1)
                                for i in range(dimension)
                            )
                            kern_place = tuple(
                                inner_list[i + 1] - 1 for i in range(dimension)
                            )
                            a = get_item(data, data_place)
                            b = get_item(kernel, kern_place)
                            if transform1 is not None:
                                a = transform1(a)
                            if transform2 is not None:
                                b = transform2(b)
                            main += a * a
                            other += b * b
                            mu += a * b

                        if inner_list[inner_flag] < window_size[inner_flag - 1]:
                            inner_list[inner_flag] += 1
                            inner_flag = dimension
                        else:
                            inner_list[inner_flag] = 1
                            inner_flag -= 1
                    except Exception as e:
                        if local_error_callback:
                            local_error_callback(e, name)

                output_places = tuple(output_start[p] + output_step[p] * (num_list[p + 1] - 1) for p in range(dimension))
                set_item(output, output_places, algorithm(main, other, mu, name))

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


# ----- statistics
def _build_ones(shape):
    """All-one kernel of the given shape (iterative)."""
    if isinstance(shape, default_contain):
        return shape
    if isinstance(shape, int):
        return [1.0] * shape
    shape = tuple(shape)
    if len(shape) == 0:
        return 1.0
    flat = [1.0] * multiple_chain(shape, 1)
    stack = flat
    for dim in range(len(shape) - 1, -1, -1):
        width = shape[dim]
        nxt = [stack[i:i + width] for i in range(0, len(stack), width)]
        stack = nxt
    return stack[0]


def _flatten_nested(obj):
    """Flatten nested iterables into a flat list (row-major, iterative)."""
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
    """Reshape a flat list into a nested window of the given shape; the
    first product(shape) values apply (C-backend semantics)."""
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
        nxt = []
        for start in range(0, len(stack), width):
            nxt.append(stack[start:start + width])
        stack = nxt
    return stack[0]


def mean_local(data, *arg, local_size=None, step=None, weight=None,
               output=None, output_start=None, output_step=None, **kwargs):
    """Local mean (average pooling), any N-D; weight is a per-window
    pattern (row-major, first product(local_size) values)."""
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
        kernel = _flat_to_window(_flatten_nested(weight), local_size)

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
    """Local variance E[X^2] - E[X]^2, any N-D."""
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


# ----- whole-tensor similarity
def cos(a, b, algorithm=_cos):
    """Whole-tensor similarity: accumulate the squared sums and the dot
    product over all positions and pass them to algorithm."""
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
        except (TypeError, IndexError, AttributeError):
            shape = tuple(shape)
            break
    if dimension == 0:
        raise ValueError("the args you gave are not tensors.")

    num_list = [None] + [1] * dimension   # current position (1-based)
    flag = dimension

    sum_a = 0.0
    sum_b = 0.0
    sum_ab = 0.0

    while flag:
        if flag == dimension:
            idx = tuple(num_list[i] - 1 for i in range(1, dimension + 1))

            val_a = get_item(a, idx)
            val_b = get_item(b, idx)

            sum_a += val_a * val_a
            sum_b += val_b * val_b
            sum_ab += val_a * val_b

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


# ----- element-wise family (A-class)
def _data_mapping_kernel(index, *, data, callback, out, r_start, r_step,
                         out_start, out_step, out_shape):
    """One element: index -> read -> callback -> write."""
    dimension = len(index)
    read = tuple(r_start[i] + index[i] * r_step[i]
                 for i in range(dimension))
    value = get_item(data, read)
    try:
        mapped = callback(value)
    except Exception:
        return
    write = tuple(out_start[i] + index[i] * out_step[i]
                  for i in range(dimension))
    if all(write[i] < out_shape[i] for i in range(dimension)):
        set_item(out, write, mapped)


def _data_filter_kernel(index, *, data, callback, hits, r_start, r_step,
                        origin, basis):
    """One element: index -> read -> predicate -> collect."""
    dimension = len(index)
    read = tuple(r_start[i] + index[i] * r_step[i]
                 for i in range(dimension))
    value = get_item(data, read)
    try:
        hit = callback(value)
    except Exception:
        return
    if hit:
        hits.append(tuple(origin[i] + basis[i] * index[i]
                          for i in range(dimension)))


def _elementwise_kernel(index, *, tensors, func, output):
    """One element: index -> multi-read -> func -> write."""
    vals = tuple(get_item(t, index) for t in tensors)
    value = func(*vals)
    if not isinstance(value, (int, float)):
        raise TypeError("elementwise func must return a number")
    set_item(output, index, value)


def _position_map_kernel(index, *, output, callback, r_start, r_step,
                         origin, scale, status):
    """One element: index -> logical -> callback -> write."""
    if status[0]:
        return
    dimension = len(index)
    real = tuple(r_start[i] + index[i] * r_step[i]
                 for i in range(dimension))
    logical = _position_transform(real, origin, scale)
    try:
        value = callback(logical)
    except Exception:
        return
    try:
        set_item(output, real, value)
    except Exception:
        status[0] = 1


def _elementwise_position_kernel(index, *, output, tensors, callback,
                                 r_start, r_step, origin, scale, status):
    """One element: index -> multi-read + logical -> callback -> write."""
    if status[0]:
        return
    dimension = len(index)
    real = tuple(r_start[i] + index[i] * r_step[i]
                 for i in range(dimension))
    logical = _position_transform(real, origin, scale)
    elements = []
    for tensor in tensors:
        try:
            elements.append(get_item(tensor, real))
        except Exception:
            status[0] = 1
            return
    try:
        value = callback(elements, logical)
    except Exception:
        return
    try:
        set_item(output, real, value)
    except Exception:
        status[0] = 1


def data_filter(data, callback, *, start=None, shape=None, step=None,
                origin=None, basis=None, iterate=None):
    """Yield the position of every element whose callback(value) is truthy
    over a sampled read region; reported position = origin + basis * local
    (defaults: origin=start, basis=step).  Callback errors are silently
    skipped.  iterate= runs the work through the injected element engine
    (see the iterate contract in docs); None keeps the inline skeleton."""
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
    if iterate is not None:
        hits = []
        iterate(_data_filter_kernel, data_shape=infer_shape(data),
                start=r_start, shape=shape, step=r_step, data=data,
                callback=callback, hits=hits, r_start=r_start,
                r_step=r_step, origin=origin, basis=basis)
        for position in hits:
            yield position
        return
    for local in _region_walk(effective):
        read = tuple(r_start[i] + local[i] * r_step[i] for i in range(dimension))
        value = get_item(data, read)
        try:
            hit = callback(value)
        except Exception:
            continue
        if hit:
            yield tuple(origin[i] + basis[i] * local[i] for i in range(dimension))


def elementwise(*tensors, func=None, output=None, iterate=None):
    """Element-wise operation: func(x1, x2, ...) per position, written into
    output (same shape).  Tensors only need get_item / set_item /
    infer_shape; callback errors propagate; func must return a number.
    Returns 0 on success.  iterate= runs the work through the injected
    element engine."""
    if func is None:
        raise TypeError("func is required")
    if not tensors:
        raise TypeError("elementwise requires at least one tensor")
    if output is None:
        raise ValueError("output is required")
    shape = infer_shape(tensors[0])
    if shape is None:
        raise ValueError("cannot infer shape of input")
    for t in tensors[1:]:
        if infer_shape(t) != shape:
            raise ValueError("the shape of two tensors are not same.")
    if infer_shape(output) != shape:
        raise ValueError("output shape does not match input")
    if iterate is not None:
        iterate(_elementwise_kernel, data_shape=shape, tensors=tensors,
                func=func, output=output)
        return 0
    dimension = len(shape)
    total = 1
    for s in shape:
        total *= s
    idx = [0] * dimension
    for _ in range(total):
        vals = tuple(get_item(t, tuple(idx)) for t in tensors)
        value = func(*vals)
        if not isinstance(value, (int, float)):
            raise TypeError("elementwise func must return a number")
        set_item(output, tuple(idx), value)
        for d in range(dimension - 1, -1, -1):
            idx[d] += 1
            if idx[d] < shape[d]:
                break
            idx[d] = 0
    return 0


def data_mapping(data, callback, *, start=None, shape=None, step=None,
                 out=None, out_start=None, out_step=None, iterate=None):
    """Map every sampled element through callback(value) into out; write
    position = out_start + out_step * local (clipped).  Callback errors are
    silently skipped.  Returns the output tensor.  iterate= runs the work
    through the injected element engine."""
    data_shape = infer_shape(data)
    if data_shape is None:
        raise ValueError("cannot infer shape of data")
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
    if iterate is not None:
        iterate(_data_mapping_kernel, data_shape=data_shape, start=r_start,
                shape=shape, step=r_step, data=data, callback=callback,
                out=out, r_start=r_start, r_step=r_step,
                out_start=out_start, out_step=out_step,
                out_shape=out_shape)
        return out
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


def position_map(output, callback, *, start=None, shape=None, step=None,
                 origin=None, scale=None, iterate=None):
    """Position-driven callback: output[real] = callback(logical) over the
    read region, logical = real * scale - origin.  Callback errors are
    silently skipped; a write failure stops with status 1.  Returns a
    status code (0 success).  iterate= runs the work through the injected
    element engine."""
    effective, r_start, r_step, origin, scale = _position_region(
        output, start, shape, step, origin, scale)
    if iterate is not None:
        status = [0]
        iterate(_position_map_kernel, data_shape=infer_shape(output),
                start=r_start, shape=shape, step=r_step, output=output,
                callback=callback, r_start=r_start, r_step=r_step,
                origin=origin, scale=scale, status=status)
        return status[0]
    for local in _region_walk(effective):
        real = tuple(r_start[i] + local[i] * r_step[i]
                     for i in range(len(effective)))
        logical = _position_transform(real, origin, scale)
        try:
            value = callback(logical)
        except Exception:
            continue
        try:
            set_item(output, real, value)
        except Exception:
            return 1
    return 0


def elementwise_position(output, *tensors, callback=None,
                         start=None, shape=None, step=None,
                         origin=None, scale=None, iterate=None):
    """Multi-tensor position callback:
    output[real] = callback([t[real] for t in tensors], logical).  Callback
    errors are silently skipped; a read or write failure stops with status
    1.  Returns a status code (0 success).  iterate= runs the work through
    the injected element engine."""
    if callback is None:
        raise TypeError("callback is required")
    effective, r_start, r_step, origin, scale = _position_region(
        output, start, shape, step, origin, scale)
    if iterate is not None:
        status = [0]
        iterate(_elementwise_position_kernel,
                data_shape=infer_shape(output), start=r_start, shape=shape,
                step=r_step, output=output, tensors=tensors,
                callback=callback, r_start=r_start, r_step=r_step,
                origin=origin, scale=scale, status=status)
        return status[0]
    for local in _region_walk(effective):
        real = tuple(r_start[i] + local[i] * r_step[i]
                     for i in range(len(effective)))
        logical = _position_transform(real, origin, scale)
        elements = []
        for tensor in tensors:
            try:
                elements.append(get_item(tensor, real))
            except Exception:
                return 1
        try:
            value = callback(elements, logical)
        except Exception:
            continue
        try:
            set_item(output, real, value)
        except Exception:
            return 1
    return 0


# ----- pooling
def _pool_axis(value, dimension, default, name):
    """Region axis: None -> default, int -> 1-tuple, else sequence; the
    length must match the data dimension (no broadcast)."""
    if value is None:
        return tuple(default)
    if isinstance(value, int):
        value = (value,)
    value = tuple(value)
    if len(value) != dimension:
        raise ValueError(name + " length does not match data dimension")
    return value


def _pool_kernel(index, *, data, output, window_start, window_step,
                 window_size, func):
    """One output element: gather the window values in row-major order,
    apply func (None = mean) and write the output."""
    dimension = len(index)
    values = []
    inner = [0] * dimension
    while True:
        place = tuple(window_start[i] + window_step[i] * index[i] + inner[i]
                      for i in range(dimension))
        values.append(get_item(data, place))
        d = dimension - 1
        while d >= 0:
            inner[d] += 1
            if inner[d] < window_size[d]:
                break
            inner[d] = 0
            d -= 1
        if d < 0:
            break
    if func is None:
        value = sum(values) / len(values)
    else:
        value = func(*values)
        if not isinstance(value, numbers.Number):
            raise TypeError("pool func must return a number")
    set_item(output, index, value)


def pool(data, *, window_size=None, func=None, step=None,
         start=None, end=None, output=None, iterate=None):
    """Sliding-window pooling: reduce every valid window of data to one
    output element.  Output i reads the window at start + step*i, so
    windows may overlap (step < window_size), tile exactly (step equals
    window_size) or be spaced apart (step > window_size); trailing
    windows that would cross end are dropped.  window_size defaults to
    all ones; func(*values) receives the window values in row-major order
    (func=None = mean, a required callable otherwise).  output is
    required, must have the window-count shape and is written in place;
    returns 0.  iterate= runs the work through the injected element
    engine."""
    shape = infer_shape(data)
    if shape is None:
        raise ValueError("cannot infer shape of input data")
    dimension = len(shape)
    window_size = _pool_axis(window_size, dimension, [1] * dimension,
                             "window_size")
    start = _pool_axis(start, dimension, [0] * dimension, "start")
    end = _pool_axis(end, dimension, list(shape), "end")
    step = _pool_axis(step, dimension, [1] * dimension, "step")
    for i in range(dimension):
        if start[i] < 0:
            raise ValueError("start entries must be non-negative")
        if step[i] <= 0:
            raise ValueError("step must be positive for all dimensions")
        if window_size[i] <= 0:
            raise ValueError("window_size must be positive for all dimensions")
        if end[i] > shape[i]:
            raise ValueError("end exceeds the data shape")
    num = []
    for i in range(dimension):
        span = end[i] - start[i] - window_size[i]
        if span < 0:
            raise ValueError("effectless args.")
        num.append(span // step[i] + 1)
    num = tuple(num)
    if output is None:
        raise ValueError("output is required")
    if infer_shape(output) != num:
        raise ValueError("output shape does not match the window count")
    if func is not None and not callable(func):
        raise TypeError("func must be callable")
    if iterate is not None:
        iterate(_pool_kernel, data_shape=num, data=data, output=output,
                window_start=start, window_step=step,
                window_size=window_size, func=func)
        return 0
    index = [0] * dimension
    while True:
        _pool_kernel(tuple(index), data=data, output=output,
                     window_start=start, window_step=step,
                     window_size=window_size, func=func)
        d = dimension - 1
        while d >= 0:
            index[d] += 1
            if index[d] < num[d]:
                break
            index[d] = 0
            d -= 1
        if d < 0:
            break
    return 0


# ----- thresholds
def _make_threshold_predicate(low, high, inclusive):
    """Predicate for the interval [low, high] (bounds optional);
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
    """data_filter over [low, high]: yields the positions whose value lies
    in the interval (endpoints per inclusive)."""
    predicate = _make_threshold_predicate(low, high, inclusive)
    return data_filter(data, predicate, **region)


def threshold_map(data, pairs, *, default_value=0.0, **region):
    """Map every sampled element via (func, value) pairs: the first truthy
    func selects its value, else default_value; callback errors are
    silently skipped."""
    def matcher(value):
        for func, mapped in pairs:
            try:
                if func(value):
                    return mapped
            except Exception:
                continue
        return default_value

    return data_mapping(data, matcher, **region)


def threshold_judge(low=None, high=None, *, inclusive=(True, True)):
    """Judge factory: 1 inside [low, high], else 0 (single-bound allowed);
    pairs with threshold_map's (func, value) iteration."""
    predicate = _make_threshold_predicate(low, high, inclusive)
    return lambda value: 1 if predicate(value) else 0
