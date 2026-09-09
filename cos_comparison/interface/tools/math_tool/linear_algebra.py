"""Linear algebra support (dimension-generic, duck typing).

Every function follows one convention:

  * the tensor output is passed out through the ``output`` keyword
    argument - never returned directly (no tensor type hard-coding);
  * the function returns an integer status (0 = success).

Status codes:
    0  success
    1  shape / length mismatch
    2  no output container (output=None) or output write failure
    3  value conversion failure (non-numeric element)

All functions accept any-dimension tensors built on the sequence
protocol (lists, tuples, custom containers, ...); scalars are written
to ``output[0]``.  The C extension (_linear_algebra) provides the same
behaviour.
"""

__all__ = (
    "CONVERSION_FAILURE",
    "NO_OUTPUT",
    "OK",
    "SHAPE_MISMATCH",
    "add",
    "clip",
    "dot",
    "flatten",
    "multiply",
    "norm",
    "normalize",
    "power",
    "scale",
    "tensor_mean",
    "tensor_sum",
)

OK = 0
SHAPE_MISMATCH = 1
NO_OUTPUT = 2
CONVERSION_FAILURE = 3


def _is_sequence(obj):
    """Duck-typed sequence check (iteration or item access)."""
    return hasattr(obj, "__iter__") or hasattr(obj, "__getitem__")


def _infer_shape(data):
    """Infer the shape of a duck-typed tensor (iterative)."""
    shape = []
    obj = data
    while _is_sequence(obj) and not isinstance(obj, (str, bytes)):
        try:
            length = len(obj)
        except TypeError:
            break
        shape.append(length)
        if length == 0:
            break
        try:
            obj = next(iter(obj))
        except (StopIteration, TypeError):
            break
    return tuple(shape)


def _flatten(data):
    """Flatten any-dimension data into a value list (explicit stack,
    no recursion)."""
    values = []
    stack = [iter(data)]
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if _is_sequence(item) and not isinstance(item, (str, bytes)):
            stack.append(iter(item))
        else:
            values.append(item)
    return values


def _convert(value):
    """Convert an element to float (None on failure)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iter_indices(shape):
    """Iterate every index tuple of a shape (iterative, no recursion)."""
    n = len(shape)
    if n == 0:
        yield ()
        return
    idx = [0] * n
    while True:
        yield tuple(idx)
        for d in range(n - 1, -1, -1):
            idx[d] += 1
            if idx[d] < shape[d]:
                break
            idx[d] = 0
        else:
            break


def _set_indexed(container, index, value):
    """Write a value through an index path (duck: __setitem__)."""
    obj = container
    for i in index[:-1]:
        obj = obj[i]
    obj[index[-1]] = value


def _get_indexed(container, index):
    """Read a value through an index path (duck: __getitem__)."""
    obj = container
    for i in index:
        obj = obj[i]
    return obj


def _write_flat(output, values, start=0):
    """Write a value list linearly into output (duck __setitem__)."""
    for i, v in enumerate(values):
        try:
            output[start + i] = v
        except (TypeError, IndexError, KeyError):
            return NO_OUTPUT
    return OK


def _write_shaped(output, shape, values):
    """Write a value list back into output following the shape."""
    for idx, v in zip(_iter_indices(shape), values):
        try:
            _set_indexed(output, idx, v)
        except (TypeError, IndexError, KeyError):
            return NO_OUTPUT
    return OK


def _to_doubles(values):
    """Convert a value list to floats (None on failure)."""
    result = []
    for v in values:
        fv = _convert(v)
        if fv is None:
            return None
        result.append(fv)
    return result


# ---------------------------------------------------------------------------
# scalar-result functions (result written to output[0])
# ---------------------------------------------------------------------------
def dot(a, b, output=None):
    """Element-wise product sum of two tensors (any dimension)."""
    if output is None:
        return NO_OUTPUT
    va = _to_doubles(_flatten(a))
    vb = _to_doubles(_flatten(b))
    if va is None or vb is None:
        return CONVERSION_FAILURE
    if len(va) != len(vb):
        return SHAPE_MISMATCH
    total = 0.0
    for x, y in zip(va, vb):
        total += x * y
    return _write_flat(output, [total])


def norm(a, output=None):
    """Frobenius (Euclidean) norm of a tensor (any dimension)."""
    if output is None:
        return NO_OUTPUT
    va = _to_doubles(_flatten(a))
    if va is None:
        return CONVERSION_FAILURE
    total = 0.0
    for v in va:
        total += v * v
    return _write_flat(output, [total ** 0.5])


def tensor_sum(a, output=None):
    """Sum of all elements of a tensor (any dimension)."""
    if output is None:
        return NO_OUTPUT
    va = _to_doubles(_flatten(a))
    if va is None:
        return CONVERSION_FAILURE
    return _write_flat(output, [sum(va)])


def tensor_mean(a, output=None):
    """Mean of all elements of a tensor (any dimension)."""
    if output is None:
        return NO_OUTPUT
    va = _to_doubles(_flatten(a))
    if va is None:
        return CONVERSION_FAILURE
    if not va:
        return _write_flat(output, [0.0])
    return _write_flat(output, [sum(va) / len(va)])


# ---------------------------------------------------------------------------
# element-wise functions (output keeps the input shape)
# ---------------------------------------------------------------------------
def _elementwise_pair(a, b, output, func):
    """Shared implementation for two-tensor element-wise operations."""
    if output is None:
        return NO_OUTPUT
    va = _to_doubles(_flatten(a))
    vb = _to_doubles(_flatten(b))
    if va is None or vb is None:
        return CONVERSION_FAILURE
    if len(va) != len(vb):
        return SHAPE_MISMATCH
    values = [func(x, y) for x, y in zip(va, vb)]
    return _write_shaped(output, _infer_shape(a), values)


def _elementwise_single(a, output, func, *args):
    """Shared implementation for single-tensor element-wise operations."""
    if output is None:
        return NO_OUTPUT
    va = _to_doubles(_flatten(a))
    if va is None:
        return CONVERSION_FAILURE
    values = [func(v, *args) for v in va]
    return _write_shaped(output, _infer_shape(a), values)


def add(a, b, output=None):
    """Element-wise addition of two tensors (same shape)."""
    return _elementwise_pair(a, b, output, lambda x, y: x + y)


def multiply(a, b, output=None):
    """Element-wise multiplication of two tensors (same shape)."""
    return _elementwise_pair(a, b, output, lambda x, y: x * y)


def scale(a, factor, output=None):
    """Element-wise multiplication by a scalar."""
    if output is None:
        return NO_OUTPUT
    fv = _convert(factor)
    if fv is None:
        return CONVERSION_FAILURE
    return _elementwise_single(a, output, lambda v, f: v * f, fv)


def normalize(a, output=None):
    """Element-wise division by the tensor norm."""
    if output is None:
        return NO_OUTPUT
    va = _to_doubles(_flatten(a))
    if va is None:
        return CONVERSION_FAILURE
    total = 0.0
    for v in va:
        total += v * v
    length = total ** 0.5
    if length == 0.0:
        values = [0.0] * len(va)
    else:
        values = [v / length for v in va]
    return _write_shaped(output, _infer_shape(a), values)


def power(a, exponent, output=None):
    """Element-wise exponentiation (scalar exponent)."""
    if output is None:
        return NO_OUTPUT
    ev = _convert(exponent)
    if ev is None:
        return CONVERSION_FAILURE
    return _elementwise_single(a, output, lambda v, e: v ** e, ev)


def clip(a, low, high, output=None):
    """Element-wise clamping into [low, high]."""
    if output is None:
        return NO_OUTPUT
    fl = _convert(low)
    fh = _convert(high)
    if fl is None or fh is None:
        return CONVERSION_FAILURE

    def _clip(v):
        if v < fl:
            return fl
        if v > fh:
            return fh
        return v

    return _elementwise_single(a, output, _clip)


def flatten(a, output=None):
    """Flatten a tensor into a 1D output (any dimension)."""
    if output is None:
        return NO_OUTPUT
    values = _to_doubles(_flatten(a))
    if values is None:
        return CONVERSION_FAILURE
    return _write_flat(output, values)
