"""Linear algebra support (dimension-generic, duck typing).

Convention: the result is written through the ``output`` keyword argument
(never returned directly - no tensor type hard-coding), and the function
returns an integer status:

    0  success
    1  shape / length mismatch
    2  no output container (output=None) or output write failure
    3  value conversion failure (non-numeric element)

Tensors are any-dimension duck containers built on the sequence protocol
(lists, tuples, custom containers, ...); scalars are written to
``output[0]``.  The C extension (_linear_algebra) provides the same
behaviour.
"""

import operator

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


def _is_container(obj):
    """Sequence protocol: iterable, or sized and indexable; text is a
    value.  A bare ``__getitem__`` without ``__len__`` is scalar-like
    (e.g. a numpy scalar), not a container."""
    if isinstance(obj, (str, bytes)):
        return False
    if hasattr(obj, "__iter__"):
        return True
    return hasattr(obj, "__getitem__") and hasattr(obj, "__len__")


def _infer_shape(data):
    """Infer the shape of a duck-typed tensor (iterative)."""
    shape = []
    obj = data
    while _is_container(obj):
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


def _convert(value):
    """Convert one element to float (None on failure)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _flat_floats(data):
    """Flatten a tensor and convert its elements to floats in one
    iterative walk (None at the first non-numeric element).

    Fast paths cover the common element types (float / int / list /
    tuple); anything else falls back to the duck container check and
    ``float``.
    """
    values = []
    append = values.append
    stack = [iter(data)]
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        kind = type(item)
        if kind is float:
            append(item)
        elif kind is int:
            append(float(item))
        elif kind is list or kind is tuple or _is_container(item):
            stack.append(iter(item))
        else:
            try:
                append(float(item))
            except (TypeError, ValueError):
                return None
    return values


def _sum_squares(values):
    """Naive left-to-right sum of squares (same order as the C side)."""
    total = 0.0
    for value in values:
        total += value * value
    return total


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


def _write_flat(output, values):
    """Write values linearly into output (duck __setitem__)."""
    for i, value in enumerate(values):
        try:
            output[i] = value
        except (TypeError, IndexError, KeyError):
            return NO_OUTPUT
    return OK


def _write_shaped(output, shape, values):
    """Write values back into output following the shape.

    A one-dimensional (or unsized-iterable, shape ()) result is written
    linearly, matching the C side.
    """
    if len(shape) <= 1:
        return _write_flat(output, values)
    for index, value in zip(_iter_indices(shape), values):
        try:
            _set_indexed(output, index, value)
        except (TypeError, IndexError, KeyError):
            return NO_OUTPUT
    return OK


# ---------------------------------------------------------------------------
# scalar-result functions (result written to output[0])
# ---------------------------------------------------------------------------

def dot(a, b, output=None):
    """Element-wise product sum of two tensors (any dimension)."""
    if output is None:
        return NO_OUTPUT
    va = _flat_floats(a)
    vb = _flat_floats(b)
    if va is None or vb is None:
        return CONVERSION_FAILURE
    if len(va) != len(vb):
        return SHAPE_MISMATCH
    total = 0.0
    for x, y in zip(va, vb):
        total += x * y
    return _write_flat(output, (total,))


def norm(a, output=None):
    """Frobenius (Euclidean) norm of a tensor (any dimension)."""
    if output is None:
        return NO_OUTPUT
    va = _flat_floats(a)
    if va is None:
        return CONVERSION_FAILURE
    return _write_flat(output, (_sum_squares(va) ** 0.5,))


def tensor_sum(a, output=None):
    """Sum of all elements of a tensor (any dimension)."""
    if output is None:
        return NO_OUTPUT
    va = _flat_floats(a)
    if va is None:
        return CONVERSION_FAILURE
    return _write_flat(output, (sum(va),))


def tensor_mean(a, output=None):
    """Mean of all elements of a tensor (any dimension)."""
    if output is None:
        return NO_OUTPUT
    va = _flat_floats(a)
    if va is None:
        return CONVERSION_FAILURE
    if not va:
        return _write_flat(output, (0.0,))
    return _write_flat(output, (sum(va) / len(va),))


# ---------------------------------------------------------------------------
# element-wise functions (output keeps the input shape)
# ---------------------------------------------------------------------------

def _elementwise_pair(a, b, output, func):
    """Shared implementation for two-tensor element-wise operations."""
    if output is None:
        return NO_OUTPUT
    va = _flat_floats(a)
    vb = _flat_floats(b)
    if va is None or vb is None:
        return CONVERSION_FAILURE
    if len(va) != len(vb):
        return SHAPE_MISMATCH
    values = (func(x, y) for x, y in zip(va, vb))
    return _write_shaped(output, _infer_shape(a), values)


def _elementwise_single(a, output, func, *args):
    """Shared implementation for single-tensor element-wise operations."""
    if output is None:
        return NO_OUTPUT
    va = _flat_floats(a)
    if va is None:
        return CONVERSION_FAILURE
    values = (func(v, *args) for v in va)
    return _write_shaped(output, _infer_shape(a), values)


def add(a, b, output=None):
    """Element-wise addition of two tensors (same shape)."""
    return _elementwise_pair(a, b, output, operator.add)


def multiply(a, b, output=None):
    """Element-wise multiplication of two tensors (same shape)."""
    return _elementwise_pair(a, b, output, operator.mul)


def scale(a, factor, output=None):
    """Element-wise multiplication by a scalar."""
    if output is None:
        return NO_OUTPUT
    fv = _convert(factor)
    if fv is None:
        return CONVERSION_FAILURE
    return _elementwise_single(a, output, operator.mul, fv)


def normalize(a, output=None):
    """Element-wise division by the tensor norm."""
    if output is None:
        return NO_OUTPUT
    va = _flat_floats(a)
    if va is None:
        return CONVERSION_FAILURE
    length = _sum_squares(va) ** 0.5
    if length == 0.0:
        values = [0.0] * len(va)
    else:
        values = (v / length for v in va)
    return _write_shaped(output, _infer_shape(a), values)


def power(a, exponent, output=None):
    """Element-wise exponentiation (scalar exponent)."""
    if output is None:
        return NO_OUTPUT
    ev = _convert(exponent)
    if ev is None:
        return CONVERSION_FAILURE
    return _elementwise_single(a, output, operator.pow, ev)


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
    values = _flat_floats(a)
    if values is None:
        return CONVERSION_FAILURE
    return _write_flat(output, values)
