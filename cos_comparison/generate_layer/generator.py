"""generate_layer - data generation attached to core algorithms."""

from .. import core

__all__ = (
    "CONVERSION_FAILURE",
    "NO_OUTPUT",
    "OK",
    "SHAPE_MISMATCH",
    "Generator",
    "TensorGenerator",
    "copy_region",
    "transform_self",
)

# status codes (shared with linear_algebra)
OK = 0
SHAPE_MISMATCH = 1
NO_OUTPUT = 2
CONVERSION_FAILURE = 3


def _region_indices(start, shape, step):
    """Iterate region indices (start/shape/step sequences, no recursion)."""
    start = tuple(start or (0,) * len(shape or ()))
    step = tuple(step or (1,) * len(start))
    dims = len(start)
    idx = list(start)
    while True:
        yield tuple(idx)
        for d in range(dims - 1, -1, -1):
            idx[d] += step[d]
            if dims == 0 or shape is None or idx[d] < start[d] + shape[d]:
                break
            idx[d] = start[d]
        else:
            break


def _read_region(data, start, shape, step):
    """Read a region via core.get_item into a value list."""
    return [core.get_item(data, i) for i in _region_indices(start, shape, step)]


def _region_view(output, out_start, shape, out_step):
    """Write-region slice view of output (out_start/out_step/shape)."""
    out_start = tuple(out_start)
    out_step = tuple(out_step or (1,) * len(out_start))
    sl = []
    for d, s in enumerate(out_start):
        end = None
        if shape is not None and d < len(shape) and shape[d] is not None:
            end = s + shape[d] * out_step[d]
        sl.append(slice(s, end, out_step[d]))
    return output[tuple(sl)]


def transform_self(tensor, func, *others, start=None, shape=None, step=None,
                   out_start=None, out_step=None):
    """Self-modifying element-wise transform: the read region is extracted
    via core.get_item, passed to core.elementwise, and the result is
    written back INTO the tensor itself (full write, an output slice view
    with set_item fallback for a write region, or set_item over the read
    region); returns a status code."""
    tensors = (tensor,) + others
    work = tensors if start is None else tuple(
        _read_region(t, start, shape, step) for t in tensors)
    if out_start is not None:
        try:
            view = _region_view(tensor, out_start, shape, out_step)
        except (TypeError, IndexError, AttributeError):
            view = None
        if view is not None:
            try:
                return core.elementwise(*work, func=func, output=view)
            except (TypeError, ValueError):
                return SHAPE_MISMATCH
        for out_idx, values in zip(
                _region_indices(out_start, shape, out_step), zip(*work)):
            try:
                core.set_item(tensor, out_idx, func(*values))
            except Exception:  # noqa: BLE001 - status through return value
                return NO_OUTPUT
        return OK
    if start is None:
        try:
            return core.elementwise(*work, func=func, output=tensor)
        except (TypeError, ValueError):
            return SHAPE_MISMATCH
    temp = [None] * (len(work[0]) if work else 0)
    try:
        core.elementwise(*work, func=func, output=temp)
    except (TypeError, ValueError):
        return SHAPE_MISMATCH
    for idx, v in zip(_region_indices(start, shape, step), temp):
        try:
            core.set_item(tensor, idx, v)
        except Exception:  # noqa: BLE001 - status through return value
            return NO_OUTPUT
    return OK


class Generator:
    __slots__ = ("data",)
    def __init__(self, data):
        self.data = data
    def fix(self, call, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        return call(self.data, *args, **kwargs)


class TensorGenerator(Generator):
    """Tensor generator: generation and self-modification over wrapped
    data through core algorithms."""

    def __init__(self, data):
        super().__init__(data)

    def generate(self, func, args=(), kwargs=None):
        """Uniform delegation entry: func(self.data, *args, **kwargs)."""
        kwargs = {} if kwargs is None else kwargs
        return func(self.data, *args, **kwargs)

    def set_point(self, index, value):
        """Write one element via the core set_item protocol."""
        return core.set_item(self.data, index, value)

    def transform_self(self, func, *others, start=None, shape=None, step=None,
                       out_start=None, out_step=None):
        """Self-modify: element-wise transform with output = self (or a
        write region of it); returns a status code."""
        return transform_self(self.data, func, *others, start=start,
                              shape=shape, step=step, out_start=out_start,
                              out_step=out_step)


#generate functions.
def copy_region(target, source, *, shape=None, source_start=None, source_step=None,
                target_start=None, target_step=None):
    """Direct region fill: copy a sub-region of source into target's
    corresponding positions (core.load_data wrapper; each side has its own
    start/step, out-of-bounds silently clipped). Returns elements copied.
    Target comes first so generate() fills the owned data naturally."""
    return core.load_data(source, target, shape=shape, source_start=source_start,
                          source_step=source_step, target_start=target_start,
                          target_step=target_step)
