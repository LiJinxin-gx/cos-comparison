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


def _region_spec(data_shape, start, shape, step):
    """Resolve a sampled region to (start, step, effective) with core
    region semantics: shape counts samples and is clipped to the data
    bounds; shape=None selects the full data adjusted for the step."""
    dimension = len(data_shape)
    start = tuple(start) if start is not None else (0,) * dimension
    if len(start) != dimension:
        raise ValueError("start length does not match data dimension")
    for v in start:
        if v < 0:
            raise ValueError("start entries must be non-negative")
    step = tuple(step) if step is not None else (1,) * dimension
    if len(step) != dimension:
        raise ValueError("step length does not match data dimension")
    for v in step:
        if v <= 0:
            raise ValueError("step entries must be positive")
    if shape is None:
        effective = tuple(
            (data_shape[i] - start[i] + step[i] - 1) // step[i]
            if start[i] < data_shape[i] else 0 for i in range(dimension))
        return start, step, effective
    shape = tuple(shape)
    if len(shape) != dimension:
        raise ValueError("shape length does not match data dimension")
    effective = []
    for i in range(dimension):
        if shape[i] < 0:
            raise ValueError("shape entries cannot be negative")
        avail = ((data_shape[i] - start[i] + step[i] - 1) // step[i]
                 if start[i] < data_shape[i] else 0)
        effective.append(min(shape[i], avail))
    return start, step, tuple(effective)


def _region_indices(start, shape, step):
    """Iterate shape sampled positions from start with step (row-major)."""
    dimension = len(start)
    total = 1
    for n in shape:
        total *= n
    idx = [0] * dimension
    for _ in range(total):
        yield tuple(start[i] + idx[i] * step[i] for i in range(dimension))
        for d in range(dimension - 1, -1, -1):
            idx[d] += 1
            if idx[d] < shape[d]:
                break
            idx[d] = 0


def _read_region(data, start, shape, step):
    """Read a sampled region via core.get_item into a value list."""
    return [core.get_item(data, i)
            for i in _region_indices(start, shape, step)]


def _write_spec(out_start, out_step):
    """Validate a write region: (out_start, out_step) with non-negative
    start and positive step."""
    out_start = tuple(out_start)
    dimension = len(out_start)
    out_step = tuple(out_step) if out_step is not None else (1,) * dimension
    if len(out_step) != dimension:
        raise ValueError("out_step length does not match out_start length")
    for v in out_start:
        if v < 0:
            raise ValueError("out_start entries must be non-negative")
    for v in out_step:
        if v <= 0:
            raise ValueError("out_step entries must be positive")
    return out_start, out_step


def _region_view(output, out_start, shape, out_step):
    """Write-region slice view of output (start/step/shape are validated)."""
    return output[tuple(
        slice(out_start[d], out_start[d] + shape[d] * out_step[d],
              out_step[d]) for d in range(len(out_start)))]


def transform_self(tensor, func, *others, start=None, shape=None, step=None,
                   out_start=None, out_step=None):
    """Self-modifying element-wise transform: the read region (start/shape/
    step, core region semantics: shape counts samples and is clipped) is
    extracted via core.get_item, passed to core.elementwise, and the
    result is written back INTO the tensor itself (full write, an output
    slice view with set_item fallback for a write region, or set_item over
    the read region); returns a status code."""
    tensors = (tensor,) + others
    if start is None:
        work, effective = tensors, None
    else:
        data_shape = core.infer_shape(tensor)
        if data_shape is None:
            return SHAPE_MISMATCH
        try:
            r_start, r_step, effective = _region_spec(
                data_shape, start, shape, step)
            work = tuple(_read_region(t, r_start, effective, r_step)
                         for t in tensors)
        except (TypeError, ValueError, IndexError):
            return SHAPE_MISMATCH
    if out_start is None:
        if start is None:
            try:
                return core.elementwise(*work, func=func, output=tensor)
            except (TypeError, ValueError):
                return SHAPE_MISMATCH
        temp = [None] * len(work[0])
        try:
            core.elementwise(*work, func=func, output=temp)
        except (TypeError, ValueError):
            return SHAPE_MISMATCH
        for idx, v in zip(_region_indices(r_start, effective, r_step), temp):
            try:
                core.set_item(tensor, idx, v)
            except Exception:  # noqa: BLE001 - status through return value
                return NO_OUTPUT
        return OK
    if effective is None:
        effective = core.infer_shape(work[0])
        if effective is None:
            return SHAPE_MISMATCH
    try:
        out_start, out_step = _write_spec(out_start, out_step)
    except ValueError:
        return SHAPE_MISMATCH
    if len(out_start) != len(effective):
        return SHAPE_MISMATCH
    try:
        view = _region_view(tensor, out_start, effective, out_step)
    except (TypeError, IndexError, AttributeError):
        view = None
    if view is not None:
        try:
            if core.infer_shape(view) == core.infer_shape(work[0]):
                return core.elementwise(*work, func=func, output=view)
        except (TypeError, ValueError):
            return SHAPE_MISMATCH
    for out_idx, values in zip(
            _region_indices(out_start, effective, out_step), zip(*work)):
        try:
            core.set_item(tensor, out_idx, func(*values))
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
