"""sense_layer - sensory stimulus reception attached to core algorithms."""

from .. import core

__all__ = (
    "CONVERSION_FAILURE",
    "NO_OUTPUT",
    "OK",
    "SHAPE_MISMATCH",
    "Receptor",
    "TensorReceptor",
    "data_match",
    "elementwise_extract",
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


def elementwise_extract(*tensors, func=None, output=None,
                        start=None, shape=None, step=None,
                        out_start=None, out_step=None):
    """Unified element-wise extraction: the read region (start/shape/step,
    core region semantics: shape counts samples and is clipped) is
    extracted via core.get_item, passed to core.elementwise; with a write
    region the output slice view is tried first and core.set_item is the
    fallback.  Returns a status code (output=None is decided by the
    underlying elementwise)."""
    if not tensors:
        return SHAPE_MISMATCH
    if start is None:
        work, effective = tensors, None
    else:
        data_shape = core.infer_shape(tensors[0])
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
        try:
            return core.elementwise(*work, func=func, output=output)
        except (TypeError, ValueError):
            if output is None or effective is None or len(effective) <= 1:
                return SHAPE_MISMATCH
        temp = [None] * len(work[0])
        try:
            core.elementwise(*work, func=func, output=temp)
        except (TypeError, ValueError):
            return SHAPE_MISMATCH
        if core.infer_shape(output) != effective:
            return SHAPE_MISMATCH
        for idx, v in zip(
                _region_indices((0,) * len(effective), effective,
                                (1,) * len(effective)), temp):
            try:
                core.set_item(output, idx, v)
            except Exception:  # noqa: BLE001 - status through return value
                return NO_OUTPUT
        return OK
    if output is None:
        return NO_OUTPUT
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
        view = _region_view(output, out_start, effective, out_step)
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
            core.set_item(output, out_idx, func(*values))
        except Exception:  # noqa: BLE001 - status through return value
            return NO_OUTPUT
    return OK


class Receptor:
    __slots__ = ("data",)
    def __init__(self, data):
        self.data = data
    def initialize(self, caller, *args, **kwargs):
        return caller(self.data, *args, **kwargs)
    def receptor(self, caller, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        return caller(self.data, *args, **kwargs)


class TensorReceptor(Receptor):
    """Tensor receptor: core algorithms attached to the wrapped data."""

    def __init__(self, data):
        super().__init__(data)

    def point(self, index):
        return core.get_item(self.data, index)

    def comparison_passive(self, output=None, **kwargs):
        return core.cos_comparison_passive(self.data, output=output, **kwargs)

    def comparison_active(self, kernel=None, output=None, **kwargs):
        return core.cos_comparison_active(
            self.data, kernel=kernel, output=output, **kwargs)

    def threshold_map(self, pairs, default_value=0.0, output=None,
                      start=None, shape=None, step=None,
                      out_start=None, out_step=None):
        """Threshold sensing map: the read region is mapped through the
        (func, value) pair sequence (first truthy func wins) into output
        (write region); returns a status code."""
        if output is None:
            return NO_OUTPUT

        def matcher(value):
            for func, mapped in pairs:
                try:
                    if func(value):
                        return mapped
                except Exception:  # noqa: BLE001, S112 - skipped like core
                    continue
            return default_value

        kwargs = {"out": output}
        for name, val in (("start", start), ("shape", shape),
                          ("step", step), ("out_start", out_start),
                          ("out_step", out_step)):
            if val is not None:
                kwargs[name] = val
        try:
            core.data_mapping(self.data, matcher, **kwargs)
        except (TypeError, ValueError):
            return SHAPE_MISMATCH
        except Exception:  # noqa: BLE001 - status through return value
            return CONVERSION_FAILURE
        return OK

    def threshold_match(self, low=None, high=None, inclusive=(True, True),
                        start=None, shape=None, step=None):
        """Threshold position iterator: data_filter + threshold_judge
        (lazily yields matching positions)."""
        kwargs = {}
        for name, val in (("start", start), ("shape", shape),
                          ("step", step)):
            if val is not None:
                kwargs[name] = val
        judge = core.threshold_judge(low, high, inclusive=inclusive)
        return core.data_filter(self.data, judge, **kwargs)


def data_match(data, template, start=None, end=None, step=None, algorithm=None,
               low=None, high=None, inclusive=(True, True)):
    """Match data against a template via the core active comparison; yields
    the output positions whose match value lies in [low, high] (interval
    optional; inclusive controls endpoint membership).  None region params
    are omitted for uniform backend handling; algorithm=None uses the core
    default."""
    kwargs = {}
    if start is not None:
        kwargs["start"] = start
    if end is not None:
        kwargs["end"] = end
    if step is not None:
        kwargs["step"] = step
    if algorithm is not None:
        kwargs["algorithm"] = algorithm
    out = core.cos_comparison_active(data, kernel=template, **kwargs)
    if low is None and high is None:
        return core.data_filter(out, lambda value: True)
    return core.threshold_filter(out, low, high, inclusive=inclusive)
