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


def elementwise_extract(*tensors, func=None, output=None,
                        start=None, shape=None, step=None,
                        out_start=None, out_step=None):
    """Unified element-wise extraction: the read region (start/shape/step)
    is extracted via core.get_item, passed to core.elementwise; without a
    write region the output is written fully, with one it tries an output
    slice view first and falls back to core.set_item.  Returns a status
    code (output=None is decided by the underlying elementwise)."""
    work = tensors if start is None else tuple(
        _read_region(t, start, shape, step) for t in tensors)
    if out_start is None:
        try:
            return core.elementwise(*work, func=func, output=output)
        except (TypeError, ValueError):
            return SHAPE_MISMATCH
    try:
        view = _region_view(output, out_start, shape, out_step)
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
