"""
Fourier transforms (procedural, recursion-free, multi-dimensional).

Generic formula (real/imag split on floats):
    F(ξ) = Σ f(x)·cos(2πξ·x) − i·Σ f(x)·sin(2πξ·x)
    f(x) = (1/N)·Σ F(ξ)·cos(2πξ·x) + i·(1/N)·Σ F(ξ)·sin(2πξ·x)
"""

import math

__all__ = (
    "dft",
    "dft_kernel_imag",
    "dft_kernel_real",
    "idft",
    "power_spectrum",
)


def _split(value):
    if isinstance(value, complex):
        return value.real, value.imag
    if isinstance(value, (tuple, list)):
        return float(value[0]), float(value[1])
    return float(value), 0.0


def _transform_1d(re, im, inverse):
    """Generic trig-variant transform on float lists, in place."""
    n = len(re)
    if n <= 1:
        return
    sign = 1.0 if inverse else -1.0
    out_re = [0.0] * n
    out_im = [0.0] * n
    for k in range(n):
        sr = si = 0.0
        for j in range(n):
            t = sign * 2.0 * math.pi * k * j / n
            cr = math.cos(t)
            sn = math.sin(t)
            sr += re[j] * cr - im[j] * sn
            si += re[j] * sn + im[j] * cr
        out_re[k] = sr
        out_im[k] = si
    re[:] = out_re
    im[:] = out_im


def _shape_of(data):
    shape = []
    obj = data
    while True:
        try:
            n = len(obj)
        except TypeError:
            break
        shape.append(n)
        if n == 0:
            break
        obj = obj[0]
    return tuple(shape)


def _axes(shape, axis):
    dim = len(shape)
    if axis is None:
        return tuple(range(dim))
    if isinstance(axis, int):
        return (axis % dim,)
    return tuple(a % dim for a in axis)


def _idx(fixed, axis, k, dim):
    return [k if i == axis else fixed[i] for i in range(dim)]


def _axes_transform(data, shape, axes, inverse):
    """Apply the 1-D transform along each axis (odometer over fixed coords)."""
    dim = len(shape)
    for axis in axes:
        size = shape[axis]
        if size <= 1:
            continue
        total = 1
        for i in range(dim):
            if i != axis:
                total *= shape[i]
        fixed = [0] * dim
        for _ in range(total):
            re_line, im_line = [], []
            for k in range(size):
                obj = data
                for p in _idx(fixed, axis, k, dim):
                    obj = obj[p]
                r, i = _split(obj)
                re_line.append(r)
                im_line.append(i)
            _transform_1d(re_line, im_line, inverse)
            for k in range(size):
                obj = data
                idx = _idx(fixed, axis, k, dim)
                for p in idx[:-1]:
                    obj = obj[p]
                obj[idx[-1]] = complex(re_line[k], im_line[k])
            for i in range(dim):
                if i == axis:
                    continue
                fixed[i] += 1
                if fixed[i] < shape[i]:
                    break
                fixed[i] = 0
    return data


def _mutable(data, shape):
    if not shape:
        return data

    def build(dims, obj):
        # iterative deep copy by shape, no recursion
        if len(dims) == 1:
            return [obj[i] for i in range(dims[0])]
        result = [None] * dims[0]
        stack = [(result, obj, 0)]
        while stack:
            dst, src, level = stack.pop()
            if level == len(dims) - 1:
                for j in range(dims[level]):
                    dst[j] = src[j]
            else:
                for i in range(dims[level]):
                    child = [None] * dims[level + 1]
                    dst[i] = child
                    stack.append((child, src[i], level + 1))
        return result

    return build(shape, data)


def _scale(data, factor):
    # explicit stack walk, no recursion
    if not isinstance(data, list):
        return data * factor
    root = [None] * len(data)
    stack = [(data, root)]
    while stack:
        src, dst = stack.pop()
        for i, item in enumerate(src):
            if isinstance(item, list):
                sub = [None] * len(item)
                dst[i] = sub
                stack.append((item, sub))
            else:
                dst[i] = item * factor
    return root


def dft(data, axis=None):
    """Generic DFT (trig formula), multi-dimensional (axis=None: all)."""
    shape = _shape_of(data)
    return _axes_transform(_mutable(data, shape), shape, _axes(shape, axis),
                           inverse=False)


def idft(data, axis=None):
    """Generic IDFT (1/N normalized), multi-dimensional (axis=None: all)."""
    shape = _shape_of(data)
    n = 1
    for s in shape:
        n *= s
    work = _axes_transform(_mutable(data, shape), shape, _axes(shape, axis),
                           inverse=True)
    return _scale(work, 1.0 / n)


def power_spectrum(data, axis=None):
    """|X[k]|^2 per element (no normalization)."""
    spectrum = dft(data, axis=axis)

    def walk(obj):
        # explicit stack walk, no recursion
        if not isinstance(obj, list):
            return abs(obj) * abs(obj)
        root = [None] * len(obj)
        stack = [(obj, root)]
        while stack:
            src, dst = stack.pop()
            for i, item in enumerate(src):
                if isinstance(item, list):
                    sub = [None] * len(item)
                    dst[i] = sub
                    stack.append((item, sub))
                else:
                    dst[i] = abs(item) * abs(item)
        return root

    return walk(spectrum)


# ---------------------------------------------------------------------------
# DFT kernels (real / imag), multi-dimensional
# ---------------------------------------------------------------------------

def _dft_kernel(shape, frequencies, func, scales=1.0,
                offsets=0.0, amplitudes=1.0, biases=0.0):
    """Pointwise kernel over the shape.

    y[i] = amplitudes[i]*func(2*pi*sum_d(scales[d]*frequencies[d]*x_d/N_d)
           + sum_d(offsets[d])) + biases[i]

    frequencies/scales/offsets are per-axis (scalar broadcast);
    amplitudes/biases are per-element (scalar broadcast).
    """
    shape = tuple(shape)
    dim = len(shape)
    if len(frequencies) != dim:
        raise ValueError("frequencies length must match shape")
    for k, n in zip(frequencies, shape):
        if not (0 <= k < n):
            raise ValueError("frequencies entries must be within shape")
    if isinstance(scales, (int, float)):
        scales = [scales] * dim
    else:
        scales = list(scales)
        if len(scales) != dim:
            raise ValueError("scales length must match shape")
    if isinstance(offsets, (int, float)):
        offsets = [offsets] * dim
    else:
        offsets = list(offsets)
        if len(offsets) != dim:
            raise ValueError("offsets length must match shape")
    total = 1
    for n in shape:
        total *= n
    if isinstance(amplitudes, (int, float)):
        amplitudes = [amplitudes] * total
    else:
        amplitudes = list(amplitudes)
        if len(amplitudes) != total:
            raise ValueError("amplitudes length must match element count")
    if isinstance(biases, (int, float)):
        biases = [biases] * total
    else:
        biases = list(biases)
        if len(biases) != total:
            raise ValueError("biases length must match element count")

    def build(dims):
        # top-down explicit stack build, no recursion
        if len(dims) == 1:
            return [0.0] * dims[0]
        root = [None] * dims[0]
        stack = [(root, 1)]
        while stack:
            node, level = stack.pop()
            for i in range(len(node)):
                if level == len(dims) - 1:
                    node[i] = [0.0] * dims[level]
                else:
                    child = [None] * dims[level]
                    node[i] = child
                    stack.append((child, level + 1))
        return root

    work = build(shape)
    idx = [0] * dim
    offset_sum = sum(offsets)
    for t in range(total):
        phase = 0.0
        for i in range(dim):
            phase += scales[i] * frequencies[i] * idx[i] / shape[i]
        obj = work
        for p in idx[:-1]:
            obj = obj[p]
        obj[idx[-1]] = amplitudes[t] * func(2.0 * math.pi * phase + offset_sum) \
            + biases[t]
        for i in range(dim - 1, -1, -1):
            idx[i] += 1
            if idx[i] < shape[i]:
                break
            idx[i] = 0
    return work


def _dims(shape, frequencies):
    if isinstance(shape, int):
        return (shape,), (frequencies,)
    return tuple(shape), tuple(frequencies)


def dft_kernel_real(shape, frequencies, scales=1.0, offsets=0.0,
                     amplitudes=1.0, biases=0.0):
    """Real-part DFT kernel (cos) with per-axis affine customization.

    y[i] = amplitudes[i]*cos(2*pi*sum_d(scales[d]*frequencies[d]*x_d/N_d)
           + sum_d(offsets[d])) + biases[i]

    frequencies/scales/offsets are per-axis (scalar broadcast for 1-D);
    amplitudes/biases are per-element (scalar broadcast).
    """
    shape, frequencies = _dims(shape, frequencies)
    return _dft_kernel(shape, frequencies, math.cos, scales=scales,
                       offsets=offsets, amplitudes=amplitudes,
                       biases=biases)


def dft_kernel_imag(shape, frequencies, scales=1.0, offsets=0.0,
                     amplitudes=1.0, biases=0.0):
    """Imag-part DFT kernel (sin) with per-axis affine customization.

    y[i] = amplitudes[i]*sin(2*pi*sum_d(scales[d]*frequencies[d]*x_d/N_d)
           + sum_d(offsets[d])) + biases[i]

    frequencies/scales/offsets are per-axis (scalar broadcast for 1-D);
    amplitudes/biases are per-element (scalar broadcast).
    """
    shape, frequencies = _dims(shape, frequencies)
    return _dft_kernel(shape, frequencies, math.sin, scales=scales,
                       offsets=offsets, amplitudes=amplitudes,
                       biases=biases)
