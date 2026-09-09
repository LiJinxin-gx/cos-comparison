/* _fourier.c - C99 optimized Fourier transforms and DFT kernels.
 *
 * Backend for cos_comparison.interface.tools.math_tool._fourier.
 * Behaviour is identical to the pure-Python fourier module; the numeric
 * cores (1-D trig transform, kernel generation) are implemented natively
 * on C double arrays.
 *
 * C99 features used deliberately: declaration-in-for, long long element
 * counts, iterative (recursion-free) odometer walks, explicit free on
 * every path, portable M_PI fallback.
 */
#define PY_SSIZE_T_CLEAN
#ifdef _MSC_VER
/* CPython extension ABI patterns under /Wall (see cos_comparison_pydll.c):
   C4191 method-table casts, C4232 PyType_GenericNew address,
   C4820 struct padding; C5045/C4711/C4710 are /Wall performance hints. */
#pragma warning(push)
#pragma warning(disable: 4191 4232 4820 5045 4711 4710)
#endif
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

/* ==================================================================
 * 1-D trig transform on C double arrays (in place)
 * ================================================================== */

static void dft_1d(double *re, double *im, Py_ssize_t n, double sign) {
    if (n <= 1) return;
    double *out_re = (double*)malloc((size_t)n * sizeof(double));
    double *out_im = (double*)malloc((size_t)n * sizeof(double));
    if (!out_re || !out_im) {
        free(out_re);
        free(out_im);
        return;   /* caller detects via PyErr */
    }
    for (Py_ssize_t k = 0; k < n; ++k) {
        double sr = 0.0, si = 0.0;
        for (Py_ssize_t j = 0; j < n; ++j) {
            double t = sign * 2.0 * M_PI * (double)k * (double)j / (double)n;
            double cr = cos(t);
            double sn = sin(t);
            sr += re[j] * cr - im[j] * sn;
            si += re[j] * sn + im[j] * cr;
        }
        out_re[k] = sr;
        out_im[k] = si;
    }
    memcpy(re, out_re, (size_t)n * sizeof(double));
    memcpy(im, out_im, (size_t)n * sizeof(double));
    free(out_re);
    free(out_im);
}

/* ==================================================================
 * Nested list helpers (list fast path, sequence fallback)
 * ================================================================== */

static PyObject *get_nested(PyObject *data, const long *idx, int dim) {
    PyObject *obj = data;
    for (int i = 0; i < dim; ++i) {
        if (PyList_Check(obj)) {
            Py_ssize_t p = (Py_ssize_t)idx[i];
            if (p < 0 || p >= PyList_GET_SIZE(obj)) {
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return NULL;
            }
            obj = PyList_GET_ITEM(obj, p);
        } else {
            PyObject *sub = PySequence_GetItem(obj, (Py_ssize_t)idx[i]);
            if (!sub) return NULL;
            obj = sub;
        }
    }
    return obj;
}

static PyObject *split(PyObject *value, double *re, double *im) {
    if (PyComplex_Check(value)) {
        *re = PyComplex_RealAsDouble(value);
        *im = PyComplex_ImagAsDouble(value);
        return PyErr_Occurred() ? NULL : value;
    }
    if (PySequence_Check(value) && !PyBytes_Check(value) &&
        !PyUnicode_Check(value)) {
        PyObject *a = PySequence_GetItem(value, 0);
        PyObject *b = PySequence_GetItem(value, 1);
        if (!a || !b) { Py_XDECREF(a); Py_XDECREF(b); return NULL; }
        *re = PyFloat_AsDouble(a);
        *im = PyFloat_AsDouble(b);
        Py_DECREF(a); Py_DECREF(b);
        return PyErr_Occurred() ? NULL : value;
    }
    *re = PyFloat_AsDouble(value);
    *im = 0.0;
    return PyErr_Occurred() ? NULL : value;
}

/* shape inference: walk nested sequences until a non-sequence */
static PyObject *shape_of(PyObject *data) {
    PyObject *shape = PyList_New(0);
    if (!shape) return NULL;
    PyObject *obj = data;
    int guard = 0;
    while (1) {
        Py_ssize_t len_o = PyObject_Size(obj);
        if (len_o < 0) {
            PyErr_Clear();
            break;
        }
        Py_ssize_t n = len_o;
        if (PyList_Append(shape, PyLong_FromSsize_t(n)) < 0) {
            Py_DECREF(shape);
            return NULL;
        }
        if (n == 0) break;
        obj = get_nested(obj, (long[]){0}, 1);
        if (!obj) { Py_DECREF(shape); return NULL; }
        if (++guard > 10000) { Py_DECREF(shape); break; }
    }
    return shape;
}

/* normalize axis argument into a list of normalized axes */
static PyObject *resolve_axes(PyObject *shape_list, PyObject *axis, Py_ssize_t *dim_out) {
    Py_ssize_t dim = PyList_GET_SIZE(shape_list);
    *dim_out = dim;
    PyObject *axes = PyList_New(0);
    if (!axes) return NULL;
    if (axis == Py_None) {
        for (Py_ssize_t i = 0; i < dim; ++i)
            if (PyList_Append(axes, PyLong_FromSsize_t(i)) < 0) {
                Py_DECREF(axes);
                return NULL;
            }
        return axes;
    }
    if (PyLong_Check(axis)) {
        Py_ssize_t a = PyLong_AsSsize_t(axis);
        if (PyErr_Occurred()) { Py_DECREF(axes); return NULL; }
        a %= dim;
        if (a < 0) a += dim;
        if (PyList_Append(axes, PyLong_FromSsize_t(a)) < 0) {
            Py_DECREF(axes);
            return NULL;
        }
        return axes;
    }
    PyObject *it = PyObject_GetIter(axis);
    if (!it) { Py_DECREF(axes); return NULL; }
    PyObject *item;
    while ((item = PyIter_Next(it))) {
        Py_ssize_t a = PyLong_AsSsize_t(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) { Py_DECREF(it); Py_DECREF(axes); return NULL; }
        a %= dim;
        if (a < 0) a += dim;
        if (PyList_Append(axes, PyLong_FromSsize_t(a)) < 0) {
            Py_DECREF(it); Py_DECREF(axes);
            return NULL;
        }
    }
    Py_DECREF(it);
    if (PyErr_Occurred()) { Py_DECREF(axes); return NULL; }
    return axes;
}

/* apply the 1-D transform along the given axis (odometer over fixed
 * coords); data must be a mutable nested list */
static int axes_transform_one(PyObject *data, PyObject *shape_list,
                               Py_ssize_t axis, double sign) {
    Py_ssize_t dim = PyList_GET_SIZE(shape_list);
    Py_ssize_t size = PyLong_AsSsize_t(PyList_GET_ITEM(shape_list, axis));
    if (PyErr_Occurred()) return -1;
    if (size <= 1) return 0;
    long long total = 1;
    for (Py_ssize_t i = 0; i < dim; ++i)
        if (i != axis) total *= (long long)PyLong_AsSsize_t(
            PyList_GET_ITEM(shape_list, i));
    if (PyErr_Occurred()) return -1;

    long *fixed = (long*)malloc((size_t)(dim > 0 ? dim : 1) * sizeof(long));
    if (!fixed) { PyErr_NoMemory(); return -1; }
    memset(fixed, 0, (size_t)(dim > 0 ? dim : 1) * sizeof(long));

    double *re = (double*)malloc((size_t)size * sizeof(double));
    double *im = (double*)malloc((size_t)size * sizeof(double));
    if (!re || !im) {
        free(fixed); free(re); free(im);
        PyErr_NoMemory();
        return -1;
    }

    long long f_count;
    for (f_count = 0; f_count < total; ++f_count) {
        /* extract the line along axis */
        for (Py_ssize_t k = 0; k < size; ++k) {
            long buf[32];
            for (Py_ssize_t i = 0; i < dim; ++i)
                buf[i] = (i == axis) ? (long)k : fixed[i];
            PyObject *obj = data;
            int ok = 1;
            for (Py_ssize_t i = 0; i < dim; ++i) {
                if (PyList_Check(obj)) {
                    Py_ssize_t p = buf[i];
                    if (p < 0 || p >= PyList_GET_SIZE(obj)) {
                        ok = 0;
                        break;
                    }
                    obj = PyList_GET_ITEM(obj, p);
                } else {
                    PyObject *sub = PySequence_GetItem(obj, buf[i]);
                    if (!sub) { ok = 0; break; }
                    obj = sub;
                }
            }
            if (!ok) {
                free(fixed); free(re); free(im);
                return -1;
            }
            if (split(obj, &re[k], &im[k]) == NULL) {
                free(fixed); free(re); free(im);
                return -1;
            }
        }
        dft_1d(re, im, size, sign);
        /* write back complex values */
        for (Py_ssize_t k = 0; k < size; ++k) {
            long buf[32];
            for (Py_ssize_t i = 0; i < dim; ++i)
                buf[i] = (i == axis) ? (long)k : fixed[i];
            PyObject *obj = data;
            int ok = 1;
            for (Py_ssize_t i = 0; i < dim - 1; ++i) {
                if (PyList_Check(obj)) {
                    Py_ssize_t p = buf[i];
                    if (p < 0 || p >= PyList_GET_SIZE(obj)) { ok = 0; break; }
                    obj = PyList_GET_ITEM(obj, p);
                } else {
                    PyObject *sub = PySequence_GetItem(obj, buf[i]);
                    if (!sub) { ok = 0; break; }
                    obj = sub;
                }
            }
            if (!ok) {
                free(fixed); free(re); free(im);
                return -1;
            }
            PyObject *cval = PyComplex_FromDoubles(re[k], im[k]);
            if (!cval) {
                free(fixed); free(re); free(im);
                return -1;
            }
            if (PyList_Check(obj)) {
                PyList_SET_ITEM(obj, buf[dim - 1], cval);
            } else {
                int setr = PyObject_SetItem(obj, PyLong_FromSsize_t(buf[dim - 1]), cval);
                Py_DECREF(cval);
                if (setr < 0) {
                    free(fixed); free(re); free(im);
                    return -1;
                }
            }
        }
        /* advance fixed (odometer over non-axis dims) */
        for (Py_ssize_t i = 0; i < dim; ++i) {
            if (i == axis) continue;
            fixed[i]++;
            if (fixed[i] < PyLong_AsSsize_t(PyList_GET_ITEM(shape_list, i)))
                break;
            fixed[i] = 0;
            if (PyErr_Occurred()) {
                free(fixed); free(re); free(im);
                return -1;
            }
        }
    }
    free(fixed);
    free(re);
    free(im);
    return 0;
}

/* deep mutable copy of data by shape (iterative, no recursion) */
static PyObject *deep_mutable(PyObject *data, PyObject *shape_list) {
    Py_ssize_t dim = PyList_GET_SIZE(shape_list);
    if (dim == 0) {
        Py_INCREF(data);
        return data;
    }
    Py_ssize_t d0 = PyLong_AsSsize_t(PyList_GET_ITEM(shape_list, 0));
    if (PyErr_Occurred()) return NULL;
    PyObject *root = PyList_New(d0);
    if (!root) return NULL;
    /* stack of (dst_list, src_obj, level) */
    PyObject *stack = PyList_New(0);
    if (!stack) { Py_DECREF(root); return NULL; }
    PyObject *init = PyTuple_Pack(3, root, data, PyLong_FromLong(0));
    if (!init || PyList_Append(stack, init) < 0) {
        Py_XDECREF(init); Py_DECREF(root); Py_DECREF(stack);
        return NULL;
    }
    Py_DECREF(init);
    while (PyList_GET_SIZE(stack) > 0) {
        PyObject *top = PyList_GET_ITEM(stack, PyList_GET_SIZE(stack) - 1);
        Py_INCREF(top);
        PyList_SetSlice(stack, PyList_GET_SIZE(stack) - 1,
                        PyList_GET_SIZE(stack), NULL);
        PyObject *dst = PyTuple_GET_ITEM(top, 0);
        PyObject *src = PyTuple_GET_ITEM(top, 1);
        long level = PyLong_AsLong(PyTuple_GET_ITEM(top, 2));
        Py_DECREF(top);
        if (PyErr_Occurred()) { Py_DECREF(root); Py_DECREF(stack); return NULL; }
        Py_ssize_t n = PyLong_AsSsize_t(PyList_GET_ITEM(shape_list, level));
        if (PyErr_Occurred()) { Py_DECREF(root); Py_DECREF(stack); return NULL; }
        if (level == dim - 1) {
            for (Py_ssize_t j = 0; j < n; ++j) {
                PyObject *item = PySequence_GetItem(src, j);
                if (!item) { Py_DECREF(root); Py_DECREF(stack); return NULL; }
                PyList_SET_ITEM(dst, j, item);
            }
        } else {
            for (Py_ssize_t i = 0; i < n; ++i) {
                Py_ssize_t child_n = PyLong_AsSsize_t(
                    PyList_GET_ITEM(shape_list, level + 1));
                if (PyErr_Occurred()) { Py_DECREF(root); Py_DECREF(stack); return NULL; }
                PyObject *child = PyList_New(child_n);
                if (!child) { Py_DECREF(root); Py_DECREF(stack); return NULL; }
                PyList_SET_ITEM(dst, i, child);
                PyObject *sub = PySequence_GetItem(src, i);
                if (!sub) { Py_DECREF(root); Py_DECREF(stack); return NULL; }
                PyObject *frame = PyTuple_Pack(3, child, sub, PyLong_FromLong(level + 1));
                Py_DECREF(sub);
                if (!frame || PyList_Append(stack, frame) < 0) {
                    Py_XDECREF(frame); Py_DECREF(root); Py_DECREF(stack);
                    return NULL;
                }
                Py_DECREF(frame);
            }
        }
    }
    Py_DECREF(stack);
    return root;
}

/* scale every leaf by factor (iterative) */
static PyObject *scale(PyObject *data, double factor) {
    if (!PyList_Check(data)) {
        return PyFloat_FromDouble(PyFloat_AsDouble(data) * factor);
    }
    Py_ssize_t n = PyList_GET_SIZE(data);
    PyObject *root = PyList_New(n);
    if (!root) return NULL;
    PyObject *stack = PyList_New(0);
    if (!stack) { Py_DECREF(root); return NULL; }
    PyObject *init = PyTuple_Pack(2, data, root);
    if (!init || PyList_Append(stack, init) < 0) {
        Py_XDECREF(init); Py_DECREF(root); Py_DECREF(stack);
        return NULL;
    }
    Py_DECREF(init);
    while (PyList_GET_SIZE(stack) > 0) {
        PyObject *top = PyList_GET_ITEM(stack, PyList_GET_SIZE(stack) - 1);
        Py_INCREF(top);
        PyList_SetSlice(stack, PyList_GET_SIZE(stack) - 1,
                        PyList_GET_SIZE(stack), NULL);
        PyObject *src = PyTuple_GET_ITEM(top, 0);
        PyObject *dst = PyTuple_GET_ITEM(top, 1);
        Py_DECREF(top);
        Py_ssize_t m = PyList_GET_SIZE(src);
        for (Py_ssize_t i = 0; i < m; ++i) {
            PyObject *item = PyList_GET_ITEM(src, i);
            if (PyList_Check(item)) {
                PyObject *sub = PyList_New(PyList_GET_SIZE(item));
                if (!sub) { Py_DECREF(root); Py_DECREF(stack); return NULL; }
                PyList_SET_ITEM(dst, i, sub);
                PyObject *frame = PyTuple_Pack(2, item, sub);
                if (!frame || PyList_Append(stack, frame) < 0) {
                    Py_XDECREF(frame); Py_DECREF(root); Py_DECREF(stack);
                    return NULL;
                }
                Py_DECREF(frame);
            } else {
                double v = PyFloat_AsDouble(item);
                if (PyErr_Occurred()) { Py_DECREF(root); Py_DECREF(stack); return NULL; }
                PyObject *scaled = PyFloat_FromDouble(v * factor);
                if (!scaled) { Py_DECREF(root); Py_DECREF(stack); return NULL; }
                PyList_SET_ITEM(dst, i, scaled);
            }
        }
    }
    Py_DECREF(stack);
    return root;
}

/* walk every leaf -> |x|^2 (iterative) */
static PyObject *spectrum_walk(PyObject *obj) {
    if (!PyList_Check(obj)) {
        double v = PyFloat_AsDouble(obj);
        if (PyErr_Occurred()) return NULL;
        return PyFloat_FromDouble(v * v);
    }
    Py_ssize_t n = PyList_GET_SIZE(obj);
    PyObject *root = PyList_New(n);
    if (!root) return NULL;
    PyObject *stack = PyList_New(0);
    if (!stack) { Py_DECREF(root); return NULL; }
    PyObject *init = PyTuple_Pack(2, obj, root);
    if (!init || PyList_Append(stack, init) < 0) {
        Py_XDECREF(init); Py_DECREF(root); Py_DECREF(stack);
        return NULL;
    }
    Py_DECREF(init);
    while (PyList_GET_SIZE(stack) > 0) {
        PyObject *top = PyList_GET_ITEM(stack, PyList_GET_SIZE(stack) - 1);
        Py_INCREF(top);
        PyList_SetSlice(stack, PyList_GET_SIZE(stack) - 1,
                        PyList_GET_SIZE(stack), NULL);
        PyObject *src = PyTuple_GET_ITEM(top, 0);
        PyObject *dst = PyTuple_GET_ITEM(top, 1);
        Py_DECREF(top);
        Py_ssize_t m = PyList_GET_SIZE(src);
        for (Py_ssize_t i = 0; i < m; ++i) {
            PyObject *item = PyList_GET_ITEM(src, i);
            if (PyList_Check(item)) {
                PyObject *sub = PyList_New(PyList_GET_SIZE(item));
                if (!sub) { Py_DECREF(root); Py_DECREF(stack); return NULL; }
                PyList_SET_ITEM(dst, i, sub);
                PyObject *frame = PyTuple_Pack(2, item, sub);
                if (!frame || PyList_Append(stack, frame) < 0) {
                    Py_XDECREF(frame); Py_DECREF(root); Py_DECREF(stack);
                    return NULL;
                }
                Py_DECREF(frame);
            } else {
                double v = PyFloat_AsDouble(item);
                if (PyErr_Occurred()) { Py_DECREF(root); Py_DECREF(stack); return NULL; }
                PyObject *sq = PyFloat_FromDouble(v * v);
                if (!sq) { Py_DECREF(root); Py_DECREF(stack); return NULL; }
                PyList_SET_ITEM(dst, i, sq);
            }
        }
    }
    Py_DECREF(stack);
    return root;
}

/* ==================================================================
 * dft / idft / power_spectrum
 * ================================================================== */

static PyObject *dft_impl(PyObject *data, PyObject *axis, double sign) {
    PyObject *shape = shape_of(data);
    if (!shape) return NULL;
    Py_ssize_t dim = 0;
    PyObject *axes = resolve_axes(shape, axis, &dim);
    if (!axes) { Py_DECREF(shape); return NULL; }
    PyObject *work = deep_mutable(data, shape);
    if (!work) { Py_DECREF(shape); Py_DECREF(axes); return NULL; }
    for (Py_ssize_t a = 0; a < PyList_GET_SIZE(axes); ++a) {
        Py_ssize_t axis_i = PyLong_AsSsize_t(PyList_GET_ITEM(axes, a));
        if (PyErr_Occurred() ||
            axes_transform_one(work, shape, axis_i, sign) < 0) {
            Py_DECREF(shape); Py_DECREF(axes); Py_DECREF(work);
            return NULL;
        }
    }
    Py_DECREF(shape);
    Py_DECREF(axes);
    return work;
}

static PyObject *py_dft(PyObject *self, PyObject *args, PyObject *kwargs) { (void)self;
    PyObject *data;
    PyObject *axis = Py_None;
    static char *kwlist[] = {"data", "axis", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist,
                                     &data, &axis))
        return NULL;
    return dft_impl(data, axis, -1.0);
}

static PyObject *py_idft(PyObject *self, PyObject *args, PyObject *kwargs) { (void)self;
    PyObject *data;
    PyObject *axis = Py_None;
    static char *kwlist[] = {"data", "axis", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist,
                                     &data, &axis))
        return NULL;
    PyObject *shape = shape_of(data);
    if (!shape) return NULL;
    double total = 1.0;
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(shape); ++i) {
        Py_ssize_t s = PyLong_AsSsize_t(PyList_GET_ITEM(shape, i));
        if (PyErr_Occurred()) { Py_DECREF(shape); return NULL; }
        total *= (double)s;
    }
    Py_DECREF(shape);
    PyObject *work = dft_impl(data, axis, 1.0);
    if (!work) return NULL;
    return scale(work, 1.0 / total);
}

static PyObject *py_power_spectrum(PyObject *self, PyObject *args,
                                   PyObject *kwargs) { (void)self;
    PyObject *data;
    PyObject *axis = Py_None;
    static char *kwlist[] = {"data", "axis", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist,
                                     &data, &axis))
        return NULL;
    PyObject *work = dft_impl(data, axis, -1.0);
    if (!work) return NULL;
    return spectrum_walk(work);
}

/* ==================================================================
 * dft_kernel_real / dft_kernel_imag
 * ================================================================== */

static int axis_param(PyObject *obj, double **out, Py_ssize_t dim, double dflt) {
    *out = NULL;
    if (obj == Py_None) {
        *out = (double*)malloc((size_t)(dim > 0 ? dim : 1) * sizeof(double));
        if (!*out) { PyErr_NoMemory(); return -1; }
        for (Py_ssize_t i = 0; i < dim; ++i) (*out)[i] = dflt;
        return 0;
    }
    if (PyNumber_Check(obj)) {
        double v = PyFloat_AsDouble(obj);
        if (PyErr_Occurred()) return -1;
        *out = (double*)malloc((size_t)(dim > 0 ? dim : 1) * sizeof(double));
        if (!*out) { PyErr_NoMemory(); return -1; }
        for (Py_ssize_t i = 0; i < dim; ++i) (*out)[i] = v;
        return 0;
    }
    PyObject *it = PyObject_GetIter(obj);
    if (!it) return -1;
    *out = (double*)malloc((size_t)(dim > 0 ? dim : 1) * sizeof(double));
    if (!*out) { Py_DECREF(it); PyErr_NoMemory(); return -1; }
    Py_ssize_t i = 0;
    PyObject *item;
    while ((item = PyIter_Next(it))) {
        double v = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred() || i >= dim) {
            if (i != dim) {
                Py_DECREF(it); free(*out); *out = NULL;
                PyErr_Format(PyExc_ValueError,
                             "parameter length must match shape");
                return -1;
            }
        }
        if (i < dim) (*out)[i] = v;
        i++;
    }
    Py_DECREF(it);
    if (PyErr_Occurred()) { free(*out); *out = NULL; return -1; }
    if (i != dim) {
        free(*out); *out = NULL;
        PyErr_Format(PyExc_ValueError, "parameter length must match shape");
        return -1;
    }
    return 0;
}

static int element_param(PyObject *obj, double **out, long long total, double dflt) {
    *out = NULL;
    if (obj == Py_None) {
        *out = (double*)malloc((size_t)total * sizeof(double));
        if (!*out) { PyErr_NoMemory(); return -1; }
        for (long long i = 0; i < total; ++i) (*out)[i] = dflt;
        return 0;
    }
    if (PyNumber_Check(obj)) {
        double v = PyFloat_AsDouble(obj);
        if (PyErr_Occurred()) return -1;
        *out = (double*)malloc((size_t)total * sizeof(double));
        if (!*out) { PyErr_NoMemory(); return -1; }
        for (long long i = 0; i < total; ++i) (*out)[i] = v;
        return 0;
    }
    PyObject *it = PyObject_GetIter(obj);
    if (!it) return -1;
    *out = (double*)malloc((size_t)total * sizeof(double));
    if (!*out) { Py_DECREF(it); PyErr_NoMemory(); return -1; }
    long long i = 0;
    PyObject *item;
    while ((item = PyIter_Next(it))) {
        double v = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            Py_DECREF(it); free(*out); *out = NULL;
            return -1;
        }
        if (i < total) (*out)[i] = v;
        i++;
    }
    Py_DECREF(it);
    if (PyErr_Occurred()) { free(*out); *out = NULL; return -1; }
    if (i != total) {
        free(*out); *out = NULL;
        PyErr_Format(PyExc_ValueError,
                     "parameter length must match element count");
        return -1;
    }
    return 0;
}

static PyObject *kernel_impl(PyObject *shape_arg, PyObject *freq_arg,
                              int use_cos, PyObject *scales,
                              PyObject *offsets, PyObject *amplitudes,
                              PyObject *biases) {
    /* normalize shape / frequencies */
    PyObject *shape_list;
    PyObject *freq_list;    if (PyLong_Check(shape_arg)) {
        shape_list = PyList_New(1);
        freq_list = PyList_New(1);
        if (!shape_list || !freq_list) {
            Py_XDECREF(shape_list); Py_XDECREF(freq_list);
            return NULL;
        }
        Py_INCREF(shape_arg);
        PyList_SET_ITEM(shape_list, 0, shape_arg);
        Py_INCREF(freq_arg);
        PyList_SET_ITEM(freq_list, 0, freq_arg);
    } else {
        shape_list = PySequence_List(shape_arg);
        freq_list = PySequence_List(freq_arg);
    }    if (!shape_list || !freq_list) {
        Py_XDECREF(shape_list); Py_XDECREF(freq_list);
        return NULL;
    }
    Py_ssize_t dim = PyList_GET_SIZE(shape_list);
    if (PyList_GET_SIZE(freq_list) != dim) {
        Py_DECREF(shape_list); Py_DECREF(freq_list);
        PyErr_SetString(PyExc_ValueError,
                        "frequencies length must match shape");
        return NULL;
    }    /* validate frequencies within shape */
    for (Py_ssize_t i = 0; i < dim; ++i) {        PyObject *k_o = PyList_GET_ITEM(freq_list, i);
        PyObject *n_o = PyList_GET_ITEM(shape_list, i);
        long k = PyLong_AsLong(k_o);
        long n = PyLong_AsLong(n_o);
        if (PyErr_Occurred()) {
            Py_DECREF(shape_list); Py_DECREF(freq_list);
            return NULL;
        }
        if (k < 0 || k >= n) {
            Py_DECREF(shape_list); Py_DECREF(freq_list);
            PyErr_SetString(PyExc_ValueError,
                            "frequencies entries must be within shape");
            return NULL;
        }
    }    long long total = 1;
    for (Py_ssize_t i = 0; i < dim; ++i) {
        long n = PyLong_AsLong(PyList_GET_ITEM(shape_list, i));
        if (PyErr_Occurred()) {
            Py_DECREF(shape_list); Py_DECREF(freq_list);
            return NULL;
        }
        total *= (long long)n;
    }    double *sc = NULL, *off = NULL, *amp = NULL, *bia = NULL;    if (axis_param(scales, &sc, dim, 1.0) < 0 ||
        axis_param(offsets, &off, dim, 0.0) < 0 ||
        element_param(amplitudes, &amp, total, 1.0) < 0 ||
        element_param(biases, &bia, total, 0.0) < 0) {
        free(sc); free(off); free(amp); free(bia);
        Py_DECREF(shape_list); Py_DECREF(freq_list);
        return NULL;
    }    /* build the nested result (iterative) */
    PyObject *work = NULL;
    if (dim == 1) {
        work = PyList_New(PyLong_AsSsize_t(PyList_GET_ITEM(shape_list, 0)));
    } else {
        work = PyList_New(PyLong_AsSsize_t(PyList_GET_ITEM(shape_list, 0)));
        PyObject *stack = PyList_New(0);
        if (!stack) {
            free(sc); free(off); free(amp); free(bia);
            Py_DECREF(shape_list); Py_DECREF(freq_list);
            Py_DECREF(work);
            return NULL;
        }
        PyObject *init = PyTuple_Pack(3, work, PyLong_FromLong(1),
                                      PyLong_FromLong(0));
        if (!init || PyList_Append(stack, init) < 0) {
            Py_XDECREF(init); Py_DECREF(stack);
            free(sc); free(off); free(amp); free(bia);
            Py_DECREF(shape_list); Py_DECREF(freq_list);
            Py_DECREF(work);
            return NULL;
        }
        Py_DECREF(init);
        while (PyList_GET_SIZE(stack) > 0) {
            PyObject *top = PyList_GET_ITEM(stack, PyList_GET_SIZE(stack) - 1);
            Py_INCREF(top);
            PyList_SetSlice(stack, PyList_GET_SIZE(stack) - 1,
                            PyList_GET_SIZE(stack), NULL);
            PyObject *node = PyTuple_GET_ITEM(top, 0);
            long level = PyLong_AsLong(PyTuple_GET_ITEM(top, 1));
            Py_DECREF(top);
            if (PyErr_Occurred()) {
                Py_DECREF(stack);
                free(sc); free(off); free(amp); free(bia);
                Py_DECREF(shape_list); Py_DECREF(freq_list);
                Py_DECREF(work);
                return NULL;
            }
            Py_ssize_t m = PyList_GET_SIZE(node);
            for (Py_ssize_t i = 0; i < m; ++i) {
                if (level == dim - 1) {
                    Py_ssize_t leaf = PyLong_AsSsize_t(
                        PyList_GET_ITEM(shape_list, level));
                    if (PyErr_Occurred()) {
                        Py_DECREF(stack);
                        free(sc); free(off); free(amp); free(bia);
                        Py_DECREF(shape_list); Py_DECREF(freq_list);
                        Py_DECREF(work);
                        return NULL;
                    }
                    PyObject *row = PyList_New(leaf);
                    if (!row) {
                        Py_DECREF(stack);
                        free(sc); free(off); free(amp); free(bia);
                        Py_DECREF(shape_list); Py_DECREF(freq_list);
                        Py_DECREF(work);
                        return NULL;
                    }
                    PyList_SET_ITEM(node, i, row);
                } else {
                    Py_ssize_t child_n = PyLong_AsSsize_t(
                        PyList_GET_ITEM(shape_list, level));
                    if (PyErr_Occurred()) {
                        Py_DECREF(stack);
                        free(sc); free(off); free(amp); free(bia);
                        Py_DECREF(shape_list); Py_DECREF(freq_list);
                        Py_DECREF(work);
                        return NULL;
                    }
                    PyObject *child = PyList_New(child_n);
                    if (!child) {
                        Py_DECREF(stack);
                        free(sc); free(off); free(amp); free(bia);
                        Py_DECREF(shape_list); Py_DECREF(freq_list);
                        Py_DECREF(work);
                        return NULL;
                    }
                    PyList_SET_ITEM(node, i, child);
                    PyObject *frame = PyTuple_Pack(
                        3, child, PyLong_FromLong(level + 1),
                        PyLong_FromLong(0));
                    if (!frame || PyList_Append(stack, frame) < 0) {
                        Py_XDECREF(frame); Py_DECREF(stack);
                        free(sc); free(off); free(amp); free(bia);
                        Py_DECREF(shape_list); Py_DECREF(freq_list);
                        Py_DECREF(work);
                        return NULL;
                    }
                    Py_DECREF(frame);
                }
            }
        }
        Py_DECREF(stack);
    }    /* fill by odometer */
    double offset_sum = 0.0;
    for (Py_ssize_t i = 0; i < dim; ++i) offset_sum += off[i];

    long *idx = (long*)calloc((size_t)(dim > 0 ? dim : 1), sizeof(long));
    if (!idx) {
        free(sc); free(off); free(amp); free(bia);
        Py_DECREF(shape_list); Py_DECREF(freq_list);
        Py_DECREF(work);
        PyErr_NoMemory();
        return NULL;
    }
    for (long long t = 0; t < total; ++t) {
        double phase = 0.0;
        for (Py_ssize_t i = 0; i < dim; ++i) {
            long n = PyLong_AsLong(PyList_GET_ITEM(shape_list, i));
            long k = PyLong_AsLong(PyList_GET_ITEM(freq_list, i));
            if (PyErr_Occurred()) {
                free(idx); free(sc); free(off); free(amp); free(bia);
                Py_DECREF(shape_list); Py_DECREF(freq_list);
                Py_DECREF(work);
                return NULL;
            }
            phase += sc[i] * (double)k * (double)idx[i] / (double)n;
        }
        double arg = 2.0 * M_PI * phase + offset_sum;
        double val = (use_cos ? cos(arg) : sin(arg)) * amp[t] + bia[t];        /* write at idx (nested) */
        PyObject *obj = work;
        int ok = 1;
        for (Py_ssize_t i = 0; i < dim - 1; ++i) {
            if (PyList_Check(obj)) {
                obj = PyList_GET_ITEM(obj, idx[i]);
            } else {
                PyObject *sub = PySequence_GetItem(obj, idx[i]);
                if (!sub) { ok = 0; break; }
                obj = sub;
            }
        }
        if (!ok) {
            free(idx); free(sc); free(off); free(amp); free(bia);
            Py_DECREF(shape_list); Py_DECREF(freq_list);
            Py_DECREF(work);
            return NULL;
        }
        if (PyList_Check(obj)) {
            PyList_SET_ITEM(obj, idx[dim - 1], PyFloat_FromDouble(val));
        } else {
            int setr = PyObject_SetItem(
                obj, PyLong_FromLong(idx[dim - 1]),
                PyFloat_FromDouble(val));
            if (setr < 0) {
                free(idx); free(sc); free(off); free(amp); free(bia);
                Py_DECREF(shape_list); Py_DECREF(freq_list);
                Py_DECREF(work);
                return NULL;
            }
        }
        for (Py_ssize_t i = dim - 1; i >= 0; --i) {
            idx[i]++;
            if (idx[i] < PyLong_AsLong(PyList_GET_ITEM(shape_list, i)))
                break;
            idx[i] = 0;
            if (PyErr_Occurred()) {
                free(idx); free(sc); free(off); free(amp); free(bia);
                Py_DECREF(shape_list); Py_DECREF(freq_list);
                Py_DECREF(work);
                return NULL;
            }
        }
    }
    free(idx);
    free(sc); free(off); free(amp); free(bia);
    Py_DECREF(shape_list);
    Py_DECREF(freq_list);
    return work;
}

static PyObject *py_dft_kernel_real(PyObject *self, PyObject *args,
                                    PyObject *kwargs) { (void)self;
    PyObject *shape, *frequencies;
    PyObject *scales = Py_None, *offsets = Py_None;
    PyObject *amplitudes = Py_None, *biases = Py_None;
    static char *kwlist[] = {"shape", "frequencies", "scales", "offsets",
                             "amplitudes", "biases", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OOOO", kwlist,
                                     &shape, &frequencies, &scales,
                                     &offsets, &amplitudes, &biases))
        return NULL;
    return kernel_impl(shape, frequencies, 1, scales, offsets,
                        amplitudes, biases);
}

static PyObject *py_dft_kernel_imag(PyObject *self, PyObject *args,
                                    PyObject *kwargs) { (void)self;
    PyObject *shape, *frequencies;
    PyObject *scales = Py_None, *offsets = Py_None;
    PyObject *amplitudes = Py_None, *biases = Py_None;
    static char *kwlist[] = {"shape", "frequencies", "scales", "offsets",
                             "amplitudes", "biases", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OOOO", kwlist,
                                     &shape, &frequencies, &scales,
                                     &offsets, &amplitudes, &biases))
        return NULL;
    return kernel_impl(shape, frequencies, 0, scales, offsets,
                        amplitudes, biases);
}

/* ==================================================================
 * Module definition
 * ================================================================== */

static PyMethodDef methods[] = {
    {"dft", (PyCFunction)py_dft, METH_VARARGS | METH_KEYWORDS,
     "Generic DFT (trig formula), multi-dimensional."},
    {"idft", (PyCFunction)py_idft, METH_VARARGS | METH_KEYWORDS,
     "Generic IDFT (1/N normalized), multi-dimensional."},
    {"power_spectrum", (PyCFunction)py_power_spectrum,
     METH_VARARGS | METH_KEYWORDS, "|X[k]|^2 per element."},
    {"dft_kernel_real", (PyCFunction)py_dft_kernel_real,
     METH_VARARGS | METH_KEYWORDS, "Real-part DFT kernel (cos)."},
    {"dft_kernel_imag", (PyCFunction)py_dft_kernel_imag,
     METH_VARARGS | METH_KEYWORDS, "Imag-part DFT kernel (sin)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_fourier",
    "C99 optimized Fourier transforms and DFT kernels (math_tool).",
    0,
    methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__fourier(void);

PyMODINIT_FUNC PyInit__fourier(void) {
    return PyModuleDef_Init(&moduledef);
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
