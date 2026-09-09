/*
 * _linear_algebra.c - dimension-generic linear algebra (C99, duck typing).
 *
 * Every function follows one convention: the tensor output is passed out
 * through the keyword argument "output" (never returned directly - no
 * tensor type hard-coding), and the function returns an integer status:
 *     0  success
 *     1  shape / length mismatch
 *     2  no output container (output=None) or output write failure
 *     3  value conversion failure (non-numeric element)
 *
 * Data access is duck-typed through the Python sequence / item protocols
 * (PyObject_GetIter / PyObject_GetItem / PyObject_SetItem), so any
 * container built on the sequence protocol works.  The behaviour matches
 * the pure-Python fallback (linear_algebra.py) exactly.
 *
 * C99 features used deliberately: stdint types, restrict-qualified
 * pointers, designated initializers, inline helpers, function pointers.
 */

#ifdef _MSC_VER
/* CPython extension ABI patterns under /Wall (see cos_comparison_pydll.c):
   C4191 method-table casts, C4232 PyType_GenericNew address,
   C4820 struct padding; C5045/C4711/C4710 are /Wall performance hints. */
#pragma warning(push)
#pragma warning(disable: 4191 4232 4820 5045 4711 4710)
#endif
#include <Python.h>
#include <math.h>
#include <stdint.h>

/* status codes (must match the Python fallback) */
#define LA_OK                 0
#define LA_SHAPE_MISMATCH     1
#define LA_NO_OUTPUT          2
#define LA_CONVERSION_FAILURE 3

/* ------------------------------------------------------------------ */
/* helpers (duck-typed)                                                */
/* ------------------------------------------------------------------ */

static inline int la_is_seq(PyObject *obj)
{
    return PyObject_HasAttrString(obj, "__iter__")
        || PyObject_HasAttrString(obj, "__getitem__");
}

static inline int la_convert(PyObject *obj, double *out)
{
    if (PyFloat_Check(obj)) {
        *out = PyFloat_AS_DOUBLE(obj);
        return 1;
    }
    if (PyLong_Check(obj)) {
        long long v = PyLong_AsLongLong(obj);
        *out = (double)v;
        return !PyErr_Occurred();
    }
    {
        PyObject *f = PyNumber_Float(obj);   /* duck numeric conversion */
        if (f == NULL) {
            PyErr_Clear();
            return 0;
        }
        *out = PyFloat_AS_DOUBLE(f);
        Py_DECREF(f);
        return 1;
    }
}

/* flatten any-dimension data into a PyList of leaf values
 * (explicit iterator stack - no recursion); NULL on failure */
static PyObject *la_flatten(PyObject *data)
{
    PyObject *values = PyList_New(0);
    PyObject *stack = PyList_New(0);
    PyObject *it;
    if (values == NULL || stack == NULL) {
        Py_XDECREF(values);
        Py_XDECREF(stack);
        return NULL;
    }
    it = PyObject_GetIter(data);
    if (it == NULL) {
        Py_DECREF(values);
        Py_DECREF(stack);
        return NULL;
    }
    if (PyList_Append(stack, it) < 0) {
        Py_DECREF(it);
        Py_DECREF(values);
        Py_DECREF(stack);
        return NULL;
    }
    Py_DECREF(it);
    while (PyList_GET_SIZE(stack) > 0) {
        Py_ssize_t top = PyList_GET_SIZE(stack) - 1;
        PyObject *iter = PyList_GET_ITEM(stack, top);
        PyObject *item = PyIter_Next(iter);
        if (item == NULL) {
            if (PyErr_Occurred()) {
                Py_DECREF(values);
                Py_DECREF(stack);
                return NULL;
            }
            if (PyList_SetSlice(stack, top, top + 1, NULL) < 0) {
                Py_DECREF(values);
                Py_DECREF(stack);
                return NULL;
            }
            continue;
        }
        if (PyUnicode_Check(item) || PyBytes_Check(item)
            || !la_is_seq(item)) {
            if (PyList_Append(values, item) < 0) {
                Py_DECREF(item);
                Py_DECREF(values);
                Py_DECREF(stack);
                return NULL;
            }
        }
        else {
            PyObject *sub = PyObject_GetIter(item);
            if (sub == NULL) {
                Py_DECREF(item);
                Py_DECREF(values);
                Py_DECREF(stack);
                return NULL;
            }
            if (PyList_Append(stack, sub) < 0) {
                Py_DECREF(sub);
                Py_DECREF(item);
                Py_DECREF(values);
                Py_DECREF(stack);
                return NULL;
            }
            Py_DECREF(sub);
        }
        Py_DECREF(item);
    }
    Py_DECREF(stack);
    return values;
}

/* convert a PyList of leaves to a double array (malloc'd);
 * returns 0 on conversion failure (array freed, *values = NULL) */
static int la_to_doubles(PyObject *list, double **values, Py_ssize_t *n)
{
    Py_ssize_t len = PyList_GET_SIZE(list);
    double *buf;
    if (len == 0) {
        *values = NULL;
        *n = 0;
        return 1;
    }
    buf = (double *)PyMem_Malloc((size_t)len * sizeof(double));
    if (buf == NULL) {
        PyErr_NoMemory();
        return 0;
    }
    for (Py_ssize_t i = 0; i < len; i++) {
        if (!la_convert(PyList_GET_ITEM(list, i), &buf[i])) {
            PyMem_Free(buf);
            *values = NULL;
            return 0;
        }
    }
    *values = buf;
    *n = len;
    return 1;
}

/* duck-typed item write: output[i] = value (0 on failure) */
static inline int la_set_item(PyObject *output, Py_ssize_t i, double value)
{
    PyObject *key = PyLong_FromSsize_t(i);
    PyObject *val = PyFloat_FromDouble(value);
    int rc;
    if (key == NULL || val == NULL) {
        Py_XDECREF(key);
        Py_XDECREF(val);
        return 0;
    }
    rc = PyObject_SetItem(output, key, val);
    Py_DECREF(key);
    Py_DECREF(val);
    if (rc < 0) {
        PyErr_Clear();
        return 0;
    }
    return 1;
}

/* write values linearly into output (0 on write failure) */
static int la_write_flat(PyObject *output, const double *restrict values,
                          Py_ssize_t n)
{
    for (Py_ssize_t i = 0; i < n; i++) {
        if (!la_set_item(output, i, values[i])) {
            return 0;
        }
    }
    return 1;
}

/* write values back following a shape (nested item paths, iterative) */
static int la_write_shaped(PyObject *output, PyObject *shape,
                            const double *restrict values, Py_ssize_t n)
{
    Py_ssize_t dims = PyTuple_GET_SIZE(shape);
    Py_ssize_t *idx;
    Py_ssize_t i, d;
    if (dims == 0) {
        return la_write_flat(output, values, n);
    }
    idx = (Py_ssize_t *)PyMem_Malloc((size_t)dims * sizeof(Py_ssize_t));
    if (idx == NULL) {
        PyErr_NoMemory();
        return 0;
    }
    for (d = 0; d < dims; d++) {
        idx[d] = 0;
    }
    i = 0;
    for (;;) {
        PyObject *obj = output;   /* borrowed at the first level */
        int owned = 0;
        int rc = 0;
        if (i >= n) {
            PyMem_Free(idx);
            return 0;
        }
        for (d = 0; d < dims - 1; d++) {
            PyObject *key = PyLong_FromSsize_t(idx[d]);
            PyObject *next;
            if (key == NULL) {
                if (owned) {
                    Py_DECREF(obj);
                }
                PyMem_Free(idx);
                return 0;
            }
            next = PyObject_GetItem(obj, key);
            Py_DECREF(key);
            if (next == NULL) {
                PyErr_Clear();
                if (owned) {
                    Py_DECREF(obj);
                }
                PyMem_Free(idx);
                return 0;
            }
            if (owned) {
                Py_DECREF(obj);
            }
            obj = next;
            owned = 1;
        }
        if (la_set_item(obj, idx[dims - 1], values[i])) {
            rc = 1;
        }
        if (owned) {
            Py_DECREF(obj);
        }
        if (!rc) {
            PyMem_Free(idx);
            return 0;
        }
        i++;
        for (d = dims - 1; d >= 0; d--) {
            idx[d]++;
            if (idx[d] < PyLong_AsSsize_t(PyTuple_GET_ITEM(shape, d))) {
                break;
            }
            idx[d] = 0;
            if (d == 0) {
                PyMem_Free(idx);
                return i == n ? 1 : 0;
            }
        }
    }
}

/* infer the shape of a duck-typed tensor (iterative) */
static PyObject *la_shape(PyObject *data)
{
    PyObject *shape = PyList_New(0);
    PyObject *obj = data;   /* borrowed (first level) */
    int borrowed = 1;
    if (shape == NULL) {
        return NULL;
    }
    for (;;) {
        Py_ssize_t len;
        PyObject *item;
        if (PyUnicode_Check(obj) || PyBytes_Check(obj)
            || !la_is_seq(obj)) {
            break;
        }
        len = PyObject_Length(obj);
        if (len < 0) {
            PyErr_Clear();
            break;
        }
        {
            PyObject *lobj = PyLong_FromSsize_t(len);
            int rc = lobj == NULL ? -1 : PyList_Append(shape, lobj);
            Py_XDECREF(lobj);
            if (rc < 0) {
                if (!borrowed) {
                    Py_DECREF(obj);
                }
                Py_DECREF(shape);
                return NULL;
            }
        }
        if (len == 0) {
            break;
        }
        item = PyObject_GetIter(obj);
        if (item == NULL) {
            PyErr_Clear();
            break;
        }
        {
            PyObject *next = PyIter_Next(item);
            Py_DECREF(item);
            if (next == NULL) {
                PyErr_Clear();
                break;
            }
            if (!borrowed) {
                Py_DECREF(obj);
            }
            obj = next;
            borrowed = 0;
        }
    }
    if (!borrowed) {
        Py_DECREF(obj);
    }
    {
        PyObject *result = PyList_AsTuple(shape);
        Py_DECREF(shape);
        return result;
    }
}

/* ------------------------------------------------------------------ */
/* shared element-wise implementations (function pointers, C99)         */
/* ------------------------------------------------------------------ */

typedef double (*la_binop)(double, double);

static PyObject *la_pair(PyObject *a, PyObject *b, PyObject *output,
                          la_binop op)
{
    PyObject *fa, *fb;
    double *va = NULL, *vb = NULL;
    Py_ssize_t na, nb;
    double *out_buf;
    int status = LA_OK;
    PyObject *shape;

    if (output == Py_None) {
        return PyLong_FromLong(LA_NO_OUTPUT);
    }
    fa = la_flatten(a);
    fb = la_flatten(b);
    if (fa == NULL || fb == NULL) {
        Py_XDECREF(fa);
        Py_XDECREF(fb);
        return NULL;
    }
    if (!la_to_doubles(fa, &va, &na)) {
        Py_DECREF(fa);
        Py_DECREF(fb);
        return PyLong_FromLong(LA_CONVERSION_FAILURE);
    }
    if (!la_to_doubles(fb, &vb, &nb)) {
        PyMem_Free(va);
        Py_DECREF(fa);
        Py_DECREF(fb);
        return PyLong_FromLong(LA_CONVERSION_FAILURE);
    }
    Py_DECREF(fa);
    Py_DECREF(fb);
    if (na != nb) {
        PyMem_Free(va);
        PyMem_Free(vb);
        return PyLong_FromLong(LA_SHAPE_MISMATCH);
    }
    out_buf = (double *)PyMem_Malloc(
        (size_t)(na > 0 ? na : 1) * sizeof(double));
    if (out_buf == NULL) {
        PyMem_Free(va);
        PyMem_Free(vb);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t i = 0; i < na; i++) {
        out_buf[i] = op(va[i], vb[i]);
    }
    PyMem_Free(va);
    PyMem_Free(vb);
    shape = la_shape(a);
    if (shape == NULL) {
        PyMem_Free(out_buf);
        return NULL;
    }
    if (!la_write_shaped(output, shape, out_buf, na)) {
        status = LA_NO_OUTPUT;
    }
    PyMem_Free(out_buf);
    Py_DECREF(shape);
    return PyLong_FromLong(status);
}

static PyObject *la_single(PyObject *a, PyObject *output,
                            double (*op)(double, double), double extra)
{
    PyObject *fa;
    double *va = NULL;
    Py_ssize_t na;
    double *out_buf;
    int status = LA_OK;
    PyObject *shape;

    if (output == Py_None) {
        return PyLong_FromLong(LA_NO_OUTPUT);
    }
    fa = la_flatten(a);
    if (fa == NULL) {
        return NULL;
    }
    if (!la_to_doubles(fa, &va, &na)) {
        Py_DECREF(fa);
        return PyLong_FromLong(LA_CONVERSION_FAILURE);
    }
    Py_DECREF(fa);
    out_buf = (double *)PyMem_Malloc(
        (size_t)(na > 0 ? na : 1) * sizeof(double));
    if (out_buf == NULL) {
        PyMem_Free(va);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t i = 0; i < na; i++) {
        out_buf[i] = op(va[i], extra);
    }
    PyMem_Free(va);
    shape = la_shape(a);
    if (shape == NULL) {
        PyMem_Free(out_buf);
        return NULL;
    }
    if (!la_write_shaped(output, shape, out_buf, na)) {
        status = LA_NO_OUTPUT;
    }
    PyMem_Free(out_buf);
    Py_DECREF(shape);
    return PyLong_FromLong(status);
}

/* ------------------------------------------------------------------ */
/* the linear algebra functions                                         */
/* ------------------------------------------------------------------ */

static inline double la_b_add(double x, double y) { return x + y; }
static inline double la_b_mul(double x, double y) { return x * y; }
static inline double la_u_scale(double v, double f) { return v * f; }
static inline double la_u_power(double v, double e)
{
    return pow(v, e);
}

static double la_clip_low = 0.0;
static double la_clip_high = 0.0;

static double la_clip_impl(double v, double extra)
{
    (void)extra;
    if (v < la_clip_low) {
        return la_clip_low;
    }
    if (v > la_clip_high) {
        return la_clip_high;
    }
    return v;
}

static PyObject *py_la_dot(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"a", "b", "output", NULL};
    PyObject *a, *b, *output = Py_None;
    PyObject *fa, *fb;
    double *va = NULL, *vb = NULL;
    Py_ssize_t na, nb;
    double total;
    int status;

    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist,
                                     &a, &b, &output)) {
        return NULL;
    }
    if (output == Py_None) {
        return PyLong_FromLong(LA_NO_OUTPUT);
    }
    fa = la_flatten(a);
    fb = la_flatten(b);
    if (fa == NULL || fb == NULL) {
        Py_XDECREF(fa);
        Py_XDECREF(fb);
        return NULL;
    }
    if (!la_to_doubles(fa, &va, &na)
        || !la_to_doubles(fb, &vb, &nb)) {
        PyMem_Free(va);
        PyMem_Free(vb);
        Py_DECREF(fa);
        Py_DECREF(fb);
        return PyLong_FromLong(LA_CONVERSION_FAILURE);
    }
    Py_DECREF(fa);
    Py_DECREF(fb);
    if (na != nb) {
        PyMem_Free(va);
        PyMem_Free(vb);
        return PyLong_FromLong(LA_SHAPE_MISMATCH);
    }
    total = 0.0;
    for (Py_ssize_t i = 0; i < na; i++) {
        total += va[i] * vb[i];
    }
    PyMem_Free(va);
    PyMem_Free(vb);
    status = la_write_flat(output, &total, 1) ? LA_OK : LA_NO_OUTPUT;
    return PyLong_FromLong(status);
}

static PyObject *py_la_scalar_reduce(PyObject *self, PyObject *args,
                                     PyObject *kwargs, int mean_flag)
{
    static char *kwlist[] = {"a", "output", NULL};
    PyObject *a, *output = Py_None;
    PyObject *fa;
    double *va = NULL;
    Py_ssize_t na;
    double result = 0.0;
    int status;

    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist,
                                     &a, &output)) {
        return NULL;
    }
    if (output == Py_None) {
        return PyLong_FromLong(LA_NO_OUTPUT);
    }
    fa = la_flatten(a);
    if (fa == NULL) {
        return NULL;
    }
    if (!la_to_doubles(fa, &va, &na)) {
        Py_DECREF(fa);
        return PyLong_FromLong(LA_CONVERSION_FAILURE);
    }
    Py_DECREF(fa);
    if (mean_flag && na > 0) {
        for (Py_ssize_t i = 0; i < na; i++) {
            result += va[i];
        }
        result /= (double)na;
    }
    else if (mean_flag) {
        result = 0.0;
    }
    else {
        for (Py_ssize_t i = 0; i < na; i++) {
            result += va[i];
        }
    }
    PyMem_Free(va);
    status = la_write_flat(output, &result, 1) ? LA_OK : LA_NO_OUTPUT;
    return PyLong_FromLong(status);
}

static PyObject *py_la_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"a", "output", NULL};
    PyObject *a, *output = Py_None;
    PyObject *fa;
    double *va = NULL;
    Py_ssize_t na;
    double total = 0.0;
    int status;

    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist,
                                     &a, &output)) {
        return NULL;
    }
    if (output == Py_None) {
        return PyLong_FromLong(LA_NO_OUTPUT);
    }
    fa = la_flatten(a);
    if (fa == NULL) {
        return NULL;
    }
    if (!la_to_doubles(fa, &va, &na)) {
        Py_DECREF(fa);
        return PyLong_FromLong(LA_CONVERSION_FAILURE);
    }
    Py_DECREF(fa);
    for (Py_ssize_t i = 0; i < na; i++) {
        total += va[i] * va[i];
    }
    PyMem_Free(va);
    total = sqrt(total);
    status = la_write_flat(output, &total, 1) ? LA_OK : LA_NO_OUTPUT;
    return PyLong_FromLong(status);
}

static PyObject *py_la_normalize(PyObject *self, PyObject *args,
                                 PyObject *kwargs)
{
    static char *kwlist[] = {"a", "output", NULL};
    PyObject *a, *output = Py_None;
    PyObject *fa, *shape;
    double *va = NULL;
    Py_ssize_t na;
    double *out_buf;
    double total = 0.0, length;
    int status = LA_OK;

    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist,
                                     &a, &output)) {
        return NULL;
    }
    if (output == Py_None) {
        return PyLong_FromLong(LA_NO_OUTPUT);
    }
    fa = la_flatten(a);
    if (fa == NULL) {
        return NULL;
    }
    if (!la_to_doubles(fa, &va, &na)) {
        Py_DECREF(fa);
        return PyLong_FromLong(LA_CONVERSION_FAILURE);
    }
    Py_DECREF(fa);
    for (Py_ssize_t i = 0; i < na; i++) {
        total += va[i] * va[i];
    }
    length = sqrt(total);
    out_buf = (double *)PyMem_Malloc(
        (size_t)(na > 0 ? na : 1) * sizeof(double));
    if (out_buf == NULL) {
        PyMem_Free(va);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t i = 0; i < na; i++) {
        out_buf[i] = (length == 0.0) ? 0.0 : va[i] / length;
    }
    PyMem_Free(va);
    shape = la_shape(a);
    if (shape == NULL) {
        PyMem_Free(out_buf);
        return NULL;
    }
    if (!la_write_shaped(output, shape, out_buf, na)) {
        status = LA_NO_OUTPUT;
    }
    PyMem_Free(out_buf);
    Py_DECREF(shape);
    return PyLong_FromLong(status);
}

static PyObject *py_la_add(PyObject *self, PyObject *args, PyObject *kwargs)
{
    static char *kwlist[] = {"a", "b", "output", NULL};
    PyObject *a, *b, *output = Py_None;
    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist,
                                     &a, &b, &output)) {
        return NULL;
    }
    return la_pair(a, b, output, la_b_add);
}

static PyObject *py_la_multiply(PyObject *self, PyObject *args,
                                PyObject *kwargs)
{
    static char *kwlist[] = {"a", "b", "output", NULL};
    PyObject *a, *b, *output = Py_None;
    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist,
                                     &a, &b, &output)) {
        return NULL;
    }
    return la_pair(a, b, output, la_b_mul);
}

static PyObject *py_la_scale(PyObject *self, PyObject *args,
                             PyObject *kwargs)
{
    static char *kwlist[] = {"a", "factor", "output", NULL};
    PyObject *a, *factor, *output = Py_None;
    double f;
    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist,
                                     &a, &factor, &output)) {
        return NULL;
    }
    if (!la_convert(factor, &f)) {
        return PyLong_FromLong(LA_CONVERSION_FAILURE);
    }
    return la_single(a, output, la_u_scale, f);
}

static PyObject *py_la_power(PyObject *self, PyObject *args,
                             PyObject *kwargs)
{
    static char *kwlist[] = {"a", "exponent", "output", NULL};
    PyObject *a, *exponent, *output = Py_None;
    double e;
    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist,
                                     &a, &exponent, &output)) {
        return NULL;
    }
    if (!la_convert(exponent, &e)) {
        return PyLong_FromLong(LA_CONVERSION_FAILURE);
    }
    return la_single(a, output, la_u_power, e);
}

static PyObject *py_la_clip(PyObject *self, PyObject *args,
                            PyObject *kwargs)
{
    static char *kwlist[] = {"a", "low", "high", "output", NULL};
    PyObject *a, *low, *high, *output = Py_None;
    double fl, fh;
    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|O", kwlist,
                                     &a, &low, &high, &output)) {
        return NULL;
    }
    if (!la_convert(low, &fl) || !la_convert(high, &fh)) {
        return PyLong_FromLong(LA_CONVERSION_FAILURE);
    }
    la_clip_low = fl;
    la_clip_high = fh;
    return la_single(a, output, la_clip_impl, 0.0);
}

static PyObject *py_la_flatten(PyObject *self, PyObject *args,
                               PyObject *kwargs)
{
    static char *kwlist[] = {"a", "output", NULL};
    PyObject *a, *output = Py_None;
    PyObject *fa;
    double *va = NULL;
    Py_ssize_t na;
    int status;

    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist,
                                     &a, &output)) {
        return NULL;
    }
    if (output == Py_None) {
        return PyLong_FromLong(LA_NO_OUTPUT);
    }
    fa = la_flatten(a);
    if (fa == NULL) {
        return NULL;
    }
    if (!la_to_doubles(fa, &va, &na)) {
        Py_DECREF(fa);
        return PyLong_FromLong(LA_CONVERSION_FAILURE);
    }
    Py_DECREF(fa);
    status = la_write_flat(output, va, na) ? LA_OK : LA_NO_OUTPUT;
    PyMem_Free(va);
    return PyLong_FromLong(status);
}

static PyObject *py_la_tensor_sum(PyObject *self, PyObject *args,
                                  PyObject *kwargs)
{
    return py_la_scalar_reduce(self, args, kwargs, 0);
}

static PyObject *py_la_tensor_mean(PyObject *self, PyObject *args,
                                   PyObject *kwargs)
{
    return py_la_scalar_reduce(self, args, kwargs, 1);
}

/* ------------------------------------------------------------------ */
/* module definition                                                    */
/* ------------------------------------------------------------------ */

static PyMethodDef methods[] = {
    {"dot", (PyCFunction)py_la_dot, METH_VARARGS | METH_KEYWORDS,
     "dot(a, b, output=None): element-wise product sum; output[0] "
     "receives the scalar; returns the status int."},
    {"norm", (PyCFunction)py_la_norm, METH_VARARGS | METH_KEYWORDS,
     "norm(a, output=None): Frobenius norm; output[0] receives the "
     "scalar."},
    {"normalize", (PyCFunction)py_la_normalize, METH_VARARGS | METH_KEYWORDS,
     "normalize(a, output=None): element-wise division by the norm."},
    {"scale", (PyCFunction)py_la_scale, METH_VARARGS | METH_KEYWORDS,
     "scale(a, factor, output=None): element-wise scalar multiply."},
    {"add", (PyCFunction)py_la_add, METH_VARARGS | METH_KEYWORDS,
     "add(a, b, output=None): element-wise addition."},
    {"multiply", (PyCFunction)py_la_multiply, METH_VARARGS | METH_KEYWORDS,
     "multiply(a, b, output=None): element-wise multiplication."},
    {"power", (PyCFunction)py_la_power, METH_VARARGS | METH_KEYWORDS,
     "power(a, exponent, output=None): element-wise exponentiation."},
    {"clip", (PyCFunction)py_la_clip, METH_VARARGS | METH_KEYWORDS,
     "clip(a, low, high, output=None): element-wise clamping."},
    {"flatten", (PyCFunction)py_la_flatten, METH_VARARGS | METH_KEYWORDS,
     "flatten(a, output=None): flatten into a 1D output."},
    {"tensor_sum", (PyCFunction)py_la_tensor_sum,
     METH_VARARGS | METH_KEYWORDS,
     "tensor_sum(a, output=None): sum of all elements."},
    {"tensor_mean", (PyCFunction)py_la_tensor_mean,
     METH_VARARGS | METH_KEYWORDS,
     "tensor_mean(a, output=None): mean of all elements."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_linear_algebra",
    "Dimension-generic linear algebra (duck typing, output keyword).",
    -1,
    methods,
    NULL,  /* m_slots */
    NULL,  /* m_traverse */
    NULL,  /* m_clear */
    NULL,  /* m_free */
};

PyMODINIT_FUNC PyInit__linear_algebra(void)
{
    return PyModule_Create(&moduledef);
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
