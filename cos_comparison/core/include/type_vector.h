#ifndef VECTOR_H
#define VECTOR_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif
#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "type_data.h"

/* ---------------------------------------------------------------------------
 * Portable SIMD & optimization hints (best-effort, degrade gracefully)
 * --------------------------------------------------------------------------- */
#if defined(_MSC_VER)
  /* MSVC: ignore vector dependencies for auto-vectorization */
  #define COS_SIMD_LOOP __pragma(loop(ivdep))
#elif defined(__GNUC__) || defined(__clang__)
  /* GCC/Clang: ignore vector dependencies for auto-vectorization */
  #define COS_SIMD_LOOP _Pragma("GCC ivdep")
#else
  /* Unknown compiler: safe no-op fallback */
  #define COS_SIMD_LOOP
#endif

/* Restrict qualifier for compiler alias analysis */
#if defined(_MSC_VER)
  #define COS_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
  #define COS_RESTRICT __restrict__
#else
  #define COS_RESTRICT
#endif

/* Inline keyword for portable C */
#if defined(_MSC_VER)
  #define COS_INLINE __inline
#elif defined(__GNUC__) || defined(__clang__)
  #define COS_INLINE static inline
#else
  #define COS_INLINE static
#endif

/* Likely/unlikely branch prediction hints */
#if defined(__GNUC__) || defined(__clang__)
  #define COS_LIKELY(x) __builtin_expect(!!(x), 1)
  #define COS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
  #define COS_LIKELY(x) (x)
  #define COS_UNLIKELY(x) (x)
#endif

/* Flag definitions for Vector.flags */
#define VECTOR_FLAG_VIEW     0x01   /* data is a view onto a Python object (owner != NULL) */
#define VECTOR_FLAG_OWNED    0x02   /* data is owned by this Vector (owner == NULL) */
#define VECTOR_FLAG_BUFFER   0x04   /* data came from Py_buffer (zero-copy) */

typedef struct {
    PyObject_HEAD
    Data     *data;           /* underlying flat array (shared or owned) */
    int      *shape;          /* shape of this view */
    int      *strides;        /* original strides for each dimension */
    int      *start_offset;   /* per-dimension start offset */
    int      *step_offset;    /* per-dimension step */
    int       dimension;      /* number of dimensions */
    int       start;          /* global start offset (flat index) */
    int       offset;         /* accumulated offset from integer indexing */
    PyObject *owner;          /* the object that owns data (ref'd), or NULL */
    Py_buffer *buf;           /* saved Py_buffer for zero-copy support, released on dealloc */
    int       flags;          /* internal flags (VECTOR_FLAG_*) */
} Vector;

static PyTypeObject VectorizeType;

/* forward declarations */
static void Vector_dealloc(Vector *self);
static int Vector_traverse(Vector *self, visitproc visit, void *arg);
static int Vector_clear(Vector *self);
static PyObject *Vector_mean(Vector *self, PyObject *args);
static PyObject *Vector_variance(Vector *self, PyObject *args);
static PyObject *Vector_repr(Vector *self);
static Py_ssize_t Vector_len(Vector *self);
static PyObject *Vector_subscript(Vector *self, PyObject *item);
static int Vector_ass_subscript(Vector *self, PyObject *item, PyObject *value);
static PyObject *Vector_add(PyObject *a, PyObject *b);
static PyObject *Vector_sub(PyObject *a, PyObject *b);
static PyObject *Vector_mul(PyObject *a, PyObject *b);
static PyObject *Vector_div(PyObject *a, PyObject *b);
static PyObject *Vector_iadd(PyObject *self, PyObject *other);
static PyObject *Vector_isub(PyObject *self, PyObject *other);
static PyObject *Vector_imul(PyObject *self, PyObject *other);
static PyObject *Vector_itruediv(PyObject *self, PyObject *other);
static PyObject *Vector_neg(PyObject *self);
static PyObject *Vector_pos(PyObject *self);
static PyObject *Vector_abs(PyObject *self);
static PyObject *Vector_get_item(Vector *self, PyObject *args);   /* __get_item__ */
static PyObject *Vector_set_item(Vector *self, PyObject *args);   /* __set_item__ */

static PyObject *Vector_cos_comparison_passive(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *Vector_cos_comparison_active(PyObject *self, PyObject *args, PyObject *kwargs);

/* Getters for Python attribute access (pure Python API compatibility) */
static PyObject* Vector_get_shape(Vector *self, void *closure) {
    PyObject *tup = PyTuple_New(self->dimension);
    if (!tup) return NULL;
    for (int i = 0; i < self->dimension; ++i) {
        PyTuple_SET_ITEM(tup, i, PyLong_FromLong(self->shape[i]));
    }
    return tup;
}

/* __shape__ method - overridable by subclasses for custom shape inference */
static PyObject* Vector_shape_method(Vector *self, PyObject *Py_UNUSED(ignored)) {
    return Vector_get_shape(self, NULL);
}

static PyObject* Vector_get_dimension(Vector *self, void *closure) {
    return PyLong_FromLong(self->dimension);
}

static PyObject* Vector_get_strides(Vector *self, void *closure) {
    PyObject *tup = PyTuple_New(self->dimension);
    if (!tup) return NULL;
    for (int i = 0; i < self->dimension; ++i) {
        PyTuple_SET_ITEM(tup, i, PyLong_FromLong(self->strides[i]));
    }
    return tup;
}

static PyObject* Vector_get_start_offset(Vector *self, void *closure) {
    PyObject *tup = PyTuple_New(self->dimension);
    if (!tup) return NULL;
    for (int i = 0; i < self->dimension; ++i) {
        PyTuple_SET_ITEM(tup, i, PyLong_FromLong(self->start_offset[i]));
    }
    return tup;
}

static PyObject* Vector_get_step_offset(Vector *self, void *closure) {
    PyObject *tup = PyTuple_New(self->dimension);
    if (!tup) return NULL;
    for (int i = 0; i < self->dimension; ++i) {
        PyTuple_SET_ITEM(tup, i, PyLong_FromLong(self->step_offset[i]));
    }
    return tup;
}

static PyObject* Vector_get_offset(Vector *self, void *closure) {
    return PyLong_FromLong(self->offset);
}

static PyObject* Vector_get_start(Vector *self, void *closure) {
    return PyLong_FromLong(self->start);
}

static PyObject* Vector_get_vector(Vector *self, void *closure) {
    /* Iterate all elements (handles non-contiguous views) */
    int total = 1;
    for (int i = 0; i < self->dimension; ++i) total *= self->shape[i];
    PyObject *list = PyList_New(total);
    if (!list) return NULL;
    double *data = (double*)self->data->data;
    
    /* Carry-based iteration, no recursion */
    int *idx = (int*)PyMem_Malloc(self->dimension * sizeof(int));
    if (!idx) { Py_DECREF(list); return PyErr_NoMemory(); }
    memset(idx, 0, self->dimension * sizeof(int));
    int pos = 0;
    
    while (1) {
        int flat = self->start + self->offset;
        for (int i = 0; i < self->dimension; ++i) {
            flat += self->strides[i] * (self->start_offset[i] + idx[i] * self->step_offset[i]);
        }
        PyList_SET_ITEM(list, pos++, PyFloat_FromDouble(data[flat]));
        
        /* Increment with carry */
        int dim = self->dimension - 1;
        while (dim >= 0) {
            idx[dim]++;
            if (idx[dim] < self->shape[dim]) break;
            idx[dim] = 0;
            dim--;
        }
        if (dim < 0) break;
    }
    PyMem_Free(idx);
    return list;
}

static PyGetSetDef Vector_getseters[] = {
    {"tensor_size", (getter)Vector_get_shape, NULL,
     "Tuple representing the shape of the current tensor view (backward compatibility alias for shape).", NULL},
    {"shape", (getter)Vector_get_shape, NULL,
     "Tuple representing the shape of the current tensor view.", NULL},
    {"dimension", (getter)Vector_get_dimension, NULL,
     "Number of dimensions (rank) of the tensor.", NULL},
    {"strides", (getter)Vector_get_strides, NULL,
     "Tuple representing the strides of the current tensor view.", NULL},
    {"start_offset", (getter)Vector_get_start_offset, NULL,
     "Tuple of per-dimension start offsets.", NULL},
    {"step_offset", (getter)Vector_get_step_offset, NULL,
     "Tuple of per-dimension step sizes.", NULL},
    {"offset", (getter)Vector_get_offset, NULL,
     "Accumulated offset from integer indexing.", NULL},
    {"start", (getter)Vector_get_start, NULL,
     "Global start offset in the flat underlying array.", NULL},
    {"vector", (getter)Vector_get_vector, NULL,
     "Flat list of underlying data (copy, for API compatibility with pure Python backend).", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef Vector_methods[] = {
    {"__shape__", (PyCFunction)Vector_shape_method, METH_NOARGS,
        "Shape protocol method for fast infer_shape. Can be overridden by subclasses."},
    {"mean", (PyCFunction)Vector_mean, METH_NOARGS,
        "Compute the mean of the current slice."},
    {"variance", (PyCFunction)Vector_variance, METH_NOARGS,
        "Compute the variance of the current slice."},
    {"__get_item__", (PyCFunction)Vector_get_item, METH_VARARGS,
        "Support multi-index slicing and value retrieval."},
    {"__set_item__", (PyCFunction)Vector_set_item, METH_VARARGS,
        "Support multi-index value assignment (fast path)."},
    {"__cos_comparison_passive__", (PyCFunction)Vector_cos_comparison_passive, METH_VARARGS | METH_KEYWORDS,
        "Optimized passive mode computation for Vector types (overload)."},
    {"cos_comparison_passive", (PyCFunction)Vector_cos_comparison_passive, METH_VARARGS | METH_KEYWORDS,
        "Passive mode comparison as instance method."},
    {"__cos_comparison_active__", (PyCFunction)Vector_cos_comparison_active, METH_VARARGS | METH_KEYWORDS,
        "Optimized active mode computation for Vector types (overload)."},
    {"cos_comparison_active", (PyCFunction)Vector_cos_comparison_active, METH_VARARGS | METH_KEYWORDS,
        "Active mode comparison as instance method."},
    {NULL, NULL, 0, NULL}
};

static PyMappingMethods Vector_as_mapping = {
    (lenfunc)Vector_len,
    (binaryfunc)Vector_subscript,
    (objobjargproc)Vector_ass_subscript,
};

/* Sequence protocol for default iteration support (matches pure Python behavior) */
static PyObject* Vector_sq_item(Vector *self, Py_ssize_t i) {
    if (self->dimension == 0) {
        PyErr_SetString(PyExc_IndexError, "scalar tensor has no elements");
        return NULL;
    }
    if (i < 0) i += self->shape[0];
    if (i < 0 || i >= self->shape[0]) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }
    return Vector_subscript(self, PyLong_FromSsize_t(i));
}

static PySequenceMethods Vector_as_sequence = {
    (lenfunc)Vector_len,                      /* sq_length */
    0,                                        /* sq_concat */
    0,                                        /* sq_repeat */
    (ssizeargfunc)Vector_sq_item,             /* sq_item */
    0,                                        /* sq_slice */
    0,                                        /* sq_ass_item */
    0,                                        /* sq_ass_slice */
    0,                                        /* sq_contains */
    0,                                        /* sq_inplace_concat */
    0,                                        /* sq_inplace_repeat */
};

/* ------------------------------------------------------------------
Buffer protocol export (memoryview(vector) support)
Exposes the underlying contiguous double/uchar storage using the
Vector's logical start/offset/step slicing parameters. Read-only.
------------------------------------------------------------------ */
static int Vector_getbuffer(PyObject *self, Py_buffer *view, int flags) {
    Vector *v = (Vector*)self;
    Data *d = v->data;
    if (!d || !d->data) {
        PyErr_SetString(PyExc_BufferError, "vector has no underlying data");
        return -1;
    }
    if (flags & PyBUF_WRITABLE) {
        PyErr_SetString(PyExc_BufferError, "vector buffer is read-only");
        return -1;
    }
    int itemsize = (d->dtype == 1) ? 1 : 8;
    const char *format = (d->dtype == 1) ? "B" : "d";
    int want_nd = !!(flags & PyBUF_ND);
    int want_strides = !!(flags & PyBUF_STRIDES);
    int want_format = !!(flags & PyBUF_FORMAT);
    Py_ssize_t *shape_buf = NULL, *strides_buf = NULL;
if (want_nd || want_strides) {
            size_t n = (v->dimension > 0) ? (size_t)v->dimension : 1;
            shape_buf = (Py_ssize_t*)PyMem_Malloc(sizeof(Py_ssize_t) * n);
            if (!shape_buf) {
                PyErr_NoMemory();
                return -1;
            }
            if (want_strides) {
                strides_buf = (Py_ssize_t*)PyMem_Malloc(sizeof(Py_ssize_t) * n);
                if (!strides_buf) {
                    PyMem_Free(shape_buf);
                    PyErr_NoMemory();
                    return -1;
                }
            }
            for (int i = 0; i < v->dimension; ++i) {
                shape_buf[i] = v->shape[i];
                if (strides_buf) strides_buf[i] = (Py_ssize_t)v->strides[i] * v->step_offset[i] * itemsize;
            }
            if (!want_nd) {
                PyMem_Free(shape_buf);
                shape_buf = NULL;
            }
            if (!want_strides) {
                PyMem_Free(strides_buf);
                strides_buf = NULL;
            }
    }
    Py_ssize_t offset = (Py_ssize_t)v->start + (Py_ssize_t)v->offset;
    Py_ssize_t total = 1;
    for (int i = 0; i < v->dimension; ++i) {
        offset += (Py_ssize_t)v->strides[i] * v->start_offset[i];
        total *= v->shape[i];
    }
    Py_ssize_t len = total * itemsize;
    Py_INCREF(self);
    view->buf = (void*)((char*)d->data + offset * itemsize);
    view->obj = self;
    view->len = len;
    view->readonly = 1;
    view->itemsize = itemsize;
    view->format = want_format ? (char*)format : NULL;
    view->ndim = want_nd ? v->dimension : 0;
    view->shape = shape_buf;
    view->strides = strides_buf;
    view->suboffsets = NULL;
    view->internal = NULL;
    return 0;
}

static void Vector_releasebuffer(PyObject *self, Py_buffer *view) {
    (void)self;
    PyMem_Free(view->shape);
    PyMem_Free(view->strides);
}

static PyBufferProcs Vector_as_buffer = {
    (getbufferproc)Vector_getbuffer,      /* bf_getbuffer */
    (releasebufferproc)Vector_releasebuffer, /* bf_releasebuffer */
};

// Forward declarations for number methods
static PyObject* Vector_pow(PyObject *a, PyObject *b, PyObject *mod);
static PyObject* Vector_ipow(PyObject *self, PyObject *other, PyObject *mod);

static PyNumberMethods Vector_as_number = {
    (binaryfunc)Vector_add,          /* nb_add              (1) */
    (binaryfunc)Vector_sub,          /* nb_subtract         (2) */
    (binaryfunc)Vector_mul,          /* nb_multiply         (3) */
    0,                               /* nb_remainder        (4) */
    0,                               /* nb_divmod           (5) */
    (ternaryfunc)Vector_pow,         /* nb_power            (6) */
    (unaryfunc)Vector_neg,           /* nb_negative         (7) */
    (unaryfunc)Vector_pos,           /* nb_positive         (8) */
    (unaryfunc)Vector_abs,           /* nb_absolute         (9) */
    0,                               /* nb_bool            (10) */
    0,                               /* nb_invert          (11) */
    0,                               /* nb_lshift          (12) */
    0,                               /* nb_rshift          (13) */
    0,                               /* nb_and             (14) */
    0,                               /* nb_xor             (15) */
    0,                               /* nb_or              (16) */
    0,                               /* nb_int             (17) */
    0,                               /* nb_reserved        (18) */
    0,                               /* nb_float           (19) */
    (binaryfunc)Vector_iadd,         /* nb_inplace_add     (20) */
    (binaryfunc)Vector_isub,         /* nb_inplace_subtract(21) */
    (binaryfunc)Vector_imul,         /* nb_inplace_multiply(22) */
    0,                               /* nb_inplace_remainder(23) */
    (ternaryfunc)Vector_ipow,        /* nb_inplace_power   (24) */
    0,                               /* nb_inplace_lshift  (25) */
    0,                               /* nb_inplace_rshift  (26) */
    0,                               /* nb_inplace_and     (27) */
    0,                               /* nb_inplace_xor     (28) */
    0,                               /* nb_inplace_or      (29) */
    0,                               /* nb_floor_divide    (30) */
    (binaryfunc)Vector_div,          /* nb_true_divide     (31) */
    0,                               /* nb_inplace_floor_divide(32) */
    (binaryfunc)Vector_itruediv,     /* nb_inplace_true_divide (33) */
    0,                               /* nb_index           (34) */
    0,                               /* nb_matrix_multiply (35) */
    0,                               /* nb_inplace_matrix_multiply(36) */
};

static inline int _multiple_chain(const int *arr, int n) {
    int result = 1;
    for (int i = 0; i < n; ++i) result *= arr[i];
    return result;
}

/* -----------------------------------------------------------------------------
 * High-frequency inline helpers - extracted to reduce code duplication
 * --------------------------------------------------------------------------- */

/* Calculate flat index from multi-dimensional indices (matches pure Python logic exactly) */
static inline int _vector_calc_flat_index(const Vector *self, PyObject *index_tuple, int n) {
    int ptr = self->start + self->offset;
    for (int i = 0; i < n; ++i) {
        PyObject *idx_obj = PyTuple_GET_ITEM(index_tuple, i);
        int idx = (int)PyLong_AsLong(idx_obj);
        if (idx < 0) idx += self->shape[i];
        ptr += self->strides[i] * (self->start_offset[i] + idx * self->step_offset[i]);
    }
    return ptr;
}

/* Bounds check for multi-dimensional indices */
static inline int _vector_check_indices(const Vector *self, PyObject *index_tuple, int n) {
    for (int i = 0; i < n; ++i) {
        PyObject *idx_obj = PyTuple_GET_ITEM(index_tuple, i);
        int idx = (int)PyLong_AsLong(idx_obj);
        if (idx < 0) idx += self->shape[i];
        if (idx < 0 || idx >= self->shape[i]) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return -1;
        }
    }
    return 0;
}

/* Create a new view of the same type as self (supports subclasses, GC-safe) */
static inline Vector* _vector_new_view(Vector *self) {
    PyTypeObject *type = Py_TYPE(self);
    Vector *view = (Vector*)type->tp_alloc(type, 0);
    if (!view) return NULL;
    
    view->data = self->data;
    view->owner = self->owner ? self->owner : (PyObject*)self;
    Py_INCREF(view->owner);
    view->flags = self->flags | VECTOR_FLAG_VIEW;
    view->dimension = self->dimension;
    view->start = self->start;
    view->offset = self->offset;
    
    // Allocate and copy shape
    view->shape = (int*)malloc((size_t)(self->dimension) * sizeof(int));
    if (!view->shape) { Py_DECREF(view); return NULL; }
    memcpy(view->shape, self->shape, self->dimension * sizeof(int));
    
    // Allocate and copy strides
    view->strides = (int*)malloc((size_t)(self->dimension) * sizeof(int));
    if (!view->strides) { free(view->shape); Py_DECREF(view); return NULL; }
    memcpy(view->strides, self->strides, self->dimension * sizeof(int));
    
    // Allocate and copy start_offset
    view->start_offset = (int*)malloc((size_t)(self->dimension) * sizeof(int));
    if (!view->start_offset) { free(view->shape); free(view->strides); Py_DECREF(view); return NULL; }
    memcpy(view->start_offset, self->start_offset, self->dimension * sizeof(int));
    
    // Allocate and copy step_offset
    view->step_offset = (int*)malloc((size_t)(self->dimension) * sizeof(int));
    if (!view->step_offset) { free(view->shape); free(view->strides); free(view->start_offset); Py_DECREF(view); return NULL; }
    memcpy(view->step_offset, self->step_offset, self->dimension * sizeof(int));
    
    view->buf = NULL;
    return view;
}

static int _flatten_list_to_data(PyObject *obj, double *out, int *idx, int dim, const int *shape) {
    if (dim == 0) {
        PyObject *num = PyNumber_Float(obj);
        if (!num) return -1;
        out[*idx] = PyFloat_AsDouble(num);
        Py_DECREF(num);
        (*idx)++;
        return 0;
    }

    /* Iterative flatten using carry mechanism - no recursion, no stack overflow */
    int *num_list = (int*)malloc(((size_t)(dim) + 1) * sizeof(int));
    if (!num_list) { PyErr_NoMemory(); return -1; }
    for (int i = 0; i <= dim; ++i) num_list[i] = 1;
    int flag = dim;
    int pos = 0;
    int *indices = (int*)malloc((size_t)(dim) * sizeof(int));
    if (!indices) { free(num_list); PyErr_NoMemory(); return -1; }

    while (flag) {
        if (flag == dim) {
            for (int i = 0; i < dim; ++i) indices[i] = num_list[i + 1] - 1;
            /* Navigate to nested item iteratively */
            PyObject *current = obj;
            Py_INCREF(current);
            int valid = 1;
            for (int i = 0; i < dim; ++i) {
                if (!PySequence_Check(current)) {
                    valid = 0;
                    break;
                }
                Py_ssize_t cur_len = PySequence_Size(current);
                if (cur_len != shape[i]) {
                    valid = 0;
                    break;
                }
                PyObject *next = PySequence_GetItem(current, indices[i]);
                Py_DECREF(current);
                if (!next) { valid = 0; break; }
                current = next;
            }
            if (!valid) {
                Py_XDECREF(current);
                free(num_list);
                free(indices);
                PyErr_SetString(PyExc_ValueError, "inconsistent tensor shape");
                return -1;
            }
            PyObject *num = PyNumber_Float(current);
            Py_DECREF(current);
            if (!num) { free(num_list); free(indices); return -1; }
            out[pos] = PyFloat_AsDouble(num);
            Py_DECREF(num);
            pos++;
        }
        if (num_list[flag] < shape[flag - 1]) {
            num_list[flag]++;
            flag = dim;
        } else {
            num_list[flag] = 1;
            flag--;
        }
    }
    *idx = pos;
    free(num_list);
    free(indices);
    return 0;
}

/* Forward declarations for sequence parsing helpers */
static int _parse_shape_tuple(PyObject *obj, int **out_shape, int *out_dim);
static int _parse_int_sequence(PyObject *obj, int **out_arr, int *out_len);
static int _override_int_array(PyObject *obj, int *dest, int dest_len, const char *name);

static int _infer_shape(PyObject *obj, int **shape, int *dimension) {
    int dim = 0;
    int *sh = NULL;
    
    /* Priority 1: Try PyBuffer protocol first (fastest for buffer objects) */
    if (PyObject_CheckBuffer(obj)) {
        Py_buffer view;
        if (PyObject_GetBuffer(obj, &view, PyBUF_FORMAT | PyBUF_ND) == 0) {
            if (view.ndim > 0) {
                dim = view.ndim;
                sh = (int*)malloc((size_t)(dim) * sizeof(int));
                if (!sh) {
                    PyBuffer_Release(&view);
                    PyErr_NoMemory();
                    return -1;
                }
                for (int i = 0; i < dim; ++i) {
                    sh[i] = (int)view.shape[i];
                }
                PyBuffer_Release(&view);
                *shape = sh;
                *dimension = dim;
                return 0;
            }
            PyBuffer_Release(&view);
        } else {
            PyErr_Clear();
        }
    }
    
    /* Priority 2: Try __shape__ method (fast path for our own tensors, overridable) */
    if (PyObject_HasAttrString(obj, "__shape__")) {
        PyObject *method = PyObject_GetAttrString(obj, "__shape__");
        if (method && PyCallable_Check(method)) {
            PyObject *result = PyObject_CallNoArgs(method);
            Py_DECREF(method);
            if (result && result != Py_None) {
                /* Accept any sequence type (tuple, list, etc.) from __shape__ method */
                int *sh = NULL;
                int dim = 0;
                if (_parse_shape_tuple(result, &sh, &dim) == 0) {
                    Py_DECREF(result);
                    *shape = sh;
                    *dimension = dim;
                    return 0;
                }
                Py_DECREF(result);
            } else if (result) {
                Py_DECREF(result);
            }
        } else if (method) {
            Py_DECREF(method);
        } else {
            PyErr_Clear();
        }
    }
    
    /* Priority 3: Fallback - iterative length detection (general algorithm, no recursion) */
    int cap = 4;
    sh = (int*)malloc((size_t)(cap) * sizeof(int));
    if (!sh) { PyErr_NoMemory(); return -1; }
    PyObject *cur = obj;
    Py_INCREF(cur);
    while (PySequence_Check(cur)) {
        Py_ssize_t len = PySequence_Size(cur);
        if (dim >= cap) {
            cap *= 2;
            int *new_sh = (int*)realloc(sh, (size_t)cap * sizeof(int));
            if (!new_sh) {
                free(sh);
                Py_XDECREF(cur);
                PyErr_NoMemory();
                return -1;
            }
            sh = new_sh;
        }
        sh[dim++] = (int)len;
        if (len == 0) break;
        PyObject *first = PySequence_GetItem(cur, 0);
        Py_DECREF(cur);
        cur = first;
        if (!cur) { free(sh); return -1; }
    }
    Py_XDECREF(cur);
    *shape = sh;
    *dimension = dim;
    return 0;
}

/* Helper to parse a shape tuple */
static int _parse_shape_tuple(PyObject *obj, int **out_shape, int *out_dim) {
    /* Accept any sequence type (tuple, list, etc.) - not just tuples.
       Uses Python sequence protocol for maximum generality and duck-typing support. */
    if (!PySequence_Check(obj)) return -1;
    Py_ssize_t len = PySequence_Size(obj);
    if (len < 0) return -1;
    *out_dim = (int)len;
    *out_shape = (int*)malloc((size_t)(*out_dim) * sizeof(int));
    if (!*out_shape) return -1;
    for (int i = 0; i < *out_dim; ++i) {
        PyObject *item = PySequence_GetItem(obj, i);
        if (!item) { free(*out_shape); *out_shape = NULL; return -1; }
        /* Accept any integer-like object via __index__ (duck typing) */
        PyObject *idx = PyNumber_Index(item);
        Py_DECREF(item);
        if (!idx) { free(*out_shape); *out_shape = NULL; return -1; }
        (*out_shape)[i] = (int)PyLong_AsLong(idx);
        Py_DECREF(idx);
        if (PyErr_Occurred()) { free(*out_shape); *out_shape = NULL; return -1; }
    }
    return 0;
}

/* Parse any sequence of integers into a newly allocated int array.
   Uses Python sequence protocol - works with tuple, list, and any sequence type.
   Returns 0 on success, -1 on failure. Caller must free *out_arr on success. */
static int _parse_int_sequence(PyObject *obj, int **out_arr, int *out_len) {
    if (!PySequence_Check(obj)) return -1;
    Py_ssize_t len = PySequence_Size(obj);
    if (len < 0) return -1;
    *out_len = (int)len;
    *out_arr = (int*)malloc((size_t)(*out_len) * sizeof(int));
    if (!*out_arr) return -1;
    for (int i = 0; i < *out_len; ++i) {
        PyObject *item = PySequence_GetItem(obj, i);
        if (!item) { free(*out_arr); *out_arr = NULL; return -1; }
        PyObject *idx = PyNumber_Index(item);
        Py_DECREF(item);
        if (!idx) { free(*out_arr); *out_arr = NULL; return -1; }
        (*out_arr)[i] = (int)PyLong_AsLong(idx);
        Py_DECREF(idx);
        if (PyErr_Occurred()) { free(*out_arr); *out_arr = NULL; return -1; }
    }
    return 0;
}

/* Override a destination int array with values from a sequence object.
   If obj is Py_None or not a sequence, does nothing (returns 0).
   If sequence length doesn't match dest_len, raises ValueError.
   Uses Python sequence protocol - works with tuple, list, and any sequence type.
   Returns 0 on success, -1 on failure. */
static int _override_int_array(PyObject *obj, int *dest, int dest_len, const char *name) {
    if (obj == Py_None) return 0;
    if (!PySequence_Check(obj)) return 0;
    Py_ssize_t n = PySequence_Size(obj);
    if (n < 0) return -1;
    if ((int)n != dest_len) {
        PyErr_Format(PyExc_ValueError, "%s length must match shape length", name);
        return -1;
    }
    for (int i = 0; i < (int)n; ++i) {
        PyObject *item = PySequence_GetItem(obj, i);
        if (!item) return -1;
        PyObject *idx = PyNumber_Index(item);
        Py_DECREF(item);
        if (!idx) {
            PyErr_Format(PyExc_TypeError, "%s must be sequence of integers", name);
            return -1;
        }
        dest[i] = (int)PyLong_AsLong(idx);
        Py_DECREF(idx);
        if (PyErr_Occurred()) return -1;
    }
    return 0;
}

/* Vector initialization – matches pure Python API */
static int Vector_init(Vector *self, PyObject *args, PyObject *kwargs) {
    PyObject *vector = NULL;
    PyObject *shape_obj = Py_None;
    PyObject *strides_obj = Py_None;
    PyObject *start_offset_obj = Py_None;
    PyObject *step_offset_obj = Py_None;
    int start = 0;
    int offset = 0;
    
    /* All arguments are keyword-only (matches pure Python: def __init__(self, *, ...)) */
    if (PyTuple_Size(args) > 0) {
        PyErr_SetString(PyExc_TypeError, "function takes no positional arguments");
        return -1;
    }
    
    static char *kwlist[] = {"vector", "shape", "start", "strides", "offset", "start_offset", "step_offset", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOiOiOO", kwlist,
                                     &vector, &shape_obj, &start, &strides_obj, &offset, &start_offset_obj, &step_offset_obj))
        return -1;

    /* Default values */
    if (!vector) {
        vector = PyTuple_New(1);
        if (!vector) return -1;
        PyTuple_SET_ITEM(vector, 0, PyLong_FromLong(1));
    }

    self->flags = 0;
    self->buf = NULL;
    self->offset = offset;
    
    /* Case 1: vector is another Vector -> create a sub-view */
    if (PyObject_IsInstance(vector, (PyObject*)&VectorizeType)) {
        Vector *src = (Vector*)vector;
        self->data = src->data;
        self->owner = src->owner ? src->owner : vector;
        Py_INCREF(self->owner);
        self->flags |= VECTOR_FLAG_VIEW;
        
        self->shape = (int*)malloc((size_t)(src->dimension) * sizeof(int));
        self->strides = (int*)malloc((size_t)(src->dimension) * sizeof(int));
        self->start_offset = (int*)malloc((size_t)(src->dimension) * sizeof(int));
        self->step_offset = (int*)malloc((size_t)(src->dimension) * sizeof(int));
        if (!self->shape || !self->strides || !self->start_offset || !self->step_offset) {
            PyErr_NoMemory();
            return -1;
        }
        memcpy(self->shape, src->shape, src->dimension * sizeof(int));
        memcpy(self->strides, src->strides, src->dimension * sizeof(int));
        memcpy(self->start_offset, src->start_offset, src->dimension * sizeof(int));
        memcpy(self->step_offset, src->step_offset, src->dimension * sizeof(int));
        self->dimension = src->dimension;
        self->start = src->start + start;
        return 0;
    }
    
    /* Case 2: try to get a Py_buffer (zero-copy for array.array, bytes, memoryview, numpy arrays etc.) */
    Py_buffer view = {0};
    if (PyObject_GetBuffer(vector, &view, PyBUF_SIMPLE | PyBUF_FORMAT) == 0) {
        int *shape = NULL;
        int dim = 0;
        int dtype = 0; /* 0 = double, 1 = unsigned char */
        size_t elem_size = sizeof(double);
        int need_convert = 0;
        
        /* Detect element type from buffer format string */
        int conv_type = 0; /* 0 = double, 1 = unsigned char, 2 = float, 3 = int, 4 = short, 5 = long, 6 = long long */
        if (view.format) {
            if (strcmp(view.format, "d") == 0) {
                need_convert = 0;
            } else if (strcmp(view.format, "B") == 0 || strcmp(view.format, "b") == 0) {
                need_convert = 0;
                dtype = 1;
                elem_size = sizeof(unsigned char);
            } else if (strcmp(view.format, "f") == 0) {
                need_convert = 1;
                conv_type = 2;
                elem_size = sizeof(float);
            } else if (strcmp(view.format, "i") == 0 || strcmp(view.format, "I") == 0) {
                need_convert = 1;
                conv_type = 3;
                elem_size = sizeof(int);
            } else if (strcmp(view.format, "h") == 0 || strcmp(view.format, "H") == 0) {
                need_convert = 1;
                conv_type = 4;
                elem_size = sizeof(short);
            } else if (strcmp(view.format, "l") == 0 || strcmp(view.format, "L") == 0) {
                need_convert = 1;
                conv_type = 5;
                elem_size = sizeof(long);
            } else if (strcmp(view.format, "q") == 0 || strcmp(view.format, "Q") == 0) {
                need_convert = 1;
                conv_type = 6;
                elem_size = sizeof(long long);
            } else {
                /* Unknown format: fall back to sequence copying */
                PyBuffer_Release(&view);
                goto fallback_sequence;
            }
        }
        
        if (shape_obj != Py_None) {
            if (_parse_shape_tuple(shape_obj, &shape, &dim) < 0) {
                PyBuffer_Release(&view);
                PyErr_SetString(PyExc_ValueError, "invalid shape tuple");
                return -1;
            }
        } else {
            dim = 1;
            shape = (int*)malloc(sizeof(int));
            if (!shape) { PyBuffer_Release(&view); PyErr_NoMemory(); return -1; }
            shape[0] = (int)(view.len / elem_size);
        }
        
        int total = _multiple_chain(shape, dim);
        Data *data = NULL;
        
        if (!need_convert) {
            /* Zero-copy path: directly use buffer memory */
            data = (Data*)calloc((size_t)(1), sizeof(Data));
            if (!data) { free(shape); PyBuffer_Release(&view); PyErr_NoMemory(); return -1; }
            data->dimension = dim;
            /* Data owns its own shape copy so Data_free can safely release it */
            data->shape = (int*)malloc((size_t)(dim) * sizeof(int));
            if (!data->shape) { free(data); free(shape); PyBuffer_Release(&view); PyErr_NoMemory(); return -1; }
            memcpy(data->shape, shape, (size_t)(dim) * sizeof(int));
            data->strides = (int*)malloc((size_t)(dim) * sizeof(int));
            if (!data->strides) { free(data->shape); free(data); free(shape); PyBuffer_Release(&view); PyErr_NoMemory(); return -1; }
            int stride = 1;
            for (int i = dim - 1; i >= 0; --i) {
                data->strides[i] = stride;
                stride *= shape[i];
            }
            data->data = view.buf;
            data->owns_data = 0;
            data->dtype = dtype;
            
            /* Save buffer to release on dealloc (fixes dangling pointer bug) */
            self->buf = (Py_buffer*)malloc(sizeof(Py_buffer));
            if (!self->buf) { Data_free(data); free(shape); PyBuffer_Release(&view); PyErr_NoMemory(); return -1; }
            memcpy(self->buf, &view, sizeof(Py_buffer));
            
            self->data = data;
            self->owner = vector;
            Py_INCREF(self->owner);
            self->flags |= VECTOR_FLAG_BUFFER;
        } else {
            /* Convert path: copy and convert to owned double array */
            data = Data_create(dim, shape);
            if (!data) { free(shape); PyBuffer_Release(&view); PyErr_NoMemory(); return -1; }
            double * COS_RESTRICT out = (double*)data->data;
            const char * COS_RESTRICT in = (const char*)view.buf;
            
            COS_SIMD_LOOP
            for (int i = 0; i < total; ++i) {
                double val;
                const char *p = in + i * (size_t)elem_size;
                switch (conv_type) {
                    case 2: { float    tmp; memcpy(&tmp, p, sizeof(tmp)); val = (double)tmp; } break;
                    case 3: { int      tmp; memcpy(&tmp, p, sizeof(tmp)); val = (double)tmp; } break;
                    case 4: { short    tmp; memcpy(&tmp, p, sizeof(tmp)); val = (double)tmp; } break;
                    case 5: { long     tmp; memcpy(&tmp, p, sizeof(tmp)); val = (double)tmp; } break;
                    case 6: { long long tmp; memcpy(&tmp, p, sizeof(tmp)); val = (double)tmp; } break;
                    default: val = 0.0;
                }
                out[i] = val;
            }
            
            /* Release buffer immediately since we copied */
            PyBuffer_Release(&view);
            self->data = data;
            self->owner = NULL;
            self->flags |= VECTOR_FLAG_OWNED;
        }
        
        self->shape = shape;
        self->dimension = dim;
        self->start = start;
        
        // Allocate and compute strides for Vector
        self->strides = (int*)malloc((size_t)(dim) * sizeof(int));
        if (!self->strides) { PyErr_NoMemory(); return -1; }
        if (dim > 0) {
            self->strides[dim - 1] = 1;
            for (int i = dim - 2; i >= 0; --i) {
                self->strides[i] = self->strides[i+1] * self->shape[i+1];
            }
        }
        // Allocate and init start/step offsets
        self->start_offset = (int*)malloc((size_t)(dim) * sizeof(int));
        self->step_offset = (int*)malloc((size_t)(dim) * sizeof(int));
        if (!self->start_offset || !self->step_offset) { PyErr_NoMemory(); return -1; }
        for (int i = 0; i < dim; ++i) {
            self->start_offset[i] = 0;
            self->step_offset[i] = 1;
        }
        
        // Override strides/start_offset/step_offset if provided (accepts any sequence type)
        if (_override_int_array(strides_obj, self->strides, self->dimension, "strides") < 0) return -1;
        if (_override_int_array(start_offset_obj, self->start_offset, self->dimension, "start_offset") < 0) return -1;
        if (_override_int_array(step_offset_obj, self->step_offset, self->dimension, "step_offset") < 0) return -1;
        
        return 0;
    }
    
    PyErr_Clear();  /* not a buffer, fallback to copying */
fallback_sequence:
    
    /* Case 3: copy data into a new double array */
    int *shape = NULL;
    int dim = 0;
    int explicit_shape = 0;

    /* Try explicit shape first (accepts any sequence type: tuple, list, etc.) */
    if (shape_obj != Py_None && _parse_shape_tuple(shape_obj, &shape, &dim) == 0) {
        explicit_shape = 1;
    } else {
        /* No explicit shape: infer from nested structure */
        if (_infer_shape(vector, &shape, &dim) < 0) {
            return -1;
        }
        if (dim <= 0) {
            free(shape);
            PyErr_SetString(PyExc_ValueError, "not a tensor");
            return -1;
        }
    }
    
    Data *data = Data_create(dim, shape);   /* creates owned double array */
    if (!data) { free(shape); PyErr_NoMemory(); return -1; }
    
    int idx = 0;
    if (explicit_shape) {
        /* Explicit shape provided: vector is 1D flat data, copy directly */
        Py_ssize_t vec_len = PySequence_Size(vector);
        int total = _multiple_chain(shape, dim);
        if (vec_len < total) {
            Data_free(data);
            free(shape);
            PyErr_SetString(PyExc_ValueError, "inconsistent tensor shape");
            return -1;
        }
        for (int i = 0; i < total; ++i) {
            PyObject *item = PySequence_GetItem(vector, i);
            if (!item) { Data_free(data); free(shape); return -1; }
            double val = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (val == -1.0 && PyErr_Occurred()) { Data_free(data); free(shape); return -1; }
            Data_set_flat(data, i, val);
        }
    } else {
        /* No explicit shape: flatten nested structure */
        if (_flatten_list_to_data(vector, data->data, &idx, dim, shape) < 0) {
            Data_free(data);
            free(shape);
            return -1;
        }
    }
    
    self->data = data;
    self->owner = NULL;
    self->flags |= VECTOR_FLAG_OWNED;
    self->start = start;
    
    self->shape = shape;
    self->dimension = dim;
    // Allocate and compute strides
    self->strides = (int*)malloc((size_t)(dim) * sizeof(int));
    if (!self->strides) { Data_free(data); free(shape); PyErr_NoMemory(); return -1; }
    if (dim > 0) {
        self->strides[dim - 1] = 1;
        for (int i = dim - 2; i >= 0; --i) {
            self->strides[i] = self->strides[i+1] * self->shape[i+1];
        }
    }
    // Allocate and init offsets
    self->start_offset = (int*)malloc((size_t)(dim) * sizeof(int));
    self->step_offset = (int*)malloc((size_t)(dim) * sizeof(int));
    if (!self->start_offset || !self->step_offset) { Data_free(data); PyErr_NoMemory(); return -1; }
    for (int i = 0; i < dim; ++i) {
        self->start_offset[i] = 0;
        self->step_offset[i] = 1;
    }
    
    // Override strides/start_offset/step_offset if provided (accepts any sequence type)
    if (_override_int_array(strides_obj, self->strides, self->dimension, "strides") < 0) return -1;
    if (_override_int_array(start_offset_obj, self->start_offset, self->dimension, "start_offset") < 0) return -1;
    if (_override_int_array(step_offset_obj, self->step_offset, self->dimension, "step_offset") < 0) return -1;
    
    return 0;
}

static int Vector_traverse(Vector *self, visitproc visit, void *arg) {
    Py_VISIT(self->owner);
    return 0;
}

static int Vector_clear(Vector *self) {
    Py_CLEAR(self->owner);
    return 0;
}

static void Vector_dealloc(Vector *self) {
    PyObject_GC_UnTrack(self);
    Vector_clear(self);
    if (self->data && !(self->flags & VECTOR_FLAG_VIEW)) {
        Data_free(self->data);
    }
    // Release saved Py_buffer if present
    if (self->buf) {
        PyBuffer_Release(self->buf);
        free(self->buf);
    }
    if (self->shape) free(self->shape);
    if (self->strides) free(self->strides);
    if (self->start_offset) free(self->start_offset);
    if (self->step_offset) free(self->step_offset);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static Py_ssize_t Vector_len(Vector *self) {
    if (self->dimension == 0) return 0;
    return self->shape[0];
}

/* Vector_subscript: supports any mix of int/slice indices, arbitrary steps
 * Deeply optimized C implementation with SIMD hints, two-pass strategy
 * Matches pure Python behavior: tuple = multi-index, single int/slice = single index
 */
static PyObject *Vector_subscript(Vector *self, PyObject *item) {
    PyObject *index_tuple;
    int tuple_created = 0;
    
    /* Normalize: tuple stays as-is, single int/slice gets wrapped into 1-element tuple */
    if (PyTuple_Check(item)) {
        index_tuple = item;
    } else {
        index_tuple = PyTuple_Pack(1, item);
        if (!index_tuple) return NULL;
        tuple_created = 1;
    }
    
    int n_idx = (int)PyTuple_Size(index_tuple);
    
    /* First pass: count how many dimensions remain after integer indexing */
    int new_dim = 0;
    for (int i = 0; i < n_idx; ++i) {
        PyObject *idx_obj = PyTuple_GET_ITEM(index_tuple, i);
        if (PySlice_Check(idx_obj)) {
            new_dim++;
        } else if (PyLong_Check(idx_obj)) {
            /* integer: dimension collapses, not counted in new_dim */
        } else {
            if (tuple_created) Py_DECREF(index_tuple);
            PyErr_SetString(PyExc_TypeError, "indices must be integers or slices");
            return NULL;
        }
    }
    /* Add remaining unindexed dimensions */
    for (int i = n_idx; i < self->dimension; ++i) {
        new_dim++;
    }
    
    /* Allocate new arrays for the result view */
    int *new_shape = (int*)malloc((size_t)(new_dim) * sizeof(int));
    int *new_strides = (int*)malloc((size_t)(new_dim) * sizeof(int));
    int *new_so = (int*)malloc((size_t)(new_dim) * sizeof(int));
    int *new_sto = (int*)malloc((size_t)(new_dim) * sizeof(int));
    int new_offset = self->offset;
    if (!new_shape || !new_strides || !new_so || !new_sto) {
        if (tuple_created) Py_DECREF(index_tuple);
        free(new_shape); free(new_strides); free(new_so); free(new_sto);
        return PyErr_NoMemory();
    }
    
    int pos = 0;
    /* Process each index - SIMD hint for compiler auto-vectorization */
    COS_SIMD_LOOP
    for (int i = 0; i < n_idx; ++i) {
        PyObject *idx_obj = PyTuple_GET_ITEM(index_tuple, i);
        if (PyLong_Check(idx_obj)) {
            /* Integer index: add to offset, dimension removed */
            int idx = (int)PyLong_AsLong(idx_obj);
            int dim_len = self->shape[i];
            if (idx < 0) idx += dim_len;
            if (idx < 0 || idx >= dim_len) {
                if (tuple_created) Py_DECREF(index_tuple);
                free(new_shape); free(new_strides); free(new_so); free(new_sto);
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return NULL;
            }
            new_offset += self->strides[i] * (self->start_offset[i] + idx * self->step_offset[i]);
        } else if (PySlice_Check(idx_obj)) {
            /* Slice index: dimension remains */
            Py_ssize_t slice_start, slice_stop, slice_step, slice_len;
            if (PySlice_GetIndicesEx(idx_obj, self->shape[i], &slice_start, &slice_stop, &slice_step, &slice_len) < 0) {
                if (tuple_created) Py_DECREF(index_tuple);
                free(new_shape); free(new_strides); free(new_so); free(new_sto);
                return NULL;
            }
            if (slice_step == 0) {
                if (tuple_created) Py_DECREF(index_tuple);
                free(new_shape); free(new_strides); free(new_so); free(new_sto);
                PyErr_SetString(PyExc_ValueError, "slice step cannot be zero");
                return NULL;
            }
            new_shape[pos] = (int)slice_len;
            new_strides[pos] = self->strides[i];
            new_so[pos] = self->start_offset[i] + (int)slice_start * self->step_offset[i];
            new_sto[pos] = self->step_offset[i] * (int)slice_step;
            pos++;
        }
    }
    /* Add remaining unindexed dimensions - SIMD hint */
    COS_SIMD_LOOP
    for (int i = n_idx; i < self->dimension; ++i) {
        new_shape[pos] = self->shape[i];
        new_strides[pos] = self->strides[i];
        new_so[pos] = self->start_offset[i];
        new_sto[pos] = self->step_offset[i];
        pos++;
    }
    
    if (tuple_created) Py_DECREF(index_tuple);
    
    /* If all dimensions collapsed, return scalar */
    if (new_dim == 0) {
        free(new_shape); free(new_strides); free(new_so); free(new_sto);
        int flat = self->start + new_offset;
        double val = Data_get_flat(self->data, flat);
        return PyFloat_FromDouble(val);
    }
    
    /* Otherwise create new view - use Py_TYPE for subclass support (no hardcoded type) */
    PyTypeObject *type = Py_TYPE(self);
    Vector *view = (Vector*)type->tp_alloc(type, 0);
    if (!view) {
        free(new_shape); free(new_strides); free(new_so); free(new_sto);
        return NULL;
    }
    view->data = self->data;
    view->owner = self->owner ? self->owner : (PyObject*)self;
    Py_INCREF(view->owner);
    view->flags = self->flags | VECTOR_FLAG_VIEW;
    view->dimension = new_dim;
    view->start = self->start;
    view->offset = new_offset;
    view->shape = new_shape;
    view->strides = new_strides;
    view->start_offset = new_so;
    view->step_offset = new_sto;
    view->buf = NULL;
    
    return (PyObject*)view;
}

/* Helper: iterate all flat indices of a view using carry method (no recursion) */
static int _vector_iter_total(Vector *v) {
    int total = 1;
    for (int i = 0; i < v->dimension; ++i) total *= v->shape[i];
    return total;
}

static int _vector_get_flat_indices(Vector *v, int *out_indices, int max) {
    int total = _vector_iter_total(v);
    if (total > max) return -1;
    if (v->dimension == 0) {
        out_indices[0] = v->start + v->offset;
        return 1;
    }
    int *idx = (int*)PyMem_Malloc(v->dimension * sizeof(int));
    if (!idx) return -1;
    memset(idx, 0, v->dimension * sizeof(int));
    int pos = 0;
    while (1) {
        int flat = v->start + v->offset;
        for (int i = 0; i < v->dimension; ++i) {
            flat += v->strides[i] * (v->start_offset[i] + idx[i] * v->step_offset[i]);
        }
        out_indices[pos++] = flat;
        /* Increment with carry */
        int dim = v->dimension - 1;
        while (dim >= 0) {
            idx[dim]++;
            if (idx[dim] < v->shape[dim]) break;
            idx[dim] = 0;
            dim--;
        }
        if (dim < 0) break;
    }
    PyMem_Free(idx);
    return total;
}

static int Vector_ass_subscript(Vector *self, PyObject *item, PyObject *value) {
    /* First get the target view by calling subscript (reuse our getitem logic) */
    PyObject *target_obj = Vector_subscript(self, item);
    if (!target_obj) return -1;
    
    /* If target is scalar, set directly */
    if (!PyObject_IsInstance(target_obj, (PyObject*)&VectorizeType)) {
        /* Scalar assignment */
        double val = PyFloat_AsDouble(value);
        if (val == -1.0 && PyErr_Occurred()) {
            Py_DECREF(target_obj);
            return -1;
        }
        /* Find the flat index */
        int flat;
        if (PyTuple_Check(item)) {
            flat = _vector_calc_flat_index(self, item, (int)PyTuple_Size(item));
        } else {
            int idx = (int)PyLong_AsLong(item);
            if (idx < 0) idx += self->shape[0];
            flat = self->start + self->offset + self->strides[0] * (self->start_offset[0] + idx * self->step_offset[0]);
        }
        Data_set_flat(self->data, flat, val);
        Py_DECREF(target_obj);
        return 0;
    }
    
    Vector *target = (Vector*)target_obj;
    int total = _vector_iter_total(target);
    
    /* Allocate buffer for target indices */
    int *target_indices = (int*)PyMem_Malloc(total * sizeof(int));
    if (!target_indices) { Py_DECREF(target_obj); return -1; }
    _vector_get_flat_indices(target, target_indices, total);
    
    /* Scalar broadcast */
    if (PyFloat_Check(value) || PyLong_Check(value)) {
        double scalar = PyFloat_AsDouble(value);
        if (PyErr_Occurred()) { PyMem_Free(target_indices); Py_DECREF(target_obj); return -1; }
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            Data_set_flat(self->data, target_indices[i], scalar);
        }
        PyMem_Free(target_indices);
        Py_DECREF(target_obj);
        return 0;
    }
    
    /* Sequence assignment (accepts any sequence type: list, tuple, etc.) */
    if (PySequence_Check(value)) {
        Py_ssize_t seq_len = PySequence_Size(value);
        if ((int)seq_len != total) {
            PyErr_Format(PyExc_ValueError, "expected %d values, got %zd", total, seq_len);
            PyMem_Free(target_indices); Py_DECREF(target_obj);
            return -1;
        }
        for (int i = 0; i < total; ++i) {
            PyObject *item_val = PySequence_GetItem(value, i);
            if (!item_val) { PyMem_Free(target_indices); Py_DECREF(target_obj); return -1; }
            double d = PyFloat_AsDouble(item_val);
            Py_DECREF(item_val);
            if (PyErr_Occurred()) { PyMem_Free(target_indices); Py_DECREF(target_obj); return -1; }
            Data_set_flat(self->data, target_indices[i], d);
        }
        PyMem_Free(target_indices);
        Py_DECREF(target_obj);
        return 0;
    }
    
    /* Vector assignment */
    if (PyObject_IsInstance(value, (PyObject*)&VectorizeType)) {
        Vector *src = (Vector*)value;
        int src_total = _vector_iter_total(src);
        if (src_total != total) {
            PyErr_Format(PyExc_ValueError, "cannot assign %d values to %d elements", src_total, total);
            PyMem_Free(target_indices); Py_DECREF(target_obj);
            return -1;
        }
        int *src_indices = (int*)PyMem_Malloc(total * sizeof(int));
        if (!src_indices) { PyMem_Free(target_indices); Py_DECREF(target_obj); return -1; }
        _vector_get_flat_indices(src, src_indices, total);
        for (int i = 0; i < total; ++i) {
            double d = Data_get_flat(src->data, src_indices[i]);
            Data_set_flat(self->data, target_indices[i], d);
        }
        PyMem_Free(src_indices);
        PyMem_Free(target_indices);
        Py_DECREF(target_obj);
        return 0;
    }
    
    /* Buffer assignment */
    if (PyObject_CheckBuffer(value)) {
        Py_buffer buf;
        if (PyObject_GetBuffer(value, &buf, PyBUF_FORMAT | PyBUF_STRIDED) < 0) {
            PyMem_Free(target_indices); Py_DECREF(target_obj);
            return -1;
        }
        if (buf.ndim != 1 || buf.shape[0] != total) {
            PyBuffer_Release(&buf);
            PyErr_Format(PyExc_ValueError, "buffer length does not match %d elements", total);
            PyMem_Free(target_indices); Py_DECREF(target_obj);
            return -1;
        }
        /* Support common formats */
        int elem_size = (int)buf.itemsize;
        for (int i = 0; i < total; ++i) {
            const char *p = (const char*)buf.buf + i * buf.strides[0];
            double val;
            if (strcmp(buf.format, "d") == 0) { memcpy(&val, p, sizeof(val)); }
            else if (strcmp(buf.format, "f") == 0) { float tmp; memcpy(&tmp, p, sizeof(tmp)); val = (double)tmp; }
            else if (strcmp(buf.format, "i") == 0 || strcmp(buf.format, "I") == 0) { int tmp; memcpy(&tmp, p, sizeof(tmp)); val = (double)tmp; }
            else if (strcmp(buf.format, "l") == 0 || strcmp(buf.format, "L") == 0) { long tmp; memcpy(&tmp, p, sizeof(tmp)); val = (double)tmp; }
            else if (strcmp(buf.format, "q") == 0 || strcmp(buf.format, "Q") == 0) { long long tmp; memcpy(&tmp, p, sizeof(tmp)); val = (double)tmp; }
            else if (strcmp(buf.format, "h") == 0 || strcmp(buf.format, "H") == 0) { short tmp; memcpy(&tmp, p, sizeof(tmp)); val = (double)tmp; }
            else if (strcmp(buf.format, "B") == 0 || strcmp(buf.format, "b") == 0) { unsigned char tmp; memcpy(&tmp, p, sizeof(tmp)); val = (double)tmp; }
            else {
                PyBuffer_Release(&buf);
                PyErr_SetString(PyExc_TypeError, "unsupported buffer format");
                PyMem_Free(target_indices); Py_DECREF(target_obj);
                return -1;
            }
            Data_set_flat(self->data, target_indices[i], val);
        }
        PyBuffer_Release(&buf);
        PyMem_Free(target_indices);
        Py_DECREF(target_obj);
        return 0;
    }
    
    PyErr_SetString(PyExc_TypeError, "value must be scalar, sequence, Vector, or buffer-like object");
    PyMem_Free(target_indices);
    Py_DECREF(target_obj);
    return -1;
}

static PyObject *Vector_mean(Vector *self, PyObject *args) {
    int total = _vector_iter_total(self);
    if (total == 0) Py_RETURN_NONE;
    
    int *indices = (int*)PyMem_Malloc(total * sizeof(int));
    if (!indices) return PyErr_NoMemory();
    _vector_get_flat_indices(self, indices, total);
    
    double mean = 0.0;
    COS_SIMD_LOOP
    for (int i = 0; i < total; ++i) {
        double val = Data_get_flat(self->data, indices[i]);
        double delta = val - mean;
        mean += delta / (i + 1);
    }
    PyMem_Free(indices);
    return PyFloat_FromDouble(mean);
}

static PyObject *Vector_variance(Vector *self, PyObject *args) {
    int total = _vector_iter_total(self);
    if (total == 0) Py_RETURN_NONE;
    
    int *indices = (int*)PyMem_Malloc(total * sizeof(int));
    if (!indices) return PyErr_NoMemory();
    _vector_get_flat_indices(self, indices, total);
    
    double mean = 0.0;
    double M2 = 0.0;
    COS_SIMD_LOOP
    for (int i = 0; i < total; ++i) {
        double val = Data_get_flat(self->data, indices[i]);
        double delta = val - mean;
        mean += delta / (i + 1);
        double delta2 = val - mean;
        M2 += delta * delta2;
    }
    PyMem_Free(indices);
    double var = M2 / (double)total;
    return PyFloat_FromDouble(var);
}

static PyObject *Vector_repr(Vector *self) {
    return PyUnicode_FromFormat(
                                "<vector_map_as_tensor: dim=%d, start=%d, offset=%d>",
                                self->dimension, self->start, self->offset);
}

static inline int _shape_equal(const int *a, const int *b, int dim) {
    for (int i = 0; i < dim; ++i)
        if (a[i] != b[i]) return 0;
    return 1;
}

static Vector* _new_vector_like(Vector *src) {
    PyTypeObject *type = Py_TYPE(src);
    Vector *result = (Vector*)type->tp_alloc(type, 0);
    if (!result) return NULL;
    int ndim = src->dimension;
    result->data = Data_create(ndim, src->shape);
    if (!result->data) { Py_DECREF(result); return NULL; }
    result->owner = NULL;
    result->flags = VECTOR_FLAG_OWNED;
    result->shape = (int*)malloc((size_t)(ndim) * sizeof(int));
    if (!result->shape) { Data_free(result->data); Py_DECREF(result); return NULL; }
    memcpy(result->shape, src->shape, ndim * sizeof(int));
    // Precompute strides for new tensor
    result->strides = (int*)malloc((size_t)(ndim) * sizeof(int));
    if (!result->strides) { free(result->shape); Data_free(result->data); Py_DECREF(result); return NULL; }
    result->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        result->strides[i] = result->strides[i+1] * result->shape[i+1];
    }
    // Init offsets
    result->start_offset = (int*)malloc((size_t)(ndim) * sizeof(int));
    result->step_offset = (int*)malloc((size_t)(ndim) * sizeof(int));
    if (!result->start_offset || !result->step_offset) {
        free(result->shape); free(result->strides); Data_free(result->data); Py_DECREF(result);
        return NULL;
    }
    for (int i = 0; i < ndim; ++i) {
        result->start_offset[i] = 0;
        result->step_offset[i] = 1;
    }
    result->dimension = ndim;
    result->start = 0;
    result->offset = 0;
    result->buf = NULL;
    
    return result;
}

static PyObject *Vector_add(PyObject *a, PyObject *b) {
    if (PyObject_IsInstance(a, (PyObject*)&VectorizeType) && PyObject_IsInstance(b, (PyObject*)&VectorizeType)) {
        Vector *va = (Vector*)a;
        Vector *vb = (Vector*)b;
        if (va->dimension != vb->dimension ||
            !_shape_equal(va->shape, vb->shape, va->dimension)) {
            PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same.");
            return NULL;
        }
        Vector *result = _new_vector_like(va);
        if (!result) return NULL;
        int total = _vector_iter_total(va);
        int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
        int *idx_b = (int*)PyMem_Malloc(total * sizeof(int));
        if (!idx_a || !idx_b) { Py_XDECREF(result); PyMem_Free(idx_a); PyMem_Free(idx_b); return PyErr_NoMemory(); }
        _vector_get_flat_indices(va, idx_a, total);
        _vector_get_flat_indices(vb, idx_b, total);
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val_a = Data_get_flat(va->data, idx_a[i]);
            double val_b = Data_get_flat(vb->data, idx_b[i]);
            Data_set_flat(result->data, i, val_a + val_b);
        }
        PyMem_Free(idx_a); PyMem_Free(idx_b);
        return (PyObject*)result;
    } else if (PyObject_IsInstance(a, (PyObject*)&VectorizeType) && (PyLong_Check(b) || PyFloat_Check(b))) {
        Vector *va = (Vector*)a;
        double scalar = PyFloat_AsDouble(b);
        Vector *result = _new_vector_like(va);
        if (!result) return NULL;
        int total = _vector_iter_total(va);
        int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
        if (!idx_a) { Py_DECREF(result); return PyErr_NoMemory(); }
        _vector_get_flat_indices(va, idx_a, total);
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(va->data, idx_a[i]);
            Data_set_flat(result->data, i, val + scalar);
        }
        PyMem_Free(idx_a);
        return (PyObject*)result;
    } else if ((PyLong_Check(a) || PyFloat_Check(a)) && PyObject_IsInstance(b, (PyObject*)&VectorizeType)) {
        return Vector_add(b, a);
    }
    PyErr_SetString(PyExc_TypeError, "unsupported operand type(s) for +");
    return NULL;
}

static PyObject *Vector_sub(PyObject *a, PyObject *b) {
    if (PyObject_IsInstance(a, (PyObject*)&VectorizeType) && PyObject_IsInstance(b, (PyObject*)&VectorizeType)) {
        Vector *va = (Vector*)a;
        Vector *vb = (Vector*)b;
        if (va->dimension != vb->dimension ||
            !_shape_equal(va->shape, vb->shape, va->dimension)) {
            PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same.");
            return NULL;
        }
        Vector *result = _new_vector_like(va);
        if (!result) return NULL;
        int total = _vector_iter_total(va);
        int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
        int *idx_b = (int*)PyMem_Malloc(total * sizeof(int));
        if (!idx_a || !idx_b) { Py_XDECREF(result); PyMem_Free(idx_a); PyMem_Free(idx_b); return PyErr_NoMemory(); }
        _vector_get_flat_indices(va, idx_a, total);
        _vector_get_flat_indices(vb, idx_b, total);
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val_a = Data_get_flat(va->data, idx_a[i]);
            double val_b = Data_get_flat(vb->data, idx_b[i]);
            Data_set_flat(result->data, i, val_a - val_b);
        }
        PyMem_Free(idx_a); PyMem_Free(idx_b);
        return (PyObject*)result;
    } else if (PyObject_IsInstance(a, (PyObject*)&VectorizeType) && (PyLong_Check(b) || PyFloat_Check(b))) {
        Vector *va = (Vector*)a;
        double scalar = PyFloat_AsDouble(b);
        Vector *result = _new_vector_like(va);
        if (!result) return NULL;
        int total = _vector_iter_total(va);
        int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
        if (!idx_a) { Py_DECREF(result); return PyErr_NoMemory(); }
        _vector_get_flat_indices(va, idx_a, total);
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(va->data, idx_a[i]);
            Data_set_flat(result->data, i, val - scalar);
        }
        PyMem_Free(idx_a);
        return (PyObject*)result;
    } else if ((PyLong_Check(a) || PyFloat_Check(a)) && PyObject_IsInstance(b, (PyObject*)&VectorizeType)) {
        Vector *vb = (Vector*)b;
        double scalar = PyFloat_AsDouble(a);
        Vector *result = _new_vector_like(vb);
        if (!result) return NULL;
        int total = _vector_iter_total(vb);
        int *idx_b = (int*)PyMem_Malloc(total * sizeof(int));
        if (!idx_b) { Py_DECREF(result); return PyErr_NoMemory(); }
        _vector_get_flat_indices(vb, idx_b, total);
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(vb->data, idx_b[i]);
            Data_set_flat(result->data, i, scalar - val);
        }
        PyMem_Free(idx_b);
        return (PyObject*)result;
    }
    PyErr_SetString(PyExc_TypeError, "unsupported operand type(s) for -");
    return NULL;
}

static PyObject *Vector_mul(PyObject *a, PyObject *b) {
    if (PyObject_IsInstance(a, (PyObject*)&VectorizeType) && PyObject_IsInstance(b, (PyObject*)&VectorizeType)) {
        Vector *va = (Vector*)a;
        Vector *vb = (Vector*)b;
        if (va->dimension != vb->dimension ||
            !_shape_equal(va->shape, vb->shape, va->dimension)) {
            PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same.");
            return NULL;
        }
        Vector *result = _new_vector_like(va);
        if (!result) return NULL;
        int total = _vector_iter_total(va);
        int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
        int *idx_b = (int*)PyMem_Malloc(total * sizeof(int));
        if (!idx_a || !idx_b) { Py_XDECREF(result); PyMem_Free(idx_a); PyMem_Free(idx_b); return PyErr_NoMemory(); }
        _vector_get_flat_indices(va, idx_a, total);
        _vector_get_flat_indices(vb, idx_b, total);
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val_a = Data_get_flat(va->data, idx_a[i]);
            double val_b = Data_get_flat(vb->data, idx_b[i]);
            Data_set_flat(result->data, i, val_a * val_b);
        }
        PyMem_Free(idx_a); PyMem_Free(idx_b);
        return (PyObject*)result;
    } else if (PyObject_IsInstance(a, (PyObject*)&VectorizeType) && (PyLong_Check(b) || PyFloat_Check(b))) {
        Vector *va = (Vector*)a;
        double scalar = PyFloat_AsDouble(b);
        Vector *result = _new_vector_like(va);
        if (!result) return NULL;
        int total = _vector_iter_total(va);
        int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
        if (!idx_a) { Py_DECREF(result); return PyErr_NoMemory(); }
        _vector_get_flat_indices(va, idx_a, total);
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(va->data, idx_a[i]);
            Data_set_flat(result->data, i, val * scalar);
        }
        PyMem_Free(idx_a);
        return (PyObject*)result;
    } else if ((PyLong_Check(a) || PyFloat_Check(a)) && PyObject_IsInstance(b, (PyObject*)&VectorizeType)) {
        return Vector_mul(b, a);
    }
    PyErr_SetString(PyExc_TypeError, "unsupported operand type(s) for *");
    return NULL;
}

static PyObject *Vector_div(PyObject *a, PyObject *b) {
    if (PyObject_IsInstance(a, (PyObject*)&VectorizeType) && PyObject_IsInstance(b, (PyObject*)&VectorizeType)) {
        Vector *va = (Vector*)a;
        Vector *vb = (Vector*)b;
        if (va->dimension != vb->dimension ||
            !_shape_equal(va->shape, vb->shape, va->dimension)) {
            PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same.");
            return NULL;
        }
        Vector *result = _new_vector_like(va);
        if (!result) return NULL;
        int total = _vector_iter_total(va);
        int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
        int *idx_b = (int*)PyMem_Malloc(total * sizeof(int));
        if (!idx_a || !idx_b) { Py_XDECREF(result); PyMem_Free(idx_a); PyMem_Free(idx_b); return PyErr_NoMemory(); }
        _vector_get_flat_indices(va, idx_a, total);
        _vector_get_flat_indices(vb, idx_b, total);
        for (int i = 0; i < total; ++i) {
            double val_b = Data_get_flat(vb->data, idx_b[i]);
            if (val_b == 0.0) {
                Py_DECREF(result);
                PyMem_Free(idx_a); PyMem_Free(idx_b);
                PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
                return NULL;
            }
            double val_a = Data_get_flat(va->data, idx_a[i]);
            Data_set_flat(result->data, i, val_a / val_b);
        }
        PyMem_Free(idx_a); PyMem_Free(idx_b);
        return (PyObject*)result;
    } else if (PyObject_IsInstance(a, (PyObject*)&VectorizeType) && (PyLong_Check(b) || PyFloat_Check(b))) {
        Vector *va = (Vector*)a;
        double scalar = PyFloat_AsDouble(b);
        if (scalar == 0.0) {
            PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
            return NULL;
        }
        Vector *result = _new_vector_like(va);
        if (!result) return NULL;
        int total = _vector_iter_total(va);
        int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
        if (!idx_a) { Py_DECREF(result); return PyErr_NoMemory(); }
        _vector_get_flat_indices(va, idx_a, total);
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(va->data, idx_a[i]);
            Data_set_flat(result->data, i, val / scalar);
        }
        PyMem_Free(idx_a);
        return (PyObject*)result;
    } else if ((PyLong_Check(a) || PyFloat_Check(a)) && PyObject_IsInstance(b, (PyObject*)&VectorizeType)) {
        Vector *vb = (Vector*)b;
        double scalar = PyFloat_AsDouble(a);
        Vector *result = _new_vector_like(vb);
        if (!result) return NULL;
        int total = _vector_iter_total(vb);
        int *idx_b = (int*)PyMem_Malloc(total * sizeof(int));
        if (!idx_b) { Py_DECREF(result); return PyErr_NoMemory(); }
        _vector_get_flat_indices(vb, idx_b, total);
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(vb->data, idx_b[i]);
            if (val == 0.0) {
                Py_DECREF(result);
                PyMem_Free(idx_b);
                PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
                return NULL;
            }
            Data_set_flat(result->data, i, scalar / val);
        }
        PyMem_Free(idx_b);
        return (PyObject*)result;
    }
    PyErr_SetString(PyExc_TypeError, "unsupported operand type(s) for /");
    return NULL;
}

static PyObject *Vector_pow(PyObject *a, PyObject *b, PyObject *mod) {
    (void)mod;
    if (PyObject_IsInstance(a, (PyObject*)&VectorizeType) && PyObject_IsInstance(b, (PyObject*)&VectorizeType)) {
        Vector *va = (Vector*)a;
        Vector *vb = (Vector*)b;
        if (va->dimension != vb->dimension ||
            !_shape_equal(va->shape, vb->shape, va->dimension)) {
            PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same.");
            return NULL;
        }
        Vector *result = _new_vector_like(va);
        if (!result) return NULL;
        int total = _vector_iter_total(va);
        int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
        int *idx_b = (int*)PyMem_Malloc(total * sizeof(int));
        if (!idx_a || !idx_b) { Py_XDECREF(result); PyMem_Free(idx_a); PyMem_Free(idx_b); return PyErr_NoMemory(); }
        _vector_get_flat_indices(va, idx_a, total);
        _vector_get_flat_indices(vb, idx_b, total);
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val_a = Data_get_flat(va->data, idx_a[i]);
            double val_b = Data_get_flat(vb->data, idx_b[i]);
            Data_set_flat(result->data, i, pow(val_a, val_b));
        }
        PyMem_Free(idx_a); PyMem_Free(idx_b);
        return (PyObject*)result;
    } else if (PyObject_IsInstance(a, (PyObject*)&VectorizeType) && (PyLong_Check(b) || PyFloat_Check(b))) {
        Vector *va = (Vector*)a;
        double scalar = PyFloat_AsDouble(b);
        if (PyErr_Occurred()) return NULL;
        Vector *result = _new_vector_like(va);
        if (!result) return NULL;
        int total = _vector_iter_total(va);
        int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
        if (!idx_a) { Py_DECREF(result); return PyErr_NoMemory(); }
        _vector_get_flat_indices(va, idx_a, total);
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(va->data, idx_a[i]);
            Data_set_flat(result->data, i, pow(val, scalar));
        }
        PyMem_Free(idx_a);
        return (PyObject*)result;
    }
    PyErr_SetString(PyExc_TypeError, "unsupported operand type(s) for **");
    return NULL;
}

/* In-place operations: modify self in place, return self */
static PyObject *Vector_iadd(PyObject *self, PyObject *other) {
    if (!PyObject_IsInstance(self, (PyObject*)&VectorizeType) || !PyObject_IsInstance(other, (PyObject*)&VectorizeType)) {
        PyErr_SetString(PyExc_TypeError, "operands must be vector_map_as_tensor");
        return NULL;
    }
    Vector *va = (Vector*)self;
    Vector *vb = (Vector*)other;
    if (va->dimension != vb->dimension ||
        !_shape_equal(va->shape, vb->shape, va->dimension)) {
        PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same.");
        return NULL;
    }
    int total = _vector_iter_total(va);
    int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
    int *idx_b = (int*)PyMem_Malloc(total * sizeof(int));
    if (!idx_a || !idx_b) { PyMem_Free(idx_a); PyMem_Free(idx_b); return PyErr_NoMemory(); }
    _vector_get_flat_indices(va, idx_a, total);
    _vector_get_flat_indices(vb, idx_b, total);
    COS_SIMD_LOOP
    for (int i = 0; i < total; ++i) {
        double new_val = Data_get_flat(va->data, idx_a[i]) + Data_get_flat(vb->data, idx_b[i]);
        Data_set_flat(va->data, idx_a[i], new_val);
    }
    PyMem_Free(idx_a); PyMem_Free(idx_b);
    Py_INCREF(self);
    return self;
}

static PyObject *Vector_isub(PyObject *self, PyObject *other) {
    if (!PyObject_IsInstance(self, (PyObject*)&VectorizeType) || !PyObject_IsInstance(other, (PyObject*)&VectorizeType)) {
        PyErr_SetString(PyExc_TypeError, "operands must be vector_map_as_tensor");
        return NULL;
    }
    Vector *va = (Vector*)self;
    Vector *vb = (Vector*)other;
    if (va->dimension != vb->dimension ||
        !_shape_equal(va->shape, vb->shape, va->dimension)) {
        PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same.");
        return NULL;
    }
    int total = _vector_iter_total(va);
    int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
    int *idx_b = (int*)PyMem_Malloc(total * sizeof(int));
    if (!idx_a || !idx_b) { PyMem_Free(idx_a); PyMem_Free(idx_b); return PyErr_NoMemory(); }
    _vector_get_flat_indices(va, idx_a, total);
    _vector_get_flat_indices(vb, idx_b, total);
    COS_SIMD_LOOP
    for (int i = 0; i < total; ++i) {
        double new_val = Data_get_flat(va->data, idx_a[i]) - Data_get_flat(vb->data, idx_b[i]);
        Data_set_flat(va->data, idx_a[i], new_val);
    }
    PyMem_Free(idx_a); PyMem_Free(idx_b);
    Py_INCREF(self);
    return self;
}

static PyObject *Vector_imul(PyObject *self, PyObject *other) {
    if (PyObject_IsInstance(self, (PyObject*)&VectorizeType) && PyObject_IsInstance(other, (PyObject*)&VectorizeType)) {
        Vector *va = (Vector*)self;
        Vector *vb = (Vector*)other;
        if (va->dimension != vb->dimension ||
            !_shape_equal(va->shape, vb->shape, va->dimension)) {
            PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same.");
            return NULL;
        }
        int total = _vector_iter_total(va);
        int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
        int *idx_b = (int*)PyMem_Malloc(total * sizeof(int));
        if (!idx_a || !idx_b) { PyMem_Free(idx_a); PyMem_Free(idx_b); return PyErr_NoMemory(); }
        _vector_get_flat_indices(va, idx_a, total);
        _vector_get_flat_indices(vb, idx_b, total);
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double new_val = Data_get_flat(va->data, idx_a[i]) * Data_get_flat(vb->data, idx_b[i]);
            Data_set_flat(va->data, idx_a[i], new_val);
        }
        PyMem_Free(idx_a); PyMem_Free(idx_b);
        Py_INCREF(self);
        return self;
    } else if (PyObject_IsInstance(self, (PyObject*)&VectorizeType) && (PyLong_Check(other) || PyFloat_Check(other))) {
        Vector *va = (Vector*)self;
        double scalar = PyFloat_AsDouble(other);
        int total = _vector_iter_total(va);
        int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
        if (!idx_a) return PyErr_NoMemory();
        _vector_get_flat_indices(va, idx_a, total);
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(va->data, idx_a[i]);
            Data_set_flat(va->data, idx_a[i], val * scalar);
        }
        PyMem_Free(idx_a);
        Py_INCREF(self);
        return self;
    }
    PyErr_SetString(PyExc_TypeError, "unsupported operand type(s) for *=");
    return NULL;
}

static PyObject *Vector_itruediv(PyObject *self, PyObject *other) {
    if (PyObject_IsInstance(self, (PyObject*)&VectorizeType) && PyObject_IsInstance(other, (PyObject*)&VectorizeType)) {
        Vector *va = (Vector*)self;
        Vector *vb = (Vector*)other;
        if (va->dimension != vb->dimension ||
            !_shape_equal(va->shape, vb->shape, va->dimension)) {
            PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same.");
            return NULL;
        }
        int total = _vector_iter_total(va);
        int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
        int *idx_b = (int*)PyMem_Malloc(total * sizeof(int));
        if (!idx_a || !idx_b) { PyMem_Free(idx_a); PyMem_Free(idx_b); return PyErr_NoMemory(); }
        _vector_get_flat_indices(va, idx_a, total);
        _vector_get_flat_indices(vb, idx_b, total);
        for (int i = 0; i < total; ++i) {
            double val_b = Data_get_flat(vb->data, idx_b[i]);
            if (val_b == 0.0) {
                PyMem_Free(idx_a); PyMem_Free(idx_b);
                PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
                return NULL;
            }
            double val_a = Data_get_flat(va->data, idx_a[i]);
            Data_set_flat(va->data, idx_a[i], val_a / val_b);
        }
        PyMem_Free(idx_a); PyMem_Free(idx_b);
        Py_INCREF(self);
        return self;
    } else if (PyObject_IsInstance(self, (PyObject*)&VectorizeType) && (PyLong_Check(other) || PyFloat_Check(other))) {
        Vector *va = (Vector*)self;
        double scalar = PyFloat_AsDouble(other);
        if (scalar == 0.0) {
            PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
            return NULL;
        }
        int total = _vector_iter_total(va);
        int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
        if (!idx_a) return PyErr_NoMemory();
        _vector_get_flat_indices(va, idx_a, total);
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(va->data, idx_a[i]);
            Data_set_flat(va->data, idx_a[i], val / scalar);
        }
        PyMem_Free(idx_a);
        Py_INCREF(self);
        return self;
    }
    PyErr_SetString(PyExc_TypeError, "unsupported operand type(s) for /=");
    return NULL;
}

static PyObject *Vector_ipow(PyObject *self, PyObject *other, PyObject *mod) {
    (void)mod;
    if (PyObject_IsInstance(self, (PyObject*)&VectorizeType) && PyObject_IsInstance(other, (PyObject*)&VectorizeType)) {
        Vector *va = (Vector*)self;
        Vector *vb = (Vector*)other;
        if (va->dimension != vb->dimension ||
            !_shape_equal(va->shape, vb->shape, va->dimension)) {
            PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same.");
            return NULL;
        }
        int total = _vector_iter_total(va);
        int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
        int *idx_b = (int*)PyMem_Malloc(total * sizeof(int));
        if (!idx_a || !idx_b) { PyMem_Free(idx_a); PyMem_Free(idx_b); return PyErr_NoMemory(); }
        _vector_get_flat_indices(va, idx_a, total);
        _vector_get_flat_indices(vb, idx_b, total);
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val_a = Data_get_flat(va->data, idx_a[i]);
            double val_b = Data_get_flat(vb->data, idx_b[i]);
            Data_set_flat(va->data, idx_a[i], pow(val_a, val_b));
        }
        PyMem_Free(idx_a); PyMem_Free(idx_b);
        Py_INCREF(self);
        return self;
    } else if (PyObject_IsInstance(self, (PyObject*)&VectorizeType) && (PyLong_Check(other) || PyFloat_Check(other))) {
        Vector *va = (Vector*)self;
        double scalar = PyFloat_AsDouble(other);
        if (PyErr_Occurred()) return NULL;
        int total = _vector_iter_total(va);
        int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
        if (!idx_a) return PyErr_NoMemory();
        _vector_get_flat_indices(va, idx_a, total);
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(va->data, idx_a[i]);
            Data_set_flat(va->data, idx_a[i], pow(val, scalar));
        }
        PyMem_Free(idx_a);
        Py_INCREF(self);
        return self;
    }
    PyErr_SetString(PyExc_TypeError, "unsupported operand type(s) for **=");
    return NULL;
}

/* Unary operators */
static PyObject *Vector_neg(PyObject *self) {
    Vector *va = (Vector*)self;
    Vector *result = _new_vector_like(va);
    if (!result) return NULL;
    int total = _vector_iter_total(va);
    int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
    if (!idx_a) { Py_DECREF(result); return PyErr_NoMemory(); }
    _vector_get_flat_indices(va, idx_a, total);
    COS_SIMD_LOOP
    for (int i = 0; i < total; ++i) {
        double val = Data_get_flat(va->data, idx_a[i]);
        Data_set_flat(result->data, i, -val);
    }
    PyMem_Free(idx_a);
    return (PyObject*)result;
}

static PyObject *Vector_pos(PyObject *self) {
    Vector *va = (Vector*)self;
    Vector *result = _new_vector_like(va);
    if (!result) return NULL;
    int total = _vector_iter_total(va);
    int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
    if (!idx_a) { Py_DECREF(result); return PyErr_NoMemory(); }
    _vector_get_flat_indices(va, idx_a, total);
    COS_SIMD_LOOP
    for (int i = 0; i < total; ++i) {
        double val = Data_get_flat(va->data, idx_a[i]);
        Data_set_flat(result->data, i, val);
    }
    PyMem_Free(idx_a);
    return (PyObject*)result;
}

static PyObject *Vector_abs(PyObject *self) {
    Vector *va = (Vector*)self;
    int total = _vector_iter_total(va);
    int *idx_a = (int*)PyMem_Malloc(total * sizeof(int));
    if (!idx_a) return PyErr_NoMemory();
    _vector_get_flat_indices(va, idx_a, total);
    double sum_sq = 0.0;
    COS_SIMD_LOOP
    for (int i = 0; i < total; ++i) {
        double val = Data_get_flat(va->data, idx_a[i]);
        sum_sq += val * val;
    }
    PyMem_Free(idx_a);
    return PyFloat_FromDouble(sqrt(sum_sq));
}

/* ---------- New methods added ---------- */

/* Implementation of __get_item__ (multi-index) */
static PyObject *Vector_get_item(Vector *self, PyObject *args) {
    int n = (int)PyTuple_Size(args);
    if (n == 0) {
        PyErr_SetString(PyExc_TypeError, "__get_item__ expected at least one index");
        return NULL;
    }
    /* Build index tuple and reuse subscript logic */
    PyObject *idx = n == 1 ? PyTuple_GET_ITEM(args, 0) : args;
    Py_INCREF(idx);
    PyObject *res = Vector_subscript(self, idx);
    Py_DECREF(idx);
    return res;
}

/* __set_item__: fast multi-index value assignment, takes (index_tuple, value) as arguments */
static PyObject *Vector_set_item(Vector *self, PyObject *args) {
    PyObject *indexs, *val_obj;
    if (!PyArg_ParseTuple(args, "OO", &indexs, &val_obj)) {
        return NULL;
    }
    if (!PyTuple_Check(indexs)) {
        PyErr_SetString(PyExc_TypeError, "__set_item__ first argument must be a tuple of indices");
        return NULL;
    }
    int n = (int)PyTuple_Size(indexs);
    if (n == 0) {
        Py_RETURN_NONE;
    }
    /* Reuse ass_subscript logic */
    int res = Vector_ass_subscript(self, indexs, val_obj);
    if (res < 0) return NULL;
    Py_RETURN_NONE;
}

/* __iter__: use the sequence iterator */
static PyObject *Vector_iter(Vector *self) {
    return PySeqIter_New((PyObject*)self);
}

/* ---------- End of new methods ---------- */

static PyTypeObject VectorizeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cos_comparison_pydll.vector_map_as_tensor",
    .tp_basicsize = sizeof(Vector),
    .tp_itemsize  = 0,
    .tp_dealloc   = (destructor)Vector_dealloc,
    .tp_traverse  = (traverseproc)Vector_traverse,
    .tp_clear     = (inquiry)Vector_clear,
    .tp_repr      = (reprfunc)Vector_repr,
    .tp_as_number = &Vector_as_number,
    .tp_as_sequence = &Vector_as_sequence,
    .tp_as_mapping = &Vector_as_mapping,
    .tp_as_buffer = &Vector_as_buffer,
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_doc       = "Maps a 1D list to a multi-dimensional tensor view (C-backed).",
    .tp_methods   = Vector_methods,
    .tp_getset    = Vector_getseters,
    .tp_init      = (initproc)Vector_init,
    .tp_new       = PyType_GenericNew,
};

#endif
