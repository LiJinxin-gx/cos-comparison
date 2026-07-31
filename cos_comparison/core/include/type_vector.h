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

/* -----------------------------------------------------------------------------
 * Optional SIMD auto-vectorization hints - safe no-op fallback for all compilers
 * These are only hints to the compiler, no architecture-specific intrinsics used
 * --------------------------------------------------------------------------- */
#if defined(_MSC_VER)
  /* MSVC: ignore vector dependencies for auto-vectorization */
  #define COS_SIMD_LOOP __pragma(loop(ivdep))
#elif defined(__GNUC__) || defined(__clang__)
  /* GCC/Clang: ignore vector dependencies for auto-vectorization */
  #define COS_SIMD_LOOP _Pragma("GCC ivdep")
#else
  /* Unknown compiler: no-op, no effect */
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

/* Backward compatibility helpers */
static inline int Vector_end(Vector *self) {
    if (self->dimension == 0) return self->start + self->offset;
    int last = self->dimension - 1;
    return self->start + self->offset +
        (self->start_offset[last] + self->shape[last] * self->step_offset[last]) * self->strides[last];
}
static inline int Vector_cache(Vector *self) {
    if (self->dimension == 0) return 1;
    int last = self->dimension - 1;
    return self->strides[last] * self->step_offset[last];
}

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

static PyObject* Vector_get_strides(Vector *self, void *closure) {
    PyObject *tup = PyTuple_New(self->dimension);
    if (!tup) return NULL;
    for (int i = 0; i < self->dimension; ++i) {
        PyTuple_SET_ITEM(tup, i, PyLong_FromLong(self->strides[i]));
    }
    return tup;
}

static PyObject* Vector_get_p(Vector *self, void *closure) {
    /* Backward compatibility: always 0 in new indexing scheme */
    return PyLong_FromLong(0);
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
    {"strides", (getter)Vector_get_strides, NULL,
     "Tuple representing the strides of the current tensor view.", NULL},
    {"start_offset", (getter)Vector_get_start_offset, NULL,
     "Tuple of per-dimension start offsets.", NULL},
    {"step_offset", (getter)Vector_get_step_offset, NULL,
     "Tuple of per-dimension step sizes.", NULL},
    {"offset", (getter)Vector_get_offset, NULL,
     "Accumulated offset from integer indexing.", NULL},
    {"p", (getter)Vector_get_p, NULL,
     "Backward compatibility attribute, always 0.", NULL},
    {"vector", (getter)Vector_get_vector, NULL,
     "Flat list of underlying data (copy, for API compatibility with pure Python backend).", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef Vector_methods[] = {
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
    view->shape = (int*)malloc(self->dimension * sizeof(int));
    if (!view->shape) { Py_DECREF(view); return NULL; }
    memcpy(view->shape, self->shape, self->dimension * sizeof(int));
    
    // Allocate and copy strides
    view->strides = (int*)malloc(self->dimension * sizeof(int));
    if (!view->strides) { free(view->shape); Py_DECREF(view); return NULL; }
    memcpy(view->strides, self->strides, self->dimension * sizeof(int));
    
    // Allocate and copy start_offset
    view->start_offset = (int*)malloc(self->dimension * sizeof(int));
    if (!view->start_offset) { free(view->shape); free(view->strides); Py_DECREF(view); return NULL; }
    memcpy(view->start_offset, self->start_offset, self->dimension * sizeof(int));
    
    // Allocate and copy step_offset
    view->step_offset = (int*)malloc(self->dimension * sizeof(int));
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
    int *num_list = (int*)malloc((dim + 1) * sizeof(int));
    if (!num_list) { PyErr_NoMemory(); return -1; }
    for (int i = 0; i <= dim; ++i) num_list[i] = 1;
    int flag = dim;
    int pos = 0;
    int *indices = (int*)malloc(dim * sizeof(int));
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

static int _infer_shape(PyObject *obj, int **shape, int *dimension) {
    int dim = 0;
    int cap = 4;
    int *sh = (int*)malloc(cap * sizeof(int));
    if (!sh) { PyErr_NoMemory(); return -1; }
    PyObject *cur = obj;
    Py_INCREF(cur);
    while (PySequence_Check(cur)) {
        Py_ssize_t len = PySequence_Size(cur);
        if (dim >= cap) {
            cap *= 2;
            int *new_sh = (int*)realloc(sh, cap * sizeof(int));
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
    if (!PyTuple_Check(obj)) return -1;
    *out_dim = (int)PyTuple_Size(obj);
    *out_shape = (int*)malloc(*out_dim * sizeof(int));
    if (!*out_shape) return -1;
    for (int i = 0; i < *out_dim; ++i) {
        PyObject *item = PyTuple_GetItem(obj, i);
        if (!item || !PyLong_Check(item)) {
            free(*out_shape);
            return -1;
        }
        (*out_shape)[i] = (int)PyLong_AsLong(item);
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
    int p = 0;
    int offset = 0;
    
    static char *kwlist[] = {"vector", "shape", "start", "p", "strides", "offset", "start_offset", "step_offset", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OiiiOO", kwlist,
                                     &vector, &shape_obj, &start, &p, &strides_obj, &offset, &start_offset_obj, &step_offset_obj))
        return -1;

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
        
        // Backward compatibility: handle old p parameter
        int new_dim = src->dimension - p;
        int offset_p = 0;
        if (p > 0) {
            for (int i = 0; i < p; ++i) {
                offset_p += src->strides[i] * (src->start_offset[i] + 0 * src->step_offset[i]);
            }
        }
        self->offset += offset_p;
        
        self->shape = (int*)malloc(new_dim * sizeof(int));
        self->strides = (int*)malloc(new_dim * sizeof(int));
        self->start_offset = (int*)malloc(new_dim * sizeof(int));
        self->step_offset = (int*)malloc(new_dim * sizeof(int));
        if (!self->shape || !self->strides || !self->start_offset || !self->step_offset) {
            PyErr_NoMemory();
            return -1;
        }
        memcpy(self->shape, src->shape + p, new_dim * sizeof(int));
        memcpy(self->strides, src->strides + p, new_dim * sizeof(int));
        memcpy(self->start_offset, src->start_offset + p, new_dim * sizeof(int));
        memcpy(self->step_offset, src->step_offset + p, new_dim * sizeof(int));
        self->dimension = new_dim;
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
            data = (Data*)calloc(1, sizeof(Data));
            if (!data) { free(shape); PyBuffer_Release(&view); PyErr_NoMemory(); return -1; }
            data->dimension = dim;
            data->shape = shape;
            data->strides = (int*)malloc(dim * sizeof(int));
            if (!data->strides) { free(data); free(shape); PyBuffer_Release(&view); PyErr_NoMemory(); return -1; }
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
            self->flags |= VECTOR_FLAG_VIEW | VECTOR_FLAG_BUFFER;
        } else {
            /* Convert path: copy and convert to owned double array */
            data = Data_create(dim, shape);
            if (!data) { free(shape); PyBuffer_Release(&view); PyErr_NoMemory(); return -1; }
            double * COS_RESTRICT out = (double*)data->data;
            const char * COS_RESTRICT in = (const char*)view.buf;
            
            COS_SIMD_LOOP
            for (int i = 0; i < total; ++i) {
                double val;
                const char *p = in + i*elem_size;
                switch (conv_type) {
                    case 2: val = (double)*(const float*)p; break;
                    case 3: val = (double)*(const int*)p; break;
                    case 4: val = (double)*(const short*)p; break;
                    case 5: val = (double)*(const long*)p; break;
                    case 6: val = (double)*(const long long*)p; break;
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
        
        // Backward compatibility: handle old p parameter
        if (p != 0) {
            int new_dim = dim - p;
            int *new_shape = (int*)malloc(new_dim * sizeof(int));
            int *new_strides = (int*)malloc(new_dim * sizeof(int));
            int *new_so = (int*)malloc(new_dim * sizeof(int));
            int *new_sto = (int*)malloc(new_dim * sizeof(int));
            if (!new_shape || !new_strides || !new_so || !new_sto) {
                PyErr_NoMemory();
                return -1;
            }
            memcpy(new_shape, shape + p, new_dim * sizeof(int));
            // Allocate and compute strides
            self->strides = (int*)malloc(dim * sizeof(int));
            if (!self->strides) { PyErr_NoMemory(); return -1; }
            self->strides[dim - 1] = 1;
            for (int i = dim - 2; i >= 0; --i) {
                self->strides[i] = self->strides[i+1] * self->shape[i+1];
            }
            memcpy(new_strides, self->strides + p, new_dim * sizeof(int));
            free(self->strides);
            free(shape);
            for (int i = 0; i < new_dim; ++i) { new_so[i] = 0; new_sto[i] = 1; }
            self->shape = new_shape;
            self->strides = new_strides;
            self->start_offset = new_so;
            self->step_offset = new_sto;
            self->dimension = new_dim;
        } else {
            // Allocate and compute strides for Vector
            self->strides = (int*)malloc(dim * sizeof(int));
            if (!self->strides) { PyErr_NoMemory(); return -1; }
            self->strides[dim - 1] = 1;
            for (int i = dim - 2; i >= 0; --i) {
                self->strides[i] = self->strides[i+1] * self->shape[i+1];
            }
            // Allocate and init start/step offsets
            self->start_offset = (int*)malloc(dim * sizeof(int));
            self->step_offset = (int*)malloc(dim * sizeof(int));
            if (!self->start_offset || !self->step_offset) { PyErr_NoMemory(); return -1; }
            for (int i = 0; i < dim; ++i) {
                self->start_offset[i] = 0;
                self->step_offset[i] = 1;
            }
        }
        
        // Override strides if provided
        if (strides_obj != Py_None && PyTuple_Check(strides_obj)) {
            int n = (int)PyTuple_Size(strides_obj);
            if (n != self->dimension) {
                PyErr_SetString(PyExc_ValueError, "strides length must match shape length");
                return -1;
            }
            for (int i = 0; i < n; ++i) {
                self->strides[i] = (int)PyLong_AsLong(PyTuple_GetItem(strides_obj, i));
            }
        }
        // Override start_offset if provided
        if (start_offset_obj != Py_None && PyTuple_Check(start_offset_obj)) {
            int n = (int)PyTuple_Size(start_offset_obj);
            if (n != self->dimension) {
                PyErr_SetString(PyExc_ValueError, "start_offset length must match shape length");
                return -1;
            }
            for (int i = 0; i < n; ++i) {
                self->start_offset[i] = (int)PyLong_AsLong(PyTuple_GetItem(start_offset_obj, i));
            }
        }
        // Override step_offset if provided
        if (step_offset_obj != Py_None && PyTuple_Check(step_offset_obj)) {
            int n = (int)PyTuple_Size(step_offset_obj);
            if (n != self->dimension) {
                PyErr_SetString(PyExc_ValueError, "step_offset length must match shape length");
                return -1;
            }
            for (int i = 0; i < n; ++i) {
                self->step_offset[i] = (int)PyLong_AsLong(PyTuple_GetItem(step_offset_obj, i));
            }
        }
        
        return 0;
    }
    
    PyErr_Clear();  /* not a buffer, fallback to copying */
fallback_sequence:
    
    /* Case 3: copy data into a new double array */
    int *shape = NULL;
    int dim = 0;
    int explicit_shape = 0;

    /* Try explicit shape first */
    if (shape_obj != Py_None && PyTuple_Check(shape_obj) && _parse_shape_tuple(shape_obj, &shape, &dim) == 0) {
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
    
    // Backward compatibility: handle old p parameter
    if (p != 0) {
        int new_dim = dim - p;
        int *new_shape = (int*)malloc(new_dim * sizeof(int));
        int *new_strides = (int*)malloc(new_dim * sizeof(int));
        int *new_so = (int*)malloc(new_dim * sizeof(int));
        int *new_sto = (int*)malloc(new_dim * sizeof(int));
        if (!new_shape || !new_strides || !new_so || !new_sto) {
            PyErr_NoMemory();
            return -1;
        }
        memcpy(new_shape, shape + p, new_dim * sizeof(int));
        // Compute full strides first
        int *full_strides = (int*)malloc(dim * sizeof(int));
        full_strides[dim - 1] = 1;
        for (int i = dim - 2; i >= 0; --i) {
            full_strides[i] = full_strides[i+1] * shape[i+1];
        }
        memcpy(new_strides, full_strides + p, new_dim * sizeof(int));
        free(full_strides);
        free(shape);
        for (int i = 0; i < new_dim; ++i) { new_so[i] = 0; new_sto[i] = 1; }
        self->shape = new_shape;
        self->strides = new_strides;
        self->start_offset = new_so;
        self->step_offset = new_sto;
        self->dimension = new_dim;
    } else {
        self->shape = shape;
        self->dimension = dim;
        // Allocate and compute strides
        self->strides = (int*)malloc(dim * sizeof(int));
        if (!self->strides) { Data_free(data); free(shape); PyErr_NoMemory(); return -1; }
        self->strides[dim - 1] = 1;
        for (int i = dim - 2; i >= 0; --i) {
            self->strides[i] = self->strides[i+1] * self->shape[i+1];
        }
        // Allocate and init offsets
        self->start_offset = (int*)malloc(dim * sizeof(int));
        self->step_offset = (int*)malloc(dim * sizeof(int));
        if (!self->start_offset || !self->step_offset) { Data_free(data); PyErr_NoMemory(); return -1; }
        for (int i = 0; i < dim; ++i) {
            self->start_offset[i] = 0;
            self->step_offset[i] = 1;
        }
    }
    
    // Override strides if provided
    if (strides_obj != Py_None && PyTuple_Check(strides_obj)) {
        int n = (int)PyTuple_Size(strides_obj);
        if (n != self->dimension) {
            PyErr_SetString(PyExc_ValueError, "strides length must match shape length");
            return -1;
        }
        for (int i = 0; i < n; ++i) {
            self->strides[i] = (int)PyLong_AsLong(PyTuple_GetItem(strides_obj, i));
        }
    }
    // Override start_offset if provided
    if (start_offset_obj != Py_None && PyTuple_Check(start_offset_obj)) {
        int n = (int)PyTuple_Size(start_offset_obj);
        if (n != self->dimension) {
            PyErr_SetString(PyExc_ValueError, "start_offset length must match shape length");
            return -1;
        }
        for (int i = 0; i < n; ++i) {
            self->start_offset[i] = (int)PyLong_AsLong(PyTuple_GetItem(start_offset_obj, i));
        }
    }
    // Override step_offset if provided
    if (step_offset_obj != Py_None && PyTuple_Check(step_offset_obj)) {
        int n = (int)PyTuple_Size(step_offset_obj);
        if (n != self->dimension) {
            PyErr_SetString(PyExc_ValueError, "step_offset length must match shape length");
            return -1;
        }
        for (int i = 0; i < n; ++i) {
            self->step_offset[i] = (int)PyLong_AsLong(PyTuple_GetItem(step_offset_obj, i));
        }
    }
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

/* Vector_subscript: supports any mix of int/slice indices, arbitrary steps */
static PyObject *Vector_subscript(Vector *self, PyObject *item) {
    /* Normalize single index to tuple */
    PyObject *index_tuple;
    int tuple_created = 0;
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
    int *new_shape = (int*)malloc(new_dim * sizeof(int));
    int *new_strides = (int*)malloc(new_dim * sizeof(int));
    int *new_so = (int*)malloc(new_dim * sizeof(int));
    int *new_sto = (int*)malloc(new_dim * sizeof(int));
    int new_offset = self->offset;
    if (!new_shape || !new_strides || !new_so || !new_sto) {
        if (tuple_created) Py_DECREF(index_tuple);
        free(new_shape); free(new_strides); free(new_so); free(new_sto);
        return PyErr_NoMemory();
    }
    
    int pos = 0;
    /* Process each index */
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
    /* Add remaining unindexed dimensions */
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
    
    /* Otherwise create new view */
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
    
    /* List/tuple assignment */
    if (PyList_Check(value) || PyTuple_Check(value)) {
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
            if (strcmp(buf.format, "d") == 0) val = *(const double*)p;
            else if (strcmp(buf.format, "f") == 0) val = (double)*(const float*)p;
            else if (strcmp(buf.format, "i") == 0 || strcmp(buf.format, "I") == 0) val = (double)*(const int*)p;
            else if (strcmp(buf.format, "l") == 0 || strcmp(buf.format, "L") == 0) val = (double)*(const long*)p;
            else if (strcmp(buf.format, "q") == 0 || strcmp(buf.format, "Q") == 0) val = (double)*(const long long*)p;
            else if (strcmp(buf.format, "h") == 0 || strcmp(buf.format, "H") == 0) val = (double)*(const short*)p;
            else if (strcmp(buf.format, "B") == 0 || strcmp(buf.format, "b") == 0) val = (double)*(const unsigned char*)p;
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
                                "<vector_map_as_tensor: dim=%d, start=%d, end=%d, offset=%d>",
                                self->dimension, self->start, Vector_end(self), self->offset);
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
    result->shape = (int*)malloc(ndim * sizeof(int));
    if (!result->shape) { Data_free(result->data); Py_DECREF(result); return NULL; }
    memcpy(result->shape, src->shape, ndim * sizeof(int));
    // Precompute strides for new tensor
    result->strides = (int*)malloc(ndim * sizeof(int));
    if (!result->strides) { free(result->shape); Data_free(result->data); Py_DECREF(result); return NULL; }
    result->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        result->strides[i] = result->strides[i+1] * result->shape[i+1];
    }
    // Init offsets
    result->start_offset = (int*)malloc(ndim * sizeof(int));
    result->step_offset = (int*)malloc(ndim * sizeof(int));
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
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_doc       = "Maps a 1D list to a multi-dimensional tensor view (C-backed).",
    .tp_methods   = Vector_methods,
    .tp_getset    = Vector_getseters,
    .tp_init      = (initproc)Vector_init,
    .tp_new       = PyType_GenericNew,
};

#endif
