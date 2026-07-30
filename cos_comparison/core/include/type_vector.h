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
    Data     *data;      /* underlying flat array (shared or owned) */
    int      *shape;     /* shape of this view */
    int      *strides;   /* strides for each dimension (precomputed) */
    int       dimension; /* number of dimensions */
    int       start;     /* start offset (flat index) */
    int       p;         /* current depth */
    PyObject *owner;     /* the object that owns data (ref'd), or NULL */
    Py_buffer *buf;      /* saved Py_buffer for zero-copy support, released on dealloc */
    int       flags;     /* internal flags (VECTOR_FLAG_*) */
} Vector;

/* Helper inline functions for strides-based access */
static inline int Vector_end(Vector *self) {
    return self->start + self->shape[self->p] * self->strides[self->p];
}
static inline int Vector_cache(Vector *self) {
    return self->strides[self->p];
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
    int ndim = self->dimension - self->p;
    PyObject *tup = PyTuple_New(ndim);
    if (!tup) return NULL;
    for (int i = 0; i < ndim; ++i) {
        PyTuple_SET_ITEM(tup, i, PyLong_FromLong(self->shape[self->p + i]));
    }
    return tup;
}

static PyObject* Vector_get_strides(Vector *self, void *closure) {
    int ndim = self->dimension - self->p;
    PyObject *tup = PyTuple_New(ndim);
    if (!tup) return NULL;
    for (int i = 0; i < ndim; ++i) {
        PyTuple_SET_ITEM(tup, i, PyLong_FromLong(self->strides[self->p + i]));
    }
    return tup;
}

static PyObject* Vector_get_vector(Vector *self, void *closure) {
    int len = Vector_end(self) - self->start;
    PyObject *list = PyList_New(len);
    if (!list) return NULL;
    double *data = (double*)self->data->data;
    COS_SIMD_LOOP
    for (int i = 0; i < len; ++i) {
        PyList_SET_ITEM(list, i, PyFloat_FromDouble(data[self->start + i]));
    }
    return list;
}

static PyGetSetDef Vector_getseters[] = {
    {"tensor_size", (getter)Vector_get_shape, NULL,
     "Tuple representing the shape of the current tensor view (backward compatibility alias for shape).", NULL},
    {"shape", (getter)Vector_get_shape, NULL,
     "Tuple representing the shape of the current tensor view.", NULL},
    {"strides", (getter)Vector_get_strides, NULL,
     "Tuple representing the strides of the current tensor view.", NULL},
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
    if (i < 0) i += self->shape[self->p];
    if (i < 0 || i >= self->shape[self->p]) {
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
    int ptr = self->start;
    for (int i = 0; i < n; ++i) {
        PyObject *idx_obj = PyTuple_GET_ITEM(index_tuple, i);
        int idx = (int)PyLong_AsLong(idx_obj);
        ptr += idx * self->strides[self->p + i];
    }
    return ptr;
}

/* Bounds check for multi-dimensional indices */
static inline int _vector_check_indices(const Vector *self, PyObject *index_tuple, int n) {
    for (int i = 0; i < n; ++i) {
        PyObject *idx_obj = PyTuple_GET_ITEM(index_tuple, i);
        int idx = (int)PyLong_AsLong(idx_obj);
        if (idx < 0) idx += self->shape[self->p + i];
        if (idx < 0 || idx >= self->shape[self->p + i]) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return -1;
        }
    }
    return 0;
}

/* Create a new view of the same type as self (supports subclasses, GC-safe) */
static inline Vector* _vector_new_view(Vector *self, int new_start, int new_p) {
    PyTypeObject *type = Py_TYPE(self);
    Vector *view = (Vector*)type->tp_alloc(type, 0);
    if (!view) return NULL;
    
    view->data = self->data;
    view->owner = self->owner ? self->owner : (PyObject*)self;
    Py_INCREF(view->owner);
    view->flags = self->flags | VECTOR_FLAG_VIEW;
    view->dimension = self->dimension;
    
    // Allocate and copy shape
    view->shape = (int*)malloc(self->dimension * sizeof(int));
    if (!view->shape) {
        Py_DECREF(view);
        return NULL;
    }
    memcpy(view->shape, self->shape, self->dimension * sizeof(int));
    
    // Allocate and copy strides (views share the same strides as parent)
    view->strides = (int*)malloc(self->dimension * sizeof(int));
    if (!view->strides) {
        free(view->shape);
        Py_DECREF(view);
        return NULL;
    }
    memcpy(view->strides, self->strides, self->dimension * sizeof(int));
    
    view->start = new_start;
    view->p = new_p;
    
    /* tp_alloc for GC types automatically tracks objects in Python 3.13+, no manual GC_Track needed */
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

/* Vector initialization – matches pure Python API: (vector, tensor_size, start=0, end=None, p=0, cache=None) */
static int Vector_init(Vector *self, PyObject *args, PyObject *kwargs) {
    PyObject *vector = NULL;
    PyObject *shape_obj = Py_None;
    PyObject *strides_obj = Py_None;
    int start = 0;
    int p = 0;
    
    static char *kwlist[] = {"vector", "shape", "start", "p", "strides", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OiiO", kwlist,
                                     &vector, &shape_obj, &start, &p, &strides_obj))
        return -1;

    self->flags = 0;
    self->buf = NULL;
    
    /* Case 1: vector is another Vector -> create a sub-view */
    if (PyObject_IsInstance(vector, (PyObject*)&VectorizeType)) {
        Vector *src = (Vector*)vector;
        self->data = src->data;
        self->owner = src->owner ? src->owner : vector;
        Py_INCREF(self->owner);
        self->flags |= VECTOR_FLAG_VIEW;
        self->shape = (int*)malloc(src->dimension * sizeof(int));
        if (!self->shape) { PyErr_NoMemory(); return -1; }
        memcpy(self->shape, src->shape, src->dimension * sizeof(int));
        // Allocate and copy strides
        self->strides = (int*)malloc(src->dimension * sizeof(int));
        if (!self->strides) { free(self->shape); PyErr_NoMemory(); return -1; }
        memcpy(self->strides, src->strides, src->dimension * sizeof(int));
        self->dimension = src->dimension;
        self->start = src->start + start;
        self->p = src->p + p;
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
        self->p = p;
        // Allocate and compute strides for Vector
        self->strides = (int*)malloc(dim * sizeof(int));
        if (!self->strides) { Data_free(data); free(shape); PyErr_NoMemory(); return -1; }
        self->strides[dim - 1] = 1;
        for (int i = dim - 2; i >= 0; --i) {
            self->strides[i] = self->strides[i+1] * self->shape[i+1];
        }
        // Override strides if provided
        if (strides_obj != Py_None && PyTuple_Check(strides_obj)) {
            int n = (int)PyTuple_Size(strides_obj);
            if (n != dim) {
                PyErr_SetString(PyExc_ValueError, "strides length must match shape length");
                return -1;
            }
            for (int i = 0; i < n; ++i) {
                self->strides[i] = (int)PyLong_AsLong(PyTuple_GetItem(strides_obj, i));
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
            // Preserve original exception (index error, type error, etc.) for upper layer handling
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
    self->shape = shape;
    self->dimension = dim;
    self->start = start;
    self->p = p;
    // Allocate and compute strides
    self->strides = (int*)malloc(dim * sizeof(int));
    if (!self->strides) { Data_free(data); free(shape); PyErr_NoMemory(); return -1; }
    self->strides[dim - 1] = 1;
    for (int i = dim - 2; i >= 0; --i) {
        self->strides[i] = self->strides[i+1] * self->shape[i+1];
    }
    // Override strides if provided
    if (strides_obj != Py_None && PyTuple_Check(strides_obj)) {
        int n = (int)PyTuple_Size(strides_obj);
        if (n != dim) {
            PyErr_SetString(PyExc_ValueError, "strides length must match shape length");
            return -1;
        }
        for (int i = 0; i < n; ++i) {
            self->strides[i] = (int)PyLong_AsLong(PyTuple_GetItem(strides_obj, i));
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
    if (self->shape)
        free(self->shape);
    if (self->strides)
        free(self->strides);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static Py_ssize_t Vector_len(Vector *self) {
    if (self->p >= self->dimension) return 0;
    return self->shape[self->p];
}

/* Vector_subscript: supports both integer and tuple indices, and slice */
static PyObject *Vector_subscript(Vector *self, PyObject *item) {
    /* Handle slice (single) */
    if (PySlice_Check(item)) {
        Py_ssize_t start, stop, step, length;
        length = self->shape[self->p];
        if (PySlice_GetIndicesEx(item, length, &start, &stop, &step, &length) < 0)
            return NULL;
        if (step != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "Non-unit step slicing not supported");
            return NULL;
        }
        int new_start = self->start + (int)start * Vector_cache(self);
        Vector *slice = _vector_new_view(self, new_start, self->p);
        if (!slice) return NULL;
        slice->shape[self->p] = (int)(stop - start); // Update current dimension size for slice
        return (PyObject*)slice;
    }

    /* Handle tuple of indices (may contain slices) */
    if (PyTuple_Check(item)) {
        int n = (int)PyTuple_Size(item);
        if (n == 0) {
            PyErr_SetString(PyExc_IndexError, "empty tuple index");
            return NULL;
        }
        Vector *cur = self;
        Py_INCREF(cur);
        for (int i = 0; i < n; ++i) {
            PyObject *idx_obj = PyTuple_GetItem(item, i);
            if (!idx_obj) { Py_DECREF(cur); return NULL; }
            if (PySlice_Check(idx_obj)) {
                // Slicing at current dimension
                Py_ssize_t start, stop, step, length;
                length = cur->shape[cur->p];
                if (PySlice_GetIndicesEx(idx_obj, length, &start, &stop, &step, &length) < 0) {
                    Py_DECREF(cur);
                    return NULL;
                }
                if (step != 1) {
                    Py_DECREF(cur);
                    PyErr_SetString(PyExc_NotImplementedError, "Non-unit step slicing not supported");
                    return NULL;
                }
                int new_start = cur->start + (int)start * Vector_cache(cur);
                // Create sliced view at current depth, p unchanged
                Vector *slice = _vector_new_view(cur, new_start, cur->p);
                if (!slice) { Py_DECREF(cur); return NULL; }
                slice->shape[cur->p] = (int)(stop - start); // Update current dimension size for slice
                Py_DECREF(cur);
                cur = slice;
                // Continue with remaining indices (if any)
                // But if we sliced, we cannot further index because the shape might change.
                // Actually, slicing does not change p, so we can continue indexing.
                continue;
            }
            if (!PyLong_Check(idx_obj)) {
                Py_DECREF(cur);
                PyErr_SetString(PyExc_TypeError, "indices must be integers or slices");
                return NULL;
            }
            int idx = (int)PyLong_AsLong(idx_obj);
            int dim_len = cur->shape[cur->p];
            if (idx < 0) idx += dim_len;
            if (idx < 0 || idx >= dim_len) {
                Py_DECREF(cur);
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return NULL;
            }
            if (cur->p == cur->dimension - 1) {
                /* leaf: return value */
                double val = Data_get_flat(cur->data, cur->start + idx * Vector_cache(cur));
                Py_DECREF(cur);
                return PyFloat_FromDouble(val);
            } else {
                /* create subview (matches __getitem__ single index behavior) */
                int new_start = cur->start + idx * Vector_cache(cur);
                int new_p = cur->p + 1;
                Vector *sub = _vector_new_view(cur, new_start, new_p);
                if (!sub) { Py_DECREF(cur); return NULL; }
                Py_DECREF(cur);
                cur = sub;
            }
        }
        return (PyObject*)cur;
    }

    /* integer index (single) */
    if (!PyLong_Check(item)) {
        PyErr_SetString(PyExc_TypeError, "indices must be integers, slices, or tuple");
        return NULL;
    }
    int idx = (int)PyLong_AsLong(item);
    if (idx < 0) idx += self->shape[self->p];
    if (idx < 0 || idx >= self->shape[self->p]) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }
    
    if (self->p < self->dimension - 1) {
        int new_start = self->start + idx * Vector_cache(self);
        int new_p = self->p + 1;
        
        Vector *result = _vector_new_view(self, new_start, new_p);
        if (!result) return NULL;
        return (PyObject*)result;
    } else {
        return PyFloat_FromDouble(Data_get_flat(self->data, self->start + idx * Vector_cache(self)));
    }
}

static int Vector_ass_subscript(Vector *self, PyObject *item, PyObject *value) {
    /* Handle tuple of all integers: fast path */
    if (PyTuple_Check(item)) {
        int n = (int)PyTuple_Size(item);
        int all_int = 1;
        for (int i = 0; i < n; ++i) {
            if (!PyLong_Check(PyTuple_GetItem(item, i))) {
                all_int = 0;
                break;
            }
        }
        if (all_int) {
            int ndim = self->dimension - self->p;
            if (n != ndim) {
                PyErr_Format(PyExc_IndexError, "expected %d indices, got %d", ndim, n);
                return -1;
            }
            /* Fast path: use inline index calculation */
            if (_vector_check_indices(self, item, n) < 0) return -1;
            int ptr = _vector_calc_flat_index(self, item, n);
            double val = PyFloat_AsDouble(value);
            if (val == -1.0 && PyErr_Occurred()) {
                return -1;
            }
            Data_set_flat(self->data, ptr, val);
            return 0;
        }
    }
    /* Handle single integer index assignment (leaf only) */
    if (PyLong_Check(item)) {
        int idx = (int)PyLong_AsLong(item);
        int ndim = self->dimension - self->p;
        if (ndim != 1) {
            PyErr_SetString(PyExc_IndexError, "single index assignment only allowed at 1D view");
            return -1;
        }
        if (idx < 0) idx += self->shape[self->p];
        if (idx < 0 || idx >= self->shape[self->p]) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return -1;
        }
        double val = PyFloat_AsDouble(value);
        if (val == -1.0 && PyErr_Occurred()) {
            return -1;
        }
        Data_set_flat(self->data, self->start + idx * Vector_cache(self), val);
        return 0;
    }
    /* Handle slice assignment (only at leaf) */
    if (PySlice_Check(item)) {
        if (self->p != self->dimension - 1) {
            PyErr_SetString(PyExc_IndexError, "Slice assignment only allowed at leaf dimension");
            return -1;
        }
        Py_ssize_t start, stop, step, slice_len;
        Py_ssize_t length = self->shape[self->p];
        if (PySlice_GetIndicesEx(item, length, &start, &stop, &step, &slice_len) < 0)
            return -1;
        if (step != 1) {
            PyErr_SetString(PyExc_NotImplementedError, "Non-unit step slicing not supported");
            return -1;
        }
        int count = (int)(stop - start);
        int start_idx = (int)start;
        // Convert value to list of doubles
        if (value == Py_None) {
            // Clear slice? Not supported.
            PyErr_SetString(PyExc_TypeError, "Cannot assign None to slice");
            return -1;
        }
        if (PyFloat_Check(value) || PyLong_Check(value)) {
            // Scalar: assign to all
            double scalar = PyFloat_AsDouble(value);
            if (PyErr_Occurred()) return -1;
            COS_SIMD_LOOP
            for (int i = 0; i < count; ++i) {
                Data_set_flat(self->data, self->start + (start_idx + i) * Vector_cache(self), scalar);
            }
            return 0;
        } else if (PyList_Check(value) || PyTuple_Check(value)) {
            // Sequence
            Py_ssize_t seq_len = PySequence_Size(value);
            if ((int)seq_len != count) {
                PyErr_SetString(PyExc_ValueError, "length of value does not match slice length");
                return -1;
            }
            COS_SIMD_LOOP
            for (int i = 0; i < count; ++i) {
                PyObject *item_val = PySequence_GetItem(value, i);
                if (!item_val) return -1;
                double d = PyFloat_AsDouble(item_val);
                Py_DECREF(item_val);
                if (PyErr_Occurred()) return -1;
                Data_set_flat(self->data, self->start + (start_idx + i) * Vector_cache(self), d);
            }
            return 0;
        } else if (PyObject_IsInstance(value, (PyObject*)&VectorizeType)) {
            Vector *v = (Vector*)value;
            if (Vector_end(v) - v->start != count) {
                PyErr_SetString(PyExc_ValueError, "length of Vector does not match slice length");
                return -1;
            }
            COS_SIMD_LOOP
            for (int i = 0; i < count; ++i) {
                double d = Data_get_flat(v->data, v->start + i);
                Data_set_flat(self->data, self->start + (start_idx + i) * Vector_cache(self), d);
            }
            return 0;
        } else if (PyObject_CheckBuffer(value)) {
            Py_buffer buf;
            if (PyObject_GetBuffer(value, &buf, PyBUF_FORMAT | PyBUF_STRIDED) < 0)
                return -1;
            // Check buffer is 1D
            if (buf.ndim != 1) {
                PyBuffer_Release(&buf);
                PyErr_SetString(PyExc_TypeError, "buffer must be 1-dimensional for slice assignment");
                return -1;
            }
            if (buf.shape[0] != count) {
                PyBuffer_Release(&buf);
                PyErr_SetString(PyExc_ValueError, "buffer length does not match slice length");
                return -1;
            }
            int dtype = 0; /* 0 = double, 1 = unsigned char */
            if (buf.format) {
                if (strcmp(buf.format, "B") == 0 || strcmp(buf.format, "b") == 0) {
                    dtype = 1;
                    if (buf.itemsize != sizeof(unsigned char)) {
                        PyBuffer_Release(&buf);
                        PyErr_SetString(PyExc_TypeError, "unsigned char buffer has invalid itemsize");
                        return -1;
                    }
                } else if (strcmp(buf.format, "d") == 0) {
                    dtype = 0;
                    if (buf.itemsize != sizeof(double)) {
                        PyBuffer_Release(&buf);
                        PyErr_SetString(PyExc_TypeError, "double buffer has invalid itemsize");
                        return -1;
                    }
                }
                /* Other formats fall back to double for compatibility */
            }
            if (dtype == 0) {
                double *buf_ptr = (double*)buf.buf;
                Py_ssize_t stride = buf.strides[0] / sizeof(double);
                COS_SIMD_LOOP
                for (int i = 0; i < count; ++i) {
                    double d = buf_ptr[i * stride];
                    Data_set_flat(self->data, self->start + (start_idx + i) * Vector_cache(self), d);
                }
            } else {
                unsigned char *buf_ptr = (unsigned char*)buf.buf;
                Py_ssize_t stride = buf.strides[0] / sizeof(unsigned char);
                COS_SIMD_LOOP
                for (int i = 0; i < count; ++i) {
                    double d = (double)buf_ptr[i * stride];
                    Data_set_flat(self->data, self->start + (start_idx + i) * Vector_cache(self), d);
                }
            }
            PyBuffer_Release(&buf);
            return 0;
        } else {
            PyErr_SetString(PyExc_TypeError, "value must be scalar, sequence, Vector, or buffer-like object");
            return -1;
        }
    }

    /* Handle tuple (slice in tuple not supported yet) */
    if (PyTuple_Check(item)) {
        int tuple_len = (int)PyTuple_Size(item);
        for (int i = 0; i < tuple_len; ++i) {
            PyObject *obj = PyTuple_GetItem(item, i);
            if (PySlice_Check(obj)) {
                PyErr_SetString(PyExc_NotImplementedError, "Slice assignment in tuple not supported");
                return -1;
            }
        }
        // All integer tuples are already handled by the fast path above
        PyErr_SetString(PyExc_IndexError, "invalid tuple index");
        return -1;
    }

    /* Single integer */
    if (!PyLong_Check(item)) {
        PyErr_SetString(PyExc_TypeError, "Key must be int, slice, or tuple");
        return -1;
    }
    int idx = (int)PyLong_AsLong(item);
    if (idx < 0) idx += self->shape[self->p];
    if (idx < 0 || idx >= self->shape[self->p]) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return -1;
    }
    if (self->p < self->dimension - 1) {
        PyErr_SetString(PyExc_IndexError, "Cannot assign to a non-leaf tensor");
        return -1;
    }
    double d = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) return -1;
    Data_set_flat(self->data, self->start + idx * Vector_cache(self), d);
    return 0;
}

static PyObject *Vector_mean(Vector *self, PyObject *args) {
    int len = Vector_end(self) - self->start;
    if (len == 0) Py_RETURN_NONE;
    // Welford's online algorithm for numerically stable mean (no overflow)
    double mean = 0.0;
    COS_SIMD_LOOP
    for (int i = 0; i < len; ++i) {
        double val = Data_get_flat(self->data, self->start + i);
        double delta = val - mean;
        mean += delta / (i + 1);
    }
    return PyFloat_FromDouble(mean);
}

static PyObject *Vector_variance(Vector *self, PyObject *args) {
    int len = Vector_end(self) - self->start;
    if (len == 0) Py_RETURN_NONE;
    // Welford's online algorithm for numerically stable variance (no large sum overflow)
    double mean = 0.0;
    double M2 = 0.0;
    COS_SIMD_LOOP
    for (int i = 0; i < len; ++i) {
        double val = Data_get_flat(self->data, self->start + i);
        double delta = val - mean;
        mean += delta / (i + 1);
        double delta2 = val - mean;
        M2 += delta * delta2;
    }
    double var = M2 / (double)len; // population variance
    return PyFloat_FromDouble(var);
}

static PyObject *Vector_repr(Vector *self) {
    return PyUnicode_FromFormat(
                                "<vector_map_as_tensor: dim=%d, start=%d, end=%d, p=%d, cache=%d>",
                                self->dimension, self->start, Vector_end(self), self->p, Vector_cache(self));
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
    int ndim = src->dimension - src->p; // New tensor uses shape from current p onwards
    int *new_shape = src->shape + src->p;
    result->data = Data_create(ndim, new_shape);
    if (!result->data) { Py_DECREF(result); return NULL; }
    result->owner = NULL;
    result->flags = VECTOR_FLAG_OWNED;
    result->shape = (int*)malloc(ndim * sizeof(int));
    if (!result->shape) { Data_free(result->data); Py_DECREF(result); return NULL; }
    memcpy(result->shape, new_shape, ndim * sizeof(int));
    // Precompute strides for new tensor
    result->strides = (int*)malloc(ndim * sizeof(int));
    if (!result->strides) { free(result->shape); Data_free(result->data); Py_DECREF(result); return NULL; }
    result->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        result->strides[i] = result->strides[i+1] * result->shape[i+1];
    }
    result->dimension = ndim;
    result->start = 0;
    result->p = 0;
    
    /* tp_alloc for GC types automatically tracks objects in Python 3.13+, no manual GC_Track needed */
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
        int total = Vector_end(va) - va->start;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val_a = Data_get_flat(va->data, va->start + i);
            double val_b = Data_get_flat(vb->data, vb->start + i);
            Data_set_flat(result->data, i, val_a + val_b);
        }
        return (PyObject*)result;
    } else if (PyObject_IsInstance(a, (PyObject*)&VectorizeType) && (PyLong_Check(b) || PyFloat_Check(b))) {
        Vector *va = (Vector*)a;
        double scalar = PyFloat_AsDouble(b);
        Vector *result = _new_vector_like(va);
        if (!result) return NULL;
        int total = Vector_end(va) - va->start;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(va->data, va->start + i);
            Data_set_flat(result->data, i, val + scalar);
        }
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
        int total = Vector_end(va) - va->start;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val_a = Data_get_flat(va->data, va->start + i);
            double val_b = Data_get_flat(vb->data, vb->start + i);
            Data_set_flat(result->data, i, val_a - val_b);
        }
        return (PyObject*)result;
    } else if (PyObject_IsInstance(a, (PyObject*)&VectorizeType) && (PyLong_Check(b) || PyFloat_Check(b))) {
        Vector *va = (Vector*)a;
        double scalar = PyFloat_AsDouble(b);
        Vector *result = _new_vector_like(va);
        if (!result) return NULL;
        int total = Vector_end(va) - va->start;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(va->data, va->start + i);
            Data_set_flat(result->data, i, val - scalar);
        }
        return (PyObject*)result;
    } else if ((PyLong_Check(a) || PyFloat_Check(a)) && PyObject_IsInstance(b, (PyObject*)&VectorizeType)) {
        Vector *vb = (Vector*)b;
        double scalar = PyFloat_AsDouble(a);
        Vector *result = _new_vector_like(vb);
        if (!result) return NULL;
        int total = Vector_end(vb) - vb->start;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(vb->data, vb->start + i);
            Data_set_flat(result->data, i, scalar - val);
        }
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
        int total = Vector_end(va) - va->start;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val_a = Data_get_flat(va->data, va->start + i);
            double val_b = Data_get_flat(vb->data, vb->start + i);
            Data_set_flat(result->data, i, val_a * val_b);
        }
        return (PyObject*)result;
    } else if (PyObject_IsInstance(a, (PyObject*)&VectorizeType) && (PyLong_Check(b) || PyFloat_Check(b))) {
        Vector *va = (Vector*)a;
        double scalar = PyFloat_AsDouble(b);
        Vector *result = _new_vector_like(va);
        if (!result) return NULL;
        int total = Vector_end(va) - va->start;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(va->data, va->start + i);
            Data_set_flat(result->data, i, val * scalar);
        }
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
        int total = Vector_end(va) - va->start;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val_b = Data_get_flat(vb->data, vb->start + i);
            if (val_b == 0.0) {
                Py_DECREF(result);
                PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
                return NULL;
            }
            double val_a = Data_get_flat(va->data, va->start + i);
            Data_set_flat(result->data, i, val_a / val_b);
        }
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
        int total = Vector_end(va) - va->start;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(va->data, va->start + i);
            Data_set_flat(result->data, i, val / scalar);
        }
        return (PyObject*)result;
    } else if ((PyLong_Check(a) || PyFloat_Check(a)) && PyObject_IsInstance(b, (PyObject*)&VectorizeType)) {
        Vector *vb = (Vector*)b;
        double scalar = PyFloat_AsDouble(a);
        Vector *result = _new_vector_like(vb);
        if (!result) return NULL;
        int total = Vector_end(vb) - vb->start;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(vb->data, vb->start + i);
            if (val == 0.0) {
                Py_DECREF(result);
                PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
                return NULL;
            }
            Data_set_flat(result->data, i, scalar / val);
        }
        return (PyObject*)result;
    }
    PyErr_SetString(PyExc_TypeError, "unsupported operand type(s) for /");
    return NULL;
}

static PyObject *Vector_pow(PyObject *a, PyObject *b, PyObject *mod) {
    // mod parameter is ignored (per Python number protocol, not supported for tensors)
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
        int total = Vector_end(va) - va->start;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val_a = Data_get_flat(va->data, va->start + i);
            double val_b = Data_get_flat(vb->data, vb->start + i);
            Data_set_flat(result->data, i, pow(val_a, val_b));
        }
        return (PyObject*)result;
    } else if (PyObject_IsInstance(a, (PyObject*)&VectorizeType) && (PyLong_Check(b) || PyFloat_Check(b))) {
        Vector *va = (Vector*)a;
        double scalar = PyFloat_AsDouble(b);
        if (PyErr_Occurred()) return NULL;
        Vector *result = _new_vector_like(va);
        if (!result) return NULL;
        int total = Vector_end(va) - va->start;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(va->data, va->start + i);
            Data_set_flat(result->data, i, pow(val, scalar));
        }
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
    int total = Vector_end(va) - va->start;
    COS_SIMD_LOOP
    for (int i = 0; i < total; ++i) {
        double new_val = Data_get_flat(va->data, va->start + i) + Data_get_flat(vb->data, vb->start + i);
        Data_set_flat(va->data, va->start + i, new_val);
    }
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
    int total = Vector_end(va) - va->start;
    COS_SIMD_LOOP
    for (int i = 0; i < total; ++i) {
        double new_val = Data_get_flat(va->data, va->start + i) - Data_get_flat(vb->data, vb->start + i);
        Data_set_flat(va->data, va->start + i, new_val);
    }
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
        int total = Vector_end(va) - va->start;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double new_val = Data_get_flat(va->data, va->start + i) * Data_get_flat(vb->data, vb->start + i);
            Data_set_flat(va->data, va->start + i, new_val);
        }
        Py_INCREF(self);
        return self;
    } else if (PyObject_IsInstance(self, (PyObject*)&VectorizeType) && (PyLong_Check(other) || PyFloat_Check(other))) {
        Vector *va = (Vector*)self;
        double scalar = PyFloat_AsDouble(other);
        int total = Vector_end(va) - va->start;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(va->data, va->start + i);
            Data_set_flat(va->data, va->start + i, val * scalar);
        }
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
        int total = Vector_end(va) - va->start;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val_b = Data_get_flat(vb->data, vb->start + i);
            if (val_b == 0.0) {
                PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
                return NULL;
            }
            double val_a = Data_get_flat(va->data, va->start + i);
            Data_set_flat(va->data, va->start + i, val_a / val_b);
        }
        Py_INCREF(self);
        return self;
    } else if (PyObject_IsInstance(self, (PyObject*)&VectorizeType) && (PyLong_Check(other) || PyFloat_Check(other))) {
        Vector *va = (Vector*)self;
        double scalar = PyFloat_AsDouble(other);
        if (scalar == 0.0) {
            PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
            return NULL;
        }
        int total = Vector_end(va) - va->start;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(va->data, va->start + i);
            Data_set_flat(va->data, va->start + i, val / scalar);
        }
        Py_INCREF(self);
        return self;
    }
    PyErr_SetString(PyExc_TypeError, "unsupported operand type(s) for /=");
    return NULL;
}

static PyObject *Vector_ipow(PyObject *self, PyObject *other, PyObject *mod) {
    // mod parameter is ignored
    (void)mod;
    if (PyObject_IsInstance(self, (PyObject*)&VectorizeType) && PyObject_IsInstance(other, (PyObject*)&VectorizeType)) {
        Vector *va = (Vector*)self;
        Vector *vb = (Vector*)other;
        if (va->dimension != vb->dimension ||
            !_shape_equal(va->shape, vb->shape, va->dimension)) {
            PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same.");
            return NULL;
        }
        int total = Vector_end(va) - va->start;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val_a = Data_get_flat(va->data, va->start + i);
            double val_b = Data_get_flat(vb->data, vb->start + i);
            Data_set_flat(va->data, va->start + i, pow(val_a, val_b));
        }
        Py_INCREF(self);
        return self;
    } else if (PyObject_IsInstance(self, (PyObject*)&VectorizeType) && (PyLong_Check(other) || PyFloat_Check(other))) {
        Vector *va = (Vector*)self;
        double scalar = PyFloat_AsDouble(other);
        if (PyErr_Occurred()) return NULL;
        int total = Vector_end(va) - va->start;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(va->data, va->start + i);
            Data_set_flat(va->data, va->start + i, pow(val, scalar));
        }
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
    int total = Vector_end(va) - va->start;
    COS_SIMD_LOOP
    for (int i = 0; i < total; ++i) {
        double val = Data_get_flat(va->data, va->start + i);
        Data_set_flat(result->data, i, -val);
    }
    return (PyObject*)result;
}

static PyObject *Vector_pos(PyObject *self) {
    Vector *va = (Vector*)self;
    Vector *result = _new_vector_like(va);
    if (!result) return NULL;
    int total = Vector_end(va) - va->start;
    COS_SIMD_LOOP
    for (int i = 0; i < total; ++i) {
        double val = Data_get_flat(va->data, va->start + i);
        Data_set_flat(result->data, i, val);
    }
    return (PyObject*)result;
}

static PyObject *Vector_abs(PyObject *self) {
    Vector *va = (Vector*)self;
    double sum_sq = 0.0;
    COS_SIMD_LOOP
    for (int i = va->start; i < Vector_end(va); ++i) {
        double val = Data_get_flat(va->data, i);
        sum_sq += val * val;
    }
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
    int remaining = self->dimension - self->p;
    if (n == remaining) {
        /* Fast scalar path: use inline index calculation */
        if (_vector_check_indices(self, args, n) < 0) return NULL;
        int ptr = _vector_calc_flat_index(self, args, n);
        return PyFloat_FromDouble(Data_get_flat(self->data, ptr));
    } else if (0 < n && n < remaining) {
        /* Subview path */
        if (_vector_check_indices(self, args, n) < 0) return NULL;
        int start_ptr = _vector_calc_flat_index(self, args, n);
        int new_p = self->p + n;
        Vector *sub = _vector_new_view(self, start_ptr, new_p);
        if (!sub) return NULL;
        return (PyObject*)sub;
    } else {
        PyErr_SetString(PyExc_IndexError, "invalid number of indices");
        return NULL;
    }
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
    int ndim = self->dimension - self->p;
    if (n == 0) {
        Py_RETURN_NONE;
    }
    if (n != ndim) {
        PyErr_Format(PyExc_IndexError, "__set_item__ expected %d indices, got %d", ndim, n);
        return NULL;
    }
    /* Bounds check and calculate flat index using inline helper */
    if (_vector_check_indices(self, indexs, n) < 0) return NULL;
    int ptr = _vector_calc_flat_index(self, indexs, n);
    double val = PyFloat_AsDouble(val_obj);
    if (val == -1.0 && PyErr_Occurred()) {
        return NULL;
    }
    Data_set_flat(self->data, ptr, val);
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
