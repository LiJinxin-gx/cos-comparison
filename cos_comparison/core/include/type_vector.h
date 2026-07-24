#ifndef VECTOR_H
#define VECTOR_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif
#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include "type_data.h"

/* Flag definitions for Vector.flags */
#define VECTOR_FLAG_VIEW     0x01   /* data is a view onto a Python object (owner != NULL) */
#define VECTOR_FLAG_OWNED    0x02   /* data is owned by this Vector (owner == NULL) */
#define VECTOR_FLAG_BUFFER   0x04   /* data came from Py_buffer (zero-copy) */

typedef struct {
    PyObject_HEAD
    Data     *data;      /* underlying flat array (shared or owned) */
    int      *shape;     /* shape of this view */
    int       dimension; /* number of dimensions */
    int       start;     /* start offset (flat index) */
    int       end;       /* end offset (exclusive, flat index) */
    int       p;         /* current depth */
    int       cache;     /* stride of current dimension (flat) */
    PyObject *owner;     /* the object that owns data (ref'd), or NULL */
    int       flags;     /* internal flags (VECTOR_FLAG_*) */
} Vector;

static PyTypeObject VectorizeType;

/* forward declarations */
static void Vector_dealloc(Vector *self);
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
static PyObject *Vector_iter(Vector *self);                       /* __iter__ */

static PyObject *Vector_cos_comparison_passive(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *Vector_cos_comparison_active(PyObject *self, PyObject *args, PyObject *kwargs);

/* Getters for Python attribute access (pure Python API compatibility) */
static PyObject* Vector_get_tensor_size(Vector *self, void *closure) {
    int ndim = self->dimension - self->p;
    PyObject *tup = PyTuple_New(ndim);
    if (!tup) return NULL;
    for (int i = 0; i < ndim; ++i) {
        PyTuple_SET_ITEM(tup, i, PyLong_FromLong(self->shape[self->p + i]));
    }
    return tup;
}

static PyObject* Vector_get_vector(Vector *self, void *closure) {
    int len = self->end - self->start;
    PyObject *list = PyList_New(len);
    if (!list) return NULL;
    double *data = (double*)self->data->data;
    for (int i = 0; i < len; ++i) {
        PyList_SET_ITEM(list, i, PyFloat_FromDouble(data[self->start + i]));
    }
    return list;
}

static PyGetSetDef Vector_getseters[] = {
    {"tensor_size", (getter)Vector_get_tensor_size, NULL,
     "Tuple representing the shape of the current tensor view.", NULL},
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
    {"__iter__", (PyCFunction)Vector_iter, METH_NOARGS,
        "Return an iterator over elements."},
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

static PyNumberMethods Vector_as_number = {
    (binaryfunc)Vector_add,          /* nb_add              (1) */
    (binaryfunc)Vector_sub,          /* nb_subtract         (2) */
    (binaryfunc)Vector_mul,          /* nb_multiply         (3) */
    0,                               /* nb_remainder        (4) */
    0,                               /* nb_divmod           (5) */
    0,                               /* nb_power            (6) */
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
    0,                               /* nb_inplace_power   (24) */
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
    int *indices = (int*)alloca(dim * sizeof(int));

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
                PyErr_SetString(PyExc_ValueError, "inconsistent tensor shape");
                return -1;
            }
            PyObject *num = PyNumber_Float(current);
            Py_DECREF(current);
            if (!num) { free(num_list); return -1; }
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
    return 0;
}

static int _infer_shape(PyObject *obj, int **shape, int *dimension) {
    int dim = 0;
    int cap = 4;
    int *sh = (int*)malloc(cap * sizeof(int));
    PyObject *cur = obj;
    Py_INCREF(cur);
    while (PySequence_Check(cur)) {
        Py_ssize_t len = PySequence_Size(cur);
        if (dim >= cap) {
            cap *= 2;
            sh = (int*)realloc(sh, cap * sizeof(int));
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
    PyObject *tensor_size_obj = Py_None;
    PyObject *end_obj = Py_None;
    PyObject *cache_obj = Py_None;
    int start = 0;
    int p = 0;
    
    static char *kwlist[] = {"vector", "tensor_size", "start", "end",
        "p", "cache", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OiOiO", kwlist,
                                     &vector, &tensor_size_obj, &start,
                                     &end_obj, &p, &cache_obj))
        return -1;

    self->flags = 0;
    
    /* Case 1: vector is another Vector -> create a sub-view */
    if (PyObject_TypeCheck(vector, &VectorizeType)) {
        Vector *src = (Vector*)vector;
        self->data = src->data;
        self->owner = src->owner ? src->owner : vector;
        Py_INCREF(self->owner);
        self->flags |= VECTOR_FLAG_VIEW;
        self->shape = (int*)malloc(src->dimension * sizeof(int));
        if (!self->shape) { PyErr_NoMemory(); return -1; }
        memcpy(self->shape, src->shape, src->dimension * sizeof(int));
        self->dimension = src->dimension;
        self->start = src->start + start;
        self->end = (end_obj == Py_None) ? src->end : src->start + (int)PyLong_AsLong(end_obj);
        self->p = src->p + p;
        if (self->p >= self->dimension) {
            self->cache = 1;
        } else if (cache_obj != Py_None && PyLong_Check(cache_obj)) {
            self->cache = (int)PyLong_AsLong(cache_obj);
        } else {
            self->cache = _multiple_chain(self->shape + self->p + 1, self->dimension - self->p - 1);
        }
        return 0;
    }
    
    /* Case 2: try to get a Py_buffer (zero-copy for array.array, bytes, etc.) */
    Py_buffer view = {0};
    if (PyObject_GetBuffer(vector, &view, PyBUF_SIMPLE) == 0) {
        int *shape = NULL;
        int dim = 0;
        if (tensor_size_obj != Py_None) {
            if (_parse_shape_tuple(tensor_size_obj, &shape, &dim) < 0) {
                PyBuffer_Release(&view);
                PyErr_SetString(PyExc_ValueError, "invalid tensor_size tuple");
                return -1;
            }
        } else {
            dim = 1;
            shape = (int*)malloc(sizeof(int));
            size_t elem_size = (view.format && strcmp(view.format, "B") == 0) ? sizeof(unsigned char) : sizeof(double);
            shape[0] = (int)(view.len / elem_size);
        }
        
        Data *data = (Data*)calloc(1, sizeof(Data));
        if (!data) { free(shape); PyBuffer_Release(&view); PyErr_NoMemory(); return -1; }
        
        data->dimension = dim;
        data->shape = shape;
        data->strides = (int*)malloc(dim * sizeof(int));
        int stride = 1;
        for (int i = dim - 1; i >= 0; --i) {
            data->strides[i] = stride;
            stride *= shape[i];
        }
        data->data = view.buf;
        data->owns_data = 0;          /* data is owned by the Python object */
        data->dtype = (view.format && strcmp(view.format, "B") == 0) ? 1 : 0;
        
        self->data = data;
        self->owner = vector;         /* keep the Python object alive */
        Py_INCREF(self->owner);
        self->flags |= VECTOR_FLAG_VIEW | VECTOR_FLAG_BUFFER;
        self->shape = shape;
        self->dimension = dim;
        self->start = start;
        if (end_obj == Py_None) {
            size_t elem_size = (data->dtype == 1) ? sizeof(unsigned char) : sizeof(double);
            self->end = (int)(view.len / elem_size);
        } else {
            self->end = (int)PyLong_AsLong(end_obj);
            if (PyErr_Occurred()) { PyBuffer_Release(&view); return -1; }
        }
        self->p = p;
        if (self->p >= self->dimension) {
            self->cache = 1;
        } else if (cache_obj != Py_None && PyLong_Check(cache_obj)) {
            self->cache = (int)PyLong_AsLong(cache_obj);
        } else {
            self->cache = (dim > self->p + 1) ? _multiple_chain(shape + self->p + 1, dim - self->p - 1) : 1;
        }
        
        PyBuffer_Release(&view);
        return 0;
    }
    
    PyErr_Clear();  /* not a buffer, fallback to copying */
    
    /* Case 3: copy data into a new double array */
    int *shape = NULL;
    int dim = 0;
    int explicit_shape = 0;

    /* Try explicit tensor_size first */
    if (tensor_size_obj != Py_None && PyTuple_Check(tensor_size_obj) && _parse_shape_tuple(tensor_size_obj, &shape, &dim) == 0) {
        explicit_shape = 1;
    } else {
        /* No explicit shape: infer from nested structure */
        if (_infer_shape(vector, &shape, &dim) < 0) {
            PyErr_SetString(PyExc_ValueError, "not a tensor");
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
    if (end_obj == Py_None) {
        self->end = Data_total(data);
    } else {
        self->end = (int)PyLong_AsLong(end_obj);
        if (PyErr_Occurred()) { Data_free(data); free(shape); return -1; }
    }
    self->p = p;
    if (self->p >= self->dimension) {
        self->cache = 1;
    } else if (cache_obj != Py_None && PyLong_Check(cache_obj)) {
        self->cache = (int)PyLong_AsLong(cache_obj);
    } else {
        self->cache = (dim > self->p + 1) ? _multiple_chain(shape + self->p + 1, dim - self->p - 1) : 1;
    }
    return 0;
}

static void Vector_dealloc(Vector *self) {
    if (self->owner) {
        Py_DECREF(self->owner);
    } else if (self->data) {
        Data_free(self->data);
    }
    if (self->shape)
        free(self->shape);
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
        int new_start = self->start + (int)start * self->cache;
        int new_end = self->start + (int)stop * self->cache;
        Vector *slice = PyObject_New(Vector, &VectorizeType);
        if (!slice) return NULL;
        slice->data = self->data;
        slice->owner = self->owner ? self->owner : (PyObject*)self;
        Py_INCREF(slice->owner);
        slice->flags = self->flags;
        slice->shape = (int*)malloc(self->dimension * sizeof(int));
        memcpy(slice->shape, self->shape, self->dimension * sizeof(int));
        slice->dimension = self->dimension;
        slice->start = new_start;
        slice->end = new_end;
        slice->p = self->p;
        slice->cache = self->cache;
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
                int new_start = cur->start + (int)start * cur->cache;
                int new_end = cur->start + (int)stop * cur->cache;
                // Create sliced view at current depth, p unchanged
                Vector *slice = PyObject_New(Vector, &VectorizeType);
                if (!slice) { Py_DECREF(cur); return NULL; }
                slice->data = cur->data;
                slice->owner = cur->owner ? cur->owner : (PyObject*)cur;
                Py_INCREF(slice->owner);
                slice->flags = cur->flags;
                slice->shape = (int*)malloc(cur->dimension * sizeof(int));
                memcpy(slice->shape, cur->shape, cur->dimension * sizeof(int));
                slice->dimension = cur->dimension;
                slice->start = new_start;
                slice->end = new_end;
                slice->p = cur->p;
                slice->cache = cur->cache;
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
                double val = Data_get_flat(cur->data, cur->start + idx * cur->cache);
                Py_DECREF(cur);
                return PyFloat_FromDouble(val);
            } else {
                /* create subview (matches __getitem__ single index behavior) */
                Vector *sub = PyObject_New(Vector, &VectorizeType);
                if (!sub) { Py_DECREF(cur); return NULL; }
                sub->data = cur->data;
                sub->owner = cur->owner ? cur->owner : (PyObject*)cur;
                Py_INCREF(sub->owner);
                sub->flags = cur->flags;
                sub->shape = (int*)malloc(cur->dimension * sizeof(int));
                memcpy(sub->shape, cur->shape, cur->dimension * sizeof(int));
                sub->dimension = cur->dimension;
                sub->start = cur->start + idx * cur->cache;
                int new_cache = cur->cache / cur->shape[cur->p];
                sub->end = sub->start + cur->cache;
                sub->p = cur->p + 1;
                sub->cache = new_cache;
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
        int new_start = self->start + idx * self->cache;
        int new_p = self->p + 1;
        int new_cache = self->cache / self->shape[self->p];
        
        Vector *result = PyObject_New(Vector, &VectorizeType);
        if (!result) return NULL;
        result->data = self->data;
        result->owner = self->owner ? self->owner : (PyObject*)self;
        Py_INCREF(result->owner);
        result->flags = self->flags;
        result->shape = (int*)malloc(self->dimension * sizeof(int));
        if (!result->shape) {
            Py_DECREF(result);
            return NULL;
        }
        memcpy(result->shape, self->shape, self->dimension * sizeof(int));
        result->dimension = self->dimension;
        result->start = new_start;
        result->end = new_start + self->cache;
        result->p = new_p;
        result->cache = new_cache;
        return (PyObject*)result;
    } else {
        return PyFloat_FromDouble(Data_get_flat(self->data, self->start + idx * self->cache));
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
            int ptr = self->start;
            int cache = self->cache;
            for (int i = 0; i < ndim; ++i) {
                PyObject *idx_obj = PyTuple_GetItem(item, i);
                int idx = (int)PyLong_AsLong(idx_obj);
                if (idx < 0 || idx >= self->shape[self->p + i]) {
                    PyErr_SetString(PyExc_IndexError, "index out of range");
                    return -1;
                }
                ptr += idx * cache;
                if (i < ndim - 1) {
                    cache /= self->shape[self->p + i];
                }
            }
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
        Data_set_flat(self->data, self->start + idx * self->cache, val);
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
            for (int i = 0; i < count; ++i) {
                Data_set_flat(self->data, self->start + (start_idx + i) * self->cache, scalar);
            }
            return 0;
        } else if (PyList_Check(value) || PyTuple_Check(value)) {
            // Sequence
            Py_ssize_t seq_len = PySequence_Size(value);
            if ((int)seq_len != count) {
                PyErr_SetString(PyExc_ValueError, "length of value does not match slice length");
                return -1;
            }
            for (int i = 0; i < count; ++i) {
                PyObject *item_val = PySequence_GetItem(value, i);
                if (!item_val) return -1;
                double d = PyFloat_AsDouble(item_val);
                Py_DECREF(item_val);
                if (PyErr_Occurred()) return -1;
                Data_set_flat(self->data, self->start + (start_idx + i) * self->cache, d);
            }
            return 0;
        } else if (PyObject_TypeCheck(value, &VectorizeType)) {
            Vector *v = (Vector*)value;
            if (v->end - v->start != count) {
                PyErr_SetString(PyExc_ValueError, "length of Vector does not match slice length");
                return -1;
            }
            for (int i = 0; i < count; ++i) {
                double d = Data_get_flat(v->data, v->start + i);
                Data_set_flat(self->data, self->start + (start_idx + i) * self->cache, d);
            }
            return 0;
        } else {
            PyErr_SetString(PyExc_TypeError, "value must be scalar, list, tuple, or Vector");
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
    Data_set_flat(self->data, self->start + idx * self->cache, d);
    return 0;
}

static PyObject *Vector_mean(Vector *self, PyObject *args) {
    int len = self->end - self->start;
    if (len == 0) Py_RETURN_NONE;
    double sum = 0.0;
    for (int i = self->start; i < self->end; ++i) {
        sum += Data_get_flat(self->data, i);
    }
    return PyFloat_FromDouble(sum / (double)len);
}

static PyObject *Vector_variance(Vector *self, PyObject *args) {
    int len = self->end - self->start;
    if (len == 0) Py_RETURN_NONE;
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = self->start; i < self->end; ++i) {
        double val = Data_get_flat(self->data, i);
        sum += val;
        sum_sq += val * val;
    }
    double mean = sum / (double)len;
    double var = sum_sq / (double)len - mean * mean;
    return PyFloat_FromDouble(var);
}

static PyObject *Vector_repr(Vector *self) {
    return PyUnicode_FromFormat(
                                "<vector_map_as_tensor: dim=%d, start=%d, end=%d, p=%d, cache=%d>",
                                self->dimension, self->start, self->end, self->p, self->cache);
}

static inline int _shape_equal(const int *a, const int *b, int dim) {
    for (int i = 0; i < dim; ++i)
        if (a[i] != b[i]) return 0;
    return 1;
}

static Vector* _new_vector_like(Vector *src) {
    Vector *result = PyObject_New(Vector, &VectorizeType);
    if (!result) return NULL;
    result->data = Data_create(src->dimension, src->shape);
    if (!result->data) { Py_DECREF(result); return NULL; }
    result->owner = NULL;
    result->flags = VECTOR_FLAG_OWNED;
    result->shape = (int*)malloc(src->dimension * sizeof(int));
    if (!result->shape) { Data_free(result->data); Py_DECREF(result); return NULL; }
    memcpy(result->shape, src->shape, src->dimension * sizeof(int));
    result->dimension = src->dimension;
    result->start = 0;
    result->end = Data_total(result->data);
    result->p = 0;
    result->cache = (src->dimension > 1) ?
    _multiple_chain(src->shape + 1, src->dimension - 1) : 1;
    return result;
}

static PyObject *Vector_add(PyObject *a, PyObject *b) {
    if (!PyObject_TypeCheck(a, &VectorizeType) || !PyObject_TypeCheck(b, &VectorizeType)) {
        PyErr_SetString(PyExc_TypeError, "operands must be vector_map_as_tensor");
        return NULL;
    }
    Vector *va = (Vector*)a;
    Vector *vb = (Vector*)b;
    if (va->dimension != vb->dimension ||
        !_shape_equal(va->shape, vb->shape, va->dimension)) {
        PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same.");
        return NULL;
    }
    Vector *result = _new_vector_like(va);
    if (!result) return NULL;
    int total = _multiple_chain(va->shape, va->dimension);
    for (int i = 0; i < total; ++i) {
        double val_a = Data_get_flat(va->data, va->start + i);
        double val_b = Data_get_flat(vb->data, vb->start + i);
        Data_set_flat(result->data, i, val_a + val_b);
    }
    return (PyObject*)result;
}

static PyObject *Vector_sub(PyObject *a, PyObject *b) {
    if (!PyObject_TypeCheck(a, &VectorizeType) || !PyObject_TypeCheck(b, &VectorizeType)) {
        PyErr_SetString(PyExc_TypeError, "operands must be vector_map_as_tensor");
        return NULL;
    }
    Vector *va = (Vector*)a;
    Vector *vb = (Vector*)b;
    if (va->dimension != vb->dimension ||
        !_shape_equal(va->shape, vb->shape, va->dimension)) {
        PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same.");
        return NULL;
    }
    Vector *result = _new_vector_like(va);
    if (!result) return NULL;
    int total = _multiple_chain(va->shape, va->dimension);
    for (int i = 0; i < total; ++i) {
        double val_a = Data_get_flat(va->data, va->start + i);
        double val_b = Data_get_flat(vb->data, vb->start + i);
        Data_set_flat(result->data, i, val_a - val_b);
    }
    return (PyObject*)result;
}

static PyObject *Vector_mul(PyObject *a, PyObject *b) {
    if (PyObject_TypeCheck(a, &VectorizeType) && PyObject_TypeCheck(b, &VectorizeType)) {
        Vector *va = (Vector*)a;
        Vector *vb = (Vector*)b;
        if (va->dimension != vb->dimension ||
            !_shape_equal(va->shape, vb->shape, va->dimension)) {
            PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same.");
            return NULL;
        }
        Vector *result = _new_vector_like(va);
        if (!result) return NULL;
        int total = _multiple_chain(va->shape, va->dimension);
        for (int i = 0; i < total; ++i) {
            double val_a = Data_get_flat(va->data, va->start + i);
            double val_b = Data_get_flat(vb->data, vb->start + i);
            Data_set_flat(result->data, i, val_a * val_b);
        }
        return (PyObject*)result;
    } else if (PyObject_TypeCheck(a, &VectorizeType) && (PyLong_Check(b) || PyFloat_Check(b))) {
        Vector *va = (Vector*)a;
        double scalar = PyFloat_AsDouble(b);
        Vector *result = _new_vector_like(va);
        if (!result) return NULL;
        int total = _multiple_chain(va->shape, va->dimension);
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(va->data, va->start + i);
            Data_set_flat(result->data, i, val * scalar);
        }
        return (PyObject*)result;
    } else if ((PyLong_Check(a) || PyFloat_Check(a)) && PyObject_TypeCheck(b, &VectorizeType)) {
        return Vector_mul(b, a);
    }
    PyErr_SetString(PyExc_TypeError, "unsupported operand type(s) for *");
    return NULL;
}

static PyObject *Vector_div(PyObject *a, PyObject *b) {
    if (PyObject_TypeCheck(a, &VectorizeType) && PyObject_TypeCheck(b, &VectorizeType)) {
        Vector *va = (Vector*)a;
        Vector *vb = (Vector*)b;
        if (va->dimension != vb->dimension ||
            !_shape_equal(va->shape, vb->shape, va->dimension)) {
            PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same.");
            return NULL;
        }
        Vector *result = _new_vector_like(va);
        if (!result) return NULL;
        int total = _multiple_chain(va->shape, va->dimension);
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
    } else if (PyObject_TypeCheck(a, &VectorizeType) && (PyLong_Check(b) || PyFloat_Check(b))) {
        Vector *va = (Vector*)a;
        double scalar = PyFloat_AsDouble(b);
        if (scalar == 0.0) {
            PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
            return NULL;
        }
        Vector *result = _new_vector_like(va);
        if (!result) return NULL;
        int total = _multiple_chain(va->shape, va->dimension);
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(va->data, va->start + i);
            Data_set_flat(result->data, i, val / scalar);
        }
        return (PyObject*)result;
    }
    PyErr_SetString(PyExc_TypeError, "unsupported operand type(s) for /");
    return NULL;
}

/* In-place operations: modify self in place, return self */
static PyObject *Vector_iadd(PyObject *self, PyObject *other) {
    if (!PyObject_TypeCheck(self, &VectorizeType) || !PyObject_TypeCheck(other, &VectorizeType)) {
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
    int total = _multiple_chain(va->shape, va->dimension);
    for (int i = 0; i < total; ++i) {
        double new_val = Data_get_flat(va->data, va->start + i) + Data_get_flat(vb->data, vb->start + i);
        Data_set_flat(va->data, va->start + i, new_val);
    }
    Py_INCREF(self);
    return self;
}

static PyObject *Vector_isub(PyObject *self, PyObject *other) {
    if (!PyObject_TypeCheck(self, &VectorizeType) || !PyObject_TypeCheck(other, &VectorizeType)) {
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
    int total = _multiple_chain(va->shape, va->dimension);
    for (int i = 0; i < total; ++i) {
        double new_val = Data_get_flat(va->data, va->start + i) - Data_get_flat(vb->data, vb->start + i);
        Data_set_flat(va->data, va->start + i, new_val);
    }
    Py_INCREF(self);
    return self;
}

static PyObject *Vector_imul(PyObject *self, PyObject *other) {
    if (PyObject_TypeCheck(self, &VectorizeType) && PyObject_TypeCheck(other, &VectorizeType)) {
        Vector *va = (Vector*)self;
        Vector *vb = (Vector*)other;
        if (va->dimension != vb->dimension ||
            !_shape_equal(va->shape, vb->shape, va->dimension)) {
            PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same.");
            return NULL;
        }
        int total = _multiple_chain(va->shape, va->dimension);
        for (int i = 0; i < total; ++i) {
            double new_val = Data_get_flat(va->data, va->start + i) * Data_get_flat(vb->data, vb->start + i);
            Data_set_flat(va->data, va->start + i, new_val);
        }
        Py_INCREF(self);
        return self;
    } else if (PyObject_TypeCheck(self, &VectorizeType) && (PyLong_Check(other) || PyFloat_Check(other))) {
        Vector *va = (Vector*)self;
        double scalar = PyFloat_AsDouble(other);
        int total = _multiple_chain(va->shape, va->dimension);
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
    if (PyObject_TypeCheck(self, &VectorizeType) && PyObject_TypeCheck(other, &VectorizeType)) {
        Vector *va = (Vector*)self;
        Vector *vb = (Vector*)other;
        if (va->dimension != vb->dimension ||
            !_shape_equal(va->shape, vb->shape, va->dimension)) {
            PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same.");
            return NULL;
        }
        int total = _multiple_chain(va->shape, va->dimension);
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
    } else if (PyObject_TypeCheck(self, &VectorizeType) && (PyLong_Check(other) || PyFloat_Check(other))) {
        Vector *va = (Vector*)self;
        double scalar = PyFloat_AsDouble(other);
        if (scalar == 0.0) {
            PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
            return NULL;
        }
        int total = _multiple_chain(va->shape, va->dimension);
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

/* Unary operators */
static PyObject *Vector_neg(PyObject *self) {
    Vector *va = (Vector*)self;
    Vector *result = _new_vector_like(va);
    if (!result) return NULL;
    int total = _multiple_chain(va->shape, va->dimension);
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
    int total = _multiple_chain(va->shape, va->dimension);
    for (int i = 0; i < total; ++i) {
        double val = Data_get_flat(va->data, va->start + i);
        Data_set_flat(result->data, i, val);
    }
    return (PyObject*)result;
}

static PyObject *Vector_abs(PyObject *self) {
    Vector *va = (Vector*)self;
    double sum_sq = 0.0;
    for (int i = va->start; i < va->end; ++i) {
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
        int ptr = self->start;
        int cache = self->cache;
        for (int i = 0; i < n; ++i) {
            PyObject *idx_obj = PyTuple_GetItem(args, i);
            int idx = (int)PyLong_AsLong(idx_obj);
            if (idx < 0 || idx >= self->shape[self->p + i]) {
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return NULL;
            }
            ptr += idx * cache;
            if (i < n - 1) {
                cache /= self->shape[self->p + i];
            }
        }
        return PyFloat_FromDouble(Data_get_flat(self->data, ptr));
    } else if (0 < n && n < remaining) {
        int start_ptr = self->start;
        int cache = self->cache;
        for (int i = 0; i < n; ++i) {
            PyObject *idx_obj = PyTuple_GetItem(args, i);
            int idx = (int)PyLong_AsLong(idx_obj);
            if (idx < 0 || idx >= self->shape[self->p + i]) {
                PyErr_SetString(PyExc_IndexError, "index out of range");
                return NULL;
            }
            start_ptr += idx * cache;
            cache /= self->shape[self->p + i];
        }
        int new_cache = cache;
        Vector *sub = PyObject_New(Vector, &VectorizeType);
        if (!sub) return NULL;
        sub->data = self->data;
        sub->owner = self->owner ? self->owner : (PyObject*)self;
        Py_INCREF(sub->owner);
        sub->flags = self->flags;
        sub->shape = (int*)malloc(self->dimension * sizeof(int));
        memcpy(sub->shape, self->shape, self->dimension * sizeof(int));
        sub->dimension = self->dimension;
        sub->start = start_ptr;
        sub->end = start_ptr + new_cache;
        sub->p = self->p + n;
        sub->cache = new_cache;
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
    int ptr = self->start;
    int cache = self->cache;
    for (int i = 0; i < ndim; ++i) {
        PyObject *idx_obj = PyTuple_GetItem(indexs, i);
        int idx = (int)PyLong_AsLong(idx_obj);
        if (idx < 0 || idx >= self->shape[self->p + i]) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return NULL;
        }
        ptr += idx * cache;
        if (i < ndim - 1) {
            cache /= self->shape[self->p + i];
        }
    }
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
    .tp_repr      = (reprfunc)Vector_repr,
    .tp_as_number = &Vector_as_number,
    .tp_as_mapping = &Vector_as_mapping,
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc       = "Maps a 1D list to a multi-dimensional tensor view (C-backed).",
    .tp_methods   = Vector_methods,
    .tp_getset    = Vector_getseters,
    .tp_init      = (initproc)Vector_init,
    .tp_new       = PyType_GenericNew,
};

#endif