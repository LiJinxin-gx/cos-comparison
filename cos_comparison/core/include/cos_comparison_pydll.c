#ifdef Py_DEBUG
#undef Py_DEBUG
#endif
#ifdef _DEBUG
#undef _DEBUG
#endif

#ifdef _MSC_VER
/* CPython extension ABI patterns under /Wall:
   - C4191: method-table casts of keyword functions to PyCFunction
     (required by the CPython METH_KEYWORDS convention)
   - C4232: address of dllimport PyType_GenericNew assigned to tp_new
     (standard CPython slot pattern)
   - C4820: padding inside the ABI-fixed C struct layouts (Data etc.)
   - C5045/C4711/C4710: /Wall performance hints (not defects)
   Keep these on for every other warning under /WX. */
#pragma warning(push)
#pragma warning(disable: 4191 4232 4820 5045 4711 4710)
#endif

#include <math.h>
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif
#include <Python.h>
#include "type.h"
#include "core.h"
#include "type_vector.h"

/* ------------------------------------------------------------------
Helper: get nested item by multi-dimensional indices (iterative)
------------------------------------------------------------------ */
static PyObject* get_nested_item(PyObject *obj, int *indices, int dim) {
    PyObject *current = obj;
    Py_INCREF(current);
    for (int i = 0; i < dim; ++i) {
        if (!PySequence_Check(current)) {
            Py_DECREF(current);
            PyErr_SetString(PyExc_TypeError, "expected sequence");
            return NULL;
        }
        PyObject *next = PySequence_GetItem(current, indices[i]);
        Py_DECREF(current);
        if (!next) return NULL;
        current = next;
    }
    return current;
}

/* ------------------------------------------------------------------
Helper: flatten nested data to a flat double array (iterative)
------------------------------------------------------------------ */
static int flatten_list(PyObject *obj, double *out, int *idx, int dim, const int *shape) {
    /* Degenerate shape (any dimension <= 0): nothing to flatten. */
    long long total = 1;
    for (int i = 0; i < dim; ++i) {
        if (shape[i] <= 0) { *idx = 0; return 0; }
        total *= (long long)shape[i];
        if (total > INT_MAX) return -1;   /* overflow guard: out is int-sized */
    }
    if (dim == 0) {
        PyObject *num = PyNumber_Float(obj);
        if (!num) return -1;
        out[*idx] = PyFloat_AsDouble(num);
        Py_DECREF(num);
        (*idx)++;
        return 0;
    }
    
    int *num_list = (int*)malloc(((size_t)(dim) + 1) * sizeof(int));
    if (!num_list) { PyErr_NoMemory(); return -1; }
    for (int i = 0; i <= dim; ++i) num_list[i] = 1;
    int flag = dim;
    int pos = 0;
    int *indices = (int*)malloc((size_t)(dim) * sizeof(int)); // Use malloc instead of alloca for portability
    if (!indices) { free(num_list); PyErr_NoMemory(); return -1; }
    
    while (flag) {
        if (flag == dim) {
            for (int i = 0; i < dim; ++i) indices[i] = num_list[i+1] - 1;
            PyObject *item = get_nested_item(obj, indices, dim);
            if (!item) { free(indices); free(num_list); return -1; }
            PyObject *num = PyNumber_Float(item);
            Py_DECREF(item);
            if (!num) { free(indices); free(num_list); return -1; }
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
    free(indices);
    free(num_list);
    return 0;
}

/* ------------------------------------------------------------------
Helper: nested list / Vector -> Data
------------------------------------------------------------------ */
static Data* pyobj_to_data(PyObject *obj) {
    if (PyObject_IsInstance(obj, (PyObject*)&VectorizeType)) {
        Vector *v = (Vector*)obj;
        int ndim = v->dimension;
        Data *data = (Data*)calloc((size_t)(1), sizeof(Data));
        if (!data) { PyErr_NoMemory(); return NULL; }
        data->dimension = ndim;
        data->shape = (int*)malloc((size_t)(ndim) * sizeof(int));
        if (!data->shape) { free(data); PyErr_NoMemory(); return NULL; }
        memcpy(data->shape, v->shape, ndim * sizeof(int));
        data->strides = (int*)malloc((size_t)(ndim) * sizeof(int));
        if (!data->strides) { free(data->shape); free(data); PyErr_NoMemory(); return NULL; }
        long long stride = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            data->strides[i] = (int)stride;
            if (data->shape[i] > 0 && stride > LLONG_MAX / (long long)data->shape[i]) {
                free(data->strides); free(data->shape); free(data);
                PyErr_SetString(PyExc_OverflowError, "tensor is too large");
                return NULL;
            }
            stride *= (long long)data->shape[i];
            if (stride > INT_MAX) {
                free(data->strides); free(data->shape); free(data);
                PyErr_SetString(PyExc_OverflowError, "tensor is too large");
                return NULL;
            }
        }
        int total = (int)stride;
        /* Always copy data to ensure contiguous, handles non-contiguous views */
        data->data = (double*)malloc((size_t)(total) * sizeof(double));
        if (!data->data) { free(data->strides); free(data->shape); free(data); PyErr_NoMemory(); return NULL; }
        data->owns_data = 1;
        /* Iterate all elements with carry method */
        int *indices = (int*)malloc((size_t)(total > 0 ? total : 1) * sizeof(int));
        if (!indices) { Data_free(data); PyErr_NoMemory(); return NULL; }
        int *idx = (int*)malloc((size_t)(ndim > 0 ? ndim : 1) * sizeof(int));
        if (!idx) { free(indices); Data_free(data); return NULL; }
        memset(idx, 0, ndim * sizeof(int));
        int pos = 0;
        if (total > 0) {
        while (1) {
            long long flat = (long long)v->start + v->offset;
            for (int i = 0; i < ndim; ++i) {
                long long term = (long long)v->strides[i];
                long long off = (long long)v->start_offset[i] +
                                (long long)idx[i] * v->step_offset[i];
                if (term != 0 && off != 0) {
                    /* guard the multiplication against long long overflow */
                    if ((off > 0 && (term > LLONG_MAX / off || term < LLONG_MIN / off)) ||
                        (off < 0 && (term > LLONG_MIN / off || term < LLONG_MAX / off))) {
                        free(indices); free(idx); Data_free(data);
                        PyErr_SetString(PyExc_OverflowError, "tensor is too large");
                        return NULL;
                    }
                }
                flat += term * off;
            }
            if (flat < 0 || flat > INT_MAX) {
                free(indices); free(idx); Data_free(data);
                PyErr_SetString(PyExc_OverflowError, "tensor is too large");
                return NULL;
            }
            indices[pos++] = (int)flat;
            int dim = ndim - 1;
            while (dim >= 0) {
                idx[dim]++;
                if (idx[dim] < v->shape[dim]) break;
                idx[dim] = 0;
                dim--;
            }
            if (dim < 0) break;
        }
        }
        free(idx);
        double *dptr = (double*)data->data;
        for (int i = 0; i < total; ++i) {
            dptr[i] = ((double*)v->data->data)[indices[i]];
        }
        free(indices);
        data->dtype = v->data->dtype;
        return data;
    }
    
    int *shape = NULL;
    int dimension = 0;
    if (infer_shape(obj, &shape, &dimension) < 0) return NULL;
    if (dimension == 0) {
        free(shape);
        PyErr_SetString(PyExc_ValueError, "not a tensor");
        return NULL;
    }
    Data *data = Data_create(dimension, shape);
    if (!data) { free(shape); PyErr_NoMemory(); return NULL; }
    int idx = 0;
    if (flatten_list(obj, data->data, &idx, dimension, shape) < 0) {
        Data_free(data);
        free(shape);
        return NULL;
    }
    free(shape);
    return data;
}

/* ------------------------------------------------------------------
Helper: Data -> vector_map_as_tensor (zero-copy)
------------------------------------------------------------------ */
static PyObject* data_to_vector(Data *data, PyTypeObject *type) {
    if (!type) type = &VectorizeType;
    if (!data) {
        PyErr_SetString(PyExc_ValueError, "cannot create vector from empty data");
        return NULL;
    }
    if (data->dimension < 0) {
        Data_free(data);
        PyErr_SetString(PyExc_ValueError, "cannot create vector from scalar data");
        return NULL;
    }
    Vector *vec = (Vector*)type->tp_alloc(type, 0);
    if (!vec) { Data_free(data); return NULL; }
    vec->data = data;
    vec->owner = NULL;
    vec->dimension = data->dimension;
    size_t dim_n = data->dimension > 0 ? (size_t)data->dimension : 1;
    vec->shape = (int*)malloc(dim_n * sizeof(int));
    if (!vec->shape) { Py_DECREF(vec); PyErr_NoMemory(); return NULL; }
    if (data->dimension > 0) {
        memcpy(vec->shape, data->shape, (size_t)data->dimension * sizeof(int));
    } else {
        vec->shape[0] = 1;
    }
    vec->strides = (int*)malloc(dim_n * sizeof(int));
    if (!vec->strides) { Py_DECREF(vec); PyErr_NoMemory(); return NULL; }
    if (data->dimension > 0) {
        vec->strides[data->dimension - 1] = 1;
        for (int i = data->dimension - 2; i >= 0; --i) {
            vec->strides[i] = vec->strides[i+1] * vec->shape[i+1];
        }
    } else {
        vec->strides[0] = 1;
    }
    vec->start = 0;
    vec->offset = 0;
    vec->start_offset = (int*)malloc(dim_n * sizeof(int));
    vec->step_offset = (int*)malloc(dim_n * sizeof(int));
    if (!vec->start_offset || !vec->step_offset) {
        Py_DECREF(vec);
        PyErr_NoMemory();
        return NULL;
    }
    for (int i = 0; i < vec->dimension; ++i) {
        vec->start_offset[i] = 0;
        vec->step_offset[i] = 1;
    }
    vec->buf = NULL;
    vec->flags = VECTOR_FLAG_OWNED;
    return (PyObject*)vec;
}

/* ------------------------------------------------------------------
Helper: parse int tuple/list
------------------------------------------------------------------ */
static int parse_int_seq(PyObject *obj, int **out, int *count) {
    if (!PySequence_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "expected sequence");
        return -1;
    }
    Py_ssize_t n = PySequence_Size(obj);
    if (n < 0) return -1;
    if (n > INT_MAX) {
        PyErr_SetString(PyExc_OverflowError,
            "sequence length does not fit an int");
        return -1;
    }
    int *arr = (int*)malloc((size_t)(n) * sizeof(int));
    if (!arr) { PyErr_NoMemory(); return -1; }
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = PySequence_GetItem(obj, i);
        if (!item) { free(arr); return -1; }
        PyObject *num = PyNumber_Index(item);
        Py_DECREF(item);
        if (!num) { free(arr); return -1; }
        int overflow = 0;
        long v = PyLong_AsLongAndOverflow(num, &overflow);
        Py_DECREF(num);
        if (overflow || v > INT_MAX || v < INT_MIN) {
            free(arr);
            PyErr_SetString(PyExc_OverflowError, "value out of range");
            return -1;
        }
        if (v == -1 && PyErr_Occurred()) { free(arr); return -1; }
        arr[i] = (int)v;
    }
    *out = arr;
    *count = (int)n;
    return 0;
}

/* Parse an optional integer sequence into *arr with exactly dim elements.
   obj == NULL fills every element with dflt; shorter sequences are padded
   with dflt, longer ones truncated (safe against out-of-bounds reads).
   Returns 0 on success, or -1 with *arr set to NULL and an exception set. */
static int parse_opt_int_seq(PyObject *obj, int **arr, int dim, int dflt) {
    if (obj == NULL) {
        *arr = (int*)malloc((size_t)(dim > 0 ? dim : 1) * sizeof(int));
        if (!*arr) { PyErr_NoMemory(); return -1; }
        for (int i = 0; i < dim; ++i) (*arr)[i] = dflt;
        return 0;
    }
    int cnt = 0;
    int *tmp = NULL;
    if (parse_int_seq(obj, &tmp, &cnt) < 0) { *arr = NULL; return -1; }
    *arr = (int*)malloc((size_t)(dim > 0 ? dim : 1) * sizeof(int));
    if (!*arr) { free(tmp); *arr = NULL; PyErr_NoMemory(); return -1; }
    for (int i = 0; i < dim; ++i) (*arr)[i] = (i < cnt) ? tmp[i] : dflt;
    free(tmp);
    return 0;
}

/* ------------------------------------------------------------------
Utility functions
------------------------------------------------------------------ */
static PyObject* py_multiple_chain(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *iterable, *base_obj = NULL;
    static char *kwlist[] = {"iterable", "base", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &iterable, &base_obj))
        return NULL;
    
    PyObject *result;
    if (base_obj) {
        result = base_obj;
        Py_INCREF(result);
    } else {
        result = PyLong_FromLong(1);
    }
    
    PyObject *iterator = PyObject_GetIter(iterable);
    if (!iterator) { Py_DECREF(result); return NULL; }
    
    PyObject *item;
    while ((item = PyIter_Next(iterator))) {
        PyObject *new_result = PyNumber_Multiply(result, item);
        Py_DECREF(item);
        Py_DECREF(result);
        if (!new_result) { Py_DECREF(iterator); return NULL; }
        result = new_result;
    }
    Py_DECREF(iterator);
    
    if (PyErr_Occurred()) { Py_DECREF(result); return NULL; }
    return result;
}

static PyObject* py_add_chain(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *iterable, *base_obj = NULL;
    static char *kwlist[] = {"iterable", "base", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &iterable, &base_obj))
        return NULL;
    
    PyObject *result;
    if (base_obj) {
        result = base_obj;
        Py_INCREF(result);
    } else {
        result = PyLong_FromLong(0);
    }
    
    PyObject *iterator = PyObject_GetIter(iterable);
    if (!iterator) { Py_DECREF(result); return NULL; }
    
    PyObject *item;
    while ((item = PyIter_Next(iterator))) {
        PyObject *new_result = PyNumber_Add(result, item);
        Py_DECREF(item);
        Py_DECREF(result);
        if (!new_result) { Py_DECREF(iterator); return NULL; }
        result = new_result;
    }
    Py_DECREF(iterator);
    
    if (PyErr_Occurred()) { Py_DECREF(result); return NULL; }
    return result;
}

static PyObject* py_create_void_list(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *length_list_obj = NULL, *default_obj = Py_None;
    static char *kwlist[] = {"length_list", "default", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO", kwlist, &length_list_obj, &default_obj))
        return NULL;
    int *shape = NULL; int dim = 0;
    if (length_list_obj) {
        if (parse_int_seq(length_list_obj, &shape, &dim) < 0) return NULL;
    } else {
        shape = (int*)malloc(sizeof(int));
    if (!shape) { PyErr_NoMemory(); return NULL; }
    shape[0] = 1; dim = 1;
    }
    Data *data = Data_create(dim, shape);
    if (!data) { free(shape); PyErr_NoMemory(); return NULL; }
    double fill_val = 0.0;
    if (default_obj != Py_None) {
        fill_val = PyFloat_AsDouble(default_obj);
        if (PyErr_Occurred()) { Data_free(data); free(shape); return NULL; }
    }
    int total = Data_total(data);
    COS_SIMD_LOOP
    for (int i = 0; i < total; ++i) Data_set_flat(data, i, fill_val);
    free(shape);
    return data_to_vector(data, NULL);
}

/* Helper: create independent Vector from Data (deep copy) */
static PyObject* data_to_independent_vector(Data *data, int start) {
    if (!data || data->dimension < 0) {
        Data_free(data);
        PyErr_SetString(PyExc_ValueError, "cannot create vector from empty or scalar data");
        return NULL;
    }
    Vector *vec = (Vector*)VectorizeType.tp_alloc(&VectorizeType, 0);
    if (!vec) { Data_free(data); return NULL; }
    vec->dimension = data->dimension;
    size_t dim_n = data->dimension > 0 ? (size_t)data->dimension : 1;
    vec->shape = (int*)malloc(dim_n * sizeof(int));
    if (!vec->shape) { Py_DECREF(vec); Data_free(data); PyErr_NoMemory(); return NULL; }
    if (data->dimension > 0) {
        memcpy(vec->shape, data->shape, (size_t)data->dimension * sizeof(int));
    } else {
        vec->shape[0] = 1;
    }
    vec->strides = (int*)malloc(dim_n * sizeof(int));
    if (!vec->strides) { Py_DECREF(vec); Data_free(data); PyErr_NoMemory(); return NULL; }
    if (data->dimension > 0) {
        vec->strides[data->dimension - 1] = 1;
        for (int i = data->dimension - 2; i >= 0; --i) {
            vec->strides[i] = vec->strides[i+1] * vec->shape[i+1];
        }
    } else {
        vec->strides[0] = 1;
    }
    int total = Data_total(data);
    vec->data = Data_create(data->dimension, data->shape);
    if (!vec->data) { Py_DECREF(vec); Data_free(data); PyErr_NoMemory(); return NULL; }
    memcpy(vec->data->data, data->data, total * sizeof(double));
    vec->owner = NULL;
    vec->start = start;
    vec->offset = 0;
    vec->start_offset = (int*)malloc(dim_n * sizeof(int));
    vec->step_offset = (int*)malloc(dim_n * sizeof(int));
    if (!vec->start_offset || !vec->step_offset) {
        Py_DECREF(vec); Data_free(data);
        PyErr_NoMemory();
        return NULL;
    }
    for (int i = 0; i < data->dimension; ++i) {
        vec->start_offset[i] = 0;
        vec->step_offset[i] = 1;
    }
    vec->buf = NULL;
    Data_free(data);
    vec->flags = VECTOR_FLAG_OWNED;
    return (PyObject*)vec;
}

static PyObject* py_infer_shape(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *data_obj;
    if (!PyArg_ParseTuple(args, "O", &data_obj))
        return NULL;
    
    int *shape = NULL;
    int dim = 0;
    
    if (infer_shape(data_obj, &shape, &dim) < 0) {
        PyErr_Clear();
        Py_RETURN_NONE;
    }

    /* Match pure Python: return None when no dimensions found (scalar/None) */
    if (dim == 0) {
        free(shape);
        Py_RETURN_NONE;
    }

    PyObject *tup = PyTuple_New(dim);
    if (!tup) { free(shape); return NULL; }
    for (int i = 0; i < dim; ++i) {
        PyTuple_SET_ITEM(tup, i, PyLong_FromLong(shape[i]));
    }
    free(shape);
    return tup;
}

static PyObject* py_load_as_default_data(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *data_obj;
    PyObject *start_obj = Py_None;
    PyObject *shape_obj = Py_None;
    PyObject *step_obj = Py_None;
    static char *kwlist[] = {"data", "start", "shape", "step", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOO", kwlist,
                                     &data_obj, &start_obj, &shape_obj, &step_obj))
        return NULL;

    /* Fast path: input is already our tensor type - use native slicing */
    if (PyObject_IsInstance(data_obj, (PyObject*)&VectorizeType)) {
        Vector *src = (Vector*)data_obj;
        int dim = src->dimension;
        
        /* Build slice tuple */
        PyObject *slices = PyTuple_New(dim);
        if (!slices) return NULL;
        
        for (int i = 0; i < dim; ++i) {
            PyObject *s = NULL, *e = NULL, *st = NULL;
            
            if (start_obj != Py_None && PySequence_Check(start_obj)) {
                s = PySequence_GetItem(start_obj, i);
                if (!s) { Py_DECREF(slices); return NULL; }
            }
            if (start_obj != Py_None && shape_obj != Py_None && 
                PySequence_Check(start_obj) && PySequence_Check(shape_obj)) {
                PyObject *s_item = PySequence_GetItem(start_obj, i);
                if (!s_item) { Py_DECREF(slices); return NULL; }
                PyObject *sh_item = PySequence_GetItem(shape_obj, i);
                if (!sh_item) {
                    Py_DECREF(s_item); Py_DECREF(slices); return NULL;
                }
                e = PyNumber_Add(s_item, sh_item);
                Py_DECREF(s_item);
                Py_DECREF(sh_item);
                if (!e) { Py_DECREF(slices); return NULL; }
            }
            if (step_obj != Py_None && PySequence_Check(step_obj)) {
                st = PySequence_GetItem(step_obj, i);
                if (!st) { Py_DECREF(slices); return NULL; }
            }
            
            PyObject *sl = PySlice_New(s, e, st);
            Py_XDECREF(s);
            Py_XDECREF(e);
            Py_XDECREF(st);
            if (!sl) { Py_DECREF(slices); return NULL; }
            PyTuple_SET_ITEM(slices, i, sl);
        }
        
        /* Apply slicing */
        PyObject *result = PyObject_GetItem(data_obj, slices);
        Py_DECREF(slices);
        if (!result) return NULL;
        
        /* If result is not a vector (shouldn't happen), return as-is */
        if (!PyObject_IsInstance(result, (PyObject*)&VectorizeType)) {
            return result;
        }
        Vector *res_vec = (Vector*)result;
        
        /* Create a new contiguous vector by copying all elements */
        long long total_ll = 1;
        for (int i = 0; i < res_vec->dimension; ++i) total_ll *= res_vec->shape[i];
        /* Match pure Python: a zero-size slice yields an empty tensor
           with the shape preserved (no elements to copy). */
        if (total_ll == 0) {
            PyObject *shape_tuple = PyTuple_New(res_vec->dimension);
            if (!shape_tuple) { Py_DECREF(result); return NULL; }
            for (int i = 0; i < res_vec->dimension; ++i) {
                PyTuple_SET_ITEM(shape_tuple, i, PyLong_FromLong(res_vec->shape[i]));
            }
            Py_DECREF(result);
            PyObject *vec_args = PyTuple_New(0);
            PyObject *vec_kwargs = PyDict_New();
            if (!vec_args || !vec_kwargs) {
                Py_XDECREF(vec_args); Py_XDECREF(vec_kwargs);
                Py_DECREF(shape_tuple);
                return NULL;
            }
            PyObject *empty_list = PyList_New(0);
            if (!empty_list) {
                Py_DECREF(vec_args); Py_DECREF(vec_kwargs);
                Py_DECREF(shape_tuple);
                return NULL;
            }
            PyDict_SetItemString(vec_kwargs, "vector", empty_list);
            PyDict_SetItemString(vec_kwargs, "shape", shape_tuple);
            PyDict_SetItemString(vec_kwargs, "start", PyLong_FromLong(0));
            PyObject *new_vec = PyObject_Call((PyObject*)&VectorizeType,
                                              vec_args, vec_kwargs);
            Py_DECREF(vec_args);
            Py_DECREF(vec_kwargs);
            Py_DECREF(empty_list);
            Py_DECREF(shape_tuple);
            return new_vec;
        }
        if (total_ll > (long long)INT_MAX) {
            Py_DECREF(result);
            PyErr_SetString(PyExc_OverflowError, "tensor is too large");
            return NULL;
        }
        int total = (int)total_ll;
        int *flat_indices = (int*)malloc((size_t)(total) * sizeof(int));
        if (!flat_indices) { Py_DECREF(result); PyErr_NoMemory(); return NULL; }
        if (vector_get_flat_indices(res_vec, flat_indices, total) < 0) {
            free(flat_indices); Py_DECREF(result);
            if (!PyErr_Occurred()) PyErr_NoMemory();
            return NULL;
        }
        
        /* Build flat list of values */
        PyObject *new_list = PyList_New(total);
        if (!new_list) { free(flat_indices); Py_DECREF(result); return NULL; }
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(res_vec->data, flat_indices[i]);
            PyList_SET_ITEM(new_list, i, PyFloat_FromDouble(val));
        }
        free(flat_indices);

        /* Build shape tuple (before releasing res_vec) */
        PyObject *shape_tuple = PyTuple_New(res_vec->dimension);
        if (!shape_tuple) { Py_DECREF(result); Py_DECREF(new_list); return NULL; }
        for (int i = 0; i < res_vec->dimension; ++i) {
            PyTuple_SET_ITEM(shape_tuple, i, PyLong_FromLong(res_vec->shape[i]));
        }
        Py_DECREF(result);
        
        /* Create new vector with keyword arguments */
        PyObject *vec_args = PyTuple_New(0);
        PyObject *vec_kwargs = PyDict_New();
        if (!vec_args || !vec_kwargs) {
            Py_XDECREF(vec_args); Py_XDECREF(vec_kwargs);
            Py_DECREF(shape_tuple); Py_DECREF(new_list);
            return NULL;
        }
        PyDict_SetItemString(vec_kwargs, "vector", new_list);
        PyDict_SetItemString(vec_kwargs, "shape", shape_tuple);
        PyDict_SetItemString(vec_kwargs, "start", PyLong_FromLong(0));
        
        PyObject *new_vec = PyObject_Call((PyObject*)&VectorizeType, vec_args, vec_kwargs);
        Py_DECREF(vec_args);
        Py_DECREF(vec_kwargs);
        Py_DECREF(new_list);
        Py_DECREF(shape_tuple);
        return new_vec;
    }

    int *full_shape = NULL;
    int dimension = 0;

    /* Step 1: Infer full shape of input data */
    if (infer_shape(data_obj, &full_shape, &dimension) < 0)
        return NULL;
    if (dimension == 0) {
        free(full_shape);
        PyErr_SetString(PyExc_ValueError, "cannot infer shape from scalar or empty data");
        return NULL;
    }

    /* Step 2: Process step parameter */
    int *step = (int*)malloc((size_t)(dimension) * sizeof(int));
    if (!step) { free(full_shape); PyErr_NoMemory(); return NULL; }
    
    if (step_obj == Py_None) {
        for (int i = 0; i < dimension; ++i) step[i] = 1;
    } else {
        if (!PySequence_Check(step_obj)) {
            free(step); free(full_shape);
            PyErr_SetString(PyExc_TypeError, "step must be a sequence of integers");
            return NULL;
        }
        Py_ssize_t n = PySequence_Size(step_obj);
        if ((int)n != dimension) {
            free(step); free(full_shape);
            PyErr_Format(PyExc_ValueError,
                         "step length %zd does not match data dimension %d",
                         n, dimension);
            return NULL;
        }
        for (int i = 0; i < dimension; ++i) {
            PyObject *item = PySequence_GetItem(step_obj, i);
            if (!item) { free(step); free(full_shape); return NULL; }
            PyObject *num = PyNumber_Index(item); Py_DECREF(item);
            if (!num) { free(step); free(full_shape); return NULL; }
            int overflow = 0;
            long sv = PyLong_AsLongAndOverflow(num, &overflow); Py_DECREF(num);
            if (overflow || sv > INT_MAX || sv < INT_MIN) {
                free(step); free(full_shape);
                PyErr_Format(PyExc_OverflowError, "step[%d] value out of range", i);
                return NULL;
            }
            if (sv == -1 && PyErr_Occurred()) { free(step); free(full_shape); return NULL; }
            step[i] = (int)sv;
            if (step[i] <= 0) {
                int step_val = step[i];
                free(step); free(full_shape);
                PyErr_Format(PyExc_ValueError, "step[%d] = %d must be positive", i, step_val);
                return NULL;
            }
        }
    }

    /* Step 3: Process shape parameter */
    int *shape = NULL;
    if (shape_obj == Py_None) {
        /* Default: full shape adjusted for step */
        shape = (int*)malloc((size_t)(dimension) * sizeof(int));
        if (!shape) { free(step); free(full_shape); PyErr_NoMemory(); return NULL; }
        for (int i = 0; i < dimension; ++i) {
            shape[i] = (full_shape[i] + step[i] - 1) / step[i];
        }
    } else {
        if (!PySequence_Check(shape_obj)) {
            free(step); free(full_shape);
            PyErr_SetString(PyExc_TypeError, "shape must be a sequence of integers");
            return NULL;
        }
        Py_ssize_t n = PySequence_Size(shape_obj);
        if (n == 0) {
            free(step); free(full_shape);
            PyErr_SetString(PyExc_ValueError, "shape cannot be empty");
            return NULL;
        }
        if ((int)n != dimension) {
            free(step); free(full_shape);
            PyErr_Format(PyExc_ValueError,
                         "shape length %zd does not match data dimension %d",
                         n, dimension);
            return NULL;
        }
        shape = (int*)malloc((size_t)(dimension) * sizeof(int));
        if (!shape) { free(step); free(full_shape); PyErr_NoMemory(); return NULL; }
        for (int i = 0; i < dimension; ++i) {
            PyObject *item = PySequence_GetItem(shape_obj, i);
            if (!item) { free(shape); free(step); free(full_shape); return NULL; }
            PyObject *num = PyNumber_Index(item); Py_DECREF(item);
            if (!num) { free(shape); free(step); free(full_shape); return NULL; }
            int overflow = 0;
            long sv = PyLong_AsLongAndOverflow(num, &overflow); Py_DECREF(num);
            if (overflow || sv > INT_MAX || sv < INT_MIN) {
                free(shape); free(step); free(full_shape);
                PyErr_Format(PyExc_OverflowError, "shape[%d] value out of range", i);
                return NULL;
            }
            if (sv == -1 && PyErr_Occurred()) { free(shape); free(step); free(full_shape); return NULL; }
            shape[i] = (int)sv;
            if (shape[i] < 0) {
                free(shape); free(step); free(full_shape);
                PyErr_Format(PyExc_ValueError, "shape[%d] cannot be negative", i);
                return NULL;
            }
        }
    }

    /* Step 4: Process start parameter */
    int *start = NULL;
    if (start_obj == Py_None) {
        /* Default: all zeros */
        start = (int*)calloc((size_t)(dimension), sizeof(int));
        if (!start) { free(shape); free(step); free(full_shape); PyErr_NoMemory(); return NULL; }
    } else {
        if (!PySequence_Check(start_obj)) {
            free(shape); free(step); free(full_shape);
            PyErr_SetString(PyExc_TypeError, "start must be a sequence of integers");
            return NULL;
        }
        Py_ssize_t n = PySequence_Size(start_obj);
        if ((int)n != dimension) {
            free(shape); free(step); free(full_shape);
            PyErr_Format(PyExc_ValueError,
                         "start length %zd does not match data dimension %d",
                         n, dimension);
            return NULL;
        }
        start = (int*)malloc((size_t)(dimension) * sizeof(int));
        if (!start) { free(shape); free(step); free(full_shape); PyErr_NoMemory(); return NULL; }
        for (int i = 0; i < dimension; ++i) {
            PyObject *item = PySequence_GetItem(start_obj, i);
            if (!item) { free(start); free(shape); free(step); free(full_shape); return NULL; }
            PyObject *num = PyNumber_Index(item); Py_DECREF(item);
            if (!num) { free(start); free(shape); free(step); free(full_shape); return NULL; }
            int overflow = 0;
            long sv = PyLong_AsLongAndOverflow(num, &overflow); Py_DECREF(num);
            if (overflow || sv > INT_MAX || sv < INT_MIN) {
                free(start); free(shape); free(step); free(full_shape);
                PyErr_Format(PyExc_OverflowError, "start[%d] value out of range", i);
                return NULL;
            }
            if (sv == -1 && PyErr_Occurred()) { free(start); free(shape); free(step); free(full_shape); return NULL; }
            start[i] = (int)sv;
            if (start[i] < 0) {
                free(start); free(shape); free(step); free(full_shape);
                PyErr_Format(PyExc_ValueError, "start[%d] cannot be negative", i);
                return NULL;
            }
            /* Bounds check with step (long long: guard int overflow) */
            long long end_val = (long long)start[i] + ((long long)shape[i] - 1) * step[i];
            if (shape[i] > 0 && end_val >= (long long)full_shape[i]) {
                free(start); free(shape); free(step); free(full_shape);
                PyErr_Format(PyExc_ValueError,
                             "start[%d] + (shape[%d]-1)*step[%d] = %lld out of bounds for dimension size %d",
                             i, i, i, end_val, full_shape[i]);
                return NULL;
            }
        }
    }

    /* Step 5: Flatten full input data into temporary Data */
    Data *full_data = pyobj_to_data(data_obj);
    if (!full_data) {
        free(start); free(shape); free(step); free(full_shape);
        return NULL;
    }

    /* Step 6: Create result Data with requested shape */
    Data *result_data = Data_create(dimension, shape);
    if (!result_data) {
        Data_free(full_data);
        free(start); free(shape); free(step); free(full_shape);
        PyErr_NoMemory();
        return NULL;
    }

    /* Step 7: Compute strides for full data (row-major) */
    int *full_strides = (int*)malloc((size_t)(dimension) * sizeof(int));
    if (!full_strides) {
        Data_free(result_data); Data_free(full_data);
        free(start); free(shape); free(step); free(full_shape);
        PyErr_NoMemory();
        return NULL;
    }
    int stride_val = 1;
    for (int i = dimension - 1; i >= 0; --i) {
        full_strides[i] = stride_val;
        stride_val *= full_shape[i];
    }

    /* Step 8: Copy sub-region using carry-iteration mechanism with step support */
    long long total_ll = 1;
    for (int i = 0; i < dimension; ++i) total_ll *= shape[i];

    /* Match pure Python: when the result shape has zero elements, the
       carry loop still attempts one access into the source, which raises
       IndexError on an empty container. */
    if (total_ll == 0) {
        free(full_strides);
        Data_free(result_data); Data_free(full_data);
        free(start); free(shape); free(step); free(full_shape);
        PyErr_SetString(PyExc_IndexError, "list index out of range");
        return NULL;
    }
    if (total_ll > (long long)INT_MAX) {
        free(full_strides);
        Data_free(result_data); Data_free(full_data);
        free(start); free(shape); free(step); free(full_shape);
        PyErr_SetString(PyExc_OverflowError, "tensor is too large");
        return NULL;
    }
    int *num_list = (int*)malloc(((size_t)(dimension) + 1) * sizeof(int));
    if (!num_list) {
        free(full_strides);
        Data_free(result_data); Data_free(full_data);
        free(start); free(shape); free(step); free(full_shape);
        PyErr_NoMemory();
        return NULL;
    }
    for (int i = 0; i <= dimension; ++i) num_list[i] = 0;

    int pos = 0;
    int i = dimension - 1;

    /* Iterate through result indices */
    while (1) {
        /* Calculate 1D offset in full data using start + step */
        int offset = 0;
        for (int d = 0; d < dimension; ++d) {
            offset += (start[d] + num_list[d] * step[d]) * full_strides[d];
        }
        ((double*)result_data->data)[pos] = ((double*)full_data->data)[offset];
        pos++;

        /* Increment indices with carry */
        i = dimension - 1;
        while (i >= 0) {
            num_list[i]++;
            if (num_list[i] < shape[i]) {
                break;
            }
            num_list[i] = 0;
            i--;
        }
        if (i < 0) break;
    }

    /* Step 9: Cleanup temporary resources */
    free(num_list);
    free(full_strides);
    Data_free(full_data);
    free(start);
    free(shape);
    free(step);
    free(full_shape);

    /* Step 10: Return as independent vector with start=0 */
    return data_to_independent_vector(result_data, 0);
}

/* ------------------------------------------------------------------
load_data: copy a sub-region from source to target with independent
start/step for each side.  Tries PyBuffer fast path first, falls back
to get_item/set_item on any failure.  Returns the number of elements
actually copied.  Iterative.
------------------------------------------------------------------ */
static PyObject* py_get_item(PyObject *self, PyObject *args);
static PyObject* py_set_item(PyObject *self, PyObject *args);

static PyObject* py_load_data(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *source, *target;
    PyObject *source_start_obj = Py_None;
    PyObject *source_step_obj  = Py_None;
    PyObject *shape_obj        = Py_None;
    PyObject *target_start_obj = Py_None;
    PyObject *target_step_obj  = Py_None;

    static char *kwlist[] = {"source", "target", "source_start", "source_step",
                             "shape", "target_start", "target_step", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OOOOO", kwlist,
            &source, &target, &source_start_obj, &source_step_obj,
            &shape_obj, &target_start_obj, &target_step_obj))
        return NULL;

    /* ---- infer shapes (with BufferError fallback inside infer_shape) ---- */
    int *src_shape = NULL, *tgt_shape = NULL;
    int src_dim = 0, tgt_dim = 0;
    if (infer_shape(source, &src_shape, &src_dim) < 0) return NULL;
    if (infer_shape(target, &tgt_shape, &tgt_dim) < 0) {
        free(src_shape);
        return NULL;
    }
    if (src_dim != tgt_dim) {
        free(src_shape); free(tgt_shape);
        PyErr_Format(PyExc_ValueError,
            "source and target must have the same number of dimensions (got %d and %d)",
            src_dim, tgt_dim);
        return NULL;
    }
    int dim = src_dim;
    if (dim == 0) {
        free(src_shape); free(tgt_shape);
        return PyLong_FromLong(0);
    }

    /* ---- parse integer-sequence parameters (temp-pointer pattern to
           avoid leaking the pre-allocated arrays; parse_int_seq mallocs) ---- */
    int *src_step = NULL, *tgt_step = NULL;
    int *copy_shape = NULL, *src_start = NULL, *tgt_start = NULL;
    int cnt = 0;

    /* source_step: default all 1 */
    if (source_step_obj == Py_None) {
        src_step = (int*)malloc((size_t)dim * sizeof(int));
        if (!src_step) goto oom;
        for (int i = 0; i < dim; ++i) src_step[i] = 1;
    } else {
        if (parse_int_seq(source_step_obj, &src_step, &cnt) < 0 || cnt != dim) {
            if (!PyErr_Occurred())
                PyErr_Format(PyExc_ValueError, "source_step must have %d elements", dim);
            goto fail;
        }
    }
    /* target_step: default all 1 */
    if (target_step_obj == Py_None) {
        tgt_step = (int*)malloc((size_t)dim * sizeof(int));
        if (!tgt_step) goto oom;
        for (int i = 0; i < dim; ++i) tgt_step[i] = 1;
    } else {
        if (parse_int_seq(target_step_obj, &tgt_step, &cnt) < 0 || cnt != dim) {
            if (!PyErr_Occurred())
                PyErr_Format(PyExc_ValueError, "target_step must have %d elements", dim);
            goto fail;
        }
    }
    for (int i = 0; i < dim; ++i) {
        if (src_step[i] < 1 || tgt_step[i] < 1) {
            PyErr_SetString(PyExc_ValueError, "step values must be positive integers");
            goto fail;
        }
    }

    /* shape: default full source extent with source_step */
    if (shape_obj == Py_None) {
        copy_shape = (int*)malloc((size_t)dim * sizeof(int));
        if (!copy_shape) goto oom;
        for (int i = 0; i < dim; ++i)
            copy_shape[i] = (src_shape[i] + src_step[i] - 1) / src_step[i];
    } else {
        if (parse_int_seq(shape_obj, &copy_shape, &cnt) < 0 || cnt != dim) {
            if (!PyErr_Occurred())
                PyErr_Format(PyExc_ValueError, "shape must have %d elements", dim);
            goto fail;
        }
    }
    for (int i = 0; i < dim; ++i) {
        if (copy_shape[i] < 0) {
            PyErr_SetString(PyExc_ValueError, "shape values must be non-negative");
            goto fail;
        }
    }

    /* source_start: default all 0 */
    if (source_start_obj == Py_None) {
        src_start = (int*)malloc((size_t)dim * sizeof(int));
        if (!src_start) goto oom;
        for (int i = 0; i < dim; ++i) src_start[i] = 0;
    } else {
        if (parse_int_seq(source_start_obj, &src_start, &cnt) < 0 || cnt != dim) {
            if (!PyErr_Occurred())
                PyErr_Format(PyExc_ValueError, "source_start must have %d elements", dim);
            goto fail;
        }
    }
    /* target_start: default all 0 */
    if (target_start_obj == Py_None) {
        tgt_start = (int*)malloc((size_t)dim * sizeof(int));
        if (!tgt_start) goto oom;
        for (int i = 0; i < dim; ++i) tgt_start[i] = 0;
    } else {
        if (parse_int_seq(target_start_obj, &tgt_start, &cnt) < 0 || cnt != dim) {
            if (!PyErr_Occurred())
                PyErr_Format(PyExc_ValueError, "target_start must have %d elements", dim);
            goto fail;
        }
    }
    for (int i = 0; i < dim; ++i) {
        if (src_start[i] < 0 || tgt_start[i] < 0) {
            PyErr_SetString(PyExc_ValueError, "start values must be non-negative");
            goto fail;
        }
    }

    /* ---- compute effective shape (clamp to available extents) ---- */
    long long total = 1;
    for (int i = 0; i < dim; ++i) {
        int avail_src = (src_shape[i] - src_start[i] + src_step[i] - 1) / src_step[i];
        int avail_tgt = (tgt_shape[i] - tgt_start[i] + tgt_step[i] - 1) / tgt_step[i];
        if (avail_src < 0) avail_src = 0;
        if (avail_tgt < 0) avail_tgt = 0;
        int eff = copy_shape[i];
        if (eff > avail_src) eff = avail_src;
        if (eff > avail_tgt) eff = avail_tgt;
        copy_shape[i] = eff;
        total *= (long long)eff;
    }

    free(src_shape); src_shape = NULL;
    free(tgt_shape); tgt_shape = NULL;

    if (total == 0) {
        free(src_step); free(tgt_step); free(copy_shape);
        free(src_start); free(tgt_start);
        return PyLong_FromLong(0);
    }

    /* ---- build index tuples (reused across iterations) ---- */
    PyObject *src_idx = PyTuple_New((Py_ssize_t)dim);
    PyObject *tgt_idx = PyTuple_New((Py_ssize_t)dim);
    if (!src_idx || !tgt_idx) {
        Py_XDECREF(src_idx); Py_XDECREF(tgt_idx);
        free(src_step); free(tgt_step); free(copy_shape);
        free(src_start); free(tgt_start);
        return NULL;
    }
    /* initialise with start positions (also used for the probe below) */
    for (int i = 0; i < dim; ++i) {
        PyTuple_SET_ITEM(src_idx, i, PyLong_FromLong((long)src_start[i]));
        PyTuple_SET_ITEM(tgt_idx, i, PyLong_FromLong((long)tgt_start[i]));
    }

    /* ---- attempt buffer fast path via memoryview objects ---- */
    PyObject *src_mv = NULL, *tgt_mv = NULL;
    int use_buffer = 0;

    src_mv = PyMemoryView_FromObject(source);
    if (!src_mv) {
        PyErr_Clear();
    } else if (PyMemoryView_GET_BUFFER(src_mv)->ndim != dim) {
        Py_DECREF(src_mv); src_mv = NULL;
    } else {
        tgt_mv = PyMemoryView_FromObject(target);
        if (!tgt_mv) {
            PyErr_Clear();
        } else if (PyMemoryView_GET_BUFFER(tgt_mv)->ndim != dim ||
                   PyMemoryView_GET_BUFFER(tgt_mv)->readonly) {
            Py_DECREF(tgt_mv); tgt_mv = NULL;
        } else {
            /* probe: read from source[start], write to target[start], then
               re-export target buffer and read back to verify the write is
               persistent (some objects return a fresh snapshot each export) */
            PyObject *probe_val = PyObject_GetItem(source, src_idx);
            if (probe_val) {
                if (PyObject_SetItem(tgt_mv, tgt_idx, probe_val) == 0) {
                    PyObject *verify_mv = PyMemoryView_FromObject(target);
                    if (verify_mv) {
                        PyObject *read_back = PyObject_GetItem(verify_mv, tgt_idx);
                        if (read_back) {
                            int eq = PyObject_RichCompareBool(read_back, probe_val, Py_EQ);
                            Py_DECREF(read_back);
                            if (eq == 1) use_buffer = 1;
                            else if (eq < 0) PyErr_Clear();
                        } else {
                            PyErr_Clear();
                        }
                        Py_DECREF(verify_mv);
                    } else {
                        PyErr_Clear();
                    }
                } else {
                    PyErr_Clear();
                }
                Py_DECREF(probe_val);
            } else {
                PyErr_Clear();
            }
            if (!use_buffer) {
                Py_DECREF(tgt_mv); tgt_mv = NULL;
            }
        }
        if (!use_buffer) {
            Py_DECREF(src_mv); src_mv = NULL;
        }
    }

    /* ---- copy loop (iterative carry) ---- */
    int *num = (int*)calloc((size_t)dim, sizeof(int));
    if (!num) {
        Py_XDECREF(src_mv); Py_XDECREF(tgt_mv);
        free(src_step); free(tgt_step); free(copy_shape);
        free(src_start); free(tgt_start);
        Py_DECREF(src_idx); Py_DECREF(tgt_idx);
        PyErr_NoMemory(); return NULL;
    }

    long long copied = 0;
    int buffer_failed = 0;

    for (long long count = 0; count < total; ++count) {
        /* compute source / target indices */
        for (int i = 0; i < dim; ++i) {
            int si = src_start[i] + num[i] * src_step[i];
            int ti = tgt_start[i] + num[i] * tgt_step[i];
            Py_DECREF(PyTuple_GET_ITEM(src_idx, i));
            PyTuple_SET_ITEM(src_idx, i, PyLong_FromLong((long)si));
            Py_DECREF(PyTuple_GET_ITEM(tgt_idx, i));
            PyTuple_SET_ITEM(tgt_idx, i, PyLong_FromLong((long)ti));
        }

        PyObject *val = NULL;

        if (use_buffer && !buffer_failed) {
            val = PyObject_GetItem(src_mv, src_idx);
            if (!val) { buffer_failed = 1; PyErr_Clear(); }
        }
        if (!use_buffer || buffer_failed) {
            val = PyObject_GetItem(source, src_idx);
            if (!val) {
                PyErr_Clear();
                PyObject *get_args = PyTuple_Pack(2, source, src_idx);
                if (!get_args) { copied = -1; break; }
                val = py_get_item(self, get_args);
                Py_DECREF(get_args);
                if (!val) { copied = -1; break; }
            }
        }

        if (use_buffer && !buffer_failed) {
            if (PyObject_SetItem(tgt_mv, tgt_idx, val) < 0) {
                buffer_failed = 1; PyErr_Clear();
            }
        }
        if (!use_buffer || buffer_failed) {
            if (PyObject_SetItem(target, tgt_idx, val) < 0) {
                PyErr_Clear();
                PyObject *set_args = PyTuple_Pack(3, target, tgt_idx, val);
                Py_DECREF(val); val = NULL;
                if (!set_args) { copied = -1; break; }
                PyObject *r = py_set_item(self, set_args);
                Py_DECREF(set_args);
                if (!r) { copied = -1; break; }
                Py_DECREF(r);
            }
        }

        Py_XDECREF(val);
        copied++;

        for (int i = dim - 1; i >= 0; --i) {
            num[i]++;
            if (num[i] < copy_shape[i]) break;
            num[i] = 0;
        }
    }

    Py_XDECREF(src_mv);
    Py_XDECREF(tgt_mv);
    free(num);
    free(src_step); free(tgt_step); free(copy_shape);
    free(src_start); free(tgt_start);
    Py_DECREF(src_idx); Py_DECREF(tgt_idx);

    if (copied < 0) return NULL;
    return PyLong_FromLongLong(copied);

oom:
    free(src_shape); free(tgt_shape);
    free(src_step); free(tgt_step); free(copy_shape);
    free(src_start); free(tgt_start);
    PyErr_NoMemory(); return NULL;
fail:
    free(src_shape); free(tgt_shape);
    free(src_step); free(tgt_step); free(copy_shape);
    free(src_start); free(tgt_start);
    return NULL;
}

static PyObject* py_get_item(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *obj, *index;
    if (!PyArg_ParseTuple(args, "OO", &obj, &index)) return NULL;
    PyObject *get_item_method = PyObject_GetAttrString(obj, "__get_item__");
    if (get_item_method != NULL) {
        PyObject *index_args;
        if (PyTuple_Check(index)) {
            index_args = index;
            Py_INCREF(index_args);
        } else {
            index_args = PyTuple_Pack(1, index);
            if (!index_args) { Py_DECREF(get_item_method); return NULL; }
        }
        PyObject *result = PyObject_CallObject(get_item_method, index_args);
        Py_DECREF(index_args);
        Py_DECREF(get_item_method);
        return result;
    }
    PyErr_Clear();
    if (PyTuple_Check(index)) {
        PyObject *temp = obj; Py_INCREF(temp);
        Py_ssize_t n = PyTuple_Size(index);
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject *idx_obj = PyTuple_GetItem(index, i);
            PyObject *idx_num = PyNumber_Index(idx_obj);
            if (!idx_num) { Py_DECREF(temp); return NULL; }
            Py_ssize_t idx = PyLong_AsSsize_t(idx_num);
            Py_DECREF(idx_num);
            if (idx == -1 && PyErr_Occurred()) { Py_DECREF(temp); return NULL; }
            PyObject *next = PySequence_GetItem(temp, idx);
            Py_DECREF(temp);
            if (!next) return NULL;
            temp = next;
        }
        return temp;
    } else {
        PyObject *idx_num = PyNumber_Index(index);
        if (!idx_num) return NULL;
        Py_ssize_t idx = PyLong_AsSsize_t(idx_num);
        Py_DECREF(idx_num);
        if (idx == -1 && PyErr_Occurred()) return NULL;
        return PySequence_GetItem(obj, idx);
    }
}

static PyObject* py_set_item(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *obj, *index, *value;
    if (!PyArg_ParseTuple(args, "OOO", &obj, &index, &value)) return NULL;
    
    // First try __set_item__ method (our custom interface)
    PyObject *set_item_method = PyObject_GetAttrString(obj, "__set_item__");
    if (set_item_method != NULL) {
        PyObject *result = PyObject_CallFunctionObjArgs(set_item_method, index, value, NULL);
        Py_DECREF(set_item_method);
        return result;
    }
    PyErr_Clear();
    
    // Fall back to generic sequence indexing
    if (PyTuple_Check(index)) {
        PyObject *temp = obj; Py_INCREF(temp);
        Py_ssize_t n = PyTuple_Size(index);
        for (Py_ssize_t i = 0; i < n - 1; ++i) {
            PyObject *idx_obj = PyTuple_GetItem(index, i);
            PyObject *idx_num = PyNumber_Index(idx_obj);
            if (!idx_num) { Py_DECREF(temp); return NULL; }
            Py_ssize_t idx = PyLong_AsSsize_t(idx_num);
            Py_DECREF(idx_num);
            if (idx == -1 && PyErr_Occurred()) { Py_DECREF(temp); return NULL; }
            PyObject *next = PySequence_GetItem(temp, idx);
            Py_DECREF(temp);
            if (!next) return NULL;
            temp = next;
        }
        /* Last index: set item */
        if (n == 0) {
            PyErr_SetString(PyExc_ValueError, "not enough values to unpack");
            Py_DECREF(temp);
            return NULL;
        }
        PyObject *last_idx = PyTuple_GetItem(index, n - 1);
        PyObject *last_num = PyNumber_Index(last_idx);
        if (!last_num) { Py_DECREF(temp); return NULL; }
        Py_ssize_t idx = PyLong_AsSsize_t(last_num);
        Py_DECREF(last_num);
        if (idx == -1 && PyErr_Occurred()) { Py_DECREF(temp); return NULL; }
        int result = PySequence_SetItem(temp, idx, value);
        Py_DECREF(temp);
        if (result < 0) return NULL;
        Py_RETURN_NONE;
    } else {
        PyObject *idx_num = PyNumber_Index(index);
        if (!idx_num) return NULL;
        Py_ssize_t idx = PyLong_AsSsize_t(idx_num);
        Py_DECREF(idx_num);
        if (idx == -1 && PyErr_Occurred()) return NULL;
        int result = PySequence_SetItem(obj, idx, value);
        if (result < 0) return NULL;
        Py_RETURN_NONE;
    }
}

/* ------------------------------------------------------------------
Basic similarity functions
------------------------------------------------------------------ */
static PyObject* py_cos(PyObject *self, PyObject *args) {
    (void)self;
    double a, b, ab; PyObject *name;
    if (!PyArg_ParseTuple(args, "dddO", &a, &b, &ab, &name)) return NULL;
    return PyFloat_FromDouble(cos_(a, b, ab, NULL));
}

static PyObject* py_mod(PyObject *self, PyObject *args) {
    (void)self;
    double a, b, ab; PyObject *name;
    if (!PyArg_ParseTuple(args, "dddO", &a, &b, &ab, &name)) return NULL;
    return PyFloat_FromDouble(mod_(a, b, ab, NULL));
}

static PyObject* py_cosmod(PyObject *self, PyObject *args) {
    (void)self;
    double a, b, ab; PyObject *name;
    if (!PyArg_ParseTuple(args, "dddO", &a, &b, &ab, &name)) return NULL;
    return PyFloat_FromDouble(cosmod_(a, b, ab, NULL));
}

static PyObject* py_convolution(PyObject *self, PyObject *args) {
    (void)self;
    double a, b, ab; PyObject *name;
    if (!PyArg_ParseTuple(args, "dddO", &a, &b, &ab, &name)) return NULL;
    return PyFloat_FromDouble(convolution_(a, b, ab, NULL));
}

static PyObject* py_no_done(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    (void)kwds;
    (void)args;
    Py_RETURN_NONE;
}

static PyObject* py_sqrt(PyObject *self, PyObject *args) {
    (void)self;
    double x;
    if (!PyArg_ParseTuple(args, "d", &x)) return NULL;
    return PyFloat_FromDouble(sqrt(x));
}

/* ------------------------------------------------------------------
Custom Python algorithm callback
------------------------------------------------------------------ */
static double py_algo_wrapper(double a, double b, double ab, CallbackContext *ctx) {
    if (!ctx) return 0.0;
    PyObject *py_algo = ctx->algorithm;
    PyObject *ns = ctx->name_space;
    if (!py_algo) {
        if (!ns) return 0.0;
        py_algo = PyObject_GetAttrString(ns, "algorithm");
        if (!py_algo || !PyCallable_Check(py_algo)) {
            Py_XDECREF(py_algo);
            PyErr_Clear();
            return 0.0;
        }
    }
    PyObject *result = PyObject_CallFunction(
        py_algo, "dddO", a, b, ab, ns ? ns : Py_None);
    if (py_algo != ctx->algorithm) Py_DECREF(py_algo);
    if (!result) { PyErr_Clear(); return 0.0; }
    double res = PyFloat_AsDouble(result);
    if (res == -1.0 && PyErr_Occurred()) {
        Py_DECREF(result);
        PyErr_Clear();
        return 0.0;
    }
    Py_DECREF(result);
    return res;
}

static algo_fn get_algo(PyObject *name) {
    if (!name) return cosmod_;
    if (PyUnicode_Check(name)) {
        const char *s = PyUnicode_AsUTF8(name);
        if (!s) return NULL;
        if (strcmp(s, "cos") == 0) return cos_;
        if (strcmp(s, "mod") == 0) return mod_;
        if (strcmp(s, "cosmod") == 0) return cosmod_;
        PyErr_SetString(PyExc_ValueError, "unknown algorithm"); return NULL;
    }
    if (PyCallable_Check(name)) {
        return py_algo_wrapper;
    }
    PyErr_SetString(PyExc_TypeError, "algorithm must be a string or callable"); return NULL;
}

/* ------------------------------------------------------------------
Core algorithm implementations (added)
------------------------------------------------------------------ */

/* Passive mode */
Data* cos_comparison_passive(const Data *data,
                             const int *window_size,
                             double w1, double w2, double b1, double b2,
                             const int start[], const int end[],
                             const int step[], const int d[],
                             algo_fn algorithm,
                             CallbackContext *ctx,
                             const int output_start[], const int output_step[],
                             PyObject *output_obj, Data *output) {
                                 (void)output_obj;
    if (!data || !window_size) return NULL;
    int dim = data->dimension;
    
    // Validate step > 0 for all dimensions to prevent division by zero
    for (int i = 0; i < dim; ++i) {
        if (step[i] <= 0) {
            PyErr_SetString(PyExc_ValueError, "step must be positive for all dimensions");
            return NULL;
        }
    }
    
    // Validate window_size > 0 for all dimensions
    for (int i = 0; i < dim; ++i) {
        if (window_size[i] <= 0) {
            PyErr_SetString(PyExc_ValueError, "window_size must be positive for all dimensions");
            return NULL;
        }
    }
    
    int *num = (int*)malloc((size_t)(dim) * sizeof(int));
    if (!num) { PyErr_NoMemory(); return NULL; }
    for (int i = 0; i < dim; ++i) {
        int eff = end[i] - start[i] - window_size[i] - d[i];
        if (eff < 0) { free(num); return NULL; }
        num[i] = (step[i] != 0) ? eff / step[i] + 1 : 0;
    }
    int output_is_data = (output != NULL);
    if (!output_is_data) {
        output = compute_output_shape(dim, num, output_start, output_step);
        if (!output) { free(num); return NULL; }
    }
    int *num_list = (int*)malloc(((size_t)(dim) + 1) * sizeof(int));
    int *inner_list = (int*)malloc(((size_t)(dim) + 1) * sizeof(int));
    int *main_place = (int*)malloc((size_t)(dim) * sizeof(int));
    int *other_place = (int*)malloc((size_t)(dim) * sizeof(int));
    int *out_idx = (int*)malloc((size_t)(dim) * sizeof(int));
    if (!num_list || !inner_list || !main_place || !other_place || !out_idx) {
        free(num); if (!output_is_data) Data_free(output);
        free(num_list); free(inner_list); free(main_place); free(other_place); free(out_idx);
        PyErr_NoMemory(); return NULL;
    }
    for (int i = 0; i <= dim; ++i) { num_list[i] = 1; inner_list[i] = 1; }
    int flag = dim;
    double main_sum = 0.0, other_sum = 0.0, mu_sum = 0.0;
    while (flag > 0) {
        if (flag == dim) {
            for (int i = 1; i <= dim; ++i) inner_list[i] = 1;
            int inner_flag = dim;
            main_sum = 0.0; other_sum = 0.0; mu_sum = 0.0;
            while (inner_flag > 0) {
                if (inner_flag == dim) {
                    for (int i = 0; i < dim; ++i) {
                        main_place[i] = start[i] + step[i] * (num_list[i+1] - 1) + (inner_list[i+1] - 1);
                        other_place[i] = main_place[i] + d[i];
                    }
                    double a = w1 * Data_get(data, main_place) + b1;
                    double b = w2 * Data_get(data, other_place) + b2;
                    main_sum += a * a;
                    other_sum += b * b;
                    mu_sum += a * b;
                    if (inner_list[dim] < window_size[dim - 1]) {
                        inner_list[dim]++;
                    } else {
                        inner_list[dim] = 1;
                        inner_flag--;
                    }
                } else {
                    if (inner_list[inner_flag] < window_size[inner_flag - 1]) {
                        inner_list[inner_flag]++;
                        inner_flag = dim;
                    } else {
                        inner_list[inner_flag] = 1;
                        inner_flag--;
                    }
                }
            }
            for (int i = 0; i < dim; ++i)
                out_idx[i] = output_start[i] + output_step[i] * (num_list[i+1] - 1);
            double res = algorithm ? algorithm(main_sum, other_sum, mu_sum, ctx)
                                   : cosmod_(main_sum, other_sum, mu_sum, ctx);
            Data_set(output, out_idx, res);
            if (num_list[dim] < num[dim - 1]) {
                num_list[dim]++;
            } else {
                num_list[dim] = 1;
                flag--;
            }
        } else {
            if (num_list[flag] < num[flag - 1]) {
                num_list[flag]++;
                flag = dim;
            } else {
                num_list[flag] = 1;
                flag--;
            }
        }
    }
    free(num); free(num_list); free(inner_list);
    free(main_place); free(other_place); free(out_idx);
    return output;
}

/* Active mode */
Data* cos_comparison_active(const Data *data, const Data *kernel,
                            double w1, double w2, double b1, double b2,
                            const int start[], const int end[],
                            const int step[],
                            algo_fn algorithm,
                            CallbackContext *ctx,
                            const int output_start[], const int output_step[],
                            PyObject *output_obj, Data *output) {
                                (void)output_obj;
    if (!data || !kernel || data->dimension != kernel->dimension) return NULL;
    int dim = data->dimension;
    const int *window_size = kernel->shape;
    
    // Validate step > 0 for all dimensions to prevent division by zero
    for (int i = 0; i < dim; ++i) {
        if (step[i] <= 0) {
            PyErr_SetString(PyExc_ValueError, "step must be positive for all dimensions");
            return NULL;
        }
    }
    
    // Validate kernel (window) size > 0 for all dimensions
    for (int i = 0; i < dim; ++i) {
        if (window_size[i] <= 0) {
            PyErr_SetString(PyExc_ValueError, "kernel size must be positive for all dimensions");
            return NULL;
        }
    }
    
    int *num = (int*)malloc((size_t)(dim) * sizeof(int));
    if (!num) { PyErr_NoMemory(); return NULL; }
    for (int i = 0; i < dim; ++i) {
        int eff = end[i] - start[i] - window_size[i];
        if (eff < 0) { free(num); return NULL; }
        num[i] = (step[i] != 0) ? eff / step[i] + 1 : 0;
    }
    int output_is_data = (output != NULL);
    if (!output_is_data) {
        output = compute_output_shape(dim, num, output_start, output_step);
        if (!output) { free(num); return NULL; }
    }
    int *num_list = (int*)malloc(((size_t)(dim) + 1) * sizeof(int));
    int *inner_list = (int*)malloc(((size_t)(dim) + 1) * sizeof(int));
    int *data_place = (int*)malloc((size_t)(dim) * sizeof(int));
    int *kern_place = (int*)malloc((size_t)(dim) * sizeof(int));
    int *out_idx = (int*)malloc((size_t)(dim) * sizeof(int));
    if (!num_list || !inner_list || !data_place || !kern_place || !out_idx) {
        free(num); if (!output_is_data) Data_free(output);
        free(num_list); free(inner_list); free(data_place); free(kern_place); free(out_idx);
        PyErr_NoMemory(); return NULL;
    }
    for (int i = 0; i <= dim; ++i) { num_list[i] = 1; inner_list[i] = 1; }
    int flag = dim;
    double main_sum = 0.0, other_sum = 0.0, mu_sum = 0.0;
    while (flag > 0) {
        if (flag == dim) {
            for (int i = 1; i <= dim; ++i) inner_list[i] = 1;
            int inner_flag = dim;
            main_sum = 0.0; other_sum = 0.0; mu_sum = 0.0;
            while (inner_flag > 0) {
                if (inner_flag == dim) {
                    for (int i = 0; i < dim; ++i) {
                        data_place[i] = start[i] + step[i] * (num_list[i+1] - 1) + (inner_list[i+1] - 1);
                        kern_place[i] = inner_list[i+1] - 1;
                    }
                    double a = w1 * Data_get(data, data_place) + b1;
                    double b = w2 * Data_get(kernel, kern_place) + b2;
                    main_sum += a * a;
                    other_sum += b * b;
                    mu_sum += a * b;
                    if (inner_list[dim] < window_size[dim - 1]) {
                        inner_list[dim]++;
                    } else {
                        inner_list[dim] = 1;
                        inner_flag--;
                    }
                } else {
                    if (inner_list[inner_flag] < window_size[inner_flag - 1]) {
                        inner_list[inner_flag]++;
                        inner_flag = dim;
                    } else {
                        inner_list[inner_flag] = 1;
                        inner_flag--;
                    }
                }
            }
            for (int i = 0; i < dim; ++i)
                out_idx[i] = output_start[i] + output_step[i] * (num_list[i+1] - 1);
            double res = algorithm ? algorithm(main_sum, other_sum, mu_sum, ctx)
                                   : cosmod_(main_sum, other_sum, mu_sum, ctx);
            Data_set(output, out_idx, res);
            if (num_list[dim] < num[dim - 1]) {
                num_list[dim]++;
            } else {
                num_list[dim] = 1;
                flag--;
            }
        } else {
            if (num_list[flag] < num[flag - 1]) {
                num_list[flag]++;
                flag = dim;
            } else {
                num_list[flag] = 1;
                flag--;
            }
        }
    }
    free(num); free(num_list); free(inner_list);
    free(data_place); free(kern_place); free(out_idx);
    return output;
}

/* Full tensor similarity */
double cos_full(const Data *a, const Data *b, algo_fn algorithm, CallbackContext *ctx) {
    if (!a || !b || !Data_shape_equal(a, b)) {
        return COS_NAN; /* NULL or shape mismatch */
    }
    int total = Data_total(a);
    double sum_a = 0.0, sum_b = 0.0, sum_ab = 0.0;
    if (a->dtype == 0 && b->dtype == 0) {
        /* Fast path: both tensors are double arrays (owned data is
           malloc-aligned; zero-copy buffers are alignment-checked at
           construction).  Plain ISO C pointer arithmetic; no `restrict`
           because a and b may alias the same buffer (cos_full(t, t)). */
        const double *pa = (const double*)a->data;
        const double *pb = (const double*)b->data;
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double va = pa[i];
            double vb = pb[i];
            sum_a += va * va;
            sum_b += vb * vb;
            sum_ab += va * vb;
        }
    } else {
        COS_SIMD_LOOP
        for (int i = 0; i < total; ++i) {
            double va = Data_get_flat(a, i);
            double vb = Data_get_flat(b, i);
            sum_a += va * va;
            sum_b += vb * vb;
            sum_ab += va * vb;
        }
    }
    return algorithm ? algorithm(sum_a, sum_b, sum_ab, ctx)
                     : cosmod_(sum_a, sum_b, sum_ab, ctx);
}

/* Local mean */
Data* cos_local_mean(const Data *data,
                     const int window_size[],
                     const int start[], const int end[],
                     const int step[],
                     const int output_start[], const int output_step[],
                     PyObject *output_obj, Data *output, const double weights[]) {
                         (void)output_obj;
    if (!data || !window_size) return NULL;
    int dim = data->dimension;
    
    // Validate step > 0 for all dimensions to prevent division by zero
    for (int i = 0; i < dim; ++i) {
        if (step[i] <= 0) {
            PyErr_SetString(PyExc_ValueError, "step must be positive for all dimensions");
            return NULL;
        }
    }
    
    // Validate local_size > 0 for all dimensions
    for (int i = 0; i < dim; ++i) {
        if (window_size[i] <= 0) {
            PyErr_SetString(PyExc_ValueError, "local_size must be positive for all dimensions");
            return NULL;
        }
    }
    
    int *num = (int*)calloc((size_t)(dim > 0 ? dim : 1), sizeof(int));
    if (!num) { PyErr_NoMemory(); return NULL; }
    for (int i = 0; i < dim; ++i) {
        int eff = end[i] - start[i] - window_size[i];
        if (eff < 0) { free(num); return NULL; }
        num[i] = (step[i] != 0) ? eff / step[i] + 1 : 0;
    }
    int output_is_data = (output != NULL);
    if (!output_is_data) {
        output = compute_output_shape(dim, num, output_start, output_step);
        if (!output) { free(num); return NULL; }
    }
    long long N = 1;
    for (int i = 0; i < dim; ++i) {
        if (N > LLONG_MAX / (long long)window_size[i]) {
            free(num); if (!output_is_data) Data_free(output);
            PyErr_SetString(PyExc_OverflowError, "local window too large");
            return NULL;
        }
        N *= (long long)window_size[i];
        if (N > INT_MAX) {
            free(num); if (!output_is_data) Data_free(output);
            PyErr_SetString(PyExc_OverflowError, "local window too large");
            return NULL;
        }
    }
    if (N <= 0) { free(num); if (!output_is_data) Data_free(output); return NULL; }
    /* Precompute window strides so weights are indexed in row-major order */
    int *wstrides = (int*)malloc((size_t)(dim) * sizeof(int));
    if (!wstrides) { free(num); if (!output_is_data) Data_free(output); PyErr_NoMemory(); return NULL; }
    {
        long long acc = 1;
        for (int i = dim - 1; i >= 0; --i) {
            wstrides[i] = (int)acc;
            if (acc > LLONG_MAX / (long long)window_size[i]) {
                free(num); free(wstrides); if (!output_is_data) Data_free(output);
                PyErr_SetString(PyExc_OverflowError, "local window too large");
                return NULL;
            }
            acc *= (long long)window_size[i];
            if (acc > INT_MAX) {
                free(num); free(wstrides); if (!output_is_data) Data_free(output);
                PyErr_SetString(PyExc_OverflowError, "local window too large");
                return NULL;
            }
        }
    }
    int *num_list = (int*)malloc(((size_t)(dim) + 1) * sizeof(int));
    int *inner_list = (int*)malloc(((size_t)(dim) + 1) * sizeof(int));
    int *data_place = (int*)malloc((size_t)(dim) * sizeof(int));
    int *out_idx = (int*)malloc((size_t)(dim) * sizeof(int));
    if (!num_list || !inner_list || !data_place || !out_idx) {
        free(num); free(wstrides); free(num_list); free(inner_list);
        free(data_place); free(out_idx);
        if (!output_is_data) Data_free(output);
        PyErr_NoMemory(); return NULL;
    }
    for (int i = 0; i <= dim; ++i) { num_list[i] = 1; inner_list[i] = 1; }
    int flag = dim;
    double sum_x = 0.0;
    while (flag > 0) {
        if (flag == dim) {
            for (int i = 1; i <= dim; ++i) inner_list[i] = 1;
            int inner_flag = dim;
            sum_x = 0.0;
            while (inner_flag > 0) {
                if (inner_flag == dim) {
                    long long wflat = 0;
                    for (int i = 0; i < dim; ++i) {
                        data_place[i] = start[i] + step[i] * (num_list[i+1] - 1) + (inner_list[i+1] - 1);
                        wflat += (long long)(inner_list[i+1] - 1) * wstrides[i];
                    }
                    double x = Data_get(data, data_place);
                    sum_x += weights ? weights[(int)wflat] * x : x;
                    if (inner_list[dim] < window_size[dim - 1]) {
                        inner_list[dim]++;
                    } else {
                        inner_list[dim] = 1;
                        inner_flag--;
                    }
                } else {
                    if (inner_list[inner_flag] < window_size[inner_flag - 1]) {
                        inner_list[inner_flag]++;
                        inner_flag = dim;
                    } else {
                        inner_list[inner_flag] = 1;
                        inner_flag--;
                    }
                }
            }
            for (int i = 0; i < dim; ++i)
                out_idx[i] = output_start[i] + output_step[i] * (num_list[i+1] - 1);
            double mean = sum_x / (double)N;
            Data_set(output, out_idx, mean);
            if (num_list[dim] < num[dim - 1]) {
                num_list[dim]++;
            } else {
                num_list[dim] = 1;
                flag--;
            }
        } else {
            if (num_list[flag] < num[flag - 1]) {
                num_list[flag]++;
                flag = dim;
            } else {
                num_list[flag] = 1;
                flag--;
            }
        }
    }
    free(num); free(num_list); free(inner_list);
    free(data_place); free(out_idx); free(wstrides);
    return output;
}

/* Local variance */
Data* cos_local_variance(const Data *data,
                         const int window_size[],
                         const int start[], const int end[],
                         const int step[],
                         const int output_start[], const int output_step[],
                         PyObject *output_obj, Data *output) {
                             (void)output_obj;
    if (!data || !window_size) return NULL;
    int dim = data->dimension;
    
    // Validate step > 0 for all dimensions to prevent division by zero
    for (int i = 0; i < dim; ++i) {
        if (step[i] <= 0) {
            PyErr_SetString(PyExc_ValueError, "step must be positive for all dimensions");
            return NULL;
        }
    }
    
    // Validate local_size > 0 for all dimensions
    for (int i = 0; i < dim; ++i) {
        if (window_size[i] <= 0) {
            PyErr_SetString(PyExc_ValueError, "local_size must be positive for all dimensions");
            return NULL;
        }
    }
    
    int *num = (int*)calloc((size_t)(dim > 0 ? dim : 1), sizeof(int));
    if (!num) { PyErr_NoMemory(); return NULL; }
    for (int i = 0; i < dim; ++i) {
        int eff = end[i] - start[i] - window_size[i];
        if (eff < 0) { free(num); return NULL; }
        num[i] = (step[i] != 0) ? eff / step[i] + 1 : 0;
    }
    int output_is_data = (output != NULL);
    if (!output_is_data) {
        output = compute_output_shape(dim, num, output_start, output_step);
        if (!output) { free(num); return NULL; }
    }
    long long N = 1;
    for (int i = 0; i < dim; ++i) {
        if (N > LLONG_MAX / (long long)window_size[i]) {
            free(num); if (!output_is_data) Data_free(output);
            PyErr_SetString(PyExc_OverflowError, "local window too large");
            return NULL;
        }
        N *= (long long)window_size[i];
        if (N > INT_MAX) {
            free(num); if (!output_is_data) Data_free(output);
            PyErr_SetString(PyExc_OverflowError, "local window too large");
            return NULL;
        }
    }
    if (N <= 0) { free(num); if (!output_is_data) Data_free(output); return NULL; }
    int *num_list = (int*)malloc(((size_t)(dim) + 1) * sizeof(int));
    int *inner_list = (int*)malloc(((size_t)(dim) + 1) * sizeof(int));
    int *data_place = (int*)malloc((size_t)(dim) * sizeof(int));
    int *out_idx = (int*)malloc((size_t)(dim) * sizeof(int));
    if (!num_list || !inner_list || !data_place || !out_idx) {
        free(num); if (!output_is_data) Data_free(output);
        free(num_list); free(inner_list); free(data_place); free(out_idx);
        PyErr_NoMemory(); return NULL;
    }
    for (int i = 0; i <= dim; ++i) { num_list[i] = 1; inner_list[i] = 1; }
    int flag = dim;
    double sum_x = 0.0, sum_x2 = 0.0;
    while (flag > 0) {
        if (flag == dim) {
            for (int i = 1; i <= dim; ++i) inner_list[i] = 1;
            int inner_flag = dim;
            sum_x = 0.0; sum_x2 = 0.0;
            while (inner_flag > 0) {
                if (inner_flag == dim) {
                    for (int i = 0; i < dim; ++i) {
                        data_place[i] = start[i] + step[i] * (num_list[i+1] - 1) + (inner_list[i+1] - 1);
                    }
                    double val = Data_get(data, data_place);
                    sum_x += val;
                    sum_x2 += val * val;
                    if (inner_list[dim] < window_size[dim - 1]) {
                        inner_list[dim]++;
                    } else {
                        inner_list[dim] = 1;
                        inner_flag--;
                    }
                } else {
                    if (inner_list[inner_flag] < window_size[inner_flag - 1]) {
                        inner_list[inner_flag]++;
                        inner_flag = dim;
                    } else {
                        inner_list[inner_flag] = 1;
                        inner_flag--;
                    }
                }
            }
            for (int i = 0; i < dim; ++i)
                out_idx[i] = output_start[i] + output_step[i] * (num_list[i+1] - 1);
            double mean = sum_x / (double)N;
            double variance = sum_x2 / (double)N - mean * mean;
            Data_set(output, out_idx, variance);
            if (num_list[dim] < num[dim - 1]) {
                num_list[dim]++;
            } else {
                num_list[dim] = 1;
                flag--;
            }
        } else {
            if (num_list[flag] < num[flag - 1]) {
                num_list[flag]++;
                flag = dim;
            } else {
                num_list[flag] = 1;
                flag--;
            }
        }
    }
    free(num); free(num_list); free(inner_list);
    free(data_place); free(out_idx);
    return output;
}

/* ------------------------------------------------------------------
Python wrapper functions (full implementations)
------------------------------------------------------------------ */

/* Passive mode Python wrapper */
/* Derive "<pkg>.core.cos_comparison" from this module's own name (never
 * hard-code the package name) - used to delegate the iterate path of the
 * passive/active functions to the pure Python reference implementation. */
static PyObject* _import_pure_core(PyObject *self) {
    const char *full = PyModule_GetName(self);
    if (!full) return NULL;
    const char *dot = strrchr(full, '.');
    if (!dot) {
        PyErr_SetString(PyExc_ImportError,
                        "cannot derive the package name");
        return NULL;
    }
    size_t len = (size_t)(dot - full);
    char *buf = (char*)malloc(len + sizeof(".cos_comparison"));
    if (!buf) { PyErr_NoMemory(); return NULL; }
    memcpy(buf, full, len);
    memcpy(buf + len, ".cos_comparison", sizeof(".cos_comparison"));
    PyObject *mod = PyImport_ImportModule(buf);
    free(buf);
    return mod;
}

static PyObject* py_passive(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *data_obj, *window_size_obj = NULL, *start_obj = NULL, *end_obj = NULL,
             *step_obj = NULL, *d_obj = NULL, *algo_name = NULL, *output_obj = NULL,
             *output_start_obj = NULL, *output_step_obj = NULL, *start_callback = NULL,
             *end_callback = NULL, *global_error_callback = NULL, *local_error_callback = NULL,
             *return_callback = NULL, *iterate_obj = NULL,
             *transform1_obj = NULL, *transform2_obj = NULL;
    int use_ns = 1;
    PyObject *hook_obj = NULL;
    double w1 = 1.0, w2 = 1.0, b1 = 0.0, b2 = 0.0;
        static char *kwlist[] = {
        "data", "window_size", "w1", "w2", "b1", "b2",
        "start", "end", "step", "d", "algorithm",
        "output", "output_start", "output_step",
        "start_callback", "end_callback",
        "global_error_callback", "local_error_callback",
        "return_callback",
        "use_namespace", "namespace_hook", "iterate", "transform1", "transform2", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OddddOOOOOOOOOOOOOpOOOO", kwlist,
                                     &data_obj, &window_size_obj,
                                     &w1, &w2, &b1, &b2,
                                     &start_obj, &end_obj, &step_obj, &d_obj,
                                     &algo_name, &output_obj,
                                     &output_start_obj, &output_step_obj,
                                     &start_callback, &end_callback,
                                     &global_error_callback, &local_error_callback,
                                     &return_callback, &use_ns, &hook_obj,
                                     &iterate_obj, &transform1_obj, &transform2_obj))
        return NULL;

    if (PyObject_HasAttrString(data_obj, "__cos_comparison_passive__")) {
        PyObject *method = PyObject_GetAttrString(data_obj, "__cos_comparison_passive__");
        if (method) {
            // Create new args without the first element (data_obj), since method is bound
            Py_ssize_t argc = PyTuple_GET_SIZE(args);
            PyObject *new_args = PyTuple_New(argc - 1);
            for (Py_ssize_t i = 1; i < argc; ++i) {
                PyObject *item = PyTuple_GET_ITEM(args, i);
                Py_INCREF(item);
                PyTuple_SET_ITEM(new_args, i - 1, item);
            }
            PyObject *result = PyObject_Call(method, new_args, kwargs);
            Py_DECREF(new_args);
            Py_DECREF(method);
            return result;
        }
        PyErr_Clear();
    }

    /* iterate/transform paths: delegate to the pure Python reference
     * implementation (the iterate mechanism, kernels and transforms live
     * there; the C core keeps the plain fast path when neither is given). */
    if ((iterate_obj != NULL && iterate_obj != Py_None) ||
        (transform1_obj != NULL && transform1_obj != Py_None) ||
        (transform2_obj != NULL && transform2_obj != Py_None)) {
        PyObject *pure = _import_pure_core(self);
        if (!pure) return NULL;
        PyObject *fn = PyObject_GetAttrString(pure, "cos_comparison_passive");
        Py_DECREF(pure);
        if (!fn) return NULL;
        PyObject *result = PyObject_Call(fn, args, kwargs);
        Py_DECREF(fn);
        return result;
    }

    Data *data = pyobj_to_data(data_obj);
    if (!data) return NULL;
    int dim = data->dimension;

    int *window_size = NULL;
    if (parse_opt_int_seq(window_size_obj, &window_size, dim, 1) < 0) { Data_free(data); return NULL; }

    int *start = NULL;
    if (parse_opt_int_seq(start_obj, &start, dim, 0) < 0) { free(window_size); Data_free(data); return NULL; }

    int *end = NULL;
    if (end_obj == NULL) {
        end = (int*)malloc((size_t)(dim > 0 ? dim : 1) * sizeof(int));
        if (!end) { free(start); free(window_size); Data_free(data); PyErr_NoMemory(); return NULL; }
        for (int i = 0; i < dim; ++i) end[i] = data->shape[i];
    } else if (parse_opt_int_seq(end_obj, &end, dim, -1) < 0) {
        free(start); free(window_size); Data_free(data); return NULL;
    }

    int *step = NULL;
    if (parse_opt_int_seq(step_obj, &step, dim, 1) < 0) { free(end); free(start); free(window_size); Data_free(data); return NULL; }

    int *d = NULL;
    if (parse_opt_int_seq(d_obj, &d, dim, 0) < 0) { free(step); free(end); free(start); free(window_size); Data_free(data); return NULL; }
    if (d_obj == NULL && dim > 0) d[0] = 1;

    int *output_start = NULL;
    if (parse_opt_int_seq(output_start_obj, &output_start, dim, 0) < 0) { free(d); free(step); free(end); free(start); free(window_size); Data_free(data); return NULL; }

    int *output_step = NULL;
    if (parse_opt_int_seq(output_step_obj, &output_step, dim, 1) < 0) { free(output_start); free(d); free(step); free(end); free(start); free(window_size); Data_free(data); return NULL; }

    algo_fn algo = get_algo(algo_name);
    if (!algo) {
        free(output_step); free(output_start); free(d); free(step);
        free(end); free(start); free(window_size); Data_free(data);
        if (global_error_callback && PyCallable_Check(global_error_callback)) {
            PyObject *ns = PyObject_CallObject((PyObject*)&FuncNameSpaceType, NULL);
            if (ns) {
                PyObject *exc = PyErr_Occurred();
                PyObject *res = PyObject_CallFunctionObjArgs(
                    global_error_callback, exc ? exc : Py_None, ns, NULL);
                Py_XDECREF(res); Py_DECREF(ns);
            }
        }
        return NULL;
    }

    PyObject *name_space = NULL;
    CallbackContext ctx = {0};
    if (PyCallable_Check(algo_name)) {
        Py_INCREF(algo_name);
        ctx.algorithm = algo_name;
    }
    if (use_ns && (start_callback || end_callback ||
                   global_error_callback ||
                   local_error_callback ||
                   return_callback || ctx.algorithm)) {
        PyObject *hook = hook_obj ? hook_obj :
            (PyObject *)&FuncNameSpaceType;
        PyObject *fill = PyDict_New();
        name_space = NULL;
        if (fill) {
            if (output_obj) { Py_INCREF(output_obj); PyDict_SetItemString(fill, "output", output_obj); }
            PyObject *os_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(os_tuple, i, PyLong_FromLong(output_start[i]));
            PyDict_SetItemString(fill, "output_start", os_tuple); Py_DECREF(os_tuple);
            PyObject *ost_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(ost_tuple, i, PyLong_FromLong(output_step[i]));
            PyDict_SetItemString(fill, "output_step", ost_tuple); Py_DECREF(ost_tuple);
            PyObject *ws_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(ws_tuple, i, PyLong_FromLong(window_size[i]));
            PyDict_SetItemString(fill, "window_size", ws_tuple); Py_DECREF(ws_tuple);
            PyObject *start_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(start_tuple, i, PyLong_FromLong(start[i]));
            PyDict_SetItemString(fill, "start", start_tuple); Py_DECREF(start_tuple);
            PyObject *end_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(end_tuple, i, PyLong_FromLong(end[i]));
            PyDict_SetItemString(fill, "end", end_tuple); Py_DECREF(end_tuple);
            PyObject *d_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(d_tuple, i, PyLong_FromLong(d[i]));
            PyDict_SetItemString(fill, "d", d_tuple); Py_DECREF(d_tuple);
            PyObject *step_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(step_tuple, i, PyLong_FromLong(step[i]));
            PyDict_SetItemString(fill, "step", step_tuple); Py_DECREF(step_tuple);
            if (algo_name) { Py_INCREF(algo_name); PyDict_SetItemString(fill, "algorithm", algo_name); }
            int *num = (int*)malloc((size_t)(dim) * sizeof(int));
    if (!num) { PyErr_NoMemory(); return NULL; }
            for (int i = 0; i < dim; ++i) num[i] = (step[i] != 0) ? (end[i] - start[i] - window_size[i] - d[i]) / step[i] + 1 : 0;
            PyObject *num_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(num_tuple, i, PyLong_FromLong(num[i]));
            PyDict_SetItemString(fill, "num", num_tuple); Py_DECREF(num_tuple);
            free(num);
            name_space = PyObject_Call(hook, PyTuple_New(0), fill);
            Py_DECREF(fill);
        }
        if (name_space) {
            ctx.local_error_callback = local_error_callback;
            ctx.name_space = name_space;
        }
    }

    if (start_callback && PyCallable_Check(start_callback)) {
        PyObject *res = PyObject_CallFunctionObjArgs(start_callback, name_space ? name_space : Py_None, NULL);
        Py_XDECREF(res);
    }

    Data *result = NULL;
    // Save user's output_start/output_step and set to 0/1 for core call when output is provided
    int *saved_os = NULL, *saved_ost = NULL;
    if (output_obj && output_obj != Py_None) {
        saved_os = (int*)malloc((size_t)(dim) * sizeof(int));
        saved_ost = (int*)malloc((size_t)(dim) * sizeof(int));
        if (!saved_os || !saved_ost) {
            free(saved_os); free(saved_ost);
            PyErr_NoMemory();
            return NULL;
        }
        memcpy(saved_os, output_start, dim * sizeof(int));
        memcpy(saved_ost, output_step, dim * sizeof(int));
        for (int i = 0; i < dim; ++i) {
            output_start[i] = 0;
            output_step[i] = 1;
        }
    }
    if (!name_space && !ctx.algorithm) {
        Py_BEGIN_ALLOW_THREADS
        result = cos_comparison_passive(data, window_size, w1, w2, b1, b2,
                                        start, end, step, d,
                                        algo, &ctx,
                                        output_start, output_step,
                                        NULL, NULL);
        Py_END_ALLOW_THREADS
    } else {
        result = cos_comparison_passive(data, window_size, w1, w2, b1, b2,
                                        start, end, step, d,
                                        algo, &ctx,
                                        output_start, output_step,
                                        NULL, NULL);
    }
    // Restore user's output_start/output_step
    if (saved_os) {
        memcpy(output_start, saved_os, dim * sizeof(int));
        memcpy(output_step, saved_ost, dim * sizeof(int));
        free(saved_os);
        free(saved_ost);
    }

    if (!result) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "effectless args.");
        if (global_error_callback && PyCallable_Check(global_error_callback)) {
            PyObject *exc = PyErr_Occurred();
            if (exc) { Py_INCREF(exc); PyErr_Clear(); PyObject *res = PyObject_CallFunctionObjArgs(global_error_callback, exc, name_space ? name_space : Py_None, NULL); Py_XDECREF(res);  Py_DECREF(exc); }
        }
        // Free allocated memory before return
        free(output_step); free(output_start); free(d); free(step);
        free(end); free(start); free(window_size); Data_free(data);
        Py_XDECREF(name_space);
        return NULL;
    }

    PyObject *py_result;
    if (output_obj && output_obj != Py_None) {
        // Write result to user-provided output object
        int r_dim = result->dimension;
        int *idx = (int*)malloc((size_t)(r_dim) * sizeof(int));
        int *out_idx = (int*)malloc((size_t)(r_dim) * sizeof(int));
        if (!idx || !out_idx) { free(idx); free(out_idx); Data_free(result); free(output_step); free(output_start); free(d); free(step); free(end); free(start); free(window_size); Data_free(data); Py_XDECREF(name_space); PyErr_NoMemory(); return NULL; }
        // Check if output is our C Vector type for direct write
        Data *out_data = NULL;
        if (PyObject_IsInstance(output_obj, (PyObject*)&VectorizeType)) {
            Vector *vec = (Vector*)output_obj;
            out_data = vec->data;
        }
        for (int i = 0; i < r_dim; ++i) idx[i] = 0;
        if (out_data) {
            for (int i = 0; i < r_dim; ++i) {
                int last = output_start[i] + output_step[i] * (result->shape[i] - 1);
                if (output_start[i] < 0 || last >= out_data->shape[i]) {
                    free(idx); free(out_idx); Data_free(result);
                    free(output_step); free(output_start); free(d); free(step); free(end); free(start); free(window_size); Data_free(data); Py_XDECREF(name_space);
                    PyErr_SetString(PyExc_IndexError, "index out of range");
                    return NULL;
                }
            }
        }
        int flag = r_dim - 1;
        while (flag >= 0) {
            if (flag == r_dim - 1) {
                // Calculate output index
                for (int i = 0; i < r_dim; ++i) {
                    out_idx[i] = output_start[i] + output_step[i] * idx[i];
                }
                // Get value from result
                double val = Data_get(result, idx);
                // Write to output: direct Data write for Vector, else generic sequence
                if (out_data) {
                    Data_set(out_data, out_idx, val);
                } else {
                    py_set_item_value(output_obj, out_idx, r_dim, 0, val);
                }
                // Advance last dimension
                idx[flag]++;
                if (idx[flag] >= result->shape[flag]) {
                    idx[flag] = 0;
                    flag--;
                }
            } else {
                // Carry over
                idx[flag]++;
                if (idx[flag] >= result->shape[flag]) {
                    idx[flag] = 0;
                    flag--;
                } else {
                    flag = r_dim - 1;
                }
            }
        }
        free(idx); free(out_idx);
        Data_free(result);
        py_result = output_obj;
        Py_INCREF(py_result);
    } else {
        PyTypeObject *result_type = PyObject_IsInstance(data_obj, (PyObject*)&VectorizeType) ? Py_TYPE(data_obj) : NULL;
        py_result = data_to_vector(result, result_type);
    }

    if (end_callback && PyCallable_Check(end_callback)) {
        if (name_space)
            PyObject_SetAttrString(name_space, "output", py_result);
        PyObject *res = PyObject_CallFunctionObjArgs(end_callback, name_space ? name_space : Py_None, NULL);
        Py_XDECREF(res);
    }

    if (return_callback && PyCallable_Check(return_callback)) {
        PyObject *ret = PyObject_CallFunctionObjArgs(return_callback, py_result, name_space ? name_space : Py_None, NULL);
        Py_DECREF(py_result);
        py_result = ret;
    }

    // Free all allocated memory
    free(output_step); free(output_start); free(d); free(step);
    free(end); free(start); free(window_size); Data_free(data);

    Py_XDECREF(name_space);
    return py_result;
}

/* Active mode Python wrapper */
static PyObject* py_active(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *data_obj, *kernel_obj = NULL, *start_obj = NULL, *end_obj = NULL,
             *step_obj = NULL, *algo_name = NULL, *output_obj = NULL,
             *output_start_obj = NULL, *output_step_obj = NULL, *start_callback = NULL,
             *end_callback = NULL, *global_error_callback = NULL, *local_error_callback = NULL,
             *return_callback = NULL, *iterate_obj = NULL,
             *transform1_obj = NULL, *transform2_obj = NULL;
    int use_ns = 1;
    PyObject *hook_obj = NULL;
    double w1 = 1.0, w2 = 1.0, b1 = 0.0, b2 = 0.0;
        static char *kwlist[] = {
        "data", "kernel", "w1", "w2", "b1", "b2",
        "start", "end", "step", "algorithm",
        "output", "output_start", "output_step",
        "start_callback", "end_callback",
        "global_error_callback", "local_error_callback",
        "return_callback",
        "use_namespace", "namespace_hook", "iterate", "transform1", "transform2", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OddddOOOOOOOOOOOOpOOOO", kwlist,
                                     &data_obj, &kernel_obj,
                                     &w1, &w2, &b1, &b2,
                                     &start_obj, &end_obj, &step_obj,
                                     &algo_name, &output_obj,
                                     &output_start_obj, &output_step_obj,
                                     &start_callback, &end_callback,
                                     &global_error_callback, &local_error_callback,
                                     &return_callback, &use_ns, &hook_obj,
                                     &iterate_obj, &transform1_obj, &transform2_obj))
        return NULL;

    /* Match pure Python signature cos_comparison_active(data, *arg, kernel=None):
       extra positional args are absorbed by *arg and ignored; kernel must be
       passed as a keyword. If a second positional was given but "kernel" was
       not explicitly in kwargs, treat it as *arg (kernel stays NULL). */
    if (kernel_obj && (!kwargs || !PyDict_GetItemString(kwargs, "kernel"))
        && PyTuple_GET_SIZE(args) >= 2) {
        kernel_obj = NULL;
    }

    if (!kernel_obj) { PyErr_SetString(PyExc_ValueError, "kernel must be provided for active mode"); return NULL; }

    if (PyObject_HasAttrString(data_obj, "__cos_comparison_active__")) {
        PyObject *method = PyObject_GetAttrString(data_obj, "__cos_comparison_active__");
        if (method) {
            // Create new args without the first element (data_obj), since method is bound
            Py_ssize_t argc = PyTuple_GET_SIZE(args);
            PyObject *new_args = PyTuple_New(argc - 1);
            for (Py_ssize_t i = 1; i < argc; ++i) {
                PyObject *item = PyTuple_GET_ITEM(args, i);
                Py_INCREF(item);
                PyTuple_SET_ITEM(new_args, i - 1, item);
            }
            PyObject *result = PyObject_Call(method, new_args, kwargs);
            Py_DECREF(new_args);
            Py_DECREF(method);
            return result;
        }
        PyErr_Clear();
    }

    /* iterate/transform paths: delegate to the pure Python reference implementation. */
    if ((iterate_obj != NULL && iterate_obj != Py_None) ||
        (transform1_obj != NULL && transform1_obj != Py_None) ||
        (transform2_obj != NULL && transform2_obj != Py_None)) {
        PyObject *pure = _import_pure_core(self);
        if (!pure) return NULL;
        PyObject *fn = PyObject_GetAttrString(pure, "cos_comparison_active");
        Py_DECREF(pure);
        if (!fn) return NULL;
        PyObject *result = PyObject_Call(fn, args, kwargs);
        Py_DECREF(fn);
        return result;
    }

    Data *data = pyobj_to_data(data_obj);
    if (!data) return NULL;
    Data *kernel = pyobj_to_data(kernel_obj);
    if (!kernel) { Data_free(data); return NULL; }
    int dim = data->dimension;

    int *start = NULL;
    if (parse_opt_int_seq(start_obj, &start, dim, 0) < 0) { Data_free(data); Data_free(kernel); return NULL; }

    int *end = NULL;
    if (end_obj == NULL) {
        end = (int*)malloc((size_t)(dim > 0 ? dim : 1) * sizeof(int));
        if (!end) { free(start); Data_free(data); Data_free(kernel); PyErr_NoMemory(); return NULL; }
        for (int i = 0; i < dim; ++i) end[i] = data->shape[i];
    } else if (parse_opt_int_seq(end_obj, &end, dim, -1) < 0) {
        free(start); Data_free(data); Data_free(kernel); return NULL;
    }

    int *step = NULL;
    if (parse_opt_int_seq(step_obj, &step, dim, 1) < 0) { free(end); free(start); Data_free(data); Data_free(kernel); return NULL; }

    int *output_start = NULL;
    if (parse_opt_int_seq(output_start_obj, &output_start, dim, 0) < 0) { free(step); free(end); free(start); Data_free(data); Data_free(kernel); return NULL; }

    int *output_step = NULL;
    if (parse_opt_int_seq(output_step_obj, &output_step, dim, 1) < 0) { free(output_start); free(step); free(end); free(start); Data_free(data); Data_free(kernel); return NULL; }

    algo_fn algo = get_algo(algo_name);
    if (!algo) {
        free(output_step); free(output_start); free(step);
        free(end); free(start); Data_free(data); Data_free(kernel);
        if (global_error_callback && PyCallable_Check(global_error_callback)) {
            PyObject *ns = PyObject_CallObject((PyObject*)&FuncNameSpaceType, NULL);
            if (ns) {
                PyObject *exc = PyErr_Occurred();
                PyObject *res = PyObject_CallFunctionObjArgs(
                    global_error_callback, exc ? exc : Py_None, ns, NULL);
                Py_XDECREF(res); Py_DECREF(ns);
            }
        }
        return NULL;
    }

    PyObject *name_space = NULL;
    CallbackContext ctx = {0};
    if (PyCallable_Check(algo_name)) {
        Py_INCREF(algo_name);
        ctx.algorithm = algo_name;
    }
    if (use_ns && (start_callback || end_callback ||
                   global_error_callback ||
                   local_error_callback ||
                   return_callback || ctx.algorithm)) {
        PyObject *hook = hook_obj ? hook_obj :
            (PyObject *)&FuncNameSpaceType;
        PyObject *fill = PyDict_New();
        name_space = NULL;
        if (fill) {
            if (output_obj) { Py_INCREF(output_obj); PyDict_SetItemString(fill, "output", output_obj); }
            PyObject *os_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(os_tuple, i, PyLong_FromLong(output_start[i]));
            PyDict_SetItemString(fill, "output_start", os_tuple); Py_DECREF(os_tuple);
            PyObject *ost_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(ost_tuple, i, PyLong_FromLong(output_step[i]));
            PyDict_SetItemString(fill, "output_step", ost_tuple); Py_DECREF(ost_tuple);
            PyObject *ws_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(ws_tuple, i, PyLong_FromLong(kernel->shape[i]));
            PyDict_SetItemString(fill, "window_size", ws_tuple); Py_DECREF(ws_tuple);
            Py_INCREF(kernel_obj); PyDict_SetItemString(fill, "kernel", kernel_obj);
            PyObject *start_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(start_tuple, i, PyLong_FromLong(start[i]));
            PyDict_SetItemString(fill, "start", start_tuple); Py_DECREF(start_tuple);
            PyObject *end_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(end_tuple, i, PyLong_FromLong(end[i]));
            PyDict_SetItemString(fill, "end", end_tuple); Py_DECREF(end_tuple);
            PyObject *step_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(step_tuple, i, PyLong_FromLong(step[i]));
            PyDict_SetItemString(fill, "step", step_tuple); Py_DECREF(step_tuple);
            if (algo_name) { Py_INCREF(algo_name); PyDict_SetItemString(fill, "algorithm", algo_name); }
            int *num = (int*)malloc((size_t)(dim) * sizeof(int));
    if (!num) { PyErr_NoMemory(); return NULL; }
            for (int i = 0; i < dim; ++i) num[i] = (step[i] != 0) ? (end[i] - start[i] - kernel->shape[i]) / step[i] + 1 : 0;
            PyObject *num_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(num_tuple, i, PyLong_FromLong(num[i]));
            PyDict_SetItemString(fill, "num", num_tuple); Py_DECREF(num_tuple);
            free(num);
            name_space = PyObject_Call(hook, PyTuple_New(0), fill);
            Py_DECREF(fill);
        }
        if (name_space) {
            ctx.local_error_callback = local_error_callback;
            ctx.name_space = name_space;
        }
    }

    if (start_callback && PyCallable_Check(start_callback)) {
        PyObject *res = PyObject_CallFunctionObjArgs(start_callback, name_space ? name_space : Py_None, NULL);
        Py_XDECREF(res);
    }

    Data *result = NULL;
    // Save user's output_start/output_step and set to 0/1 for core call when output is provided
    int *saved_os = NULL, *saved_ost = NULL;
    if (output_obj && output_obj != Py_None) {
        saved_os = (int*)malloc((size_t)(dim) * sizeof(int));
        saved_ost = (int*)malloc((size_t)(dim) * sizeof(int));
        if (!saved_os || !saved_ost) {
            free(saved_os); free(saved_ost);
            PyErr_NoMemory();
            return NULL;
        }
        memcpy(saved_os, output_start, dim * sizeof(int));
        memcpy(saved_ost, output_step, dim * sizeof(int));
        for (int i = 0; i < dim; ++i) {
            output_start[i] = 0;
            output_step[i] = 1;
        }
    }
    if (!name_space && !ctx.algorithm) {
        Py_BEGIN_ALLOW_THREADS
        result = cos_comparison_active(data, kernel, w1, w2, b1, b2,
                                       start, end, step,
                                       algo, &ctx,
                                       output_start, output_step,
                                       NULL, NULL);
        Py_END_ALLOW_THREADS
    } else {
        result = cos_comparison_active(data, kernel, w1, w2, b1, b2,
                                       start, end, step,
                                       algo, &ctx,
                                       output_start, output_step,
                                       NULL, NULL);
    }
    // Restore user's output_start/output_step
    if (saved_os) {
        memcpy(output_start, saved_os, dim * sizeof(int));
        memcpy(output_step, saved_ost, dim * sizeof(int));
        free(saved_os);
        free(saved_ost);
    }

    if (!result) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "effectless args.");
        if (global_error_callback && PyCallable_Check(global_error_callback)) {
            PyObject *exc = PyErr_Occurred();
            if (exc) { Py_INCREF(exc); PyErr_Clear(); PyObject *res = PyObject_CallFunctionObjArgs(global_error_callback, exc, name_space ? name_space : Py_None, NULL); Py_XDECREF(res);  Py_DECREF(exc); }
        }
        // Free allocated memory before return
        free(output_step); free(output_start); free(step);
        free(end); free(start); Data_free(data); Data_free(kernel);
        Py_XDECREF(name_space);
        return NULL;
    }

    PyObject *py_result;
    if (output_obj && output_obj != Py_None) {
        // Write result to user-provided output object
        int r_dim = result->dimension;
        int *idx = (int*)malloc((size_t)(r_dim) * sizeof(int));
        int *out_idx = (int*)malloc((size_t)(r_dim) * sizeof(int));
        if (!idx || !out_idx) { free(idx); free(out_idx); Data_free(result); free(output_step); free(output_start); free(step); free(end); free(start); Data_free(data); Data_free(kernel); Py_XDECREF(name_space); PyErr_NoMemory(); return NULL; }
        // Check if output is our C Vector type for direct write
        Data *out_data = NULL;
        if (PyObject_IsInstance(output_obj, (PyObject*)&VectorizeType)) {
            Vector *vec = (Vector*)output_obj;
            out_data = vec->data;
        }
        for (int i = 0; i < r_dim; ++i) idx[i] = 0;
        if (out_data) {
            for (int i = 0; i < r_dim; ++i) {
                int last = output_start[i] + output_step[i] * (result->shape[i] - 1);
                if (output_start[i] < 0 || last >= out_data->shape[i]) {
                    free(idx); free(out_idx); Data_free(result);
                    free(output_step); free(output_start); free(step); free(end); free(start); Data_free(data); Data_free(kernel); Py_XDECREF(name_space);
                    PyErr_SetString(PyExc_IndexError, "index out of range");
                    return NULL;
                }
            }
        }
        int flag = r_dim - 1;
        while (flag >= 0) {
            if (flag == r_dim - 1) {
                // Calculate output index
                for (int i = 0; i < r_dim; ++i) {
                    out_idx[i] = output_start[i] + output_step[i] * idx[i];
                }
                // Get value from result
                double val = Data_get(result, idx);
                // Write to output: direct Data write for Vector, else generic sequence
                if (out_data) {
                    Data_set(out_data, out_idx, val);
                } else {
                    py_set_item_value(output_obj, out_idx, r_dim, 0, val);
                }
                // Advance last dimension
                idx[flag]++;
                if (idx[flag] >= result->shape[flag]) {
                    idx[flag] = 0;
                    flag--;
                }
            } else {
                // Carry over
                idx[flag]++;
                if (idx[flag] >= result->shape[flag]) {
                    idx[flag] = 0;
                    flag--;
                } else {
                    flag = r_dim - 1;
                }
            }
        }
        free(idx); free(out_idx);
        Data_free(result);
        py_result = output_obj;
        Py_INCREF(py_result);
    } else {
        PyTypeObject *result_type = PyObject_IsInstance(data_obj, (PyObject*)&VectorizeType) ? Py_TYPE(data_obj) : NULL;
        py_result = data_to_vector(result, result_type);
    }

    if (end_callback && PyCallable_Check(end_callback)) {
        if (name_space)
            PyObject_SetAttrString(name_space, "output", py_result);
        PyObject *res = PyObject_CallFunctionObjArgs(end_callback, name_space ? name_space : Py_None, NULL);
        Py_XDECREF(res);
    }

    if (return_callback && PyCallable_Check(return_callback)) {
        PyObject *ret = PyObject_CallFunctionObjArgs(return_callback, py_result, name_space ? name_space : Py_None, NULL);
        Py_DECREF(py_result);
        py_result = ret;
    }

    // Free all allocated memory
    free(output_step); free(output_start); free(step);
    free(end); free(start); Data_free(data); Data_free(kernel);

    Py_XDECREF(name_space);
    return py_result;
}

/* Full tensor similarity Python wrapper */
static PyObject* py_cos_full(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *a_obj, *b_obj, *algo_name = NULL;
        static char *kwlist[] = {"a", "b", "algorithm", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist, &a_obj, &b_obj, &algo_name))
        return NULL;

    Data *a = pyobj_to_data(a_obj);
    if (!a) return NULL;
    Data *b = pyobj_to_data(b_obj);
    if (!b) { Data_free(a); return NULL; }

    if (a->dimension != b->dimension) { PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same."); Data_free(a); Data_free(b); return NULL; }
    for (int i = 0; i < a->dimension; ++i) {
        if (a->shape[i] != b->shape[i]) { PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same."); Data_free(a); Data_free(b); return NULL; }
    }

    /* Match pure Python: empty tensors raise IndexError (no elements). */
    {
        long long total_a = 1;
        for (int i = 0; i < a->dimension; ++i) total_a *= (long long)a->shape[i];
        if (total_a == 0) {
            Data_free(a); Data_free(b);
            PyErr_SetString(PyExc_IndexError, "list index out of range");
            return NULL;
        }
    }

    algo_fn algo;
    if (algo_name) {
        algo = get_algo(algo_name);
    } else {
        PyObject *tmp = PyUnicode_FromString("cos");
        algo = tmp ? get_algo(tmp) : NULL;
        Py_XDECREF(tmp);
    }
    if (!algo) { Data_free(a); Data_free(b); return NULL; }

    CallbackContext ctx = {0};
    PyObject *ns = NULL;
    if (algo_name && PyCallable_Check(algo_name)) {
        ns = PyObject_CallObject((PyObject*)&FuncNameSpaceType, NULL);
        if (ns) { Py_INCREF(algo_name); PyObject_SetAttrString(ns, "algorithm", algo_name); ctx.name_space = ns; }
    }

    double result = 0.0;
    if (!ns) {
        Py_BEGIN_ALLOW_THREADS
        result = cos_full(a, b, algo, NULL);
        Py_END_ALLOW_THREADS
    } else {
        result = cos_full(a, b, algo, &ctx);
    }

    Py_XDECREF(ns);
    Data_free(a); Data_free(b);
    return PyFloat_FromDouble(result);
}

/* Local mean Python wrapper */
static PyObject* py_mean_local(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *data_obj, *local_size_obj = NULL, *step_obj = NULL,
             *weight_obj = NULL,
             *output_obj = NULL, *output_start_obj = NULL, *output_step_obj = NULL,
             *iterate_obj = NULL, *transform1_obj = NULL, *transform2_obj = NULL;
        static char *kwlist[] = {"data", "local_size", "step", "weight", "output", "output_start", "output_step", "iterate", "transform1", "transform2", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOOOOOOOO", kwlist,
                                     &data_obj, &local_size_obj, &step_obj, &weight_obj,
                                     &output_obj, &output_start_obj, &output_step_obj,
                                     &iterate_obj, &transform1_obj, &transform2_obj))
        return NULL;

    /* iterate/transform paths: delegate to the pure Python reference implementation. */
    if ((iterate_obj != NULL && iterate_obj != Py_None) ||
        (transform1_obj != NULL && transform1_obj != Py_None) ||
        (transform2_obj != NULL && transform2_obj != Py_None)) {
        PyObject *pure = _import_pure_core(self);
        if (!pure) return NULL;
        PyObject *fn = PyObject_GetAttrString(pure, "mean_local");
        Py_DECREF(pure);
        if (!fn) return NULL;
        PyObject *result = PyObject_Call(fn, args, kwargs);
        Py_DECREF(fn);
        return result;
    }

    Data *data = pyobj_to_data(data_obj);
    if (!data) return NULL;
    int dim = data->dimension;

    int *local_size = NULL; int ls_dim = 0;
    /* Accept a single int as a 1-D window (consistent with pure Python) */
    if (local_size_obj && PyIndex_Check(local_size_obj)) {
        local_size = (int*)malloc(sizeof(int));
        if (!local_size) { Data_free(data); PyErr_NoMemory(); return NULL; }
        PyObject *ls_py = PyNumber_Index(local_size_obj);
        if (!ls_py) { free(local_size); Data_free(data); return NULL; }
        local_size[0] = (int)PyLong_AsLong(ls_py);
        Py_DECREF(ls_py);
        if (PyErr_Occurred()) { free(local_size); Data_free(data); return NULL; }
        ls_dim = 1;
    } else if (local_size_obj) {
        if (parse_int_seq(local_size_obj, &local_size, &ls_dim) < 0) { Data_free(data); return NULL; }
    } else {
        ls_dim = dim; local_size = (int*)malloc((size_t)(ls_dim) * sizeof(int));
        for (int i = 0; i < ls_dim; ++i) local_size[i] = 1;
    }
    if (ls_dim != dim) {
        free(local_size); Data_free(data);
        PyErr_Format(PyExc_ValueError, "local_size dimension %d does not match data dimension %d", ls_dim, dim);
        return NULL;
    }
    /* Optional per-element weights: convert to a flat double array (NULL = ones) */
    double *weights = NULL;
    if (weight_obj && weight_obj != Py_None) {
        Data *wdata = pyobj_to_data(weight_obj);
        if (!wdata) { free(local_size); Data_free(data); return NULL; }
        if (wdata->dimension != dim) {
            int wdim = wdata->dimension;
            Data_free(wdata); free(local_size); Data_free(data);
            PyErr_Format(PyExc_ValueError, "weight dimension %d must match data dimension %d", wdim, dim);
            return NULL;
        }
        int wtot = 1;
        for (int i = 0; i < dim; ++i) {
            if (wdata->shape[i] < local_size[i]) {
                Data_free(wdata); free(local_size); Data_free(data);
                PyErr_Format(PyExc_ValueError, "weight shape must be at least the local_size window");
                return NULL;
            }
            wtot *= local_size[i];
        }
        weights = (double*)malloc((size_t)(wtot) * sizeof(double));
        if (!weights) { Data_free(wdata); free(local_size); Data_free(data); PyErr_NoMemory(); return NULL; }
        for (int i = 0; i < wtot; ++i) weights[i] = Data_get_flat(wdata, i);
        Data_free(wdata);
    } else {
        /* One weight per window element (placeholder to keep single code path) */
        int wtot = 1;
        for (int i = 0; i < dim; ++i) wtot *= local_size[i];
        weights = (double*)malloc((size_t)(wtot) * sizeof(double));
        if (!weights) { free(local_size); Data_free(data); PyErr_NoMemory(); return NULL; }
        for (int i = 0; i < wtot; ++i) weights[i] = 1.0;
    }

    int *start = (int*)malloc((size_t)(dim > 0 ? dim : 1) * sizeof(int));
    if (!start) { free(local_size); free(weights); Data_free(data); PyErr_NoMemory(); return NULL; }
    for (int i = 0; i < dim; ++i) start[i] = 0;
    int *end = (int*)malloc((size_t)(dim > 0 ? dim : 1) * sizeof(int));
    if (!end) { free(start); free(local_size); free(weights); Data_free(data); PyErr_NoMemory(); return NULL; }
    for (int i = 0; i < dim; ++i) end[i] = data->shape[i];

    int *step = NULL;
    if (parse_opt_int_seq(step_obj, &step, dim, 1) < 0) { free(end); free(start); free(local_size); free(weights); Data_free(data); return NULL; }

    int *output_start = NULL;
    if (parse_opt_int_seq(output_start_obj, &output_start, dim, 0) < 0) { free(step); free(end); free(start); free(local_size); free(weights); Data_free(data); return NULL; }

    int *output_step = NULL;
    if (parse_opt_int_seq(output_step_obj, &output_step, dim, 1) < 0) { free(output_start); free(step); free(end); free(start); free(local_size); free(weights); Data_free(data); return NULL; }

    Data *result = NULL;
    // Save user's output_start/output_step and set to 0/1 for core call when output is provided
    int *saved_os = NULL, *saved_ost = NULL;
    if (output_obj && output_obj != Py_None) {
        saved_os = (int*)malloc((size_t)(dim) * sizeof(int));
        saved_ost = (int*)malloc((size_t)(dim) * sizeof(int));
        if (!saved_os || !saved_ost) {
            free(saved_os); free(saved_ost);
            PyErr_NoMemory();
            return NULL;
        }
        memcpy(saved_os, output_start, dim * sizeof(int));
        memcpy(saved_ost, output_step, dim * sizeof(int));
        for (int i = 0; i < dim; ++i) {
            output_start[i] = 0;
            output_step[i] = 1;
        }
    }
    if (!output_obj) {
        Py_BEGIN_ALLOW_THREADS
        result = cos_local_mean(data, local_size, start, end, step,
                                output_start, output_step, NULL, NULL, weights);
        Py_END_ALLOW_THREADS
    } else {
        result = cos_local_mean(data, local_size, start, end, step,
                                output_start, output_step, NULL, NULL, weights);
    }
    // Restore user's output_start/output_step
    if (saved_os) {
        memcpy(output_start, saved_os, dim * sizeof(int));
        memcpy(output_step, saved_ost, dim * sizeof(int));
        free(saved_os);
        free(saved_ost);
    }

    if (!result) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "effectless args.");
        // Free allocated memory before return
        free(output_step); free(output_start); free(step);
        free(end); free(start); free(local_size); free(weights); Data_free(data);
        return NULL;
    }

    PyObject *py_result;
    if (output_obj && output_obj != Py_None) {
        // Write result to user-provided output object
        int r_dim = result->dimension;
        int *idx = (int*)malloc((size_t)(r_dim) * sizeof(int));
        int *out_idx = (int*)malloc((size_t)(r_dim) * sizeof(int));
        if (!idx || !out_idx) { free(idx); free(out_idx); Data_free(result); free(output_step); free(output_start); free(step); free(end); free(start); free(local_size); free(weights); Data_free(data); PyErr_NoMemory(); return NULL; }
        // Check if output is our C Vector type for direct write
        Data *out_data = NULL;
        if (PyObject_IsInstance(output_obj, (PyObject*)&VectorizeType)) {
            Vector *vec = (Vector*)output_obj;
            out_data = vec->data;
        }
        for (int i = 0; i < r_dim; ++i) idx[i] = 0;
        if (out_data) {
            for (int i = 0; i < r_dim; ++i) {
                int last = output_start[i] + output_step[i] * (result->shape[i] - 1);
                if (output_start[i] < 0 || last >= out_data->shape[i]) {
                    free(idx); free(out_idx); Data_free(result);
                    free(output_step); free(output_start); free(step); free(end); free(start); free(local_size); free(weights); Data_free(data);
                    PyErr_SetString(PyExc_IndexError, "index out of range");
                    return NULL;
                }
            }
        }
        int flag = r_dim - 1;
        while (flag >= 0) {
            if (flag == r_dim - 1) {
                // Calculate output index
                for (int i = 0; i < r_dim; ++i) {
                    out_idx[i] = output_start[i] + output_step[i] * idx[i];
                }
                // Get value from result
                double val = Data_get(result, idx);
                // Write to output: direct Data write for Vector, else generic sequence
                if (out_data) {
                    Data_set(out_data, out_idx, val);
                } else {
                    py_set_item_value(output_obj, out_idx, r_dim, 0, val);
                }
                // Advance last dimension
                idx[flag]++;
                if (idx[flag] >= result->shape[flag]) {
                    idx[flag] = 0;
                    flag--;
                }
            } else {
                // Carry over
                idx[flag]++;
                if (idx[flag] >= result->shape[flag]) {
                    idx[flag] = 0;
                    flag--;
                } else {
                    flag = r_dim - 1;
                }
            }
        }
        free(idx); free(out_idx);
        Data_free(result);
        py_result = output_obj;
        Py_INCREF(py_result);
    } else {
        PyTypeObject *result_type = PyObject_IsInstance(data_obj, (PyObject*)&VectorizeType) ? Py_TYPE(data_obj) : NULL;
        py_result = data_to_vector(result, result_type);
    }

    // Free all allocated memory
    free(output_step); free(output_start); free(step);
    free(end); free(start); free(local_size); free(weights); Data_free(data);

    return py_result;
}

/* Local variance Python wrapper */
static PyObject* py_local_variance(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *data_obj, *local_size_obj = NULL, *step_obj = NULL,
             *output_obj = NULL, *output_start_obj = NULL, *output_step_obj = NULL,
             *iterate_obj = NULL, *transform1_obj = NULL, *transform2_obj = NULL;
        static char *kwlist[] = {"data", "local_size", "step", "output", "output_start", "output_step", "iterate", "transform1", "transform2", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOOOOOOO", kwlist,
                                     &data_obj, &local_size_obj, &step_obj,
                                     &output_obj, &output_start_obj, &output_step_obj,
                                     &iterate_obj, &transform1_obj, &transform2_obj))
        return NULL;

    /* iterate/transform paths: delegate to the pure Python reference implementation. */
    if ((iterate_obj != NULL && iterate_obj != Py_None) ||
        (transform1_obj != NULL && transform1_obj != Py_None) ||
        (transform2_obj != NULL && transform2_obj != Py_None)) {
        PyObject *pure = _import_pure_core(self);
        if (!pure) return NULL;
        PyObject *fn = PyObject_GetAttrString(pure, "local_variance");
        Py_DECREF(pure);
        if (!fn) return NULL;
        PyObject *result = PyObject_Call(fn, args, kwargs);
        Py_DECREF(fn);
        return result;
    }

    Data *data = pyobj_to_data(data_obj);
    if (!data) return NULL;
    int dim = data->dimension;

    int *local_size = NULL; int ls_dim = 0;
    if (local_size_obj && PyIndex_Check(local_size_obj)) {
        local_size = (int*)malloc(sizeof(int));
        if (!local_size) { Data_free(data); PyErr_NoMemory(); return NULL; }
        PyObject *ls_py = PyNumber_Index(local_size_obj);
        if (!ls_py) { free(local_size); Data_free(data); return NULL; }
        local_size[0] = (int)PyLong_AsLong(ls_py);
        Py_DECREF(ls_py);
        if (PyErr_Occurred()) { free(local_size); Data_free(data); return NULL; }
        ls_dim = 1;
    } else if (local_size_obj) {
        if (parse_int_seq(local_size_obj, &local_size, &ls_dim) < 0) { Data_free(data); return NULL; }
    } else {
        ls_dim = dim; local_size = (int*)malloc((size_t)(ls_dim) * sizeof(int));
        for (int i = 0; i < ls_dim; ++i) local_size[i] = 1;
    }
    if (ls_dim != dim) {
        free(local_size); Data_free(data);
        PyErr_Format(PyExc_ValueError, "local_size dimension %d does not match data dimension %d", ls_dim, dim);
        return NULL;
    }

    int *start = (int*)malloc((size_t)(dim > 0 ? dim : 1) * sizeof(int));
    if (!start) { free(local_size); Data_free(data); PyErr_NoMemory(); return NULL; }
    for (int i = 0; i < dim; ++i) start[i] = 0;
    int *end = (int*)malloc((size_t)(dim > 0 ? dim : 1) * sizeof(int));
    if (!end) { free(start); free(local_size); Data_free(data); PyErr_NoMemory(); return NULL; }
    for (int i = 0; i < dim; ++i) end[i] = data->shape[i];

    int *step = NULL;
    if (parse_opt_int_seq(step_obj, &step, dim, 1) < 0) { free(end); free(start); free(local_size); Data_free(data); return NULL; }

    int *output_start = NULL;
    if (parse_opt_int_seq(output_start_obj, &output_start, dim, 0) < 0) { free(step); free(end); free(start); free(local_size); Data_free(data); return NULL; }

    int *output_step = NULL;
    if (parse_opt_int_seq(output_step_obj, &output_step, dim, 1) < 0) { free(output_start); free(step); free(end); free(start); free(local_size); Data_free(data); return NULL; }

    Data *result = NULL;
    // Save user's output_start/output_step and set to 0/1 for core call when output is provided
    int *saved_os = NULL, *saved_ost = NULL;
    if (output_obj && output_obj != Py_None) {
        saved_os = (int*)malloc((size_t)(dim) * sizeof(int));
        saved_ost = (int*)malloc((size_t)(dim) * sizeof(int));
        if (!saved_os || !saved_ost) {
            free(saved_os); free(saved_ost);
            PyErr_NoMemory();
            return NULL;
        }
        memcpy(saved_os, output_start, dim * sizeof(int));
        memcpy(saved_ost, output_step, dim * sizeof(int));
        for (int i = 0; i < dim; ++i) {
            output_start[i] = 0;
            output_step[i] = 1;
        }
    }
    if (!output_obj) {
        Py_BEGIN_ALLOW_THREADS
        result = cos_local_variance(data, local_size, start, end, step,
                                    output_start, output_step, NULL, NULL);
        Py_END_ALLOW_THREADS
    } else {
        result = cos_local_variance(data, local_size, start, end, step,
                                    output_start, output_step, NULL, NULL);
    }
    // Restore user's output_start/output_step
    if (saved_os) {
        memcpy(output_start, saved_os, dim * sizeof(int));
        memcpy(output_step, saved_ost, dim * sizeof(int));
        free(saved_os);
        free(saved_ost);
    }

    if (!result) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "effectless args.");
        // Free allocated memory before return
        free(output_step); free(output_start); free(step);
        free(end); free(start); free(local_size); Data_free(data);
        return NULL;
    }

    PyObject *py_result;
    if (output_obj && output_obj != Py_None) {
        // Write result to user-provided output object
        int r_dim = result->dimension;
        int *idx = (int*)malloc((size_t)(r_dim) * sizeof(int));
        int *out_idx = (int*)malloc((size_t)(r_dim) * sizeof(int));
        if (!idx || !out_idx) { free(idx); free(out_idx); Data_free(result); free(output_step); free(output_start); free(step); free(end); free(start); free(local_size); Data_free(data); PyErr_NoMemory(); return NULL; }
        // Check if output is our C Vector type for direct write
        Data *out_data = NULL;
        if (PyObject_IsInstance(output_obj, (PyObject*)&VectorizeType)) {
            Vector *vec = (Vector*)output_obj;
            out_data = vec->data;
        }
        for (int i = 0; i < r_dim; ++i) idx[i] = 0;
        if (out_data) {
            for (int i = 0; i < r_dim; ++i) {
                int last = output_start[i] + output_step[i] * (result->shape[i] - 1);
                if (output_start[i] < 0 || last >= out_data->shape[i]) {
                    free(idx); free(out_idx); Data_free(result);
                    free(output_step); free(output_start); free(step); free(end); free(start); free(local_size); Data_free(data);
                    PyErr_SetString(PyExc_IndexError, "index out of range");
                    return NULL;
                }
            }
        }
        int flag = r_dim - 1;
        while (flag >= 0) {
            if (flag == r_dim - 1) {
                // Calculate output index
                for (int i = 0; i < r_dim; ++i) {
                    out_idx[i] = output_start[i] + output_step[i] * idx[i];
                }
                // Get value from result
                double val = Data_get(result, idx);
                // Write to output: direct Data write for Vector, else generic sequence
                if (out_data) {
                    Data_set(out_data, out_idx, val);
                } else {
                    py_set_item_value(output_obj, out_idx, r_dim, 0, val);
                }
                // Advance last dimension
                idx[flag]++;
                if (idx[flag] >= result->shape[flag]) {
                    idx[flag] = 0;
                    flag--;
                }
            } else {
                // Carry over
                idx[flag]++;
                if (idx[flag] >= result->shape[flag]) {
                    idx[flag] = 0;
                    flag--;
                } else {
                    flag = r_dim - 1;
                }
            }
        }
        free(idx); free(out_idx);
        Data_free(result);
        py_result = output_obj;
        Py_INCREF(py_result);
    } else {
        PyTypeObject *result_type = PyObject_IsInstance(data_obj, (PyObject*)&VectorizeType) ? Py_TYPE(data_obj) : NULL;
        py_result = data_to_vector(result, result_type);
    }

    // Free all allocated memory
    free(output_step); free(output_start); free(step);
    free(end); free(start); free(local_size); Data_free(data);

    return py_result;
}

/* ------------------------------------------------------------------
Vector optimized methods (unchanged)
------------------------------------------------------------------ */
static PyObject *Vector_cos_comparison_passive(PyObject *self, PyObject *args, PyObject *kwargs) {
    /* Vector passive wrapper (delegates to core loop) */
    Vector *vec = (Vector*)self;
    PyObject *window_size_obj = NULL;
    double w1 = 1.0, w2 = 1.0, b1 = 0.0, b2 = 0.0;
    PyObject *start_obj = NULL;
    PyObject *end_obj = NULL;
    PyObject *step_obj = NULL;
    PyObject *d_obj = NULL;
    PyObject *algo_name = NULL;
    PyObject *output_obj = NULL;
    PyObject *output_start_obj = NULL;
    PyObject *output_step_obj = NULL;
    PyObject *start_callback = NULL;
    PyObject *end_callback = NULL;
    PyObject *global_error_callback = NULL;
    PyObject *local_error_callback = NULL;
    PyObject *return_callback = NULL;
    int use_ns = 1;
    PyObject *hook_obj = NULL;
        static char *kwlist[] = {
        "window_size", "w1", "w2", "b1", "b2",
        "start", "end", "step", "d", "algorithm",
        "output", "output_start", "output_step",
        "start_callback", "end_callback",
        "global_error_callback", "local_error_callback",
        "return_callback",
        "use_namespace", "namespace_hook", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OddddOOOOOOOOOOOOOpO", kwlist,
                                     &window_size_obj,
                                     &w1, &w2, &b1, &b2,
                                     &start_obj, &end_obj, &step_obj, &d_obj,
                                     &algo_name, &output_obj,
                                     &output_start_obj, &output_step_obj,
                                     &start_callback, &end_callback,
                                     &global_error_callback, &local_error_callback,
                                     &return_callback, &use_ns, &hook_obj))
        return NULL;

    /* Create contiguous copy of view data for core computation */
    Data *contig_data = Data_create(vec->dimension, vec->shape);
    if (!contig_data) return NULL;
    long long total_ll = 1;
    for (int i = 0; i < vec->dimension; ++i) total_ll *= vec->shape[i];
    int total = (int)total_ll;
    int *indices = (int*)malloc((size_t)(total > 0 ? total : 1) * sizeof(int));
    if (!indices) { Data_free(contig_data); PyErr_NoMemory(); return NULL; }
    /* Iterate all indices with carry method */
    if (total > 0) {
        int *idx = (int*)malloc((size_t)(vec->dimension > 0 ? vec->dimension : 1) * sizeof(int));
        if (!idx) { free(indices); Data_free(contig_data); PyErr_NoMemory(); return NULL; }
        memset(idx, 0, vec->dimension * sizeof(int));
        int pos = 0;
        while (1) {
            int flat = vec->start + vec->offset;
            for (int i = 0; i < vec->dimension; ++i) {
                flat += vec->strides[i] * (vec->start_offset[i] + idx[i] * vec->step_offset[i]);
            }
            indices[pos++] = flat;
            int dim = vec->dimension - 1;
            while (dim >= 0) {
                idx[dim]++;
                if (idx[dim] < vec->shape[dim]) break;
                idx[dim] = 0;
                dim--;
            }
            if (dim < 0) break;
        }
        free(idx);
    }
    for (int i = 0; i < total; ++i) {
        Data_set_flat(contig_data, i, Data_get_flat(vec->data, indices[i]));
    }
    free(indices);
    
    Data *data = contig_data;
    int dim = vec->dimension;

    int *window_size = NULL;
    if (parse_opt_int_seq(window_size_obj, &window_size, dim, 1) < 0) { Data_free(contig_data); return NULL; }

    int *start = NULL;
    if (parse_opt_int_seq(start_obj, &start, dim, 0) < 0) { free(window_size); Data_free(contig_data); return NULL; }

    int *end = NULL;
    if (end_obj == NULL) {
        end = (int*)malloc((size_t)(dim > 0 ? dim : 1) * sizeof(int));
        if (!end) { Data_free(contig_data); PyErr_NoMemory(); return NULL; }
        for (int i = 0; i < dim; ++i) end[i] = data->shape[i];
    } else if (parse_opt_int_seq(end_obj, &end, dim, -1) < 0) {
        free(window_size); free(start); Data_free(contig_data); return NULL;
    }

    int *step = NULL;
    if (parse_opt_int_seq(step_obj, &step, dim, 1) < 0) { free(window_size); free(start); free(end); Data_free(contig_data); return NULL; }

    int *d = NULL;
    if (parse_opt_int_seq(d_obj, &d, dim, 0) < 0) { free(window_size); free(start); free(end); free(step); Data_free(contig_data); return NULL; }
    if (d_obj == NULL && dim > 0) d[0] = 1;

    algo_fn algo = get_algo(algo_name);
    if (!algo) {
        free(window_size); free(start); free(end); free(step); free(d);
        Data_free(contig_data);
        return NULL;
    }

    int *output_start = NULL;
    if (parse_opt_int_seq(output_start_obj, &output_start, dim, 0) < 0) { Data_free(contig_data); return NULL; }

    int *output_step = NULL;
    if (parse_opt_int_seq(output_step_obj, &output_step, dim, 1) < 0) { Data_free(contig_data); return NULL; }

    PyObject *name_space = NULL; CallbackContext ctx = {0};
    if (start_callback || end_callback || global_error_callback || local_error_callback || return_callback || PyCallable_Check(algo_name)) {
        name_space = PyObject_CallObject((PyObject*)&FuncNameSpaceType, NULL);
        if (name_space) {
            /* Fill attributes (same as py_passive) */
            if (output_obj) { Py_INCREF(output_obj); PyObject_SetAttrString(name_space, "output", output_obj); }
            PyObject *os_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(os_tuple, i, PyLong_FromLong(output_start[i]));
            PyObject_SetAttrString(name_space, "output_start", os_tuple); Py_DECREF(os_tuple);
            PyObject *ost_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(ost_tuple, i, PyLong_FromLong(output_step[i]));
            PyObject_SetAttrString(name_space, "output_step", ost_tuple); Py_DECREF(ost_tuple);
            PyObject *ws_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(ws_tuple, i, PyLong_FromLong(window_size[i]));
            PyObject_SetAttrString(name_space, "window_size", ws_tuple); Py_DECREF(ws_tuple);
            PyObject *start_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(start_tuple, i, PyLong_FromLong(start[i]));
            PyObject_SetAttrString(name_space, "start", start_tuple); Py_DECREF(start_tuple);
            PyObject *end_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(end_tuple, i, PyLong_FromLong(end[i]));
            PyObject_SetAttrString(name_space, "end", end_tuple); Py_DECREF(end_tuple);
            PyObject *d_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(d_tuple, i, PyLong_FromLong(d[i]));
            PyObject_SetAttrString(name_space, "d", d_tuple); Py_DECREF(d_tuple);
            PyObject *step_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(step_tuple, i, PyLong_FromLong(step[i]));
            PyObject_SetAttrString(name_space, "step", step_tuple); Py_DECREF(step_tuple);
            if (algo_name) { Py_INCREF(algo_name); PyObject_SetAttrString(name_space, "algorithm", algo_name); }
            int *num = (int*)malloc((size_t)(dim) * sizeof(int));
    if (!num) { PyErr_NoMemory(); return NULL; }
            for (int i = 0; i < dim; ++i) num[i] = (step[i] != 0) ? (end[i] - start[i] - window_size[i] - d[i]) / step[i] + 1 : 0;
            PyObject *num_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(num_tuple, i, PyLong_FromLong(num[i]));
            PyObject_SetAttrString(name_space, "num", num_tuple); Py_DECREF(num_tuple);
            free(num);
            ctx.local_error_callback = local_error_callback;
            ctx.name_space = name_space;
        }
    }

    if (start_callback && PyCallable_Check(start_callback) && name_space) {
        PyObject *res = PyObject_CallFunctionObjArgs(start_callback, name_space, NULL);
        Py_XDECREF(res);
    }

    Data *result = NULL;
    // Save user's output_start/output_step and set to 0/1 for core call when output is provided
    int *saved_os = NULL, *saved_ost = NULL;
    if (output_obj && output_obj != Py_None) {
        saved_os = (int*)malloc((size_t)(dim) * sizeof(int));
        saved_ost = (int*)malloc((size_t)(dim) * sizeof(int));
        if (!saved_os || !saved_ost) {
            free(saved_os); free(saved_ost);
            PyErr_NoMemory();
            return NULL;
        }
        memcpy(saved_os, output_start, dim * sizeof(int));
        memcpy(saved_ost, output_step, dim * sizeof(int));
        for (int i = 0; i < dim; ++i) {
            output_start[i] = 0;
            output_step[i] = 1;
        }
    }
    if (!name_space) {
        Py_BEGIN_ALLOW_THREADS
        result = cos_comparison_passive(data, window_size, w1, w2, b1, b2,
                                        start, end, step, d,
                                        algo, &ctx,
                                        output_start, output_step,
                                        NULL, NULL);
        Py_END_ALLOW_THREADS
    } else {
        result = cos_comparison_passive(data, window_size, w1, w2, b1, b2,
                                        start, end, step, d,
                                        algo, &ctx,
                                        output_start, output_step,
                                        NULL, NULL);
    }
    // Restore user's output_start/output_step
    if (saved_os) {
        memcpy(output_start, saved_os, dim * sizeof(int));
        memcpy(output_step, saved_ost, dim * sizeof(int));
        free(saved_os);
        free(saved_ost);
    }

    if (!result) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "effectless args.");
        if (global_error_callback && PyCallable_Check(global_error_callback) && name_space) {
            PyObject *exc = PyErr_Occurred();
            if (exc) { Py_INCREF(exc); PyErr_Clear(); PyObject *res = PyObject_CallFunctionObjArgs(global_error_callback, exc, name_space, NULL); Py_XDECREF(res);  Py_DECREF(exc); }
        }
        // Free allocated memory before return
        free(window_size); free(start); free(end); free(step); free(d);
        free(output_start); free(output_step);
        Py_XDECREF(name_space);
        return NULL;
    }

    PyObject *py_result;
    if (output_obj && output_obj != Py_None) {
        // Write result to user-provided output object
        int r_dim = result->dimension;
        int *idx = (int*)malloc((size_t)(r_dim) * sizeof(int));
        int *out_idx = (int*)malloc((size_t)(r_dim) * sizeof(int));
        if (!idx || !out_idx) { free(idx); free(out_idx); Data_free(result); free(output_step); free(output_start); free(d); free(step); free(end); free(start); free(window_size); Data_free(contig_data); Py_XDECREF(name_space); PyErr_NoMemory(); return NULL; }
        // Check if output is our C Vector type for direct write
        Data *out_data = NULL;
        if (PyObject_IsInstance(output_obj, (PyObject*)&VectorizeType)) {
            Vector *out_vec = (Vector*)output_obj;
            out_data = out_vec->data;
        }
        for (int i = 0; i < r_dim; ++i) idx[i] = 0;
        if (out_data) {
            for (int i = 0; i < r_dim; ++i) {
                int last = output_start[i] + output_step[i] * (result->shape[i] - 1);
                if (output_start[i] < 0 || last >= out_data->shape[i]) {
                    free(idx); free(out_idx); Data_free(result);
                    free(output_step); free(output_start); free(d); free(step); free(end); free(start); free(window_size); Data_free(contig_data); Py_XDECREF(name_space);
                    PyErr_SetString(PyExc_IndexError, "index out of range");
                    return NULL;
                }
            }
        }
        int flag = r_dim - 1;
        while (flag >= 0) {
            if (flag == r_dim - 1) {
                // Calculate output index
                for (int i = 0; i < r_dim; ++i) {
                    out_idx[i] = output_start[i] + output_step[i] * idx[i];
                }
                // Get value from result
                double val = Data_get(result, idx);
                // Write to output: direct Data write for Vector, else generic sequence
                if (out_data) {
                    Data_set(out_data, out_idx, val);
                } else {
                    py_set_item_value(output_obj, out_idx, r_dim, 0, val);
                }
                // Advance last dimension
                idx[flag]++;
                if (idx[flag] >= result->shape[flag]) {
                    idx[flag] = 0;
                    flag--;
                }
            } else {
                // Carry over
                idx[flag]++;
                if (idx[flag] >= result->shape[flag]) {
                    idx[flag] = 0;
                    flag--;
                } else {
                    flag = r_dim - 1;
                }
            }
        }
        free(idx); free(out_idx);
        Data_free(result);
        py_result = output_obj;
        Py_INCREF(py_result);
    } else {
        py_result = data_to_vector(result, Py_TYPE(self));
    }

    if (end_callback && PyCallable_Check(end_callback) && name_space) {
        PyObject_SetAttrString(name_space, "output", py_result);
        PyObject *res = PyObject_CallFunctionObjArgs(end_callback, name_space, NULL);
        Py_XDECREF(res);
    }

    if (return_callback && PyCallable_Check(return_callback) && name_space) {
        PyObject *ret = PyObject_CallFunctionObjArgs(return_callback, py_result, name_space, NULL);
        Py_DECREF(py_result);
        py_result = ret;
    }

    // Free all allocated memory
    free(window_size); free(start); free(end); free(step); free(d);
    free(output_start); free(output_step);
    Data_free(contig_data);

    Py_XDECREF(name_space);
    return py_result;
}

static PyObject *Vector_cos_comparison_active(PyObject *self, PyObject *args, PyObject *kwargs) {
    /* Vector active wrapper (delegates to core loop) */
    Vector *vec = (Vector*)self;

    PyObject *kernel_obj = NULL;
    double w1 = 1.0, w2 = 1.0, b1 = 0.0, b2 = 0.0;
    PyObject *start_obj = NULL;
    PyObject *end_obj = NULL;
    PyObject *step_obj = NULL;
    PyObject *algo_name = NULL;
    PyObject *output_obj = NULL;
    PyObject *output_start_obj = NULL;
    PyObject *output_step_obj = NULL;
    PyObject *start_callback = NULL;
    PyObject *end_callback = NULL;
    PyObject *global_error_callback = NULL;
    PyObject *local_error_callback = NULL;
    PyObject *return_callback = NULL;
    int use_ns = 1;
    PyObject *hook_obj = NULL;
        static char *kwlist[] = {
        "kernel", "w1", "w2", "b1", "b2",
        "start", "end", "step", "algorithm",
        "output", "output_start", "output_step",
        "start_callback", "end_callback",
        "global_error_callback", "local_error_callback",
        "return_callback",
        "use_namespace", "namespace_hook", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ddddOOOOOOOOOOOOpO", kwlist,
                                     &kernel_obj,
                                     &w1, &w2, &b1, &b2,
                                     &start_obj, &end_obj, &step_obj,
                                     &algo_name, &output_obj,
                                     &output_start_obj, &output_step_obj,
                                     &start_callback, &end_callback,
                                     &global_error_callback, &local_error_callback,
                                     &return_callback, &use_ns, &hook_obj))
        return NULL;

    if (!kernel_obj) { PyErr_SetString(PyExc_ValueError, "kernel must be provided for active mode"); return NULL; }

    /* Create contiguous copy of view data for core computation */
    Data *contig_data = Data_create(vec->dimension, vec->shape);
    if (!contig_data) return NULL;
    long long total_ll = 1;
    for (int i = 0; i < vec->dimension; ++i) total_ll *= vec->shape[i];
    int total = (int)total_ll;
    int *indices = (int*)malloc((size_t)(total > 0 ? total : 1) * sizeof(int));
    if (!indices) { Data_free(contig_data); PyErr_NoMemory(); return NULL; }
    /* Iterate all indices with carry method */
    if (total > 0) {
        int *idx = (int*)malloc((size_t)(vec->dimension > 0 ? vec->dimension : 1) * sizeof(int));
        if (!idx) { free(indices); Data_free(contig_data); PyErr_NoMemory(); return NULL; }
        memset(idx, 0, vec->dimension * sizeof(int));
        int pos = 0;
        while (1) {
            int flat = vec->start + vec->offset;
            for (int i = 0; i < vec->dimension; ++i) {
                flat += vec->strides[i] * (vec->start_offset[i] + idx[i] * vec->step_offset[i]);
            }
            indices[pos++] = flat;
            int dim = vec->dimension - 1;
            while (dim >= 0) {
                idx[dim]++;
                if (idx[dim] < vec->shape[dim]) break;
                idx[dim] = 0;
                dim--;
            }
            if (dim < 0) break;
        }
        free(idx);
    }
    for (int i = 0; i < total; ++i) {
        Data_set_flat(contig_data, i, Data_get_flat(vec->data, indices[i]));
    }
    free(indices);
    
    Data *data = contig_data;
    int dim = vec->dimension;
    Data *kernel = pyobj_to_data(kernel_obj);
    if (!kernel) { Data_free(contig_data); return NULL; }

    int *start = NULL;
    if (parse_opt_int_seq(start_obj, &start, dim, 0) < 0) { Data_free(kernel); Data_free(contig_data); return NULL; }

    int *end = NULL;
    if (end_obj == NULL) {
        end = (int*)malloc((size_t)(dim > 0 ? dim : 1) * sizeof(int));
        if (!end) { Data_free(kernel); Data_free(contig_data); PyErr_NoMemory(); return NULL; }
        for (int i = 0; i < dim; ++i) end[i] = data->shape[i];
    } else if (parse_opt_int_seq(end_obj, &end, dim, -1) < 0) {
        free(start); Data_free(kernel); Data_free(contig_data); return NULL;
    }

    int *step = NULL;
    if (parse_opt_int_seq(step_obj, &step, dim, 1) < 0) { free(start); free(end); Data_free(kernel); Data_free(contig_data); return NULL; }

    algo_fn algo = get_algo(algo_name);
    if (!algo) { free(start); free(end); free(step); Data_free(kernel); Data_free(contig_data); return NULL; }

    int *output_start = NULL;
    if (parse_opt_int_seq(output_start_obj, &output_start, dim, 0) < 0) { free(start); free(end); free(step); Data_free(kernel); Data_free(contig_data); return NULL; }

    int *output_step = NULL;
    if (parse_opt_int_seq(output_step_obj, &output_step, dim, 1) < 0) { free(start); free(end); free(step); Data_free(kernel); Data_free(contig_data); return NULL; }

    PyObject *name_space = NULL; CallbackContext ctx = {0};
    if (start_callback || end_callback || global_error_callback || local_error_callback || return_callback || PyCallable_Check(algo_name)) {
        name_space = PyObject_CallObject((PyObject*)&FuncNameSpaceType, NULL);
        if (name_space) {
            /* Fill attributes similarly */
            if (output_obj) { Py_INCREF(output_obj); PyObject_SetAttrString(name_space, "output", output_obj); }
            PyObject *os_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(os_tuple, i, PyLong_FromLong(output_start[i]));
            PyObject_SetAttrString(name_space, "output_start", os_tuple); Py_DECREF(os_tuple);
            PyObject *ost_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(ost_tuple, i, PyLong_FromLong(output_step[i]));
            PyObject_SetAttrString(name_space, "output_step", ost_tuple); Py_DECREF(ost_tuple);
            PyObject *ws_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(ws_tuple, i, PyLong_FromLong(kernel->shape[i]));
            PyObject_SetAttrString(name_space, "window_size", ws_tuple); Py_DECREF(ws_tuple);
            Py_INCREF(kernel_obj); PyObject_SetAttrString(name_space, "kernel", kernel_obj);
            PyObject *start_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(start_tuple, i, PyLong_FromLong(start[i]));
            PyObject_SetAttrString(name_space, "start", start_tuple); Py_DECREF(start_tuple);
            PyObject *end_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(end_tuple, i, PyLong_FromLong(end[i]));
            PyObject_SetAttrString(name_space, "end", end_tuple); Py_DECREF(end_tuple);
            PyObject *step_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(step_tuple, i, PyLong_FromLong(step[i]));
            PyObject_SetAttrString(name_space, "step", step_tuple); Py_DECREF(step_tuple);
            if (algo_name) { Py_INCREF(algo_name); PyObject_SetAttrString(name_space, "algorithm", algo_name); }
            int *num = (int*)malloc((size_t)(dim) * sizeof(int));
    if (!num) { PyErr_NoMemory(); return NULL; }
            for (int i = 0; i < dim; ++i) num[i] = (step[i] != 0) ? (end[i] - start[i] - kernel->shape[i]) / step[i] + 1 : 0;
            PyObject *num_tuple = PyTuple_New(dim);
            for (int i = 0; i < dim; ++i) PyTuple_SET_ITEM(num_tuple, i, PyLong_FromLong(num[i]));
            PyObject_SetAttrString(name_space, "num", num_tuple); Py_DECREF(num_tuple);
            free(num);
            ctx.local_error_callback = local_error_callback;
            ctx.name_space = name_space;
        }
    }

    if (start_callback && PyCallable_Check(start_callback) && name_space) {
        PyObject *res = PyObject_CallFunctionObjArgs(start_callback, name_space, NULL);
        Py_XDECREF(res);
    }

    Data *result = NULL;
    // Save user's output_start/output_step and set to 0/1 for core call when output is provided
    int *saved_os = NULL, *saved_ost = NULL;
    if (output_obj && output_obj != Py_None) {
        saved_os = (int*)malloc((size_t)(dim) * sizeof(int));
        saved_ost = (int*)malloc((size_t)(dim) * sizeof(int));
        if (!saved_os || !saved_ost) {
            free(saved_os); free(saved_ost);
            PyErr_NoMemory();
            return NULL;
        }
        memcpy(saved_os, output_start, dim * sizeof(int));
        memcpy(saved_ost, output_step, dim * sizeof(int));
        for (int i = 0; i < dim; ++i) {
            output_start[i] = 0;
            output_step[i] = 1;
        }
    }
    if (!name_space) {
        Py_BEGIN_ALLOW_THREADS
        result = cos_comparison_active(data, kernel, w1, w2, b1, b2,
                                       start, end, step,
                                       algo, &ctx,
                                       output_start, output_step,
                                       NULL, NULL);
        Py_END_ALLOW_THREADS
    } else {
        result = cos_comparison_active(data, kernel, w1, w2, b1, b2,
                                       start, end, step,
                                       algo, &ctx,
                                       output_start, output_step,
                                       NULL, NULL);
    }
    // Restore user's output_start/output_step
    if (saved_os) {
        memcpy(output_start, saved_os, dim * sizeof(int));
        memcpy(output_step, saved_ost, dim * sizeof(int));
        free(saved_os);
        free(saved_ost);
    }

    if (!result) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "effectless args.");
        if (global_error_callback && PyCallable_Check(global_error_callback) && name_space) {
            PyObject *exc = PyErr_Occurred();
            if (exc) { Py_INCREF(exc); PyErr_Clear(); PyObject *res = PyObject_CallFunctionObjArgs(global_error_callback, exc, name_space, NULL); Py_XDECREF(res);  Py_DECREF(exc); }
        }
        // Free allocated memory before return
        Data_free(kernel); free(start); free(end); free(step);
        free(output_start); free(output_step);
        Py_XDECREF(name_space);
        return NULL;
    }

    PyObject *py_result;
    if (output_obj && output_obj != Py_None) {
        // Write result to user-provided output object
        int r_dim = result->dimension;
        int *idx = (int*)malloc((size_t)(r_dim) * sizeof(int));
        int *out_idx = (int*)malloc((size_t)(r_dim) * sizeof(int));
        if (!idx || !out_idx) { free(idx); free(out_idx); Data_free(result); free(output_step); free(output_start); free(step); free(end); free(start); Data_free(contig_data); Data_free(kernel); Py_XDECREF(name_space); PyErr_NoMemory(); return NULL; }
        // Check if output is our C Vector type for direct write
        Data *out_data = NULL;
        if (PyObject_IsInstance(output_obj, (PyObject*)&VectorizeType)) {
            Vector *out_vec = (Vector*)output_obj;
            out_data = out_vec->data;
        }
        for (int i = 0; i < r_dim; ++i) idx[i] = 0;
        if (out_data) {
            for (int i = 0; i < r_dim; ++i) {
                int last = output_start[i] + output_step[i] * (result->shape[i] - 1);
                if (output_start[i] < 0 || last >= out_data->shape[i]) {
                    free(idx); free(out_idx); Data_free(result);
                    free(output_step); free(output_start); free(step); free(end); free(start); Data_free(contig_data); Data_free(kernel); Py_XDECREF(name_space);
                    PyErr_SetString(PyExc_IndexError, "index out of range");
                    return NULL;
                }
            }
        }
        int flag = r_dim - 1;
        while (flag >= 0) {
            if (flag == r_dim - 1) {
                // Calculate output index
                for (int i = 0; i < r_dim; ++i) {
                    out_idx[i] = output_start[i] + output_step[i] * idx[i];
                }
                // Get value from result
                double val = Data_get(result, idx);
                // Write to output: direct Data write for Vector, else generic sequence
                if (out_data) {
                    Data_set(out_data, out_idx, val);
                } else {
                    py_set_item_value(output_obj, out_idx, r_dim, 0, val);
                }
                // Advance last dimension
                idx[flag]++;
                if (idx[flag] >= result->shape[flag]) {
                    idx[flag] = 0;
                    flag--;
                }
            } else {
                // Carry over
                idx[flag]++;
                if (idx[flag] >= result->shape[flag]) {
                    idx[flag] = 0;
                    flag--;
                } else {
                    flag = r_dim - 1;
                }
            }
        }
        free(idx); free(out_idx);
        Data_free(result);
        py_result = output_obj;
        Py_INCREF(py_result);
    } else {
        py_result = data_to_vector(result, Py_TYPE(self));
    }

    if (end_callback && PyCallable_Check(end_callback) && name_space) {
        PyObject_SetAttrString(name_space, "output", py_result);
        PyObject *res = PyObject_CallFunctionObjArgs(end_callback, name_space, NULL);
        Py_XDECREF(res);
    }

    if (return_callback && PyCallable_Check(return_callback) && name_space) {
        PyObject *ret = PyObject_CallFunctionObjArgs(return_callback, py_result, name_space, NULL);
        Py_DECREF(py_result);
        py_result = ret;
    }

    // Free all allocated memory
    Data_free(kernel); free(start); free(end); free(step);
    free(output_start); free(output_step);
    Data_free(contig_data);

    Py_XDECREF(name_space);
    return py_result;
}

/* ------------------------------------------------------------------
VectorChainCompute native type (replaces PyRun_String Python closure)
------------------------------------------------------------------ */
typedef struct {
    PyObject_HEAD
    PyObject *a;
} VectorChainComputeObject;

static void VectorChainCompute_dealloc(VectorChainComputeObject *self) {
    Py_XDECREF(self->a);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// compute method: returns tuple of dot products
static PyObject* VectorChainCompute_compute(VectorChainComputeObject *self, PyObject *vector) {
    PyObject *a = self->a;
    if (!a || !vector) {
        PyErr_SetString(PyExc_ValueError, "Invalid state");
        return NULL;
    }
    Py_ssize_t leng = PyObject_Length(a);
    if (leng < 0) return NULL;
    PyObject *result = PyTuple_New(leng);
    if (!result) return NULL;

    for (Py_ssize_t i = 0; i < leng; i++) {
        PyObject *row = PySequence_GetItem(a, i);
        if (!row) { Py_DECREF(result); return NULL; }
        PyObject *vec_iter = PyObject_GetIter(vector);
        PyObject *row_iter = PyObject_GetIter(row);
        if (!vec_iter || !row_iter) {
            Py_XDECREF(vec_iter); Py_XDECREF(row_iter); Py_DECREF(row);
            Py_DECREF(result);
            return NULL;
        }
        double dot = 0.0;
        PyObject *v_item = NULL, *r_item = NULL;
        while ((v_item = PyIter_Next(vec_iter)) && (r_item = PyIter_Next(row_iter))) {
            double v = PyFloat_AsDouble(v_item);
            double r = PyFloat_AsDouble(r_item);
            dot += v * r;
            Py_DECREF(v_item); Py_DECREF(r_item);
        }
        Py_XDECREF(v_item); Py_XDECREF(r_item);
        Py_DECREF(vec_iter); Py_DECREF(row_iter);
        Py_DECREF(row);
        if (PyErr_Occurred()) { Py_DECREF(result); return NULL; }
        PyTuple_SET_ITEM(result, i, PyFloat_FromDouble(dot));
    }
    return result;
}

// fix method: update internal a state
static PyObject* VectorChainCompute_fix(VectorChainComputeObject *self, PyObject *new_a) {
    PyObject *tmp = self->a;
    Py_INCREF(new_a);
    self->a = new_a;
    Py_XDECREF(tmp);
    Py_RETURN_NONE;
}

// get method: return current a
static PyObject* VectorChainCompute_get(VectorChainComputeObject *self, PyObject *args) {
    (void)args;
    if (self->a) {
        Py_INCREF(self->a);
        return self->a;
    }
    Py_RETURN_NONE;
}

static PyMethodDef VectorChainCompute_methods[] = {
    {"compute", (PyCFunction)VectorChainCompute_compute, METH_O,
     "Compute dot products between input vector and each row of internal state."},
    {"fix", (PyCFunction)VectorChainCompute_fix, METH_O,
     "Update internal state to new value."},
    {"get", (PyCFunction)VectorChainCompute_get, METH_NOARGS,
     "Return current internal state."},
    {NULL}
};

static PyTypeObject VectorChainComputeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "cos_comparison_pydll.VectorChainCompute",
    .tp_basicsize = sizeof(VectorChainComputeObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)VectorChainCompute_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Native chain compute state object",
    .tp_methods = VectorChainCompute_methods,
    .tp_new = PyType_GenericNew,
};

// vector_chain_compute factory function
static PyMethodDef vcc_compute_def = {"compute", (PyCFunction)VectorChainCompute_compute, METH_O, NULL};
static PyMethodDef vcc_fix_def = {"fix", (PyCFunction)VectorChainCompute_fix, METH_O, NULL};
static PyMethodDef vcc_get_def = {"get", (PyCFunction)VectorChainCompute_get, METH_NOARGS, NULL};

static PyObject* py_vector_chain_compute(PyObject *self, PyObject *A) {
    (void)self;
    VectorChainComputeObject *obj = PyObject_New(VectorChainComputeObject, &VectorChainComputeType);
    if (!obj) return NULL;
    obj->a = A;
    Py_INCREF(A);

    // Create bound C functions directly (no need to fetch from tp_dict)
    PyObject *compute = PyCFunction_New(&vcc_compute_def, (PyObject*)obj);
    PyObject *fix = PyCFunction_New(&vcc_fix_def, (PyObject*)obj);
    PyObject *get = PyCFunction_New(&vcc_get_def, (PyObject*)obj);
    if (!compute || !fix || !get) {
        Py_XDECREF(compute); Py_XDECREF(fix); Py_XDECREF(get);
        Py_DECREF(obj);
        return NULL;
    }
    PyObject *result = PyTuple_Pack(3, compute, fix, get);
    Py_DECREF(compute); Py_DECREF(fix); Py_DECREF(get);
    Py_DECREF(obj);
    return result;
}

/* ------------------------------------------------------------------
 * data_filter / data_mapping: callback-based element walk over a sampled
 * region.  Callbacks are stateless (value only); errors are silently
 * skipped.  Iterative odometer walk, never recursive.  All arrays are
 * malloc'd and freed on every path.
 * ------------------------------------------------------------------ */

static int check_region_len(PyObject *obj, int dimension, const char *name) {
    if (obj == NULL || obj == Py_None) return 0;
    if (!PyTuple_Check(obj) || PyTuple_Size(obj) != dimension) {
        PyErr_Format(PyExc_ValueError, "%s length does not match data dimension", name);
        return -1;
    }
    return 0;
}

/* Resolve the sampled read region (start/shape/step -> effective sizes and
   element total), with the same clipping semantics as load_data's source
   side.  Allocates r_start/r_step/effective (caller frees each).  Returns 0
   on success, -1 with a Python exception set otherwise.  data_shape is
   const: the caller's shape buffer is never modified. */
static int resolve_read_region(PyObject *start, PyObject *shape_obj,
                                PyObject *step, const int *data_shape,
                                int dimension,
                                int **r_start_out, int **r_step_out,
                                int **effective_out, long long *total_out) {
    int *r_start = NULL, *r_step = NULL, *req_shape = NULL;
    if (parse_opt_int_seq(start, &r_start, dimension, 0) < 0
        || parse_opt_int_seq(step, &r_step, dimension, 1) < 0
        || parse_opt_int_seq(shape_obj, &req_shape, dimension, -1) < 0) {
        free(r_start); free(r_step); free(req_shape);
        return -1;
    }
    for (int i = 0; i < dimension; ++i) {
        if (r_start[i] < 0) {
            PyErr_SetString(PyExc_ValueError, "start entries must be non-negative");
            free(r_start); free(r_step); free(req_shape);
            return -1;
        }
        if (r_step[i] <= 0) {
            PyErr_SetString(PyExc_ValueError, "step entries must be positive");
            free(r_start); free(r_step); free(req_shape);
            return -1;
        }
    }
    int *effective = (int*)malloc((size_t)(dimension > 0 ? dimension : 1) * sizeof(int));
    if (!effective) {
        free(r_start); free(r_step); free(req_shape);
        PyErr_NoMemory();
        return -1;
    }
    long long total = 1;
    for (int i = 0; i < dimension; ++i) {
        int avail = (r_start[i] < data_shape[i])
            ? (int)((long long)(data_shape[i] - r_start[i] + r_step[i] - 1) / r_step[i]) : 0;
        int n = (req_shape[i] >= 0) ? req_shape[i] : avail;
        effective[i] = (n < avail) ? n : avail;
        total *= (long long)effective[i];
    }
    free(req_shape);
    *r_start_out = r_start;
    *r_step_out = r_step;
    *effective_out = effective;
    *total_out = total;
    return 0;
}

/* ==================================================================
 * Optional iterate path: a module-level C kernel (PyObject* in /
 * PyObject* out) handed to custom iterators.  One implementation with a
 * mode selector carried on the PyCFunction self argument; the five
 * PyMethodDef entries below name the per-function kernels - they are
 * NOT registered in the module method table.
 * ================================================================== */

static PyObject *g_module = NULL;   /* module object for internal calls */

static PyObject* _ints_to_tuple(const int *values, int dimension) {
    PyObject *tuple = PyTuple_New(dimension);
    if (tuple == NULL) return NULL;
    for (int i = 0; i < dimension; ++i) {
        PyObject *item = PyLong_FromLong(values[i]);
        if (item == NULL) { Py_DECREF(tuple); return NULL; }
        PyTuple_SET_ITEM(tuple, i, item);
    }
    return tuple;
}

static PyObject* _dbls_to_tuple(const double *values, int dimension) {
    PyObject *tuple = PyTuple_New(dimension);
    if (tuple == NULL) return NULL;
    for (int i = 0; i < dimension; ++i) {
        PyObject *item = PyFloat_FromDouble(values[i]);
        if (item == NULL) { Py_DECREF(tuple); return NULL; }
        PyTuple_SET_ITEM(tuple, i, item);
    }
    return tuple;
}

static int _parse_int_tuple(PyObject *obj, int **out, int dimension) {
    if (obj == NULL || !PySequence_Check(obj)) return -1;
    if (PySequence_Size(obj) != dimension) return -1;
    int *values = (int*)malloc(sizeof(int) *
                               (size_t)(dimension > 0 ? dimension : 1));
    if (values == NULL) return -1;
    for (int i = 0; i < dimension; ++i) {
        PyObject *item = PySequence_GetItem(obj, i);
        if (item == NULL) { free(values); return -1; }
        long v = PyLong_AsLong(item);
        Py_DECREF(item);
        if (v == -1 && PyErr_Occurred()) { free(values); return -1; }
        values[i] = (int)v;
    }
    *out = values;
    return 0;
}

static int _parse_dbl_tuple(PyObject *obj, double **out, int dimension) {
    if (obj == NULL || !PySequence_Check(obj)) return -1;
    if (PySequence_Size(obj) != dimension) return -1;
    double *values = (double*)malloc(sizeof(double) *
                                     (size_t)(dimension > 0 ? dimension : 1));
    if (values == NULL) return -1;
    for (int i = 0; i < dimension; ++i) {
        PyObject *item = PySequence_GetItem(obj, i);
        if (item == NULL) { free(values); return -1; }
        double v = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (v == -1.0 && PyErr_Occurred()) { free(values); return -1; }
        values[i] = v;
    }
    *out = values;
    return 0;
}

static PyObject* _make_kernel(PyMethodDef *def, long mode) {
    PyObject *mode_obj = PyLong_FromLong(mode);
    if (mode_obj == NULL) return NULL;
    PyObject *kernel = PyCFunction_New(def, mode_obj);
    Py_DECREF(mode_obj);
    return kernel;
}

/* mode: 0 data_mapping, 1 data_filter, 2 elementwise,
 *       3 position_map, 4 elementwise_position */
static PyObject* _iterate_kernel(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
    long mode = PyLong_AsLong(self);
    if (mode == -1 && PyErr_Occurred()) return NULL;
    PyObject *index_obj = NULL;
    if (!PyArg_ParseTuple(args, "O", &index_obj)) return NULL;
    Py_ssize_t dim_ss = PySequence_Size(index_obj);
    if (dim_ss < 0) {
        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError, "index must be a sequence");
        return NULL;
    }
    int dimension = (int)dim_ss;
    int *index = NULL;
    if (_parse_int_tuple(index_obj, &index, dimension) < 0) {
        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError, "index must be an int sequence");
        return NULL;
    }

    if (mode == 0) {
        /* data_mapping */
        PyObject *data = PyDict_GetItemString(kwargs, "data");
        PyObject *callback = PyDict_GetItemString(kwargs, "callback");
        PyObject *out = PyDict_GetItemString(kwargs, "out");
        int *r_start = NULL, *r_step = NULL;
        int *w_start = NULL, *w_step = NULL, *out_shape = NULL;
        int ok = (data != NULL && callback != NULL && out != NULL
            && _parse_int_tuple(PyDict_GetItemString(kwargs, "r_start"),
                                &r_start, dimension) == 0
            && _parse_int_tuple(PyDict_GetItemString(kwargs, "r_step"),
                                &r_step, dimension) == 0
            && _parse_int_tuple(PyDict_GetItemString(kwargs, "out_start"),
                                &w_start, dimension) == 0
            && _parse_int_tuple(PyDict_GetItemString(kwargs, "out_step"),
                                &w_step, dimension) == 0
            && _parse_int_tuple(PyDict_GetItemString(kwargs, "out_shape"),
                                &out_shape, dimension) == 0);
        if (!ok) {
            PyErr_Clear();
            PyErr_SetString(PyExc_TypeError,
                            "data_mapping kernel: bad arguments");
            free(index); free(r_start); free(r_step);
            free(w_start); free(w_step); free(out_shape);
            return NULL;
        }
        PyObject *read = PyTuple_New(dimension);
        for (int i = 0; i < dimension; ++i)
            PyTuple_SET_ITEM(read, i, PyLong_FromLong(
                (long long)r_start[i] + (long long)index[i] * r_step[i]));
        PyObject *get_args = PyTuple_Pack(2, data, read);
        Py_DECREF(read);
        PyObject *value = get_args ? py_get_item(g_module, get_args) : NULL;
        Py_XDECREF(get_args);
        if (value == NULL) {
            free(index); free(r_start); free(r_step);
            free(w_start); free(w_step); free(out_shape);
            return NULL;
        }
        PyObject *cb_res = PyObject_CallFunctionObjArgs(callback, value, NULL);
        Py_DECREF(value);
        if (cb_res == NULL) {
            PyErr_Clear();  /* callback errors silently skipped */
        } else {
            int in_bounds = 1;
            for (int i = 0; i < dimension; ++i) {
                long w = (long long)w_start[i]
                         + (long long)index[i] * w_step[i];
                if (w >= out_shape[i]) { in_bounds = 0; break; }
            }
            if (in_bounds) {
                PyObject *write = PyTuple_New(dimension);
                for (int i = 0; i < dimension; ++i)
                    PyTuple_SET_ITEM(write, i, PyLong_FromLong(
                        (long long)w_start[i]
                        + (long long)index[i] * w_step[i]));
                PyObject *set_args = PyTuple_Pack(3, out, write, cb_res);
                Py_DECREF(write);
                PyObject *r = set_args ? py_set_item(g_module, set_args)
                                       : NULL;
                Py_XDECREF(set_args);
                Py_DECREF(cb_res);
                if (r == NULL) {
                    free(index); free(r_start); free(r_step);
                    free(w_start); free(w_step); free(out_shape);
                    return NULL;
                }
                Py_DECREF(r);
            } else {
                Py_DECREF(cb_res);
            }
        }
        free(index); free(r_start); free(r_step);
        free(w_start); free(w_step); free(out_shape);
        Py_RETURN_NONE;
    } else if (mode == 1) {
        /* data_filter */
        PyObject *data = PyDict_GetItemString(kwargs, "data");
        PyObject *callback = PyDict_GetItemString(kwargs, "callback");
        PyObject *hits = PyDict_GetItemString(kwargs, "hits");
        int *r_start = NULL, *r_step = NULL;
        int *origin = NULL, *basis = NULL;
        int ok = (data != NULL && callback != NULL && hits != NULL
            && _parse_int_tuple(PyDict_GetItemString(kwargs, "r_start"),
                                &r_start, dimension) == 0
            && _parse_int_tuple(PyDict_GetItemString(kwargs, "r_step"),
                                &r_step, dimension) == 0
            && _parse_int_tuple(PyDict_GetItemString(kwargs, "origin"),
                                &origin, dimension) == 0
            && _parse_int_tuple(PyDict_GetItemString(kwargs, "basis"),
                                &basis, dimension) == 0);
        if (!ok) {
            PyErr_Clear();
            PyErr_SetString(PyExc_TypeError,
                            "data_filter kernel: bad arguments");
            free(index); free(r_start); free(r_step);
            free(origin); free(basis);
            return NULL;
        }
        PyObject *read = PyTuple_New(dimension);
        for (int i = 0; i < dimension; ++i)
            PyTuple_SET_ITEM(read, i, PyLong_FromLong(
                (long long)r_start[i] + (long long)index[i] * r_step[i]));
        PyObject *get_args = PyTuple_Pack(2, data, read);
        Py_DECREF(read);
        PyObject *value = get_args ? py_get_item(g_module, get_args) : NULL;
        Py_XDECREF(get_args);
        if (value == NULL) {
            free(index); free(r_start); free(r_step);
            free(origin); free(basis);
            return NULL;
        }
        PyObject *cb_res = PyObject_CallFunctionObjArgs(callback, value, NULL);
        Py_DECREF(value);
        if (cb_res == NULL) {
            PyErr_Clear();  /* callback errors silently skipped */
        } else {
            int truthy = PyObject_IsTrue(cb_res);
            Py_DECREF(cb_res);
            if (truthy < 0) {
                PyErr_Clear();
                truthy = 0;
            }
            if (truthy) {
                PyObject *pos = PyTuple_New(dimension);
                for (int i = 0; i < dimension; ++i)
                    PyTuple_SET_ITEM(pos, i, PyLong_FromLong(
                        (long long)origin[i]
                        + (long long)basis[i] * index[i]));
                int app = PyList_Append(hits, pos);
                Py_DECREF(pos);
                if (app < 0) {
                    free(index); free(r_start); free(r_step);
                    free(origin); free(basis);
                    return NULL;
                }
            }
        }
        free(index); free(r_start); free(r_step);
        free(origin); free(basis);
        Py_RETURN_NONE;
    } else if (mode == 2) {
        /* elementwise */
        PyObject *tensors = PyDict_GetItemString(kwargs, "tensors");
        PyObject *func = PyDict_GetItemString(kwargs, "func");
        PyObject *output = PyDict_GetItemString(kwargs, "output");
        if (tensors == NULL || func == NULL || output == NULL) {
            PyErr_SetString(PyExc_TypeError,
                            "elementwise kernel: bad arguments");
            free(index);
            return NULL;
        }
        Py_ssize_t nt = PyTuple_Size(tensors);
        PyObject *vals = PyTuple_New(nt);
        if (vals == NULL) { free(index); return NULL; }
        for (Py_ssize_t t = 0; t < nt; ++t) {
            PyObject *get_args = PyTuple_Pack(
                2, PyTuple_GET_ITEM(tensors, t), index_obj);
            PyObject *v = get_args ? py_get_item(g_module, get_args) : NULL;
            Py_XDECREF(get_args);
            if (v == NULL) {
                Py_DECREF(vals); free(index);
                return NULL;
            }
            PyTuple_SET_ITEM(vals, t, v);
        }
        PyObject *value = PyObject_Call(func, vals, NULL);
        Py_DECREF(vals);
        if (value == NULL) { free(index); return NULL; }
        if (!PyNumber_Check(value)) {
            Py_DECREF(value);
            PyErr_SetString(PyExc_TypeError,
                            "elementwise func must return a number");
            free(index);
            return NULL;
        }
        PyObject *set_args = PyTuple_Pack(3, output, index_obj, value);
        Py_DECREF(value);
        PyObject *r = set_args ? py_set_item(g_module, set_args) : NULL;
        Py_XDECREF(set_args);
        if (r == NULL) { free(index); return NULL; }
        Py_DECREF(r);
        free(index);
        Py_RETURN_NONE;
    } else if (mode == 3) {
        /* position_map */
        PyObject *output = PyDict_GetItemString(kwargs, "output");
        PyObject *callback = PyDict_GetItemString(kwargs, "callback");
        PyObject *status = PyDict_GetItemString(kwargs, "status");
        int *r_start = NULL, *r_step = NULL;
        double *origin_d = NULL, *scale_d = NULL;
        int ok = (output != NULL && callback != NULL && status != NULL
            && _parse_int_tuple(PyDict_GetItemString(kwargs, "r_start"),
                                &r_start, dimension) == 0
            && _parse_int_tuple(PyDict_GetItemString(kwargs, "r_step"),
                                &r_step, dimension) == 0
            && _parse_dbl_tuple(PyDict_GetItemString(kwargs, "origin"),
                                &origin_d, dimension) == 0
            && _parse_dbl_tuple(PyDict_GetItemString(kwargs, "scale"),
                                &scale_d, dimension) == 0);
        if (!ok) {
            PyErr_Clear();
            PyErr_SetString(PyExc_TypeError,
                            "position_map kernel: bad arguments");
            free(index); free(r_start); free(r_step);
            free(origin_d); free(scale_d);
            return NULL;
        }
        PyObject *real = PyTuple_New(dimension);
        PyObject *logical = PyTuple_New(dimension);
        for (int i = 0; i < dimension; ++i) {
            double rr = (double)((long long)r_start[i]
                                 + (long long)index[i] * r_step[i]);
            PyTuple_SET_ITEM(real, i, PyLong_FromLongLong((long long)rr));
            double logical_val = rr * scale_d[i] - origin_d[i];
            long long li = (long long)logical_val;
            if ((double)li == logical_val)
                PyTuple_SET_ITEM(logical, i, PyLong_FromLongLong(li));
            else
                PyTuple_SET_ITEM(logical, i, PyFloat_FromDouble(logical_val));
        }
        PyObject *cb_res = PyObject_CallFunctionObjArgs(callback, logical,
                                                        NULL);
        Py_DECREF(logical);
        if (cb_res == NULL) {
            PyErr_Clear();  /* callback errors silently skipped */
        } else {
            PyObject *set_args = PyTuple_Pack(3, output, real, cb_res);
            Py_DECREF(cb_res);
            PyObject *r = set_args ? py_set_item(g_module, set_args) : NULL;
            Py_XDECREF(set_args);
            if (r == NULL) {
                PyErr_Clear();
                PyList_SetItem(status, 0, PyLong_FromLong(1));
                Py_DECREF(real);
                free(index); free(r_start); free(r_step);
                free(origin_d); free(scale_d);
                Py_RETURN_NONE;
            }
            Py_DECREF(r);
        }
        Py_DECREF(real);
        free(index); free(r_start); free(r_step);
        free(origin_d); free(scale_d);
        Py_RETURN_NONE;
    } else {
        /* elementwise_position */
        PyObject *output = PyDict_GetItemString(kwargs, "output");
        PyObject *tensors = PyDict_GetItemString(kwargs, "tensors");
        PyObject *callback = PyDict_GetItemString(kwargs, "callback");
        PyObject *status = PyDict_GetItemString(kwargs, "status");
        int *r_start = NULL, *r_step = NULL;
        double *origin_d = NULL, *scale_d = NULL;
        int ok = (output != NULL && tensors != NULL && callback != NULL
            && status != NULL
            && _parse_int_tuple(PyDict_GetItemString(kwargs, "r_start"),
                                &r_start, dimension) == 0
            && _parse_int_tuple(PyDict_GetItemString(kwargs, "r_step"),
                                &r_step, dimension) == 0
            && _parse_dbl_tuple(PyDict_GetItemString(kwargs, "origin"),
                                &origin_d, dimension) == 0
            && _parse_dbl_tuple(PyDict_GetItemString(kwargs, "scale"),
                                &scale_d, dimension) == 0);
        if (!ok) {
            PyErr_Clear();
            PyErr_SetString(PyExc_TypeError,
                            "elementwise_position kernel: bad arguments");
            free(index); free(r_start); free(r_step);
            free(origin_d); free(scale_d);
            return NULL;
        }
        PyObject *real = PyTuple_New(dimension);
        PyObject *logical = PyTuple_New(dimension);
        for (int i = 0; i < dimension; ++i) {
            double rr = (double)((long long)r_start[i]
                                 + (long long)index[i] * r_step[i]);
            PyTuple_SET_ITEM(real, i, PyLong_FromLongLong((long long)rr));
            double logical_val = rr * scale_d[i] - origin_d[i];
            long long li = (long long)logical_val;
            if ((double)li == logical_val)
                PyTuple_SET_ITEM(logical, i, PyLong_FromLongLong(li));
            else
                PyTuple_SET_ITEM(logical, i, PyFloat_FromDouble(logical_val));
        }
        Py_ssize_t nt = PyTuple_Size(tensors);
        PyObject *elements = PyTuple_New(nt);
        for (Py_ssize_t t = 0; t < nt; ++t) {
            PyObject *get_args = PyTuple_Pack(
                2, PyTuple_GET_ITEM(tensors, t), real);
            PyObject *v = get_args ? py_get_item(g_module, get_args) : NULL;
            Py_XDECREF(get_args);
            if (v == NULL) {
                PyErr_Clear();
                Py_DECREF(elements); Py_DECREF(logical); Py_DECREF(real);
                PyList_SetItem(status, 0, PyLong_FromLong(1));
                free(index); free(r_start); free(r_step);
                free(origin_d); free(scale_d);
                Py_RETURN_NONE;
            }
            PyTuple_SET_ITEM(elements, t, v);
        }
        PyObject *value = PyObject_CallFunctionObjArgs(callback, elements,
                                                       logical, NULL);
        Py_DECREF(elements);
        Py_DECREF(logical);
        if (value == NULL) {
            PyErr_Clear();  /* callback errors silently skipped */
        } else {
            PyObject *set_args = PyTuple_Pack(3, output, real, value);
            Py_DECREF(value);
            PyObject *r = set_args ? py_set_item(g_module, set_args) : NULL;
            Py_XDECREF(set_args);
            if (r == NULL) {
                PyErr_Clear();
                PyList_SetItem(status, 0, PyLong_FromLong(1));
                Py_DECREF(real);
                free(index); free(r_start); free(r_step);
                free(origin_d); free(scale_d);
                Py_RETURN_NONE;
            }
            Py_DECREF(r);
        }
        Py_DECREF(real);
        free(index); free(r_start); free(r_step);
        free(origin_d); free(scale_d);
        Py_RETURN_NONE;
    }
}

static PyMethodDef _kernel_def_mapping = {
    "_data_mapping_kernel",
    (PyCFunction)(void(*)(void))_iterate_kernel,
    METH_VARARGS | METH_KEYWORDS, NULL};
static PyMethodDef _kernel_def_filter = {
    "_data_filter_kernel",
    (PyCFunction)(void(*)(void))_iterate_kernel,
    METH_VARARGS | METH_KEYWORDS, NULL};
static PyMethodDef _kernel_def_elementwise = {
    "_elementwise_kernel",
    (PyCFunction)(void(*)(void))_iterate_kernel,
    METH_VARARGS | METH_KEYWORDS, NULL};
static PyMethodDef _kernel_def_position = {
    "_position_map_kernel",
    (PyCFunction)(void(*)(void))_iterate_kernel,
    METH_VARARGS | METH_KEYWORDS, NULL};
static PyMethodDef _kernel_def_elementwise_position = {
    "_elementwise_position_kernel",
    (PyCFunction)(void(*)(void))_iterate_kernel,
    METH_VARARGS | METH_KEYWORDS, NULL};

static PyObject* py_data_filter(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"data", "callback", "start", "shape", "step",
                             "origin", "basis", "iterate", NULL};
    PyObject *data, *callback;
    PyObject *start = NULL, *shape_obj = NULL, *step = NULL;
    PyObject *origin = NULL, *basis = NULL;
    PyObject *iterate = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OOOOOO", kwlist,
            &data, &callback, &start, &shape_obj, &step, &origin, &basis,
            &iterate))
        return NULL;

    /* Match pure Python: None input cannot be shaped. */
    if (data == Py_None) {
        PyErr_SetString(PyExc_ValueError, "cannot infer shape of input data");
        return NULL;
    }

    int *data_shape = NULL, dimension = 0;
    if (infer_shape(data, &data_shape, &dimension) < 0) return NULL;

    if (check_region_len(start, dimension, "start") < 0
        || check_region_len(step, dimension, "step") < 0
        || check_region_len(shape_obj, dimension, "shape") < 0
        || check_region_len(origin, dimension, "origin") < 0
        || check_region_len(basis, dimension, "basis") < 0) {
        free(data_shape);
        return NULL;
    }

    int *r_start = NULL, *r_step = NULL, *effective = NULL;
    long long total = 0;
    if (resolve_read_region(start, shape_obj, step, data_shape, dimension,
                             &r_start, &r_step, &effective, &total) < 0) {
        free(data_shape);
        return NULL;
    }

    int *r_origin = NULL, *r_basis = NULL;
    if (parse_opt_int_seq(origin, &r_origin, dimension, 0) < 0
        || parse_opt_int_seq(basis, &r_basis, dimension, 1) < 0) {
        free(data_shape); free(r_start); free(r_step); free(effective);
        return NULL;
    }
    /* origin defaults to start, basis defaults to step */
    for (int i = 0; i < dimension; ++i) {
        if (origin == NULL || origin == Py_None) r_origin[i] = r_start[i];
        if (basis == NULL || basis == Py_None) r_basis[i] = r_step[i];
    }

    PyObject *result = PyList_New(0);
    if (!result) {
        free(data_shape); free(r_start); free(r_step); free(effective);
        free(r_origin); free(r_basis);
        PyErr_NoMemory();
        return NULL;
    }

    if (iterate != NULL && iterate != Py_None) {
        /* optional custom iterator: hand it the C kernel + index info */
        PyObject *res = NULL;
        PyObject *kernel = _make_kernel(&_kernel_def_filter, 1);
        PyObject *call_args = kernel ? PyTuple_Pack(1, kernel) : NULL;
        Py_XDECREF(kernel);
        PyObject *call_kwargs = call_args ? PyDict_New() : NULL;
        PyObject *data_shape_t = call_kwargs
            ? _ints_to_tuple(data_shape, dimension) : NULL;
        PyObject *r_start_t = data_shape_t
            ? _ints_to_tuple(r_start, dimension) : NULL;
        PyObject *r_step_t = r_start_t
            ? _ints_to_tuple(r_step, dimension) : NULL;
        PyObject *origin_t = r_step_t
            ? _ints_to_tuple(r_origin, dimension) : NULL;
        PyObject *basis_t = origin_t
            ? _ints_to_tuple(r_basis, dimension) : NULL;
        if (basis_t != NULL) {
            PyDict_SetItemString(call_kwargs, "data_shape", data_shape_t);
            PyDict_SetItemString(call_kwargs, "start", r_start_t);
            PyDict_SetItemString(call_kwargs, "shape",
                                 (shape_obj && shape_obj != Py_None)
                                     ? shape_obj : Py_None);
            PyDict_SetItemString(call_kwargs, "step", r_step_t);
            PyDict_SetItemString(call_kwargs, "data", data);
            PyDict_SetItemString(call_kwargs, "callback", callback);
            PyDict_SetItemString(call_kwargs, "hits", result);
            PyDict_SetItemString(call_kwargs, "r_start", r_start_t);
            PyDict_SetItemString(call_kwargs, "r_step", r_step_t);
            PyDict_SetItemString(call_kwargs, "origin", origin_t);
            PyDict_SetItemString(call_kwargs, "basis", basis_t);
            res = PyObject_Call(iterate, call_args, call_kwargs);
        }
        Py_XDECREF(data_shape_t); Py_XDECREF(r_start_t);
        Py_XDECREF(r_step_t); Py_XDECREF(origin_t); Py_XDECREF(basis_t);
        Py_XDECREF(call_kwargs); Py_XDECREF(call_args);
        free(r_origin); free(r_basis); free(effective);
        free(r_start); free(r_step); free(data_shape);
        if (res == NULL) { Py_DECREF(result); return NULL; }
        Py_DECREF(res);
        return result;
    }

    int *local = (int*)calloc((size_t)(dimension > 0 ? dimension : 1), sizeof(int));
    if (!local) {
        Py_DECREF(result); free(data_shape); free(r_start); free(r_step);
        free(effective); free(r_origin); free(r_basis);
        PyErr_NoMemory();
        return NULL;
    }

    for (long long k = 0; k < total; ++k) {
        PyObject *read = PyTuple_New(dimension);
        for (int i = 0; i < dimension; ++i)
            PyTuple_SET_ITEM(read, i, PyLong_FromLongLong((long long)r_start[i] + (long long)local[i] * r_step[i]));
        PyObject *get_args = PyTuple_Pack(2, data, read);
        Py_DECREF(read);
        PyObject *value = py_get_item(self, get_args);
        Py_DECREF(get_args);
        if (!value) {
            Py_DECREF(result); free(local); free(r_origin); free(r_basis);
            free(effective); free(r_start); free(r_step); free(data_shape);
            return NULL;
        }
        PyObject *cb_res = PyObject_CallFunctionObjArgs(callback, value, NULL);
        Py_DECREF(value);
        if (!cb_res) {
            PyErr_Clear();  /* callback errors silently skipped */
        } else {
            int truthy = PyObject_IsTrue(cb_res);
            Py_DECREF(cb_res);
            if (truthy < 0) {
                PyErr_Clear();
                truthy = 0;
            }
            if (truthy) {
                PyObject *pos = PyTuple_New(dimension);
                for (int i = 0; i < dimension; ++i)
                    PyTuple_SET_ITEM(pos, i, PyLong_FromLongLong((long long)r_origin[i] + (long long)r_basis[i] * local[i]));
                int app = PyList_Append(result, pos);
                Py_DECREF(pos);
                if (app < 0) {
                    Py_DECREF(result); free(local); free(r_origin); free(r_basis);
                    free(effective); free(r_start); free(r_step); free(data_shape);
                    return NULL;
                }
            }
        }
        for (int i = dimension - 1; i >= 0; --i) {
            local[i]++;
            if (local[i] < effective[i]) break;
            local[i] = 0;
        }
    }
    free(local); free(r_origin); free(r_basis); free(effective);
    free(r_start); free(r_step); free(data_shape);
    return result;
}

static PyObject* py_data_mapping(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"data", "callback", "start", "shape", "step",
                             "out", "out_start", "out_step", "iterate", NULL};
    PyObject *data, *callback;
    PyObject *start = NULL, *shape_obj = NULL, *step = NULL;
    PyObject *out = NULL, *out_start = NULL, *out_step = NULL;
    PyObject *iterate = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OOOOOOO", kwlist,
            &data, &callback, &start, &shape_obj, &step, &out, &out_start,
            &out_step, &iterate))
        return NULL;

    /* Match pure Python: None input cannot be shaped. */
    if (data == Py_None) {
        PyErr_SetString(PyExc_ValueError, "cannot infer shape of input data");
        return NULL;
    }

    int *data_shape = NULL, dimension = 0;
    if (infer_shape(data, &data_shape, &dimension) < 0) return NULL;

    if (check_region_len(start, dimension, "start") < 0
        || check_region_len(step, dimension, "step") < 0
        || check_region_len(shape_obj, dimension, "shape") < 0
        || check_region_len(out_start, dimension, "out_start") < 0
        || check_region_len(out_step, dimension, "out_step") < 0) {
        free(data_shape);
        return NULL;
    }

    int *r_start = NULL, *r_step = NULL, *effective = NULL;
    long long total = 0;
    if (resolve_read_region(start, shape_obj, step, data_shape, dimension,
                             &r_start, &r_step, &effective, &total) < 0) {
        free(data_shape);
        return NULL;
    }

    /* output: default-allocate shaped like the read region */
    if (out == NULL || out == Py_None) {
        PyObject *shape_tuple = PyTuple_New(dimension);
        for (int i = 0; i < dimension; ++i)
            PyTuple_SET_ITEM(shape_tuple, i, PyLong_FromLong(effective[i]));
        PyObject *void_args = PyTuple_Pack(1, shape_tuple);
        Py_DECREF(shape_tuple);
        out = py_create_void_list(self, void_args, NULL);
        Py_DECREF(void_args);
        if (!out) { free(effective); free(r_start); free(r_step); free(data_shape); return NULL; }
    } else {
        Py_INCREF(out);
    }

    int *out_shape = NULL, out_dim = 0;
    if (infer_shape(out, &out_shape, &out_dim) < 0) {
        Py_DECREF(out); free(effective); free(r_start); free(r_step); free(data_shape);
        return NULL;
    }
    if (out_dim != dimension) {
        PyErr_SetString(PyExc_ValueError, "output dimension does not match data dimension");
        Py_DECREF(out); free(out_shape); free(effective); free(r_start); free(r_step); free(data_shape);
        return NULL;
    }

    int *w_start = NULL, *w_step = NULL;
    if (parse_opt_int_seq(out_start, &w_start, dimension, 0) < 0
        || parse_opt_int_seq(out_step, &w_step, dimension, 1) < 0) {
        Py_DECREF(out); free(out_shape); free(effective); free(r_start); free(r_step); free(data_shape);
        return NULL;
    }
    for (int i = 0; i < dimension; ++i) {
        if (w_start[i] < 0) {
            PyErr_SetString(PyExc_ValueError, "out_start entries must be non-negative");
            Py_DECREF(out); free(out_shape); free(w_start); free(w_step);
            free(effective); free(r_start); free(r_step); free(data_shape);
            return NULL;
        }
        if (w_step[i] <= 0) {
            PyErr_SetString(PyExc_ValueError, "out_step entries must be positive");
            Py_DECREF(out); free(out_shape); free(w_start); free(w_step);
            free(effective); free(r_start); free(r_step); free(data_shape);
            return NULL;
        }
    }

    if (iterate != NULL && iterate != Py_None) {
        /* optional custom iterator: hand it the C kernel + index info */
        PyObject *res = NULL;
        PyObject *kernel = _make_kernel(&_kernel_def_mapping, 0);
        PyObject *call_args = kernel ? PyTuple_Pack(1, kernel) : NULL;
        Py_XDECREF(kernel);
        PyObject *call_kwargs = call_args ? PyDict_New() : NULL;
        PyObject *data_shape_t = call_kwargs
            ? _ints_to_tuple(data_shape, dimension) : NULL;
        PyObject *r_start_t = data_shape_t
            ? _ints_to_tuple(r_start, dimension) : NULL;
        PyObject *r_step_t = r_start_t
            ? _ints_to_tuple(r_step, dimension) : NULL;
        PyObject *w_start_t = r_step_t
            ? _ints_to_tuple(w_start, dimension) : NULL;
        PyObject *w_step_t = w_start_t
            ? _ints_to_tuple(w_step, dimension) : NULL;
        PyObject *out_shape_t = w_step_t
            ? _ints_to_tuple(out_shape, dimension) : NULL;
        if (out_shape_t != NULL) {
            PyDict_SetItemString(call_kwargs, "data_shape", data_shape_t);
            PyDict_SetItemString(call_kwargs, "start", r_start_t);
            PyDict_SetItemString(call_kwargs, "shape",
                                 (shape_obj && shape_obj != Py_None)
                                     ? shape_obj : Py_None);
            PyDict_SetItemString(call_kwargs, "step", r_step_t);
            PyDict_SetItemString(call_kwargs, "data", data);
            PyDict_SetItemString(call_kwargs, "callback", callback);
            PyDict_SetItemString(call_kwargs, "out", out);
            PyDict_SetItemString(call_kwargs, "r_start", r_start_t);
            PyDict_SetItemString(call_kwargs, "r_step", r_step_t);
            PyDict_SetItemString(call_kwargs, "out_start", w_start_t);
            PyDict_SetItemString(call_kwargs, "out_step", w_step_t);
            PyDict_SetItemString(call_kwargs, "out_shape", out_shape_t);
            res = PyObject_Call(iterate, call_args, call_kwargs);
        }
        Py_XDECREF(data_shape_t); Py_XDECREF(r_start_t);
        Py_XDECREF(r_step_t); Py_XDECREF(w_start_t);
        Py_XDECREF(w_step_t); Py_XDECREF(out_shape_t);
        Py_XDECREF(call_kwargs); Py_XDECREF(call_args);
        free(out_shape); free(w_start); free(w_step);
        free(effective); free(r_start); free(r_step); free(data_shape);
        if (res == NULL) { Py_DECREF(out); return NULL; }
        Py_DECREF(res);
        return out;
    }

    int *local = (int*)calloc((size_t)(dimension > 0 ? dimension : 1), sizeof(int));
    if (!local) {
        Py_DECREF(out); free(out_shape); free(w_start); free(w_step);
        free(effective); free(r_start); free(r_step); free(data_shape);
        PyErr_NoMemory();
        return NULL;
    }

    for (long long k = 0; k < total; ++k) {
        PyObject *read = PyTuple_New(dimension);
        for (int i = 0; i < dimension; ++i)
            PyTuple_SET_ITEM(read, i, PyLong_FromLongLong((long long)r_start[i] + (long long)local[i] * r_step[i]));
        PyObject *get_args = PyTuple_Pack(2, data, read);
        Py_DECREF(read);
        PyObject *value = py_get_item(self, get_args);
        Py_DECREF(get_args);
        if (!value) {
            Py_DECREF(out); free(local); free(out_shape); free(w_start); free(w_step);
            free(effective); free(r_start); free(r_step); free(data_shape);
            return NULL;
        }
        PyObject *cb_res = PyObject_CallFunctionObjArgs(callback, value, NULL);
        Py_DECREF(value);
        if (!cb_res) {
            PyErr_Clear();  /* callback errors silently skipped */
        } else {
            int in_bounds = 1;
            for (int i = 0; i < dimension; ++i) {
                long w = (long long)w_start[i] + (long long)local[i] * w_step[i];
                if (w >= out_shape[i]) { in_bounds = 0; break; }
            }
            if (in_bounds) {
                PyObject *write = PyTuple_New(dimension);
                for (int i = 0; i < dimension; ++i)
                    PyTuple_SET_ITEM(write, i, PyLong_FromLongLong((long long)w_start[i] + (long long)local[i] * w_step[i]));
                PyObject *set_args = PyTuple_Pack(3, out, write, cb_res);
                Py_DECREF(write);
                PyObject *r = py_set_item(self, set_args);
                Py_DECREF(set_args);
                if (!r) {
                    Py_DECREF(cb_res); Py_DECREF(out); free(local); free(out_shape);
                    free(w_start); free(w_step); free(effective); free(r_start); free(r_step); free(data_shape);
                    return NULL;
                }
                Py_DECREF(r);
            }
            Py_DECREF(cb_res);
        }
        for (int i = dimension - 1; i >= 0; --i) {
            local[i]++;
            if (local[i] < effective[i]) break;
            local[i] = 0;
        }
    }
    free(local); free(out_shape); free(w_start); free(w_step); free(effective);
    free(r_start); free(r_step); free(data_shape);
    return out;
}

/* ------------------------------------------------------------------
Element-wise operation: elementwise(*tensors, func=f, output=o) -> 0.
Duck typing: every tensor/output only needs the sequence protocol
(get_item/set_item/infer_shape); shapes must match exactly.  func is
called with the values at each position as positional arguments in
tensor order.  Callback errors propagate (no silent skipping); the
callback return value must be numeric.  Iterative row-major walk,
never recursive.  Integrated with the internal get_item/set_item
machinery for the value access.
------------------------------------------------------------------ */
/* ------------------------------------------------------------------
 * position_map / elementwise_position: position-driven element callback
 * (C99).  Traversal and writes stay on the REAL region
 * (start/shape/step, clipped like load_data's source side); the callback
 * receives the LOGICAL coordinate redirected by origin and scaled per
 * dimension (logical_i = (real_i - origin_i) / scale_i).  Callback
 * errors are silently skipped; a read or write failure stops with
 * status 1 (0 success).
 * ------------------------------------------------------------------ */

/* parse an optional sequence into a double array (dflt when None) */
static int parse_opt_dbl_seq(PyObject *obj, double **arr, int dim,
                              double dflt, int nonzero) {
    if (obj == NULL || obj == Py_None) {
        *arr = (double *)malloc((size_t)(dim > 0 ? dim : 1) * sizeof(double));
        if (*arr == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        for (int i = 0; i < dim; ++i) {
            (*arr)[i] = dflt;
        }
        return 0;
    }
    {
        Py_ssize_t n = PySequence_Size(obj);
        if (n < 0 || n != (Py_ssize_t)dim) {
            PyErr_SetString(PyExc_ValueError,
                            "sequence length does not match dimension");
            return -1;
        }
        *arr = (double *)malloc((size_t)(dim > 0 ? dim : 1) * sizeof(double));
        if (*arr == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        for (int i = 0; i < dim; ++i) {
            PyObject *item = PySequence_GetItem(obj, i);
            double v;
            if (item == NULL) {
                free(*arr);
                *arr = NULL;
                return -1;
            }
            v = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (v == -1.0 && PyErr_Occurred()) {
                free(*arr);
                *arr = NULL;
                return -1;
            }
            if (nonzero && v == 0.0) {
                PyErr_SetString(PyExc_ValueError,
                                "scale entries must be non-zero");
                free(*arr);
                *arr = NULL;
                return -1;
            }
            (*arr)[i] = v;
        }
    }
    return 0;
}

static PyObject* py_position_map(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
    static char *kwlist[] = {"output", "callback", "start", "shape", "step",
                             "origin", "scale", "iterate", NULL};
    PyObject *output, *callback;
    PyObject *start = NULL, *shape_obj = NULL, *step = NULL;
    PyObject *origin = NULL, *scale = NULL;
    PyObject *iterate = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OOOOOO", kwlist,
            &output, &callback, &start, &shape_obj, &step, &origin, &scale,
            &iterate))
        return NULL;
    if (output == Py_None) {
        PyErr_SetString(PyExc_ValueError, "cannot infer shape of output");
        return NULL;
    }

    int *data_shape = NULL, dimension = 0;
    if (infer_shape(output, &data_shape, &dimension) < 0) return NULL;
    if (check_region_len(start, dimension, "start") < 0
        || check_region_len(step, dimension, "step") < 0
        || check_region_len(shape_obj, dimension, "shape") < 0
        || check_region_len(origin, dimension, "origin") < 0
        || check_region_len(scale, dimension, "scale") < 0) {
        free(data_shape);
        return NULL;
    }

    int *r_start = NULL, *r_step = NULL, *effective = NULL;
    long long total = 0;
    if (resolve_read_region(start, shape_obj, step, data_shape, dimension,
                             &r_start, &r_step, &effective, &total) < 0) {
        free(data_shape);
        return NULL;
    }
    double *origin_d = NULL, *scale_d = NULL;
    if (parse_opt_dbl_seq(origin, &origin_d, dimension, 0.0, 0) < 0
        || parse_opt_dbl_seq(scale, &scale_d, dimension, 1.0, 0) < 0) {
        free(data_shape); free(effective); free(r_start); free(r_step);
        free(origin_d); free(scale_d);
        return NULL;
    }

    if (iterate != NULL && iterate != Py_None) {
        /* optional custom iterator: hand it the C kernel + index info */
        long status_val = 0;
        PyObject *res = NULL;
        PyObject *status = PyList_New(1);
        if (status != NULL) PyList_SET_ITEM(status, 0, PyLong_FromLong(0));
        PyObject *kernel = status ? _make_kernel(&_kernel_def_position, 3)
                                  : NULL;
        PyObject *call_args = kernel ? PyTuple_Pack(1, kernel) : NULL;
        Py_XDECREF(kernel);
        PyObject *call_kwargs = call_args ? PyDict_New() : NULL;
        PyObject *data_shape_t = call_kwargs
            ? _ints_to_tuple(data_shape, dimension) : NULL;
        PyObject *r_start_t = data_shape_t
            ? _ints_to_tuple(r_start, dimension) : NULL;
        PyObject *r_step_t = r_start_t
            ? _ints_to_tuple(r_step, dimension) : NULL;
        PyObject *origin_t = r_step_t
            ? _dbls_to_tuple(origin_d, dimension) : NULL;
        PyObject *scale_t = origin_t
            ? _dbls_to_tuple(scale_d, dimension) : NULL;
        if (scale_t != NULL) {
            PyDict_SetItemString(call_kwargs, "data_shape", data_shape_t);
            PyDict_SetItemString(call_kwargs, "start", r_start_t);
            PyDict_SetItemString(call_kwargs, "shape",
                                 (shape_obj && shape_obj != Py_None)
                                     ? shape_obj : Py_None);
            PyDict_SetItemString(call_kwargs, "step", r_step_t);
            PyDict_SetItemString(call_kwargs, "output", output);
            PyDict_SetItemString(call_kwargs, "callback", callback);
            PyDict_SetItemString(call_kwargs, "r_start", r_start_t);
            PyDict_SetItemString(call_kwargs, "r_step", r_step_t);
            PyDict_SetItemString(call_kwargs, "origin", origin_t);
            PyDict_SetItemString(call_kwargs, "scale", scale_t);
            PyDict_SetItemString(call_kwargs, "status", status);
            res = PyObject_Call(iterate, call_args, call_kwargs);
        }
        Py_XDECREF(data_shape_t); Py_XDECREF(r_start_t);
        Py_XDECREF(r_step_t); Py_XDECREF(origin_t); Py_XDECREF(scale_t);
        Py_XDECREF(call_kwargs); Py_XDECREF(call_args);
        free(data_shape); free(effective); free(r_start); free(r_step);
        free(origin_d); free(scale_d);
        if (res == NULL) { Py_XDECREF(status); return NULL; }
        Py_DECREF(res);
        if (status != NULL) {
            PyObject *sv = PyList_GET_ITEM(status, 0);
            status_val = PyLong_AsLong(sv);
            if (status_val == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                status_val = 0;
            }
            Py_DECREF(status);
        }
        return PyLong_FromLong(status_val);
    }

    int *local = (int *)calloc((size_t)(dimension > 0 ? dimension : 1),
                               sizeof(int));
    if (local == NULL) {
        free(data_shape); free(effective); free(r_start); free(r_step);
        free(origin_d); free(scale_d);
        PyErr_NoMemory();
        return NULL;
    }

    for (long long k = 0; k < total; ++k) {
        PyObject *real = PyTuple_New(dimension);
        PyObject *logical = PyTuple_New(dimension);
        if (real == NULL || logical == NULL) {
            Py_XDECREF(real); Py_XDECREF(logical);
            free(local); free(data_shape); free(effective); free(r_start);
            free(r_step); free(origin_d); free(scale_d);
            return NULL;
        }
        for (int i = 0; i < dimension; ++i) {
            double rr = (double)((long long)r_start[i]
                                 + (long long)local[i] * r_step[i]);
            PyTuple_SET_ITEM(real, i, PyLong_FromLong(
                (long long)r_start[i] + (long long)local[i] * r_step[i]));
            /* logical: keep an integer when exactly divisible (type
             * safety), otherwise float */
            {
                double logical_val = rr * scale_d[i] - origin_d[i];
                long long li = (long long)logical_val;
                if ((double)li == logical_val) {
                    PyTuple_SET_ITEM(logical, i,
                                     PyLong_FromLongLong(li));
                } else {
                    PyTuple_SET_ITEM(logical, i,
                                     PyFloat_FromDouble(logical_val));
                }
            }
        }
        PyObject *cb_res = PyObject_CallFunctionObjArgs(
            callback, logical, NULL);
        Py_DECREF(logical);
        if (cb_res == NULL) {
            PyErr_Clear();  /* callback errors silently skipped */
        } else {
            PyObject *set_args = PyTuple_Pack(3, output, real, cb_res);
            PyObject *r = set_args ? py_set_item(self, set_args) : NULL;
            Py_XDECREF(set_args);
            Py_DECREF(cb_res);
            if (r == NULL) {
                PyErr_Clear();
                Py_DECREF(real);
                free(local); free(data_shape); free(effective);
                free(r_start); free(r_step); free(origin_d); free(scale_d);
                return PyLong_FromLong(1);  /* write failure -> status 1 */
            }
            Py_DECREF(r);
        }
        Py_DECREF(real);
        for (int i = dimension - 1; i >= 0; --i) {
            local[i]++;
            if (local[i] < effective[i]) break;
            local[i] = 0;
        }
    }
    free(local); free(data_shape); free(effective); free(r_start);
    free(r_step); free(origin_d); free(scale_d);
    return PyLong_FromLong(0);
}

static PyObject* py_elementwise_position(PyObject *self, PyObject *args,
                                         PyObject *kwargs) {
    Py_ssize_t n = PyTuple_Size(args);
    if (n < 1) {
        PyErr_SetString(PyExc_TypeError,
                        "elementwise_position requires an output");
        return NULL;
    }
    PyObject *output = PyTuple_GET_ITEM(args, 0);
    PyObject *callback = kwargs ? PyDict_GetItemString(kwargs, "callback")
                                : NULL;
    if (callback == NULL || !PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "callback is required");
        return NULL;
    }
    PyObject *start = kwargs ? PyDict_GetItemString(kwargs, "start") : NULL;
    PyObject *shape_obj = kwargs ? PyDict_GetItemString(kwargs, "shape") : NULL;
    PyObject *step = kwargs ? PyDict_GetItemString(kwargs, "step") : NULL;
    PyObject *origin = kwargs ? PyDict_GetItemString(kwargs, "origin") : NULL;
    PyObject *scale = kwargs ? PyDict_GetItemString(kwargs, "scale") : NULL;
    PyObject *iterate = kwargs ? PyDict_GetItemString(kwargs, "iterate")
                               : NULL;
    if (output == Py_None) {
        PyErr_SetString(PyExc_ValueError, "cannot infer shape of output");
        return NULL;
    }

    int *data_shape = NULL, dimension = 0;
    if (infer_shape(output, &data_shape, &dimension) < 0) return NULL;
    if (check_region_len(start, dimension, "start") < 0
        || check_region_len(step, dimension, "step") < 0
        || check_region_len(shape_obj, dimension, "shape") < 0
        || check_region_len(origin, dimension, "origin") < 0
        || check_region_len(scale, dimension, "scale") < 0) {
        free(data_shape);
        return NULL;
    }

    int *r_start = NULL, *r_step = NULL, *effective = NULL;
    long long total = 0;
    if (resolve_read_region(start, shape_obj, step, data_shape, dimension,
                             &r_start, &r_step, &effective, &total) < 0) {
        free(data_shape);
        return NULL;
    }
    double *origin_d = NULL, *scale_d = NULL;
    if (parse_opt_dbl_seq(origin, &origin_d, dimension, 0.0, 0) < 0
        || parse_opt_dbl_seq(scale, &scale_d, dimension, 1.0, 0) < 0) {
        free(data_shape); free(effective); free(r_start); free(r_step);
        free(origin_d); free(scale_d);
        return NULL;
    }

    if (iterate != NULL && iterate != Py_None) {
        /* optional custom iterator: hand it the C kernel + index info */
        long status_val = 0;
        PyObject *res = NULL;
        PyObject *status = PyList_New(1);
        if (status != NULL) PyList_SET_ITEM(status, 0, PyLong_FromLong(0));
        PyObject *tensors = PyTuple_GetSlice(args, 1, n);
        PyObject *kernel = (status && tensors)
            ? _make_kernel(&_kernel_def_elementwise_position, 4) : NULL;
        PyObject *call_args = kernel ? PyTuple_Pack(1, kernel) : NULL;
        Py_XDECREF(kernel);
        PyObject *call_kwargs = call_args ? PyDict_New() : NULL;
        PyObject *data_shape_t = call_kwargs
            ? _ints_to_tuple(data_shape, dimension) : NULL;
        PyObject *r_start_t = data_shape_t
            ? _ints_to_tuple(r_start, dimension) : NULL;
        PyObject *r_step_t = r_start_t
            ? _ints_to_tuple(r_step, dimension) : NULL;
        PyObject *origin_t = r_step_t
            ? _dbls_to_tuple(origin_d, dimension) : NULL;
        PyObject *scale_t = origin_t
            ? _dbls_to_tuple(scale_d, dimension) : NULL;
        if (scale_t != NULL) {
            PyDict_SetItemString(call_kwargs, "data_shape", data_shape_t);
            PyDict_SetItemString(call_kwargs, "start", r_start_t);
            PyDict_SetItemString(call_kwargs, "shape",
                                 (shape_obj && shape_obj != Py_None)
                                     ? shape_obj : Py_None);
            PyDict_SetItemString(call_kwargs, "step", r_step_t);
            PyDict_SetItemString(call_kwargs, "output", output);
            PyDict_SetItemString(call_kwargs, "tensors", tensors);
            PyDict_SetItemString(call_kwargs, "callback", callback);
            PyDict_SetItemString(call_kwargs, "r_start", r_start_t);
            PyDict_SetItemString(call_kwargs, "r_step", r_step_t);
            PyDict_SetItemString(call_kwargs, "origin", origin_t);
            PyDict_SetItemString(call_kwargs, "scale", scale_t);
            PyDict_SetItemString(call_kwargs, "status", status);
            res = PyObject_Call(iterate, call_args, call_kwargs);
        }
        Py_XDECREF(data_shape_t); Py_XDECREF(r_start_t);
        Py_XDECREF(r_step_t); Py_XDECREF(origin_t); Py_XDECREF(scale_t);
        Py_XDECREF(tensors);
        Py_XDECREF(call_kwargs); Py_XDECREF(call_args);
        free(data_shape); free(effective); free(r_start); free(r_step);
        free(origin_d); free(scale_d);
        if (res == NULL) { Py_XDECREF(status); return NULL; }
        Py_DECREF(res);
        if (status != NULL) {
            PyObject *sv = PyList_GET_ITEM(status, 0);
            status_val = PyLong_AsLong(sv);
            if (status_val == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                status_val = 0;
            }
            Py_DECREF(status);
        }
        return PyLong_FromLong(status_val);
    }

    int *local = (int *)calloc((size_t)(dimension > 0 ? dimension : 1),
                               sizeof(int));
    if (local == NULL) {
        free(data_shape); free(effective); free(r_start); free(r_step);
        free(origin_d); free(scale_d);
        PyErr_NoMemory();
        return NULL;
    }

    for (long long k = 0; k < total; ++k) {
        PyObject *real = PyTuple_New(dimension);
        PyObject *logical = PyTuple_New(dimension);
        PyObject *elements = PyList_New(0);
        if (real == NULL || logical == NULL || elements == NULL) {
            Py_XDECREF(real); Py_XDECREF(logical); Py_XDECREF(elements);
            free(local); free(data_shape); free(effective); free(r_start);
            free(r_step); free(origin_d); free(scale_d);
            return NULL;
        }
        for (int i = 0; i < dimension; ++i) {
            double rr = (double)((long long)r_start[i]
                                 + (long long)local[i] * r_step[i]);
            PyTuple_SET_ITEM(real, i, PyLong_FromLong(
                (long long)r_start[i] + (long long)local[i] * r_step[i]));
            /* logical: keep an integer when exactly divisible (type
             * safety), otherwise float */
            {
                double logical_val = rr * scale_d[i] - origin_d[i];
                long long li = (long long)logical_val;
                if ((double)li == logical_val) {
                    PyTuple_SET_ITEM(logical, i,
                                     PyLong_FromLongLong(li));
                } else {
                    PyTuple_SET_ITEM(logical, i,
                                     PyFloat_FromDouble(logical_val));
                }
            }
        }
        for (Py_ssize_t t = 1; t < n; ++t) {
            PyObject *get_args = PyTuple_Pack(2, PyTuple_GET_ITEM(args, t),
                                              real);
            PyObject *item = get_args ? py_get_item(self, get_args) : NULL;
            Py_XDECREF(get_args);
            if (item == NULL) {
                PyErr_Clear();
                Py_DECREF(real); Py_DECREF(logical); Py_DECREF(elements);
                free(local); free(data_shape); free(effective);
                free(r_start); free(r_step); free(origin_d); free(scale_d);
                return PyLong_FromLong(1);  /* read failure -> status 1 */
            }
            if (PyList_Append(elements, item) < 0) {
                Py_DECREF(item); Py_DECREF(real); Py_DECREF(logical);
                Py_DECREF(elements);
                free(local); free(data_shape); free(effective);
                free(r_start); free(r_step); free(origin_d); free(scale_d);
                return NULL;
            }
            Py_DECREF(item);
        }
        PyObject *cb_args = PyTuple_Pack(2, elements, logical);
        PyObject *cb_res = cb_args ? PyObject_CallObject(callback, cb_args)
                                   : NULL;
        Py_XDECREF(cb_args);
        Py_DECREF(elements);
        Py_DECREF(logical);
        if (cb_res == NULL) {
            PyErr_Clear();  /* callback errors silently skipped */
        } else {
            PyObject *set_args = PyTuple_Pack(3, output, real, cb_res);
            PyObject *r = set_args ? py_set_item(self, set_args) : NULL;
            Py_XDECREF(set_args);
            Py_DECREF(cb_res);
            if (r == NULL) {
                PyErr_Clear();
                Py_DECREF(real);
                free(local); free(data_shape); free(effective);
                free(r_start); free(r_step); free(origin_d); free(scale_d);
                return PyLong_FromLong(1);
            }
            Py_DECREF(r);
        }
        Py_DECREF(real);
        for (int i = dimension - 1; i >= 0; --i) {
            local[i]++;
            if (local[i] < effective[i]) break;
            local[i] = 0;
        }
    }
    free(local); free(data_shape); free(effective); free(r_start);
    free(r_step); free(origin_d); free(scale_d);
    return PyLong_FromLong(0);
}

static PyObject* py_elementwise(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t n = PyTuple_Size(args);
    if (n < 1) {
        PyErr_SetString(PyExc_TypeError, "elementwise requires at least one tensor");
        return NULL;
    }
    PyObject *func = kwargs ? PyDict_GetItemString(kwargs, "func") : NULL;
    if (func == NULL || !PyCallable_Check(func)) {
        PyErr_SetString(PyExc_TypeError, "func is required");
        return NULL;
    }
    PyObject *output = kwargs ? PyDict_GetItemString(kwargs, "output") : NULL;
    if (output == NULL) {
        PyErr_SetString(PyExc_ValueError, "output is required");
        return NULL;
    }
    PyObject *iterate = kwargs ? PyDict_GetItemString(kwargs, "iterate")
                               : NULL;

    int *shape = NULL;
    int dimension = 0;
    if (infer_shape(PyTuple_GET_ITEM(args, 0), &shape, &dimension) < 0)
        return NULL;
    for (Py_ssize_t i = 1; i < n; ++i) {
        int *other = NULL;
        int other_dim = 0;
        if (infer_shape(PyTuple_GET_ITEM(args, i), &other, &other_dim) < 0) {
            free(shape);
            return NULL;
        }
        if (other_dim != dimension ||
            memcmp(shape, other, (size_t)dimension * sizeof(int)) != 0) {
            free(shape); free(other);
            PyErr_SetString(PyExc_ValueError,
                            "the shape of two tensors are not same.");
            return NULL;
        }
        free(other);
    }
    {
        int *out_shape = NULL;
        int out_dim = 0;
        if (infer_shape(output, &out_shape, &out_dim) < 0) {
            free(shape);
            return NULL;
        }
        if (out_dim != dimension ||
            memcmp(shape, out_shape, (size_t)dimension * sizeof(int)) != 0) {
            free(shape); free(out_shape);
            PyErr_SetString(PyExc_ValueError,
                            "output shape does not match input");
            return NULL;
        }
        free(out_shape);
    }

    if (iterate != NULL && iterate != Py_None) {
        /* optional custom iterator: hand it the C kernel + index info */
        PyObject *res = NULL;
        PyObject *kernel = _make_kernel(&_kernel_def_elementwise, 2);
        PyObject *call_args = kernel ? PyTuple_Pack(1, kernel) : NULL;
        Py_XDECREF(kernel);
        PyObject *call_kwargs = call_args ? PyDict_New() : NULL;
        PyObject *shape_t = call_kwargs
            ? _ints_to_tuple(shape, dimension) : NULL;
        if (shape_t != NULL) {
            PyDict_SetItemString(call_kwargs, "data_shape", shape_t);
            PyDict_SetItemString(call_kwargs, "tensors", args);
            PyDict_SetItemString(call_kwargs, "func", func);
            PyDict_SetItemString(call_kwargs, "output", output);
            res = PyObject_Call(iterate, call_args, call_kwargs);
        }
        Py_XDECREF(shape_t); Py_XDECREF(call_kwargs); Py_XDECREF(call_args);
        free(shape);
        if (res == NULL) return NULL;
        Py_DECREF(res);
        return PyLong_FromLong(0);
    }

    long long total_ll = 1;
    for (int i = 0; i < dimension; ++i)
        total_ll *= (long long)shape[i];

    int *idx = (int*)calloc((size_t)(dimension > 0 ? dimension : 1),
                            sizeof(int));
    if (!idx) { free(shape); PyErr_NoMemory(); return NULL; }

    for (long long k = 0; k < total_ll; ++k) {
        PyObject *index = PyTuple_New(dimension);
        if (!index) { free(idx); free(shape); return NULL; }
        for (int i = 0; i < dimension; ++i)
            PyTuple_SET_ITEM(index, i, PyLong_FromLong(idx[i]));

        PyObject *vals = PyTuple_New(n);
        if (!vals) { Py_DECREF(index); free(idx); free(shape); return NULL; }
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject *gi = PyTuple_Pack(2, PyTuple_GET_ITEM(args, i), index);
            if (!gi) { Py_DECREF(vals); Py_DECREF(index); free(idx); free(shape); return NULL; }
            PyObject *value = py_get_item(self, gi);
            Py_DECREF(gi);
            if (!value) { Py_DECREF(vals); Py_DECREF(index); free(idx); free(shape); return NULL; }
            PyTuple_SET_ITEM(vals, i, value);
        }
        Py_DECREF(index);

        /* func receives the values as positional arguments in tensor order */
        PyObject *call_args = PyTuple_New(n);
        if (!call_args) { Py_DECREF(vals); free(idx); free(shape); return NULL; }
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject *item = PyTuple_GET_ITEM(vals, i);
            Py_INCREF(item);   /* SET_ITEM steals: own the borrowed item */
            PyTuple_SET_ITEM(call_args, i, item);
        }
        PyObject *cb_res = PyObject_Call(func, call_args, NULL);
        Py_DECREF(call_args);
        Py_DECREF(vals);
        if (!cb_res) {
            free(idx); free(shape);
            return NULL;   /* callback errors propagate */
        }
        double num = PyFloat_AsDouble(cb_res);
        Py_DECREF(cb_res);
        if (num == -1.0 && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError,
                            "elementwise func must return a number");
            free(idx); free(shape);
            return NULL;
        }

        {
            PyObject *write = PyTuple_New(dimension);
            for (int i = 0; i < dimension; ++i)
                PyTuple_SET_ITEM(write, i, PyLong_FromLong(idx[i]));
            PyObject *set_args = PyTuple_Pack(3, output, write,
                                              PyFloat_FromDouble(num));
            Py_DECREF(write);
            PyObject *r = py_set_item(self, set_args);
            Py_DECREF(set_args);
            if (!r) { free(idx); free(shape); return NULL; }
            Py_DECREF(r);
        }

        for (int d = dimension - 1; d >= 0; --d) {
            idx[d]++;
            if (idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }
    free(idx);
    free(shape);
    return PyLong_FromLong(0);
}

/* ------------------------------------------------------------------
Method table
------------------------------------------------------------------ */
static PyMethodDef methods[] = {
    {"multiple_chain", (PyCFunction)py_multiple_chain, METH_VARARGS | METH_KEYWORDS,
        "Multiply all elements in an iterable."},
    {"add_chain", (PyCFunction)py_add_chain, METH_VARARGS | METH_KEYWORDS,
        "Add all elements in an iterable."},
    {"create_void_list", (PyCFunction)py_create_void_list, METH_VARARGS | METH_KEYWORDS,
        "Create a multi-dimensional nested list filled with default value."},
    {"load_as_default_data", (PyCFunction)py_load_as_default_data, METH_VARARGS | METH_KEYWORDS,
        "Load data as a default data type."},
    {"load_data", (PyCFunction)py_load_data, METH_VARARGS | METH_KEYWORDS,
        "Copy a sub-region from source to target with independent start/step."},
    {"infer_shape", (PyCFunction)py_infer_shape, METH_VARARGS,
        "Infer the shape of multi-dimensional data."},
    {"get_item", py_get_item, METH_VARARGS,
        "Get item from nested list with multi-dimensional index."},
    {"set_item", py_set_item, METH_VARARGS,
        "Set item in nested list with multi-dimensional index."},
    {"data_filter", (PyCFunction)py_data_filter, METH_VARARGS | METH_KEYWORDS,
        "Yield positions whose callback(value) is truthy over a sampled region."},
    {"data_mapping", (PyCFunction)py_data_mapping, METH_VARARGS | METH_KEYWORDS,
        "Map every sampled element through callback(value) into the output."},
    {"position_map", (PyCFunction)py_position_map,
     METH_VARARGS | METH_KEYWORDS,
     "position_map(output, callback, start/shape/step, origin/scale): "
     "position-driven element callback -> 0."},
    {"elementwise_position", (PyCFunction)py_elementwise_position,
     METH_VARARGS | METH_KEYWORDS,
     "elementwise_position(output, *tensors, callback=..., origin/scale): "
     "multi-tensor element callback by position -> 0."},
    {"elementwise", (PyCFunction)py_elementwise, METH_VARARGS | METH_KEYWORDS,
        "Element-wise operation: elementwise(*tensors, func=f, output=o) -> 0."},
    {"_cos", py_cos, METH_VARARGS, "inner cos algorithm."},
    {"_mod", py_mod, METH_VARARGS, "inner mod algorithm."},
    {"_cosmod", py_cosmod, METH_VARARGS, "inner cosmod algorithm."},
    {"_convolution", py_convolution, METH_VARARGS, "inner convolution algorithm."},
    {"no_done", (PyCFunction)py_no_done, METH_VARARGS | METH_KEYWORDS, "Placeholder callback that does nothing."},
    {"sqrt", py_sqrt, METH_VARARGS, "Square root function (C math.h implementation)."},
    {"cos_comparison_passive", (PyCFunction)py_passive, METH_VARARGS | METH_KEYWORDS,
        "Passive mode: sliding window local comparison."},
    {"cos_comparison_passive_1d", (PyCFunction)py_passive, METH_VARARGS | METH_KEYWORDS,
        "Passive mode (1D alias)."},
    {"cos_comparison_passive_2d", (PyCFunction)py_passive, METH_VARARGS | METH_KEYWORDS,
        "Passive mode (2D alias)."},
    {"cos_comparison_passive_3d", (PyCFunction)py_passive, METH_VARARGS | METH_KEYWORDS,
        "Passive mode (3D alias)."},
    {"cos_comparison_passive_4d", (PyCFunction)py_passive, METH_VARARGS | METH_KEYWORDS,
        "Passive mode (4D alias)."},
    {"cos_comparison_active", (PyCFunction)py_active, METH_VARARGS | METH_KEYWORDS,
        "Active mode: template matching with kernel."},
    {"cos_comparison_active_1d", (PyCFunction)py_active, METH_VARARGS | METH_KEYWORDS,
        "Active mode (1D alias)."},
    {"cos_comparison_active_2d", (PyCFunction)py_active, METH_VARARGS | METH_KEYWORDS,
        "Active mode (2D alias)."},
    {"cos_comparison_active_3d", (PyCFunction)py_active, METH_VARARGS | METH_KEYWORDS,
        "Active mode (3D alias)."},
    {"cos_comparison_active_4d", (PyCFunction)py_active, METH_VARARGS | METH_KEYWORDS,
        "Active mode (4D alias)."},
    {"cos", (PyCFunction)py_cos_full, METH_VARARGS | METH_KEYWORDS,
        "Compute similarity between two whole tensors."},
    {"cos_1d", (PyCFunction)py_cos_full, METH_VARARGS | METH_KEYWORDS,
        "Compute similarity (1D alias)."},
    {"cos_2d", (PyCFunction)py_cos_full, METH_VARARGS | METH_KEYWORDS,
        "Compute similarity (2D alias)."},
    {"cos_3d", (PyCFunction)py_cos_full, METH_VARARGS | METH_KEYWORDS,
        "Compute similarity (3D alias)."},
    {"cos_4d", (PyCFunction)py_cos_full, METH_VARARGS | METH_KEYWORDS,
        "Compute similarity (4D alias)."},
    {"mean_local", (PyCFunction)py_mean_local, METH_VARARGS | METH_KEYWORDS,
        "Compute local mean (average pooling)."},
    {"mean_local_1d", (PyCFunction)py_mean_local, METH_VARARGS | METH_KEYWORDS,
        "Local mean (1D alias)."},
    {"mean_local_2d", (PyCFunction)py_mean_local, METH_VARARGS | METH_KEYWORDS,
        "Local mean (2D alias)."},
    {"mean_local_3d", (PyCFunction)py_mean_local, METH_VARARGS | METH_KEYWORDS,
        "Local mean (3D alias)."},
    {"mean_local_4d", (PyCFunction)py_mean_local, METH_VARARGS | METH_KEYWORDS,
        "Local mean (4D alias)."},
    {"local_variance", (PyCFunction)py_local_variance, METH_VARARGS | METH_KEYWORDS,
        "Compute local variance."},
    {"vector_chain_compute", (PyCFunction)py_vector_chain_compute, METH_O,
        "Create chain compute state, returns (compute, fix, get) tuple."},
    {"local_variance_1d", (PyCFunction)py_local_variance, METH_VARARGS | METH_KEYWORDS,
        "Local variance (1D alias)."},
    {"local_variance_2d", (PyCFunction)py_local_variance, METH_VARARGS | METH_KEYWORDS,
        "Local variance (2D alias)."},
    {"local_variance_3d", (PyCFunction)py_local_variance, METH_VARARGS | METH_KEYWORDS,
        "Local variance (3D alias)."},
    {"local_variance_4d", (PyCFunction)py_local_variance, METH_VARARGS | METH_KEYWORDS,
        "Local variance (4D alias)."},
    {NULL, NULL, 0, NULL}
};

/* ------------------------------------------------------------------
Module definition (multi-phase: safe re-init, free-threaded ready)
------------------------------------------------------------------ */
static int module_exec(PyObject *module) {
    g_module = module;   /* module object for the iterate kernels */
    if (Add_Object(module)) {
        return -1;
    }

    // Add NaN constant
    if (PyModule_AddObject(module, "NaN", PyFloat_FromDouble(COS_NAN)) < 0) {
        return -1;
    }

    // Add private_dict (internal algorithm dictionary, matches pure Python backend)
    PyObject *private_dict = PyDict_New();
    if (!private_dict) {
        return -1;
    }
    PyObject *cos_algo = PyObject_GetAttrString(module, "_cos");
    PyObject *mod_algo = PyObject_GetAttrString(module, "_mod");
    PyObject *cosmod_algo = PyObject_GetAttrString(module, "_cosmod");
    PyObject *conv_algo = PyObject_GetAttrString(module, "_convolution");
    PyObject *default_algo = PyObject_GetAttrString(module, "_cosmod");
    if (!cos_algo || !mod_algo || !cosmod_algo || !conv_algo || !default_algo) {
        Py_XDECREF(cos_algo);
        Py_XDECREF(mod_algo);
        Py_XDECREF(cosmod_algo);
        Py_XDECREF(conv_algo);
        Py_XDECREF(default_algo);
        Py_DECREF(private_dict);
        return -1;
    }
    PyDict_SetItemString(private_dict, "_cos", cos_algo);
    PyDict_SetItemString(private_dict, "_mod", mod_algo);
    PyDict_SetItemString(private_dict, "_cosmod", cosmod_algo);
    PyDict_SetItemString(private_dict, "_convolution", conv_algo);
    PyDict_SetItemString(private_dict, "_default_algorithm", default_algo);
    Py_DECREF(cos_algo);
    Py_DECREF(mod_algo);
    Py_DECREF(cosmod_algo);
    Py_DECREF(conv_algo);
    Py_DECREF(default_algo);
    if (PyModule_AddObject(module, "private_dict", private_dict) < 0) {
        Py_DECREF(private_dict);
        return -1;
    }

    // Register VectorChainCompute native type
    if (PyType_Ready(&VectorChainComputeType) < 0) {
        return -1;
    }

    // Add all_names (public API list, matches pure Python backend)
    PyObject *all_names = PyList_New(0);
    if (!all_names) {
        return -1;
    }
    {
        static const char *const _all_names[] = {
            "NaN", "sqrt",
            "cos_comparison_passive", "cos_comparison_passive_1d", "cos_comparison_passive_2d", "cos_comparison_passive_3d", "cos_comparison_passive_4d",
            "cos_comparison_active", "cos_comparison_active_1d", "cos_comparison_active_2d", "cos_comparison_active_3d", "cos_comparison_active_4d",
            "cos", "cos_1d", "cos_2d", "cos_3d", "cos_4d",
            "mean_local", "mean_local_1d", "mean_local_2d", "mean_local_3d", "mean_local_4d",
            "local_variance", "local_variance_1d", "local_variance_2d", "local_variance_3d", "local_variance_4d",
            "multiple_chain", "add_chain", "no_done", "create_void_list", "load_as_default_data", "load_data", "infer_shape", "get_item", "set_item", "_cos", "_mod", "_cosmod", "_convolution",
            "data_filter", "data_mapping", "elementwise", "position_map", "elementwise_position", "threshold_filter", "threshold_map", "threshold_judge",
            "vector_chain_compute",
            "vector_map_as_tensor", "func_name_space", "default_contain",
            "private_dict"
        };
        size_t i;
        for (i = 0; i < sizeof(_all_names) / sizeof(_all_names[0]); ++i) {
            PyObject *name = PyUnicode_FromString(_all_names[i]);
            if (name == NULL || PyList_Append(all_names, name) < 0) {
                Py_XDECREF(name);
                Py_DECREF(all_names);
                return -1;
            }
            Py_DECREF(name);
        }
    }
    if (PyModule_AddObject(module, "all_names", all_names) < 0) {
        Py_DECREF(all_names);
        return -1;
    }

    /* Backfill the threshold_* wrappers from the pure Python backend so the
       module surface matches (data iteration itself is native). */
    {
        static const char *const _fill_names[] = {
            "threshold_filter", "threshold_map", "threshold_judge"
        };
        PyObject *pure = PyImport_ImportModule(
            "cos_comparison.core.cos_comparison");
        if (pure) {
            for (size_t i = 0; i < sizeof(_fill_names) / sizeof(_fill_names[0]); ++i) {
                PyObject *attr = PyObject_GetAttrString(pure, _fill_names[i]);
                if (!attr) { PyErr_Clear(); continue; }
                if (PyModule_AddObject(module, (char*)_fill_names[i], attr) < 0) {
                    Py_DECREF(attr);
                }
            }
            Py_DECREF(pure);
        } else {
            PyErr_Clear();
        }
    }

    return 0;
}

static PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, (void*)module_exec},
#if PY_VERSION_HEX >= 0x030D0000
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cos_comparison_pydll",
    "cos-comparison C extension backend.",
    0,
    methods,
    module_slots,
    NULL,
    NULL,
    NULL
};

/* ------------------------------------------------------------------
Module init
------------------------------------------------------------------ */
PyMODINIT_FUNC PyInit_cos_comparison_pydll(void) {
    return PyModuleDef_Init(&moduledef);
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
