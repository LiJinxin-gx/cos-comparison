#ifdef Py_DEBUG
#undef Py_DEBUG
#endif
#ifdef _DEBUG
#undef _DEBUG
#endif

#include <math.h>
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#endif
#include "type.h"
#include "core.h"
#include "type_vector.h"

/* ------------------------------------------------------------------
Helper: get nested item by multi-dimensional indices (iterative)
------------------------------------------------------------------ */
static PyObject* _get_nested_item(PyObject *obj, int *indices, int dim) {
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
Helper: flatten nested list to flat array (iterative, no recursion)
------------------------------------------------------------------ */
static int _flatten_list(PyObject *obj, double *out, int *idx, int dim, const int *shape) {
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
            PyObject *item = _get_nested_item(obj, indices, dim);
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
Helper: nested list / Vector → Data
------------------------------------------------------------------ */
static Data* _pyobj_to_data(PyObject *obj) {
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
        int stride = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            data->strides[i] = stride;
            stride *= data->shape[i];
        }
        int total = stride;
        /* Always copy data to ensure contiguous, handles non-contiguous views */
        data->data = (double*)malloc((size_t)(total) * sizeof(double));
        if (!data->data) { free(data->strides); free(data->shape); free(data); PyErr_NoMemory(); return NULL; }
        data->owns_data = 1;
        /* Iterate all elements with carry method */
        int *indices = (int*)malloc((size_t)(total) * sizeof(int));
        if (!indices) { Data_free(data); return NULL; }
        int *idx = (int*)malloc((size_t)(ndim) * sizeof(int));
        memset(idx, 0, ndim * sizeof(int));
        int pos = 0;
        while (1) {
            int flat = v->start + v->offset;
            for (int i = 0; i < ndim; ++i) {
                flat += v->strides[i] * (v->start_offset[i] + idx[i] * v->step_offset[i]);
            }
            indices[pos++] = flat;
            int dim = ndim - 1;
            while (dim >= 0) {
                idx[dim]++;
                if (idx[dim] < v->shape[dim]) break;
                idx[dim] = 0;
                dim--;
            }
            if (dim < 0) break;
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
    if (_infer_shape(obj, &shape, &dimension) < 0) return NULL;
    if (dimension == 0) {
        free(shape);
        PyErr_SetString(PyExc_ValueError, "not a tensor");
        return NULL;
    }
    Data *data = Data_create(dimension, shape);
    if (!data) { free(shape); PyErr_NoMemory(); return NULL; }
    int idx = 0;
    if (_flatten_list(obj, data->data, &idx, dimension, shape) < 0) {
        Data_free(data);
        free(shape);
        return NULL;
    }
    free(shape);
    return data;
}

/* ------------------------------------------------------------------
Helper: Data → vector_map_as_tensor (zero-copy)
------------------------------------------------------------------ */
static PyObject* _data_to_vector(Data *data, PyTypeObject *type) {
    if (!type) type = &VectorizeType;
    Vector *vec = (Vector*)type->tp_alloc(type, 0);
    if (!vec) { Data_free(data); return NULL; }
    vec->data = data;
    vec->owner = NULL;
    vec->dimension = data->dimension;
    vec->shape = (int*)malloc((size_t)(data->dimension) * sizeof(int));
    if (!vec->shape) { Py_DECREF(vec); Data_free(data); return NULL; }
    memcpy(vec->shape, data->shape, data->dimension * sizeof(int));
    // Allocate and compute strides
    vec->strides = (int*)malloc((size_t)(data->dimension) * sizeof(int));
    if (!vec->strides) { free(vec->shape); Py_DECREF(vec); Data_free(data); return NULL; }
    vec->strides[data->dimension - 1] = 1;
    for (int i = data->dimension - 2; i >= 0; --i) {
        vec->strides[i] = vec->strides[i+1] * vec->shape[i+1];
    }
    vec->start = 0;
    vec->offset = 0;
    vec->start_offset = (int*)malloc((size_t)(data->dimension) * sizeof(int));
    vec->step_offset = (int*)malloc((size_t)(data->dimension) * sizeof(int));
    for (int i = 0; i < data->dimension; ++i) {
        vec->start_offset[i] = 0;
        vec->step_offset[i] = 1;
    }
    vec->buf = NULL;
    vec->flags = VECTOR_FLAG_OWNED;
    // tp_alloc already tracks GC objects in Python 3.14, no need to call again
    // PyObject_GC_Track(vec);
    return (PyObject*)vec;
}

/* ------------------------------------------------------------------
Helper: parse int tuple/list
------------------------------------------------------------------ */
static int _parse_int_seq(PyObject *obj, int **out, int *count) {
    if (!PySequence_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "expected sequence");
        return -1;
    }
    Py_ssize_t n = PySequence_Size(obj);
    int *arr = (int*)malloc((size_t)(n) * sizeof(int));
    if (!arr) { PyErr_NoMemory(); return -1; }
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = PySequence_GetItem(obj, i);
        if (!item) { free(arr); return -1; }
        PyObject *num = PyNumber_Index(item);
        Py_DECREF(item);
        if (!num) { free(arr); return -1; }
        arr[i] = (int)PyLong_AsLong(num);
        Py_DECREF(num);
        if (PyErr_Occurred()) { free(arr); return -1; }
    }
    *out = arr;
    *count = (int)n;
    return 0;
}

/* ------------------------------------------------------------------
Utility functions
------------------------------------------------------------------ */
static PyObject* py_multiple_chain(PyObject *self, PyObject *args, PyObject *kwargs) {
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
    PyObject *length_list_obj = NULL, *default_obj = Py_None;
    static char *kwlist[] = {"length_list", "default", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO", kwlist, &length_list_obj, &default_obj))
        return NULL;
    int *shape = NULL; int dim = 0;
    if (length_list_obj) {
        if (_parse_int_seq(length_list_obj, &shape, &dim) < 0) return NULL;
    } else {
        shape = (int*)malloc(sizeof(int)); shape[0] = 1; dim = 1;
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
    return _data_to_vector(data, NULL);
}

/* Helper: create independent Vector from Data (deep copy) */
static PyObject* _data_to_independent_vector(Data *data, int start) {
    Vector *vec = (Vector*)VectorizeType.tp_alloc(&VectorizeType, 0);
    if (!vec) { Data_free(data); return NULL; }
    vec->dimension = data->dimension;
    vec->shape = (int*)malloc((size_t)(data->dimension) * sizeof(int));
    if (!vec->shape) { Py_DECREF(vec); Data_free(data); return NULL; }
    memcpy(vec->shape, data->shape, data->dimension * sizeof(int));
    // Allocate and compute strides
    vec->strides = (int*)malloc((size_t)(data->dimension) * sizeof(int));
    if (!vec->strides) { free(vec->shape); Py_DECREF(vec); Data_free(data); return NULL; }
    vec->strides[data->dimension - 1] = 1;
    for (int i = data->dimension - 2; i >= 0; --i) {
        vec->strides[i] = vec->strides[i+1] * vec->shape[i+1];
    }
    int total = Data_total(data);
    vec->data = Data_create(data->dimension, data->shape);
    if (!vec->data) { free(vec->strides); free(vec->shape); Py_DECREF(vec); Data_free(data); return NULL; }
    memcpy(vec->data->data, data->data, total * sizeof(double));
    vec->owner = NULL;
    vec->start = start;
    vec->offset = 0;
    vec->start_offset = (int*)malloc((size_t)(data->dimension) * sizeof(int));
    vec->step_offset = (int*)malloc((size_t)(data->dimension) * sizeof(int));
    for (int i = 0; i < data->dimension; ++i) {
        vec->start_offset[i] = 0;
        vec->step_offset[i] = 1;
    }
    vec->buf = NULL;
    Data_free(data);
    vec->flags = VECTOR_FLAG_OWNED;
    // tp_alloc already tracks GC objects in Python 3.14
    // PyObject_GC_Track(vec);
    return (PyObject*)vec;
}

static PyObject* py_infer_shape(PyObject *self, PyObject *args) {
    PyObject *data_obj;
    if (!PyArg_ParseTuple(args, "O", &data_obj))
        return NULL;
    
    int *shape = NULL;
    int dim = 0;
    
    if (_infer_shape(data_obj, &shape, &dim) < 0) {
        PyErr_Clear();
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
            }
            if (start_obj != Py_None && shape_obj != Py_None && 
                PySequence_Check(start_obj) && PySequence_Check(shape_obj)) {
                PyObject *s_item = PySequence_GetItem(start_obj, i);
                PyObject *sh_item = PySequence_GetItem(shape_obj, i);
                if (s_item && sh_item) {
                    e = PyNumber_Add(s_item, sh_item);
                }
                Py_XDECREF(s_item);
                Py_XDECREF(sh_item);
            }
            if (step_obj != Py_None && PySequence_Check(step_obj)) {
                st = PySequence_GetItem(step_obj, i);
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
        int total = _multiple_chain(res_vec->shape, res_vec->dimension);
        int *flat_indices = (int*)malloc((size_t)(total) * sizeof(int));
        if (!flat_indices) { Py_DECREF(result); return PyErr_NoMemory(); }
        _vector_get_flat_indices(res_vec, flat_indices, total);
        
        /* Build flat list of values */
        PyObject *new_list = PyList_New(total);
        if (!new_list) { free(flat_indices); Py_DECREF(result); return NULL; }
        for (int i = 0; i < total; ++i) {
            double val = Data_get_flat(res_vec->data, flat_indices[i]);
            PyList_SET_ITEM(new_list, i, PyFloat_FromDouble(val));
        }
        free(flat_indices);
        Py_DECREF(result);
        
        /* Build shape tuple */
        PyObject *shape_tuple = PyTuple_New(res_vec->dimension);
        if (!shape_tuple) { Py_DECREF(new_list); return NULL; }
        for (int i = 0; i < res_vec->dimension; ++i) {
            PyTuple_SET_ITEM(shape_tuple, i, PyLong_FromLong(res_vec->shape[i]));
        }
        
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
    if (_infer_shape(data_obj, &full_shape, &dimension) < 0)
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
            step[i] = (int)PyLong_AsLong(num); Py_DECREF(num);
            if (step[i] <= 0) {
                free(step); free(full_shape);
                PyErr_Format(PyExc_ValueError, "step[%d] = %d must be positive", i, step[i]);
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
            shape[i] = (int)PyLong_AsLong(num); Py_DECREF(num);
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
            start[i] = (int)PyLong_AsLong(num); Py_DECREF(num);
            if (start[i] < 0) {
                free(start); free(shape); free(step); free(full_shape);
                PyErr_Format(PyExc_ValueError, "start[%d] cannot be negative", i);
                return NULL;
            }
            /* Bounds check with step */
            if (shape[i] > 0 && start[i] + (shape[i] - 1) * step[i] >= full_shape[i]) {
                free(start); free(shape); free(step); free(full_shape);
                PyErr_Format(PyExc_ValueError,
                             "start[%d] + (shape[%d]-1)*step[%d] = %d out of bounds for dimension size %d",
                             i, i, i, start[i] + (shape[i] - 1) * step[i], full_shape[i]);
                return NULL;
            }
        }
    }

    /* Step 5: Flatten full input data into temporary Data */
    Data *full_data = _pyobj_to_data(data_obj);
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
    int total = 1;
    for (int i = 0; i < dimension; ++i) total *= shape[i];

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
    return _data_to_independent_vector(result_data, 0);
}

static PyObject* _get_item_recursive(PyObject *obj, PyObject *index, int depth) {
    PyObject *current = obj;
    Py_INCREF(current);
    Py_ssize_t n = PyTuple_Size(index);
    for (int i = depth; i < n; ++i) {
        PyObject *idx = PyTuple_GetItem(index, i);
        if (!idx) { Py_DECREF(current); return NULL; }
        if (!PySequence_Check(current)) { Py_DECREF(current); return NULL; }
        PyObject *next = PySequence_GetItem(current, PyLong_AsSsize_t(idx));
        Py_DECREF(current);
        if (!next) return NULL;
        current = next;
    }
    return current;
}

static PyObject* py_get_item(PyObject *self, PyObject *args) {
    PyObject *obj, *index;
    if (!PyArg_ParseTuple(args, "OO", &obj, &index)) return NULL;
    PyObject *get_item_method = PyObject_GetAttrString(obj, "__get_item__");
    if (get_item_method != NULL) {
        PyObject *index_args = PyTuple_Check(index) ? index : PyTuple_Pack(1, index);
        Py_INCREF(index_args);
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
            Py_ssize_t idx = PyLong_AsSsize_t(idx_obj);
            if (idx == -1 && PyErr_Occurred()) { Py_DECREF(temp); return NULL; }
            PyObject *next = PySequence_GetItem(temp, idx);
            Py_DECREF(temp);
            if (!next) return NULL;
            temp = next;
        }
        return temp;
    } else {
        Py_ssize_t idx = PyLong_AsSsize_t(index);
        if (idx == -1 && PyErr_Occurred()) return NULL;
        return PySequence_GetItem(obj, idx);
    }
}

static PyObject* py_set_item(PyObject *self, PyObject *args) {
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
            Py_ssize_t idx = PyLong_AsSsize_t(idx_obj);
            if (idx == -1 && PyErr_Occurred()) { Py_DECREF(temp); return NULL; }
            PyObject *next = PySequence_GetItem(temp, idx);
            Py_DECREF(temp);
            if (!next) return NULL;
            temp = next;
        }
        // Last index: set item
        PyObject *last_idx = PyTuple_GetItem(index, n - 1);
        Py_ssize_t idx = PyLong_AsSsize_t(last_idx);
        if (idx == -1 && PyErr_Occurred()) { Py_DECREF(temp); return NULL; }
        int result = PySequence_SetItem(temp, idx, value);
        Py_DECREF(temp);
        if (result < 0) return NULL;
        Py_RETURN_NONE;
    } else {
        Py_ssize_t idx = PyLong_AsSsize_t(index);
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
    double a, b, ab; PyObject *name;
    if (!PyArg_ParseTuple(args, "dddO", &a, &b, &ab, &name)) return NULL;
    return PyFloat_FromDouble(_cos_(a, b, ab, NULL));
}

static PyObject* py_mod(PyObject *self, PyObject *args) {
    double a, b, ab; PyObject *name;
    if (!PyArg_ParseTuple(args, "dddO", &a, &b, &ab, &name)) return NULL;
    return PyFloat_FromDouble(_mod_(a, b, ab, NULL));
}

static PyObject* py_cosmod(PyObject *self, PyObject *args) {
    double a, b, ab; PyObject *name;
    if (!PyArg_ParseTuple(args, "dddO", &a, &b, &ab, &name)) return NULL;
    return PyFloat_FromDouble(_cosmod_(a, b, ab, NULL));
}

static PyObject* py_no_done(PyObject *self, PyObject *args) {
    Py_RETURN_NONE;
}

static PyObject* py_sqrt(PyObject *self, PyObject *args) {
    double x;
    if (!PyArg_ParseTuple(args, "d", &x)) return NULL;
    return PyFloat_FromDouble(sqrt(x));
}

/* ------------------------------------------------------------------
Custom Python algorithm callback
------------------------------------------------------------------ */
static double _py_algo_wrapper(double a, double b, double ab, CallbackContext *ctx) {
    if (!ctx || !ctx->name_space) return 0.0;
    PyObject *py_algo = PyObject_GetAttrString(ctx->name_space, "algorithm");
    if (!py_algo || !PyCallable_Check(py_algo)) {
        Py_XDECREF(py_algo);
        return 0.0;
    }
    PyObject *result = PyObject_CallFunction(py_algo, "dddO", a, b, ab, ctx->name_space);
    Py_DECREF(py_algo);
    if (!result) { return 0.0; }
    double res = PyFloat_AsDouble(result);
    Py_DECREF(result);
    return res;
}

static algo_fn _get_algo(PyObject *name) {
    if (!name) return _cosmod_;
    if (PyUnicode_Check(name)) {
        const char *s = PyUnicode_AsUTF8(name);
        if (!s) return NULL;
        if (strcmp(s, "cos") == 0) return _cos_;
        if (strcmp(s, "mod") == 0) return _mod_;
        if (strcmp(s, "cosmod") == 0) return _cosmod_;
        PyErr_SetString(PyExc_ValueError, "unknown algorithm"); return NULL;
    }
    if (PyCallable_Check(name)) {
        return _py_algo_wrapper;
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
    for (int i = 0; i < dim; ++i) {
        int eff = end[i] - start[i] - window_size[i] - d[i];
        if (eff < 0) { free(num); return NULL; }
        num[i] = eff / step[i] + 1;
    }
    int output_is_data = (output != NULL);
    if (!output_is_data) {
        output = _compute_output_shape(dim, num, output_start, output_step);
        if (!output) { free(num); return NULL; }
    }
    int *num_list = (int*)malloc(((size_t)(dim) + 1) * sizeof(int));
    int *inner_list = (int*)malloc(((size_t)(dim) + 1) * sizeof(int));
    int *main_place = (int*)malloc((size_t)(dim) * sizeof(int));
    int *other_place = (int*)malloc((size_t)(dim) * sizeof(int));
    int *out_idx = (int*)malloc((size_t)(dim) * sizeof(int));
    for (int i = 0; i <= dim; ++i) { num_list[i] = 1; inner_list[i] = 1; }
    int flag = dim;
    double main_sum, other_sum, mu_sum;
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
                                   : _cosmod_(main_sum, other_sum, mu_sum, ctx);
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
    return output_is_data ? output : output;
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
    for (int i = 0; i < dim; ++i) {
        int eff = end[i] - start[i] - window_size[i];
        if (eff < 0) { free(num); return NULL; }
        num[i] = eff / step[i] + 1;
    }
    int output_is_data = (output != NULL);
    if (!output_is_data) {
        output = _compute_output_shape(dim, num, output_start, output_step);
        if (!output) { free(num); return NULL; }
    }
    int *num_list = (int*)malloc(((size_t)(dim) + 1) * sizeof(int));
    int *inner_list = (int*)malloc(((size_t)(dim) + 1) * sizeof(int));
    int *data_place = (int*)malloc((size_t)(dim) * sizeof(int));
    int *kern_place = (int*)malloc((size_t)(dim) * sizeof(int));
    int *out_idx = (int*)malloc((size_t)(dim) * sizeof(int));
    for (int i = 0; i <= dim; ++i) { num_list[i] = 1; inner_list[i] = 1; }
    int flag = dim;
    double main_sum, other_sum, mu_sum;
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
                                   : _cosmod_(main_sum, other_sum, mu_sum, ctx);
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
    return output_is_data ? output : output;
}

/* Full tensor similarity */
double cos_full(const Data *a, const Data *b, algo_fn algorithm, CallbackContext *ctx) {
    if (!Data_shape_equal(a, b)) {
        return NAN; /* Shape mismatch, return NaN intentionally */
    }
    int total = Data_total(a);
    double sum_a = 0.0, sum_b = 0.0, sum_ab = 0.0;
    COS_SIMD_LOOP
    for (int i = 0; i < total; ++i) {
        double va = Data_get_flat(a, i);
        double vb = Data_get_flat(b, i);
        sum_a += va * va;
        sum_b += vb * vb;
        sum_ab += va * vb;
    }
    return algorithm ? algorithm(sum_a, sum_b, sum_ab, ctx)
                     : _cosmod_(sum_a, sum_b, sum_ab, ctx);
}

/* Local mean */
Data* cos_local_mean(const Data *data,
                     const int window_size[],
                     const int start[], const int end[],
                     const int step[],
                     const int output_start[], const int output_step[],
                     PyObject *output_obj, Data *output, const double weights[]) {
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
    
    int *num = (int*)malloc((size_t)(dim) * sizeof(int));
    for (int i = 0; i < dim; ++i) {
        int eff = end[i] - start[i] - window_size[i];
        if (eff < 0) { free(num); return NULL; }
        num[i] = eff / step[i] + 1;
    }
    int output_is_data = (output != NULL);
    if (!output_is_data) {
        output = _compute_output_shape(dim, num, output_start, output_step);
        if (!output) { free(num); return NULL; }
    }
    int N = 1;
    for (int i = 0; i < dim; ++i) N *= window_size[i];
    /* Precompute window strides so weights are indexed in row-major order */
    int *wstrides = (int*)malloc((size_t)(dim) * sizeof(int));
    if (!wstrides) { free(num); if (!output_is_data) Data_free(output); PyErr_NoMemory(); return NULL; }
    {
        int acc = 1;
        for (int i = dim - 1; i >= 0; --i) { wstrides[i] = acc; acc *= window_size[i]; }
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
    double sum_x;
    while (flag > 0) {
        if (flag == dim) {
            for (int i = 1; i <= dim; ++i) inner_list[i] = 1;
            int inner_flag = dim;
            sum_x = 0.0;
            while (inner_flag > 0) {
                if (inner_flag == dim) {
                    int wflat = 0;
                    for (int i = 0; i < dim; ++i) {
                        data_place[i] = start[i] + step[i] * (num_list[i+1] - 1) + (inner_list[i+1] - 1);
                        wflat += (inner_list[i+1] - 1) * wstrides[i];
                    }
                    double x = Data_get(data, data_place);
                    sum_x += weights ? weights[wflat] * x : x;
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
    return output_is_data ? output : output;
}

/* Local variance */
/* Local variance */
Data* cos_local_variance(const Data *data,
                         const int window_size[],
                         const int start[], const int end[],
                         const int step[],
                         const int output_start[], const int output_step[],
                         PyObject *output_obj, Data *output) {
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
    
    int *num = (int*)malloc((size_t)(dim) * sizeof(int));
    for (int i = 0; i < dim; ++i) {
        int eff = end[i] - start[i] - window_size[i];
        if (eff < 0) { free(num); return NULL; }
        num[i] = eff / step[i] + 1;
    }
    int output_is_data = (output != NULL);
    if (!output_is_data) {
        output = _compute_output_shape(dim, num, output_start, output_step);
        if (!output) { free(num); return NULL; }
    }
    int N = 1;
    for (int i = 0; i < dim; ++i) N *= window_size[i];
    int *num_list = (int*)malloc(((size_t)(dim) + 1) * sizeof(int));
    int *inner_list = (int*)malloc(((size_t)(dim) + 1) * sizeof(int));
    int *data_place = (int*)malloc((size_t)(dim) * sizeof(int));
    int *out_idx = (int*)malloc((size_t)(dim) * sizeof(int));
    for (int i = 0; i <= dim; ++i) { num_list[i] = 1; inner_list[i] = 1; }
    int flag = dim;
    double sum_x, sum_x2;
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
    return output_is_data ? output : output;
}

/* ------------------------------------------------------------------
Python wrapper functions (full implementations)
------------------------------------------------------------------ */

/* Passive mode Python wrapper */
static PyObject* py_passive(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *data_obj, *window_size_obj = NULL, *start_obj = NULL, *end_obj = NULL,
             *step_obj = NULL, *d_obj = NULL, *algo_name = NULL, *output_obj = NULL,
             *output_start_obj = NULL, *output_step_obj = NULL, *start_callback = NULL,
             *end_callback = NULL, *global_error_callback = NULL, *local_error_callback = NULL,
             *return_callback = NULL;
    double w1 = 1.0, w2 = 1.0, b1 = 0.0, b2 = 0.0;
    int release_gil = 0;
    static char *kwlist[] = {
        "data", "window_size", "w1", "w2", "b1", "b2",
        "start", "end", "step", "d", "algorithm",
        "output", "output_start", "output_step",
        "start_callback", "end_callback",
        "global_error_callback", "local_error_callback",
        "return_callback", "release_gil", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OddddOOOOOOOOOOOOOp", kwlist,
                                     &data_obj, &window_size_obj,
                                     &w1, &w2, &b1, &b2,
                                     &start_obj, &end_obj, &step_obj, &d_obj,
                                     &algo_name, &output_obj,
                                     &output_start_obj, &output_step_obj,
                                     &start_callback, &end_callback,
                                     &global_error_callback, &local_error_callback,
                                     &return_callback, &release_gil))
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

    Data *data = _pyobj_to_data(data_obj);
    if (!data) return NULL;
    int dim = data->dimension;

    int *window_size = NULL; int ws_count = 0;
    if (window_size_obj) {
        if (_parse_int_seq(window_size_obj, &window_size, &ws_count) < 0) { Data_free(data); return NULL; }
    } else {
        window_size = (int*)malloc((size_t)(dim) * sizeof(int));
        for (int i = 0; i < dim; ++i) window_size[i] = 1;
        ws_count = dim;
    }

    int *start = (int*)malloc((size_t)(dim) * sizeof(int));
    if (start_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(start_obj, &tmp, &cnt) < 0) { free(window_size); Data_free(data); return NULL; }
        memcpy(start, tmp, dim * sizeof(int)); free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) start[i] = 0;
    }

    int *end = (int*)malloc((size_t)(dim) * sizeof(int));
    if (end_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(end_obj, &tmp, &cnt) < 0) { free(start); free(window_size); Data_free(data); return NULL; }
        memcpy(end, tmp, dim * sizeof(int)); free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) end[i] = data->shape[i];
    }

    int *step = (int*)malloc((size_t)(dim) * sizeof(int));
    if (step_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(step_obj, &tmp, &cnt) < 0) { free(end); free(start); free(window_size); Data_free(data); return NULL; }
        memcpy(step, tmp, dim * sizeof(int)); free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) step[i] = 1;
    }

    int *d = (int*)malloc((size_t)(dim) * sizeof(int));
    if (d_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(d_obj, &tmp, &cnt) < 0) { free(step); free(end); free(start); free(window_size); Data_free(data); return NULL; }
        memcpy(d, tmp, dim * sizeof(int)); free(tmp);
    } else {
        d[0] = 1; for (int i = 1; i < dim; ++i) d[i] = 0;
    }

    int *output_start = (int*)malloc((size_t)(dim) * sizeof(int));
    if (output_start_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(output_start_obj, &tmp, &cnt) < 0) { free(d); free(step); free(end); free(start); free(window_size); Data_free(data); return NULL; }
        memcpy(output_start, tmp, dim * sizeof(int)); free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) output_start[i] = 0;
    }

    int *output_step = (int*)malloc((size_t)(dim) * sizeof(int));
    if (output_step_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(output_step_obj, &tmp, &cnt) < 0) { free(output_start); free(d); free(step); free(end); free(start); free(window_size); Data_free(data); return NULL; }
        memcpy(output_step, tmp, dim * sizeof(int)); free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) output_step[i] = 1;
    }

    algo_fn algo = _get_algo(algo_name);
    if (!algo) {
        free(output_step); free(output_start); free(d); free(step);
        free(end); free(start); free(window_size); Data_free(data);
        if (global_error_callback && PyCallable_Check(global_error_callback)) {
            PyObject *ns = PyObject_CallObject((PyObject*)&FuncNameSpaceType, NULL);
            if (ns) {
                PyObject *res = PyObject_CallFunctionObjArgs(global_error_callback, ns, NULL);
                Py_XDECREF(res); Py_DECREF(ns);
            }
        }
        return NULL;
    }

    PyObject *name_space = NULL;
    CallbackContext ctx = {0};
    if (start_callback || end_callback || global_error_callback || local_error_callback || return_callback || PyCallable_Check(algo_name)) {
        name_space = PyObject_CallObject((PyObject*)&FuncNameSpaceType, NULL);
        if (name_space) {
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
            PyObject *linear_tuple = PyTuple_New(4);
            PyTuple_SET_ITEM(linear_tuple, 0, PyFloat_FromDouble(w1));
            PyTuple_SET_ITEM(linear_tuple, 1, PyFloat_FromDouble(w2));
            PyTuple_SET_ITEM(linear_tuple, 2, PyFloat_FromDouble(b1));
            PyTuple_SET_ITEM(linear_tuple, 3, PyFloat_FromDouble(b2));
            PyObject_SetAttrString(name_space, "linear", linear_tuple); Py_DECREF(linear_tuple);
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
            for (int i = 0; i < dim; ++i) num[i] = (end[i] - start[i] - window_size[i] - d[i]) / step[i] + 1;
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
        memcpy(saved_os, output_start, dim * sizeof(int));
        memcpy(saved_ost, output_step, dim * sizeof(int));
        for (int i = 0; i < dim; ++i) {
            output_start[i] = 0;
            output_step[i] = 1;
        }
    }
    if (release_gil && !name_space) {
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
            if (exc) { PyErr_Clear(); PyObject *res = PyObject_CallFunctionObjArgs(global_error_callback, exc, name_space, NULL); Py_XDECREF(res); }
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
        // Check if output is our C Vector type for direct write
        Data *out_data = NULL;
        if (PyObject_IsInstance(output_obj, (PyObject*)&VectorizeType)) {
            Vector *vec = (Vector*)output_obj;
            out_data = vec->data;
        }
        for (int i = 0; i < r_dim; ++i) idx[i] = 0;
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
                    _py_set_item(output_obj, out_idx, r_dim, 0, val);
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
        py_result = _data_to_vector(result, result_type);
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
    free(output_step); free(output_start); free(d); free(step);
    free(end); free(start); free(window_size); Data_free(data);

    Py_XDECREF(name_space);
    return py_result;
}

/* Active mode Python wrapper */
static PyObject* py_active(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *data_obj, *kernel_obj = NULL, *start_obj = NULL, *end_obj = NULL,
             *step_obj = NULL, *algo_name = NULL, *output_obj = NULL,
             *output_start_obj = NULL, *output_step_obj = NULL, *start_callback = NULL,
             *end_callback = NULL, *global_error_callback = NULL, *local_error_callback = NULL,
             *return_callback = NULL;
    double w1 = 1.0, w2 = 1.0, b1 = 0.0, b2 = 0.0;
    int release_gil = 0;
    static char *kwlist[] = {
        "data", "kernel", "w1", "w2", "b1", "b2",
        "start", "end", "step", "algorithm",
        "output", "output_start", "output_step",
        "start_callback", "end_callback",
        "global_error_callback", "local_error_callback",
        "return_callback", "release_gil", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|ddddOOOOOOOOOOOOp", kwlist,
                                     &data_obj, &kernel_obj,
                                     &w1, &w2, &b1, &b2,
                                     &start_obj, &end_obj, &step_obj,
                                     &algo_name, &output_obj,
                                     &output_start_obj, &output_step_obj,
                                     &start_callback, &end_callback,
                                     &global_error_callback, &local_error_callback,
                                     &return_callback, &release_gil))
        return NULL;

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

    Data *data = _pyobj_to_data(data_obj);
    if (!data) return NULL;
    Data *kernel = _pyobj_to_data(kernel_obj);
    if (!kernel) { Data_free(data); return NULL; }
    int dim = data->dimension;

    int *start = (int*)malloc((size_t)(dim) * sizeof(int));
    if (start_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(start_obj, &tmp, &cnt) < 0) { Data_free(data); Data_free(kernel); return NULL; }
        memcpy(start, tmp, dim * sizeof(int)); free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) start[i] = 0;
    }

    int *end = (int*)malloc((size_t)(dim) * sizeof(int));
    if (end_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(end_obj, &tmp, &cnt) < 0) { free(start); Data_free(data); Data_free(kernel); return NULL; }
        memcpy(end, tmp, dim * sizeof(int)); free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) end[i] = data->shape[i];
    }

    int *step = (int*)malloc((size_t)(dim) * sizeof(int));
    if (step_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(step_obj, &tmp, &cnt) < 0) { free(end); free(start); Data_free(data); Data_free(kernel); return NULL; }
        memcpy(step, tmp, dim * sizeof(int)); free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) step[i] = 1;
    }

    int *output_start = (int*)malloc((size_t)(dim) * sizeof(int));
    if (output_start_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(output_start_obj, &tmp, &cnt) < 0) { free(step); free(end); free(start); Data_free(data); Data_free(kernel); return NULL; }
        memcpy(output_start, tmp, dim * sizeof(int)); free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) output_start[i] = 0;
    }

    int *output_step = (int*)malloc((size_t)(dim) * sizeof(int));
    if (output_step_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(output_step_obj, &tmp, &cnt) < 0) { free(output_start); free(step); free(end); free(start); Data_free(data); Data_free(kernel); return NULL; }
        memcpy(output_step, tmp, dim * sizeof(int)); free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) output_step[i] = 1;
    }

    algo_fn algo = _get_algo(algo_name);
    if (!algo) {
        free(output_step); free(output_start); free(step);
        free(end); free(start); Data_free(data); Data_free(kernel);
        if (global_error_callback && PyCallable_Check(global_error_callback)) {
            PyObject *ns = PyObject_CallObject((PyObject*)&FuncNameSpaceType, NULL);
            if (ns) {
                PyObject *res = PyObject_CallFunctionObjArgs(global_error_callback, ns, NULL);
                Py_XDECREF(res); Py_DECREF(ns);
            }
        }
        return NULL;
    }

    PyObject *name_space = NULL;
    CallbackContext ctx = {0};
    if (start_callback || end_callback || global_error_callback || local_error_callback || return_callback || PyCallable_Check(algo_name)) {
        name_space = PyObject_CallObject((PyObject*)&FuncNameSpaceType, NULL);
        if (name_space) {
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
            PyObject *linear_tuple = PyTuple_New(4);
            PyTuple_SET_ITEM(linear_tuple, 0, PyFloat_FromDouble(w1));
            PyTuple_SET_ITEM(linear_tuple, 1, PyFloat_FromDouble(w2));
            PyTuple_SET_ITEM(linear_tuple, 2, PyFloat_FromDouble(b1));
            PyTuple_SET_ITEM(linear_tuple, 3, PyFloat_FromDouble(b2));
            PyObject_SetAttrString(name_space, "linear", linear_tuple); Py_DECREF(linear_tuple);
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
            for (int i = 0; i < dim; ++i) num[i] = (end[i] - start[i] - kernel->shape[i]) / step[i] + 1;
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
        memcpy(saved_os, output_start, dim * sizeof(int));
        memcpy(saved_ost, output_step, dim * sizeof(int));
        for (int i = 0; i < dim; ++i) {
            output_start[i] = 0;
            output_step[i] = 1;
        }
    }
    if (release_gil && !name_space) {
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
            if (exc) { PyErr_Clear(); PyObject *res = PyObject_CallFunctionObjArgs(global_error_callback, exc, name_space, NULL); Py_XDECREF(res); }
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
        // Check if output is our C Vector type for direct write
        Data *out_data = NULL;
        if (PyObject_IsInstance(output_obj, (PyObject*)&VectorizeType)) {
            Vector *vec = (Vector*)output_obj;
            out_data = vec->data;
        }
        for (int i = 0; i < r_dim; ++i) idx[i] = 0;
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
                    _py_set_item(output_obj, out_idx, r_dim, 0, val);
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
        py_result = _data_to_vector(result, result_type);
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
    free(output_step); free(output_start); free(step);
    free(end); free(start); Data_free(data); Data_free(kernel);

    Py_XDECREF(name_space);
    return py_result;
}

/* Full tensor similarity Python wrapper */
static PyObject* py_cos_full(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *a_obj, *b_obj, *algo_name = NULL;
    int release_gil = 0;
    static char *kwlist[] = {"a", "b", "algorithm", "release_gil", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|Op", kwlist, &a_obj, &b_obj, &algo_name, &release_gil))
        return NULL;

    Data *a = _pyobj_to_data(a_obj);
    if (!a) return NULL;
    Data *b = _pyobj_to_data(b_obj);
    if (!b) { Data_free(a); return NULL; }

    if (a->dimension != b->dimension) { PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same."); Data_free(a); Data_free(b); return NULL; }
    for (int i = 0; i < a->dimension; ++i) {
        if (a->shape[i] != b->shape[i]) { PyErr_SetString(PyExc_ValueError, "the shape of two tensors are not same."); Data_free(a); Data_free(b); return NULL; }
    }

    algo_fn algo = _get_algo(algo_name ? algo_name : PyUnicode_FromString("cos"));
    if (!algo) { Data_free(a); Data_free(b); return NULL; }

    CallbackContext ctx = {0};
    PyObject *ns = NULL;
    if (algo_name && PyCallable_Check(algo_name)) {
        ns = PyObject_CallObject((PyObject*)&FuncNameSpaceType, NULL);
        if (ns) { Py_INCREF(algo_name); PyObject_SetAttrString(ns, "algorithm", algo_name); ctx.name_space = ns; }
    }

    double result = 0.0;
    if (release_gil && !ns) {
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
    PyObject *data_obj, *local_size_obj = NULL, *step_obj = NULL,
             *weight_obj = NULL,
             *output_obj = NULL, *output_start_obj = NULL, *output_step_obj = NULL;
    int release_gil = 0;
    static char *kwlist[] = {"data", "local_size", "step", "weight", "output", "output_start", "output_step", "release_gil", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOOOOOp", kwlist,
                                     &data_obj, &local_size_obj, &step_obj, &weight_obj,
                                     &output_obj, &output_start_obj, &output_step_obj, &release_gil))
        return NULL;

    Data *data = _pyobj_to_data(data_obj);
    if (!data) return NULL;
    int dim = data->dimension;
    Data *output_data = NULL;
    if (output_obj && PyObject_IsInstance(output_obj, (PyObject*)&VectorizeType)) {
        Vector *v = (Vector*)output_obj;
        output_data = v->data;
    }

    int *local_size = NULL; int ls_dim = 0;
    /* Accept a single int as a 1-D window (consistent with pure Python) */
    if (local_size_obj && PyLong_Check(local_size_obj)) {
        local_size = (int*)malloc(sizeof(int));
        if (!local_size) { Data_free(data); return PyErr_NoMemory(); }
        local_size[0] = (int)PyLong_AsLong(local_size_obj);
        ls_dim = 1;
    } else if (local_size_obj) {
        if (_parse_int_seq(local_size_obj, &local_size, &ls_dim) < 0) { Data_free(data); return NULL; }
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
        Data *wdata = _pyobj_to_data(weight_obj);
        if (!wdata) { free(local_size); Data_free(data); return NULL; }
        if (wdata->dimension != dim) {
            Data_free(wdata); free(local_size); Data_free(data);
            PyErr_Format(PyExc_ValueError, "weight dimension %d must match data dimension %d", wdata->dimension, dim);
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
        if (!weights) { Data_free(wdata); free(local_size); Data_free(data); return PyErr_NoMemory(); }
        for (int i = 0; i < wtot; ++i) weights[i] = Data_get_flat(wdata, i);
        Data_free(wdata);
    } else {
        /* One weight per window element (placeholder to keep single code path) */
        int wtot = 1;
        for (int i = 0; i < dim; ++i) wtot *= local_size[i];
        weights = (double*)malloc((size_t)(wtot) * sizeof(double));
        if (!weights) { free(local_size); Data_free(data); return PyErr_NoMemory(); }
        for (int i = 0; i < wtot; ++i) weights[i] = 1.0;
    }

    int *start = (int*)malloc((size_t)(dim) * sizeof(int));
    for (int i = 0; i < dim; ++i) start[i] = 0;
    int *end = (int*)malloc((size_t)(dim) * sizeof(int));
    for (int i = 0; i < dim; ++i) end[i] = data->shape[i];

    int *step = (int*)malloc((size_t)(dim) * sizeof(int));
    if (step_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(step_obj, &tmp, &cnt) < 0) { free(end); free(start); free(local_size); free(weights); Data_free(data); return NULL; }
        memcpy(step, tmp, dim * sizeof(int)); free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) step[i] = 1;
    }

    int *output_start = (int*)malloc((size_t)(dim) * sizeof(int));
    if (output_start_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(output_start_obj, &tmp, &cnt) < 0) { free(step); free(end); free(start); free(local_size); free(weights); Data_free(data); return NULL; }
        memcpy(output_start, tmp, dim * sizeof(int)); free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) output_start[i] = 0;
    }

    int *output_step = (int*)malloc((size_t)(dim) * sizeof(int));
    if (output_step_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(output_step_obj, &tmp, &cnt) < 0) { free(output_start); free(step); free(end); free(start); free(local_size); free(weights); Data_free(data); return NULL; }
        memcpy(output_step, tmp, dim * sizeof(int)); free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) output_step[i] = 1;
    }

    Data *result = NULL;
    // Save user's output_start/output_step and set to 0/1 for core call when output is provided
    int *saved_os = NULL, *saved_ost = NULL;
    if (output_obj && output_obj != Py_None) {
        saved_os = (int*)malloc((size_t)(dim) * sizeof(int));
        saved_ost = (int*)malloc((size_t)(dim) * sizeof(int));
        memcpy(saved_os, output_start, dim * sizeof(int));
        memcpy(saved_ost, output_step, dim * sizeof(int));
        for (int i = 0; i < dim; ++i) {
            output_start[i] = 0;
            output_step[i] = 1;
        }
    }
    if (release_gil && !output_obj) {
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
        // Check if output is our C Vector type for direct write
        Data *out_data = NULL;
        if (PyObject_IsInstance(output_obj, (PyObject*)&VectorizeType)) {
            Vector *vec = (Vector*)output_obj;
            out_data = vec->data;
        }
        for (int i = 0; i < r_dim; ++i) idx[i] = 0;
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
                    _py_set_item(output_obj, out_idx, r_dim, 0, val);
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
        py_result = _data_to_vector(result, result_type);
    }

    // Free all allocated memory
    free(output_step); free(output_start); free(step);
    free(end); free(start); free(local_size); free(weights); Data_free(data);

    return py_result;
}

/* Local variance Python wrapper */
static PyObject* py_local_variance(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *data_obj, *local_size_obj = NULL, *step_obj = NULL,
             *output_obj = NULL, *output_start_obj = NULL, *output_step_obj = NULL;
    int release_gil = 0;
    static char *kwlist[] = {"data", "local_size", "step", "output", "output_start", "output_step", "release_gil", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOOOOp", kwlist,
                                     &data_obj, &local_size_obj, &step_obj,
                                     &output_obj, &output_start_obj, &output_step_obj, &release_gil))
        return NULL;

    Data *data = _pyobj_to_data(data_obj);
    if (!data) return NULL;
    int dim = data->dimension;

    int *local_size = NULL; int ls_dim = 0;
    if (local_size_obj && PyLong_Check(local_size_obj)) {
        local_size = (int*)malloc(sizeof(int));
        if (!local_size) { Data_free(data); return PyErr_NoMemory(); }
        local_size[0] = (int)PyLong_AsLong(local_size_obj);
        ls_dim = 1;
    } else if (local_size_obj) {
        if (_parse_int_seq(local_size_obj, &local_size, &ls_dim) < 0) { Data_free(data); return NULL; }
    } else {
        ls_dim = dim; local_size = (int*)malloc((size_t)(ls_dim) * sizeof(int));
        for (int i = 0; i < ls_dim; ++i) local_size[i] = 1;
    }
    if (ls_dim != dim) {
        free(local_size); Data_free(data);
        PyErr_Format(PyExc_ValueError, "local_size dimension %d does not match data dimension %d", ls_dim, dim);
        return NULL;
    }

    int *start = (int*)malloc((size_t)(dim) * sizeof(int));
    for (int i = 0; i < dim; ++i) start[i] = 0;
    int *end = (int*)malloc((size_t)(dim) * sizeof(int));
    for (int i = 0; i < dim; ++i) end[i] = data->shape[i];

    int *step = (int*)malloc((size_t)(dim) * sizeof(int));
    if (step_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(step_obj, &tmp, &cnt) < 0) { free(end); free(start); free(local_size); Data_free(data); return NULL; }
        memcpy(step, tmp, dim * sizeof(int)); free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) step[i] = 1;
    }

    int *output_start = (int*)malloc((size_t)(dim) * sizeof(int));
    if (output_start_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(output_start_obj, &tmp, &cnt) < 0) { free(step); free(end); free(start); free(local_size); Data_free(data); return NULL; }
        memcpy(output_start, tmp, dim * sizeof(int)); free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) output_start[i] = 0;
    }

    int *output_step = (int*)malloc((size_t)(dim) * sizeof(int));
    if (output_step_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(output_step_obj, &tmp, &cnt) < 0) { free(output_start); free(step); free(end); free(start); free(local_size); Data_free(data); return NULL; }
        memcpy(output_step, tmp, dim * sizeof(int)); free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) output_step[i] = 1;
    }

    Data *result = NULL;
    // Save user's output_start/output_step and set to 0/1 for core call when output is provided
    int *saved_os = NULL, *saved_ost = NULL;
    if (output_obj && output_obj != Py_None) {
        saved_os = (int*)malloc((size_t)(dim) * sizeof(int));
        saved_ost = (int*)malloc((size_t)(dim) * sizeof(int));
        memcpy(saved_os, output_start, dim * sizeof(int));
        memcpy(saved_ost, output_step, dim * sizeof(int));
        for (int i = 0; i < dim; ++i) {
            output_start[i] = 0;
            output_step[i] = 1;
        }
    }
    if (release_gil && !output_obj) {
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
        // Check if output is our C Vector type for direct write
        Data *out_data = NULL;
        if (PyObject_IsInstance(output_obj, (PyObject*)&VectorizeType)) {
            Vector *vec = (Vector*)output_obj;
            out_data = vec->data;
        }
        for (int i = 0; i < r_dim; ++i) idx[i] = 0;
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
                    _py_set_item(output_obj, out_idx, r_dim, 0, val);
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
        py_result = _data_to_vector(result, result_type);
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
    /* Same as your existing implementation */
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
    int release_gil = 0;
    static char *kwlist[] = {
        "window_size", "w1", "w2", "b1", "b2",
        "start", "end", "step", "d", "algorithm",
        "output", "output_start", "output_step",
        "start_callback", "end_callback",
        "global_error_callback", "local_error_callback",
        "return_callback", "release_gil", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OddddOOOOOOOOOOOOOp", kwlist,
                                     &window_size_obj,
                                     &w1, &w2, &b1, &b2,
                                     &start_obj, &end_obj, &step_obj, &d_obj,
                                     &algo_name, &output_obj,
                                     &output_start_obj, &output_step_obj,
                                     &start_callback, &end_callback,
                                     &global_error_callback, &local_error_callback,
                                     &return_callback, &release_gil))
        return NULL;

    /* Create contiguous copy of view data for core computation */
    Data *contig_data = Data_create(vec->dimension, vec->shape);
    if (!contig_data) return NULL;
    int total = 1;
    for (int i = 0; i < vec->dimension; ++i) total *= vec->shape[i];
    int *indices = (int*)malloc((size_t)(total) * sizeof(int));
    if (!indices) { Data_free(contig_data); return PyErr_NoMemory(); }
    /* Iterate all indices with carry method */
    {
        int *idx = (int*)malloc((size_t)(vec->dimension) * sizeof(int));
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

    int *window_size = NULL; int ws_count = 0;
    if (window_size_obj) {
        if (_parse_int_seq(window_size_obj, &window_size, &ws_count) < 0) return NULL;
    } else {
        window_size = (int*)malloc((size_t)(dim) * sizeof(int));
        for (int i = 0; i < dim; ++i) window_size[i] = 1;
        ws_count = dim;
    }

    int *start = (int*)malloc((size_t)(dim) * sizeof(int));
    if (start_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(start_obj, &tmp, &cnt) < 0) { free(window_size); free(start); return NULL; }
        for (int i = 0; i < dim; ++i) start[i] = (i < cnt) ? tmp[i] : 0;
        free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) start[i] = 0;
    }

    int *end = (int*)malloc((size_t)(dim) * sizeof(int));
    if (end_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(end_obj, &tmp, &cnt) < 0) { free(window_size); free(start); free(end); return NULL; }
        for (int i = 0; i < dim; ++i) end[i] = (i < cnt) ? tmp[i] : data->shape[i];
        free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) end[i] = data->shape[i];
    }

    int *step = (int*)malloc((size_t)(dim) * sizeof(int));
    if (step_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(step_obj, &tmp, &cnt) < 0) { free(window_size); free(start); free(end); free(step); return NULL; }
        for (int i = 0; i < dim; ++i) step[i] = (i < cnt) ? tmp[i] : 1;
        free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) step[i] = 1;
    }

    int *d = (int*)malloc((size_t)(dim) * sizeof(int));
    if (d_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(d_obj, &tmp, &cnt) < 0) { free(window_size); free(start); free(end); free(step); free(d); return NULL; }
        for (int i = 0; i < dim; ++i) d[i] = (i < cnt) ? tmp[i] : 0;
        free(tmp);
    } else {
        d[0] = 1;
        for (int i = 1; i < dim; ++i) d[i] = 0;
    }

    algo_fn algo = _get_algo(algo_name);
    if (!algo) {
        free(window_size); free(start); free(end); free(step); free(d);
        return NULL;
    }

    int *output_start = NULL; int *output_step = NULL;
    if (output_start_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(output_start_obj, &tmp, &cnt) < 0) {
            free(window_size); free(start); free(end); free(step); free(d);
            return NULL;
        }
        output_start = (int*)malloc((size_t)(dim) * sizeof(int));
        for (int i = 0; i < dim; ++i) output_start[i] = (i < cnt) ? tmp[i] : 0;
        free(tmp);
    } else {
        output_start = (int*)malloc((size_t)(dim) * sizeof(int));
        for (int i = 0; i < dim; ++i) output_start[i] = 0;
    }
    if (output_step_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(output_step_obj, &tmp, &cnt) < 0) {
            free(window_size); free(start); free(end); free(step); free(d); free(output_start);
            return NULL;
        }
        output_step = (int*)malloc((size_t)(dim) * sizeof(int));
        for (int i = 0; i < dim; ++i) output_step[i] = (i < cnt) ? tmp[i] : 1;
        free(tmp);
    } else {
        output_step = (int*)malloc((size_t)(dim) * sizeof(int));
        for (int i = 0; i < dim; ++i) output_step[i] = 1;
    }

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
            PyObject *linear_tuple = PyTuple_New(4);
            PyTuple_SET_ITEM(linear_tuple, 0, PyFloat_FromDouble(w1));
            PyTuple_SET_ITEM(linear_tuple, 1, PyFloat_FromDouble(w2));
            PyTuple_SET_ITEM(linear_tuple, 2, PyFloat_FromDouble(b1));
            PyTuple_SET_ITEM(linear_tuple, 3, PyFloat_FromDouble(b2));
            PyObject_SetAttrString(name_space, "linear", linear_tuple); Py_DECREF(linear_tuple);
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
            for (int i = 0; i < dim; ++i) num[i] = (end[i] - start[i] - window_size[i] - d[i]) / step[i] + 1;
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
        memcpy(saved_os, output_start, dim * sizeof(int));
        memcpy(saved_ost, output_step, dim * sizeof(int));
        for (int i = 0; i < dim; ++i) {
            output_start[i] = 0;
            output_step[i] = 1;
        }
    }
    if (release_gil && !name_space) {
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
            if (exc) { PyErr_Clear(); PyObject *res = PyObject_CallFunctionObjArgs(global_error_callback, exc, name_space, NULL); Py_XDECREF(res); }
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
        // Check if output is our C Vector type for direct write
        Data *out_data = NULL;
        if (PyObject_IsInstance(output_obj, (PyObject*)&VectorizeType)) {
            Vector *vec = (Vector*)output_obj;
            out_data = vec->data;
        }
        for (int i = 0; i < r_dim; ++i) idx[i] = 0;
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
                    _py_set_item(output_obj, out_idx, r_dim, 0, val);
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
        py_result = _data_to_vector(result, Py_TYPE(self));
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
    /* Same as your existing implementation, omitted for brevity but identical logic */
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
    int release_gil = 0;
    static char *kwlist[] = {
        "kernel", "w1", "w2", "b1", "b2",
        "start", "end", "step", "algorithm",
        "output", "output_start", "output_step",
        "start_callback", "end_callback",
        "global_error_callback", "local_error_callback",
        "return_callback", "release_gil", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ddddOOOOOOOOOOOOp", kwlist,
                                     &kernel_obj,
                                     &w1, &w2, &b1, &b2,
                                     &start_obj, &end_obj, &step_obj,
                                     &algo_name, &output_obj,
                                     &output_start_obj, &output_step_obj,
                                     &start_callback, &end_callback,
                                     &global_error_callback, &local_error_callback,
                                     &return_callback, &release_gil))
        return NULL;

    if (!kernel_obj) { PyErr_SetString(PyExc_ValueError, "kernel must be provided for active mode"); return NULL; }

    /* Create contiguous copy of view data for core computation */
    Data *contig_data = Data_create(vec->dimension, vec->shape);
    if (!contig_data) return NULL;
    int total = 1;
    for (int i = 0; i < vec->dimension; ++i) total *= vec->shape[i];
    int *indices = (int*)malloc((size_t)(total) * sizeof(int));
    if (!indices) { Data_free(contig_data); return PyErr_NoMemory(); }
    /* Iterate all indices with carry method */
    {
        int *idx = (int*)malloc((size_t)(vec->dimension) * sizeof(int));
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
    Data *kernel = _pyobj_to_data(kernel_obj);
    if (!kernel) return NULL;

    int *start = (int*)malloc((size_t)(dim) * sizeof(int));
    if (start_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(start_obj, &tmp, &cnt) < 0) { Data_free(kernel); free(start); return NULL; }
        for (int i = 0; i < dim; ++i) start[i] = (i < cnt) ? tmp[i] : 0;
        free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) start[i] = 0;
    }

    int *end = (int*)malloc((size_t)(dim) * sizeof(int));
    if (end_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(end_obj, &tmp, &cnt) < 0) { Data_free(kernel); free(start); free(end); return NULL; }
        for (int i = 0; i < dim; ++i) end[i] = (i < cnt) ? tmp[i] : data->shape[i];
        free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) end[i] = data->shape[i];
    }

    int *step = (int*)malloc((size_t)(dim) * sizeof(int));
    if (step_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(step_obj, &tmp, &cnt) < 0) { Data_free(kernel); free(start); free(end); free(step); return NULL; }
        for (int i = 0; i < dim; ++i) step[i] = (i < cnt) ? tmp[i] : 1;
        free(tmp);
    } else {
        for (int i = 0; i < dim; ++i) step[i] = 1;
    }

    algo_fn algo = _get_algo(algo_name);
    if (!algo) { Data_free(kernel); free(start); free(end); free(step); return NULL; }

    int *output_start = NULL; int *output_step = NULL;
    if (output_start_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(output_start_obj, &tmp, &cnt) < 0) { Data_free(kernel); free(start); free(end); free(step); return NULL; }
        output_start = (int*)malloc((size_t)(dim) * sizeof(int));
        for (int i = 0; i < dim; ++i) output_start[i] = (i < cnt) ? tmp[i] : 0;
        free(tmp);
    } else {
        output_start = (int*)malloc((size_t)(dim) * sizeof(int));
        for (int i = 0; i < dim; ++i) output_start[i] = 0;
    }
    if (output_step_obj) {
        int cnt; int *tmp;
        if (_parse_int_seq(output_step_obj, &tmp, &cnt) < 0) { Data_free(kernel); free(start); free(end); free(step); free(output_start); return NULL; }
        output_step = (int*)malloc((size_t)(dim) * sizeof(int));
        for (int i = 0; i < dim; ++i) output_step[i] = (i < cnt) ? tmp[i] : 1;
        free(tmp);
    } else {
        output_step = (int*)malloc((size_t)(dim) * sizeof(int));
        for (int i = 0; i < dim; ++i) output_step[i] = 1;
    }

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
            PyObject *linear_tuple = PyTuple_New(4);
            PyTuple_SET_ITEM(linear_tuple, 0, PyFloat_FromDouble(w1));
            PyTuple_SET_ITEM(linear_tuple, 1, PyFloat_FromDouble(w2));
            PyTuple_SET_ITEM(linear_tuple, 2, PyFloat_FromDouble(b1));
            PyTuple_SET_ITEM(linear_tuple, 3, PyFloat_FromDouble(b2));
            PyObject_SetAttrString(name_space, "linear", linear_tuple); Py_DECREF(linear_tuple);
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
            for (int i = 0; i < dim; ++i) num[i] = (end[i] - start[i] - kernel->shape[i]) / step[i] + 1;
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
        memcpy(saved_os, output_start, dim * sizeof(int));
        memcpy(saved_ost, output_step, dim * sizeof(int));
        for (int i = 0; i < dim; ++i) {
            output_start[i] = 0;
            output_step[i] = 1;
        }
    }
    if (release_gil && !name_space) {
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
            if (exc) { PyErr_Clear(); PyObject *res = PyObject_CallFunctionObjArgs(global_error_callback, exc, name_space, NULL); Py_XDECREF(res); }
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
        // Check if output is our C Vector type for direct write
        Data *out_data = NULL;
        if (PyObject_IsInstance(output_obj, (PyObject*)&VectorizeType)) {
            Vector *vec = (Vector*)output_obj;
            out_data = vec->data;
        }
        for (int i = 0; i < r_dim; ++i) idx[i] = 0;
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
                    _py_set_item(output_obj, out_idx, r_dim, 0, val);
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
        py_result = _data_to_vector(result, Py_TYPE(self));
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
        PyObject *v_item, *r_item;
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
    if (PyType_Ready(&VectorChainComputeType) < 0) return NULL;
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
    {"infer_shape", (PyCFunction)py_infer_shape, METH_VARARGS,
        "Infer the shape of multi-dimensional data."},
    {"get_item", py_get_item, METH_VARARGS,
        "Get item from nested list with multi-dimensional index."},
    {"set_item", py_set_item, METH_VARARGS,
        "Set item in nested list with multi-dimensional index."},
    {"_cos", py_cos, METH_VARARGS, "inner cos algorithm."},
    {"_mod", py_mod, METH_VARARGS, "inner mod algorithm."},
    {"_cosmod", py_cosmod, METH_VARARGS, "inner cosmod algorithm."},
    {"no_done", py_no_done, METH_VARARGS, "Placeholder callback that does nothing."},
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
    if (Add_Object(module)) {
        return -1;
    }

    // Add NaN constant
    if (PyModule_AddObject(module, "NaN", PyFloat_FromDouble(NAN)) < 0) {
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
    PyObject *default_algo = PyObject_GetAttrString(module, "_cosmod");
    if (!cos_algo || !mod_algo || !cosmod_algo || !default_algo) {
        Py_XDECREF(cos_algo);
        Py_XDECREF(mod_algo);
        Py_XDECREF(cosmod_algo);
        Py_XDECREF(default_algo);
        Py_DECREF(private_dict);
        return -1;
    }
    PyDict_SetItemString(private_dict, "_cos", cos_algo);
    PyDict_SetItemString(private_dict, "_mod", mod_algo);
    PyDict_SetItemString(private_dict, "_cosmod", cosmod_algo);
    PyDict_SetItemString(private_dict, "_default_algorithm", default_algo);
    Py_DECREF(cos_algo);
    Py_DECREF(mod_algo);
    Py_DECREF(cosmod_algo);
    Py_DECREF(default_algo);
    if (PyModule_AddObject(module, "private_dict", private_dict) < 0) {
        Py_DECREF(private_dict);
        return -1;
    }

    // Register VectorChainCompute native type
    if (PyType_Ready(&VectorChainComputeType) < 0) {
        return -1;
    }

    // Add __all__ (public API list, matches pure Python backend)
    PyObject *__all__ = Py_BuildValue("[sssssssssssssssssssssssssssssssssssssssssss]",
        "NaN", "sqrt",
        "cos_comparison_passive", "cos_comparison_passive_1d", "cos_comparison_passive_2d", "cos_comparison_passive_3d", "cos_comparison_passive_4d",
        "cos_comparison_active", "cos_comparison_active_1d", "cos_comparison_active_2d", "cos_comparison_active_3d", "cos_comparison_active_4d",
        "cos", "cos_1d", "cos_2d", "cos_3d", "cos_4d",
        "mean_local", "mean_local_1d", "mean_local_2d", "mean_local_3d", "mean_local_4d",
        "local_variance", "local_variance_1d", "local_variance_2d", "local_variance_3d", "local_variance_4d",
        "multiple_chain", "add_chain", "no_done", "create_void_list", "load_as_default_data", "infer_shape", "get_item", "set_item", "_cos", "_mod", "_cosmod",
        "vector_chain_compute",
        "vector_map_as_tensor", "func_name_space", "default_contain",
        "private_dict"
    );
    if (!__all__) {
        return -1;
    }
    if (PyModule_AddObject(module, "__all__", __all__) < 0) {
        Py_DECREF(__all__);
        return -1;
    }

    return 0;
}

static PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, module_exec},
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
