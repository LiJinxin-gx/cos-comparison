#ifndef CORE_H
#define CORE_H

#include "type_data.h"
#include <Python.h>

/* Callback context for error handling and name space passing */
typedef struct {
    PyObject *local_error_callback;
    PyObject *name_space;
} CallbackContext;

static inline double _cos_(double a, double b, double ab, CallbackContext *ctx) {
    double c = a * b;
    if (c) return ab / sqrt(c);
    return (a == b) ? 1.0 : 0.0;
}

static inline double _mod_(double a, double b, double ab, CallbackContext *ctx) {
    double c = a * b;
    if (c) return 2 * sqrt(c) / (a + b);
    return (a == b) ? 1.0 : 0.0;
}

static inline double _cosmod_(double a, double b, double ab, CallbackContext *ctx) {
    double c = a * b;
    if (c) return 2 * ab / (a + b);
    return (a == b) ? 1.0 : 0.0;
}

typedef double (*algo_fn)(double, double, double, CallbackContext*);

static inline Data* _compute_output_shape(int dim, const int num[],
                                          const int output_start[], const int output_step[]) {
    int *out_shape = (int*)malloc(dim * sizeof(int));
    for (int i = 0; i < dim; ++i)
        out_shape[i] = (num[i] - 1) * output_step[i] + 1;
    Data *out = Data_create(dim, out_shape);
    free(out_shape);
    return out;
}

/* Helper: get item from arbitrary Python object, supporting __get_item__ overload */
static inline double _py_get_item(PyObject *obj, const int idx[], int dim) {
    PyObject *get_item_method = PyObject_GetAttrString(obj, "__get_item__");
    if (get_item_method != NULL) {
        PyObject *args = PyTuple_New(dim);
        if (!args) {
            Py_DECREF(get_item_method);
            return NAN;
        }
        for (int i = 0; i < dim; ++i) {
            PyObject *idx_obj = PyLong_FromLong(idx[i]);
            if (!idx_obj) {
                /* Cleanup already inserted items */
                for (int j = 0; j < i; ++j) {
                    Py_DECREF(PyTuple_GET_ITEM(args, j));
                }
                Py_DECREF(args);
                Py_DECREF(get_item_method);
                return NAN;
            }
            PyTuple_SET_ITEM(args, i, idx_obj);
        }
        PyObject *res = PyObject_CallObject(get_item_method, args);
        Py_DECREF(args);
        Py_DECREF(get_item_method);
        if (res == NULL) return NAN;
        double val = PyFloat_AsDouble(res);
        Py_DECREF(res);
        return val;
    }
    PyErr_Clear();
    PyObject *current = obj;
    for (int i = 0; i < dim; ++i) {
        PyObject *next = PySequence_GetItem(current, idx[i]);
        if (i > 0) Py_DECREF(current);
        if (next == NULL) {
            Py_XDECREF(current);
            return NAN;
        }
        current = next;
    }
    double val = PyFloat_AsDouble(current);
    Py_DECREF(current);
    return val;
}

/* Helper: write value to arbitrary nested Python object */
static inline void _py_set_item(PyObject *obj, const int idx[], int dim, int depth, double value) {
    if (depth == dim - 1) {
        PyObject *set_item_method = PyObject_GetAttrString(obj, "__set_item__");
        if (set_item_method != NULL) {
            PyObject *args = PyTuple_New(2);
            PyTuple_SET_ITEM(args, 0, PyLong_FromLong(idx[depth]));
            PyTuple_SET_ITEM(args, 1, PyFloat_FromDouble(value));
            PyObject_CallObject(set_item_method, args);
            Py_DECREF(args);
            Py_DECREF(set_item_method);
            return;
        }
        PyErr_Clear();
        PyObject *val = PyFloat_FromDouble(value);
        PyObject_SetItem(obj, PyLong_FromLong(idx[depth]), val);
        Py_DECREF(val);
        return;
    }
    PyObject *next;
    PyObject *get_item_method = PyObject_GetAttrString(obj, "__get_item__");
    if (get_item_method != NULL) {
        PyObject *args = PyTuple_New(1);
        PyTuple_SET_ITEM(args, 0, PyLong_FromLong(idx[depth]));
        next = PyObject_CallObject(get_item_method, args);
        Py_DECREF(args);
        Py_DECREF(get_item_method);
    } else {
        PyErr_Clear();
        next = PySequence_GetItem(obj, idx[depth]);
    }
    if (next == NULL) return;
    _py_set_item(next, idx, dim, depth + 1, value);
    Py_DECREF(next);
}

/* Core algorithms – same as ctypes version but with CallbackContext and output_obj */
Data* cos_comparison_passive(const Data *data,
                             const int window_size[],
                             double w1, double w2, double b1, double b2,
                             const int start[], const int end[],
                             const int step[], const int d[],
                             algo_fn algorithm,
                             CallbackContext *ctx,
                             const int output_start[], const int output_step[],
                             PyObject *output_obj, Data *output);

Data* cos_comparison_active(const Data *data, const Data *kernel,
                            double w1, double w2, double b1, double b2,
                            const int start[], const int end[],
                            const int step[],
                            algo_fn algorithm,
                            CallbackContext *ctx,
                            const int output_start[], const int output_step[],
                            PyObject *output_obj, Data *output);

double cos_full(const Data *a, const Data *b, algo_fn algorithm, CallbackContext *ctx);

Data* cos_local_mean(const Data *data,
                     const int window_size[],
                     const int start[], const int end[],
                     const int step[],
                     const int output_start[], const int output_step[],
                     PyObject *output_obj, Data *output);

Data* cos_local_variance(const Data *data,
                         const int window_size[],
                         const int start[], const int end[],
                         const int step[],
                         const int output_start[], const int output_step[],
                         PyObject *output_obj, Data *output);

#endif