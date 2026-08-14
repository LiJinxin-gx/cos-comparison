#ifndef DEFAULT_CONTAIN_H
#define DEFAULT_CONTAIN_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif
#include <Python.h>

typedef struct {
	PyObject_HEAD
	PyObject *default_dict;
	PyObject *default_value;
} DefaultContain;

static void DefaultContain_dealloc(DefaultContain *self) {
	Py_XDECREF(self->default_dict);
	Py_XDECREF(self->default_value);
	Py_TYPE(self)->tp_free((PyObject*)self);
}

static int DefaultContain_init(DefaultContain *self, PyObject *args, PyObject *kwargs) {
	PyObject *default_value = NULL;
	PyObject *default_dict = Py_None;
	static char *kwlist[] = {"default", "default_dict", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &default_value, &default_dict))
		return -1;
	if (default_dict != Py_None && PyDict_Check(default_dict)) {
		self->default_dict = PyDict_Copy(default_dict);
		if (!self->default_dict) return -1;
	} else {
		self->default_dict = PyDict_New();
		if (!self->default_dict) return -1;
	}
	self->default_value = default_value;
	Py_INCREF(default_value);
	return 0;
}

static int DefaultContain_contains(DefaultContain *self, PyObject *item) {
	return 1;
}

static Py_ssize_t DefaultContain_len(DefaultContain *self) {
	return 1;
}

static PyObject* DefaultContain_getitem(DefaultContain *self, PyObject *item) {
	PyObject *value = PyDict_GetItem(self->default_dict, item);
	if (value) {
		Py_INCREF(value);
		return value;
	}
	Py_INCREF(self->default_value);
	return self->default_value;
}

/* Designated initializers (C99 6.7.8): sq_contains must land in its own
   slot - a positional initializer would put it into sq_slice. */
static PySequenceMethods DefaultContain_as_sequence = {
	.sq_length   = (lenfunc)DefaultContain_len,
	.sq_contains = (objobjproc)DefaultContain_contains,
};

static PyMappingMethods DefaultContain_as_mapping = {
	.mp_subscript = (binaryfunc)DefaultContain_getitem,
};

static PyObject* DefaultContain_repr(DefaultContain *self) {
	return PyUnicode_FromFormat("<default_contain: default=%R>", self->default_value);
}

static PyTypeObject DefaultContainType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name      = "cos_comparison_pydll.default_contain",
	.tp_basicsize = sizeof(DefaultContain),
	.tp_itemsize  = 0,
	.tp_dealloc   = (destructor)DefaultContain_dealloc,
	.tp_repr      = (reprfunc)DefaultContain_repr,
	.tp_as_sequence = &DefaultContain_as_sequence,
	.tp_as_mapping  = &DefaultContain_as_mapping,
	.tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
	.tp_doc       = "Container that returns default value for unset keys.",
	.tp_init      = (initproc)DefaultContain_init,
	.tp_new       = PyType_GenericNew,
};

#endif

