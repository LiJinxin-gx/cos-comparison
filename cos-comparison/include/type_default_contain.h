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
	PyObject *default_value = Py_None;
	static char *kwlist[] = {"default_value", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &default_value))
		return -1;
	self->default_dict = PyDict_New();
	if (!self->default_dict) return -1;
	self->default_value = default_value;
	Py_INCREF(default_value);
	return 0;
}

static int DefaultContain_contains(DefaultContain *self, PyObject *item) {
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

static int DefaultContain_setitem(DefaultContain *self, PyObject *item, PyObject *value) {
	if (value == NULL) {
		return PyDict_DelItem(self->default_dict, item);
	}
	return PyDict_SetItem(self->default_dict, item, value);
}

static PySequenceMethods DefaultContain_as_sequence = {
	0, 0, 0, 0,
	(objobjproc)DefaultContain_contains,
};

static PyMappingMethods DefaultContain_as_mapping = {
	0,
	(binaryfunc)DefaultContain_getitem,
	(objobjargproc)DefaultContain_setitem,
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

