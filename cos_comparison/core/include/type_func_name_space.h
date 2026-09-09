#ifndef FUNC_NAME_SPACE_H
#define FUNC_NAME_SPACE_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif
#include <Python.h>

typedef struct {
	PyObject_HEAD
	PyObject *dict;
} FuncNameSpace;

static void FuncNameSpace_dealloc(FuncNameSpace *self) {
	Py_XDECREF(self->dict);
	Py_TYPE(self)->tp_free((PyObject*)self);
}

static int FuncNameSpace_init(FuncNameSpace *self, PyObject *args, PyObject *kwargs) {
    (void)args;
	self->dict = PyDict_New();
	if (!self->dict) return -1;
	if (kwargs) {
		PyObject *key, *value;
		Py_ssize_t pos = 0;
		while (PyDict_Next(kwargs, &pos, &key, &value)) {
			if (PyDict_SetItem(self->dict, key, value) < 0) return -1;
		}
	}
	return 0;
}

static PyObject* FuncNameSpace_getattro(FuncNameSpace *self, PyObject *name) {
	PyObject *value = PyDict_GetItem(self->dict, name);
	if (value) {
		Py_INCREF(value);
		return value;
	}
	return PyObject_GenericGetAttr((PyObject*)self, name);
}

static int FuncNameSpace_setattro(FuncNameSpace *self, PyObject *name, PyObject *value) {
	if (value == NULL) {
		return PyDict_DelItem(self->dict, name);
	}
	return PyDict_SetItem(self->dict, name, value);
}

static PyObject* FuncNameSpace_repr(FuncNameSpace *self) {
	return PyUnicode_FromFormat("<func_name_space: %R>", self->dict);
}

/* Mapping protocol (**s unpacking, dict(ns), len, item access) - native
 * dict operations only. */
static Py_ssize_t FuncNameSpace_length(FuncNameSpace *self) {
	return PyDict_Size(self->dict);
}

static PyObject* FuncNameSpace_subscript(FuncNameSpace *self,
                                         PyObject *key) {
	PyObject *value = PyDict_GetItemWithError(self->dict, key);
	if (value) {
		Py_INCREF(value);
		return value;
	}
	if (PyErr_Occurred()) {
		return NULL;
	}
	PyErr_SetObject(PyExc_KeyError, key);
	return NULL;
}

static PyObject* FuncNameSpace_keys(FuncNameSpace *self,
                                    PyObject *noargs) {
	(void)noargs;
	return PyDict_Keys(self->dict);
}

static PyMappingMethods FuncNameSpace_mapping = {
	(lenfunc)FuncNameSpace_length,        /* mp_length */
	(binaryfunc)FuncNameSpace_subscript,  /* mp_subscript */
	0,                                    /* mp_ass_subscript */
};

static PyMethodDef FuncNameSpace_methods[] = {
	{"keys", (PyCFunction)FuncNameSpace_keys, METH_NOARGS,
	 "keys(): the namespace keys as a dict_keys view."},
	{NULL, NULL, 0, NULL},
};

static PyTypeObject FuncNameSpaceType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name      = "cos_comparison_pydll.func_name_space",
	.tp_basicsize = sizeof(FuncNameSpace),
	.tp_itemsize  = 0,
	.tp_dealloc   = (destructor)FuncNameSpace_dealloc,
	.tp_repr      = (reprfunc)FuncNameSpace_repr,
	.tp_getattro  = (getattrofunc)FuncNameSpace_getattro,
	.tp_setattro  = (setattrofunc)FuncNameSpace_setattro,
	.tp_as_mapping = &FuncNameSpace_mapping,
	.tp_methods   = FuncNameSpace_methods,
	.tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
	.tp_doc       = "Dynamic namespace for callback parameters.",
	.tp_init      = (initproc)FuncNameSpace_init,
	.tp_new       = PyType_GenericNew,
};

#endif

