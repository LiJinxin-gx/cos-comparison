#ifndef TYPE_H
#define TYPE_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif
#include <Python.h>

#ifndef VECTOR_H
#include "type_vector.h"
#endif
#ifndef FUNC_NAME_SPACE_H
#include "type_func_name_space.h"
#endif
#ifndef DEFAULT_CONTAIN_H
#include "type_default_contain.h"
#endif

typedef struct {
	const char *name;
	PyTypeObject *type;
} submodule;

static submodule submodules[] = {
	{"vector_map_as_tensor", (PyTypeObject*)&VectorizeType},
	{"func_name_space", (PyTypeObject*)&FuncNameSpaceType},
	{"default_contain", (PyTypeObject*)&DefaultContainType},
};

static int Add_Object(PyObject *module) {
	int count = sizeof(submodules) / sizeof(submodule);
	for (int i = 0; i < count; i++) {
		submodule *m = &submodules[i];
		if (PyType_Ready(m->type) < 0) {
			return i + 1;
		}
		PyModule_AddObject(module, m->name, (PyObject*)m->type);
	}
	return 0;
}

#endif

