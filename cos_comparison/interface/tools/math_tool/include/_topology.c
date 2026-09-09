/* _topology.c - C99 optimized topology algorithms and graph types.
 *
 * Backend for cos_comparison.interface.tools.math_tool._topology.
 * Behaviour is identical to the pure-Python topology module: the two
 * graph types (Graph / DirectedGraph) and the pure functions (BFS
 * shortest path, Euler characteristic by cell) are implemented natively.
 * Type conversion is the C extension's responsibility; the Python module
 * only performs the priority import.
 *
 * C99 features used deliberately: declaration-in-for, long long element
 * arithmetic with overflow fallback to Python big ints, index-headed
 * PyList queues (no O(n) pop(0)), iterative (recursion-free) Tarjan /
 * Kahn / BFS.
 */
#define PY_SSIZE_T_CLEAN
#ifdef _MSC_VER
/* CPython extension ABI patterns under /Wall (see cos_comparison_pydll.c):
   C4191 method-table casts, C4232 PyType_GenericNew address,
   C4820 struct padding; C5045/C4711/C4710 are /Wall performance hints. */
#pragma warning(push)
#pragma warning(disable: 4191 4232 4820 5045 4711 4710)
#endif
#include <Python.h>
#include <limits.h>
#include <stddef.h>

/* ==================================================================
 * Internal helpers
 * ================================================================== */

static int lock_graph(PyObject *lock) {
    /* with lock: acquire via __enter__ (None / no-op lock: skip). */
    if (lock == NULL || lock == Py_None) return 0;
    PyObject *r = PyObject_CallMethod(lock, "__enter__", NULL);
    if (!r) return -1;
    Py_DECREF(r);
    return 0;
}

static void unlock_graph(PyObject *lock) {
    if (lock == NULL || lock == Py_None) return;
    PyObject *r = PyObject_CallMethod(lock, "__exit__", "OOO",
                                      Py_None, Py_None, Py_None);
    Py_XDECREF(r);
}

/* ==================================================================
 * Graph type (undirected multigraph, union-find DSU)
 * ================================================================== */

typedef struct {
    PyObject_HEAD
    PyObject *vertices;     /* PySet */
    PyObject *parent;       /* PyDict: vertex -> vertex */
    PyObject *rank;         /* PyDict: vertex -> int */
    Py_ssize_t edges;
    Py_ssize_t components;
    PyObject *lock;         /* lock object or Py_None */
} GraphObject;

static PyTypeObject GraphType;

static int graph_add_vertex(GraphObject *self, PyObject *v) {
    int contains = PySet_Contains(self->vertices, v);
    if (contains < 0) return -1;
    if (contains) return 0;
    if (PySet_Add(self->vertices, v) < 0) return -1;
    Py_INCREF(v);
    if (PyDict_SetItem(self->parent, v, v) < 0 ||
        PyDict_SetItem(self->rank, v, PyLong_FromLong(0)) < 0) {
        Py_DECREF(v);
        return -1;
    }
    Py_DECREF(v);
    self->components++;
    return 0;
}

static PyObject *graph_find(GraphObject *self, PyObject *x) {
    /* two-pass path compression (iterative) */
    PyObject *root = x;
    PyObject *p;
    while ((p = PyDict_GetItem(self->parent, root)) != NULL &&
           PyObject_RichCompareBool(root, p, Py_NE) == 1) {
        root = p;
    }
    PyObject *cur = x;
    while (PyObject_RichCompareBool(cur, root, Py_NE) == 1) {
        PyObject *parent = PyDict_GetItem(self->parent, cur);
        if (PyDict_SetItem(self->parent, cur, root) < 0) return NULL;
        cur = parent;
    }
    Py_INCREF(root);
    return root;
}

static int graph_union(GraphObject *self, PyObject *a, PyObject *b) {
    PyObject *ra = graph_find(self, a);
    if (!ra) return -1;
    PyObject *rb = graph_find(self, b);
    if (!rb) { Py_DECREF(ra); return -1; }
    int same = PyObject_RichCompareBool(ra, rb, Py_EQ);
    if (same < 0) { Py_DECREF(ra); Py_DECREF(rb); return -1; }
    if (same) { Py_DECREF(ra); Py_DECREF(rb); return 0; }
    PyObject *rra = PyDict_GetItem(self->rank, ra);
    PyObject *rrb = PyDict_GetItem(self->rank, rb);
    long ra_rank = rra ? PyLong_AsLong(rra) : 0;
    long rb_rank = rrb ? PyLong_AsLong(rrb) : 0;
    if (PyErr_Occurred()) { Py_DECREF(ra); Py_DECREF(rb); return -1; }
    PyObject *new_root = ra, *old_root = rb;
    if (ra_rank < rb_rank) {
        new_root = rb; old_root = ra;
    }
    if (PyDict_SetItem(self->parent, old_root, new_root) < 0) {
        Py_DECREF(ra); Py_DECREF(rb); return -1;
    }
    if (ra_rank == rb_rank) {
        long nr = PyLong_AsLong(PyDict_GetItem(self->rank, new_root));
        if (PyErr_Occurred()) { Py_DECREF(ra); Py_DECREF(rb); return -1; }
        if (PyDict_SetItem(self->rank, new_root,
                           PyLong_FromLong(nr + 1)) < 0) {
            Py_DECREF(ra); Py_DECREF(rb); return -1;
        }
    }
    Py_DECREF(ra); Py_DECREF(rb);
    self->components--;
    return 0;
}

static PyObject *graph_new_graph(PyTypeObject *type, PyObject *args,
                                  PyObject *kwargs) {
    PyObject *lock = Py_None;
    static char *kwlist[] = {"lock", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &lock))
        return NULL;
    GraphObject *self = (GraphObject*)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->vertices = PySet_New(NULL);
    self->parent = PyDict_New();
    self->rank = PyDict_New();
    Py_INCREF(lock);
    self->lock = lock;
    self->edges = 0;
    self->components = 0;
    if (!self->vertices || !self->parent || !self->rank) {
        Py_DECREF(self);
        return NULL;
    }
    return (PyObject*)self;
}

static void graph_dealloc(GraphObject *self) {
    Py_XDECREF(self->vertices);
    Py_XDECREF(self->parent);
    Py_XDECREF(self->rank);
    Py_XDECREF(self->lock);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *graph_add_edge(GraphObject *self, PyObject *args) {
    PyObject *u, *v;
    if (!PyArg_ParseTuple(args, "OO", &u, &v)) return NULL;
    if (lock_graph(self->lock) < 0) return NULL;
    if (graph_add_vertex(self, u) < 0 ||
        graph_add_vertex(self, v) < 0 ||
        graph_union(self, u, v) < 0) {
        unlock_graph(self->lock);
        return NULL;
    }
    self->edges++;
    unlock_graph(self->lock);
    Py_RETURN_NONE;
}

#define GRAPH_COUNT_METHOD(name, field)                                      \
static PyObject *graph_##name(GraphObject *self, PyObject *Py_UNUSED(ign)) { \
    if (lock_graph(self->lock) < 0) return NULL;                             \
    PyObject *r = PyLong_FromSsize_t(field);                                  \
    unlock_graph(self->lock);                                                \
    return r;                                                                 \
}

GRAPH_COUNT_METHOD(vertices_count, PySet_GET_SIZE(self->vertices))
GRAPH_COUNT_METHOD(edges_count, self->edges)
GRAPH_COUNT_METHOD(components_count, self->components)

static PyObject *graph_euler(GraphObject *self, PyObject *Py_UNUSED(ign)) {
    if (lock_graph(self->lock) < 0) return NULL;
    Py_ssize_t v = PySet_GET_SIZE(self->vertices);
    PyObject *r = PyLong_FromSsize_t(v - self->edges + self->components);
    unlock_graph(self->lock);
    return r;
}

static PyObject *graph_cycle_rank(GraphObject *self, PyObject *Py_UNUSED(ign)) {
    if (lock_graph(self->lock) < 0) return NULL;
    Py_ssize_t v = PySet_GET_SIZE(self->vertices);
    PyObject *r = PyLong_FromSsize_t(self->edges - v + self->components);
    unlock_graph(self->lock);
    return r;
}

static PyObject *graph_is_connected(GraphObject *self, PyObject *Py_UNUSED(ign)) {
    if (lock_graph(self->lock) < 0) return NULL;
    PyObject *r = PyBool_FromLong(self->components <= 1);
    unlock_graph(self->lock);
    return r;
}

static PyObject *graph_repr(GraphObject *self) {
    if (lock_graph(self->lock) < 0) return NULL;
    Py_ssize_t v = PySet_GET_SIZE(self->vertices);
    PyObject *r = PyUnicode_FromFormat(
        "<Graph: V=%zd, E=%zd, C=%zd, \xCF\x87=%zd, r=%zd>",
        v, self->edges, self->components,
        v - self->edges + self->components,
        self->edges - v + self->components);
    unlock_graph(self->lock);
    return r;
}

static PyMethodDef graph_methods[] = {
    {"add_edge", (PyCFunction)graph_add_edge, METH_VARARGS,
     "Add an undirected edge between vertices u and v (parallel allowed)."},
    {"vertices_count", (PyCFunction)graph_vertices_count, METH_NOARGS,
     "Return the number of vertices."},
    {"edges_count", (PyCFunction)graph_edges_count, METH_NOARGS,
     "Return the number of edges."},
    {"components_count", (PyCFunction)graph_components_count, METH_NOARGS,
     "Return the number of connected components."},
    {"euler_characteristic", (PyCFunction)graph_euler, METH_NOARGS,
     "Return the Euler characteristic V - E + C."},
    {"cycle_rank", (PyCFunction)graph_cycle_rank, METH_NOARGS,
     "Return the cycle rank E - V + C."},
    {"is_connected", (PyCFunction)graph_is_connected, METH_NOARGS,
     "Return True if the graph is connected."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject GraphType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_topology.Graph",
    .tp_basicsize = sizeof(GraphObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = graph_new_graph,
    .tp_dealloc = (destructor)graph_dealloc,
    .tp_repr = (reprfunc)graph_repr,
    .tp_methods = graph_methods,
    .tp_doc = "Undirected multigraph with union-find components."
};

/* ==================================================================
 * DirectedGraph type (adjacency dict, iterative Tarjan / Kahn / BFS)
 * ================================================================== */

typedef struct {
    PyObject_HEAD
    PyObject *vertices;     /* PySet */
    PyObject *adj;          /* PyDict: v -> {w: count} */
    PyObject *in_deg;       /* PyDict: v -> count */
    PyObject *out_deg;      /* PyDict: v -> count */
    PyObject *parent;       /* PyDict (weak DSU) */
    PyObject *rank;         /* PyDict */
    Py_ssize_t components;
    Py_ssize_t edges;
    PyObject *lock;
} DirectedGraphObject;

static PyTypeObject DirectedGraphType;

static int dg_add_vertex(DirectedGraphObject *self, PyObject *v) {
    int contains = PySet_Contains(self->vertices, v);
    if (contains < 0) return -1;
    if (contains) return 0;
    if (PySet_Add(self->vertices, v) < 0) return -1;
    Py_INCREF(v);
    if (PyDict_SetItem(self->parent, v, v) < 0 ||
        PyDict_SetItem(self->rank, v, PyLong_FromLong(0)) < 0) {
        Py_DECREF(v);
        return -1;
    }
    Py_DECREF(v);
    self->components++;
    return 0;
}

static PyObject *dg_find(DirectedGraphObject *self, PyObject *x) {
    PyObject *root = x;
    PyObject *p;
    while ((p = PyDict_GetItem(self->parent, root)) != NULL &&
           PyObject_RichCompareBool(root, p, Py_NE) == 1) {
        root = p;
    }
    PyObject *cur = x;
    while (PyObject_RichCompareBool(cur, root, Py_NE) == 1) {
        PyObject *parent = PyDict_GetItem(self->parent, cur);
        if (PyDict_SetItem(self->parent, cur, root) < 0) return NULL;
        cur = parent;
    }
    Py_INCREF(root);
    return root;
}

static int dg_union(DirectedGraphObject *self, PyObject *a, PyObject *b) {
    PyObject *ra = dg_find(self, a);
    if (!ra) return -1;
    PyObject *rb = dg_find(self, b);
    if (!rb) { Py_DECREF(ra); return -1; }
    int same = PyObject_RichCompareBool(ra, rb, Py_EQ);
    if (same < 0) { Py_DECREF(ra); Py_DECREF(rb); return -1; }
    if (same) { Py_DECREF(ra); Py_DECREF(rb); return 0; }
    long ra_rank = PyLong_AsLong(PyDict_GetItem(self->rank, ra));
    long rb_rank = PyLong_AsLong(PyDict_GetItem(self->rank, rb));
    if (PyErr_Occurred()) { Py_DECREF(ra); Py_DECREF(rb); return -1; }
    PyObject *new_root = ra, *old_root = rb;
    if (ra_rank < rb_rank) { new_root = rb; old_root = ra; }
    if (PyDict_SetItem(self->parent, old_root, new_root) < 0) {
        Py_DECREF(ra); Py_DECREF(rb); return -1;
    }
    if (ra_rank == rb_rank) {
        long nr = PyLong_AsLong(PyDict_GetItem(self->rank, new_root));
        if (PyErr_Occurred()) { Py_DECREF(ra); Py_DECREF(rb); return -1; }
        if (PyDict_SetItem(self->rank, new_root,
                           PyLong_FromLong(nr + 1)) < 0) {
            Py_DECREF(ra); Py_DECREF(rb); return -1;
        }
    }
    Py_DECREF(ra); Py_DECREF(rb);
    self->components--;
    return 0;
}

static int dg_bump(DirectedGraphObject *self, PyObject *key,
                    PyObject *table, int delta) { (void)self;
    PyObject *cur = PyDict_GetItem(table, key);
    long n = cur ? PyLong_AsLong(cur) : 0;
    if (PyErr_Occurred()) return -1;
    return PyDict_SetItem(table, key, PyLong_FromLong(n + delta));
}

static PyObject *dg_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    PyObject *lock = Py_None;
    static char *kwlist[] = {"lock", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &lock))
        return NULL;
    DirectedGraphObject *self =
        (DirectedGraphObject*)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->vertices = PySet_New(NULL);
    self->adj = PyDict_New();
    self->in_deg = PyDict_New();
    self->out_deg = PyDict_New();
    self->parent = PyDict_New();
    self->rank = PyDict_New();
    Py_INCREF(lock);
    self->lock = lock;
    self->components = 0;
    self->edges = 0;
    if (!self->vertices || !self->adj || !self->in_deg || !self->out_deg ||
        !self->parent || !self->rank) {
        Py_DECREF(self);
        return NULL;
    }
    return (PyObject*)self;
}

static void dg_dealloc(DirectedGraphObject *self) {
    Py_XDECREF(self->vertices);
    Py_XDECREF(self->adj);
    Py_XDECREF(self->in_deg);
    Py_XDECREF(self->out_deg);
    Py_XDECREF(self->parent);
    Py_XDECREF(self->rank);
    Py_XDECREF(self->lock);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *dg_add_edge(DirectedGraphObject *self, PyObject *args) {
    PyObject *u, *v;
    if (!PyArg_ParseTuple(args, "OO", &u, &v)) return NULL;
    if (lock_graph(self->lock) < 0) return NULL;
    if (dg_add_vertex(self, u) < 0 || dg_add_vertex(self, v) < 0) {
        unlock_graph(self->lock);
        return NULL;
    }
    self->edges++;
    if (dg_bump(self, u, self->out_deg, 1) < 0 ||
        dg_bump(self, v, self->in_deg, 1) < 0) {
        unlock_graph(self->lock);
        return NULL;
    }
    PyObject *targets = PyDict_GetItem(self->adj, u);
    if (targets == NULL) {
        targets = PyDict_New();
        if (!targets || PyDict_SetItem(self->adj, u, targets) < 0) {
            Py_XDECREF(targets);
            unlock_graph(self->lock);
            return NULL;
        }
    }
    PyObject *cur = PyDict_GetItem(targets, v);
    long n = cur ? PyLong_AsLong(cur) : 0;
    if (PyErr_Occurred() ||
        PyDict_SetItem(targets, v, PyLong_FromLong(n + 1)) < 0 ||
        dg_union(self, u, v) < 0) {
        unlock_graph(self->lock);
        return NULL;
    }
    unlock_graph(self->lock);
    Py_RETURN_NONE;
}

static PyObject *dg_vertices_count(DirectedGraphObject *self,
                                    PyObject *Py_UNUSED(ign)) {
    return PyLong_FromSsize_t(PySet_GET_SIZE(self->vertices));
}

static PyObject *dg_edges_count(DirectedGraphObject *self,
                                 PyObject *Py_UNUSED(ign)) {
    return PyLong_FromSsize_t(self->edges);
}

static PyObject *dg_degree(DirectedGraphObject *self, PyObject *args,
                            int in_out) {
    PyObject *v;
    if (!PyArg_ParseTuple(args, "O", &v)) return NULL;
    if (lock_graph(self->lock) < 0) return NULL;
    PyObject *table = in_out ? self->in_deg : self->out_deg;
    PyObject *cur = PyDict_GetItem(table, v);
    PyObject *r = cur ? PyLong_FromLong(PyLong_AsLong(cur))
                      : PyLong_FromLong(0);
    if (cur && PyErr_Occurred()) r = NULL;
    unlock_graph(self->lock);
    return r;
}

static PyObject *dg_in_degree(DirectedGraphObject *self, PyObject *args) {
    return dg_degree(self, args, 1);
}

static PyObject *dg_out_degree(DirectedGraphObject *self, PyObject *args) {
    return dg_degree(self, args, 0);
}

static PyObject *dg_neighbors(DirectedGraphObject *self, PyObject *args) {
    PyObject *v;
    if (!PyArg_ParseTuple(args, "O", &v)) return NULL;
    if (lock_graph(self->lock) < 0) return NULL;
    int known = PySet_Contains(self->vertices, v);
    if (known < 0) { unlock_graph(self->lock); return NULL; }
    if (!known) {
        unlock_graph(self->lock);
        PyErr_Format(PyExc_KeyError, "Unknown vertex: %R", v);
        return NULL;
    }
    PyObject *targets = PyDict_GetItem(self->adj, v);
    PyObject *keys = targets ? PyDict_Keys(targets) : PyList_New(0);
    PyObject *r = PySequence_Tuple(keys);
    Py_DECREF(keys);
    unlock_graph(self->lock);
    return r;
}

static PyObject *dg_weak_count(DirectedGraphObject *self,
                                PyObject *Py_UNUSED(ign)) {
    return PyLong_FromSsize_t(self->components);
}

static PyObject *dg_is_weakly_connected(DirectedGraphObject *self,
                                         PyObject *Py_UNUSED(ign)) {
    return PyBool_FromLong(self->components <= 1);
}

/* ---- iterative Tarjan (matches the Python frames-based version) ---- */
static PyObject *dg_strong_components(DirectedGraphObject *self,
                                       PyObject *Py_UNUSED(ign)) {
    if (lock_graph(self->lock) < 0) return NULL;
    PyObject *index = PyDict_New();
    PyObject *lowlink = PyDict_New();
    PyObject *on_stack = PySet_New(NULL);
    PyObject *stack = PyList_New(0);
    PyObject *frames = PyList_New(0);
    PyObject *components = PyList_New(0);
    if (!index || !lowlink || !on_stack || !stack || !frames ||
        !components) {
        Py_XDECREF(index); Py_XDECREF(lowlink); Py_XDECREF(on_stack);
        Py_XDECREF(stack); Py_XDECREF(frames); Py_XDECREF(components);
        unlock_graph(self->lock);
        return PyErr_NoMemory();
    }
    long counter = 0;

    PyObject *viter = PyObject_GetIter(self->vertices);
    if (!viter) {
        Py_DECREF(index); Py_DECREF(lowlink); Py_DECREF(on_stack);
        Py_DECREF(stack); Py_DECREF(frames); Py_DECREF(components);
        unlock_graph(self->lock);
        return NULL;
    }
    PyObject *root;
    int ok = 1;
    while ((root = PyIter_Next(viter))) {
        int has = PyDict_Contains(index, root);
        if (has < 0) { Py_DECREF(root); ok = 0; break; }
        if (has) { Py_DECREF(root); continue; }
        if (PyDict_SetItem(index, root, PyLong_FromLong(counter)) < 0 ||
            PyDict_SetItem(lowlink, root, PyLong_FromLong(counter)) < 0) {
            Py_DECREF(root); ok = 0; break;
        }
        counter++;
        if (PyList_Append(stack, root) < 0 ||
            PySet_Add(on_stack, root) < 0) {
            Py_DECREF(root); ok = 0; break;
        }
        PyObject *targets = PyDict_GetItem(self->adj, root);
        PyObject *keys = targets ? PyDict_Keys(targets) : PyList_New(0);
        PyObject *it = PyObject_GetIter(keys);
        Py_DECREF(keys);
        PyObject *frame = PyTuple_Pack(2, root, it);
        Py_DECREF(it);
        Py_DECREF(root);
        if (!frame || PyList_Append(frames, frame) < 0) {
            Py_XDECREF(frame); ok = 0; break;
        }
        Py_DECREF(frame);

        while (PyList_GET_SIZE(frames) > 0) {
            PyObject *top = PyList_GET_ITEM(frames, PyList_GET_SIZE(frames) - 1);
            PyObject *v = PyTuple_GET_ITEM(top, 0);
            PyObject *frame_it = PyTuple_GET_ITEM(top, 1);
            int advanced = 0;
            PyObject *w;
            while ((w = PyIter_Next(frame_it))) {
                int w_has = PyDict_Contains(index, w);
                if (w_has < 0) { Py_DECREF(w); ok = 0; break; }
                if (!w_has) {
                    if (PyDict_SetItem(index, w, PyLong_FromLong(counter)) < 0 ||
                        PyDict_SetItem(lowlink, w, PyLong_FromLong(counter)) < 0 ||
                        PyList_Append(stack, w) < 0 ||
                        PySet_Add(on_stack, w) < 0) {
                        Py_DECREF(w); ok = 0; break;
                    }
                    counter++;
                    PyObject *wt = PyDict_GetItem(self->adj, w);
                    PyObject *wkeys = wt ? PyDict_Keys(wt) : PyList_New(0);
                    PyObject *wit = PyObject_GetIter(wkeys);
                    Py_DECREF(wkeys);
                    PyObject *wframe = PyTuple_Pack(2, w, wit);
                    Py_DECREF(wit);
                    Py_DECREF(w);
                    if (!wframe || PyList_Append(frames, wframe) < 0) {
                        Py_XDECREF(wframe); ok = 0; break;
                    }
                    Py_DECREF(wframe);
                    advanced = 1;
                    break;
                }
                int on = PySet_Contains(on_stack, w);
                if (on < 0) { Py_DECREF(w); ok = 0; break; }
                if (on) {
                    long idx_w = PyLong_AsLong(PyDict_GetItem(index, w));
                    long low_v = PyLong_AsLong(PyDict_GetItem(lowlink, v));
                    if (PyErr_Occurred()) { Py_DECREF(w); ok = 0; break; }
                    if (idx_w < low_v) {
                        if (PyDict_SetItem(lowlink, v, PyLong_FromLong(idx_w)) < 0) {
                            Py_DECREF(w); ok = 0; break;
                        }
                    }
                }
                Py_DECREF(w);
            }
            if (!ok) break;
            if (advanced) continue;
            PyObject *popped = PyList_GET_ITEM(frames, PyList_GET_SIZE(frames) - 1);
            Py_INCREF(popped);
            PyList_SetSlice(frames, PyList_GET_SIZE(frames) - 1,
                            PyList_GET_SIZE(frames), NULL);
            if (PyList_GET_SIZE(frames) > 0) {
                PyObject *parent = PyTuple_GET_ITEM(
                    PyList_GET_ITEM(frames, PyList_GET_SIZE(frames) - 1), 0);
                long low_v = PyLong_AsLong(PyDict_GetItem(lowlink, v));
                long low_p = PyLong_AsLong(PyDict_GetItem(lowlink, parent));
                if (PyErr_Occurred()) { Py_DECREF(popped); ok = 0; break; }
                if (low_v < low_p) {
                    if (PyDict_SetItem(lowlink, parent,
                                       PyLong_FromLong(low_v)) < 0) {
                        Py_DECREF(popped); ok = 0; break;
                    }
                }
            }
            long low_v = PyLong_AsLong(PyDict_GetItem(lowlink, v));
            long idx_v = PyLong_AsLong(PyDict_GetItem(index, v));
            if (PyErr_Occurred()) { Py_DECREF(popped); ok = 0; break; }
            if (low_v == idx_v) {
                PyObject *comp = PyList_New(0);
                if (!comp) { Py_DECREF(popped); ok = 0; break; }
                while (1) {
                    PyObject *cw = PyList_GET_ITEM(stack,
                                                   PyList_GET_SIZE(stack) - 1);
                    Py_INCREF(cw);
                    PyList_SetSlice(stack, PyList_GET_SIZE(stack) - 1,
                                    PyList_GET_SIZE(stack), NULL);
                    PySet_Discard(on_stack, cw);
                    if (PyList_Append(comp, cw) < 0) {
                        Py_DECREF(cw); Py_DECREF(comp);
                        Py_DECREF(popped); ok = 0; break;
                    }
                    int same = PyObject_RichCompareBool(cw, v, Py_EQ);
                    Py_DECREF(cw);
                    if (same < 0) {
                        Py_DECREF(comp); Py_DECREF(popped); ok = 0; break;
                    }
                    if (same) break;
                }
                if (!ok) break;
                if (PyList_Append(components, comp) < 0) {
                    Py_DECREF(comp); Py_DECREF(popped); ok = 0; break;
                }
                Py_DECREF(comp);
            }
            Py_DECREF(popped);
        }
        if (!ok) break;
    }
    Py_DECREF(viter);
    Py_DECREF(index); Py_DECREF(lowlink); Py_DECREF(on_stack);
    Py_DECREF(stack); Py_DECREF(frames);
    unlock_graph(self->lock);
    if (!ok) {
        Py_DECREF(components);
        return NULL;
    }
    return components;
}

/* ---- Kahn topological sort ---- */
static PyObject *dg_topological_sort(DirectedGraphObject *self,
                                      PyObject *Py_UNUSED(ign)) {
    if (lock_graph(self->lock) < 0) return NULL;
    PyObject *indeg = PyDict_New();
    PyObject *queue = PyList_New(0);
    PyObject *order = PyList_New(0);
    if (!indeg || !queue || !order) {
        Py_XDECREF(indeg); Py_XDECREF(queue); Py_XDECREF(order);
        unlock_graph(self->lock);
        return PyErr_NoMemory();
    }
    /* indeg[v] = in_deg.get(v, 0) for v in vertices */
    PyObject *viter = PyObject_GetIter(self->vertices);
    if (!viter) {
        Py_DECREF(indeg); Py_DECREF(queue); Py_DECREF(order);
        unlock_graph(self->lock);
        return NULL;
    }
    PyObject *v;
    int ok = 1;
    while ((v = PyIter_Next(viter))) {
        PyObject *cur = PyDict_GetItem(self->in_deg, v);
        long n = cur ? PyLong_AsLong(cur) : 0;
        Py_DECREF(v);
        if (PyErr_Occurred()) { ok = 0; break; }
        if (n == 0) {
            Py_INCREF(v);
            if (PyDict_SetItem(indeg, v, PyLong_FromLong(0)) < 0 ||
                PyList_Append(queue, v) < 0) {
                Py_DECREF(v); ok = 0; break;
            }
            Py_DECREF(v);
        } else if (PyDict_SetItem(indeg, v, PyLong_FromLong(n)) < 0) {
            ok = 0; break;
        }
    }
    Py_DECREF(viter);
    if (!ok) {
        Py_DECREF(indeg); Py_DECREF(queue); Py_DECREF(order);
        unlock_graph(self->lock);
        return NULL;
    }
    Py_ssize_t head = 0;
    while (head < PyList_GET_SIZE(queue)) {
        PyObject *q = PyList_GET_ITEM(queue, head++);
        Py_INCREF(q);
        if (PyList_Append(order, q) < 0) {
            Py_DECREF(q); ok = 0; break;
        }
        PyObject *targets = PyDict_GetItem(self->adj, q);
        if (targets) {
            PyObject *titer = PyObject_GetIter(targets);
            if (!titer) { Py_DECREF(q); ok = 0; break; }
            PyObject *w;
            while ((w = PyIter_Next(titer))) {
                PyObject *cur = PyDict_GetItem(indeg, w);
                long n = cur ? PyLong_AsLong(cur) : 0;
                if (PyErr_Occurred()) { Py_DECREF(w); ok = 0; break; }
                PyObject *cnt = PyDict_GetItem(targets, w);
                long c = cnt ? PyLong_AsLong(cnt) : 0;
                if (PyErr_Occurred()) { Py_DECREF(w); ok = 0; break; }
                n -= c;
                if (PyDict_SetItem(indeg, w, PyLong_FromLong(n)) < 0) {
                    Py_DECREF(w); ok = 0; break;
                }
                if (n == 0 && PyList_Append(queue, w) < 0) {
                    Py_DECREF(w); ok = 0; break;
                }
                Py_DECREF(w);
            }
            Py_DECREF(titer);
            if (!ok) { Py_DECREF(q); break; }
        }
        Py_DECREF(q);
        if (!ok) break;
    }
    Py_DECREF(indeg); Py_DECREF(queue);
    unlock_graph(self->lock);
    if (!ok) { Py_DECREF(order); return NULL; }
    if (PyList_GET_SIZE(order) != PySet_GET_SIZE(self->vertices)) {
        Py_DECREF(order);
        Py_RETURN_NONE;
    }
    return order;
}

/* ---- reachable / shortest_path (BFS over _adj inside the lock) ---- */
static PyObject *dg_reachable(DirectedGraphObject *self, PyObject *args) {
    PyObject *src, *dst;
    if (!PyArg_ParseTuple(args, "OO", &src, &dst)) return NULL;
    if (lock_graph(self->lock) < 0) return NULL;
    int has_src = PySet_Contains(self->vertices, src);
    int has_dst = PySet_Contains(self->vertices, dst);
    if (has_src < 0 || has_dst < 0) {
        unlock_graph(self->lock);
        return NULL;
    }
    if (!has_src || !has_dst) {
        unlock_graph(self->lock);
        Py_RETURN_FALSE;
    }
    int eq = PyObject_RichCompareBool(src, dst, Py_EQ);
    if (eq < 0) { unlock_graph(self->lock); return NULL; }
    if (eq == 1) { unlock_graph(self->lock); Py_RETURN_TRUE; }

    PyObject *seen = PySet_New(NULL);
    PyObject *queue = PyList_New(0);
    if (!seen || !queue) {
        Py_XDECREF(seen); Py_XDECREF(queue);
        unlock_graph(self->lock);
        return PyErr_NoMemory();
    }
    if (PySet_Add(seen, src) < 0 || PyList_Append(queue, src) < 0) {
        Py_DECREF(seen); Py_DECREF(queue);
        unlock_graph(self->lock);
        return NULL;
    }
    Py_ssize_t head = 0;
    int found = 0;
    while (head < PyList_GET_SIZE(queue)) {
        PyObject *v = PyList_GET_ITEM(queue, head++);
        PyObject *targets = PyDict_GetItem(self->adj, v);
        if (!targets) continue;
        PyObject *titer = PyObject_GetIter(targets);
        if (!titer) {
            Py_DECREF(seen); Py_DECREF(queue);
            unlock_graph(self->lock);
            return NULL;
        }
        PyObject *w;
        while ((w = PyIter_Next(titer))) {
            int d = PyObject_RichCompareBool(w, dst, Py_EQ);
            if (d < 0) { Py_DECREF(w); Py_DECREF(titer);
                         Py_DECREF(seen); Py_DECREF(queue);
                         unlock_graph(self->lock); return NULL; }
            if (d == 1) { Py_DECREF(w); Py_DECREF(titer); found = 1; break; }
            int s = PySet_Contains(seen, w);
            if (s < 0) { Py_DECREF(w); Py_DECREF(titer);
                         Py_DECREF(seen); Py_DECREF(queue);
                         unlock_graph(self->lock); return NULL; }
            if (!s) {
                if (PySet_Add(seen, w) < 0 || PyList_Append(queue, w) < 0) {
                    Py_DECREF(w); Py_DECREF(titer);
                    Py_DECREF(seen); Py_DECREF(queue);
                    unlock_graph(self->lock); return NULL;
                }
            }
            Py_DECREF(w);
        }
        Py_DECREF(titer);
        if (found || PyErr_Occurred()) break;
    }
    Py_DECREF(seen); Py_DECREF(queue);
    unlock_graph(self->lock);
    if (PyErr_Occurred()) return NULL;
    return PyBool_FromLong(found);
}

/* BFS shortest path over the internal adjacency (inside the lock). */
static PyObject *dg_bfs(DirectedGraphObject *self, PyObject *src,
                         PyObject *dst) {
    int eq = PyObject_RichCompareBool(src, dst, Py_EQ);
    if (eq < 0) return NULL;
    if (eq == 1) {
        PyObject *single = PyList_New(1);
        if (!single) return NULL;
        Py_INCREF(src);
        PyList_SET_ITEM(single, 0, src);
        return single;
    }
    PyObject *prev = PyDict_New();
    PyObject *queue = PyList_New(0);
    if (!prev || !queue) {
        Py_XDECREF(prev); Py_XDECREF(queue);
        return PyErr_NoMemory();
    }
    Py_INCREF(Py_None);
    if (PyDict_SetItem(prev, src, Py_None) < 0 ||
        PyList_Append(queue, src) < 0) {
        Py_DECREF(prev); Py_DECREF(queue);
        return NULL;
    }
    Py_ssize_t head = 0;
    while (head < PyList_GET_SIZE(queue)) {
        PyObject *v = PyList_GET_ITEM(queue, head++);
        PyObject *targets = PyDict_GetItem(self->adj, v);
        if (!targets) continue;
        PyObject *titer = PyObject_GetIter(targets);
        if (!titer) { Py_DECREF(prev); Py_DECREF(queue); return NULL; }
        PyObject *w;
        int done = 0;
        while ((w = PyIter_Next(titer))) {
            int s = PyDict_Contains(prev, w);
            if (s < 0) { Py_DECREF(w); Py_DECREF(titer);
                         Py_DECREF(prev); Py_DECREF(queue); return NULL; }
            if (!s) {
                Py_INCREF(v);
                if (PyDict_SetItem(prev, w, v) < 0 ||
                    PyList_Append(queue, w) < 0) {
                    Py_DECREF(v); Py_DECREF(w); Py_DECREF(titer);
                    Py_DECREF(prev); Py_DECREF(queue); return NULL;
                }
                Py_DECREF(v);
                int f = PyObject_RichCompareBool(w, dst, Py_EQ);
                if (f < 0) { Py_DECREF(w); Py_DECREF(titer);
                             Py_DECREF(prev); Py_DECREF(queue); return NULL; }
                if (f == 1) {
                    PyObject *path = PyList_New(0);
                    if (!path) { Py_DECREF(w); Py_DECREF(titer);
                                 Py_DECREF(prev); Py_DECREF(queue);
                                 return NULL; }
                    PyObject *cur = w;
                    Py_INCREF(cur);
                    while (PyObject_RichCompareBool(cur, src, Py_EQ) != 1) {
                        if (PyList_Append(path, cur) < 0) {
                            Py_DECREF(cur); Py_DECREF(w); Py_DECREF(titer);
                            Py_DECREF(prev); Py_DECREF(queue);
                            Py_DECREF(path); return NULL;
                        }
                        PyObject *p = PyDict_GetItem(prev, cur);
                        Py_DECREF(cur);
                        cur = p;
                        Py_INCREF(cur);
                    }
                    Py_DECREF(cur);
                    if (PyList_Append(path, src) < 0 ||
                        PyList_Reverse(path) < 0) {
                        Py_DECREF(w); Py_DECREF(titer);
                        Py_DECREF(prev); Py_DECREF(queue);
                        Py_DECREF(path); return NULL;
                    }
                    Py_DECREF(w); Py_DECREF(titer);
                    Py_DECREF(prev); Py_DECREF(queue);
                    return path;
                }
            }
            Py_DECREF(w);
        }
        Py_DECREF(titer);
        if (done || PyErr_Occurred()) break;
    }
    Py_DECREF(prev); Py_DECREF(queue);
    if (PyErr_Occurred()) return NULL;
    Py_RETURN_NONE;
}

static PyObject *dg_shortest_path(DirectedGraphObject *self, PyObject *args) {
    PyObject *src, *dst;
    if (!PyArg_ParseTuple(args, "OO", &src, &dst)) return NULL;
    if (lock_graph(self->lock) < 0) return NULL;
    PyObject *r = dg_bfs(self, src, dst);
    unlock_graph(self->lock);
    return r;
}

/* ---- Eulerian tests ---- */
static int dg_non_isolated_weakly_connected(DirectedGraphObject *self) {
    /* all vertices with non-zero degree share one weak component */
    int seen_active = 0;
    PyObject *root = NULL;
    PyObject *viter = PyObject_GetIter(self->vertices);
    if (!viter) return -1;
    PyObject *v;
    while ((v = PyIter_Next(viter))) {
        PyObject *in_c = PyDict_GetItem(self->in_deg, v);
        PyObject *out_c = PyDict_GetItem(self->out_deg, v);
        long in_n = in_c ? PyLong_AsLong(in_c) : 0;
        long out_n = out_c ? PyLong_AsLong(out_c) : 0;
        Py_DECREF(v);
        if (PyErr_Occurred()) { Py_DECREF(viter); return -1; }
        if (in_n == 0 && out_n == 0) continue;
        if (!seen_active) {
            root = dg_find(self, v);
            if (!root) { Py_DECREF(viter); return -1; }
            seen_active = 1;
        } else {
            PyObject *r = dg_find(self, v);
            if (!r) { Py_DECREF(root); Py_DECREF(viter); return -1; }
            int same = PyObject_RichCompareBool(r, root, Py_EQ);
            Py_DECREF(r);
            Py_DECREF(root);
            root = NULL;
            if (same < 0) { Py_DECREF(viter); return -1; }
            if (!same) { Py_DECREF(viter); return 0; }
        }
    }
    Py_DECREF(viter);
    Py_XDECREF(root);
    if (PyErr_Occurred()) return -1;
    return 1;
}

static PyObject *dg_has_eulerian_circuit(DirectedGraphObject *self,
                                          PyObject *Py_UNUSED(ign)) {
    if (lock_graph(self->lock) < 0) return NULL;
    PyObject *viter = PyObject_GetIter(self->vertices);
    if (!viter) { unlock_graph(self->lock); return NULL; }
    PyObject *v;
    int ok = 1;
    while ((v = PyIter_Next(viter))) {
        PyObject *in_c = PyDict_GetItem(self->in_deg, v);
        PyObject *out_c = PyDict_GetItem(self->out_deg, v);
        long in_n = in_c ? PyLong_AsLong(in_c) : 0;
        long out_n = out_c ? PyLong_AsLong(out_c) : 0;
        Py_DECREF(v);
        if (PyErr_Occurred() || in_n != out_n) {
            if (PyErr_Occurred()) ok = 0;
            else {
                Py_DECREF(viter);
                unlock_graph(self->lock);
                Py_RETURN_FALSE;
            }
            break;
        }
    }
    Py_DECREF(viter);
    if (!ok) { unlock_graph(self->lock); return NULL; }
    int w = dg_non_isolated_weakly_connected(self);
    unlock_graph(self->lock);
    if (w < 0) return NULL;
    return PyBool_FromLong(w);
}

static PyObject *dg_has_eulerian_path(DirectedGraphObject *self,
                                       PyObject *Py_UNUSED(ign)) {
    if (lock_graph(self->lock) < 0) return NULL;
    int w = dg_non_isolated_weakly_connected(self);
    if (w < 0) { unlock_graph(self->lock); return NULL; }
    if (!w) { unlock_graph(self->lock); Py_RETURN_FALSE; }
    long start = 0, end = 0;
    PyObject *viter = PyObject_GetIter(self->vertices);
    if (!viter) { unlock_graph(self->lock); return NULL; }
    PyObject *v;
    while ((v = PyIter_Next(viter))) {
        PyObject *in_c = PyDict_GetItem(self->in_deg, v);
        PyObject *out_c = PyDict_GetItem(self->out_deg, v);
        long in_n = in_c ? PyLong_AsLong(in_c) : 0;
        long out_n = out_c ? PyLong_AsLong(out_c) : 0;
        Py_DECREF(v);
        if (PyErr_Occurred()) {
            Py_DECREF(viter); unlock_graph(self->lock); return NULL;
        }
        long d = out_n - in_n;
        if (d == 1) start++;
        else if (d == -1) end++;
        else if (d != 0) {
            Py_DECREF(viter);
            unlock_graph(self->lock);
            Py_RETURN_FALSE;
        }
    }
    Py_DECREF(viter);
    unlock_graph(self->lock);
    return PyBool_FromLong(start <= 1 && start == end);
}

static PyObject *dg_repr(DirectedGraphObject *self) {
    if (lock_graph(self->lock) < 0) return NULL;
    Py_ssize_t v = PySet_GET_SIZE(self->vertices);
    PyObject *r = PyUnicode_FromFormat(
        "<DirectedGraph: V=%zd, E=%zd, weak C=%zd>",
        v, self->edges, self->components);
    unlock_graph(self->lock);
    return r;
}

static PyMethodDef dgraph_methods[] = {
    {"add_edge", (PyCFunction)dg_add_edge, METH_VARARGS,
     "Add a directed arc from u to v (parallel and self-loops allowed)."},
    {"vertices_count", (PyCFunction)dg_vertices_count, METH_NOARGS,
     "Return the number of vertices."},
    {"edges_count", (PyCFunction)dg_edges_count, METH_NOARGS,
     "Return the number of arcs."},
    {"in_degree", (PyCFunction)dg_in_degree, METH_VARARGS,
     "Return the in-degree of v."},
    {"out_degree", (PyCFunction)dg_out_degree, METH_VARARGS,
     "Return the out-degree of v."},
    {"neighbors", (PyCFunction)dg_neighbors, METH_VARARGS,
     "Return the distinct successors of v as a tuple."},
    {"weak_components_count", (PyCFunction)dg_weak_count, METH_NOARGS,
     "Return the number of weak components."},
    {"is_weakly_connected", (PyCFunction)dg_is_weakly_connected,
     METH_NOARGS, "Return True if weakly connected."},
    {"strong_components", (PyCFunction)dg_strong_components, METH_NOARGS,
     "Return the strongly connected components (iterative Tarjan)."},
    {"strong_components_count", (PyCFunction)dg_strong_components,
     METH_NOARGS, "Return the number of strongly connected components."},
    {"topological_sort", (PyCFunction)dg_topological_sort, METH_NOARGS,
     "Return a topological order or None if cyclic (Kahn)."},
    {"is_dag", (PyCFunction)dg_topological_sort, METH_NOARGS,
     "Return True if the graph is a DAG."},
    {"reachable", (PyCFunction)dg_reachable, METH_VARARGS,
     "Return True if dst is reachable from src."},
    {"shortest_path", (PyCFunction)dg_shortest_path, METH_VARARGS,
     "Return the shortest directed path [src, ..., dst] or None."},
    {"has_eulerian_circuit", (PyCFunction)dg_has_eulerian_circuit,
     METH_NOARGS, "Return True if an Eulerian circuit exists."},
    {"has_eulerian_path", (PyCFunction)dg_has_eulerian_path,
     METH_NOARGS, "Return True if an Eulerian path exists."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject DirectedGraphType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_topology.DirectedGraph",
    .tp_basicsize = sizeof(DirectedGraphObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = dg_new,
    .tp_dealloc = (destructor)dg_dealloc,
    .tp_repr = (reprfunc)dg_repr,
    .tp_methods = dgraph_methods,
    .tp_doc = "Directed multigraph with iterative algorithms."
};

/* ==================================================================
 * Pure functions
 * ================================================================== */

static PyObject *py_euler_characteristic(PyObject *self, PyObject *args) { (void)self;
    PyObject *cell_list;
    if (!PyArg_ParseTuple(args, "O", &cell_list)) return NULL;
    PyObject *iter = PyObject_GetIter(cell_list);
    if (!iter) return NULL;
    long long factor = 1;
    long long acc = 0;
    int big = 0;
    PyObject *item;
    while ((item = PyIter_Next(iter))) {
        if (Py_TYPE(item) != &PyLong_Type) {
            Py_DECREF(item); Py_DECREF(iter);
            PyErr_SetString(PyExc_TypeError,
                            "Cell must be a positive integer.");
            return NULL;
        }
        long long cell = PyLong_AsLongLong(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            PyErr_Clear();
            big = 1;
            break;
        }
        if (cell <= 0) {
            Py_DECREF(iter);
            PyErr_SetString(PyExc_ValueError,
                            "Cell must be a positive integer.");
            return NULL;
        }
        if ((factor != 0 && cell > LLONG_MAX / factor) ||
            (factor != 0 && cell < LLONG_MIN / factor)) {
            big = 1;
            break;
        }
        long long term = cell * factor;
        if ((term > 0 && acc > LLONG_MAX - term) ||
            (term < 0 && acc < LLONG_MIN - term)) {
            big = 1;
            break;
        }
        acc += term;
        factor = -factor;
    }
    if (PyErr_Occurred()) {
        Py_DECREF(iter);
        return NULL;
    }
    if (big) {
        Py_DECREF(iter);
        PyObject *new_iter = PyObject_GetIter(cell_list);
        if (!new_iter) return NULL;
        PyObject *f = PyLong_FromLong(1);
        PyObject *res = PyLong_FromLong(0);
        if (!f || !res) {
            Py_XDECREF(f); Py_XDECREF(res); Py_DECREF(new_iter);
            return NULL;
        }
        PyObject *it;
        while ((it = PyIter_Next(new_iter))) {
            if (Py_TYPE(it) != &PyLong_Type) {
                Py_DECREF(it);
                Py_DECREF(new_iter); Py_DECREF(f); Py_DECREF(res);
                PyErr_SetString(PyExc_TypeError,
                                "Cell must be a positive integer.");
                return NULL;
            }
            long long c = PyLong_AsLongLong(it);
            if (PyErr_Occurred()) PyErr_Clear();
            else if (c <= 0) {
                Py_DECREF(it);
                Py_DECREF(new_iter); Py_DECREF(f); Py_DECREF(res);
                PyErr_SetString(PyExc_ValueError,
                                "Cell must be a positive integer.");
                return NULL;
            }
            PyObject *term = PyNumber_Multiply(it, f);
            PyObject *nr = PyNumber_Add(res, term);
            Py_XDECREF(term);
            Py_DECREF(res);
            if (!nr) {
                Py_DECREF(it); Py_DECREF(new_iter); Py_DECREF(f);
                return NULL;
            }
            res = nr;
            PyObject *nf = PyNumber_Negative(f);
            Py_DECREF(f);
            if (!nf) {
                Py_DECREF(it); Py_DECREF(new_iter); Py_DECREF(res);
                return NULL;
            }
            f = nf;
            Py_DECREF(it);
        }
        Py_DECREF(new_iter); Py_DECREF(f);
        return res;
    }
    Py_DECREF(iter);
    return PyLong_FromLongLong(acc);
}

static PyObject *py_shortest_path(PyObject *self, PyObject *args) { (void)self;
    PyObject *graph, *src, *dst;
    if (!PyArg_ParseTuple(args, "OOO", &graph, &src, &dst)) return NULL;
    int eq = PyObject_RichCompareBool(src, dst, Py_EQ);
    if (eq < 0) return NULL;
    if (eq == 1) {
        PyObject *single = PyList_New(1);
        if (!single) return NULL;
        Py_INCREF(src);
        PyList_SET_ITEM(single, 0, src);
        return single;
    }
    PyObject *neighbors = PyObject_GetAttrString(graph, "neighbors");
    if (!neighbors) return NULL;
    if (!PyCallable_Check(neighbors)) {
        Py_DECREF(neighbors);
        PyErr_SetString(PyExc_TypeError,
                        "graph must expose a callable neighbors(v)");
        return NULL;
    }
    PyObject *prev = PyDict_New();
    PyObject *queue = PyList_New(0);
    if (!prev || !queue) {
        Py_XDECREF(prev); Py_XDECREF(queue); Py_DECREF(neighbors);
        return PyErr_NoMemory();
    }
    Py_INCREF(Py_None);
    if (PyDict_SetItem(prev, src, Py_None) < 0 ||
        PyList_Append(queue, src) < 0) {
        Py_DECREF(prev); Py_DECREF(queue); Py_DECREF(neighbors);
        return NULL;
    }
    Py_ssize_t head = 0;
    while (head < PyList_GET_SIZE(queue)) {
        PyObject *v = PyList_GET_ITEM(queue, head++);
        PyObject *call_args = PyTuple_Pack(1, v);
        if (!call_args) {
            Py_DECREF(prev); Py_DECREF(queue); Py_DECREF(neighbors);
            return NULL;
        }
        PyObject *nbrs = PyObject_CallObject(neighbors, call_args);
        Py_DECREF(call_args);
        if (!nbrs) {
            if (PyErr_ExceptionMatches(PyExc_KeyError)) {
                PyErr_Clear();
                continue;
            }
            Py_DECREF(prev); Py_DECREF(queue); Py_DECREF(neighbors);
            return NULL;
        }
        PyObject *niter = PyObject_GetIter(nbrs);
        Py_DECREF(nbrs);
        if (!niter) {
            Py_DECREF(prev); Py_DECREF(queue); Py_DECREF(neighbors);
            return NULL;
        }
        PyObject *w;
        while ((w = PyIter_Next(niter))) {
            int seen = PyDict_Contains(prev, w);
            if (seen < 0) {
                Py_DECREF(w); Py_DECREF(niter);
                Py_DECREF(prev); Py_DECREF(queue); Py_DECREF(neighbors);
                return NULL;
            }
            if (seen == 0) {
                Py_INCREF(v);
                if (PyDict_SetItem(prev, w, v) < 0) {
                    Py_DECREF(w); Py_DECREF(niter);
                    Py_DECREF(prev); Py_DECREF(queue); Py_DECREF(neighbors);
                    return NULL;
                }
                Py_DECREF(v);
                int found = PyObject_RichCompareBool(w, dst, Py_EQ);
                if (found < 0) {
                    Py_DECREF(w); Py_DECREF(niter);
                    Py_DECREF(prev); Py_DECREF(queue); Py_DECREF(neighbors);
                    return NULL;
                }
                if (found == 1) {
                    PyObject *path = PyList_New(0);
                    if (!path) {
                        Py_DECREF(w); Py_DECREF(niter);
                        Py_DECREF(prev); Py_DECREF(queue);
                        Py_DECREF(neighbors);
                        return NULL;
                    }
                    PyObject *cur = w;
                    Py_INCREF(cur);
                    while (PyObject_RichCompareBool(cur, src, Py_EQ) != 1) {
                        if (PyList_Append(path, cur) < 0) {
                            Py_DECREF(cur); Py_DECREF(w); Py_DECREF(niter);
                            Py_DECREF(prev); Py_DECREF(queue);
                            Py_DECREF(neighbors); Py_DECREF(path);
                            return NULL;
                        }
                        PyObject *p = PyDict_GetItem(prev, cur);
                        Py_DECREF(cur);
                        cur = p;
                        Py_INCREF(cur);
                    }
                    Py_DECREF(cur);
                    if (PyList_Append(path, src) < 0 ||
                        PyList_Reverse(path) < 0) {
                        Py_DECREF(w); Py_DECREF(niter);
                        Py_DECREF(prev); Py_DECREF(queue);
                        Py_DECREF(neighbors); Py_DECREF(path);
                        return NULL;
                    }
                    Py_DECREF(w); Py_DECREF(niter);
                    Py_DECREF(prev); Py_DECREF(queue); Py_DECREF(neighbors);
                    return path;
                }
                if (PyList_Append(queue, w) < 0) {
                    Py_DECREF(w); Py_DECREF(niter);
                    Py_DECREF(prev); Py_DECREF(queue); Py_DECREF(neighbors);
                    return NULL;
                }
            }
            Py_DECREF(w);
        }
        Py_DECREF(niter);
        if (PyErr_Occurred()) {
            Py_DECREF(prev); Py_DECREF(queue); Py_DECREF(neighbors);
            return NULL;
        }
    }
    Py_DECREF(prev); Py_DECREF(queue); Py_DECREF(neighbors);
    Py_RETURN_NONE;
}

/* ==================================================================
 * Module definition
 * ================================================================== */

static PyMethodDef methods[] = {
    {"shortest_path_between", py_shortest_path, METH_VARARGS,
     "BFS shortest path [src, ..., dst] or None (duck neighbors protocol)."},
    {"Euler_characteristic_compute_by_cell", py_euler_characteristic,
     METH_VARARGS,
     "Euler characteristic from a cell list."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_topology",
    "C99 optimized topology algorithms and graph types (math_tool).",
    0,
    methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__topology(void);

PyMODINIT_FUNC PyInit__topology(void) {
    if (PyType_Ready(&GraphType) < 0 || PyType_Ready(&DirectedGraphType) < 0)
        return NULL;
    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;
    Py_INCREF(&GraphType);
    if (PyModule_AddObject(m, "Graph", (PyObject*)&GraphType) < 0) {
        Py_DECREF(&GraphType);
        Py_DECREF(m);
        return NULL;
    }
    Py_INCREF(&DirectedGraphType);
    if (PyModule_AddObject(m, "DirectedGraph",
                           (PyObject*)&DirectedGraphType) < 0) {
        Py_DECREF(&DirectedGraphType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
