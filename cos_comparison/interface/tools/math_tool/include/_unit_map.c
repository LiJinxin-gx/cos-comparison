/*
 * _unit_map.c - run-folding unit mapper (C99, duck typing).
 *
 * Mirrors the pure-Python reference (unit_map.py) exactly: any objects
 * are accepted as units (compared with ==), consecutive-equal runs fold
 * into single real flag elements; the content table (object <-> flag) is
 * instance-held and cumulative; put() writes through the sequence
 * protocol with a file-pointer model and per-call buffering; run
 * statistics are internal; state export/restore and clear are provided.
 *
 * Memory discipline: the tables are Python-owned containers (PyDict for
 * hashable units, a linear PyList of records otherwise); no C-side
 * unbounded buffers exist - the mapper keeps only the pending run and
 * the queued (closed, not yet output) runs.  All references are owned
 * and released on every path (dealloc / error cleanup).
 *
 * C99 features used deliberately: stdint.h types, designated
 * initializers, inline helpers, function pointers.
 */

#ifdef _MSC_VER
/* CPython extension ABI patterns under /Wall (see cos_comparison_pydll.c):
   C4191 method-table casts, C4232 PyType_GenericNew address,
   C4820 struct padding; C5045/C4711/C4710 are /Wall performance hints. */
#pragma warning(push)
#pragma warning(disable: 4191 4232 4820 5045 4711 4710)
#endif
#include <Python.h>
#include <math.h>
#include <stdint.h>

/* Python 3.8/3.9 compatibility: Py_NewRef is 3.10+. */
#if PY_VERSION_HEX < 0x030A0000
static inline PyObject *Py_NewRef(PyObject *obj) {
    Py_INCREF(obj);
    return obj;
}
#endif

/* ------------------------------------------------------------------ */
/* object type                                                          */
/* ------------------------------------------------------------------ */

typedef struct {
    PyObject_HEAD
    double start;
    double step;
    long threshold;     /* fold frequency threshold (>= 1)            */
    PyObject *length;   /* None/int/tuple (normalized) or NULL        */
    PyObject *shape;    /* unit-grid shape tuple or NULL              */
    PyObject *lens;     /* per-dimension caps tuple or NULL           */
    PyObject *partial;  /* (item, is_raw, next, remaining) or NULL    */
    int output_started; /* any element written?                       */
    long long units_in;     /* units pushed                            */
    long long queued_units; /* pushed but not yet emitted              */
    PyObject *table;    /* dict: hashable obj -> float flag          */
    PyObject *plain_sig;/* dict: content signature -> [rec, ...]     */
    PyObject *flags;    /* dict: float flag -> obj (reverse)          */
    PyObject *counts;   /* dict: hashable obj -> [count, runs]        */
    PyObject *pending;  /* (obj, length) tuple or NULL                */
    PyObject *queued;   /* list of (obj, length) closed, not output   */
    PyObject *last_out; /* output object of the previous put or NULL  */
    Py_ssize_t out_pos; /* file pointer within the current output     */
    long nunique;       /* total unique units (flag sequence order)   */
} UnitMapObject;

/* ------------------------------------------------------------------ */
/* helpers                                                              */
/* ------------------------------------------------------------------ */

static int um_hashable(PyObject *obj)
{
    Py_hash_t h = PyObject_Hash(obj);
    if (h == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }
    return 1;
}


/* Bounded 61-bit container signature: iterative post-order fold
 * (children before parents); order-insensitive for mappings (pairs
 * sorted by key hash); collisions resolved by deep equality. */
#define UM_SIG_MASK 0x1FFFFFFFFFFFFFFFLL
#define UM_SIG_MUL  1000003LL

typedef enum { UM_EXPAND, UM_COLLAPSE } um_frame_phase;

static int um_hash_long(PyObject *obj, long long *out)
{
    Py_hash_t h = PyObject_Hash(obj);
    if (h == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        {
            /* unhashable leaf: fold its repr instead */
            PyObject *r = PyObject_Repr(obj);
            if (r == NULL) {
                return -1;
            }
            h = PyObject_Hash(r);
            Py_DECREF(r);
            if (h == -1 && PyErr_Occurred()) {
                return -1;
            }
        }
    }
    *out = (long long)h & UM_SIG_MASK;
    return 0;
}

static int um_memo_put(PyObject *memo, PyObject *node, long long v)
{
    PyObject *key = PyLong_FromVoidPtr(node);
    PyObject *val = PyLong_FromLongLong(v);
    int rc;
    if (key == NULL || val == NULL) {
        Py_XDECREF(key);
        Py_XDECREF(val);
        return -1;
    }
    rc = PyDict_SetItem(memo, key, val);
    Py_DECREF(key);
    Py_DECREF(val);
    return rc;
}

static int um_memo_get(PyObject *memo, PyObject *node, long long *out)
{
    PyObject *key = PyLong_FromVoidPtr(node);
    PyObject *val;
    if (key == NULL) {
        return -1;
    }
    val = PyDict_GetItemWithError(memo, key);
    Py_DECREF(key);
    if (val == NULL) {
        if (PyErr_Occurred()) {
            return -1;
        }
        return 1;
    }
    *out = PyLong_AsLongLong(val);
    return PyErr_Occurred() ? -1 : 0;
}

static int um_push_frame(PyObject ***nodes, int **phases, size_t *cap,
                         size_t *len, PyObject *node, int phase)
{
    if (*len >= *cap) {
        size_t ncap = *cap == 0 ? 64 : *cap * 2;
        PyObject **nn = (PyObject **)PyMem_Malloc(
            ncap * sizeof(PyObject *));
        int *np = (int *)PyMem_Malloc(ncap * sizeof(int));
        if (nn == NULL || np == NULL) {
            PyMem_Free(nn);
            PyMem_Free(np);
            return -1;
        }
        if (*len > 0) {
            memcpy(nn, *nodes, *len * sizeof(PyObject *));
            memcpy(np, *phases, *len * sizeof(int));
        }
        PyMem_Free(*nodes);
        PyMem_Free(*phases);
        *nodes = nn;
        *phases = np;
        *cap = ncap;
    }
    (*nodes)[*len] = node;
    (*phases)[*len] = phase;
    *len += 1;
    return 0;
}

/* Protocol helpers: duck sequence / mapping / set (text excluded,
 * mappings are never sequences). */
static int um_is_mapping_proto(PyObject *obj)
{
    if (!PyObject_HasAttrString(obj, "keys")) return 0;
    return PyObject_HasAttrString(obj, "__getitem__");
}

static int um_is_seq_proto(PyObject *obj)
{
    if (PyUnicode_Check(obj) || PyBytes_Check(obj)) return 0;
    if (um_is_mapping_proto(obj)) return 0;
    if (!PySequence_Check(obj)) return 0;
    if (PyObject_Length(obj) < 0) { PyErr_Clear(); return 0; }
    return 1;
}

static int um_is_set_proto(PyObject *obj)
{
    if (PyUnicode_Check(obj) || PyBytes_Check(obj)) return 0;
    if (um_is_mapping_proto(obj)) return 0;
    if (PyObject_HasAttrString(obj, "__getitem__")) return 0;
    if (!PyObject_HasAttrString(obj, "__contains__")) return 0;
    return PyObject_HasAttrString(obj, "__iter__");
}

/* Structural labels: 1 = mutable (list/set-like), 2 = immutable. */
static int um_seq_label(PyObject *obj)
{
    return PyObject_HasAttrString(obj, "__setitem__") ? 1 : 2;
}

static int um_set_label(PyObject *obj)
{
    return PyObject_HasAttrString(obj, "add") ? 1 : 2;
}

/* Children of a container (stable order); mapping children are pushed
 * as [k, v, ...].  Returns the child count (0 for atoms). */
static Py_ssize_t um_children(PyObject *node, PyObject ***out)
{
    Py_ssize_t n;
    Py_ssize_t i;
    PyObject **kids;
    if (um_is_seq_proto(node)) {
        n = PySequence_Size(node);
        if (n < 0) {
            return -1;
        }
        kids = (PyObject **)PyMem_Malloc(
            (size_t)(n > 0 ? n : 1) * sizeof(PyObject *));
        if (kids == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        for (i = 0; i < n; ++i) {
            kids[i] = PySequence_GetItem(node, i);
            if (kids[i] == NULL) {
                Py_ssize_t j;
                for (j = 0; j < i; ++j) {
                    Py_DECREF(kids[j]);
                }
                PyMem_Free(kids);
                return -1;
            }
        }
        *out = kids;
        return n;
    }
    if (um_is_mapping_proto(node)) {
        PyObject *keys = PyObject_CallMethod(node, "keys", NULL);
        PyObject *keys_list = keys != NULL ? PySequence_List(keys) : NULL;
        Py_XDECREF(keys);
        if (keys_list == NULL) {
            return -1;
        }
        n = PyList_GET_SIZE(keys_list);
        kids = (PyObject **)PyMem_Malloc(
            (size_t)(n > 0 ? n * 2 : 2) * sizeof(PyObject *));
        if (kids == NULL) {
            Py_DECREF(keys_list);
            PyErr_NoMemory();
            return -1;
        }
        for (i = 0; i < n; ++i) {
            PyObject *k = PyList_GET_ITEM(keys_list, i);
            PyObject *v = PyObject_GetItem(node, k);
            if (v == NULL) {
                Py_ssize_t j;
                for (j = 0; j < i * 2; ++j) {
                    Py_DECREF(kids[j]);
                }
                PyMem_Free(kids);
                Py_DECREF(keys_list);
                return -1;
            }
            Py_INCREF(k);
            kids[i * 2] = k;
            kids[i * 2 + 1] = v;
        }
        Py_DECREF(keys_list);
        *out = kids;
        return n * 2;
    }
    if (um_is_set_proto(node)) {
        PyObject *items = PySequence_List(node);
        if (items == NULL) {
            return -1;
        }
        n = PyList_GET_SIZE(items);
        kids = (PyObject **)PyMem_Malloc(
            (size_t)(n > 0 ? n : 1) * sizeof(PyObject *));
        if (kids == NULL) {
            Py_DECREF(items);
            PyErr_NoMemory();
            return -1;
        }
        for (i = 0; i < n; ++i) {
            PyObject *item = PyList_GET_ITEM(items, i);
            Py_INCREF(item);
            kids[i] = item;
        }
        Py_DECREF(items);
        *out = kids;
        return n;
    }
    return 0;  /* atomic leaf */
}

/* Signature of a unit (PyLong) or NULL on error - iterative post-order. */
static PyObject *um_sig(PyObject *obj, int depth)
{
    PyObject *memo = NULL;
    PyObject *result = NULL;
    PyObject **nodes = NULL;
    int *phases = NULL;
    size_t cap = 0;
    size_t len = 0;
    (void)depth;
    memo = PyDict_New();
    if (memo == NULL) {
        return NULL;
    }
    if (um_push_frame(&nodes, &phases, &cap, &len, obj, UM_EXPAND) < 0) {
        PyErr_NoMemory();
        goto done;
    }
    while (len > 0) {
        PyObject *node;
        int phase;
        PyObject **kids = NULL;
        Py_ssize_t nk;
        len -= 1;
        node = nodes[len];
        phase = phases[len];
        if (phase == UM_EXPAND) {
            nk = um_children(node, &kids);
            if (nk < 0) {
                goto done;
            }
            if (nk == 0) {
                long long v;
                if (um_hash_long(node, &v) < 0 ||
                    um_memo_put(memo, node, v) < 0) {
                    goto done;
                }
                continue;
            }
            /* push the collapse frame, then children reversed */
            if (um_push_frame(&nodes, &phases, &cap, &len, node,
                              UM_COLLAPSE) < 0) {
                Py_ssize_t j;
                for (j = 0; j < nk; ++j) {
                    Py_DECREF(kids[j]);
                }
                PyMem_Free(kids);
                PyErr_NoMemory();
                goto done;
            }
            {
                Py_ssize_t j;
                for (j = nk - 1; j >= 0; --j) {
                    if (um_push_frame(&nodes, &phases, &cap, &len,
                                      kids[j], UM_EXPAND) < 0) {
                        Py_ssize_t t;
                        for (t = 0; t <= j; ++t) {
                            Py_DECREF(kids[t]);
                        }
                        for (t = j + 1; t < nk; ++t) {
                            Py_DECREF(kids[t]);
                        }
                        PyMem_Free(kids);
                        PyErr_NoMemory();
                        goto done;
                    }
                }
                for (j = 0; j < nk; ++j) {
                    Py_DECREF(kids[j]);
                }
                PyMem_Free(kids);
            }
            continue;
        }
        /* collapse: fold the node from its children memo values */
        nk = um_children(node, &kids);
        if (nk < 0) {
            goto done;
        }
        if (nk == 0) {
            /* should not happen (atoms are folded at expand) */
            continue;
        }
        {
            long long acc;
            long long *vals = (long long *)PyMem_Malloc(
                (size_t)nk * sizeof(long long));
            Py_ssize_t j;
            if (vals == NULL) {
                Py_ssize_t t;
                for (t = 0; t < nk; ++t) {
                    Py_DECREF(kids[t]);
                }
                PyMem_Free(kids);
                PyErr_NoMemory();
                goto done;
            }
            for (j = 0; j < nk; ++j) {
                if (um_memo_get(memo, kids[j], &vals[j]) < 0) {
                    PyMem_Free(vals);
                    Py_ssize_t t;
                    for (t = 0; t < nk; ++t) {
                        Py_DECREF(kids[t]);
                    }
                    PyMem_Free(kids);
                    goto done;
                }
            }
            if (um_is_seq_proto(node)) {
                acc = um_seq_label(node);
                for (j = 0; j < nk; ++j) {
                    acc = (acc * UM_SIG_MUL + vals[j]) & UM_SIG_MASK;
                }
            } else if (um_is_mapping_proto(node)) {
                /* pairs sorted by key hash (order-insensitive) */
                acc = 3;
                {
                    Py_ssize_t m = nk / 2;
                    /* insertion sort on the pairs by key value */
                    Py_ssize_t p;
                    for (p = 1; p < m; ++p) {
                        long long kv = vals[p * 2];
                        long long vv = vals[p * 2 + 1];
                        Py_ssize_t q = p;
                        while (q > 0 &&
                               vals[(q - 1) * 2] > kv) {
                            vals[q * 2] = vals[(q - 1) * 2];
                            vals[q * 2 + 1] =
                                vals[(q - 1) * 2 + 1];
                            --q;
                        }
                        vals[q * 2] = kv;
                        vals[q * 2 + 1] = vv;
                    }
                    for (p = 0; p < m; ++p) {
                        acc = (acc * UM_SIG_MUL + vals[p * 2])
                            & UM_SIG_MASK;
                        acc = (acc * UM_SIG_MUL + vals[p * 2 + 1])
                            & UM_SIG_MASK;
                    }
                }
            } else {  /* set-like */
                acc = 4;
                for (j = 0; j < nk; ++j) {
                    acc = (acc * UM_SIG_MUL + vals[j]) & UM_SIG_MASK;
                }
            }
            PyMem_Free(vals);
            for (j = 0; j < nk; ++j) {
                Py_DECREF(kids[j]);
            }
            PyMem_Free(kids);
            if (um_memo_put(memo, node, acc) < 0) {
                goto done;
            }
        }
    }
    {
        long long top;
        if (um_memo_get(memo, obj, &top) < 0 || top < 0) {
            goto done;
        }
        result = PyLong_FromLongLong(top);
    }
done:
    PyMem_Free(nodes);
    PyMem_Free(phases);
    Py_DECREF(memo);
    return result;
}

static int um_push_frame_ab(PyObject ***sa, PyObject ***sb, size_t *cap,
                            size_t *len, PyObject *a, PyObject *b)
{
    if (*len >= *cap) {
        size_t ncap = *cap == 0 ? 64 : *cap * 2;
        PyObject **na = (PyObject **)PyMem_Malloc(
            ncap * sizeof(PyObject *));
        PyObject **nb = (PyObject **)PyMem_Malloc(
            ncap * sizeof(PyObject *));
        if (na == NULL || nb == NULL) {
            PyMem_Free(na);
            PyMem_Free(nb);
            return -1;
        }
        if (*len > 0) {
            memcpy(na, *sa, *len * sizeof(PyObject *));
            memcpy(nb, *sb, *len * sizeof(PyObject *));
        }
        PyMem_Free(*sa);
        PyMem_Free(*sb);
        *sa = na;
        *sb = nb;
        *cap = ncap;
    }
    (*sa)[*len] = a;
    (*sb)[*len] = b;
    *len += 1;
    return 0;
}

/* Iterative deep equality: containers compare through the sequence /
 * mapping / set protocols (structure labels keep list != tuple and
 * set != frozenset); leaves use their own ==. */
static int um_deep_eq(PyObject *a, PyObject *b)
{
    PyObject **sa = NULL;
    PyObject **sb = NULL;
    size_t cap = 0;
    size_t len = 0;
    int result = 0;
    if (um_push_frame_ab(&sa, &sb, &cap, &len, a, b) < 0) {
        PyErr_NoMemory();
        return -1;
    }
    while (len > 0) {
        PyObject *x;
        PyObject *y;
        len -= 1;
        x = sa[len];
        y = sb[len];
        if (x == y) {
            continue;
        }
        if (um_is_seq_proto(x)) {
            Py_ssize_t n;
            Py_ssize_t i;
            if (!um_is_seq_proto(y) ||
                um_seq_label(x) != um_seq_label(y)) {
                result = 0;
                goto done;
            }
            n = PySequence_Size(x);
            if (n < 0) {
                result = -1;
                goto done;
            }
            if (PySequence_Size(y) != n) {
                result = 0;
                goto done;
            }
            for (i = n - 1; i >= 0; --i) {
                PyObject *xi = PySequence_GetItem(x, i);
                PyObject *yi = PySequence_GetItem(y, i);
                if (xi == NULL || yi == NULL) {
                    Py_XDECREF(xi); Py_XDECREF(yi);
                    result = -1;
                    goto done;
                }
                if (um_push_frame_ab(&sa, &sb, &cap, &len, xi, yi) < 0) {
                    Py_DECREF(xi); Py_DECREF(yi);
                    PyErr_NoMemory();
                    result = -1;
                    goto done;
                }
                /* the containers keep the children alive */
                Py_DECREF(xi);
                Py_DECREF(yi);
            }
            continue;
        }
        if (um_is_mapping_proto(x)) {
            PyObject *kx;
            PyObject *ky;
            PyObject *iter;
            PyObject *k;
            int eq;
            if (!um_is_mapping_proto(y)) {
                result = 0;
                goto done;
            }
            kx = PyObject_CallMethod(x, "keys", NULL);
            ky = PyObject_CallMethod(y, "keys", NULL);
            if (kx == NULL || ky == NULL) {
                Py_XDECREF(kx); Py_XDECREF(ky);
                result = -1;
                goto done;
            }
            eq = PyObject_RichCompareBool(kx, ky, Py_EQ);
            Py_DECREF(ky);
            if (eq < 0) {
                Py_DECREF(kx);
                result = -1;
                goto done;
            }
            if (!eq) {
                Py_DECREF(kx);
                result = 0;
                goto done;
            }
            iter = PyObject_GetIter(kx);
            Py_DECREF(kx);
            if (iter == NULL) {
                result = -1;
                goto done;
            }
            while ((k = PyIter_Next(iter)) != NULL) {
                PyObject *xv = PyObject_GetItem(x, k);
                PyObject *yv = PyObject_GetItem(y, k);
                if (xv == NULL || yv == NULL) {
                    Py_XDECREF(xv); Py_XDECREF(yv);
                    Py_DECREF(k); Py_DECREF(iter);
                    result = -1;
                    goto done;
                }
                if (um_push_frame_ab(&sa, &sb, &cap, &len, xv, yv) < 0) {
                    Py_DECREF(xv); Py_DECREF(yv);
                    Py_DECREF(k); Py_DECREF(iter);
                    PyErr_NoMemory();
                    result = -1;
                    goto done;
                }
                Py_DECREF(xv);
                Py_DECREF(yv);
                Py_DECREF(k);
            }
            Py_DECREF(iter);
            if (PyErr_Occurred()) {
                result = -1;
                goto done;
            }
            continue;
        }
        if (um_is_set_proto(x)) {
            int eq;
            if (!um_is_set_proto(y) ||
                um_set_label(x) != um_set_label(y)) {
                result = 0;
                goto done;
            }
            eq = PyObject_RichCompareBool(x, y, Py_EQ);
            if (eq < 0) {
                result = -1;
                goto done;
            }
            if (!eq) {
                result = 0;
                goto done;
            }
            continue;
        }
        {
            int eq = PyObject_RichCompareBool(x, y, Py_EQ);
            if (eq < 0) {
                result = -1;
                goto done;
            }
            if (!eq) {
                result = 0;
                goto done;
            }
        }
    }
    result = 1;
done:
    PyMem_Free(sa);
    PyMem_Free(sb);
    return result;
}

/* Records of one signature bucket (usually a single ==-verified rec);
 * returns Py_None when the signature is unknown (no records). */
static PyObject *um_sig_bucket(UnitMapObject *self, PyObject *obj)
{
    PyObject *sig = um_sig(obj, 0);
    PyObject *bucket;
    if (sig == NULL) {
        return NULL;
    }
    bucket = PyDict_GetItemWithError(self->plain_sig, sig);
    Py_DECREF(sig);
    if (bucket == NULL && PyErr_Occurred()) {
        return NULL;
    }
    if (bucket == NULL) {
        bucket = Py_None;
    }
    Py_INCREF(bucket);
    return bucket;
}


/* flag_of: content-table query (returns a new reference or NULL). */
static PyObject *um_flag_of(UnitMapObject *self, PyObject *obj)
{
    PyObject *flag = PyDict_GetItemWithError(self->table, obj);
    if (flag != NULL) {
        Py_INCREF(flag);
        return flag;
    }
    if (PyErr_Occurred()) {
        if (!PyErr_ExceptionMatches(PyExc_TypeError)) {
            return NULL;
        }
        PyErr_Clear();
    }
    if (PyDict_GET_SIZE(self->plain_sig) == 0) {
        Py_RETURN_NONE;
    }
    {
        PyObject *bucket = um_sig_bucket(self, obj);
        Py_ssize_t i;
        Py_ssize_t n;
        if (bucket == NULL) {
            return NULL;
        }
        if (bucket == Py_None) {  /* signature unknown: no records */
            Py_DECREF(bucket);
            Py_RETURN_NONE;
        }
        n = PyList_GET_SIZE(bucket);
        for (i = 0; i < n; ++i) {
            PyObject *rec = PyList_GET_ITEM(bucket, i);
            int eq = um_deep_eq(PyList_GET_ITEM(rec, 0), obj);
            if (eq < 0) {
                Py_DECREF(bucket);
                return NULL;
            }
            if (eq) {
                flag = PyList_GET_ITEM(rec, 1);
                Py_INCREF(flag);
                Py_DECREF(bucket);
                return flag;
            }
        }
        Py_DECREF(bucket);
    }
    Py_RETURN_NONE;
}

/* _register: allocate the next flag (first-sight order across tables). */
static int um_register(UnitMapObject *self, PyObject *obj)
{
    PyObject *flag = NULL;
    int hashable = um_hashable(obj);
    double next = self->start + self->step * (double)self->nunique;
    flag = PyFloat_FromDouble(next);
    if (flag == NULL) {
        return -1;
    }
    if (hashable) {
        if (PyDict_SetItem(self->table, obj, flag) < 0) {
            goto fail;
        }
    } else {
        PyObject *sig = um_sig(obj, 0);
        PyObject *bucket;
        PyObject *rec;
        PyObject *z1;
        PyObject *z2;
        if (sig == NULL) {
            goto fail;
        }
        bucket = PyDict_GetItemWithError(self->plain_sig, sig);
        if (bucket == NULL && PyErr_Occurred()) {
            Py_DECREF(sig);
            goto fail;
        }
        if (bucket == NULL) {
            bucket = PyList_New(0);
            if (bucket == NULL ||
                PyDict_SetItem(self->plain_sig, sig, bucket) < 0) {
                Py_XDECREF(bucket);
                Py_DECREF(sig);
                goto fail;
            }
        }
        Py_DECREF(sig);
        rec = PyList_New(4);
        z1 = PyLong_FromLong(0);
        z2 = PyLong_FromLong(0);
        if (rec == NULL || z1 == NULL || z2 == NULL) {
            Py_XDECREF(rec);
            Py_XDECREF(z1);
            Py_XDECREF(z2);
            goto fail;
        }
        Py_INCREF(obj);
        PyList_SET_ITEM(rec, 0, obj);
        Py_INCREF(flag);
        PyList_SET_ITEM(rec, 1, flag);
        PyList_SET_ITEM(rec, 2, z1);
        PyList_SET_ITEM(rec, 3, z2);
        if (PyList_Append(bucket, rec) < 0) {
            Py_DECREF(rec);
            goto fail;
        }
        Py_DECREF(rec);
    }
    if (self->flags != NULL &&
        PyDict_SetItem(self->flags, flag, obj) < 0) {
        goto fail;
    }
    self->nunique += 1;
    Py_DECREF(flag);
    return 0;
fail:
    Py_XDECREF(flag);
    return -1;
}
/* reverse map, built lazily on first decode/get_state. */
static PyObject *um_ensure_flags(UnitMapObject *self)
{
    PyObject *rev;
    PyObject *key;
    PyObject *val;
    Py_ssize_t pos = 0;
    if (self->flags != NULL) {
        return self->flags;
    }
    rev = PyDict_New();
    if (rev == NULL) {
        return NULL;
    }
    while (PyDict_Next(self->table, &pos, &key, &val)) {
        if (PyDict_SetItem(rev, val, key) < 0) {
            Py_DECREF(rev);
            return NULL;
        }
    }
    {
        PyObject *bucket;
        Py_ssize_t bpos = 0;
        while (PyDict_Next(self->plain_sig, &bpos, &key, &bucket)) {
            Py_ssize_t j;
            for (j = 0; j < PyList_GET_SIZE(bucket); ++j) {
                PyObject *rec = PyList_GET_ITEM(bucket, j);
                if (PyDict_SetItem(rev, PyList_GET_ITEM(rec, 1),
                                   PyList_GET_ITEM(rec, 0)) < 0) {
                    Py_DECREF(rev);
                    return NULL;
                }
            }
        }
    }
    self->flags = rev;
    return rev;
}

/* bump: run statistics (total elements and run count). */
static void um_bump(UnitMapObject *self, PyObject *obj, long length)
{
    PyObject *rec;
    if (um_hashable(obj)) {
        rec = PyDict_GetItemWithError(self->counts, obj);
        if (rec != NULL) {
            PyObject *cnt = PyList_GET_ITEM(rec, 0);
            PyObject *runs = PyList_GET_ITEM(rec, 1);
            PyObject *nc = PyLong_FromLong(
                PyLong_AsLong(cnt) + length);
            PyObject *nr = PyLong_FromLong(
                PyLong_AsLong(runs) + 1);
            if (nc != NULL && nr != NULL) {
                PyList_SET_ITEM(rec, 0, nc);
                PyList_SET_ITEM(rec, 1, nr);
                return;
            }
            Py_XDECREF(nc);
            Py_XDECREF(nr);
            PyErr_Clear();
            return;
        }
        PyErr_Clear();
        rec = PyList_New(2);
        if (rec == NULL) {
            return;
        }
        PyList_SET_ITEM(rec, 0, PyLong_FromLong(length));
        PyList_SET_ITEM(rec, 1, PyLong_FromLong(1));
        if (PyList_GET_ITEM(rec, 0) == NULL ||
            PyList_GET_ITEM(rec, 1) == NULL) {
            Py_DECREF(rec);
            return;
        }
        PyDict_SetItem(self->counts, obj, rec);
        Py_DECREF(rec);
        return;
    }
    {
        PyObject *bucket = um_sig_bucket(self, obj);
        Py_ssize_t i;
        Py_ssize_t n;
        if (bucket == NULL) {
            PyErr_Clear();
            return;
        }
        if (bucket == Py_None) {  /* signature unknown: no records */
            Py_DECREF(bucket);
            return;
        }
        n = PyList_GET_SIZE(bucket);
        for (i = 0; i < n; ++i) {
            PyObject *pr = PyList_GET_ITEM(bucket, i);
            int eq = um_deep_eq(PyList_GET_ITEM(pr, 0), obj);
            if (eq < 0) {
                PyErr_Clear();
                break;
            }
            if (eq) {
                PyObject *cnt = PyList_GET_ITEM(pr, 2);
                PyObject *runs = PyList_GET_ITEM(pr, 3);
                PyObject *nc = PyLong_FromLong(
                    PyLong_AsLong(cnt) + length);
                PyObject *nr = PyLong_FromLong(
                    PyLong_AsLong(runs) + 1);
                if (nc != NULL && nr != NULL) {
                    PyList_SET_ITEM(pr, 2, nc);
                    PyList_SET_ITEM(pr, 3, nr);
                } else {
                    Py_XDECREF(nc);
                    Py_XDECREF(nr);
                    PyErr_Clear();
                }
                break;
            }
        }
        Py_DECREF(bucket);
    }
}

/* close the pending run into the queue + statistics. */
static int um_close(UnitMapObject *self)
{
    PyObject *tuple = NULL;
    if (self->pending == NULL) {
        return 0;
    }
    um_bump(self, PyTuple_GET_ITEM(self->pending, 0),
            PyLong_AsLong(PyTuple_GET_ITEM(self->pending, 1)));
    tuple = self->pending;
    self->pending = NULL;
    if (PyList_Append(self->queued, tuple) < 0) {
        Py_DECREF(tuple);
        return -1;
    }
    Py_DECREF(tuple);
    return 0;
}

/* push one unit (register on first sight, extend/close the run). */
/* fast equality (defined below with the output helpers). */
static int um_fast_eq(PyObject *a, PyObject *b);

static int um_push(UnitMapObject *self, PyObject *unit)
{
    self->units_in += 1;
    self->queued_units += 1;
    if (self->pending != NULL) {
        PyObject *pend_obj = PyTuple_GET_ITEM(self->pending, 0);
        int eq = um_fast_eq(pend_obj, unit);
        if (eq < 0) {
            return -1;
        }
        if (eq) {
            long length = PyLong_AsLong(
                PyTuple_GET_ITEM(self->pending, 1));
            PyObject *nl = PyLong_FromLong(length + 1);
            PyObject *nt;
            if (nl == NULL) {
                return -1;
            }
            nt = PyTuple_Pack(2, pend_obj, nl);
            Py_DECREF(nl);
            if (nt == NULL) {
                return -1;
            }
            Py_DECREF(self->pending);
            self->pending = nt;
            return 0;
        }
        if (um_close(self) < 0) {
            return -1;
        }
    }
    {
        PyObject *flag = um_flag_of(self, unit);
        int need;
        if (flag == NULL) {
            return -1;
        }
        need = (flag == Py_None);
        Py_DECREF(flag);
        if (need && um_register(self, unit) < 0) {
            return -1;
        }
        if (self->pending != NULL) {
            Py_DECREF(self->pending);
        }
        self->pending = PyTuple_Pack(2, unit, PyLong_FromLong(1));
        if (self->pending == NULL) {
            return -1;
        }
    }
    return 0;
}

#ifndef UM_MAXDIM
#define UM_MAXDIM 16
#endif

/* scalar type test for the fast equality path. */
static int um_scalar_type(PyObject *obj)
{
    return PyLong_Check(obj) || PyFloat_Check(obj) ||
           PyComplex_Check(obj) || PyUnicode_Check(obj) ||
           PyBytes_Check(obj);
}

/* scalar-leaf tuple test (nested containers fall back to deep_eq). */
static int um_scalar_tuple(PyObject *obj)
{
    Py_ssize_t i;
    for (i = 0; i < PyTuple_GET_SIZE(obj); ++i) {
        if (!um_scalar_type(PyTuple_GET_ITEM(obj, i))) {
            return 0;
        }
    }
    return 1;
}

/* equality with a scalar / scalar-tuple fast path (same semantics as
 * um_deep_eq; compound or duck values fall back). */
static int um_fast_eq(PyObject *a, PyObject *b)
{
    if (a == b) {
        return 1;
    }
    if (Py_TYPE(a) == Py_TYPE(b)) {
        if (um_scalar_type(a)) {
            return PyObject_RichCompareBool(a, b, Py_EQ);
        }
        if (PyTuple_Check(a) && um_scalar_tuple(a) &&
            um_scalar_tuple(b)) {
            return PyObject_RichCompareBool(a, b, Py_EQ);
        }
    }
    return um_deep_eq(a, b);
}

/* first greedy maximal-prefix piece within the per-dimension caps
 * (row-major unit grid); returns 0 on error. */
static long um_piece(UnitMapObject *self, long long start, long remaining)
{
    Py_ssize_t rank;
    long coords[UM_MAXDIM];
    long lo[UM_MAXDIM];
    long hi[UM_MAXDIM];
    long shape[UM_MAXDIM];
    long caps[UM_MAXDIM];
    long long index = start;
    long piece = 0;
    Py_ssize_t d;
    long k;
    if (self->lens == NULL) {
        return remaining;
    }
    if (self->shape == NULL) {
        long cap = PyLong_AsLong(PyTuple_GET_ITEM(self->lens, 0));
        return cap < remaining ? cap : remaining;
    }
    rank = PyTuple_GET_SIZE(self->shape);
    if (rank > UM_MAXDIM) {
        rank = UM_MAXDIM;
    }
    for (d = rank - 1; d >= 0; --d) {
        long s = PyLong_AsLong(PyTuple_GET_ITEM(self->shape, d));
        long c = PyLong_AsLong(PyTuple_GET_ITEM(self->lens, d));
        if (s < 1 || c < 1) {
            PyErr_SetString(PyExc_ValueError, "bad shape/length");
            return -1;
        }
        shape[d] = s;
        caps[d] = c;
        coords[d] = (long)(index % s);
        index /= s;
        lo[d] = coords[d];
        hi[d] = coords[d];
    }
    for (k = 0; k < remaining; ++k) {
        int too_wide = 0;
        for (d = 0; d < rank; ++d) {
            long nd = coords[d];
            long nlo = lo[d] < nd ? lo[d] : nd;
            long nhi = hi[d] > nd ? hi[d] : nd;
            if (nhi - nlo + 1 > caps[d]) {
                too_wide = 1;
                break;
            }
        }
        if (too_wide) {
            break;
        }
        for (d = 0; d < rank; ++d) {
            if (coords[d] < lo[d]) {
                lo[d] = coords[d];
            }
            if (coords[d] > hi[d]) {
                hi[d] = coords[d];
            }
        }
        piece += 1;
        for (d = rank - 1; d >= 0; --d) {
            coords[d] += 1;
            if (coords[d] < shape[d]) {
                break;
            }
            coords[d] = 0;
        }
    }
    return piece;
}

/* live count of a unit (hashable dict / signature records). */
static long um_count_of(UnitMapObject *self, PyObject *obj)
{
    if (um_hashable(obj)) {
        PyObject *rec = PyDict_GetItemWithError(self->counts, obj);
        if (rec != NULL) {
            return PyLong_AsLong(PyList_GET_ITEM(rec, 0));
        }
        PyErr_Clear();
    }
    {
        PyObject *bucket = um_sig_bucket(self, obj);
        Py_ssize_t i;
        long count = 0;
        if (bucket == NULL) {
            PyErr_Clear();
            return 0;
        }
        if (bucket != Py_None) {
            for (i = 0; i < PyList_GET_SIZE(bucket); ++i) {
                PyObject *rec = PyList_GET_ITEM(bucket, i);
                int eq = um_deep_eq(PyList_GET_ITEM(rec, 0), obj);
                if (eq < 0) {
                    PyErr_Clear();
                    break;
                }
                if (eq) {
                    count = PyLong_AsLong(PyList_GET_ITEM(rec, 2));
                    break;
                }
            }
        }
        Py_DECREF(bucket);
        return count;
    }
}

/* write one element into the output (pre-allocated slot, else append). */
static int um_write(UnitMapObject *self, PyObject *output,
                    PyObject *flag)
{
    PyObject *idx;
    PyObject *append;
    int res;
    if (PyList_Check(output)) {
        Py_ssize_t size = PyList_GET_SIZE(output);
        if (self->out_pos < size) {
            PyObject *old = PyList_GET_ITEM(output, self->out_pos);
            Py_INCREF(flag);
            PyList_SET_ITEM(output, self->out_pos, flag);
            Py_DECREF(old);
        } else if (PyList_Append(output, flag) < 0) {
            return -1;
        }
        self->out_pos += 1;
        return 1;
    }
    idx = PyLong_FromSsize_t(self->out_pos);
    if (idx == NULL) {
        return -1;
    }
    res = PyObject_SetItem(output, idx, flag);
    Py_DECREF(idx);
    if (res == 0) {
        self->out_pos += 1;
        return 1;
    }
    if (!PyErr_ExceptionMatches(PyExc_IndexError) &&
        !PyErr_ExceptionMatches(PyExc_TypeError)) {
        return -1;
    }
    PyErr_Clear();
    append = PyObject_GetAttrString(output, "append");
    if (append != NULL) {
        PyObject *r = PyObject_CallFunctionObjArgs(append, flag, NULL);
        Py_DECREF(append);
        if (r == NULL) {
            return -1;
        }
        Py_DECREF(r);
        self->out_pos += 1;
        return 1;
    }
    PyErr_Clear();
    return 0;
}

/* ------------------------------------------------------------------ */
/* methods                                                              */
/* ------------------------------------------------------------------ */

static PyObject *um_add(UnitMapObject *self, PyObject *args)
{
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    Py_ssize_t a;
    for (a = 0; a < nargs; ++a) {
        PyObject *iter = PyObject_GetIter(
            PyTuple_GET_ITEM(args, a));
        PyObject *unit;
        if (iter == NULL) {
            return NULL;
        }
        while ((unit = PyIter_Next(iter)) != NULL) {
            if (um_push(self, unit) < 0) {
                Py_DECREF(unit);
                Py_DECREF(iter);
                return NULL;
            }
            Py_DECREF(unit);
        }
        Py_DECREF(iter);
        if (PyErr_Occurred()) {
            return NULL;
        }
    }
    return PyLong_FromLong(0);
}

static PyObject *um_put(UnitMapObject *self, PyObject *args,
                        PyObject *kwargs)
{
    static char *kwlist[] = {"output", "buffering", NULL};
    PyObject *output = Py_None;
    PyObject *buffering = Py_None;
    PyObject *fresh = NULL;
    PyObject *partial = self->partial;
    PyObject *queued = self->queued;
    long limit = -1;
    int has_limit = 0;
    Py_ssize_t written = 0;
    Py_ssize_t index = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO", kwlist,
                                     &output, &buffering)) {
        return NULL;
    }
    if (buffering != Py_None) {
        limit = PyLong_AsLong(buffering);
        if (limit == -1 && PyErr_Occurred()) {
            return NULL;
        }
        has_limit = 1;
    }
    if (um_close(self) < 0) {
        return NULL;
    }
    if (output == Py_None) {
        fresh = PyList_New(0);
        if (fresh == NULL) {
            return NULL;
        }
        output = fresh;
        self->out_pos = 0;
    } else if (output != self->last_out) {
        self->out_pos = 0;
    }
    if (self->lens == NULL && self->threshold == 1 && partial == NULL) {
        /* default workflow: one element per run */
        for (index = 0; index < PyList_GET_SIZE(queued); ++index) {
            PyObject *pair = PyList_GET_ITEM(queued, index);
            PyObject *flag;
            int w;
            if (has_limit && written >= (Py_ssize_t)limit) {
                break;
            }
            flag = um_flag_of(self, PyTuple_GET_ITEM(pair, 0));
            if (flag == NULL) {
                Py_XDECREF(fresh);
                return NULL;
            }
            if (flag == Py_None) {
                Py_DECREF(flag);
                Py_XDECREF(fresh);
                PyErr_SetString(PyExc_RuntimeError, "unit not registered");
                return NULL;
            }
            w = um_write(self, output, flag);
            Py_DECREF(flag);
            if (w < 0) {
                Py_XDECREF(fresh);
                return NULL;
            }
            if (w > 0) {
                written += 1;
                self->queued_units -= PyLong_AsLong(PyTuple_GET_ITEM(pair, 1));
            }
        }
        if (index > 0) {
            PyObject *tail = PyList_GetSlice(queued, index,
                                             PyList_GET_SIZE(queued));
            if (tail == NULL) {
                Py_XDECREF(fresh);
                return NULL;
            }
            Py_SETREF(self->queued, tail);
        }
        if (written > 0) {
            self->output_started = 1;
        }
        if (fresh != NULL) {
            return fresh;
        }
        Py_INCREF(output);
        Py_XSETREF(self->last_out, output);
        return PyLong_FromSsize_t(written);
    }
    /* general element stream: eligibility + piece splitting + carry */
    {
        PyObject *item = NULL;
        int is_raw = 0;
        int have = 0;
        long long next_start = 0;
        long remaining = 0;
        if (partial != NULL) {
            item = PyTuple_GET_ITEM(partial, 0);
            is_raw = PyObject_IsTrue(PyTuple_GET_ITEM(partial, 1));
            next_start = PyLong_AsLongLong(PyTuple_GET_ITEM(partial, 2));
            remaining = PyLong_AsLong(PyTuple_GET_ITEM(partial, 3));
            Py_INCREF(item);
            have = 1;
        }
        while (1) {
            int w;
            if (has_limit && written >= (Py_ssize_t)limit) {
                break;
            }
            if (!have) {
                PyObject *pair;
                PyObject *obj;
                if (index >= PyList_GET_SIZE(queued)) {
                    break;
                }
                pair = PyList_GET_ITEM(queued, index);
                obj = PyTuple_GET_ITEM(pair, 0);
                remaining = PyLong_AsLong(PyTuple_GET_ITEM(pair, 1));
                if (self->threshold > 1 &&
                    um_count_of(self, obj) < self->threshold) {
                    is_raw = 1;
                    item = obj;
                    next_start = 0;
                } else {
                    is_raw = 0;
                    item = um_flag_of(self, obj);
                    if (item == NULL) {
                        Py_XDECREF(fresh);
                        return NULL;
                    }
                    if (item == Py_None) {
                        Py_DECREF(item);
                        Py_XDECREF(fresh);
                        PyErr_SetString(PyExc_RuntimeError,
                                        "unit not registered");
                        return NULL;
                    }
                    next_start = self->units_in - self->queued_units;
                }
                Py_INCREF(item);
                have = 1;
            }
            if (is_raw) {
                w = um_write(self, output, item);
                if (w < 0) {
                    Py_XDECREF(item);
                    Py_XDECREF(fresh);
                    return NULL;
                }
                if (w == 0) {
                    break;
                }
                written += 1;
                remaining -= 1;
                self->queued_units -= 1;
            } else {
                long piece = um_piece(self, next_start, remaining);
                if (piece < 0) {
                    Py_XDECREF(item);
                    Py_XDECREF(fresh);
                    return NULL;
                }
                w = um_write(self, output, item);
                if (w < 0) {
                    Py_XDECREF(item);
                    Py_XDECREF(fresh);
                    return NULL;
                }
                if (w == 0) {
                    break;
                }
                written += 1;
                next_start += piece;
                remaining -= piece;
                self->queued_units -= piece;
            }
            if (remaining == 0) {
                Py_CLEAR(item);
                have = 0;
                index += 1;
            }
        }
        if (have) {
            PyObject *np = Py_BuildValue(
                "(OiLL)", item, is_raw, (long long)next_start,
                (long long)remaining);
            if (np == NULL) {
                Py_XDECREF(item);
                Py_XDECREF(fresh);
                return NULL;
            }
            Py_XSETREF(self->partial, np);
        } else {
            Py_CLEAR(self->partial);
        }
        Py_XDECREF(item);
    }
    if (index > 0) {
        PyObject *tail = PyList_GetSlice(queued, index,
                                         PyList_GET_SIZE(queued));
        if (tail == NULL) {
            Py_XDECREF(fresh);
            return NULL;
        }
        Py_SETREF(self->queued, tail);
    }
    if (written > 0) {
        self->output_started = 1;
    }
    if (fresh != NULL) {
        return fresh;
    }
    Py_INCREF(output);
    Py_XSETREF(self->last_out, output);
    return PyLong_FromSsize_t(written);
}

static PyObject *um_get_state(UnitMapObject *self, PyObject *noargs)
{
    PyObject *state = PyDict_New();
    PyObject *plain = NULL;
    PyObject *queued = NULL;
    Py_ssize_t i;
    (void)noargs;
    if (state == NULL) {
        return NULL;
    }
    plain = PyDict_New();
    queued = PyList_New(PyList_GET_SIZE(self->queued));
    if (plain == NULL || queued == NULL) {
        Py_DECREF(plain);
        Py_DECREF(queued);
        Py_DECREF(state);
        return NULL;
    }
    /* deep-ish copy: signature -> bucket of copied records */
    {
        PyObject *key;
        PyObject *bucket;
        Py_ssize_t pos = 0;
        while (PyDict_Next(self->plain_sig, &pos, &key, &bucket)) {
            PyObject *copy = PyList_New(PyList_GET_SIZE(bucket));
            Py_ssize_t j;
            if (copy == NULL) {
                goto fail;
            }
            for (j = 0; j < PyList_GET_SIZE(bucket); ++j) {
                PyObject *rec = PyList_GetSlice(
                    PyList_GET_ITEM(bucket, j), 0, 4);
                if (rec == NULL) {
                    Py_DECREF(copy);
                    goto fail;
                }
                PyList_SET_ITEM(copy, j, rec);
            }
            if (PyDict_SetItem(plain, key, copy) < 0) {
                Py_DECREF(copy);
                goto fail;
            }
            Py_DECREF(copy);
        }
    }
    for (i = 0; i < PyList_GET_SIZE(self->queued); ++i) {
        PyObject *item = PyList_GET_ITEM(self->queued, i);
        Py_INCREF(item);
        PyList_SET_ITEM(queued, i, item);
    }
    {
        PyObject *v;
        v = PyFloat_FromDouble(self->start);
        if (v == NULL || PyDict_SetItemString(state, "start", v) < 0) {
            Py_XDECREF(v);
            goto fail;
        }
        Py_DECREF(v);
        v = PyFloat_FromDouble(self->step);
        if (v == NULL || PyDict_SetItemString(state, "step", v) < 0) {
            Py_XDECREF(v);
            goto fail;
        }
        Py_DECREF(v);
    }
    if (um_ensure_flags(self) == NULL) {
        goto fail;
    }
    if (PyDict_SetItemString(state, "table", self->table) < 0 ||
        PyDict_SetItemString(state, "plain_sig", plain) < 0 ||
        PyDict_SetItemString(state, "flags", self->flags) < 0 ||
        PyDict_SetItemString(state, "counts", self->counts) < 0 ||
        PyDict_SetItemString(state, "queued", queued) < 0 ||
        PyDict_SetItemString(state, "out_pos",
                             PyLong_FromSsize_t(self->out_pos)) < 0 ||
        PyDict_SetItemString(state, "n", PyLong_FromLong(self->nunique)) < 0 ||
        PyDict_SetItemString(state, "threshold",
                             PyLong_FromLong(self->threshold)) < 0 ||
        PyDict_SetItemString(state, "length",
                             self->length != NULL ? self->length
                             : Py_None) < 0 ||
        PyDict_SetItemString(state, "shape",
                             self->shape != NULL ? self->shape
                             : Py_None) < 0 ||
        PyDict_SetItemString(state, "partial",
                             self->partial != NULL ? self->partial
                             : Py_None) < 0 ||
        PyDict_SetItemString(state, "output_started",
                             PyBool_FromLong(self->output_started)) < 0 ||
        PyDict_SetItemString(state, "units_in",
                             PyLong_FromLongLong(self->units_in)) < 0 ||
        PyDict_SetItemString(state, "queued_units",
                             PyLong_FromLongLong(self->queued_units)) < 0 ||
        PyDict_SetItemString(state, "pending",
                             self->pending != NULL ? self->pending
                             : Py_None) < 0) {
        goto fail;
    }
    Py_DECREF(plain);
    Py_DECREF(queued);
    return state;
fail:
    Py_DECREF(plain);
    Py_DECREF(queued);
    Py_DECREF(state);
    return NULL;
}

/* State-field lookup through the mapping protocol (absent -> NULL). */
static PyObject *um_state_get(PyObject *st, const char *key)
{
    PyObject *k = PyUnicode_FromString(key);
    PyObject *v;
    if (k == NULL) return NULL;
    v = PyObject_GetItem(st, k);
    Py_DECREF(k);
    if (v == NULL && PyErr_ExceptionMatches(PyExc_KeyError)) {
        PyErr_Clear();
        return NULL;
    }
    return v;
}

static PyObject *um_set_state(UnitMapObject *self, PyObject *arg)
{
    PyObject *st;
    PyObject *v;
    if (!um_is_mapping_proto(arg)) {
        PyErr_SetString(PyExc_TypeError,
                        "set_state expects a state dict");
        return NULL;
    }
    st = arg;
    if ((v = um_state_get(st, "start")) != NULL) {
        self->start = PyFloat_AsDouble(v);
        Py_DECREF(v);
    }
    if ((v = um_state_get(st, "step")) != NULL) {
        self->step = PyFloat_AsDouble(v);
        Py_DECREF(v);
    }
    if ((v = um_state_get(st, "table")) != NULL) {
        PyObject *copy = PyDict_Check(v) ? PyDict_Copy(v)
                                         : PySequence_Tuple(v);
        Py_DECREF(v);
        if (copy == NULL) {
            return NULL;
        }
        Py_SETREF(self->table, copy);
    }
    if ((v = um_state_get(st, "plain_sig")) != NULL) {
        PyObject *copy;
        PyObject *key;
        PyObject *bucket;
        Py_ssize_t pos = 0;
        if (!PyDict_Check(v)) {
            Py_DECREF(v);
            PyErr_SetString(PyExc_TypeError, "plain_sig must be a dict");
            return NULL;
        }
        copy = PyDict_New();
        if (copy == NULL) {
            Py_DECREF(v);
            return NULL;
        }
        while (PyDict_Next(v, &pos, &key, &bucket)) {
            PyObject *nb = PyList_New(PySequence_Size(bucket));
            Py_ssize_t j;
            if (nb == NULL) {
                Py_DECREF(copy);
                Py_DECREF(v);
                return NULL;
            }
            for (j = 0; j < PySequence_Size(bucket); ++j) {
                PyObject *rec = PySequence_GetItem(bucket, j);
                PyObject *rcopy = rec == NULL ? NULL
                    : PySequence_List(rec);
                Py_XDECREF(rec);
                if (rcopy == NULL) {
                    Py_DECREF(nb);
                    Py_DECREF(copy);
                    Py_DECREF(v);
                    return NULL;
                }
                if (PyList_Append(nb, rcopy) < 0) {
                    Py_DECREF(rcopy);
                    Py_DECREF(nb);
                    Py_DECREF(copy);
                    Py_DECREF(v);
                    return NULL;
                }
                Py_DECREF(rcopy);
            }
            if (PyDict_SetItem(copy, key, nb) < 0) {
                Py_DECREF(nb);
                Py_DECREF(copy);
                Py_DECREF(v);
                return NULL;
            }
            Py_DECREF(nb);
        }
        Py_DECREF(v);
        Py_SETREF(self->plain_sig, copy);
    }
    if ((v = um_state_get(st, "flags")) != NULL) {
        PyObject *copy = PyDict_Check(v) ? PyDict_Copy(v)
                                         : PySequence_Tuple(v);
        Py_DECREF(v);
        if (copy == NULL) {
            return NULL;
        }
        Py_XSETREF(self->flags, copy);
    }
    if ((v = um_state_get(st, "counts")) != NULL) {
        PyObject *copy = PyDict_Check(v) ? PyDict_Copy(v)
                                         : PySequence_Tuple(v);
        Py_DECREF(v);
        if (copy == NULL) {
            return NULL;
        }
        Py_SETREF(self->counts, copy);
    }
    if ((v = um_state_get(st, "pending")) != NULL) {
        if (v == Py_None) {
            Py_DECREF(v);
            Py_CLEAR(self->pending);
        } else {
            Py_XSETREF(self->pending, v);
        }
    }
    if ((v = um_state_get(st, "queued")) != NULL) {
        PyObject *copy = PySequence_List(v);
        Py_DECREF(v);
        if (copy == NULL) {
            return NULL;
        }
        Py_SETREF(self->queued, copy);
    }
    if ((v = um_state_get(st, "out_pos")) != NULL) {
        self->out_pos = PyLong_AsSsize_t(v);
        Py_DECREF(v);
        if (PyErr_Occurred()) {
            return NULL;
        }
    }
    if ((v = um_state_get(st, "n")) != NULL) {
        self->nunique = PyLong_AsLong(v);
        Py_DECREF(v);
        if (PyErr_Occurred()) {
            return NULL;
        }
    }
    if ((v = um_state_get(st, "threshold")) != NULL) {
        self->threshold = PyLong_AsLong(v);
        Py_DECREF(v);
        if (PyErr_Occurred()) {
            return NULL;
        }
    }
    if ((v = um_state_get(st, "length")) != NULL) {
        if (v == Py_None) {
            Py_DECREF(v);
            Py_CLEAR(self->length);
        } else {
            Py_XSETREF(self->length, v);
        }
    }
    if ((v = um_state_get(st, "shape")) != NULL) {
        if (v == Py_None) {
            Py_DECREF(v);
            Py_CLEAR(self->shape);
        } else {
            Py_XSETREF(self->shape, v);
        }
    }
    if ((v = um_state_get(st, "partial")) != NULL) {
        if (v == Py_None) {
            Py_DECREF(v);
            Py_CLEAR(self->partial);
        } else {
            Py_XSETREF(self->partial, v);
        }
    }
    if ((v = um_state_get(st, "output_started")) != NULL) {
        self->output_started = PyObject_IsTrue(v);
        Py_DECREF(v);
    }
    if ((v = um_state_get(st, "units_in")) != NULL) {
        self->units_in = PyLong_AsLongLong(v);
        Py_DECREF(v);
        if (PyErr_Occurred()) {
            return NULL;
        }
    }
    if ((v = um_state_get(st, "queued_units")) != NULL) {
        self->queued_units = PyLong_AsLongLong(v);
        Py_DECREF(v);
        if (PyErr_Occurred()) {
            return NULL;
        }
    }
    /* rebuild the per-dimension caps from length + shape */
    Py_CLEAR(self->lens);
    if (self->length != NULL) {
        if (self->shape == NULL) {
            self->lens = PyTuple_Pack(1, self->length);
        } else if (PyTuple_Check(self->length)) {
            self->lens = PySequence_Tuple(self->length);
        } else {
            Py_ssize_t r = PyTuple_GET_SIZE(self->shape);
            Py_ssize_t i2;
            self->lens = PyTuple_New(r);
            if (self->lens != NULL) {
                for (i2 = 0; i2 < r; ++i2) {
                    Py_INCREF(self->length);
                    PyTuple_SET_ITEM(self->lens, i2, self->length);
                }
            }
        }
        if (self->lens == NULL) {
            return NULL;
        }
    }
    Py_CLEAR(self->last_out);
    Py_RETURN_NONE;
}

static PyObject *um_clear(UnitMapObject *self, PyObject *noargs)
{
    (void)noargs;
    PyDict_Clear(self->table);
    PyDict_Clear(self->plain_sig);
    Py_CLEAR(self->flags);
    PyDict_Clear(self->counts);
    Py_XSETREF(self->pending, NULL);
    PyList_SetSlice(self->queued, 0,
                    PyList_GET_SIZE(self->queued), NULL);
    Py_CLEAR(self->last_out);
    Py_CLEAR(self->partial);
    self->out_pos = 0;
    self->nunique = 0;
    self->output_started = 0;
    self->units_in = 0;
    self->queued_units = 0;
    Py_RETURN_NONE;
}

static PyObject *um_flag_of_method(UnitMapObject *self, PyObject *arg)
{
    return um_flag_of(self, arg);
}

static PyObject *um_decode(UnitMapObject *self, PyObject *arg)
{
    PyObject *iter = PyObject_GetIter(arg);
    PyObject *out;
    PyObject *flag;
    if (iter == NULL) {
        return NULL;
    }
    out = PyList_New(0);
    if (out == NULL) {
        Py_DECREF(iter);
        return NULL;
    }
    if (um_ensure_flags(self) == NULL) {
        Py_DECREF(out);
        Py_DECREF(iter);
        return NULL;
    }
    while ((flag = PyIter_Next(iter)) != NULL) {
        PyObject *obj = PyDict_GetItemWithError(self->flags, flag);
        PyObject *item;
        if (obj == NULL && PyErr_Occurred()) {
            Py_DECREF(flag);
            Py_DECREF(out);
            Py_DECREF(iter);
            return NULL;
        }
        item = obj != NULL ? obj : Py_None;
        Py_INCREF(item);
        if (PyList_Append(out, item) < 0) {
            Py_DECREF(item);
            Py_DECREF(flag);
            Py_DECREF(out);
            Py_DECREF(iter);
            return NULL;
        }
        Py_DECREF(item);
        Py_DECREF(flag);
    }
    Py_DECREF(iter);
    if (PyErr_Occurred()) {
        Py_DECREF(out);
        return NULL;
    }
    return out;
}

static Py_ssize_t um_length(UnitMapObject *self)
{
    return (Py_ssize_t)self->nunique;
}

static int um_contains(UnitMapObject *self, PyObject *obj)
{
    PyObject *flag = um_flag_of(self, obj);
    int res;
    if (flag == NULL) {
        return -1;
    }
    res = (flag != Py_None);
    Py_DECREF(flag);
    return res;
}


/* ------------------------------------------------------------------ */
/* frequency statistics (counts; dynamic - no shared global counter)    */
/* ------------------------------------------------------------------ */

static int um_unit_stats(UnitMapObject *self, PyObject *obj,
                         long long *cnt, long long *runs)
{
    *cnt = 0;
    *runs = 0;
    if (um_hashable(obj)) {
        PyObject *rec = PyDict_GetItemWithError(self->counts, obj);
        if (rec != NULL) {
            *cnt = PyLong_AsLongLong(PyList_GET_ITEM(rec, 0));
            *runs = PyLong_AsLongLong(PyList_GET_ITEM(rec, 1));
        } else if (PyErr_Occurred()) {
            return -1;
        }
    } else {
        PyObject *bucket = um_sig_bucket(self, obj);
        Py_ssize_t i;
        Py_ssize_t n;
        if (bucket == NULL) {
            return -1;
        }
        if (bucket != Py_None) {
            n = PyList_GET_SIZE(bucket);
            for (i = 0; i < n; ++i) {
                PyObject *rec = PyList_GET_ITEM(bucket, i);
                int eq = um_deep_eq(PyList_GET_ITEM(rec, 0), obj);
                if (eq < 0) {
                    Py_DECREF(bucket);
                    return -1;
                }
                if (eq) {
                    *cnt = PyLong_AsLongLong(PyList_GET_ITEM(rec, 2));
                    *runs = PyLong_AsLongLong(PyList_GET_ITEM(rec, 3));
                    break;
                }
            }
        }
        Py_DECREF(bucket);
    }
    /* live statistics: add the open pending run when it matches */
    if (self->pending != NULL) {
        PyObject *pend_obj = PyTuple_GET_ITEM(self->pending, 0);
        int eq = um_deep_eq(pend_obj, obj);
        if (eq < 0) {
            return -1;
        }
        if (eq) {
            *cnt += PyLong_AsLong(
                PyTuple_GET_ITEM(self->pending, 1));
            *runs += 1;
        }
    }
    return 0;
}

static PyObject *um_total(UnitMapObject *self, PyObject *noargs)
{
    long long n = 0;
    PyObject *key;
    PyObject *val;
    Py_ssize_t pos = 0;
    (void)noargs;
    while (PyDict_Next(self->counts, &pos, &key, &val)) {
        n += PyLong_AsLongLong(PyList_GET_ITEM(val, 0));
    }
    {
        PyObject *bk;
        PyObject *bv;
        Py_ssize_t bpos = 0;
        while (PyDict_Next(self->plain_sig, &bpos, &bk, &bv)) {
            Py_ssize_t i;
            for (i = 0; i < PyList_GET_SIZE(bv); ++i) {
                PyObject *rec = PyList_GET_ITEM(bv, i);
                n += PyLong_AsLongLong(PyList_GET_ITEM(rec, 2));
            }
        }
    }
    if (self->pending != NULL) {
        n += PyLong_AsLong(PyTuple_GET_ITEM(self->pending, 1));
    }
    return PyLong_FromLongLong(n);
}

static PyObject *um_count(UnitMapObject *self, PyObject *arg)
{
    long long cnt;
    long long runs;
    if (um_unit_stats(self, arg, &cnt, &runs) < 0) {
        return NULL;
    }
    return PyLong_FromLongLong(cnt);
}

static PyObject *um_runs(UnitMapObject *self, PyObject *arg)
{
    long long cnt;
    long long runs;
    if (um_unit_stats(self, arg, &cnt, &runs) < 0) {
        return NULL;
    }
    return PyLong_FromLongLong(runs);
}

static int um_mc_cmp(const void *pa, const void *pb)
{
    PyObject *a = *(PyObject *const *)pa;
    PyObject *b = *(PyObject *const *)pb;
    long long ca = PyLong_AsLongLong(PyTuple_GET_ITEM(a, 1));
    long long cb = PyLong_AsLongLong(PyTuple_GET_ITEM(b, 1));
    if (ca < cb) {
        return 1;  /* descending by count */
    }
    if (ca > cb) {
        return -1;
    }
    return 0;
}

static PyObject *um_most_common(UnitMapObject *self, PyObject *args,
                                PyObject *kwargs)
{
    static char *kwlist[] = {"k", NULL};
    PyObject *k_obj = Py_None;
    Py_ssize_t n = 0;
    Py_ssize_t cap = 0;
    Py_ssize_t i = 0;
    PyObject **items = NULL;
    PyObject *out;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist,
                                     &k_obj)) {
        return NULL;
    }
    n = PyDict_Size(self->counts);
    {
        PyObject *bk;
        PyObject *bv;
        Py_ssize_t bpos = 0;
        while (PyDict_Next(self->plain_sig, &bpos, &bk, &bv)) {
            n += PyList_GET_SIZE(bv);
        }
    }
    items = (PyObject **)PyMem_Malloc(
        (size_t)(n > 0 ? n : 1) * sizeof(PyObject *));
    if (items == NULL) {
        return PyErr_NoMemory();
    }
    cap = n;
    n = 0;
    {
        PyObject *key;
        PyObject *val;
        Py_ssize_t pos = 0;
        while (PyDict_Next(self->counts, &pos, &key, &val)) {
            long long c = PyLong_AsLongLong(PyList_GET_ITEM(val, 0));
            long long r = PyLong_AsLongLong(PyList_GET_ITEM(val, 1));
            PyObject *t = Py_BuildValue("(OLL)", key, c, r);
            if (t == NULL) {
                goto fail;
            }
            items[n++] = t;
        }
    }
    {
        PyObject *bk;
        PyObject *bv;
        Py_ssize_t bpos = 0;
        while (PyDict_Next(self->plain_sig, &bpos, &bk, &bv)) {
            Py_ssize_t j;
            for (j = 0; j < PyList_GET_SIZE(bv); ++j) {
                PyObject *rec = PyList_GET_ITEM(bv, j);
                long long c = PyLong_AsLongLong(
                    PyList_GET_ITEM(rec, 2));
                long long r = PyLong_AsLongLong(
                    PyList_GET_ITEM(rec, 3));
                PyObject *t = Py_BuildValue(
                    "(OLL)", PyList_GET_ITEM(rec, 0), c, r);
                if (t == NULL) {
                    goto fail;
                }
                items[n++] = t;
            }
        }
    }
    /* include the open pending run */
    if (self->pending != NULL) {
        PyObject *po = PyTuple_GET_ITEM(self->pending, 0);
        long long pc = PyLong_AsLong(
            PyTuple_GET_ITEM(self->pending, 1));
        int found = 0;
        Py_ssize_t j;
        for (j = 0; j < n; ++j) {
            int eq = um_deep_eq(PyTuple_GET_ITEM(items[j], 0), po);
            if (eq < 0) {
                goto fail;
            }
            if (eq) {
                PyObject *nt = Py_BuildValue(
                    "(OLL)", po,
                    PyLong_AsLongLong(PyTuple_GET_ITEM(items[j], 1))
                    + pc,
                    PyLong_AsLongLong(PyTuple_GET_ITEM(items[j], 2))
                    + 1);
                if (nt == NULL) {
                    goto fail;
                }
                Py_DECREF(items[j]);
                items[j] = nt;
                found = 1;
                break;
            }
        }
        if (!found) {
            PyObject *t = Py_BuildValue("(OLL)", po, pc, 1LL);
            if (t == NULL) {
                goto fail;
            }
            if (n >= cap) {
                cap = cap == 0 ? 8 : cap * 2;
                PyObject **ni = (PyObject **)PyMem_Realloc(
                    items, (size_t)cap * sizeof(PyObject *));
                if (ni == NULL) {
                    Py_DECREF(t);
                    PyMem_Free(items);
                    return PyErr_NoMemory();
                }
                items = ni;
            }
            items[n++] = t;
        }
    }
    qsort(items, (size_t)n, sizeof(PyObject *), um_mc_cmp);
    if (k_obj != Py_None) {
        Py_ssize_t k = PyLong_AsSsize_t(k_obj);
        if (PyErr_Occurred()) {
            goto fail;
        }
        if (k < 0) {
            k = 0;
        }
        if (k < n) {
            n = k;
        }
    }
    out = PyList_New(n);
    if (out == NULL) {
        goto fail;
    }
    for (i = 0; i < n; ++i) {
        PyList_SET_ITEM(out, i, items[i]);
    }
    PyMem_Free(items);
    return out;
fail:
    for (i = 0; i < n; ++i) {
        Py_XDECREF(items[i]);
    }
    PyMem_Free(items);
    return NULL;
}

static PyObject *um_count_vector(UnitMapObject *self, PyObject *noargs)
{
    PyObject *out;
    Py_ssize_t n = (Py_ssize_t)self->nunique;
    Py_ssize_t i;
    (void)noargs;
    out = PyList_New(n);
    if (out == NULL) {
        return NULL;
    }
    for (i = 0; i < n; ++i) {
        PyList_SET_ITEM(out, i, PyLong_FromLongLong(0));
    }
    {
        PyObject *key;
        PyObject *val;
        Py_ssize_t pos = 0;
        while (PyDict_Next(self->counts, &pos, &key, &val)) {
            PyObject *flag = PyDict_GetItemWithError(
                self->table, key);
            if (flag == NULL) {
                Py_DECREF(out);
                return NULL;
            }
            {
                long long idx = (long long)llround(
                    (PyFloat_AsDouble(flag) - self->start)
                    / self->step);
                long long c = PyLong_AsLongLong(
                    PyList_GET_ITEM(val, 0));
                PyList_SET_ITEM(out, (Py_ssize_t)idx,
                                PyLong_FromLongLong(c));
            }
        }
    }
    {
        PyObject *bk;
        PyObject *bv;
        Py_ssize_t bpos = 0;
        while (PyDict_Next(self->plain_sig, &bpos, &bk, &bv)) {
            Py_ssize_t j;
            for (j = 0; j < PyList_GET_SIZE(bv); ++j) {
                PyObject *rec = PyList_GET_ITEM(bv, j);
                long long idx = (long long)llround(
                    (PyFloat_AsDouble(PyList_GET_ITEM(rec, 1))
                     - self->start) / self->step);
                long long c = PyLong_AsLongLong(
                    PyList_GET_ITEM(rec, 2));
                PyList_SET_ITEM(out, (Py_ssize_t)idx,
                                PyLong_FromLongLong(c));
            }
        }
    }
    /* open pending run */
    if (self->pending != NULL) {
        PyObject *po = PyTuple_GET_ITEM(self->pending, 0);
        PyObject *flag = um_flag_of(self, po);
        if (flag != NULL && flag != Py_None) {
            long long idx = (long long)llround(
                (PyFloat_AsDouble(flag) - self->start) / self->step);
            long long pc = PyLong_AsLong(
                PyTuple_GET_ITEM(self->pending, 1));
            PyObject *old = PyList_GET_ITEM(out, (Py_ssize_t)idx);
            PyList_SET_ITEM(out, (Py_ssize_t)idx,
                            PyLong_FromLongLong(
                                PyLong_AsLongLong(old) + pc));
        }
        Py_XDECREF(flag);
        if (PyErr_Occurred()) {
            Py_DECREF(out);
            return NULL;
        }
    }
    return out;
}


static int um_append_run(PyObject *runs, PyObject *entry);

/* ---- ordered merge (stream concatenation, open resolution) --------- */

static PyTypeObject UnitMapType;   /* defined below the methods table */

static int um_resolve_maps(PyObject *maps, long *threshold,
                           PyObject **lens_out, PyObject **shape_out,
                           double *start_out, double *step_out)
{
    Py_ssize_t n = PyList_GET_SIZE(maps);
    PyObject *head = PyList_GET_ITEM(maps, 0);
    UnitMapObject *first = (UnitMapObject *)head;
    PyObject *lens = NULL;
    PyObject *shape = NULL;
    long thr = first->threshold;
    int any_null = (first->lens == NULL);
    Py_ssize_t i;
    if (!any_null) {
        lens = PySequence_Tuple(first->lens);
        if (lens == NULL) {
            return -1;
        }
    }
    for (i = 1; i < n; ++i) {
        UnitMapObject *m = (UnitMapObject *)PyList_GET_ITEM(maps, i);
        int same_shape;
        if (m->start != first->start || m->step != first->step) {
            PyErr_SetString(PyExc_ValueError,
                            "merge requires identical start/step");
            return -1;
        }
        if (first->shape == NULL || m->shape == NULL) {
            same_shape = (first->shape == NULL && m->shape == NULL);
        } else {
            same_shape = PyObject_RichCompareBool(first->shape, m->shape,
                                                  Py_EQ);
            if (same_shape < 0) {
                return -1;
            }
        }
        if (!same_shape) {
            PyErr_SetString(PyExc_ValueError,
                            "merge requires identical start/step/shape");
            return -1;
        }
        if (m->threshold < thr) {
            thr = m->threshold;
        }
        if (m->lens == NULL) {
            any_null = 1;
        } else if (!any_null) {
            Py_ssize_t r = PyTuple_GET_SIZE(lens);
            Py_ssize_t d;
            for (d = 0; d < r; ++d) {
                long a = PyLong_AsLong(PyTuple_GET_ITEM(lens, d));
                long b = PyLong_AsLong(PyTuple_GET_ITEM(m->lens, d));
                if (b > a) {
                    PyObject *nv = PyLong_FromLong(b);
                    PyObject *old;
                    if (nv == NULL) {
                        return -1;
                    }
                    old = PyTuple_GET_ITEM(lens, d);
                    PyTuple_SET_ITEM(lens, d, nv);
                    Py_DECREF(old);
                }
            }
        }
    }
    if (any_null) {
        Py_CLEAR(lens);
        lens = NULL;
    }
    for (i = 0; i < n; ++i) {
        UnitMapObject *m = (UnitMapObject *)PyList_GET_ITEM(maps, i);
        if (m->output_started) {
            PyErr_SetString(PyExc_ValueError,
                            "merge requires not-yet-output maps");
            Py_XDECREF(lens);
            return -1;
        }
        if (m->partial != NULL) {
            PyErr_SetString(PyExc_ValueError,
                            "merge requires maps without partial output");
            Py_XDECREF(lens);
            return -1;
        }
    }
    if (first->shape != NULL) {
        shape = first->shape;
        Py_INCREF(shape);
    }
    *threshold = thr;
    *lens_out = lens;
    *shape_out = shape;
    *start_out = first->start;
    *step_out = first->step;
    return 0;
}

static PyObject *um_merge_all_method(PyObject *self, PyObject *arg)
{
    PyObject *maps = NULL;
    PyObject *runs = NULL;
    PyObject *cargs = NULL;
    PyObject *ckw = NULL;
    PyObject *merged = NULL;
    PyObject *lens = NULL;
    PyObject *shape = NULL;
    long threshold = 1;
    double start = 0.0;
    double step = 1.0;
    Py_ssize_t i;
    (void)self;
    if (!PySequence_Check(arg) || PyUnicode_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "merge_all expects a sequence");
        return NULL;
    }
    maps = PySequence_List(arg);
    if (maps == NULL) {
        return NULL;
    }
    if (PyList_GET_SIZE(maps) == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "merge_all needs at least one map");
        Py_DECREF(maps);
        return NULL;
    }
    for (i = 0; i < PyList_GET_SIZE(maps); ++i) {
        if (!PyObject_TypeCheck(PyList_GET_ITEM(maps, i), &UnitMapType)) {
            PyErr_SetString(PyExc_TypeError,
                            "merge_all expects UnitMap maps");
            Py_DECREF(maps);
            return NULL;
        }
    }
    if (um_resolve_maps(maps, &threshold, &lens, &shape,
                        &start, &step) < 0) {
        Py_DECREF(maps);
        return NULL;
    }
    cargs = Py_BuildValue("(dd)", start, step);
    ckw = PyDict_New();
    if (cargs == NULL || ckw == NULL) {
        goto fail;
    }
    {
        PyObject *tv = PyLong_FromLong(threshold);
        int rc = tv == NULL ||
                 PyDict_SetItemString(ckw, "threshold", tv) < 0;
        Py_XDECREF(tv);
        if (rc) {
            goto fail;
        }
    }
    if (shape != NULL &&
        PyDict_SetItemString(ckw, "shape", shape) < 0) {
        goto fail;
    }
    if (lens != NULL) {
        PyObject *lv = (shape == NULL)
            ? PyLong_FromLong(PyLong_AsLong(PyTuple_GET_ITEM(lens, 0)))
            : (Py_INCREF(lens), lens);
        if (lv == NULL || PyDict_SetItemString(ckw, "length", lv) < 0) {
            Py_XDECREF(lv);
            goto fail;
        }
        Py_DECREF(lv);
    }
    merged = PyObject_Call((PyObject *)&UnitMapType, cargs, ckw);
    if (merged == NULL) {
        goto fail;
    }
    runs = PyList_New(0);
    if (runs == NULL) {
        goto fail;
    }
    for (i = 0; i < PyList_GET_SIZE(maps); ++i) {
        UnitMapObject *m = (UnitMapObject *)PyList_GET_ITEM(maps, i);
        Py_ssize_t j;
        for (j = 0; j < PyList_GET_SIZE(m->queued); ++j) {
            PyObject *pair = PyList_GET_ITEM(m->queued, j);
            PyObject *entry = Py_BuildValue(
                "[OOL]", PyTuple_GET_ITEM(pair, 0),
                PyTuple_GET_ITEM(pair, 1), 0);
            if (entry == NULL) {
                goto fail;
            }
            if (um_append_run(runs, entry) < 0) {
                Py_DECREF(entry);
                goto fail;
            }
            Py_DECREF(entry);
        }
        if (m->pending != NULL) {
            PyObject *entry = Py_BuildValue(
                "[OOL]", PyTuple_GET_ITEM(m->pending, 0),
                PyTuple_GET_ITEM(m->pending, 1), 1);
            if (entry == NULL) {
                goto fail;
            }
            if (um_append_run(runs, entry) < 0) {
                Py_DECREF(entry);
                goto fail;
            }
            Py_DECREF(entry);
        }
    }
    /* normalize: only the final run may stay open */
    for (i = 0; i + 1 < PyList_GET_SIZE(runs); ++i) {
        PyObject *entry = PyList_GET_ITEM(runs, i);
        if (PyObject_IsTrue(PyList_GET_ITEM(entry, 2))) {
            PyObject *zero = PyLong_FromLong(0);
            PyObject *old_flag;
            if (zero == NULL) {
                goto fail;
            }
            old_flag = PyList_GET_ITEM(entry, 2);
            PyList_SET_ITEM(entry, 2, zero);
            Py_DECREF(old_flag);
        }
    }
    /* register/bump in stream order */
    for (i = 0; i < PyList_GET_SIZE(runs); ++i) {
        PyObject *entry = PyList_GET_ITEM(runs, i);
        PyObject *obj = PyList_GET_ITEM(entry, 0);
        PyObject *flag = um_flag_of((UnitMapObject *)merged, obj);
        int is_open;
        if (flag == NULL) {
            goto fail;
        }
        if (flag == Py_None) {
            Py_DECREF(flag);
            if (um_register((UnitMapObject *)merged, obj) < 0) {
                goto fail;
            }
        } else {
            Py_DECREF(flag);
        }
        is_open = PyObject_IsTrue(PyList_GET_ITEM(entry, 2));
        if (is_open < 0) {
            goto fail;
        }
        if (!is_open) {
            um_bump((UnitMapObject *)merged, obj,
                    PyLong_AsLong(PyList_GET_ITEM(entry, 1)));
        }
    }
    /* pending + queued + counters */
    {
        UnitMapObject *mm = (UnitMapObject *)merged;
        PyObject *last = PyList_GET_SIZE(runs) > 0
            ? PyList_GET_ITEM(runs, PyList_GET_SIZE(runs) - 1) : NULL;
        Py_ssize_t stop = PyList_GET_SIZE(runs);
        long long uin = 0;
        long long qunits = 0;
        if (last != NULL && PyObject_IsTrue(PyList_GET_ITEM(last, 2))) {
            PyObject *pt = Py_BuildValue(
                "(OL)", PyList_GET_ITEM(last, 0),
                PyLong_AsLong(PyList_GET_ITEM(last, 1)));
            if (pt == NULL) {
                goto fail;
            }
            Py_XSETREF(mm->pending, pt);
            stop -= 1;
        }
        for (i = 0; i < stop; ++i) {
            PyObject *entry = PyList_GET_ITEM(runs, i);
            PyObject *pair = Py_BuildValue(
                "(OL)", PyList_GET_ITEM(entry, 0),
                PyLong_AsLong(PyList_GET_ITEM(entry, 1)));
            if (pair == NULL) {
                goto fail;
            }
            if (PyList_Append(mm->queued, pair) < 0) {
                Py_DECREF(pair);
                goto fail;
            }
            Py_DECREF(pair);
        }
        for (i = 0; i < PyList_GET_SIZE(maps); ++i) {
            UnitMapObject *m = (UnitMapObject *)PyList_GET_ITEM(maps, i);
            uin += m->units_in;
            qunits += m->queued_units;
        }
        mm->units_in = uin;
        mm->queued_units = qunits;
    }
    Py_DECREF(runs);
    Py_DECREF(cargs);
    Py_DECREF(ckw);
    Py_XDECREF(lens);
    Py_XDECREF(shape);
    Py_DECREF(maps);
    return merged;
fail:
    Py_XDECREF(runs);
    Py_XDECREF(cargs);
    Py_XDECREF(ckw);
    Py_XDECREF(lens);
    Py_XDECREF(shape);
    Py_XDECREF(merged);
    Py_DECREF(maps);
    return NULL;
}

/* append a run entry with seam coalescing (ordered concatenation):
 * [obj, length, is_open]; merges with the last entry when the objects
 * are deep-equal. */
static int um_append_run(PyObject *runs, PyObject *entry)
{
    Py_ssize_t n = PyList_GET_SIZE(runs);
    if (n > 0) {
        PyObject *last = PyList_GET_ITEM(runs, n - 1);
        int eq = um_deep_eq(PyList_GET_ITEM(last, 0),
                            PyList_GET_ITEM(entry, 0));
        if (eq < 0) {
            return -1;
        }
        if (eq) {
            long a = PyLong_AsLong(PyList_GET_ITEM(last, 1));
            long b = PyLong_AsLong(PyList_GET_ITEM(entry, 1));
            PyObject *sum = PyLong_FromLong(a + b);
            PyObject *open = PyList_GET_ITEM(entry, 2);
            PyObject *old_len;
            PyObject *old_open;
            if (sum == NULL) {
                return -1;
            }
            old_len = PyList_GET_ITEM(last, 1);
            old_open = PyList_GET_ITEM(last, 2);
            Py_INCREF(open);
            PyList_SET_ITEM(last, 1, sum);
            PyList_SET_ITEM(last, 2, open);
            Py_DECREF(old_len);
            Py_DECREF(old_open);
            return 0;
        }
    }
    return PyList_Append(runs, entry);
}

static PyObject *um_merge_method(UnitMapObject *self, PyObject *other)
{
    PyObject *pair = NULL;
    PyObject *result;
    if (!PyObject_TypeCheck(other, &UnitMapType)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    pair = PyList_New(2);
    if (pair == NULL) {
        return NULL;
    }
    Py_INCREF(self);
    PyList_SET_ITEM(pair, 0, (PyObject *)self);
    Py_INCREF(other);
    PyList_SET_ITEM(pair, 1, other);
    result = um_merge_all_method(NULL, pair);
    Py_DECREF(pair);
    return result;
}

static PyObject *um_add_op(PyObject *left, PyObject *right)
{
    return um_merge_method((UnitMapObject *)left, right);
}

static PyObject *um_inplace_add(PyObject *left, PyObject *right)
{
    PyObject *merged = um_merge_method((UnitMapObject *)left, right);
    UnitMapObject *a = (UnitMapObject *)left;
    UnitMapObject *b;
    if (merged == Py_NotImplemented) {
        return merged;
    }
    if (merged == NULL) {
        return NULL;
    }
    b = (UnitMapObject *)merged;
    a->start = b->start;
    a->step = b->step;
    a->threshold = b->threshold;
    Py_XINCREF(b->length);
    Py_XSETREF(a->length, b->length);
    Py_XINCREF(b->shape);
    Py_XSETREF(a->shape, b->shape);
    Py_XINCREF(b->lens);
    Py_XSETREF(a->lens, b->lens);
    Py_INCREF(b->table);
    Py_SETREF(a->table, b->table);
    Py_INCREF(b->plain_sig);
    Py_SETREF(a->plain_sig, b->plain_sig);
    Py_XINCREF(b->flags);
    Py_XSETREF(a->flags, b->flags);
    Py_INCREF(b->counts);
    Py_SETREF(a->counts, b->counts);
    Py_XINCREF(b->pending);
    Py_XSETREF(a->pending, b->pending);
    Py_INCREF(b->queued);
    Py_SETREF(a->queued, b->queued);
    Py_CLEAR(a->last_out);
    Py_CLEAR(a->partial);
    a->out_pos = 0;
    a->nunique = b->nunique;
    a->output_started = 0;
    a->units_in = b->units_in;
    a->queued_units = b->queued_units;
    Py_DECREF(merged);
    Py_INCREF(left);
    return left;
}

static PyNumberMethods um_number = {
    .nb_add = (binaryfunc)um_add_op,
    .nb_inplace_add = (binaryfunc)um_inplace_add,
};

static PyObject *um_get_start(UnitMapObject *self, void *closure)
{
    (void)closure;
    return PyFloat_FromDouble(self->start);
}

static PyObject *um_get_step(UnitMapObject *self, void *closure)
{
    (void)closure;
    return PyFloat_FromDouble(self->step);
}

static PyObject *um_get_threshold(UnitMapObject *self, void *closure)
{
    (void)closure;
    return PyLong_FromLong(self->threshold);
}

static PyObject *um_get_length(UnitMapObject *self, void *closure)
{
    (void)closure;
    if (self->length == NULL) {
        Py_RETURN_NONE;
    }
    return Py_NewRef(self->length);
}

static PyObject *um_get_shape(UnitMapObject *self, void *closure)
{
    (void)closure;
    if (self->shape == NULL) {
        Py_RETURN_NONE;
    }
    return Py_NewRef(self->shape);
}

static PyObject *um_get_units_in(UnitMapObject *self, void *closure)
{
    (void)closure;
    return PyLong_FromLongLong(self->units_in);
}

static PyObject *um_get_queued_units(UnitMapObject *self, void *closure)
{
    (void)closure;
    return PyLong_FromLongLong(self->queued_units);
}

static PyGetSetDef um_getset[] = {
    {"start", (getter)um_get_start, NULL, NULL, NULL},
    {"step", (getter)um_get_step, NULL, NULL, NULL},
    {"threshold", (getter)um_get_threshold, NULL, NULL, NULL},
    {"length", (getter)um_get_length, NULL, NULL, NULL},
    {"shape", (getter)um_get_shape, NULL, NULL, NULL},
    {"units_in", (getter)um_get_units_in, NULL, NULL, NULL},
    {"queued_units", (getter)um_get_queued_units, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyMethodDef um_methods[] = {
    {"add", (PyCFunction)um_add, METH_VARARGS,
     "add(*seq): add unit streams (any iterable, any objects)."},
    {"put", (PyCFunction)um_put, METH_VARARGS | METH_KEYWORDS,
     "put(output=None, buffering=None): write folded elements."},
    {"get_state", (PyCFunction)um_get_state, METH_NOARGS,
     "get_state(): export the mapper state."},
    {"set_state", (PyCFunction)um_set_state, METH_O,
     "set_state(state): restore an exported state."},
    {"clear", (PyCFunction)um_clear, METH_NOARGS,
     "clear(): reset table, statistics and output pointer."},
    {"flag_of", (PyCFunction)um_flag_of_method, METH_O,
     "flag_of(obj): the flag of a unit or None."},
    {"decode", (PyCFunction)um_decode, METH_O,
     "decode(flags): reverse content-table query."},
    {"total", (PyCFunction)um_total, METH_NOARGS,
     "total(): cumulative element total (dynamic)."},
    {"count", (PyCFunction)um_count, METH_O,
     "count(obj): element frequency (count) of a unit (0 unknown)."},
    {"runs", (PyCFunction)um_runs, METH_O,
     "runs(obj): run count of a unit (0 unknown)."},
    {"most_common", (PyCFunction)um_most_common,
     METH_VARARGS | METH_KEYWORDS,
     "most_common(k=None): units by count, descending."},
    {"count_vector", (PyCFunction)um_count_vector, METH_NOARGS,
     "count_vector(): per-unit counts by flag order."},
    {"merge", (PyCFunction)um_merge_method, METH_O,
     "merge(other): ordered merge returning a new map."},
    {"merge_all", (PyCFunction)um_merge_all_method,
     METH_O | METH_STATIC,
     "merge_all(maps): ordered n-ary merge of not-yet-output maps."},
    {NULL, NULL, 0, NULL},
};

static PySequenceMethods um_sequence = {
    (lenfunc)um_length,   /* sq_length */
    0,                    /* sq_concat */
    0,                    /* sq_repeat */
    0,                    /* sq_item */
    0,                    /* sq_slice */
    0,                    /* sq_ass_item */
    0,                    /* sq_ass_slice */
    (objobjproc)um_contains, /* sq_contains */
    0,                    /* sq_inplace_concat */
    0,                    /* sq_inplace_repeat */
};

static int um_tp_traverse(UnitMapObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->table);
    Py_VISIT(self->plain_sig);
    Py_VISIT(self->flags);
    Py_VISIT(self->counts);
    Py_VISIT(self->pending);
    Py_VISIT(self->queued);
    Py_VISIT(self->last_out);
    Py_VISIT(self->length);
    Py_VISIT(self->shape);
    Py_VISIT(self->lens);
    Py_VISIT(self->partial);
    return 0;
}

static int um_tp_clear(UnitMapObject *self)
{
    Py_CLEAR(self->table);
    Py_CLEAR(self->plain_sig);
    Py_CLEAR(self->flags);
    Py_CLEAR(self->counts);
    Py_CLEAR(self->pending);
    Py_CLEAR(self->queued);
    Py_CLEAR(self->last_out);
    Py_CLEAR(self->length);
    Py_CLEAR(self->shape);
    Py_CLEAR(self->lens);
    Py_CLEAR(self->partial);
    return 0;
}

static void um_dealloc(UnitMapObject *self)
{
    PyObject_GC_UnTrack(self);
    Py_XDECREF(self->table);
    Py_XDECREF(self->plain_sig);
    Py_XDECREF(self->flags);
    Py_XDECREF(self->counts);
    Py_XDECREF(self->pending);
    Py_XDECREF(self->queued);
    Py_XDECREF(self->last_out);
    Py_XDECREF(self->length);
    Py_XDECREF(self->shape);
    Py_XDECREF(self->lens);
    Py_XDECREF(self->partial);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *um_new(PyTypeObject *type, PyObject *args,
                        PyObject *kwargs)
{
    UnitMapObject *self;
    (void)args;
    (void)kwargs;
    self = (UnitMapObject *)type->tp_alloc(type, 0);
    if (self == NULL) {
        return NULL;
    }
    self->start = 0.0;
    self->step = 1.0;
    self->table = PyDict_New();
    self->plain_sig = PyDict_New();
    self->flags = NULL;
    self->counts = PyDict_New();
    self->pending = NULL;
    self->queued = PyList_New(0);
    self->last_out = NULL;
    self->out_pos = 0;
    self->nunique = 0;
    self->threshold = 1;
    self->length = NULL;
    self->shape = NULL;
    self->lens = NULL;
    self->partial = NULL;
    self->output_started = 0;
    self->units_in = 0;
    self->queued_units = 0;
    if (self->table == NULL || self->plain_sig == NULL ||
        self->counts == NULL || self->queued == NULL) {
        Py_DECREF(self);
        return NULL;
    }
    return (PyObject *)self;
}

static PyObject *um_norm_lens(PyObject *length, PyObject *shape);

static int um_init(UnitMapObject *self, PyObject *args,
                   PyObject *kwargs)
{
    static char *kwlist[] = {"start", "step", "threshold", "length",
                             "shape", NULL};
    double start = 0.0;
    double step = 1.0;
    PyObject *threshold_o = Py_None;
    PyObject *length_o = Py_None;
    PyObject *shape_o = Py_None;
    long threshold = 1;
    PyObject *shape = NULL;
    PyObject *length = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ddOOO", kwlist,
                                     &start, &step, &threshold_o,
                                     &length_o, &shape_o)) {
        return -1;
    }
    if (threshold_o != Py_None) {
        threshold = PyLong_AsLong(threshold_o);
        if (threshold == -1 && PyErr_Occurred()) {
            return -1;
        }
        if (!PyLong_Check(threshold_o) || threshold < 1) {
            PyErr_SetString(PyExc_ValueError,
                            "threshold must be an int >= 1");
            return -1;
        }
    }
    if (shape_o != Py_None) {
        Py_ssize_t r;
        shape = PySequence_Tuple(shape_o);
        if (shape == NULL) {
            return -1;
        }
        r = PyTuple_GET_SIZE(shape);
        if (r < 1) {
            PyErr_SetString(PyExc_ValueError,
                            "shape must be a tuple of positive ints");
            Py_DECREF(shape);
            return -1;
        }
        {
            Py_ssize_t d;
            for (d = 0; d < r; ++d) {
                PyObject *v = PyTuple_GET_ITEM(shape, d);
                long s = PyLong_AsLong(v);
                if (!PyLong_Check(v) || s < 1) {
                    PyErr_SetString(PyExc_ValueError,
                                    "shape must be positive ints");
                    Py_DECREF(shape);
                    return -1;
                }
            }
        }
    }
    if (length_o != Py_None) {
        if (PyLong_Check(length_o) && !PyBool_Check(length_o)) {
            long lv = PyLong_AsLong(length_o);
            if (lv < 1) {
                PyErr_SetString(PyExc_ValueError, "length must be >= 1");
                Py_XDECREF(shape);
                return -1;
            }
            if (shape != NULL) {
                Py_ssize_t r = PyTuple_GET_SIZE(shape);
                Py_ssize_t d;
                length = PyTuple_New(r);
                if (length == NULL) {
                    Py_XDECREF(shape);
                    return -1;
                }
                for (d = 0; d < r; ++d) {
                    PyObject *lv2 = PyLong_FromLong(lv);
                    if (lv2 == NULL) {
                        Py_DECREF(length);
                        Py_XDECREF(shape);
                        return -1;
                    }
                    PyTuple_SET_ITEM(length, d, lv2);
                }
            } else {
                length = length_o;
                Py_INCREF(length);
            }
        } else {
            length = PySequence_Tuple(length_o);
            if (length == NULL) {
                Py_XDECREF(shape);
                return -1;
            }
            if (shape == NULL ||
                PyTuple_GET_SIZE(length) != PyTuple_GET_SIZE(shape)) {
                PyErr_SetString(PyExc_ValueError,
                                "length tuple requires a shape of equal rank");
                Py_DECREF(length);
                Py_XDECREF(shape);
                return -1;
            }
        }
    }
    self->start = start;
    self->step = step;
    self->threshold = threshold;
    Py_XSETREF(self->shape, shape);
    Py_XSETREF(self->length, length);
    Py_CLEAR(self->lens);
    if (self->length != NULL) {
        self->lens = um_norm_lens(self->length, self->shape);
        if (self->lens == NULL) {
            return -1;
        }
    }
    return 0;
}

/* normalized per-dimension caps (tuple; scalar length -> (L,)). */
static PyObject *um_norm_lens(PyObject *length, PyObject *shape)
{
    if (shape == NULL) {
        return PyTuple_Pack(1, length);
    }
    if (PyTuple_Check(length)) {
        return PySequence_Tuple(length);
    }
    {
        Py_ssize_t r = PyTuple_GET_SIZE(shape);
        PyObject *lens = PyTuple_New(r);
        Py_ssize_t d;
        if (lens == NULL) {
            return NULL;
        }
        for (d = 0; d < r; ++d) {
            Py_INCREF(length);
            PyTuple_SET_ITEM(lens, d, length);
        }
        return lens;
    }
}

static PyTypeObject UnitMapType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "math_tool.unit_map.UnitMap",      /* tp_name */
    sizeof(UnitMapObject),             /* tp_basicsize */
    0,                                 /* tp_itemsize */
    (destructor)um_dealloc,            /* tp_dealloc */
    0,                                 /* tp_vectorcall_offset */
    0,                                 /* tp_getattr */
    0,                                 /* tp_setattr */
    0,                                 /* tp_as_async */
    0,                                 /* tp_repr */
    &um_number,                        /* tp_as_number */
    &um_sequence,                      /* tp_as_sequence */
    0,                                 /* tp_as_mapping */
    0,                                 /* tp_hash */
    0,                                 /* tp_call */
    0,                                 /* tp_str */
    0,                                 /* tp_getattro */
    0,                                 /* tp_setattro */
    0,                                 /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, /* tp_flags */
    "Run-folding unit mapper (duck units -> real flags).",
    (traverseproc)um_tp_traverse,      /* tp_traverse */
    (inquiry)um_tp_clear,              /* tp_clear */
    0,                                 /* tp_richcompare */
    0,                                 /* tp_weaklistoffset */
    0,                                 /* tp_iter */
    0,                                 /* tp_iternext */
    um_methods,                        /* tp_methods */
    0,                                 /* tp_members */
    um_getset,                         /* tp_getset */
    0,                                 /* tp_base */
    0,                                 /* tp_dict */
    0,                                 /* tp_descr_get */
    0,                                 /* tp_descr_set */
    0,                                 /* tp_dictoffset */
    (initproc)um_init,                 /* tp_init */
    0,                                 /* tp_alloc */
    um_new,                            /* tp_new */
};


/* ------------------------------------------------------------------ */
/* module-level helpers (N-D windows, iterative)                        */
/* ------------------------------------------------------------------ */

#ifndef UM_MAXDIM
#define UM_MAXDIM 16
#endif

/* Probe the leading dimensions up to ``limit`` levels (atomic-scale
 * guard: deeper nesting is unit content, not a data dimension). */
static int um_shape(PyObject *data, long *shape, int *nd, int limit)
{
    PyObject *cur = data;
    int d = 0;
    if (limit < 1) {
        limit = 1;
    }
    if (limit > UM_MAXDIM) {
        limit = UM_MAXDIM;
    }
    for (;;) {
        Py_ssize_t len;
        PyObject *first;
        if (d >= limit) {
            break;
        }
        if (!PySequence_Check(cur) || PyUnicode_Check(cur) ||
            PyBytes_Check(cur)) {
            break;
        }
        len = PySequence_Size(cur);
        if (len < 0) {
            return -1;
        }
        shape[d] = (long)len;
        ++d;
        if (len == 0) {
            break;
        }
        if (d >= limit) {
            break;
        }
        first = PySequence_GetItem(cur, 0);
        if (first == NULL) {
            return -1;
        }
        if (!PySequence_Check(first) || PyUnicode_Check(first) ||
            PyBytes_Check(first)) {
            Py_DECREF(first);
            break;
        }
        cur = first;  /* descend one level */
    }
    if (d == 0) {
        shape[0] = 1;
        d = 1;
    }
    *nd = d;
    return 0;
}

/* One leaf value at an origin (data[origin0][origin1]...). */
static PyObject *um_at(PyObject *data, const long *origin, int nd)
{
    PyObject *cur = data;
    int d;
    for (d = 0; d < nd - 1; ++d) {
        PyObject *next = PySequence_GetItem(cur, origin[d]);
        if (next == NULL) {
            return NULL;
        }
        cur = next;
    }
    return PySequence_GetItem(cur, origin[nd - 1]);
}

/* Build one window block at origin with the given size.  Leaves are
 * collected in row-major order (deepest dimension fastest) and nested
 * lists are assembled from the deepest level upwards - all iterative. */
static PyObject *um_build_block(PyObject *data, const long *origin,
                                const long *size, int nd)
{
    PyObject **rows = NULL;
    PyObject *block = NULL;
    long total = 1;
    long *pos = NULL;
    long *coord = NULL;
    long *strides = NULL;
    long i;
    int d;
    for (d = 0; d < nd; ++d) {
        total *= size[d];
    }
    rows = (PyObject **)PyMem_Malloc((size_t)total *
                                     sizeof(PyObject *));
    pos = (long *)PyMem_Malloc((size_t)nd * sizeof(long));
    coord = (long *)PyMem_Malloc((size_t)nd * sizeof(long));
    strides = (long *)PyMem_Malloc((size_t)nd * sizeof(long));
    if (rows == NULL || pos == NULL || coord == NULL ||
        strides == NULL) {
        PyMem_Free(rows);
        PyMem_Free(pos);
        PyMem_Free(coord);
        PyMem_Free(strides);
        return PyErr_NoMemory();
    }
    /* strides of the row-major leaf order */
    strides[nd - 1] = 1;
    for (d = nd - 2; d >= 0; --d) {
        strides[d] = strides[d + 1] * size[d + 1];
    }
    for (d = 0; d < nd; ++d) {
        coord[d] = 0;
        pos[d] = origin[d];
    }
    for (i = 0; i < total; ++i) {
        rows[i] = um_at(data, pos, nd);
        if (rows[i] == NULL) {
            long k;
            for (k = 0; k < i; ++k) {
                Py_DECREF(rows[k]);
            }
            PyMem_Free(rows);
            PyMem_Free(pos);
            PyMem_Free(coord);
            PyMem_Free(strides);
            return NULL;
        }
        for (d = nd - 1; d >= 0; --d) {
            coord[d] += 1;
            pos[d] += 1;
            if (coord[d] < size[d]) {
                break;
            }
            coord[d] = 0;
            pos[d] = origin[d];
        }
    }
    /* assemble: deepest slice first (each group of size[nd-1] leaves) */
    for (d = nd - 1; d >= 0; --d) {
        long groups = total / (strides[d] * size[d]);
        long g;
        if (d < nd - 1) {
            /* previous pass stored lists at the group starts */
            strides[d] = 1;  /* recomputed below */
        }
        for (g = 0; g < groups; ++g) {
            PyObject *lst = PyList_New((Py_ssize_t)size[d]);
            long base = g * strides[d] * size[d];
            long k;
            if (lst == NULL) {
                long z;
                for (z = 0; z < g * size[d] + (g ? 0 : size[d]); ++z) {
                    Py_XDECREF(z < (g * strides[d] * size[d])
                               ? rows[z] : NULL);
                }
                PyMem_Free(rows);
                PyMem_Free(pos);
                PyMem_Free(coord);
                PyMem_Free(strides);
                return NULL;
            }
            for (k = 0; k < size[d]; ++k) {
                PyList_SET_ITEM(lst, k, rows[base + k]);
            }
            rows[g * strides[d]] = lst;
        }
    }
    block = rows[0];
    Py_INCREF(block);
    PyMem_Free(rows);
    PyMem_Free(pos);
    PyMem_Free(coord);
    PyMem_Free(strides);
    return block;
}

static int um_read_region(PyObject *obj, long *vals, int nd,
                          const char *name, int scalar_ok)
{
    int d;
    if (obj == Py_None) {
        for (d = 0; d < nd; ++d) {
            vals[d] = scalar_ok ? 1 : 0;
        }
        return 0;
    }
    {
        PyObject *idx = PyNumber_Index(obj);
        if (idx != NULL) {
            long v = PyLong_AsLong(idx);
            Py_DECREF(idx);
            if (v == -1 && PyErr_Occurred()) {
                return -1;
            }
            for (d = 0; d < nd; ++d) {
                vals[d] = v;
            }
            return 0;
        }
        PyErr_Clear();
    }
    {
        Py_ssize_t n = PySequence_Size(obj);
        Py_ssize_t i;
        if (n < 0) {
            PyErr_Clear();
            PyErr_Format(PyExc_TypeError,
                         "%s must be an int or a sequence", name);
            return -1;
        }
        if (n != nd) {
            PyErr_Format(PyExc_ValueError,
                         "%s length does not match the dimension",
                         name);
            return -1;
        }
        for (i = 0; i < n; ++i) {
            PyObject *item = PySequence_GetItem(obj, i);
            long v;
            if (item == NULL) {
                return -1;
            }
            v = PyLong_AsLong(item);
            Py_DECREF(item);
            if (v == -1 && PyErr_Occurred()) {
                return -1;
            }
            vals[i] = v;
        }
    }
    return 0;
}

/* core window walk (shared by the module functions); probe_limit is the
 * dimension probe depth (atomic-scale guard, default 1). */
static PyObject *um_windows_core(PyObject *data, PyObject *local_size,
                                 PyObject *step, PyObject *start,
                                 PyObject *shape_o, long *wshape_out,
                                 int *nd_out, int probe_limit)
{
    long dshape[UM_MAXDIM];
    long lsize[UM_MAXDIM];
    long lstep[UM_MAXDIM];
    long lstart[UM_MAXDIM];
    long wshape[UM_MAXDIM];
    int nd = 1;
    int d;
    long count = 1;
    PyObject *out;
    long *pos;
    if (um_shape(data, dshape, &nd, probe_limit) < 0) {
        return NULL;
    }
    if (um_read_region(local_size, lsize, nd, "local_size", 1) < 0 ||
        um_read_region(step, lstep, nd, "step", 1) < 0 ||
        um_read_region(start, lstart, nd, "start", 0) < 0) {
        return NULL;
    }
    if (step == Py_None) {
        for (d = 0; d < nd; ++d) {
            lstep[d] = lsize[d];
        }
    }
    if (shape_o != Py_None) {
        if (um_read_region(shape_o, wshape, nd, "shape", 0) < 0) {
            return NULL;
        }
    } else {
        for (d = 0; d < nd; ++d) {
            long span = dshape[d] - lstart[d] - lsize[d];
            wshape[d] = span < 0 ? 0 : span / lstep[d] + 1;
        }
    }
    for (d = 0; d < nd; ++d) {
        count *= wshape[d];
    }
    out = PyList_New(count);
    pos = (long *)PyMem_Malloc((size_t)nd * sizeof(long));
    if (out == NULL || pos == NULL) {
        Py_XDECREF(out);
        PyMem_Free(pos);
        return PyErr_NoMemory();
    }
    for (d = 0; d < nd; ++d) {
        pos[d] = 0;
    }
    {
        long i;
        for (i = 0; i < count; ++i) {
            long origin[UM_MAXDIM];
            PyObject *block;
            for (d = 0; d < nd; ++d) {
                origin[d] = lstart[d] + pos[d] * lstep[d];
            }
            block = um_build_block(data, origin, lsize, nd);
            if (block == NULL) {
                Py_DECREF(out);
                PyMem_Free(pos);
                return NULL;
            }
            PyList_SET_ITEM(out, i, block);
            for (d = nd - 1; d >= 0; --d) {
                pos[d] += 1;
                if (pos[d] < wshape[d]) {
                    break;
                }
                pos[d] = 0;
            }
        }
    }
    PyMem_Free(pos);
    if (wshape_out != NULL && nd_out != NULL) {
        for (d = 0; d < nd; ++d) {
            wshape_out[d] = wshape[d];
        }
        *nd_out = nd;
    }
    return out;
}

static PyObject *um_window_units(PyObject *self, PyObject *args,
                                 PyObject *kwargs)
{
    static char *kwlist[] = {"data", "local_size", "step", "start",
                             "shape", "probe_dim", NULL};
    PyObject *data;
    PyObject *local_size = Py_None;
    PyObject *step = Py_None;
    PyObject *start = Py_None;
    PyObject *shape_o = Py_None;
    PyObject *probe_dim = Py_None;
    int limit;
    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOOOO", kwlist,
                                     &data, &local_size, &step, &start,
                                     &shape_o, &probe_dim)) {
        return NULL;
    }
    if (probe_dim == Py_None) {
        limit = 1;
    } else {
        limit = (int)PyLong_AsLong(probe_dim);
        if (limit == -1 && PyErr_Occurred()) {
            return NULL;
        }
    }
    return um_windows_core(data, local_size, step, start, shape_o,
                           NULL, NULL, limit);
}

/* um_map_data: stream the N-D windows one block at a time into the
 * mapper (peak memory O(1) - no window list is materialized).  A native
 * UnitMap mapper is fed through um_push directly; any other mapper falls
 * back to a per-block add() call. */
/* Build the default UnitMap for map_data: threshold / length pass
 * through; a tuple length gets the window-grid shape so per-dimension
 * caps line up with the unit grid. */
static PyObject *um_make_mapper(PyObject *threshold_o, PyObject *length_o,
                                const long *wshape, int nd)
{
    PyObject *cargs = PyTuple_New(0);
    PyObject *ckw = PyDict_New();
    PyObject *tv = NULL;
    PyObject *sh = NULL;
    PyObject *mapper;
    int d;
    if (cargs == NULL || ckw == NULL) {
        Py_XDECREF(cargs);
        Py_XDECREF(ckw);
        return NULL;
    }
    tv = (threshold_o == Py_None) ? PyLong_FromLong(1) : threshold_o;
    if (tv == NULL || PyDict_SetItemString(ckw, "threshold", tv) < 0) {
        if (threshold_o == Py_None) {
            Py_XDECREF(tv);
        }
        Py_DECREF(cargs);
        Py_DECREF(ckw);
        return NULL;
    }
    if (threshold_o == Py_None) {
        Py_DECREF(tv);
    }
    if (length_o != Py_None &&
        PyDict_SetItemString(ckw, "length", length_o) < 0) {
        Py_DECREF(cargs);
        Py_DECREF(ckw);
        return NULL;
    }
    if (PyTuple_Check(length_o)) {
        sh = PyTuple_New(nd);
        if (sh == NULL) {
            Py_DECREF(cargs);
            Py_DECREF(ckw);
            return NULL;
        }
        for (d = 0; d < nd; ++d) {
            PyObject *wv = PyLong_FromLong(wshape[d]);
            if (wv == NULL) {
                Py_DECREF(sh);
                Py_DECREF(cargs);
                Py_DECREF(ckw);
                return NULL;
            }
            PyTuple_SET_ITEM(sh, d, wv);
        }
        if (PyDict_SetItemString(ckw, "shape", sh) < 0) {
            Py_DECREF(sh);
            Py_DECREF(cargs);
            Py_DECREF(ckw);
            return NULL;
        }
        Py_DECREF(sh);
    }
    mapper = PyObject_Call((PyObject *)&UnitMapType, cargs, ckw);
    Py_DECREF(cargs);
    Py_DECREF(ckw);
    return mapper;
}

static PyObject *um_map_data(PyObject *self, PyObject *args,
                             PyObject *kwargs)
{
    static char *kwlist[] = {"data", "local_size", "step", "mapper",
                             "output", "buffering", "start", "shape",
                             "probe_dim", "threshold", "length", NULL};
    PyObject *data;
    PyObject *local_size = Py_None;
    PyObject *step = Py_None;
    PyObject *mapper = Py_None;
    PyObject *output = Py_None;
    PyObject *buffering = Py_None;
    PyObject *start = Py_None;
    PyObject *shape_o = Py_None;
    PyObject *probe_dim = Py_None;
    PyObject *threshold_o = Py_None;
    PyObject *length_o = Py_None;
    long dshape[UM_MAXDIM];
    long lsize[UM_MAXDIM];
    long lstep[UM_MAXDIM];
    long lstart[UM_MAXDIM];
    long wshape[UM_MAXDIM];
    int nd = 1;
    int d;
    int limit;
    long count = 1;
    int native;
    PyObject *res;
    long *pos;
    long i;
    (void)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOOOOOOOOO", kwlist,
                                     &data, &local_size, &step, &mapper,
                                     &output, &buffering, &start,
                                     &shape_o, &probe_dim,
                                     &threshold_o, &length_o)) {
        return NULL;
    }
    if (probe_dim == Py_None) {
        limit = 1;
    } else {
        limit = (int)PyLong_AsLong(probe_dim);
        if (limit == -1 && PyErr_Occurred()) {
            return NULL;
        }
    }
    if (um_shape(data, dshape, &nd, limit) < 0) {
        return NULL;
    }
    if (um_read_region(local_size, lsize, nd, "local_size", 1) < 0 ||
        um_read_region(step, lstep, nd, "step", 1) < 0 ||
        um_read_region(start, lstart, nd, "start", 0) < 0) {
        return NULL;
    }
    if (step == Py_None) {
        for (d = 0; d < nd; ++d) {
            lstep[d] = lsize[d];
        }
    }
    if (shape_o != Py_None) {
        if (um_read_region(shape_o, wshape, nd, "shape", 0) < 0) {
            return NULL;
        }
    } else {
        for (d = 0; d < nd; ++d) {
            long span = dshape[d] - lstart[d] - lsize[d];
            wshape[d] = span < 0 ? 0 : span / lstep[d] + 1;
        }
    }
    for (d = 0; d < nd; ++d) {
        count *= wshape[d];
    }
    if (mapper == Py_None) {
        mapper = um_make_mapper(threshold_o, length_o, wshape, nd);
        if (mapper == NULL) {
            return NULL;
        }
    } else {
        Py_INCREF(mapper);
    }
    native = PyObject_TypeCheck(mapper, &UnitMapType);
    pos = (long *)PyMem_Malloc((size_t)nd * sizeof(long));
    if (pos == NULL) {
        Py_DECREF(mapper);
        return PyErr_NoMemory();
    }
    for (d = 0; d < nd; ++d) {
        pos[d] = 0;
    }
    for (i = 0; i < count; ++i) {
        long origin[UM_MAXDIM];
        PyObject *block;
        int fed;
        for (d = 0; d < nd; ++d) {
            origin[d] = lstart[d] + pos[d] * lstep[d];
        }
        block = um_build_block(data, origin, lsize, nd);
        if (block == NULL) {
            PyMem_Free(pos);
            Py_DECREF(mapper);
            return NULL;
        }
        if (native) {
            fed = um_push((UnitMapObject *)mapper, block);
        } else {
            res = PyObject_CallMethod(mapper, "add", "(O)", block);
            fed = (res != NULL) ? 0 : -1;
            Py_XDECREF(res);
        }
        Py_DECREF(block);
        if (fed < 0) {
            PyMem_Free(pos);
            Py_DECREF(mapper);
            return NULL;
        }
        for (d = nd - 1; d >= 0; --d) {
            pos[d] += 1;
            if (pos[d] < wshape[d]) {
                break;
            }
            pos[d] = 0;
        }
    }
    PyMem_Free(pos);
    /* mapper.put(output=..., buffering=...) */
    {
        PyObject *kw = PyDict_New();
        PyObject *call;
        if (kw == NULL) {
            Py_DECREF(mapper);
            return NULL;
        }
        if (output != Py_None) {
            PyDict_SetItemString(kw, "output", output);
        }
        if (buffering != Py_None) {
            PyDict_SetItemString(kw, "buffering", buffering);
        }
        call = PyObject_CallMethod(mapper, "put", NULL, kw);
        Py_DECREF(kw);
        Py_DECREF(mapper);
        if (call == NULL) {
            return NULL;
        }
        return call;
    }
}


/* ------------------------------------------------------------------ */
/* module definition                                                    */
/* ------------------------------------------------------------------ */

static PyMethodDef module_methods[] = {
    {"window_units", (PyCFunction)um_window_units,
     METH_VARARGS | METH_KEYWORDS,
     "window_units(data, local_size, step=None, start=None, "
     "shape=None): N-D windows as a list of nested sub-blocks."},
    {"map_data", (PyCFunction)um_map_data,
     METH_VARARGS | METH_KEYWORDS,
     "map_data(data, local_size, step=None, mapper=None, "
     "output=None, buffering=None, start=None, shape=None): N-D "
     "convenience over UnitMap."},
    {NULL, NULL, 0, NULL},
};

static int module_exec(PyObject *m) {
    if (PyType_Ready(&UnitMapType) < 0) {
        return -1;
    }
    Py_INCREF(&UnitMapType);
    if (PyModule_AddObject(m, "UnitMap", (PyObject *)&UnitMapType) < 0) {
        Py_DECREF(&UnitMapType);
        return -1;
    }
    return 0;
}

static PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, (void*)module_exec},
#if defined(Py_mod_gil) && defined(Py_MOD_GIL_NOT_USED)
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_unit_map",
    "Run-folding unit mapper (duck units -> real flags).",
    0,
    module_methods,
    module_slots,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit__unit_map(void);
PyMODINIT_FUNC PyInit__unit_map(void)
{
    return PyModuleDef_Init(&moduledef);
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
