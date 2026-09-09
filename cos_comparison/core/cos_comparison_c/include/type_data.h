#ifndef TYPE_DATA_H
#define TYPE_DATA_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

/* The algorithms use `int` for shapes, strides and element counts.
   ISO C only guarantees int >= 16 bits; fail at compile time on
   targets where int is too narrow instead of silently misbehaving.
   Guarded so a translation unit mixing both type_data.h variants
   still defines the typedef exactly once. */
#ifndef COS_STATIC_ASSERT_INT32
#define COS_STATIC_ASSERT_INT32
typedef char cos_static_assert_int_at_least_32[(INT_MAX >= 2147483647) ? 1 : -1];
#endif

/* The flat iterator counts elements in long long; ISO C only guarantees
   long long >= 64 bits, but any narrower target cannot hold the counts. */
#ifndef COS_STATIC_ASSERT_LL64
#define COS_STATIC_ASSERT_LL64
typedef char cos_static_assert_ll64[(sizeof(long long) >= 8) ? 1 : -1];
#endif

/* C99 7.12 defines NAN only when the implementation supports quiet NaNs;
   provide a portable fallback for the rest. */
#ifdef NAN
#define COS_NAN NAN
#else
#define COS_NAN sqrt(-1.0)
#endif

#ifdef _WIN32
#ifdef COS_BUILD_DLL
#define COS_API __declspec(dllexport)
#else
#define COS_API __declspec(dllimport)
#endif
#else
#define COS_API
#endif

typedef struct Data {
	int     dimension;
	int    *shape;
	int    *strides;
	void   *data;
	int     owns_data;
	int     dtype;      /* 0 = double, 1 = unsigned char */
} Data;

/* Function declarations (implemented in core.c) */
COS_API Data* Data_create(int dimension, const int shape[]);
COS_API void  Data_free(Data *self);
COS_API double Data_get(const Data *self, const int index[]);
COS_API void   Data_set(Data *self, const int index[], double value);
COS_API int   Data_total(const Data *self);
COS_API int   Data_total_elements(const Data *self);
COS_API int   Data_shape_equal(const Data *a, const Data *b);

/* ---------- Pure C99 flat iterator (simulated OO via self parameter) ----------
   The iterator owns its view arrays; it is driven by self-parameter methods
   (CosFlatIterator_init / _next / _free) that the Python layer copies onto
   its wrapping class, so the C core never needs the Python C API.
   Element counts and flat indices use long long (the C99 int -> long ->
   long long type chain) so arithmetic keeps working as far as possible. */
typedef struct CosFlatIterator {
    int        dimension;
    long long  total;        /* element count (0 = empty, 1 = scalar) */
    long long  pos;          /* next element position (0-based) */
    long long  current;      /* flat index at the current position */
    long long  base;         /* global flat base offset (start + offset) */
    int       *shape;        /* owned copy of per-dimension sizes */
    long long *strides;      /* owned copy of view strides */
    int       *start_offset; /* owned copy of per-dimension start offsets */
    int       *step_offset;  /* owned copy of per-dimension step offsets */
    int       *idx;          /* per-dimension counters */
} CosFlatIterator;

COS_API int CosFlatIterator_init(CosFlatIterator *self,
    int dimension, const int *shape, const long long *strides,
    const int *start_offset, const int *step_offset, long long base);
/* advance to the next element: 1 = ok (self->current updated),
   0 = exhausted, -1 = error */
COS_API int CosFlatIterator_next(CosFlatIterator *self);
COS_API void CosFlatIterator_free(CosFlatIterator *self);

#endif /* TYPE_DATA_H */
