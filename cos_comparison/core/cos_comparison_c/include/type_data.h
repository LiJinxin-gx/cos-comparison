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

#endif /* TYPE_DATA_H */
