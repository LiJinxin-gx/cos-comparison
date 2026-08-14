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

typedef struct Data {
	int     dimension;
	int    *shape;
	int    *strides;
	void   *data;
	int     owns_data;
	int     dtype;      /* 0 = double, 1 = unsigned char */
} Data;

static inline int Data_offset(const Data *self, const int index[]) {
	int offset = 0;
	for (int i = 0; i < self->dimension; ++i)
		offset += index[i] * self->strides[i];
	return offset;
}

static inline double Data_get_flat(const Data *self, int idx) {
	if (self->dtype == 1)
		return (double)((unsigned char*)self->data)[idx];
	return ((double*)self->data)[idx];
}

static inline void Data_set_flat(Data *self, int idx, double value) {
	if (self->dtype == 1)
		((unsigned char*)self->data)[idx] = (unsigned char)value;
	else
		((double*)self->data)[idx] = value;
}

static inline double Data_get(const Data *self, const int index[]) {
	return Data_get_flat(self, Data_offset(self, index));
}

static inline void Data_set(Data *self, const int index[], double value) {
	Data_set_flat(self, Data_offset(self, index), value);
}

static inline int Data_total(const Data *self) {
	int total = 1;
	for (int i = 0; i < self->dimension; ++i) total *= self->shape[i];
	return total;
}

/* --- Add missing functions here --- */
static inline int Data_shape_equal(const Data *a, const Data *b) {
	if (a->dimension != b->dimension) return 0;
	for (int i = 0; i < a->dimension; ++i)
		if (a->shape[i] != b->shape[i]) return 0;
	return 1;
}
/* --------------------------------- */

static inline Data* Data_create(int dimension, const int shape[]) {
	Data *self = (Data*)calloc((size_t)(1), sizeof(Data));
	if (!self) return NULL;
	/* Guard against degenerate dimensions: allocate at least 1 element
	   so malloc/calloc never returns NULL due to a zero size. */
	int dim = dimension > 0 ? dimension : 1;
	self->dimension = dimension;
	self->shape = (int*)calloc((size_t)(dim), sizeof(int));
	if (!self->shape) { free(self); return NULL; }
	if (dimension > 0) memcpy(self->shape, shape, (size_t)(dimension) * sizeof(int));
	self->strides = (int*)calloc((size_t)(dim), sizeof(int));
	if (!self->strides) { free(self->shape); free(self); return NULL; }
	long long stride = 1;
	for (int i = dimension - 1; i >= 0; --i) {
		self->strides[i] = (int)stride;
		long long next = stride * (long long)shape[i];
		/* Signed int overflow is undefined behaviour (C99 6.5p5); reject
		   shapes whose strides/total would not fit an int. Degenerate
		   (zero/negative) shapes are preserved for backend parity. */
		if (next > INT_MAX || next < INT_MIN) {
			free(self->strides);
			free(self->shape);
			free(self);
			return NULL;
		}
		stride = next;
	}
	int total = (int)stride;
	/* Allocate at least 1 element to avoid zero-size allocation (C99:
	   malloc(0) may return NULL, which would be indistinguishable from
	   an out-of-memory error).  A zero-element tensor never dereferences it. */
	size_t alloc_count = total > 0 ? (size_t)total : 1;
	self->data = (double*)calloc(alloc_count, sizeof(double));
	if (!self->data) { free(self->strides); free(self->shape); free(self); return NULL; }
	self->owns_data = 1;
	self->dtype = 0;
	return self;
}

static inline void Data_free(Data *self) {
	if (!self) return;
	free(self->shape);
	free(self->strides);
	if (self->owns_data && self->data)
		free(self->data);
	free(self);
}

#endif
