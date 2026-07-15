#ifndef TYPE_DATA_H
#define TYPE_DATA_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

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

static inline int Data_total_elements(const Data *self) {
	return Data_total(self);
}
/* --------------------------------- */

static inline Data* Data_create(int dimension, const int shape[]) {
	Data *self = (Data*)calloc(1, sizeof(Data));
	if (!self) return NULL;
	self->dimension = dimension;
	self->shape = (int*)malloc(dimension * sizeof(int));
	if (!self->shape) { free(self); return NULL; }
	memcpy(self->shape, shape, dimension * sizeof(int));
	self->strides = (int*)malloc(dimension * sizeof(int));
	if (!self->strides) { free(self->shape); free(self); return NULL; }
	int stride = 1;
	for (int i = dimension - 1; i >= 0; --i) {
		self->strides[i] = stride;
		stride *= shape[i];
	}
	int total = stride;
	self->data = (double*)calloc(total, sizeof(double));
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
