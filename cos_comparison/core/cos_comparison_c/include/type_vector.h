#ifndef TYPE_VECTOR_H
#define TYPE_VECTOR_H

#include "type_data.h"

typedef struct VectorMap {
	Data    *base;
	int     *shape;
	int      dimension;
	int      start;
	int      end;
	int      p;
	int      stride;
	int     *strides;
} VectorMap;

/* Basic functions */
COS_API VectorMap* VectorMap_new(Data *base, int dimension, const int shape[],
					 int start, int end, int p, int stride);
COS_API void VectorMap_free(VectorMap *self);
COS_API double VectorMap_get(const VectorMap *self, const int index[]);
COS_API void   VectorMap_set(VectorMap *self, const int index[], double value);
COS_API int    VectorMap_len(const VectorMap *self);
COS_API double VectorMap_mean(const VectorMap *self);
COS_API double VectorMap_variance(const VectorMap *self);
COS_API VectorMap* VectorMap_subview(const VectorMap *self, int index);

/* New slice operations (top‑level dimension only) */
COS_API VectorMap* VectorMap_slice(const VectorMap *self, int start, int stop);
COS_API void VectorMap_slice_set(VectorMap *self, int start, int stop, const VectorMap *value);
COS_API void VectorMap_slice_set_scalar(VectorMap *self, int start, int stop, double value);

#endif /* TYPE_VECTOR_H */
