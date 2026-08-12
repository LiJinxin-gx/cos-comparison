#ifndef TYPE_DATA_H
#define TYPE_DATA_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

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
COS_API int   Data_offset(const Data *self, const int index[]);
COS_API double Data_get_flat(const Data *self, int idx);
COS_API void   Data_set_flat(Data *self, int idx, double value);
COS_API double Data_get(const Data *self, const int index[]);
COS_API void   Data_set(Data *self, const int index[], double value);
COS_API int   Data_total(const Data *self);
COS_API int   Data_total_elements(const Data *self);
COS_API int   Data_shape_equal(const Data *a, const Data *b);

#endif /* TYPE_DATA_H */
