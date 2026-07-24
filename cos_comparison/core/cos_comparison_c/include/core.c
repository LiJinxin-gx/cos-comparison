#define COS_BUILD_DLL
#include "type.h"
#include <math.h>
#include <stdlib.h>

#ifdef _WIN32
#define COS_API __declspec(dllexport)
#else
#define COS_API
#endif

/* ---------- Data functions ---------- */
COS_API Data* Data_create(int dimension, const int shape[]) {
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
    self->dtype = 0; /* default double */
    return self;
}

COS_API void Data_free(Data *self) {
    if (!self) return;
    free(self->shape);
    free(self->strides);
    if (self->owns_data && self->data)
        free(self->data);
    free(self);
}

COS_API int Data_offset(const Data *self, const int index[]) {
    int offset = 0;
    for (int i = 0; i < self->dimension; ++i)
        offset += index[i] * self->strides[i];
    return offset;
}

COS_API double Data_get_flat(const Data *self, int idx) {
    if (self->dtype == 1)
        return (double)((unsigned char*)self->data)[idx];
    return ((double*)self->data)[idx];
}

COS_API void Data_set_flat(Data *self, int idx, double value) {
    if (self->dtype == 1)
        ((unsigned char*)self->data)[idx] = (unsigned char)value;
    else
        ((double*)self->data)[idx] = value;
}

COS_API double Data_get(const Data *self, const int index[]) {
    return Data_get_flat(self, Data_offset(self, index));
}

COS_API void Data_set(Data *self, const int index[], double value) {
    Data_set_flat(self, Data_offset(self, index), value);
}

COS_API int Data_total(const Data *self) {
    int total = 1;
    for (int i = 0; i < self->dimension; ++i) total *= self->shape[i];
    return total;
}

COS_API int Data_total_elements(const Data *self) {
    return Data_total(self);
}

COS_API int Data_shape_equal(const Data *a, const Data *b) {
    if (a->dimension != b->dimension) return 0;
    for (int i = 0; i < a->dimension; ++i)
        if (a->shape[i] != b->shape[i]) return 0;
    return 1;
}

/* ---------- VectorMap functions ---------- */
COS_API VectorMap* VectorMap_new(Data *base, int dimension, const int shape[],
                                 int start, int end, int p, int stride) {
    if (!base || dimension <= 0) return NULL;
    VectorMap *self = (VectorMap*)malloc(sizeof(VectorMap));
    if (!self) return NULL;
    self->base = base;
    self->dimension = dimension;
    self->shape = (int*)malloc(dimension * sizeof(int));
    if (!self->shape) { free(self); return NULL; }
    memcpy(self->shape, shape, dimension * sizeof(int));
    if (end < 0) {
        int total = 1;
        for (int i = 0; i < base->dimension; ++i) total *= base->shape[i];
        self->end = total;
    } else {
        self->end = end;
    }
    self->start = start;
    self->p = p;
    self->stride = stride;
    self->strides = (int*)malloc(dimension * sizeof(int));
    if (!self->strides) { free(self->shape); free(self); return NULL; }
    int s = 1;
    for (int i = dimension - 1; i >= 0; --i) {
        self->strides[i] = s;
        s *= shape[i];
    }
    return self;
}

COS_API void VectorMap_free(VectorMap *self) {
    if (!self) return;
    free(self->shape);
    free(self->strides);
    free(self);
}

COS_API double VectorMap_get(const VectorMap *self, const int index[]) {
    if (!self || !index) return 0.0;
    int offset = self->start;
    for (int i = 0; i < self->dimension; ++i) {
        offset += index[i] * self->strides[i];
    }
    if (offset >= self->end) return 0.0;
    return Data_get_flat(self->base, offset);
}

COS_API void VectorMap_set(VectorMap *self, const int index[], double value) {
    if (!self || !index) return;
    int offset = self->start;
    for (int i = 0; i < self->dimension; ++i) {
        offset += index[i] * self->strides[i];
    }
    if (offset < self->end) {
        Data_set_flat(self->base, offset, value);
    }
}

COS_API int VectorMap_len(const VectorMap *self) {
    if (!self) return 0;
    return self->shape[self->p];
}

COS_API double VectorMap_mean(const VectorMap *self) {
    if (!self) return 0.0;
    int total = 1;
    for (int i = 0; i < self->dimension; ++i) total *= self->shape[i];
    if (total == 0) return 0.0;
    double sum = 0.0;
    int *indices = (int*)calloc(self->dimension, sizeof(int));
    if (!indices) return 0.0;
    for (int idx = 0; idx < total; ++idx) {
        int offset = self->start;
        for (int i = 0; i < self->dimension; ++i)
            offset += indices[i] * self->strides[i];
        sum += Data_get_flat(self->base, offset);
        int carry = 1;
        for (int i = self->dimension - 1; i >= 0 && carry; --i) {
            indices[i] += carry;
            if (indices[i] >= self->shape[i]) {
                indices[i] = 0;
                carry = 1;
            } else {
                carry = 0;
            }
        }
    }
    free(indices);
    return sum / (double)total;
}

COS_API double VectorMap_variance(const VectorMap *self) {
    if (!self) return 0.0;
    int total = 1;
    for (int i = 0; i < self->dimension; ++i) total *= self->shape[i];
    if (total == 0) return 0.0;
    double sum = 0.0;
    double sum_sq = 0.0;
    int *indices = (int*)calloc(self->dimension, sizeof(int));
    if (!indices) return 0.0;
    for (int idx = 0; idx < total; ++idx) {
        int offset = self->start;
        for (int i = 0; i < self->dimension; ++i)
            offset += indices[i] * self->strides[i];
        double val = Data_get_flat(self->base, offset);
        sum += val;
        sum_sq += val * val;
        int carry = 1;
        for (int i = self->dimension - 1; i >= 0 && carry; --i) {
            indices[i] += carry;
            if (indices[i] >= self->shape[i]) {
                indices[i] = 0;
                carry = 1;
            } else {
                carry = 0;
            }
        }
    }
    free(indices);
    double mean = sum / (double)total;
    return sum_sq / (double)total - mean * mean;
}

COS_API VectorMap* VectorMap_subview(const VectorMap *self, int index) {
    if (!self || index < 0 || index >= self->shape[self->p]) return NULL;
    int new_start = self->start + index * self->strides[self->p];
    int new_end = new_start + self->strides[self->p]; // length of one element along this dim
    // Actually, we need to adjust end based on remaining dimensions.
    // For a subview, we move one level deeper.
    int new_p = self->p + 1;
    int new_stride = (new_p < self->dimension) ? self->strides[new_p] : 0;
    VectorMap *sub = VectorMap_new(self->base, self->dimension, self->shape,
                                   new_start, new_end, new_p, new_stride);
    return sub;
}

/* ---------- NEW: Slice operations (top-level dimension only) ---------- */
COS_API VectorMap* VectorMap_slice(const VectorMap *self, int start, int stop) {
    if (!self) return NULL;
    int length = self->shape[self->p];
    if (start < 0) start += length;
    if (stop < 0) stop += length;
    if (start < 0) start = 0;
    if (stop > length) stop = length;
    if (start >= stop) {
        // Empty slice: return a view with start==end (zero length)
        VectorMap *empty = VectorMap_new(self->base, self->dimension, self->shape,
                                         self->start + start * self->strides[self->p],
                                         self->start + start * self->strides[self->p],
                                         self->p, self->stride);
        return empty;
    }
    int new_start = self->start + start * self->strides[self->p];
    int new_end = self->start + stop * self->strides[self->p];
    VectorMap *slice = VectorMap_new(self->base, self->dimension, self->shape,
                                     new_start, new_end, self->p, self->stride);
    return slice;
}

COS_API void VectorMap_slice_set(VectorMap *self, int start, int stop, const VectorMap *value) {
    if (!self || !value) return;
    int length = self->shape[self->p];
    if (start < 0) start += length;
    if (stop < 0) stop += length;
    if (start < 0) start = 0;
    if (stop > length) stop = length;
    if (start >= stop) return;
    int count = stop - start;
    if (value->end - value->start != count) {
        // Length mismatch
        return;
    }
    for (int i = 0; i < count; ++i) {
        int idx = self->start + (start + i) * self->strides[self->p];
        double val = Data_get_flat(value->base, value->start + i);
        Data_set_flat(self->base, idx, val);
    }
}

COS_API void VectorMap_slice_set_scalar(VectorMap *self, int start, int stop, double value) {
    if (!self) return;
    int length = self->shape[self->p];
    if (start < 0) start += length;
    if (stop < 0) stop += length;
    if (start < 0) start = 0;
    if (stop > length) stop = length;
    if (start >= stop) return;
    for (int i = 0; i < stop - start; ++i) {
        int idx = self->start + (start + i) * self->strides[self->p];
        Data_set_flat(self->base, idx, value);
    }
}

/* ---------- Core comparison algorithms ---------- */

typedef struct {
    double w1, b1;
    double w2, b2;
} linear;

typedef struct {
    linear li;
    int *start;
    int *end;
    int *step;
    int *d;
} control;

typedef void (*error_callback)(void* e, void* name);
typedef void* (*return_callback)(void* output, void* name);

typedef struct {
    void (*start)(void*);
    void (*end)(void*);
    void (*iter)(void*);
    error_callback local_error;
    error_callback global_error;
    return_callback return_cb;
} callback;

static inline double _cos_(double a, double b, double ab) {
    double c = a * b;
    if (c) return ab / sqrt(c);
    return (a == b) ? 1.0 : 0.0;
}
static inline double _mod_(double a, double b, double ab) {
    double c = a * b;
    if (c) return 2.0 * sqrt(c) / (a + b);
    return (a == b) ? 1.0 : 0.0;
}
static inline double _cosmod_(double a, double b, double ab) {
    double c = a * b;
    if (c) return 2.0 * ab / (a + b);
    return (a == b) ? 1.0 : 0.0;
}

static void no_done(void *unused) { (void)unused; }
static void no_error(void *e, void *name) { (void)e; (void)name; }
static void* no_return(void *output, void *name) { (void)name; return output; }

static control create_default_control(int dimension) {
    int *start = (int*)malloc(sizeof(*start) * dimension);
    int *end   = (int*)malloc(sizeof(*end) * dimension);
    int *step  = (int*)malloc(sizeof(*step) * dimension);
    int *d     = (int*)malloc(sizeof(*d) * dimension);
    if (!start || !end || !step || !d) {
        free(start); free(end); free(step); free(d);
        return (control){{1,0,1,0}, NULL, NULL, NULL, NULL};
    }
    for (int i = 0; i < dimension; i++) {
        start[i] = 0;
        end[i]   = -1;
        step[i]  = 1;
        d[i]     = 0;
    }
    d[0] = 1;
    linear li = {1, 0, 1, 0};
    return (control){li, start, end, step, d};
}

static int* _compute_num(const int *shape, int dim,
                         const int *start, const int *end,
                         const int *step, const int *window_size,
                         const int *d) {
    int *num = (int*)malloc(dim * sizeof(int));
    if (!num) return NULL;
    for (int i = 0; i < dim; ++i) {
        int e = (end && end[i] >= 0) ? end[i] : shape[i];
        int effective = e - start[i] - window_size[i] - d[i];
        if (effective < 0) {
            free(num);
            return NULL;
        }
        num[i] = effective / step[i] + 1;
    }
    return num;
}

static Data* _compute_output_shape(int dim, const int num[],
                                   const int output_start[], const int output_step[]) {
    int *out_shape = (int*)malloc(dim * sizeof(int));
    for (int i = 0; i < dim; ++i)
        out_shape[i] = (num[i] - 1) * output_step[i] + 1;
    Data *out = Data_create(dim, out_shape);
    free(out_shape);
    return out;
}

/* Passive mode */
COS_API Data* cos_comparison_passive(const Data *data,
                                     const int *window_size,
                                     const control *ctrl,
                                     const callback *cb,
                                     double (*sim_func)(double,double,double),
                                     Data *out_param,
                                     const int *output_start,
                                     const int *output_step,
                                     void *ctx) {
    if (!data || !window_size) return NULL;
    int dim = data->dimension;
    control default_ctrl;
    const control *c;
    if (ctrl) {
        c = ctrl;
    } else {
        default_ctrl = create_default_control(dim);
        c = &default_ctrl;
    }
    int *start = c->start ? c->start : (int*)calloc(dim, sizeof(int));
    int *end   = c->end   ? c->end   : (int*)malloc(dim * sizeof(int));
    int *step  = c->step  ? c->step  : (int*)malloc(dim * sizeof(int));
    int *d     = c->d     ? c->d     : (int*)calloc(dim, sizeof(int));
    if (!c->end)   { for (int i=0; i<dim; ++i) end[i] = -1; }
    if (!c->step)  { for (int i=0; i<dim; ++i) step[i] = 1; }
    if (!c->d)     { d[0] = 1; for (int i=1; i<dim; ++i) d[i] = 0; }
    linear lin = c->li;

    int *num = _compute_num(data->shape, dim, start, end, step, window_size, d);
    if (!num) {
        if (!ctrl) { free(start); free(end); free(step); free(d); }
        if (cb && cb->global_error) cb->global_error(NULL, ctx);
        return NULL;
    }

    int *os = (int*)malloc(dim * sizeof(int));
    int *ost = (int*)malloc(dim * sizeof(int));
    for (int i = 0; i < dim; ++i) {
        os[i] = output_start ? output_start[i] : 0;
        ost[i] = output_step ? output_step[i] : 1;
    }

    Data *out;
    if (out_param) {
        out = out_param;
    } else {
        out = _compute_output_shape(dim, num, os, ost);
        if (!out) {
            free(num); free(os); free(ost);
            if (!ctrl) { free(start); free(end); free(step); free(d); }
            if (cb && cb->global_error) cb->global_error(NULL, ctx);
            return NULL;
        }
    }

    int *num_list = (int*)malloc((dim + 1) * sizeof(int));
    int *inner_list = (int*)malloc((dim + 1) * sizeof(int));
    int *main_place = (int*)malloc(dim * sizeof(int));
    int *other_place = (int*)malloc(dim * sizeof(int));
    int *out_idx = (int*)malloc(dim * sizeof(int));
    for (int i = 0; i <= dim; ++i) { num_list[i] = 1; inner_list[i] = 1; }

    int flag = dim;
    double main_sum, other_sum, mu_sum;

    if (cb && cb->start) cb->start(ctx);

    while (flag > 0) {
        if (flag == dim) {
            for (int i = 1; i <= dim; ++i) inner_list[i] = 1;
            int inner_flag = dim;
            main_sum = 0.0;
            other_sum = 0.0;
            mu_sum = 0.0;

            while (inner_flag > 0) {
                if (inner_flag == dim) {
                    for (int i = 0; i < dim; ++i) {
                        main_place[i] = start[i] + step[i] * (num_list[i+1] - 1) + (inner_list[i+1] - 1);
                        other_place[i] = main_place[i] + d[i];
                    }
                    double a = lin.w1 * Data_get(data, main_place) + lin.b1;
                    double b = lin.w2 * Data_get(data, other_place) + lin.b2;
                    main_sum += a * a;
                    other_sum += b * b;
                    mu_sum += a * b;

                    if (inner_list[dim] < window_size[dim - 1]) {
                        inner_list[dim]++;
                    } else {
                        inner_list[dim] = 1;
                        inner_flag--;
                    }
                } else {
                    if (inner_list[inner_flag] < window_size[inner_flag - 1]) {
                        inner_list[inner_flag]++;
                        inner_flag = dim;
                    } else {
                        inner_list[inner_flag] = 1;
                        inner_flag--;
                    }
                }
            }

            for (int i = 0; i < dim; ++i)
                out_idx[i] = os[i] + ost[i] * (num_list[i+1] - 1);

            double res;
            if (sim_func) {
                res = sim_func(main_sum, other_sum, mu_sum);
            } else {
                res = _cosmod_(main_sum, other_sum, mu_sum);
            }

            Data_set(out, out_idx, res);
            if (cb && cb->iter) cb->iter(ctx);

            if (num_list[dim] < num[dim - 1]) {
                num_list[dim]++;
            } else {
                num_list[dim] = 1;
                flag--;
            }
        } else {
            if (num_list[flag] < num[flag - 1]) {
                num_list[flag]++;
                flag = dim;
            } else {
                num_list[flag] = 1;
                flag--;
            }
        }
    }

    free(num); free(os); free(ost);
    free(num_list); free(inner_list);
    free(main_place); free(other_place); free(out_idx);
    if (!ctrl) { free(start); free(end); free(step); free(d); }

    if (cb && cb->end) cb->end(ctx);
    if (cb && cb->return_cb) out = cb->return_cb(out, ctx);
    return out;
}

/* Active mode */
COS_API Data* cos_comparison_active(const Data *data,
                                    const Data *kernel,
                                    const control *ctrl,
                                    const callback *cb,
                                    double (*sim_func)(double,double,double),
                                    Data *out_param,
                                    const int *output_start,
                                    const int *output_step,
                                    void *ctx) {
    if (!data || !kernel || data->dimension != kernel->dimension) {
        if (cb && cb->global_error) cb->global_error(NULL, ctx);
        return NULL;
    }
    int dim = data->dimension;
    const int *window_size = kernel->shape;

    control default_ctrl;
    const control *c;
    if (ctrl) {
        c = ctrl;
    } else {
        default_ctrl = create_default_control(dim);
        c = &default_ctrl;
    }
    int *start = c->start ? c->start : (int*)calloc(dim, sizeof(int));
    int *end   = c->end   ? c->end   : (int*)malloc(dim * sizeof(int));
    int *step  = c->step  ? c->step  : (int*)malloc(dim * sizeof(int));
    int *d_dummy = (int*)calloc(dim, sizeof(int));
    if (!c->end)   { for (int i=0; i<dim; ++i) end[i] = -1; }
    if (!c->step)  { for (int i=0; i<dim; ++i) step[i] = 1; }
    linear lin = c->li;

    int *num = _compute_num(data->shape, dim, start, end, step, window_size, d_dummy);
    if (!num) {
        free(d_dummy);
        if (!ctrl) { free(start); free(end); free(step); }
        if (cb && cb->global_error) cb->global_error(NULL, ctx);
        return NULL;
    }

    int *os = (int*)malloc(dim * sizeof(int));
    int *ost = (int*)malloc(dim * sizeof(int));
    for (int i = 0; i < dim; ++i) {
        os[i] = output_start ? output_start[i] : 0;
        ost[i] = output_step ? output_step[i] : 1;
    }

    Data *out;
    if (out_param) {
        out = out_param;
    } else {
        out = _compute_output_shape(dim, num, os, ost);
        if (!out) {
            free(num); free(os); free(ost); free(d_dummy);
            if (!ctrl) { free(start); free(end); free(step); }
            if (cb && cb->global_error) cb->global_error(NULL, ctx);
            return NULL;
        }
    }

    int *num_list = (int*)malloc((dim + 1) * sizeof(int));
    int *inner_list = (int*)malloc((dim + 1) * sizeof(int));
    int *data_place = (int*)malloc(dim * sizeof(int));
    int *kern_place = (int*)malloc(dim * sizeof(int));
    int *out_idx = (int*)malloc(dim * sizeof(int));
    for (int i = 0; i <= dim; ++i) { num_list[i] = 1; inner_list[i] = 1; }

    int flag = dim;
    double main_sum, other_sum, mu_sum;

    if (cb && cb->start) cb->start(ctx);

    while (flag > 0) {
        if (flag == dim) {
            for (int i = 1; i <= dim; ++i) inner_list[i] = 1;
            int inner_flag = dim;
            main_sum = 0.0;
            other_sum = 0.0;
            mu_sum = 0.0;

            while (inner_flag > 0) {
                if (inner_flag == dim) {
                    for (int i = 0; i < dim; ++i) {
                        data_place[i] = start[i] + step[i] * (num_list[i+1] - 1) + (inner_list[i+1] - 1);
                        kern_place[i] = inner_list[i+1] - 1;
                    }
                    double a = lin.w1 * Data_get(data, data_place) + lin.b1;
                    double b = lin.w2 * Data_get(kernel, kern_place) + lin.b2;
                    main_sum += a * a;
                    other_sum += b * b;
                    mu_sum += a * b;

                    if (inner_list[dim] < window_size[dim - 1]) {
                        inner_list[dim]++;
                    } else {
                        inner_list[dim] = 1;
                        inner_flag--;
                    }
                } else {
                    if (inner_list[inner_flag] < window_size[inner_flag - 1]) {
                        inner_list[inner_flag]++;
                        inner_flag = dim;
                    } else {
                        inner_list[inner_flag] = 1;
                        inner_flag--;
                    }
                }
            }

            for (int i = 0; i < dim; ++i)
                out_idx[i] = os[i] + ost[i] * (num_list[i+1] - 1);

            double res;
            if (sim_func) {
                res = sim_func(main_sum, other_sum, mu_sum);
            } else {
                res = _cosmod_(main_sum, other_sum, mu_sum);
            }

            Data_set(out, out_idx, res);
            if (cb && cb->iter) cb->iter(ctx);

            if (num_list[dim] < num[dim - 1]) {
                num_list[dim]++;
            } else {
                num_list[dim] = 1;
                flag--;
            }
        } else {
            if (num_list[flag] < num[flag - 1]) {
                num_list[flag]++;
                flag = dim;
            } else {
                num_list[flag] = 1;
                flag--;
            }
        }
    }

    free(num); free(os); free(ost);
    free(num_list); free(inner_list);
    free(data_place); free(kern_place); free(out_idx);
    free(d_dummy);
    if (!ctrl) { free(start); free(end); free(step); }

    if (cb && cb->end) cb->end(ctx);
    if (cb && cb->return_cb) out = cb->return_cb(out, ctx);
    return out;
}

/* Full tensor similarity */
COS_API double cos_full(const Data *a, const Data *b,
                        double (*sim_func)(double,double,double)) {
    if (!Data_shape_equal(a, b)) {
        return NAN; /* Shape mismatch, return NaN intentionally */
    }
    int total = Data_total(a);
    double sum_a = 0.0, sum_b = 0.0, sum_ab = 0.0;
    for (int i = 0; i < total; ++i) {
        double va = Data_get_flat(a, i);
        double vb = Data_get_flat(b, i);
        sum_a += va * va;
        sum_b += vb * vb;
        sum_ab += va * vb;
    }
    return sim_func ? sim_func(sum_a, sum_b, sum_ab)
           : _cosmod_(sum_a, sum_b, sum_ab);
}

/* Local mean */
COS_API Data* cos_local_mean(const Data *data,
                             const int window_size[],
                             const control *ctrl,
                             const callback *cb,
                             Data *out_param,
                             const int output_start[],
                             const int output_step[],
                             void *ctx) {
    if (!data || !window_size) return NULL;
    int dim = data->dimension;

    control default_ctrl;
    const control *c;
    if (ctrl) {
        c = ctrl;
    } else {
        default_ctrl = create_default_control(dim);
        c = &default_ctrl;
    }

    int *start = c->start ? c->start : (int*)calloc(dim, sizeof(int));
    int *end   = c->end   ? c->end   : (int*)malloc(dim * sizeof(int));
    int *step  = c->step  ? c->step  : (int*)malloc(dim * sizeof(int));
    int *d_dummy = (int*)calloc(dim, sizeof(int));
    if (!c->end)   { for (int i=0; i<dim; ++i) end[i] = -1; }
    if (!c->step)  { for (int i=0; i<dim; ++i) step[i] = 1; }

    int *num = _compute_num(data->shape, dim, start, end, step, window_size, d_dummy);
    if (!num) {
        free(d_dummy);
        if (!ctrl) { free(start); free(end); free(step); }
        if (cb && cb->global_error) cb->global_error(NULL, ctx);
        return NULL;
    }

    int *os = (int*)malloc(dim * sizeof(int));
    int *ost = (int*)malloc(dim * sizeof(int));
    for (int i = 0; i < dim; ++i) {
        os[i] = output_start ? output_start[i] : 0;
        ost[i] = output_step ? output_step[i] : 1;
    }

    Data *out;
    if (out_param) {
        out = out_param;
    } else {
        out = _compute_output_shape(dim, num, os, ost);
        if (!out) {
            free(num); free(os); free(ost); free(d_dummy);
            if (!ctrl) { free(start); free(end); free(step); }
            if (cb && cb->global_error) cb->global_error(NULL, ctx);
            return NULL;
        }
    }

    int N = 1;
    for (int i = 0; i < dim; ++i) N *= window_size[i];

    int *num_list = (int*)malloc((dim + 1) * sizeof(int));
    int *inner_list = (int*)malloc((dim + 1) * sizeof(int));
    int *data_place = (int*)malloc(dim * sizeof(int));
    int *out_idx = (int*)malloc(dim * sizeof(int));
    for (int i = 0; i <= dim; ++i) { num_list[i] = 1; inner_list[i] = 1; }

    int flag = dim;
    double sum_x;

    if (cb && cb->start) cb->start(ctx);

    while (flag > 0) {
        if (flag == dim) {
            for (int i = 1; i <= dim; ++i) inner_list[i] = 1;
            int inner_flag = dim;
            sum_x = 0.0;

            while (inner_flag > 0) {
                if (inner_flag == dim) {
                    for (int i = 0; i < dim; ++i) {
                        data_place[i] = start[i] + step[i] * (num_list[i+1] - 1) + (inner_list[i+1] - 1);
                    }
                    double val = Data_get(data, data_place);
                    sum_x += val;

                    if (inner_list[dim] < window_size[dim - 1]) {
                        inner_list[dim]++;
                    } else {
                        inner_list[dim] = 1;
                        inner_flag--;
                    }
                } else {
                    if (inner_list[inner_flag] < window_size[inner_flag - 1]) {
                        inner_list[inner_flag]++;
                        inner_flag = dim;
                    } else {
                        inner_list[inner_flag] = 1;
                        inner_flag--;
                    }
                }
            }

            for (int i = 0; i < dim; ++i)
                out_idx[i] = os[i] + ost[i] * (num_list[i+1] - 1);
            double mean = sum_x / (double)N;
            Data_set(out, out_idx, mean);
            if (cb && cb->iter) cb->iter(ctx);

            if (num_list[dim] < num[dim - 1]) {
                num_list[dim]++;
            } else {
                num_list[dim] = 1;
                flag--;
            }
        } else {
            if (num_list[flag] < num[flag - 1]) {
                num_list[flag]++;
                flag = dim;
            } else {
                num_list[flag] = 1;
                flag--;
            }
        }
    }

    free(num); free(os); free(ost);
    free(num_list); free(inner_list); free(data_place); free(out_idx);
    free(d_dummy);
    if (!ctrl) { free(start); free(end); free(step); }

    if (cb && cb->end) cb->end(ctx);
    if (cb && cb->return_cb) out = cb->return_cb(out, ctx);
    return out;
}

/* Local variance */
COS_API Data* cos_local_variance(const Data *data,
                                 const int window_size[],
                                 const control *ctrl,
                                 const callback *cb,
                                 Data *out_param,
                                 const int output_start[],
                                 const int output_step[],
                                 void *ctx) {
    if (!data || !window_size) return NULL;
    int dim = data->dimension;

    control default_ctrl;
    const control *c;
    if (ctrl) {
        c = ctrl;
    } else {
        default_ctrl = create_default_control(dim);
        c = &default_ctrl;
    }

    int *start = c->start ? c->start : (int*)calloc(dim, sizeof(int));
    int *end   = c->end   ? c->end   : (int*)malloc(dim * sizeof(int));
    int *step  = c->step  ? c->step  : (int*)malloc(dim * sizeof(int));
    int *d_dummy = (int*)calloc(dim, sizeof(int));
    if (!c->end)   { for (int i=0; i<dim; ++i) end[i] = -1; }
    if (!c->step)  { for (int i=0; i<dim; ++i) step[i] = 1; }

    int *num = _compute_num(data->shape, dim, start, end, step, window_size, d_dummy);
    if (!num) {
        free(d_dummy);
        if (!ctrl) { free(start); free(end); free(step); }
        if (cb && cb->global_error) cb->global_error(NULL, ctx);
        return NULL;
    }

    int *os = (int*)malloc(dim * sizeof(int));
    int *ost = (int*)malloc(dim * sizeof(int));
    for (int i = 0; i < dim; ++i) {
        os[i] = output_start ? output_start[i] : 0;
        ost[i] = output_step ? output_step[i] : 1;
    }

    Data *out;
    if (out_param) {
        out = out_param;
    } else {
        out = _compute_output_shape(dim, num, os, ost);
        if (!out) {
            free(num); free(os); free(ost); free(d_dummy);
            if (!ctrl) { free(start); free(end); free(step); }
            if (cb && cb->global_error) cb->global_error(NULL, ctx);
            return NULL;
        }
    }

    int N = 1;
    for (int i = 0; i < dim; ++i) N *= window_size[i];

    int *num_list = (int*)malloc((dim + 1) * sizeof(int));
    int *inner_list = (int*)malloc((dim + 1) * sizeof(int));
    int *data_place = (int*)malloc(dim * sizeof(int));
    int *out_idx = (int*)malloc(dim * sizeof(int));
    for (int i = 0; i <= dim; ++i) { num_list[i] = 1; inner_list[i] = 1; }

    int flag = dim;
    double sum_x, sum_x2;

    if (cb && cb->start) cb->start(ctx);

    while (flag > 0) {
        if (flag == dim) {
            for (int i = 1; i <= dim; ++i) inner_list[i] = 1;
            int inner_flag = dim;
            sum_x = 0.0;
            sum_x2 = 0.0;

            while (inner_flag > 0) {
                if (inner_flag == dim) {
                    for (int i = 0; i < dim; ++i) {
                        data_place[i] = start[i] + step[i] * (num_list[i+1] - 1) + (inner_list[i+1] - 1);
                    }
                    double val = Data_get(data, data_place);
                    sum_x += val;
                    sum_x2 += val * val;

                    if (inner_list[dim] < window_size[dim - 1]) {
                        inner_list[dim]++;
                    } else {
                        inner_list[dim] = 1;
                        inner_flag--;
                    }
                } else {
                    if (inner_list[inner_flag] < window_size[inner_flag - 1]) {
                        inner_list[inner_flag]++;
                        inner_flag = dim;
                    } else {
                        inner_list[inner_flag] = 1;
                        inner_flag--;
                    }
                }
            }

            for (int i = 0; i < dim; ++i)
                out_idx[i] = os[i] + ost[i] * (num_list[i+1] - 1);
            double mean = sum_x / (double)N;
            double variance = sum_x2 / (double)N - mean * mean;
            Data_set(out, out_idx, variance);
            if (cb && cb->iter) cb->iter(ctx);

            if (num_list[dim] < num[dim - 1]) {
                num_list[dim]++;
            } else {
                num_list[dim] = 1;
                flag--;
            }
        } else {
            if (num_list[flag] < num[flag - 1]) {
                num_list[flag]++;
                flag = dim;
            } else {
                num_list[flag] = 1;
                flag--;
            }
        }
    }

    free(num); free(os); free(ost);
    free(num_list); free(inner_list); free(data_place); free(out_idx);
    free(d_dummy);
    if (!ctrl) { free(start); free(end); free(step); }

    if (cb && cb->end) cb->end(ctx);
    if (cb && cb->return_cb) out = cb->return_cb(out, ctx);
    return out;
}