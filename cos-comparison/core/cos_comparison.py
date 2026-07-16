# it provides basic functions of cos_comparsion module, an AI algorithm set
#
# core : information is produced by local comparison in raw data

#--------------------- import ------------------------------
from math import sqrt

#-------------------- constant -----------------------------
NaN = float("nan")

#------------------ tool module -----------------       
def multiple_chain(iterable, base=1):
    """
    Use base to multiple element of iterable object in turn.
    """
    for m in iterable:
        base = base * m
    return base

def add_chain(iterable, base=0):
    """
    Use base to add element of iterable object in turn.
    """
    for m in iterable:
        base = base + m
    return base

def vector_chain_compute(A):
    a=A
    def compute(vector):
        nonlocal a
        leng=len(a)
        return (sum((m*n for m,n in zip(vector,a[i]))) for i in range(leng))
    def fix(new):
        nonlocal a
        a=new
    def get():
        return a
    return compute,fix,get

def create_void_list(length_list=(1,), default=None):
    """
    create a default void list.
    """
    length_list = tuple(length_list)
    return vector_map_as_tensor(
        [default for _ in range(multiple_chain(length_list))],
        length_list
        )

def load_as_default_data(data, start=None, shape=None):
    """
    Load data as a default data type.
    start: tuple of int, start coordinates in each dimension
    shape: tuple of int, size of the region to load in each dimension
    Supports loading sub-regions from multi-dimensional data, hides underlying type details
    """
    # First infer full shape of input data
    temp = data
    full_shape = []
    while True:
        try:
            full_shape.append(len(temp))
            temp = temp[0]
        except (TypeError, IndexError, AttributeError):
            break
    if not full_shape:
        raise ValueError("data is empty or scalar; cannot infer shape")
    dimension = len(full_shape)
    full_shape = tuple(full_shape)

    # Default shape is full shape if not provided
    if shape is None:
        shape = full_shape
    else:
        shape = tuple(shape)
        if len(shape) != dimension:
            raise ValueError(f"shape length {len(shape)} does not match data dimension {dimension}")
        for i in range(dimension):
            if shape[i] < 0 or shape[i] > full_shape[i]:
                raise ValueError(f"shape[{i}] = {shape[i]} out of bounds for dimension size {full_shape[i]}")

    # Default start is all zeros if not provided
    if start is None:
        start = tuple(0 for _ in range(dimension))
    else:
        start = tuple(start)
        if len(start) != dimension:
            raise ValueError(f"start length {len(start)} does not match data dimension {dimension}")
        for i in range(dimension):
            if start[i] < 0 or start[i] + shape[i] > full_shape[i]:
                raise ValueError(f"start[{i}] = {start[i]} out of bounds for dimension size {full_shape[i]} with shape {shape[i]}")

    # Calculate strides for each dimension (to convert multi-dim index to 1d offset)
    strides = [1] * dimension
    for i in range(dimension-2, -1, -1):
        strides[i] = strides[i+1] * full_shape[i+1]

    # Calculate 1d start offset in the full data
    start_1d = 0
    for i in range(dimension):
        start_1d += start[i] * strides[i]

    # Flatten the requested region using carry mechanism
    total_elements = 1
    for s in shape:
        total_elements *= s
    vector = [0.0] * total_elements

    num_list = [None] + [1] * dimension
    flag = dimension
    pos = 0
    while flag:
        if flag == dimension:
            # Calculate multi-dim index relative to region start
            idx = tuple(start[i] + num_list[i+1] - 1 for i in range(dimension))
            vector[pos] = get_item(data, idx)
            pos += 1
        if num_list[flag] < shape[flag - 1]:
            num_list[flag] += 1
            flag = dimension
        else:
            num_list[flag] = 1
            flag -= 1

    end = start_1d + total_elements
    # Create tensor: note that the underlying vector is a copy of the region,
    # so start is 0 for the new tensor's own vector, shape is the requested region shape
    return vector_map_as_tensor(vector, shape, start=0, end=total_elements, p=0, cache=None)

def get_item(object, index):
    if hasattr(object,"__get_item__"):
        return object.__get_item__(*index) #allow to reload
    temp = object
    for p in index:
        temp = temp[p]
    return temp

def no_done(*arg,**kwarg):
    pass

#------------------- class support --------------------------
#     --------------- type support -----------------
class vector_map_as_tensor:
    __slots__ = ("vector", "tensor_size", "dimension", "start", "end", "p", "cache")
    def __init__(self, vector, tensor_size, start=0, end=None, p=0, cache=None):
        self.vector = vector
        self.tensor_size = tensor_size
        self.start = start
        self.end = len(vector) if end is None else end
        self.p = p
        self.cache = multiple_chain(tensor_size[p+1:]) if cache is None else cache
    def __repr__(self):
        return f"<vector_map_as_tensor: dim={len(self.tensor_size)-self.p}, start={self.start}, end={self.end}, p={self.p}, cache={self.cache}>"

    def __getitem__(self, index):
        # Handle tuple of integers (already supported)
        if isinstance(index, tuple) and all(isinstance(i, int) for i in index):
            return self.__get_item__(*index)

        # Handle single slice (only top-level dimension)
        if isinstance(index, slice):
            # Normalize start and stop
            length = self.tensor_size[self.p]
            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else length
            step = index.step if index.step is not None else 1
            if start < 0:
                start += length
            if stop < 0:
                stop += length
            start = max(0, min(start, length))
            stop = max(0, min(stop, length))
            if step != 1:
                raise ValueError("Non-unit step slicing not supported")
            if start >= stop:
                # Empty slice: return a view with zero length? 
                # We'll return a view with start=start, end=start (empty)
                new_start = self.start + start * self.cache
                new_end = new_start  # zero length
                return self.__class__(self.vector, self.tensor_size,
                                      start=new_start, end=new_end,
                                      p=self.p, cache=self.cache)
            new_start = self.start + start * self.cache
            new_end = self.start + stop * self.cache
            return self.__class__(self.vector, self.tensor_size,
                                  start=new_start, end=new_end,
                                  p=self.p, cache=self.cache)

        # Handle single integer
        if isinstance(index, int):
            if index < 0:
                index += self.tensor_size[self.p]
            if index < 0 or index >= self.tensor_size[self.p]:
                raise IndexError("Index out of range")
            new_start = self.start + index * self.cache
            if self.p < len(self.tensor_size) - 1:
                new_p = self.p + 1
                new_cache = self.cache // self.tensor_size[new_p]
                return self.__class__(
                    self.vector, self.tensor_size,
                    start=new_start, end=new_start + self.cache,
                    p=new_p, cache=new_cache
                )
            else:
                return self.vector[new_start]

        raise TypeError("Index must be int, slice, or tuple of ints")

    def __setitem__(self, key, value):
        # Handle tuple of integers (recursive)
        if isinstance(key, tuple) and all(isinstance(i, int) for i in key):
            obj = self
            for i in key[:-1]:
                obj = obj[i]
            obj[key[-1]] = value
            return

        # Handle single slice (only allowed at leaf dimension)
        if isinstance(key, slice):
            if self.p != len(self.tensor_size) - 1:
                raise IndexError("Slice assignment only allowed at the leaf dimension")
            length = self.tensor_size[self.p]
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else length
            step = key.step if key.step is not None else 1
            if start < 0:
                start += length
            if stop < 0:
                stop += length
            start = max(0, min(start, length))
            stop = max(0, min(stop, length))
            if step != 1:
                raise ValueError("Non-unit step slicing not supported")
            if start >= stop:
                return  # assign nothing
            # Convert value to a list if it's a vector_map_as_tensor
            if isinstance(value, vector_map_as_tensor):
                # Value must have the same length as the slice
                if value.end - value.start != stop - start:
                    raise ValueError("Length of value does not match slice length")
                for i in range(stop - start):
                    self.vector[self.start + (start + i) * self.cache] = value.vector[value.start + i]
            elif isinstance(value, (list, tuple)):
                if len(value) != stop - start:
                    raise ValueError("Length of value does not match slice length")
                for i, v in enumerate(value):
                    self.vector[self.start + (start + i) * self.cache] = v
            else:
                # Scalar: assign to all elements
                for i in range(stop - start):
                    self.vector[self.start + (start + i) * self.cache] = value
            return

        # Handle single integer
        if isinstance(key, int):
            if key < 0:
                key += self.tensor_size[self.p]
            if key < 0 or key >= self.tensor_size[self.p]:
                raise IndexError("Index out of range")
            if self.p < len(self.tensor_size) - 1:
                # Not leaf: navigate down
                child = self[key]
                child[key] = value  # This will call __setitem__ on child
                return
            else:
                self.vector[self.start + key] = value
                return

        raise TypeError("Key must be int, slice, or tuple of ints")

    def __get_item__(self, *indexs):
        p = self.p
        remaining = len(self.tensor_size) - p
        if len(indexs) == remaining:
            ptr = self.start
            cache = self.cache
            for i, idx in enumerate(indexs):
                ptr += idx * cache
                if i < remaining - 1:
                    cache //= self.tensor_size[p + i]
            return self.vector[ptr]
        elif 0 < len(indexs) < remaining:
            start_ptr = self.start
            cache = self.cache
            for i, idx in enumerate(indexs):
                start_ptr += idx * cache
                cache //= self.tensor_size[p + i]
            new_p = p + len(indexs)
            new_cache = cache
            return self.__class__(self.vector, self.tensor_size,
                                  start=start_ptr,
                                  end=start_ptr + new_cache,
                                  p=new_p,
                                  cache=new_cache)
        elif len(indexs) == 0:
            return self
        else:
            raise IndexError("It was given some effectless index.")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return self.tensor_size[self.p]

    def _check_shape(self, other):
        if type(other) != type(self):
            raise TypeError("It can not compute with other type.")
        if self.tensor_size != other.tensor_size:
            raise ValueError("the shape of two tensors are not same.")
        if self.p != other.p:
            raise ValueError("the current depths are not same.")

    # ---------- binary arithmetic ----------
    def __add__(self, other):
        self._check_shape(other)
        total = multiple_chain(self.tensor_size)
        new_vec = [0.0] * total
        for i in range(total):
            new_vec[i] = self.vector[self.start + i] + other.vector[other.start + i]
        return self.__class__(new_vec, self.tensor_size, start=0, end=total, p=0, cache=None)

    def __sub__(self, other):
        self._check_shape(other)
        total = multiple_chain(self.tensor_size)
        new_vec = [0.0] * total
        for i in range(total):
            new_vec[i] = self.vector[self.start + i] - other.vector[other.start + i]
        return self.__class__(new_vec, self.tensor_size, start=0, end=total, p=0, cache=None)

    def __mul__(self, other):
        if type(other) == type(self):
            self._check_shape(other)
            total = multiple_chain(self.tensor_size)
            new_vec = [0.0] * total
            for i in range(total):
                new_vec[i] = self.vector[self.start + i] * other.vector[other.start + i]
            return self.__class__(new_vec, self.tensor_size, start=0, end=total, p=0, cache=None)
        elif type(other) in (int, float):
            total = multiple_chain(self.tensor_size)
            new_vec = [0.0] * total
            for i in range(total):
                new_vec[i] = self.vector[self.start + i] * other
            return self.__class__(new_vec, self.tensor_size, start=0, end=total, p=0, cache=None)
        else:
            raise TypeError("It can not compute with other type.")

    def __truediv__(self, other):
        if type(other) == type(self):
            self._check_shape(other)
            total = multiple_chain(self.tensor_size)
            new_vec = [0.0] * total
            for i in range(total):
                new_vec[i] = self.vector[self.start + i] / other.vector[other.start + i]
            return self.__class__(new_vec, self.tensor_size, start=0, end=total, p=0, cache=None)
        elif type(other) in (int, float):
            if other == 0:
                raise ZeroDivisionError("division by zero")
            total = multiple_chain(self.tensor_size)
            new_vec = [0.0] * total
            for i in range(total):
                new_vec[i] = self.vector[self.start + i] / other
            return self.__class__(new_vec, self.tensor_size, start=0, end=total, p=0, cache=None)
        else:
            raise TypeError("It can not be used with other type.")

    # ---------- in-place arithmetic ----------
    def __iadd__(self, other):
        self._check_shape(other)
        for i in range(self.end - self.start):
            self.vector[self.start + i] += other.vector[other.start + i]
        return self

    def __isub__(self, other):
        self._check_shape(other)
        for i in range(self.end - self.start):
            self.vector[self.start + i] -= other.vector[other.start + i]
        return self

    def __imul__(self, other):
        if type(other) == type(self):
            self._check_shape(other)
            for i in range(self.end - self.start):
                self.vector[self.start + i] *= other.vector[other.start + i]
        elif type(other) in (int, float):
            for i in range(self.end - self.start):
                self.vector[self.start + i] *= other
        else:
            raise TypeError("It can not be used with other type.")
        return self

    def __itruediv__(self, other):
        if type(other) == type(self):
            self._check_shape(other)
            for i in range(self.end - self.start):
                self.vector[self.start + i] /= other.vector[other.start + i]
        elif type(other) in (int, float):
            if other == 0:
                raise ZeroDivisionError("division by zero")
            for i in range(self.end - self.start):
                self.vector[self.start + i] /= other
        else:
            raise TypeError("It can not be used with other type.")
        return self

    # ---------- unary arithmetic ----------
    def __neg__(self):
        total = multiple_chain(self.tensor_size)
        new_vec = [0.0] * total
        for i in range(total):
            new_vec[i] = -self.vector[self.start + i]
        return self.__class__(new_vec, self.tensor_size, start=0, end=total, p=0, cache=None)

    def __pos__(self):
        total = multiple_chain(self.tensor_size)
        new_vec = [0.0] * total
        for i in range(total):
            new_vec[i] = self.vector[self.start + i]
        return self.__class__(new_vec, self.tensor_size, start=0, end=total, p=0, cache=None)

    def __abs__(self):
        square_sum = sum(i * i for i in self.vector[self.start:self.end])
        return sqrt(square_sum)

    def mean(self):
        return sum(self.vector[self.start:self.end]) / (self.end - self.start)

    def variance(self):
        mean = self.mean()
        square_mean = sum(i * i for i in self.vector[self.start:self.end]) / (self.end - self.start)
        return square_mean - mean * mean 

#    ---------containers----------
class func_name_space:
    __slots__ = ("output", "output_start", "output_step", "window_size", "kernel",
                 "linear", "start", "end", "d", "step", "algorithm", "num")
    def __init__(self, *arg, **kwarg):
        for key, value in kwarg.items():
            setattr(self, key, value)

class default_contain:
    __slots__ = ("default", "deep", "default_dict", "leng")
    def __init__(self, default, default_dict=None):
        self.default, self.default_dict = default, (default_dict if default_dict else {})
    def __len__(self)->int:
        return -1
    def __getitem__(self, index):
        return self.default_dict.get(index, self.default)

#----------------- private module ---------------------------
_cos = lambda a, b, ab, name: ab / sqrt(a * b) if a * b else (1.0 if a == b else 0.0)
_mod = lambda a, b, ab, name: 2 * sqrt(a * b) / (a + b) if a * b else (1.0 if a == b else 0.0)
_cosmod = lambda a, b, ab, name: 2 * ab / (a + b) if a * b else (1.0 if a == b else 0.0)
_default_algorithm = _cosmod
private_dict = {
    "_cos": _cos,
    "_mod": _mod,
    "_cosmod": _cosmod,
    "_default_algorithm": _default_algorithm
}

#-------------------- core ----------------------------------
#    ------------------ passive mode ------------------------
def cos_comparison_passive(data,
                           *arg,
                           window_size=None,
                           w1=1, w2=1,
                           b1=0, b2=0,
                           start=None, end=None,
                           step=None, d=None,
                           algorithm=_default_algorithm,
                           output=None,
                           output_start=None, output_step=None,
                           start_callback=None,
                           end_callback=None,
                           global_error_callback=None,
                           local_error_callback=None,
                           return_callback=lambda output, name: output,
                           **kwargs):
    if hasattr(data, "__cos_comparison_passive__"):
        dicts = locals()
        return data.__cos_comparison_passive__(data, *arg, **dicts)  # allow to reload.

    temp = data
    length = []
    dimension = 0
    while True:
        try:
            length.append(len(temp))
            temp = temp[0]
            dimension += 1
        except:
            break

    start = start if start is not None else tuple([0 for _ in range(dimension)])
    end = end if end is not None else tuple(length)
    step = step if step is not None else tuple([1 for _ in range(dimension)])
    d = d if d is not None else tuple([1] + [0 for _ in range(dimension - 1)])
    window_size = window_size if window_size is not None else tuple([1 for _ in range(dimension)])
    num = [0 for _ in range(dimension)]
    for i in range(dimension):
        step_effective = end[i] - start[i] - window_size[i] - d[i]
        if step_effective >= 0:
            num[i] = (step_effective // step[i]) + 1
        else:
            raise ValueError("effectless args.")

    output_start = output_start if output_start is not None else tuple([0 for _ in range(dimension)])
    output_step = output_step if output_step is not None else tuple([1 for _ in range(dimension)])
    output = output if output is not None else create_void_list(
        ((n - 1) * s + 1 for n, s in zip(num, output_step))
    )

    name = func_name_space(
        output=output,
        output_start=output_start, output_step=output_step,
        window_size=window_size,
        linear=(w1, w2, b1, b2),
        start=start, end=end, step=step, d=d,
        algorithm=algorithm,
        num=num
    )

    if start_callback:
        start_callback(name)

    flag = dimension  # start at innermost dimension
    num_list = [None] + [1 for _ in num]  # 1‑based indices
    main, other, mu = 0, 0, 0

    while flag:
        try:
            if flag == dimension:
                output_temp = output
                # Inner loop over window offsets
                inner_list = [None] + [1 for _ in window_size]
                inner_flag = len(window_size)
                main = 0
                other = 0
                mu = 0

                while inner_flag:
                    try:
                        if inner_flag == dimension:
                            main_place = tuple(
                                start[i] + step[i] * (num_list[i + 1] - 1) + (inner_list[i + 1] - 1)
                                for i in range(dimension)
                            )
                            other_place = tuple(
                                main_place[i] + d[i] for i in range(dimension)
                            )
                            a = w1 * get_item(data, main_place) + b1
                            b = w2 * get_item(data, other_place) + b2
                            main += a * a
                            other += b * b
                            mu += a * b

                        # Advance inner position or carry left
                        if inner_list[inner_flag] < window_size[inner_flag - 1]:
                            inner_list[inner_flag] += 1
                            inner_flag = dimension
                        else:
                            inner_list[inner_flag] = 1
                            inner_flag -= 1
                    except Exception as e:
                        if local_error_callback:
                            local_error_callback(e, name)

                # Write result to output
                for p in range(dimension):
                    output_place = output_start[p] + output_step[p] * (num_list[p + 1] - 1)
                    if p == dimension - 1:
                        output_temp[output_place] = algorithm(main, other, mu, name)
                    else:
                        output_temp = output_temp[output_place]

            # Advance output position or carry left
            if num_list[flag] < num[flag - 1]:
                num_list[flag] += 1
                flag = dimension
            else:
                num_list[flag] = 1
                flag -= 1
        except Exception as e:
            if global_error_callback:
                global_error_callback(e, name)

    if end_callback:
        end_callback(name)

    return return_callback(output, name)

cos_comparison_passive_1d=cos_comparison_passive
cos_comparison_passive_2d=cos_comparison_passive
cos_comparison_passive_3d=cos_comparison_passive
cos_comparison_passive_4d=cos_comparison_passive

# -------------------- active mode --------------------
def cos_comparison_active(data,
                          *arg,
                          kernel=None,
                          w1=1, w2=1,
                          b1=0, b2=0,
                          start=None, end=None,
                          step=None,
                          algorithm=_default_algorithm,
                          output=None,
                          output_start=None, output_step=None,
                          start_callback=None,
                          end_callback=None,
                          global_error_callback=None,
                          local_error_callback=None,
                          return_callback=lambda output, name: output,
                          **kwargs):
    if hasattr(data, "__cos_comparison_active__"):
        dicts = locals()
        return data.__cos_comparison_active__(data, *arg, **dicts)

    if kernel is None:
        raise ValueError("kernel must be provided for active mode")

    # Infer data shape
    temp = data
    length = []
    dimension = 0
    while True:
        try:
            length.append(len(temp))
            temp = temp[0]
            dimension += 1
        except:
            break

    # Infer kernel shape (must have same dimension)
    ktemp = kernel
    kernel_shape = []
    for _ in range(dimension):
        kernel_shape.append(len(ktemp))
        ktemp = ktemp[0]

    start = start if start is not None else tuple([0 for _ in range(dimension)])
    end = end if end is not None else tuple(length)
    step = step if step is not None else tuple([1 for _ in range(dimension)])
    # In active mode, window size is determined by kernel shape
    window_size = tuple(kernel_shape)

    num = [0 for _ in range(dimension)]
    for i in range(dimension):
        step_effective = end[i] - start[i] - window_size[i]
        if step_effective >= 0:
            num[i] = (step_effective // step[i]) + 1
        else:
            raise ValueError("effectless args.")

    output_start = output_start if output_start is not None else tuple([0 for _ in range(dimension)])
    output_step = output_step if output_step is not None else tuple([1 for _ in range(dimension)])
    output = output if output is not None else create_void_list(
        ((n - 1) * s + 1 for n, s in zip(num, output_step))
    )

    name = func_name_space(
        output=output,
        output_start=output_start, output_step=output_step,
        window_size=window_size,
        kernel=kernel,
        linear=(w1, w2, b1, b2),
        start=start, end=end, step=step,
        algorithm=algorithm,
        num=num
    )

    if start_callback:
        start_callback(name)

    flag = dimension  # start at innermost dimension
    num_list = [None] + [1 for _ in num]
    main, other, mu = 0, 0, 0

    while flag:
        try:
            if flag == dimension:
                output_temp = output
                inner_list = [None] + [1 for _ in window_size]
                inner_flag = len(window_size)
                main = 0
                other = 0
                mu = 0

                while inner_flag:
                    try:
                        if inner_flag == dimension:
                            # Data window position
                            data_place = tuple(
                                start[i] + step[i] * (num_list[i + 1] - 1) + (inner_list[i + 1] - 1)
                                for i in range(dimension)
                            )
                            # Kernel position (always starting at 0 for each dim)
                            kern_place = tuple(
                                inner_list[i + 1] - 1 for i in range(dimension)
                            )
                            a = w1 * get_item(data, data_place) + b1
                            b = w2 * get_item(kernel, kern_place) + b2
                            main += a * a
                            other += b * b
                            mu += a * b

                        # Advance inner position or carry left
                        if inner_list[inner_flag] < window_size[inner_flag - 1]:
                            inner_list[inner_flag] += 1
                            inner_flag = dimension
                        else:
                            inner_list[inner_flag] = 1
                            inner_flag -= 1
                    except Exception as e:
                        if local_error_callback:
                            local_error_callback(e, name)

                # Write result to output
                for p in range(dimension):
                    output_place = output_start[p] + output_step[p] * (num_list[p + 1] - 1)
                    if p == dimension - 1:
                        output_temp[output_place] = algorithm(main, other, mu, name)
                    else:
                        output_temp = output_temp[output_place]

            # Advance output position or carry left
            if num_list[flag] < num[flag - 1]:
                num_list[flag] += 1
                flag = dimension
            else:
                num_list[flag] = 1
                flag -= 1
        except Exception as e:
            if global_error_callback:
                global_error_callback(e, name)

    if end_callback:
        end_callback(name)

    return return_callback(output, name)

cos_comparison_active_1d=cos_comparison_active
cos_comparison_active_2d=cos_comparison_active
cos_comparison_active_3d=cos_comparison_active
cos_comparison_active_4d=cos_comparison_active

#-------------- cos mode -----------
def cos_1d(a,b,algorithm=_cos):
    sum_a = 0.0
    sum_b = 0.0
    sum_ab = 0.0
    try:
        if len(a) != len(b) :
            raise ValueError("the shape of two tensors are not same.")
        shape1= len(a)
    except:
        raise ValueError("the args you gave are not tensors.")
    for p1 in range(shape1):
        a_temp = a[p1]
        b_temp = b[p1]
        sum_a += a_temp*a_temp
        sum_b += b_temp*b_temp
        sum_ab += a_temp*b_temp
    return algorithm(sum_a,sum_b,sum_ab,None)

def cos_2d(a,b,algorithm=_cos):
    sum_a = 0.0
    sum_b = 0.0
    sum_ab = 0.0
    try:
        if len(a) != len(b) or len(a[0]) !=len(b[0]) :
            raise ValueError("the shape of two tensors are not same.")
        shape1= len(a)
        shape2 = len(a[0])
    except:
        raise ValueError("the args you gave are not tensors.")
    for p1 in range(shape1):
        for p2 in range(shape2):
            a_temp = a[p1][p2]
            b_temp = b[p1][p2]
            sum_a += a_temp*a_temp
            sum_b += b_temp*b_temp
            sum_ab += a_temp*b_temp
    return algorithm(sum_a,sum_b,sum_ab,None)

def cos_3d(a,b,algorithm=_cos):
    sum_a = 0.0
    sum_b = 0.0
    sum_ab = 0.0
    try:
        if len(a) != len(b) or len(a[0]) !=len(b[0]) or len(a[0][0]) != len(b[0][0]) :
            raise ValueError("the shape of two tensors are not same.")
        shape1= len(a)
        shape2 = len(a[0])
        shape3 = len(a[0][0])
    except:
        raise ValueError("the args you gave are not tensors.")
    for p1 in range(shape1):
        for p2 in range(shape2):
            for p3 in range(shape3):
                a_temp = a[p1][p2][p3]
                b_temp = b[p1][p2][p3]
                sum_a += a_temp*a_temp
                sum_b += b_temp*b_temp
                sum_ab += a_temp*b_temp
    return algorithm(sum_a,sum_b,sum_ab,None)

def cos_4d(a,b,algorithm=_cos):
    sum_a = 0.0
    sum_b = 0.0
    sum_ab = 0.0
    try:
        if len(a) != len(b) or len(a[0]) !=len(b[0]) or len(a[0][0]) != len(b[0][0]) or len(a[0][0][0]) != len(b[0][0][0]) :
            raise ValueError("the shape of two tensors are not same.")
        shape1= len(a)
        shape2 = len(a[0])
        shape3 = len(a[0][0])
        shape4 = len(a[0][0][0])
    except:
        raise ValueError("the args you gave are not tensors.")
    for p1 in range(shape1):
        for p2 in range(shape2):
            for p3 in range(shape3):
                for p4 in range(shape4):
                    a_temp = a[p1][p2][p3][p4]
                    b_temp = b[p1][p2][p3][p4]
                    sum_a += a_temp*a_temp
                    sum_b += b_temp*b_temp
                    sum_ab += a_temp*b_temp
    return algorithm(sum_a,sum_b,sum_ab,None)

def cos(a, b, algorithm=_cos):
    """
    Compute similarity between two whole tensors (nested structures).
    The tensors are flattened element‑wise, squared sums and dot product
    are accumulated, then passed to the given algorithm.
    """
    # ----- Infer shapes and verify consistency (same as passive mode) -----
    shape = []
    tmp_a, tmp_b = a, b
    dimension = 0
    while True:
        try:
            len_a, len_b = len(tmp_a), len(tmp_b)
            if len_a != len_b:
                raise ValueError("the shape of two tensors are not same.")
            shape.append(len_a)
            dimension += 1
            tmp_a, tmp_b = tmp_a[0], tmp_b[0]
        except:
            shape = tuple(shape)
            break
    if dimension == 0:
        raise ValueError("the args you gave are not tensors.")

    # ----- Traverse all indices (1‑based, same as passive mode) -----
    num_list = [None] + [1] * dimension   # current position in each dim (1‑based)
    flag = dimension                      # start at innermost dimension

    sum_a = 0.0
    sum_b = 0.0
    sum_ab = 0.0

    while flag:
        if flag == dimension:
            # Build index tuple (0‑based) for the current position
            idx = tuple(num_list[i] - 1 for i in range(1, dimension + 1))

            val_a = get_item(a, idx)
            val_b = get_item(b, idx)

            sum_a += val_a * val_a
            sum_b += val_b * val_b
            sum_ab += val_a * val_b

        # Advance position or carry left
        if num_list[flag] < shape[flag - 1]:
            num_list[flag] += 1
            flag = dimension
        else:
            num_list[flag] = 1
            flag -= 1

    return algorithm(sum_a, sum_b, sum_ab, None)


#----------------- important module ------------------
#    ----------------------- statistic mode -----------------------
def mean_local(data, *arg, local_size=None, step=None,weight=None,
               output=None, output_start=None, output_step=None, **kwarg):
    """
    Generic local mean (arbitrary N‑dim).
    Supports external container via `output`, `output_start`, `output_step`.
    """
    if local_size is None:
        local_size = default_contain(1)
    if step is None:
        step = default_contain(1)

    # Build an all‑one kernel according to the given shape
    def _build_ones(shape):
        if isinstance(shape, default_contain):
            return shape               # let active mode handle it
        if isinstance(shape, int):
            return [1.0] * shape
        if isinstance(shape, (list, tuple)):
            if len(shape) == 0:
                return 1.0
            if len(shape) == 1:
                return _build_ones(shape[0])
            # shape is (dim0, dim1, ...): create dim0 copies of the remaining shape
            return [_build_ones(shape[1:]) for _ in range(shape[0])]
        raise TypeError("local_size must be a sequence, int, or default_contain")

    kernel = weight if weight else _build_ones(local_size)

    # Compute total number of elements in the window
    if hasattr(local_size, '__len__') and not isinstance(local_size, default_contain):
        N = multiple_chain(local_size, 1)
    else:
        N = 1   # fallback for default_contain (will be overridden by kernel size)

    return cos_comparison_active(data, *arg,
                                 kernel=kernel,
                                 step=step,
                                 output=output,
                                 output_start=output_start,
                                 output_step=output_step,
                                 algorithm=lambda a, b, ab, name: ab / N,
                                 **kwarg)


def local_variance(data, *arg, local_size=None, step=None,
                   output=None, output_start=None, output_step=None, **kwarg):
    """
    Generic local variance (arbitrary N‑dim).
    Supports external container via `output`, `output_start`, `output_step`.
    """
    if local_size is None:
        local_size = default_contain(1)
    if step is None:
        step = default_contain(1)

    def _build_ones(shape):
        if isinstance(shape, default_contain):
            return shape
        if isinstance(shape, int):
            return [1.0] * shape
        if isinstance(shape, (list, tuple)):
            if len(shape) == 0:
                return 1.0
            if len(shape) == 1:
                return _build_ones(shape[0])
            # shape is (dim0, dim1, ...): create dim0 copies of the remaining shape
            return [_build_ones(shape[1:]) for _ in range(shape[0])]
        raise TypeError("local_size must be a sequence, int, or default_contain")

    kernel = _build_ones(local_size)

    if hasattr(local_size, '__len__') and not isinstance(local_size, default_contain):
        N = multiple_chain(local_size, 1)
    else:
        N = 1

    def var_alg(a, b, ab, name):
        mean = ab / N
        return a / N - mean * mean

    return cos_comparison_active(data, *arg,
                                 kernel=kernel,
                                 step=step,
                                 output=output,
                                 output_start=output_start,
                                 output_step=output_step,
                                 algorithm=var_alg,
                                 **kwarg)

mean_local_1d = mean_local
mean_local_2d = mean_local
mean_local_3d = mean_local
mean_local_4d = mean_local

local_variance_1d = local_variance
local_variance_2d = local_variance
local_variance_3d = local_variance
local_variance_4d = local_variance
