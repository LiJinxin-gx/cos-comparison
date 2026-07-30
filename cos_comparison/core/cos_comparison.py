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
        return tuple( (sum((m*n for m,n in zip(vector,a[i]))) for i in range(leng)) )
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

    # Create tensor: note that the underlying vector is a copy of the region,
    # so start is 0 for the new tensor's own vector, shape is the requested region shape
    return vector_map_as_tensor(vector, shape, start=0, p=0)

def get_item(object, index):
    if hasattr(object,"__get_item__"):
        return object.__get_item__(*index) #allow to reload
    temp = object
    for p in index:
        temp = temp[p]
    return temp

def set_item(object,index,value):
    if hasattr(object,"__set_item__"):
        object.__set_item__(index,value)
    temp = object
    *indexp,endp=index
    for p in indexp:
        temp = temp[p]
    temp[endp] = value

def no_done(*arg,**kwarg):
    pass

#------------------- class support --------------------------
#     --------------- type support -----------------
class vector_map_as_tensor:
    __slots__ = ("vector", "shape", "strides", "start", "p")
    def __init__(self, vector, shape, start=0, p=0, strides=None):
        self.vector = vector
        self.shape = tuple(shape)
        self.start = start
        self.p = p
        # Precompute strides once at creation, strides[i] is step for dimension i in flat memory
        if strides is None:
            ndim = len(self.shape)
            strides = [1] * ndim
            for i in range(ndim - 2, -1, -1):
                strides[i] = strides[i+1] * self.shape[i+1]
            self.strides = tuple(strides)
        else:
            self.strides = tuple(strides)

    @property
    def dimension(self):
        return len(self.shape)

    @property
    def end(self):
        return self.start + self.shape[self.p] * self.strides[self.p]

    @property
    def tensor_size(self):
        # Backward compatibility alias
        return self.shape

    @property
    def cache(self):
        # Backward compatibility alias for old cache attribute
        return self.strides[self.p]

    def __repr__(self):
        return f"<vector_map_as_tensor: dim={len(self.shape)-self.p}, start={self.start}, end={self.end}, p={self.p}>"

    def __getitem__(self, index):
        # Handle tuple of indices (int/slice mix)
        if isinstance(index, tuple):
            obj = self
            for i, idx in enumerate(index):
                if isinstance(idx, slice):
                    # Only last index can be slice
                    if i != len(index) - 1:
                        raise IndexError("slice can only be the last index")
                    return obj[idx]
                obj = obj[idx]
            return obj

        # Handle single slice
        if isinstance(index, slice):
            length = self.shape[self.p]
            start, stop, step = index.indices(length)
            if step != 1:
                raise NotImplementedError("step != 1 not supported for slice")
            new_start = self.start + start * self.strides[self.p]
            new_shape = list(self.shape)
            new_shape[self.p] = stop - start
            return self.__class__(vector=self.vector, shape=tuple(new_shape),
                                  start=new_start, p=self.p, strides=self.strides)

        # Handle single integer
        if isinstance(index, int):
            if index < 0:
                index += self.shape[self.p]
            new_p = self.p + 1
            new_start = self.start + index * self.strides[self.p]
            if new_p == len(self.shape):
                return self.vector[new_start]
            return self.__class__(vector=self.vector, shape=self.shape,
                                  start=new_start, p=new_p, strides=self.strides)
        raise TypeError(f"Invalid index type: {type(index)}")

    def __setitem__(self, key, value):
        # Handle tuple of indices (int/slice mix)
        if isinstance(key, tuple):
            obj = self
            for i, idx in enumerate(key):
                if isinstance(idx, slice):
                    if i != len(key) - 1:
                        raise IndexError("slice can only be the last index")
                    obj[idx] = value
                    return
                obj = obj[idx]
            # All indices are int, set scalar value
            if isinstance(obj, vector_map_as_tensor):
                raise IndexError("not enough indices")
            # obj is scalar from vector, set it
            # Wait, no: obj is the result of __getitem__ which returns scalar for all int indices
            # We need to track the flat index
            idx = self.start
            for i, dim_i in enumerate(key):
                idx += dim_i * self.strides[self.p + i]
            self.vector[idx] = value
            return

        if isinstance(key, slice):
            if self.p != len(self.shape) - 1:
                raise IndexError("Slice assignment only allowed at leaf dimension")
            length = self.shape[self.p]
            start, stop, step = key.indices(length)
            if step != 1:
                raise NotImplementedError("step != 1 not supported for slice assignment")
            count = stop - start
            start_idx = self.start + start * self.strides[self.p]
            stride = self.strides[self.p]
            # Scalar assignment
            if isinstance(value, (int, float)):
                for i in range(count):
                    self.vector[start_idx + i * stride] = value
                return
            # List/tuple assignment
            if isinstance(value, (list, tuple)):
                if len(value) != count:
                    raise ValueError("length of sequence does not match slice length")
                for i in range(count):
                    self.vector[start_idx + i * stride] = value[i]
                return
            # Vector assignment
            if isinstance(value, vector_map_as_tensor):
                if value.end - value.start != count:
                    raise ValueError("length of Vector does not match slice length")
                val_stride = value.strides[value.p]
                for i in range(count):
                    self.vector[start_idx + i * stride] = value.vector[value.start + i * val_stride]
                return
            # Buffer protocol support (array.array, memoryview, numpy arrays, bytes, etc.)
            if hasattr(value, '__buffer__'):
                mv = memoryview(value)
                # Auto-detect format: support double ('d') and unsigned char ('B')
                if mv.format in ('d', 'B') and mv.itemsize in (8, 1):
                    if mv.format == 'd' and mv.itemsize == 8:
                        buf = mv
                    else:
                        # Cast unsigned char bytes to double (0-255 range)
                        buf = mv.cast('B')
                    if len(buf) != count:
                        raise ValueError("buffer length does not match slice length")
                    for i in range(count):
                        self.vector[start_idx + i * stride] = buf[i]
                    return
                # Fallback: try cast to double
                try:
                    buf = mv.cast('d')
                    if len(buf) != count:
                        raise ValueError("buffer length does not match slice length")
                    for i in range(count):
                        self.vector[start_idx + i * stride] = buf[i]
                    return
                except (TypeError, ValueError):
                    pass
            raise TypeError("value must be scalar, sequence, Vector, or buffer-like object")

        if isinstance(key, int):
            if key < 0:
                key += self.shape[self.p]
            if self.p == len(self.shape) - 1:
                self.vector[self.start + key * self.strides[self.p]] = value
                return
            raise IndexError("not enough indices for assignment")
        raise TypeError(f"Invalid index type: {type(key)}")

    def __get_item__(self, *indexs):
        ndim = len(self.shape) - self.p
        if len(indexs) != ndim:
            raise IndexError(f"expected {ndim} indices, got {len(indexs)}")
        idx = self.start
        for i, dim_i in enumerate(indexs):
            idx += dim_i * self.strides[self.p + i]
        return self.vector[idx]

    def __set_item__(self, indexs, value):
        remaining = len(self.shape) - self.p
        if len(indexs) == remaining:
            ptr = self.start
            for i, idx in enumerate(indexs):
                ptr += idx * self.strides[self.p + i]
            self.vector[ptr] = value
        elif len(indexs) != 0:
            raise IndexError("It was given some effectless index.")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return self.shape[self.p]

    def _check_shape(self, other):
        if not isinstance(other, vector_map_as_tensor):
            raise TypeError("It can not compute with other type.")
        if self.shape != other.shape:
            raise ValueError("the shape of two tensors are not same.")
        if self.p != other.p:
            raise ValueError("the current depths are not same.")

    # ---------- binary arithmetic ----------
    def __add__(self, other):
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            total = self.end - self.start
            new_vec = [0.0] * total
            for i in range(total):
                new_vec[i] = self.vector[self.start + i] + other.vector[other.start + i]
            return self.__class__(vector=new_vec, shape=self.shape[self.p:], start=0, p=0)
        elif isinstance(other, (int, float)):
            total = self.end - self.start
            new_vec = [0.0] * total
            for i in range(total):
                new_vec[i] = self.vector[self.start + i] + other
            return self.__class__(vector=new_vec, shape=self.shape[self.p:], start=0, p=0)
        else:
            raise TypeError("It can not compute with other type.")
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            total = self.end - self.start
            new_vec = [0.0] * total
            for i in range(total):
                new_vec[i] = self.vector[self.start + i] - other.vector[other.start + i]
            return self.__class__(vector=new_vec, shape=self.shape[self.p:], start=0, p=0)
        elif isinstance(other, (int, float)):
            total = self.end - self.start
            new_vec = [0.0] * total
            for i in range(total):
                new_vec[i] = self.vector[self.start + i] - other
            return self.__class__(vector=new_vec, shape=self.shape[self.p:], start=0, p=0)
        else:
            raise TypeError("It can not compute with other type.")
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            total = self.end - self.start
            new_vec = [0.0] * total
            for i in range(total):
                new_vec[i] = other - self.vector[self.start + i]
            return self.__class__(vector=new_vec, shape=self.shape[self.p:], start=0, p=0)
        raise TypeError("It can not compute with other type.")

    def __mul__(self, other):
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            total = self.end - self.start
            new_vec = [0.0] * total
            for i in range(total):
                new_vec[i] = self.vector[self.start + i] * other.vector[other.start + i]
            return self.__class__(vector=new_vec, shape=self.shape[self.p:], start=0, p=0)
        elif isinstance(other, (int, float)):
            total = self.end - self.start
            new_vec = [0.0] * total
            for i in range(total):
                new_vec[i] = self.vector[self.start + i] * other
            return self.__class__(vector=new_vec, shape=self.shape[self.p:], start=0, p=0)
        else:
            raise TypeError("It can not compute with other type.")
    
    def __rmul__(self, other):
        # Scalar multiplication is commutative
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            total = self.end - self.start
            new_vec = [0.0] * total
            for i in range(total):
                b = other.vector[other.start + i]
                if b == 0:
                    raise ZeroDivisionError("division by zero")
                new_vec[i] = self.vector[self.start + i] / b
            return self.__class__(vector=new_vec, shape=self.shape[self.p:], start=0, p=0)
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("division by zero")
            total = self.end - self.start
            new_vec = [0.0] * total
            for i in range(total):
                new_vec[i] = self.vector[self.start + i] / other
            return self.__class__(vector=new_vec, shape=self.shape[self.p:], start=0, p=0)
        else:
            raise TypeError("It can not be used with other type.")

    def __pow__(self, other):
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            total = self.end - self.start
            new_vec = [0.0] * total
            for i in range(total):
                new_vec[i] = self.vector[self.start + i] ** other.vector[other.start + i]
            return self.__class__(vector=new_vec, shape=self.shape[self.p:], start=0, p=0)
        elif isinstance(other, (int, float)):
            total = self.end - self.start
            new_vec = [0.0] * total
            for i in range(total):
                new_vec[i] = self.vector[self.start + i] ** other
            return self.__class__(vector=new_vec, shape=self.shape[self.p:], start=0, p=0)
        raise TypeError("unsupported operand type(s) for **")

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
        if isinstance(other, vector_map_as_tensor):
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
        if isinstance(other, vector_map_as_tensor):
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

    def __ipow__(self, other):
        if isinstance(other, vector_map_as_tensor):
            self._check_shape(other)
            for i in range(self.end - self.start):
                self.vector[self.start + i] **= other.vector[other.start + i]
        elif type(other) in (int, float):
            for i in range(self.end - self.start):
                self.vector[self.start + i] **= other
        else:
            raise TypeError("unsupported operand type(s) for **=")
        return self

    # ---------- unary arithmetic ----------
    def __neg__(self):
        total = self.end - self.start
        new_vec = [0.0] * total
        for i in range(total):
            new_vec[i] = -self.vector[self.start + i]
        return self.__class__(vector=new_vec, shape=self.shape[self.p:], start=0, p=0)

    def __pos__(self):
        total = self.end - self.start
        new_vec = [0.0] * total
        for i in range(total):
            new_vec[i] = self.vector[self.start + i]
        return self.__class__(vector=new_vec, shape=self.shape[self.p:], start=0, p=0)

    def __abs__(self):
        square_sum = sum(i * i for i in self.vector[self.start:self.end])
        return sqrt(square_sum)

    def mean(self):
        count = self.end - self.start
        if count == 0:
            return None
        # Welford's online algorithm for numerically stable mean (no overflow)
        mean = 0.0
        for i in range(count):
            val = self.vector[self.start + i]
            delta = val - mean
            mean += delta / (i + 1)
        return mean

    def variance(self):
        count = self.end - self.start
        if count == 0:
            return None
        # Welford's online algorithm for numerically stable variance (no large sum overflow)
        mean = 0.0
        M2 = 0.0
        for i in range(count):
            val = self.vector[self.start + i]
            delta = val - mean
            mean += delta / (i + 1)
            delta2 = val - mean
            M2 += delta * delta2
        return M2 / count 

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
                output_places = tuple( ( output_start[p] + output_step[p] * (num_list[p + 1] - 1) for p in range(dimension) ) )
                set_item(output,output_places,algorithm(main, other, mu, name))

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
                output_places = tuple( ( output_start[p] + output_step[p] * (num_list[p + 1] - 1) for p in range(dimension) ) )
                set_item(output,output_places,algorithm(main, other, mu, name))

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

# ---------- public API export list (matches other backends) ----------
__all__ = [
    'NaN', 'sqrt',
    'cos_comparison_passive', 'cos_comparison_passive_1d', 'cos_comparison_passive_2d', 'cos_comparison_passive_3d', 'cos_comparison_passive_4d',
    'cos_comparison_active', 'cos_comparison_active_1d', 'cos_comparison_active_2d', 'cos_comparison_active_3d', 'cos_comparison_active_4d',
    'cos', 'cos_1d', 'cos_2d', 'cos_3d', 'cos_4d',
    'mean_local', 'mean_local_1d', 'mean_local_2d', 'mean_local_3d', 'mean_local_4d',
    'local_variance', 'local_variance_1d', 'local_variance_2d', 'local_variance_3d', 'local_variance_4d',
    'multiple_chain', 'add_chain', 'no_done', 'create_void_list', 'load_as_default_data', 'get_item', 'set_item', '_cos', '_mod', '_cosmod',
    'vector_chain_compute',
    'vector_map_as_tensor', 'func_name_space', 'default_contain',
    'private_dict'
]

