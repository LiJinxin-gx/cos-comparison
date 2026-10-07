"""Unit map: fold variable-length runs of repeated units into fixed-length
real elements (symbolic data to the tensor domain).

Registration is eager (the streaming workflow is always available).
``threshold`` filters folding by total occurrence count (units below
it pass through raw); ``length`` caps the span of one folded element
(int for 1-D streams, per-dimension tuple over ``shape`` for grids);
``+`` / ``merge_all`` merge not-yet-output maps in stream order
(open resolution: min threshold, elementwise max length)."""

import operator

__all__ = ("UnitMap", "map_data", "window_units")

_DEFAULT_START = 0.0
_DEFAULT_STEP = 1.0
_SIG_MASK = (1 << 61) - 1
_SCALAR_EQ_TYPES = (int, float, complex, str, bytes, bool)


def _scalar_tuple(obj):
    """True when a tuple holds only scalar leaves (fast == is then
    equivalent to _deep_eq)."""
    for item in obj:
        if type(item) not in _SCALAR_EQ_TYPES:
            return False
    return True


def _is_mapping(obj):
    """Mapping protocol (duck): keys() + item access."""
    return hasattr(obj, "keys") and hasattr(obj, "__getitem__")


def _is_sequence(obj):
    """Sequence protocol: sized + indexable/iterable; text and mappings
    excluded (mappings use the mapping protocol)."""
    if _is_mapping(obj) or isinstance(obj, (str, bytes)):
        return False
    try:
        len(obj)
    except TypeError:
        return False
    return hasattr(obj, "__getitem__") or hasattr(obj, "__iter__")


def _is_set(obj):
    """Set protocol (duck): membership + iteration without item access;
    text and mappings are excluded."""
    return not _is_mapping(obj) and not isinstance(obj, (str, bytes)) \
        and hasattr(obj, "__contains__") and hasattr(obj, "__iter__") \
        and not hasattr(obj, "__getitem__")


def _sequence_label(obj):
    """Structural label keeping the list/tuple distinction via mutability:
    1 = mutable (list-like), 2 = immutable (tuple-like)."""
    return 1 if hasattr(obj, "__setitem__") else 2


def _set_label(obj):
    """1 = mutable set-like, 2 = immutable (frozenset-like)."""
    return 1 if hasattr(obj, "add") else 2


class UnitMap:
    """Run-folding unit mapper (duck typing - any objects as units).

    ``add`` accumulates object streams; consecutive-equal runs fold into
    single real elements (variable-length runs -> fixed-length elements).
    A cumulative content table (object <-> flag) stays queryable
    (``flag_of`` / ``decode``).  ``put`` writes folded output with a
    file-pointer model: the output position is remembered per output
    object, ``buffering`` limits the elements of one call (the rest stays
    queued for the next put).  Run statistics are internal (counts).

    Appending is a recurrence: adding data after a completed parse only
    processes the new part (the pending run continues, counts accumulate,
    new runs queue independently).  Unhashable units are resolved through
    content-signature buckets with iterative deep equality."""

    __slots__ = ("_counts", "_flags", "_last_out", "_lens", "_n",
                 "_out_pos", "_output_started", "_partial", "_pending",
                 "_plain_sig", "_queued", "_queued_units", "_units_in",
                 "length", "shape", "start", "step", "table",
                 "threshold")

    def __init__(self, start=_DEFAULT_START, step=_DEFAULT_STEP,
                 threshold=1, length=None, shape=None):
        if not isinstance(threshold, int) or threshold < 1:
            raise ValueError("threshold must be an int >= 1")
        if shape is not None:
            shape = tuple(shape)
            if not shape or any(not isinstance(s, int) or s < 1
                                for s in shape):
                raise ValueError(
                    "shape must be a tuple of positive ints")
        if length is not None:
            if isinstance(length, int):
                if length < 1:
                    raise ValueError("length must be >= 1")
                if shape is not None:
                    length = (length,) * len(shape)
            elif (isinstance(length, tuple) and length
                  and all(isinstance(v, int) and v >= 1
                          for v in length)):
                if shape is None or len(length) != len(shape):
                    raise ValueError(
                        "length tuple requires a shape of equal rank")
                length = tuple(length)
            else:
                raise ValueError(
                    "length must be None, a positive int, or a tuple "
                    "of positive ints")
        self.start = float(start)
        self.step = float(step)
        self.threshold = threshold
        self.length = length
        self.shape = shape
        if length is None:
            self._lens = None
        elif shape is None:
            self._lens = (length,)
        else:
            self._lens = tuple(length)
        self.table = {}          # hashable obj -> flag
        self._plain_sig = {}     # content signature -> [[obj, flag, ...]]
        self._flags = None       # flag -> obj (lazy reverse map)
        self._counts = {}        # hashable obj -> [count, runs]
        self._pending = None     # (obj, run_length) not closed yet
        self._queued = []        # closed runs not yet output: (obj, len)
        self._n = 0              # total unique units (flag sequence)
        self._last_out = None
        self._out_pos = 0
        self._partial = None     # (item, is_raw, next, remaining)
        self._output_started = False
        self._units_in = 0       # units pushed
        self._queued_units = 0   # units pushed but not yet emitted

    # -- lookup / registration ------------------------------------------

    def _hashable(self, obj):
        try:
            hash(obj)
            return True
        except TypeError:
            return False

    def _sig_bucket(self, obj):
        """Records of one signature bucket (usually a single ==-verified
        record), or None when the signature is unknown."""
        return self._plain_sig.get(_fold_root(obj, _SIG_MASK))

    def flag_of(self, obj):
        """Content-table query: the flag of a unit, or None."""
        try:
            flag = self.table.get(obj)
        except TypeError:
            pass
        else:
            if flag is not None:
                return flag
            if not self._plain_sig:
                return None
        recs = self._sig_bucket(obj)
        if recs:
            for rec in recs:
                if _deep_eq(rec[0], obj):
                    return rec[1]
        return None

    def _flags_dict(self):
        """Reverse map, built lazily on first use."""
        rev = self._flags
        if rev is None:
            rev = {}
            for obj, flag in self.table.items():
                rev[flag] = obj
            for bucket in self._plain_sig.values():
                for rec in bucket:
                    rev[rec[1]] = rec[0]
            self._flags = rev
        return rev

    def decode(self, flags):
        """Reverse query over the content table (compatibility surface):
        a flag -> unit list (unknown flags decode to None)."""
        rev = self._flags_dict()
        out = []
        for flag in flags:
            out.append(rev.get(flag))
        return out

    def __contains__(self, obj):
        return self.flag_of(obj) is not None

    def __len__(self):
        return len(self.table) + sum(len(v) for v in
                                     self._plain_sig.values())

    def _register(self, obj):
        """Allocate the next flag for a new unit (first sight)."""
        flag = self.start + self.step * self._n
        self._n += 1
        if self._hashable(obj):
            self.table[obj] = flag
        else:  # unhashable unit: signature-bucket table (== lookup)
            rec = [obj, flag, 0, 0]
            sig = _fold_root(obj, _SIG_MASK)
            bucket = self._plain_sig.get(sig)
            if bucket is None:
                self._plain_sig[sig] = [rec]
            else:
                bucket.append(rec)
        if self._flags is not None:
            self._flags[flag] = obj
        return flag

    def _bump(self, obj, length):
        """Record run statistics: total elements and run count."""
        try:
            rec = self._counts.get(obj)
        except TypeError:
            recs = self._sig_bucket(obj)
            if recs:
                for record in recs:
                    if _deep_eq(record[0], obj):
                        record[2] += length
                        record[3] += 1
                        return
            return
        if rec is None:
            self._counts[obj] = [length, 1]
        else:
            rec[0] += length
            rec[1] += 1

    def _stat_rec(self, obj):
        """The statistic record [count, runs] of a unit, or None."""
        try:
            return self._counts.get(obj)
        except TypeError:
            pass
        recs = self._sig_bucket(obj)
        if recs:
            for rec in recs:
                if _deep_eq(rec[0], obj):
                    return [rec[2], rec[3]]
        return None

    # -- frequency statistics (counts; dynamic - no shared counter) ----

    def _unit_stats(self, obj):
        """(count, runs) of a unit - closed runs plus the open pending
        run (statistics are live)."""
        rec = self._stat_rec(obj)
        c = rec[0] if rec is not None else 0
        r = rec[1] if rec is not None else 0
        if self._pending is not None and _deep_eq(self._pending[0], obj):
            c += self._pending[1]
            r += 1
        return c, r

    def _units(self):
        """Iterate all registered units (hashable then plain)."""
        yield from self.table
        for bucket in self._plain_sig.values():
            for rec in bucket:
                yield rec[0]

    def total(self):
        """Cumulative element total - dynamic: summed on demand over the
        per-unit counts plus the open pending run (no shared global
        counter, so concurrent adds cannot corrupt it)."""
        n = 0
        for obj in self._units():
            n += self._unit_stats(obj)[0]
        return n

    def count(self, obj):
        """Element frequency (count) of a unit; 0 when unregistered."""
        return self._unit_stats(obj)[0]

    def runs(self, obj):
        """Run count of a unit; 0 when unregistered."""
        return self._unit_stats(obj)[1]

    def most_common(self, k=None):
        """Units sorted by element count (descending) as
        [(obj, count, runs), ...]; ``k`` truncates (None = all)."""
        items = []
        for obj in self._units():
            c, r = self._unit_stats(obj)
            items.append((obj, c, r))
        items.sort(key=lambda t: t[1], reverse=True)
        if k is not None:
            items = items[:k]
        return items

    def count_vector(self):
        """Per-unit counts ordered by flag sequence (index 0 = first
        registered unit) - a plain int vector, ready for tensor
        mapping."""
        vec = [0] * len(self)
        for obj in self._units():
            flag = self.flag_of(obj)
            idx = round((flag - self.start) / self.step)
            vec[idx] = self._unit_stats(obj)[0]
        return vec

    # -- input -----------------------------------------------------------

    def add(self, *seq):
        """Add unit streams (any iterable, any objects).  Consecutive
        equal units close/continue the pending run.  Appending is
        incremental: only the new units are processed (recurrence)."""
        for units in seq:
            for unit in units:
                self._push(unit)
        return 0

    def _push(self, unit):
        self._units_in += 1
        self._queued_units += 1
        pend = self._pending
        if pend is not None:
            pobj = pend[0]
            if pobj is unit:
                self._pending = (unit, pend[1] + 1)
                return
            kind = type(pobj)
            if kind is type(unit) and kind is tuple:
                equal = pobj == unit
                if equal and not (_scalar_tuple(pobj)
                                  and _scalar_tuple(unit)):
                    equal = _deep_eq(pobj, unit)
            elif kind is type(unit) and kind in _SCALAR_EQ_TYPES:
                equal = pobj == unit
            else:
                equal = _deep_eq(pobj, unit)
            if equal:
                self._pending = (unit, pend[1] + 1)
                return
            self._close()
        if self._hashable(unit) and not self._plain_sig:
            if unit not in self.table:   # first sight registers
                self._register(unit)
        elif self.flag_of(unit) is None:
            self._register(unit)
        self._pending = (unit, 1)

    def _register_get(self, obj):
        """Flag lookup + registration (direct table access when no
        unhashable units exist)."""
        if self._plain_sig:
            flag = self.flag_of(obj)
        else:
            flag = self.table.get(obj)
        if flag is None:
            flag = self._register(obj)
        return flag

    def _piece(self, start, run_len):
        """Length of the first greedy maximal-prefix piece within the
        per-dimension length caps (no caps -> the whole run)."""
        lens = self._lens
        if lens is None:
            return run_len
        shape = self.shape
        if shape is None:
            cap = lens[0]
            return cap if cap < run_len else run_len
        rank = len(shape)
        coords = [0] * rank
        index = start
        for d in range(rank - 1, -1, -1):
            coords[d] = index % shape[d]
            index //= shape[d]
        lo = coords[:]
        hi = coords[:]
        piece = 0
        for _ in range(run_len):
            too_wide = False
            for d in range(rank):
                nd = coords[d]
                nlo = lo[d] if lo[d] < nd else nd
                nhi = hi[d] if hi[d] > nd else nd
                if nhi - nlo + 1 > lens[d]:
                    too_wide = True
                    break
            if too_wide:
                break
            for d in range(rank):
                if coords[d] < lo[d]:
                    lo[d] = coords[d]
                if coords[d] > hi[d]:
                    hi[d] = coords[d]
            piece += 1
            for d in range(rank - 1, -1, -1):
                coords[d] += 1
                if coords[d] < shape[d]:
                    break
                coords[d] = 0
        return piece

    def _close(self):
        obj, length = self._pending
        self._queued.append((obj, length))
        self._bump(obj, length)
        self._pending = None

    # -- output ----------------------------------------------------------

    def put(self, output=None, buffering=None):
        """Write the folded elements.

        Default configuration keeps the original workflow; ``threshold``
        filters folding by total count (low units pass through raw);
        ``length`` splits runs into per-dimension-bounded pieces sharing
        one flag.  Buffering stops between elements; the unfinished run
        is carried in ``_partial``."""
        if self._pending is not None:
            self._close()
        fresh = output is None
        if fresh:
            output = []
            self._out_pos = 0
            write = output.append
        else:
            self._out_pos = 0 if output is not self._last_out \
                else self._out_pos
            write = None
        written = 0
        threshold = self.threshold
        queued = self._queued
        partial = self._partial
        if self._lens is None and threshold == 1 and partial is None:
            # original eager loop (fresh outputs append directly)
            for obj, run_len in queued:
                if buffering is not None and written >= buffering:
                    break
                flag = self._register_get(obj)
                if write is not None:
                    write(flag)
                    written += 1
                    self._queued_units -= run_len
                elif self._write(output, flag):
                    written += 1
                    self._queued_units -= run_len
            if written:
                self._output_started = True
            if fresh:
                self._queued = queued[written:]
                self._out_pos = written
                return output
            if written:
                del queued[:written]
            self._last_out = output
            return written
        counts = self._counts
        index = 0
        item = None
        is_raw = False
        next_start = 0
        remaining = 0
        have = False
        if partial is not None:
            item, is_raw, next_start, remaining = partial
            have = True
        while True:
            if buffering is not None and written >= buffering:
                break
            if not have:
                if index >= len(queued):
                    break
                obj, run_len = queued[index]
                if threshold > 1:
                    if self._hashable(obj):
                        rec = counts.get(obj)
                        count = rec[0] if rec is not None else 0
                    else:
                        count = self._unit_stats(obj)[0]
                else:
                    count = threshold
                if count < threshold:
                    is_raw = True
                    item = obj
                    next_start = 0
                    remaining = run_len
                else:
                    is_raw = False
                    item = self._register_get(obj)
                    next_start = self._units_in - self._queued_units
                    remaining = run_len
                have = True
            if is_raw:
                if write is not None:
                    write(item)
                elif not self._write(output, item):
                    break
                written += 1
                remaining -= 1
                self._queued_units -= 1
            else:
                piece = self._piece(next_start, remaining)
                if write is not None:
                    write(item)
                elif not self._write(output, item):
                    break
                written += 1
                next_start += piece
                remaining -= piece
                self._queued_units -= piece
            if remaining == 0:
                have = False
                index += 1
        if have:
            self._partial = (item, is_raw, next_start, remaining)
        else:
            self._partial = None
        if index:
            del queued[:index]
        if written:
            self._output_started = True
        if fresh:
            self._out_pos = written
            return output
        self._last_out = output
        return written

    def _write(self, output, flag):
        """Write one element: pre-allocated slots first (sequence
        protocol), otherwise append (accumulating containers)."""
        try:
            output[self._out_pos] = flag
            self._out_pos += 1
            return True
        except (TypeError, IndexError):
            pass
        append = getattr(output, "append", None)
        if append is not None:
            append(flag)
            self._out_pos += 1
            return True
        return False

    # -- ordered merge (stream concatenation) ----------------------------

    def _open_runs(self):
        """Closed queue runs plus the open pending run, as
        [(obj, length, is_open)]."""
        runs = [(obj, run_len, False) for obj, run_len in self._queued]
        if self._pending is not None:
            runs.append((self._pending[0], self._pending[1], True))
        return runs

    @staticmethod
    def _resolve(maps):
        """Open resolution: min threshold, elementwise max length (None
        absorbs); start/step/shape must match; not-yet-output maps only
        (partial output states are rejected)."""
        head = maps[0]
        threshold = head.threshold
        lens = head._lens
        for other in maps[1:]:
            if (head.start, head.step, head.shape) != (
                    other.start, other.step, other.shape):
                raise ValueError(
                    "merge requires identical start/step/shape")
            if other.threshold < threshold:
                threshold = other.threshold
            other_lens = other._lens
            if lens is None or other_lens is None:
                lens = None
            else:
                lens = tuple(a if a > b else b
                             for a, b in zip(lens, other_lens))
        for mapper in maps:
            if mapper._output_started:
                raise ValueError(
                    "merge requires not-yet-output maps")
            if mapper._partial is not None:
                raise ValueError(
                    "merge requires maps without partial output")
        return threshold, lens

    @classmethod
    def merge_all(cls, maps):
        """Ordered n-ary merge.

        ``A + B`` equals folding the concatenated streams (sequential,
        non-commutative, associative); parameter conflicts resolve to the
        loosest condition (min threshold, max length).  The merged map
        registers units in stream order during the merge, so flags follow
        concat-order first sight and the streaming workflow stays
        available."""
        maps = list(maps)
        if not maps:
            raise ValueError("merge_all needs at least one map")
        threshold, lens = cls._resolve(maps)
        head = maps[0]
        merged = cls(head.start, head.step, threshold=threshold,
                     shape=head.shape)
        if lens is not None:
            merged.length = lens if len(lens) > 1 else lens[0]
            merged._lens = lens
        runs = []
        for mapper in maps:
            part = mapper._open_runs()
            if not part:
                continue
            if runs and _deep_eq(runs[-1][0], part[0][0]):
                last = runs.pop()
                runs.append((last[0], last[1] + part[0][1], part[0][2]))
                runs.extend(part[1:])
            else:
                runs.extend(part)
        if runs:
            runs = [(obj, run_len, False)
                    for obj, run_len, _open in runs[:-1]] + [runs[-1]]
        for obj, run_len, is_open in runs:
            if merged.flag_of(obj) is None:
                merged._register(obj)
            if not is_open:
                merged._bump(obj, run_len)
        if runs and runs[-1][2]:
            merged._pending = (runs[-1][0], runs[-1][1])
            tail = runs[:-1]
        else:
            merged._pending = None
            tail = runs
        merged._queued = [(obj, run_len) for obj, run_len, _open in tail]
        merged._units_in = sum(mapper._units_in for mapper in maps)
        merged._queued_units = sum(mapper._queued_units for mapper in maps)
        return merged

    def __add__(self, other):
        if not isinstance(other, UnitMap):
            return NotImplemented
        return UnitMap.merge_all([self, other])

    def merge(self, other):
        """Ordered merge returning a new map (``self + other``)."""
        return self.__add__(other)

    def __iadd__(self, other):
        merged = self.__add__(other)
        if merged is NotImplemented:
            return NotImplemented
        self.start = merged.start
        self.step = merged.step
        self.threshold = merged.threshold
        self.length = merged.length
        self.shape = merged.shape
        self._lens = merged._lens
        self.table = merged.table
        self._plain_sig = merged._plain_sig
        self._flags = merged._flags
        self._counts = merged._counts
        self._pending = merged._pending
        self._queued = merged._queued
        self._out_pos = 0
        self._partial = None
        self._n = merged._n
        self._last_out = None
        self._output_started = False
        self._units_in = merged._units_in
        self._queued_units = merged._queued_units
        return self

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    # -- state -----------------------------------------------------------

    def get_state(self):
        """Export the state: table/signature records, statistics, output
        pointer, queued and pending runs (importable via set_state)."""
        plain = {sig: [list(r) for r in recs]
                 for sig, recs in self._plain_sig.items()}
        return {
            "start": self.start, "step": self.step,
            "table": dict(self.table), "plain_sig": plain,
            "counts": dict(self._counts),
            "pending": self._pending,
            "queued": list(self._queued),
            "out_pos": self._out_pos,
            "n": self._n,
            "threshold": self.threshold,
            "length": self.length,
            "shape": self.shape,
            "partial": self._partial,
            "output_started": self._output_started,
            "units_in": self._units_in,
            "queued_units": self._queued_units,
            "flags": self._flags_dict(),
        }

    def set_state(self, state):
        """Restore an exported state."""
        self.start = state.get("start", _DEFAULT_START)
        self.step = state.get("step", _DEFAULT_STEP)
        self.table = dict(state.get("table", {}))
        self._plain_sig = {
            sig: [list(r) for r in recs]
            for sig, recs in state.get("plain_sig", {}).items()}
        self._flags = dict(state.get("flags", {}))
        self._counts = dict(state.get("counts", {}))
        self._pending = state.get("pending")
        self._queued = [tuple(r) for r in state.get("queued", [])]
        self._out_pos = state.get("out_pos", 0)
        self.threshold = state.get("threshold", 1)
        self.length = state.get("length")
        self.shape = state.get("shape")
        self._partial = state.get("partial")
        self._output_started = state.get("output_started", False)
        self._units_in = state.get("units_in", 0)
        self._queued_units = state.get("queued_units", 0)
        if self.length is None:
            self._lens = None
        elif self.shape is None:
            self._lens = (self.length,)
        else:
            self._lens = tuple(self.length)
        self._n = state.get("n", len(self.table)
                            + sum(len(v) for v in
                                  self._plain_sig.values()))
        self._last_out = None
        return 0

    def clear(self):
        """Clear everything (table, statistics, runs, output pointer)."""
        self.table = {}
        self._plain_sig = {}
        self._flags = None
        self._counts = {}
        self._pending = None
        self._queued = []
        self._last_out = None
        self._out_pos = 0
        self._partial = None
        self._output_started = False
        self._units_in = 0
        self._queued_units = 0
        self._n = 0
        return 0


# ----------------------------------------------------------------------
# iterative helpers (no recursion)

def _fold_root(obj, mask):
    """Iterative post-order fold of a container tree into a signature
    integer (children folded before parents; deterministic;
    order-insensitive for dicts - pairs sorted by element hashes)."""

    def children_of(node):
        if _is_sequence(node):
            return list(node)
        if _is_mapping(node):
            flat = []
            for k in sorted(node.keys(), key=hash):
                flat.append(k)
                flat.append(node[k])
            return flat
        if _is_set(node):
            return list(node)
        return None

    stack = [(obj, False)]
    order = []
    while stack:
        node, done = stack.pop()
        if done:
            order.append(node)
            continue
        kids = children_of(node)
        if kids is None:
            order.append(node)
            continue
        stack.append((node, True))
        for k in reversed(kids):
            stack.append((k, False))
    memo = {}
    for node in order:
        kids = children_of(node)
        if kids is None:
            try:
                memo[id(node)] = hash(node) % mask
            except TypeError:
                memo[id(node)] = hash(repr(node)) % mask
            continue
        if _is_sequence(node):
            acc = _sequence_label(node)
            for k in kids:
                acc = (acc * 1000003 + memo[id(k)]) & mask
            memo[id(node)] = acc
        elif _is_mapping(node):
            acc = 3
            items = sorted(
                ((memo[id(k)], memo[id(node[k])]) for k in node.keys()),
                key=lambda p: p[0])
            for kh, vh in items:
                acc = (acc * 1000003 + kh) & mask
                acc = (acc * 1000003 + vh) & mask
            memo[id(node)] = acc
        else:  # set-like (members hashable)
            acc = 4
            for x in sorted(node, key=hash):
                acc = (acc * 1000003 + memo[id(x)]) & mask
            memo[id(node)] = acc
    return memo[id(obj)]


def _deep_eq(a, b):
    """Iterative deep equality: containers compare through the sequence /
    mapping / set protocols (structure labels keep list != tuple and
    set != frozenset); leaves use their own ==."""
    stack = [(a, b)]
    while stack:
        x, y = stack.pop()
        if x is y:
            continue
        if _is_sequence(x):
            if (not _is_sequence(y) or len(x) != len(y)
                    or _sequence_label(x) != _sequence_label(y)):
                return False
            for xi, yi in zip(x, y):
                stack.append((xi, yi))
            continue
        if _is_mapping(x):
            if not _is_mapping(y) or x.keys() != y.keys():
                return False
            for k in x:
                stack.append((x[k], y[k]))
            continue
        if _is_set(x):
            if (not _is_set(y) or _set_label(x) != _set_label(y)
                    or x != y):
                return False
            continue
        if x != y:
            return False
    return True


def _nd_shape(data, limit):
    """Probe the leading dimensions of a duck container up to ``limit``
    levels (atomic-scale guard: deeper nesting is unit content, not a
    data dimension)."""
    shape = []
    cur = data
    for _ in range(limit):
        if not _is_sequence(cur):
            break
        shape.append(len(cur))
        if not cur:
            break
        cur = cur[0]
    return tuple(shape)


def _block_at(data, origin, size, depth):
    """Iteratively collect one window block: leaves at every offset of
    the window grid (row-major), then nested lists assembled from the
    deepest level up - no recursion."""
    total = 1
    for s in size:
        total *= s
    rows = [None] * total
    strides = [1] * depth
    for d in range(depth - 2, -1, -1):
        strides[d] = strides[d + 1] * size[d + 1]
    pos = list(origin)
    for i in range(total):
        node = data
        for d in range(depth - 1):
            node = node[pos[d]]
        rows[i] = node[pos[depth - 1]]
        for d in range(depth - 1, -1, -1):
            pos[d] += 1
            if pos[d] < origin[d] + size[d]:
                break
            pos[d] = origin[d]
    for d in range(depth - 1, -1, -1):
        stride = strides[d]
        groups = total // (stride * size[d])
        for g in range(groups):
            base = g * stride * size[d]
            rows[g * stride] = rows[base:base + size[d]]
    return rows[0]


def window_units(data, local_size=1, step=None, start=None, shape=None,
                 probe_dim=None):
    """N-D data -> window units (duck data; odometer region walk, all
    iterative).  ``probe_dim`` limits the dimension probe (the atomic
    scale): deeper nesting stays unit content.  Default probe_dim=1 -
    the data is treated as a 1-D stream (multidimensional data must pass
    probe_dim explicitly).  Yields one unit per window."""
    import itertools
    if probe_dim is None:
        probe_dim = 1
    nd = _nd_shape(data, probe_dim)
    if not _is_sequence(local_size):
        local_size = (operator.index(local_size),) * len(nd)
    if step is None:
        step = local_size
    elif not _is_sequence(step):
        step = (operator.index(step),) * len(nd)
    if start is None:
        start = (0,) * len(nd)
    if shape is None:
        shape = tuple(max(0, (nd[i] - start[i] - local_size[i]) // step[i]
                          + 1) for i in range(len(nd)))
    for idx in itertools.product(*[range(s) for s in shape]):
        origin = tuple(start[i] + idx[i] * step[i] for i in range(len(nd)))
        yield _block_at(data, origin, local_size, len(nd))


def map_data(data, local_size=1, step=None, mapper=None, output=None,
             buffering=None, start=None, shape=None, probe_dim=None,
             threshold=1, length=None):
    """N-D convenience: window units -> UnitMap flags (streamed - windows
    are fed one at a time, no window list is materialized; appending runs
    through the same mapper is incremental).  Returns the written count
    (or a fresh list when output is None)."""
    if mapper is None:
        mapper = UnitMap(threshold=threshold, length=length)
    mapper.add(window_units(data, local_size, step, start, shape,
                            probe_dim))
    return mapper.put(output=output, buffering=buffering)
