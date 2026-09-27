"""Unit map: fold variable-length runs of repeated units into fixed-length
real elements (symbolic data to the tensor domain)."""

import operator

__all__ = ("UnitMap", "map_data", "window_units")

_DEFAULT_START = 0.0
_DEFAULT_STEP = 1.0
_SIG_MASK = (1 << 61) - 1


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

    __slots__ = ("_counts", "_flags", "_last_out", "_n", "_out_pos",
                 "_pending", "_plain_sig", "_queued", "start", "step",
                 "table")

    def __init__(self, start=_DEFAULT_START, step=_DEFAULT_STEP):
        self.start = float(start)
        self.step = float(step)
        self.table = {}          # hashable obj -> flag
        self._plain_sig = {}     # content signature -> [[obj, flag, ...]]
        self._flags = {}         # flag -> obj (reverse)
        self._counts = {}        # hashable obj -> [count, runs]
        self._pending = None     # (obj, run_length) not closed yet
        self._queued = []        # closed runs not yet output: (obj, len)
        self._n = 0              # total unique units (flag sequence)
        self._last_out = None
        self._out_pos = 0

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
        if self._hashable(obj):
            flag = self.table.get(obj)
            if flag is not None:
                return flag
        recs = self._sig_bucket(obj)
        if recs:
            for rec in recs:
                if _deep_eq(rec[0], obj):
                    return rec[1]
        return None

    def decode(self, flags):
        """Reverse query over the content table (compatibility surface):
        a flag -> unit list (unknown flags decode to None)."""
        out = []
        for flag in flags:
            out.append(self._flags.get(flag))
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
        self._flags[flag] = obj
        return flag

    def _bump(self, obj, length):
        """Record run statistics: total elements and run count."""
        if self._hashable(obj):
            rec = self._counts.get(obj)
            if rec is None:
                self._counts[obj] = [length, 1]
            else:
                rec[0] += length
                rec[1] += 1
            return
        recs = self._sig_bucket(obj)
        if recs:
            for rec in recs:
                if _deep_eq(rec[0], obj):
                    rec[2] += length
                    rec[3] += 1
                    return

    def _stat_rec(self, obj):
        """The statistic record [count, runs] of a unit, or None."""
        if self._hashable(obj):
            return self._counts.get(obj)
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
        pend = self._pending
        if pend is not None and _deep_eq(pend[0], unit):
            self._pending = (unit, pend[1] + 1)
            return
        if pend is not None:
            self._close()
        if self.flag_of(unit) is None:  # first sight registers the unit
            self._register(unit)
        self._pending = (unit, 1)

    def _close(self):
        obj, length = self._pending
        self._queued.append((obj, length))
        self._bump(obj, length)
        self._pending = None

    # -- output ----------------------------------------------------------

    def put(self, output=None, buffering=None):
        """Write the folded elements.  ``output``: duck container with
        ``append`` (preferred) or ``__setitem__`` (sequence protocol);
        None returns a fresh list.  Position is remembered per output
        object (file pointer); ``buffering`` limits this call's elements
        (None = unlimited) - the rest stays queued.  The pending run at
        the input tail is closed by the put.  Returns the written count
        (or the fresh list when output is None)."""
        if self._pending is not None:
            self._close()
        fresh = output is None
        if fresh:
            output = []
        self._out_pos = 0 if output is not self._last_out \
            else self._out_pos
        written = 0
        for obj, _length in self._queued:
            if buffering is not None and written >= buffering:
                break
            flag = self.flag_of(obj)
            if flag is None:
                flag = self._register(obj)
            if self._write(output, flag):
                written += 1
        if fresh:
            self._queued = self._queued[written:]
            return output
        if written:
            del self._queued[:written]
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

    # -- state -----------------------------------------------------------

    def get_state(self):
        """Export the state: table/signature records, statistics, output
        pointer, queued and pending runs (importable via set_state)."""
        plain = {sig: [list(r) for r in recs]
                 for sig, recs in self._plain_sig.items()}
        return {
            "start": self.start, "step": self.step,
            "table": dict(self.table), "plain_sig": plain,
            "flags": dict(self._flags),
            "counts": dict(self._counts),
            "pending": self._pending,
            "queued": list(self._queued),
            "out_pos": self._out_pos,
            "n": self._n,
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
        self._n = state.get("n", len(self.table)
                            + sum(len(v) for v in
                                  self._plain_sig.values()))
        self._last_out = None
        return 0

    def clear(self):
        """Clear everything (table, statistics, runs, output pointer)."""
        self.table = {}
        self._plain_sig = {}
        self._flags = {}
        self._counts = {}
        self._pending = None
        self._queued = []
        self._last_out = None
        self._out_pos = 0
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
             buffering=None, start=None, shape=None, probe_dim=None):
    """N-D convenience: window units -> UnitMap flags (streamed - windows
    are fed one at a time, no window list is materialized; appending runs
    through the same mapper is incremental).  Returns the written count
    (or a fresh list when output is None)."""
    if mapper is None:
        mapper = UnitMap()
    mapper.add(window_units(data, local_size, step, start, shape,
                            probe_dim))
    return mapper.put(output=output, buffering=buffering)
