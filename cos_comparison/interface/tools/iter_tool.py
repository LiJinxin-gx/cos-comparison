"""
Iterator tools: chained/zipped/windowed/flattened iteration and lazy
map/filter/group/batch helpers (state-machine driven, no recursion).
"""

from collections.abc import Iterable

__all__ = (
    "IterChain",
    "IterFlatten",
    "IterWindow",
    "IterWrap",
    "IterZip",
    "iter_batch",
    "iter_cycle",
    "iter_filter",
    "iter_group",
    "iter_map",
)


class IterWrap:
    """Delegate iterator over an iterable with injected slots."""
    __slots__ = ("iter_func", "iterable", "next_func")
    def __init__(self, iterable=(), iter_func=None, next_func=None):
        self.iterable = iterable
        self.iter_func = iter_func if iter_func is not None else iter
        self.next_func = next_func if next_func is not None else next
    def __iter__(self):
        source = self.iter_func(self.iterable)
        next_func = self.next_func
        while True:
            try:
                yield next_func(source)
            except StopIteration:
                return


class IterChain:
    """Chain several iterables; explicit state machine, no recursion."""
    __slots__ = ("count", "current", "index", "sources")
    def __init__(self, *sources):
        self.sources = list(sources)
        self.count = len(self.sources)
        self.index = 0
        self.current = iter(self.sources[0]) if self.sources else iter(())
    def __iter__(self):
        return self
    def __next__(self):
        while True:
            try:
                return next(self.current)
            except StopIteration:
                if self.index + 1 >= self.count:
                    raise
                self.index += 1
                self.current = iter(self.sources[self.index])


class IterZip:
    """Zip iterables; stops at the shortest one (state machine)."""
    __slots__ = ("active", "iters")
    def __init__(self, *iterables):
        self.iters = [iter(it) for it in iterables]
        self.active = len(self.iters) > 0
    def __iter__(self):
        return self
    def __next__(self):
        if not self.active:
            raise StopIteration
        out = []
        for it in self.iters:
            try:
                out.append(next(it))
            except StopIteration:
                self.active = False
                raise
        return tuple(out)


class IterWindow:
    """Sliding-window iteration over an iterable (step may exceed size:
    whole windows are then skipped between yields)."""
    __slots__ = ("it", "size", "step", "window")
    def __init__(self, iterable, size, step=1):
        if size < 1:
            raise ValueError("IterWindow size must be a positive integer")
        if step < 1:
            raise ValueError("IterWindow step must be a positive integer")
        self.it = iter(iterable)
        self.size = size
        self.step = step
        self.window = []
    def __iter__(self):
        return self
    def __next__(self):
        while len(self.window) < self.size:
            try:
                self.window.append(next(self.it))
            except StopIteration:
                break
        if len(self.window) < self.size:
            raise StopIteration
        out = list(self.window)
        if self.step < self.size:
            self.window = self.window[self.step:]
        else:
            self.window = []
            for _ in range(self.step - self.size):
                try:
                    next(self.it)
                except StopIteration:
                    break
        return out


class IterFlatten:
    """Flatten nested iterables; explicit stack, no recursion."""
    __slots__ = ("done", "root", "stack")
    def __init__(self, iterable):
        self.root = iterable
        self.stack = []
        self.done = False
    def __iter__(self):
        return self
    def __next__(self):
        if not self.stack:
            if self.done:
                raise StopIteration
            self.stack.append(iter(self.root))
            self.done = True
        while self.stack:
            it = self.stack[-1]
            try:
                item = next(it)
            except StopIteration:
                self.stack.pop()
                continue
            if isinstance(item, Iterable) and not isinstance(
                    item, (str, bytes)):
                self.stack.append(iter(item))
            else:
                return item
        raise StopIteration


def iter_map(func, iterable):
    """Lazy mapping."""
    for item in iterable:
        yield func(item)


def iter_filter(func, iterable):
    """Lazy filtering."""
    for item in iterable:
        if func(item):
            yield item


def iter_group(iterable, key_func):
    """Yield (key, [items]) for consecutive equal keys."""
    current_key = object()
    group = []
    for item in iterable:
        key = key_func(item)
        if group and key != current_key:
            yield current_key, group
            group = []
        current_key = key
        group.append(item)
    if group:
        yield current_key, group


def iter_batch(iterable, size):
    """Yield batches of fixed size."""
    if size < 1:
        raise ValueError("iter_batch size must be a positive integer")
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_cycle(*iterables):
    """Yield items cycling through iterables forever (caller breaks);
    sources are re-iterated on exhaustion, empty ones are dropped."""
    sources = [iter(it) for it in iterables]
    if not sources:
        return
    index = 0
    while True:
        try:
            item = next(sources[index])
        except StopIteration:
            fresh = iter(iterables[index])
            try:
                item = next(fresh)
            except StopIteration:
                del sources[index]
                iterables = iterables[:index] + iterables[index + 1:]
                if not sources:
                    return
                index %= len(sources)
                continue
            sources[index] = fresh
        yield item
        index = (index + 1) % len(sources)
