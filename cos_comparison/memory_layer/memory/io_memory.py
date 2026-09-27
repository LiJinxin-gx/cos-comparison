# IOStream as memory carrier: a delegating Memory whose working defaults
# provide MapMemory-style key/value storage on the stream.

import ast

from ...interface.api.io_api import IOStream
from .basememory import Memory
from .inner_memory import Transaction


def _stream_method(name):
    """Build a delegation for an IOStream method:
    func(stream, *args, **kwargs) == getattr(stream, name)(*args, **kwargs),
    so operation slots accept any callable taking the stream first."""
    def func(stream, *args, **kwargs):
        return getattr(stream, name)(*args, **kwargs)
    return func


def _default_capability(mem, kind):
    """Mode-adaptive capability (working default): an opened stream answers
    via its probe (readable / writable); an unopened IOFile is judged from
    its mode string; anything else is assumed capable."""
    stream = mem.memory
    obj = getattr(stream, "obj", None)
    if kind == "read":
        if obj is not None:
            return bool(getattr(obj, "readable", lambda: True)())
        mode = getattr(stream, "mode", None)
        if mode is None:
            return True
        return ("r" in mode) or ("+" in mode)
    if obj is not None:
        return bool(getattr(obj, "writable", lambda: True)())
    mode = getattr(stream, "mode", None)
    if mode is None:
        return True
    return any(ch in mode for ch in "wa+")


def _ensure_open(mem):
    stream = mem.memory
    if stream is None:
        raise RuntimeError("IOStreamMemory: no stream carrier set")
    if getattr(stream, "obj", None) is None:
        mem.open_func(stream)


def _ensure_index(mem):
    if mem.index is None:
        mem.index = {}
        if mem.capability_func(mem, "read"):
            _scan(mem)


def _scan(mem):
    """Rebuild the key -> offset index by walking existing records; a later
    duplicate key overwrites the earlier offset (dict overwrite semantics)."""
    stream = mem.memory
    mem.seek_func(stream, 0)
    index = {}
    while True:
        offset = mem.tell_func(stream)
        record = mem.decode_func(mem)
        if record is None:
            break  # end of stream / truncated or foreign tail
        key, _content = record
        try:
            index[key] = offset
        except TypeError:
            break  # unhashable key in the stream: stop scanning
    mem.index = index


def _write_record(mem, record):
    stream = mem.memory
    offset = mem.tell_func(stream)
    mem.write_func(stream, mem.encode_func(mem, record.key, record.value))
    mem.index[record.key] = offset


# --------------------------------------------------------------------------
# working-default memory slots (MapMemory-aligned key/value: save queues,
# commit appends records, refer recalls)
# --------------------------------------------------------------------------

def _default_save(mem, key, value):
    """Queue save (MapMemory-compatible): visible only after commit;
    returns nothing."""
    try:
        hash(key)
    except TypeError:
        raise TypeError("IOStreamMemory: key must be hashable") from None
    _ensure_open(mem)
    _ensure_index(mem)
    mem.cache.append(Transaction(key, value))


def _default_commit(mem):
    """Write queued records to the stream tail and flush; written records
    stay indexed, the failing tail stays queued."""
    if not mem.cache:
        return 0
    _ensure_open(mem)
    if not mem.capability_func(mem, "write"):
        raise RuntimeError(
            "IOStreamMemory: stream is not writable (commit needs write "
            "access)")
    _ensure_index(mem)
    stream = mem.memory
    mem.seek_func(stream, 0, 2)  # append at the tail whatever the reads
    applied = 0
    try:
        while applied < len(mem.cache):
            _write_record(mem, mem.cache[applied])
            applied += 1
    finally:
        mem.cache = mem.cache[applied:]
        mem.flush_func(stream)
    return applied


def _default_rollback(mem):
    mem.cache = []


def _default_refer(mem, key):
    """Recall committed record ``key`` from the stream (lazy index scan on
    first use); KeyError for unknown / not-yet-committed keys."""
    _ensure_open(mem)
    if not mem.capability_func(mem, "read"):
        raise RuntimeError(
            "IOStreamMemory: stream is not readable (refer needs read "
            "access)")
    _ensure_index(mem)
    offset = mem.index.get(key)
    if offset is None:
        raise KeyError(key)
    stream = mem.memory
    mem.seek_func(stream, offset)
    record = mem.decode_func(mem)
    if record is None:
        raise RuntimeError(
            f"IOStreamMemory: record at offset {offset} is not decodable")
    return record[1]


def _default_initialize(mem):
    """Explicitly (re)scan the stream and rebuild the index; returns the
    number of restored records."""
    _ensure_open(mem)
    mem.index = {}
    if mem.capability_func(mem, "read"):
        _scan(mem)
    return len(mem.index)


def _default_close(mem):
    if mem.close_commit and mem.cache:
        mem.commit()
    stream = mem.memory
    mem.memory = None
    if stream is not None:
        mem.stream_close_func(stream)


# --------------------------------------------------------------------------
# working-default record format.  Text mode stores one repr line per key and
# per value (resolved with ast.literal_eval, so literal values - int / float
# / str / bool / None / tuple / list / dict - round-trip with their type);
# binary mode adds a length-prefixed raw content line after the repr key.
# --------------------------------------------------------------------------

def _default_encode(mem, key, content):
    if mem.binary:
        header = f"{key!r}\n{len(content)}\n".encode()
        return header + content
    return f"{key!r}\n{content!r}\n"


def _default_decode(mem):
    stream = mem.memory
    binary = mem.binary
    key_line = mem.readline_func(stream)
    if not key_line:
        return None
    if binary:
        key_line = key_line.rstrip(b"\r\n").decode("utf-8")
    else:
        key_line = key_line.rstrip("\r\n")
    try:
        key = ast.literal_eval(key_line)
    except (ValueError, SyntaxError):
        return None
    header = mem.readline_func(stream)
    if not header:
        return None
    if binary:
        try:
            length = int(header.rstrip(b"\r\n"))
        except ValueError:
            return None
        content = mem.read_func(stream, length)
        if len(content) != length:
            return None
    else:
        try:
            content = ast.literal_eval(header.rstrip("\r\n"))
        except (ValueError, SyntaxError):
            return None
    return (key, content)


class IOStreamMemory(Memory):
    """IOStream-backed memory - a delegating Memory (every operation is an
    injected slot with a working default) whose defaults store MapMemory
    key/value records on the stream.

    * The memory-level slots (``save`` / ``commit`` / ``rollback`` /
      ``refer`` / ``initialize`` / ``close``) forward to ``save_func`` /
      ``commit_func`` / ...; their signatures are decided by the injected
      functions - the working defaults align with MapMemory (``save(key,
      value)`` / ``refer(key)``, key must be hashable; uncommitted records
      are invisible until ``commit``; re-saving a key overwrites - the
      index keeps the newest offset).
    * Carrier must be an io_api ``IOStream`` (``IOFile`` / ``IOMemory``).
    * The stream-interaction slots (``open_func`` / ``read_func`` /
      ``write_func`` / ``seek_func`` / ``tell_func`` / ``flush_func`` /
      ``readline_func`` / ``stream_close_func`` / ``capability_func``)
      transcribe the carrier methods by default; ``encode_func`` /
      ``decode_func`` are the record-format extension points - the working
      default is ``repr(key)`` + length-prefixed content (``binary=True``
      counts bytes), resolved with ``ast.literal_eval`` so hashable literal
      keys round-trip through a restart.  Custom record semantics are
      injected by replacing the memory-level slots.
    * Mode-adaptive: readable streams support ``refer`` (the index is
      scanned lazily on first use, existing records recalled and new keys
      indexed after them); write-only streams skip the scan; read-only
      streams reject ``commit``.
    """
    __slots__ = (
        "binary",
        "cache",
        "capability_func",
        "close_commit",
        "decode_func",
        "encode_func",
        "flush_func",
        "index",
        "open_func",
        "read_func",
        "readline_func",
        "seek_func",
        "stream_close_func",
        "tell_func",
        "write_func",
    )
    def __init__(self, stream=None, binary=False, close_commit=False,
                 open_func=None, read_func=None, write_func=None,
                 seek_func=None, tell_func=None, flush_func=None,
                 readline_func=None, stream_close_func=None,
                 capability_func=None, encode_func=None, decode_func=None,
                 init_func=None, save_func=None, commit_func=None,
                 rollback_func=None, refer_func=None, close_func=None):
        if stream is not None and not isinstance(stream, IOStream):
            raise TypeError(
                "IOStreamMemory: carrier must be an io_api IOStream "
                "(IOFile / IOMemory)")
        self.binary = binary
        self.cache = []
        self.close_commit = close_commit
        self.index = None  # None = not scanned yet (lazy restore)
        # stream-interaction slots: working defaults transcribe the carrier
        self.open_func = open_func if open_func is not None else _stream_method("open")
        self.read_func = read_func if read_func is not None else _stream_method("read")
        self.write_func = write_func if write_func is not None else _stream_method("write")
        self.seek_func = seek_func if seek_func is not None else _stream_method("seek")
        self.tell_func = tell_func if tell_func is not None else _stream_method("tell")
        self.flush_func = flush_func if flush_func is not None else _stream_method("flush")
        self.readline_func = readline_func if readline_func is not None else _stream_method("readline")
        self.stream_close_func = stream_close_func if stream_close_func is not None else _stream_method("close")
        self.capability_func = capability_func if capability_func is not None else _default_capability
        self.encode_func = encode_func if encode_func is not None else _default_encode
        self.decode_func = decode_func if decode_func is not None else _default_decode
        # memory-level slots: working defaults give MapMemory-style
        # key/value behaviour; every signature below is a default only
        init_func = init_func if init_func is not None else _default_initialize
        save_func = save_func if save_func is not None else _default_save
        commit_func = commit_func if commit_func is not None else _default_commit
        rollback_func = rollback_func if rollback_func is not None else _default_rollback
        refer_func = refer_func if refer_func is not None else _default_refer
        close_func = close_func if close_func is not None else _default_close
        super().__init__(memory=stream, init_func=init_func,
                         save_func=save_func, commit_func=commit_func,
                         rollback_func=rollback_func, refer_func=refer_func,
                         close_func=close_func)

    def __getitem__(self, key):
        """Mapping-protocol recall (MemoryWrap-compatible)."""
        return self.refer(key)

    def keys(self):
        _ensure_index(self)
        return self.index.keys()

    def __len__(self):
        _ensure_index(self)
        return len(self.index)

    def __contains__(self, key):
        _ensure_index(self)
        return key in self.index


"""
Explicit public exports (prevents import-star namespace pollution).
"""
__all__ = (
    "IOStreamMemory",
)
