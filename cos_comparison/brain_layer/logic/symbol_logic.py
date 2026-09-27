"""
Symbolic rule-chain reasoning over an implication fragment.

Rules (binds) are reason -> result arcs.  The judge decides the correctness
of a full statement (a, b[, limit[, is_true]]): a claim of truth must be
derivable (reflexive-transitive closure of the available rules, BFS); a
claim of non-truth must not be.  limit bounds both endpoints (No_limit
contains everything).  The verdict is determined within the rule library:
correct -> sure_true, otherwise Logic.SURE (determined not-true) - matching
the probabilistic sibling, where underivable pairs resolve to 0.0.

Logic is a 2-bit flag set: Logic(0) unset, Logic.TRUE true (not yet
certain), Logic.SURE sure/determined (alone: determined not-true),
sure_true = TRUE|SURE (surely true); the boolean projection means "carries
TRUE" (use is_true / is_sure / is_uncertain).  Rule status is truth-gating:
only rules carrying Logic.TRUE participate.  Event identity follows the
objects' own equality (keep kinds type-consistent, reuse one object per
proposition).
"""

# ---- imports ----
from enum import Flag,auto
from ...core import no_done
from ...interface.tools.math_tool.topology import DirectedGraph, shortest_path_between

# ---- value constants ----
class Logic(Flag):
    TRUE = auto()
    SURE = auto()
    
Logic_true = Logic.TRUE
Logic_sure = Logic.SURE

sure_true = Logic.TRUE | Logic.SURE


def is_true(status):
    """Logic projection: the status carries Logic.TRUE (true)."""
    return bool(status & Logic_true)


def is_sure(status):
    """Logic projection: the status is determined (sure)."""
    return bool(status & Logic_sure)


def is_uncertain(status):
    """Logic projection: not determined (unset or merely true)."""
    return not (status & Logic_sure)

class No_limit:
    def __contains__(self,other):
        return True

# ---- type support ----
class Variable:
    """Truth-value variable: equality is truth-value based (two variables
    with the same value are the same event, whatever their names)."""
    __slots__ = ("name","value")
    def __init__(self,name,value=None):
        self.name = name
        self.value = value
    def __eq__(self,other):
        if isinstance(other, Variable):
            if self.value is other.value:
                return True  # identity: reflexive even for NaN values
            return self.value==other.value
        return NotImplemented
    def __hash__(self):
        try:
            return hash(self.value)
        except TypeError:
            # unhashable values: repr fallback (hashable values unchanged)
            return hash(repr(self.value))

# ---- errors ----
class UnsupportedError(Exception):
    pass

class LogicError(Exception):
    pass

# ---- propositions and rules ----
class Atomic_proposition:
    __slots__=("logic_args","arg_names","status") 
    def __init__(self,*logic_args,arg_names=None,status=sure_true):
        # arg_names=None -> positional access (a_list = *a);
        # else -> named access (a_dict = **a); names must be unique and
        # cover the arguments one-to-one.
        self.logic_args = list(logic_args)
        if arg_names is not None:
            arg_names = tuple(arg_names)
            if len(arg_names) != len(self.logic_args):
                raise ValueError(
                    "arg_names length does not match the argument count")
            if len(set(arg_names)) != len(arg_names):
                raise ValueError("arg_names entries must be unique")
        self.arg_names = arg_names
        self.status = status
    def __bool__(self)->bool:
        return bool(self.status & Logic_true)
    def __iter__(self):
        for logic_arg in self.logic_args:
            yield logic_arg
    def __len__(self):
        return len(self.logic_args)
    def keys(self):
        if self.arg_names is None:
            raise UnsupportedError("It does not support the operation because itdid not use 'arg_names'.")
        return tuple(self.arg_names)
    def __getitem__(self,key):
        if self.arg_names is None:
            raise UnsupportedError("It does not support the operation because itdid not use 'arg_names'.")
        else:
            if key in self.arg_names:
                for i,name in enumerate(self.arg_names):
                    if name==key:
                        return self.logic_args[i]
            else:
                raise ValueError(f"It does not have the arg '{key}'.")
    def __setitem__(self,key,value):
        if self.arg_names is None:
            raise UnsupportedError("It does not support the operation because itdid not use 'arg_names'.")
        else:
            if key in self.arg_names:
                for i,name in enumerate(self.arg_names):
                    if name==key:
                        self.logic_args[i] = value
            else:
                raise ValueError(f"It does not have the arg '{key}'.")

class Logic_bind:
    __slots__=("reason","result","limit","status")
    def __init__(self,reason,result,limit=None,status=sure_true):
        self.reason=reason
        self.result=result
        self.limit=limit if limit is not None else No_limit()
        self.status=status
    def __bool__(self)->bool:
        return bool(self.status & Logic_true)
    def __iter__(self):
        for logic in self.__slots__:
            yield getattr(self,logic)
    def keys(self):
        return self.__slots__
    def __len__(self):
        return len(self.__slots__)
    def __getitem__(self,key):
        if key in self.__slots__:
            return getattr(self,key)
        raise KeyError(key)
    def __setitem__(self,key,value):
        return setattr(self,key,value)
    def __contains__(self,key):
        return key in self.__slots__

class Logic_context:
    __slots__ = ("name","binds","extension","init_func","add_func","pop_func","judge_func")
    def __init__(self,name="",binds=None,
                 init_func=None,
                 add_func=None,
                 pop_func=None,
                 judge_func=None):
        self.name = name
        self.extension = None # extension slot usable by callbacks.
        self.binds = binds if binds is not None else []
        self.init_func = init_func if init_func is not None else no_done
        self.add_func = add_func if add_func is not None else no_done
        self.pop_func = pop_func if pop_func is not None else no_done
        self.judge_func = judge_func if judge_func is not None else default_judge_func
    def initialize(self,*args,**kwargs):
        return self.init_func(self.binds,*args,**kwargs)
    def add(self,logic_bind,**kwargs):
        return self.add_func(self.binds,logic_bind,**kwargs)
    def pop(self,logic_bind,**kwargs):
        return self.pop_func(self.binds,logic_bind,**kwargs)
    def logic_judge(self, *args_bind, **kwargs_bind):
        """Protocol-style judge: the bound statement is unpacked and
        delegated (*args_bind, **kwargs_bind); the default judge decides
        the correctness of the full statement (see default_judge_func)."""
        return self.judge_func(self.binds, *args_bind, **kwargs_bind)


# ---- judge helpers ----
def _as_binds(context):
    """Duck protocol: a context exposing .binds, or a bare binds container."""
    return getattr(context, "binds", context)


_MISSING = object()


def _bind_field(bind, field):
    """Duck rule-field access (mapping get first, then attribute);
    TypeError when the bind lacks the field."""
    get = getattr(bind, "get", None)
    if callable(get):
        value = get(field, _MISSING)
        if value is not _MISSING:
            return value
    try:
        return getattr(bind, field)
    except AttributeError:
        raise TypeError("bind object lacks the %r field" % field) from None


def _rule_status(bind):
    """Duck rule availability: attribute `status` or mapping 'status' key;
    None when the rule carries no status (always available)."""
    get = getattr(bind, "get", None)
    if callable(get):
        value = get("status", _MISSING)
        if value is not _MISSING:
            return value
    return getattr(bind, "status", None)


def default_judge_func(context, *statement, **kwargs_bind):
    """Judge the correctness of a full statement.

    A statement is the sequence (a, b[, limit[, is_true]]) - the unpacked
    Logic_bind slots; (a, b) alone keeps the plain derivation query, and
    a/b may alternatively arrive as kwargs.  The statement is CORRECT when
    its claim agrees with the available rules (rules without Logic.TRUE
    status are excluded): a truth claim (is_true carries Logic.TRUE) must
    be derivable (a == b or reachable) within limit; a falsehood claim
    must not be.      Returns sure_true when correct, Logic.SURE otherwise (determined
    not-true within the rule library).  With return_path=True a correct
    truth claim returns its proving rule list ([] when a == b), anything
    else returns None; a path_func path without supporting rules raises
    LogicError.  graph_factory / path_func are injectable via kwargs."""
    binds = _as_binds(context)
    if len(statement) >= 2:
        a, b = statement[0], statement[1]
        limit = statement[2] if len(statement) >= 3 else None
        is_true = statement[3] if len(statement) >= 4 else sure_true
    elif "a" in kwargs_bind and "b" in kwargs_bind:
        a, b = kwargs_bind["a"], kwargs_bind["b"]
        limit = kwargs_bind.get("limit")
        is_true = kwargs_bind.get("is_true", sure_true)
    else:
        return Logic.SURE
    if limit is None:
        limit = No_limit()
    return_path = kwargs_bind.get("return_path", False)
    wants_true = bool(is_true & Logic_true)

    if not ((a in limit) and (b in limit)):
        return None if return_path else Logic.SURE  # statement out of bounds

    graph_factory = kwargs_bind.get("graph_factory", DirectedGraph)
    path_func = kwargs_bind.get("path_func", shortest_path_between)
    if a == b:
        proof = []
    else:
        g = graph_factory()
        edges = []
        for bind in binds:
            status = _rule_status(bind)
            if status is not None and not (status & Logic_true):
                continue  # rule marked unavailable (no logical claim)
            reason = _bind_field(bind, "reason")
            result = _bind_field(bind, "result")
            edges.append((reason, result, bind))
            g.add_edge(reason, result)
        path = path_func(g, a, b)
        if path is None:
            proof = None
        else:
            proof = []
            for i in range(1, len(path)):
                matched = None
                for reason, result, bind in edges:
                    if reason == path[i - 1] and result == path[i]:
                        matched = bind
                        break
                if matched is None:
                    raise LogicError(
                        "path_func returned a segment without a supporting "
                        "rule")
                proof.append(matched)

    correct = (proof is not None) == wants_true
    if not return_path:
        return sure_true if correct else Logic.SURE
    return proof if (correct and wants_true) else None


# ---- exports ----
__all__ = (
    "Logic",
    "Logic_true",
    "Logic_sure",
    "sure_true",
    "is_true",
    "is_sure",
    "is_uncertain",
    "No_limit",
    "Variable",
    "UnsupportedError",
    "LogicError",
    "Atomic_proposition",
    "Logic_bind",
    "Logic_context",
    "default_judge_func",
)
