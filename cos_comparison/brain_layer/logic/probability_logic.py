"""
Probabilistic logic system for uncertain reasoning.
Implements event classes, probability operations and Bayesian inference primitives.
"""

# Relative probability axioms (no absolute probability; all values are
# conditional P(X|C), global_event is the relative benchmark):
#   A1  relativism: every probability is P(X|C)
#   A2  reflexivity: P(X|X) = 1
#   A3  chain rule (Markov approximation along dependency arcs):
#       P(A|C) = prod_i P(v_i | v_{i-1})  for C=v0 -> ... -> vk=A
#   A4  Bayes duality (reference-invariant ratio P(A)/P(B) by the symmetry
#       of the joint P(AB)): P(A|B) = P(B|A) * P(A|C) / P(B|C)
#   A5  union (exact, commutative): P(A+B|C) = P(A|C) + P(B|C) - P(AB|C)
# Absolute forms degenerate from C = global_event (A3/A4 -> classic rules).

#----------- event class -----------
class UnionEvent(frozenset): #A+B+...
    __slots__ = ("name",)
    def __init__(self,*event):
        self.name = "UnionEvent"
    def __new__(cls,*event):
        return super().__new__(cls,event)
    def __eq__(self,other):
        # Content equality within the same event kind only: a union must not
        # equal an intersection (or a plain frozenset) even with equal members.
        if type(self) is not type(other):
            return False
        return frozenset.__eq__(self,other)
    def __ne__(self,other):
        return not self == other
    def __hash__(self):
        return hash(("Union",super().__hash__()))

class IntersectionEvent(frozenset): #AB...
    __slots__ = ("name",)
    def __init__(self,*event):
        self.name = "IntersectionEvent"
    def __new__(cls,*event):
        return super().__new__(cls,event)
    def __eq__(self,other):
        # Content equality within the same event kind only.
        if type(self) is not type(other):
            return False
        return frozenset.__eq__(self,other)
    def __ne__(self,other):
        return not self == other
    def __hash__(self):
        return hash(("Intersection",super().__hash__()))

class GlobalEvent: 
    # In face,there is not any absolute global event,the class make a relative benchmark in a context.
    def __repr__(self):
        return f"<GlobalEvent : id={id(self)} , hash={hash(self)}>"

global_event = GlobalEvent()
#It provide a Singleton to make a relative benchmark in module.

#------------ event_bind ---------------
class event_bind:
    #We should know that proibability of all event is relative.
    __slots__=("name","event","binds","id")
    def __init__(self,name="",event=None):
        self.name=name if name else  str(event)
        self.event=event
        self.binds = {} #It stores Conditional Probability , like P(A | B)
        self.id=id(self)
    def __iter__(self):
        for bind in self.binds:
            yield (self.event,bind,self.binds[bind])
    def get_bind(self,event):
        return self.binds[event]
    def bind_exist(self,event):
        if event in self.binds:
            return True
        else:
            return False
    def bind(self,event,relative_p):
        if 0<=relative_p<=1 :
            self.binds[event] = relative_p #if it has existed,overwrite it.
            return 0
        else:
            raise ValueError("effectless args")
    def unbind(self,event):
        try:
            del self.binds[event]
            return 0
        except:
            return 1

#--------- context ----------
from ...core import no_done
from ...interface.tools.math_tool.topology import DirectedGraph, shortest_path_between

class event_context:
    #logic context container and processer to isolate different logical context.
    #Protocol-style (function delegation) context: every operation is a
    #delegated slot (init_func / add_func / probability_func), mirroring
    #Logic_context / Memory / Communicate.  The default implementations handle
    #the concrete behaviour; injecting a function replaces it entirely.
    #Default binds construction uses the protocol-class container (EventBinds)
    #so the cached-graph / statistics engine is ready out of the box; passing
    #an explicit binds (plain dict or protocol container) keeps the stateless
    #default slots instead.
    __slots__ = ("name","binds","extension","init_func","add_func","probability_func")
    def __init__(self,name="",binds=None,init_func=None,add_func=None,probability_func=None):
        self.name = name
        self.extension = None #It provides extension to use by callback functions.
        if binds is None:
            # Default object construction (no truthiness substitution): the
            # protocol-class container carries the engine state, and the
            # default slots are its class methods (no instance construction).
            binds = EventBinds()
            init_func = init_func if init_func is not None else default_context_init
            add_func = add_func if add_func is not None else EventBinds.add_bind
            probability_func = probability_func if probability_func is not None else EventBinds.resolve
        self.binds = binds
        self.init_func = init_func if init_func is not None else no_done
        self.add_func = add_func if add_func is not None else default_add_bind
        self.probability_func = probability_func if probability_func is not None else default_probability_func
    def __iter__(self):
        for bind in self.binds:
            yield (*bind,self.binds[bind])
    def initialize(self,*args,**kwargs):
        return self.init_func(self.binds,*args,**kwargs)
    def add_bind(self,binds):
        return self.add_func(self.binds,binds)
    def bind_probability(self,A,B=global_event):
        """
        A : event or tuple ,B : event or tuple (default value : global_event) , return value : P(A|B)
        Delegated to the probability slot (default: chain-rule / rigorous
        resolution over the dependency graph; exact hits first).
        """
        return self.probability_func(self,A,B)


def _as_binds(context):
    """Duck protocol: a context exposing `.binds`, or a bare binds container."""
    return getattr(context, "binds", context)


def default_add_bind(binds, items):
    """add_func slot default: store (outcome, condition, p) triples."""
    for bind in items:
        binds[(bind[0], bind[1])] = bind[2]


def default_context_init(binds, *args, **kwargs):
    """init_func slot default: warm the graph cache if present (duck)."""
    ensure = getattr(binds, "_ensure_graph", None)
    if ensure is not None:
        ensure()
    return 0


def _build_graph(binds, graph_factory=DirectedGraph):
    """Dependency graph: arc condition -> outcome per binding."""
    g = graph_factory()
    for (outcome, condition) in binds:
        g.add_edge(condition, outcome)
    return g


def _reference_candidates(binds):
    """Benchmark first, then binding conditions (insertion order, unique)."""
    seen = {global_event}
    candidates = [global_event]
    for (outcome, condition) in binds:
        if condition not in seen:
            seen.add(condition)
            candidates.append(condition)
    return candidates


def _resolve(binds, A, B, strict=False, graph=None, graph_factory=DirectedGraph,
             stats=None):
    """
    Two-stage resolution of P(A|B):
      direct hit (reflexive / binding)          -> exact
      strict: Bayes P(A|B) = P(B|A)*P(A|C)/P(B|C) (A4; direct refs only)
      fallback: chain-rule product on shortest path (A3)
    """
    if A == B:
        return 1.0
    if (A, B) in binds:
        if stats is not None:
            stats["hits"] += 1
        return binds[(A, B)]
    if strict:
        pba = binds.get((B, A))
        if pba is not None:
            for C in _reference_candidates(binds):
                if C in (A, B):
                    continue
                pa = binds.get((A, C))
                pb = binds.get((B, C))
                if pa is not None and pb not in (None, 0):
                    if stats is not None:
                        stats["bayes"] += 1
                    return pba * pa / pb
    if graph is None:
        graph = _build_graph(binds, graph_factory)
    path = shortest_path_between(graph, B, A)
    if path is None:
        if stats is not None:
            stats["miss"] += 1
        return 0.0
    p = 1.0
    for i in range(1, len(path)):
        seg = binds.get((path[i], path[i - 1]))
        if seg is None:
            if stats is not None:
                stats["miss"] += 1
            return 0.0
        p *= seg
    if stats is not None:
        stats["chain"] += 1
    return p


def chain_probability(binds, A, B, graph=None, graph_factory=DirectedGraph):
    """P(A|B) by direct hit or chain-rule product (A3); 0.0 when unresolved."""
    return _resolve(binds, A, B, graph=graph, graph_factory=graph_factory)


def default_probability_func(context, A, B, graph_factory=DirectedGraph):
    """Default probability slot: exact hits first, chain fallback (A3)."""
    return _resolve(_as_binds(context), A, B, graph_factory=graph_factory)


def strict_probability_func(context, A, B, graph_factory=DirectedGraph):
    """Rigorous slot: exact hit, then relative Bayes (A4, direct refs only),
    then chain fallback (A3)."""
    return _resolve(_as_binds(context), A, B, strict=True,
                    graph_factory=graph_factory)


def chain_intersection(context, A, B, C):
    """P(AB|C) ~= P(A|B)*P(B|C)   (chain rule, Markov approximation)."""
    return (default_probability_func(context, A, B)
            * default_probability_func(context, B, C))


def union_probability(context, A, B, C):
    """P(A+B|C) = P(A|C) + P(B|C) - P(AB|C)   (A5, exact)."""
    return (default_probability_func(context, A, C)
            + default_probability_func(context, B, C)
            - chain_intersection(context, A, B, C))


def consistency_diagnostic(context, A, B, C):
    """Both Bayes decompositions; equal iff P(C|A) == P(C|B) (diagnostic)."""
    left = default_probability_func(context, B, A) * default_probability_func(context, A, C)
    right = default_probability_func(context, A, B) * default_probability_func(context, B, C)
    return {"left": left, "right": right, "consistent": left == right}


class EventContextProtocol:
    """Protocol-class engine (deploy form): state (binds, cached graph, stats,
    strict switch) + slot methods; `deploy` wires it onto a context uniformly."""

    __slots__ = ("binds", "strict", "_graph", "_graph_ver", "_binds_len", "stats")

    def __init__(self, binds=None, strict=False):
        self.binds = binds if binds is not None else {}
        self.strict = strict
        self._graph = None
        self._graph_ver = -1
        self._binds_len = len(self.binds)
        self.stats = {"adds": 0, "hits": 0, "bayes": 0, "chain": 0, "miss": 0}

    def _sync_binds(self, binds):
        """Follow the container the context exposes (duck protocol)."""
        if binds is not None and binds is not self.binds:
            self.binds = binds
            self._binds_len = len(binds)
            self._graph_ver = -1

    def _ensure_graph(self):
        """Cached graph, rebuilt when the binds grew (len check) or invalidated."""
        if len(self.binds) != self._binds_len or self._graph_ver < 0:
            self._graph = _build_graph(self.binds)
            self._graph_ver += 1
            self._binds_len = len(self.binds)
        return self._graph

    def add_bind(self, binds, items):
        """add_func slot: store (outcome, condition, p) triples."""
        self._sync_binds(binds)
        default_add_bind(self.binds, items)
        self._graph_ver = -1
        self._binds_len = len(self.binds)
        self.stats["adds"] += 1

    def resolve(self, context, A, B):
        """probability_func slot: two-stage resolve, cached graph + stats."""
        self._sync_binds(getattr(context, "binds", None))
        return _resolve(self.binds, A, B, strict=self.strict,
                        graph=self._ensure_graph(), stats=self.stats)

    def deploy(self, context):
        """Wire slot methods onto the context; attach self to extension."""
        self._sync_binds(getattr(context, "binds", None))
        context.binds = self.binds
        context.add_func = self.add_bind
        context.probability_func = self.resolve
        context.extension = self
        return context


class EventBinds(dict):
    """Binds-as-engine protocol class: state on the binds container itself;
    class methods injected into the slots (no separate engine instance)."""

    def __bool__(self):
        return True  # never dropped by legacy truthiness substitution

    def __init__(self, *args, strict=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.strict = strict
        self._graph = None
        self._graph_ver = -1
        self._len = len(self)
        self.stats = {"adds": 0, "hits": 0, "bayes": 0, "chain": 0, "miss": 0}

    def add_bind(self, items):
        """add_func slot (self signature: the binds instance is self)."""
        for bind in items:
            self[(bind[0], bind[1])] = bind[2]
        self._graph_ver = -1
        self._len = len(self)
        self.stats["adds"] += 1

    def _ensure_graph(self):
        if len(self) != self._len or self._graph_ver < 0:
            self._graph = _build_graph(self)
            self._graph_ver += 1
            self._len = len(self)
        return self._graph

    def resolve(context, A, B):
        """probability_func slot (class-body, no self; state via context.binds)."""
        binds = getattr(context, "binds")
        return _resolve(binds, A, B, strict=binds.strict,
                        graph=binds._ensure_graph(), stats=binds.stats)

"""
Explicit public exports (prevents import-star namespace pollution).
"""
__all__ = (
    "UnionEvent",
    "IntersectionEvent",
    "GlobalEvent",
    "event_bind",
    "event_context",
    "default_add_bind",
    "default_context_init",
    "chain_probability",
    "default_probability_func",
    "strict_probability_func",
    "chain_intersection",
    "union_probability",
    "consistency_diagnostic",
    "EventContextProtocol",
    "EventBinds",
)
