"""
Tools for solving topology problems.
"""

import operator
from collections import deque

from ..context_tool import VoidContext as default_lock


def _bfs(neighbors, src, dst):
    """BFS over a neighbors callable (unknown vertices may raise KeyError).
    Returns [src, ..., dst] or None."""
    if src == dst:
        return [src]
    prev = {src: None}
    queue = deque([src])
    while queue:
        v = queue.popleft()
        try:
            nbrs = neighbors(v)
        except KeyError:
            continue
        for w in nbrs:
            if w not in prev:
                prev[w] = v
                if w == dst:
                    path = [w]
                    while path[-1] != src:
                        path.append(prev[path[-1]])
                    path.reverse()
                    return path
                queue.append(w)
    return None


def shortest_path_between(graph, src, dst):
    """BFS shortest path over any object exposing neighbors(v);
    returns [src, ..., dst] or None."""
    return _bfs(graph.neighbors, src, dst)

def Euler_characteristic_compute_by_cell(cell_list):
    factor = 1
    Euler_characteristic = 0
    for cell in cell_list:
        # integer protocol (__index__); bool stays excluded (ambiguous)
        if isinstance(cell, bool) or not hasattr(cell, "__index__"):
            raise TypeError("Cell must be a positive integer.")
        cell = operator.index(cell)
        if cell <= 0:
            raise ValueError("Cell must be a positive integer.")
        Euler_characteristic += cell * factor
        factor *= -1
    return Euler_characteristic

class Graph:
    """Undirected multigraph with parallel edges; union-find maintains
    connected components for cycle rank and Euler characteristic."""

    __slots__ = ("_vertices", "_edges", "_parent", "_rank", "_components", "_lock")

    def __init__(self, lock=None):
        self._vertices = set()
        self._edges = 0
        self._parent = {}
        self._rank = {}
        self._components = 0
        self._lock = lock if lock is not None else default_lock()

    def _add_vertex(self, v):
        """Add a vertex if it does not already exist."""
        if v not in self._vertices:
            self._vertices.add(v)
            self._parent[v] = v
            self._rank[v] = 0
            self._components += 1

    def _find(self, x):
        """Find the root of x with path compression (iterative, two-pass)."""
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != x:
            parent = self._parent[x]
            self._parent[x] = root
            x = parent
        return root

    def _union(self, a, b):
        """Union two vertices. Returns True if they were previously in different components."""
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return False
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        self._components -= 1
        return True

    def add_edge(self, u, v):
        """Add an undirected edge between u and v (parallel edges allowed)."""
        with self._lock:
            self._add_vertex(u)
            self._add_vertex(v)
            self._edges += 1
            self._union(u, v)

    def vertices_count(self):
        """Return the number of vertices."""
        with self._lock:
            return len(self._vertices)

    def edges_count(self):
        """Return the number of edges."""
        with self._lock:
            return self._edges

    def components_count(self):
        """Return the number of connected components."""
        with self._lock:
            return self._components

    def euler_characteristic(self):
        """Euler characteristic χ = V - E + C (equals 1 - cycle rank)."""
        with self._lock:
            return len(self._vertices) - self._edges + self._components

    def cycle_rank(self):
        """Cycle rank r = E - V + C: the number of independent cycles."""
        with self._lock:
            return self._edges - len(self._vertices) + self._components

    def is_connected(self):
        """Return True if the graph is connected (single component)."""
        with self._lock:
            return self._components <= 1

    def __repr__(self):
        with self._lock:
            v, e, c = len(self._vertices), self._edges, self._components
            return (f"<Graph: V={v}, E={e}, C={c}, "
                    f"χ={v - e + c}, r={e - v + c}>")


class DirectedGraph:
    """Directed multigraph with parallel arcs and self-loops; iterative
    (recursion-free) algorithms: Tarjan SCC, Kahn topological sort, BFS
    reachability, Eulerian path/circuit tests."""

    __slots__ = ("_vertices", "_adj", "_in", "_out", "_parent", "_rank",
                 "_components", "_edges", "_lock")

    def __init__(self, lock=None):
        self._vertices = set()
        self._adj = {}
        self._in = {}
        self._out = {}
        self._parent = {}
        self._rank = {}
        self._components = 0
        self._edges = 0
        self._lock = lock if lock is not None else default_lock()

    def _add_vertex(self, v):
        """Add a vertex if it does not already exist."""
        if v not in self._vertices:
            self._vertices.add(v)
            self._parent[v] = v
            self._rank[v] = 0
            self._components += 1

    def _find(self, x):
        """Find the root of x with path compression (iterative, two-pass)."""
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != x:
            parent = self._parent[x]
            self._parent[x] = root
            x = parent
        return root

    def _union(self, a, b):
        """Union two vertices. Returns True if they were previously in different components."""
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return False
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        self._components -= 1
        return True

    def add_edge(self, u, v):
        """
        Add a directed arc from u to v.
        Supports parallel arcs and self-loops.
        """
        with self._lock:
            self._add_vertex(u)
            self._add_vertex(v)
            self._edges += 1
            self._out[u] = self._out.get(u, 0) + 1
            self._in[v] = self._in.get(v, 0) + 1
            targets = self._adj.get(u)
            if targets is None:
                targets = self._adj[u] = {}
            targets[v] = targets.get(v, 0) + 1
            self._union(u, v)

    def vertices_count(self):
        """Return the number of vertices."""
        with self._lock:
            return len(self._vertices)

    def edges_count(self):
        """Return the number of arcs."""
        with self._lock:
            return self._edges

    def in_degree(self, v):
        """Return the in-degree of v (parallel arcs counted)."""
        with self._lock:
            return self._in.get(v, 0)

    def out_degree(self, v):
        """Return the out-degree of v (parallel arcs counted)."""
        with self._lock:
            return self._out.get(v, 0)

    def neighbors(self, v):
        """Return the distinct successors of v as a tuple (parallel arcs deduplicated)."""
        with self._lock:
            if v not in self._vertices:
                raise KeyError("Unknown vertex: %r" % (v,))
            return tuple(self._adj.get(v, {}).keys())

    def weak_components_count(self):
        """Return the number of weak components (direction ignored)."""
        with self._lock:
            return self._components

    def is_weakly_connected(self):
        """Return True if the graph is weakly connected (single weak component)."""
        with self._lock:
            return self._components <= 1

    def strong_components(self):
        """Strongly connected components as vertex lists (iterative Tarjan)."""
        with self._lock:
            index = {}
            lowlink = {}
            stack = []
            on_stack = set()
            components = []
            counter = 0
            for root in self._vertices:
                if root in index:
                    continue
                index[root] = counter
                lowlink[root] = counter
                counter += 1
                stack.append(root)
                on_stack.add(root)
                frames = [(root, iter(self._adj.get(root, {}).keys()))]
                while frames:
                    v, it = frames[-1]
                    advanced = False
                    for w in it:
                        if w not in index:
                            index[w] = counter
                            lowlink[w] = counter
                            counter += 1
                            stack.append(w)
                            on_stack.add(w)
                            frames.append((w, iter(self._adj.get(w, {}).keys())))
                            advanced = True
                            break
                        if w in on_stack and index[w] < lowlink[v]:
                            lowlink[v] = index[w]
                    if advanced:
                        continue
                    frames.pop()
                    if frames:
                        parent = frames[-1][0]
                        if lowlink[v] < lowlink[parent]:
                            lowlink[parent] = lowlink[v]
                    if lowlink[v] == index[v]:
                        comp = []
                        while True:
                            w = stack.pop()
                            on_stack.discard(w)
                            comp.append(w)
                            if w == v:
                                break
                        components.append(comp)
            return components

    def strong_components_count(self):
        """Return the number of strongly connected components."""
        return len(self.strong_components())

    def topological_sort(self):
        """Topological order as a list, or None on a directed cycle (Kahn)."""
        with self._lock:
            indeg = {v: self._in.get(v, 0) for v in self._vertices}
            queue = deque(v for v in self._vertices if indeg[v] == 0)
            order = []
            while queue:
                v = queue.popleft()
                order.append(v)
                for w, cnt in self._adj.get(v, {}).items():
                    indeg[w] -= cnt
                    if indeg[w] == 0:
                        queue.append(w)
            if len(order) != len(self._vertices):
                return None
            return order

    def is_dag(self):
        """Return True if the graph is a directed acyclic graph."""
        return self.topological_sort() is not None

    def reachable(self, src, dst):
        """True if dst is reachable from src (iterative BFS)."""
        with self._lock:
            if src not in self._vertices or dst not in self._vertices:
                return False
            if src == dst:
                return True
            seen = {src}
            queue = deque([src])
            while queue:
                v = queue.popleft()
                for w in self._adj.get(v, {}):
                    if w == dst:
                        return True
                    if w not in seen:
                        seen.add(w)
                        queue.append(w)
            return False

    def shortest_path(self, src, dst):
        """Shortest directed path [src, ..., dst] or None if unreachable
        (iterative BFS)."""
        with self._lock:
            # Access adjacency directly: neighbors() would re-acquire the
            # lock and deadlock on non-reentrant locks.
            return _bfs(lambda v: self._adj.get(v, {}), src, dst)

    def _non_isolated_weakly_connected(self):
        """Return True if all vertices with non-zero degree share one weak component."""
        active = [v for v in self._vertices if self._in.get(v, 0) or self._out.get(v, 0)]
        if not active:
            return True
        root = self._find(active[0])
        for v in active[1:]:
            if self._find(v) != root:
                return False
        return True

    def has_eulerian_circuit(self):
        """True iff every non-isolated vertex is in one weak component with
        in-degree == out-degree."""
        with self._lock:
            for v in self._vertices:
                if self._in.get(v, 0) != self._out.get(v, 0):
                    return False
            return self._non_isolated_weakly_connected()

    def has_eulerian_path(self):
        """True iff one weak component and balanced degrees except at most one
        start (out-in == 1) and one end (in-out == 1)."""
        with self._lock:
            if not self._non_isolated_weakly_connected():
                return False
            start, end = 0, 0
            for v in self._vertices:
                d = self._out.get(v, 0) - self._in.get(v, 0)
                if d == 1:
                    start += 1
                elif d == -1:
                    end += 1
                elif d != 0:
                    return False
            return start <= 1 and start == end

    def __repr__(self):
        with self._lock:
            return (f"<DirectedGraph: V={len(self._vertices)}, E={self._edges}, "
                    f"weak C={self._components}>")
