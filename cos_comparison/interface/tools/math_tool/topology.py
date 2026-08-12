"""
It provides some tools to solve problems about topology.
"""

from .context_tool import VoidContext as default_lock

def Euler_characteristic_compute_by_cell(cell_list):
    factor = 1
    Euler_characteristic = 0
    for cell in cell_list:
        if type(cell) is int:
            if cell <= 0:
                raise ValueError("Cell must be a positive integer.")
            Euler_characteristic += cell * factor
            factor *= -1
        else:
            raise TypeError("Cell must be a positive integer.")
    return Euler_characteristic

class Graph:
    """
    Undirected multigraph with support for parallel edges and connected components.
    Uses a union-find (DSU) to maintain component count, enabling accurate
    computation of cycle rank (independent cycles) and Euler characteristic.
    """

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
        """Find with path compression."""
        if self._parent[x] != x:
            self._parent[x] = self._find(self._parent[x])
        return self._parent[x]

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
        Add an undirected edge between vertices u and v.
        Supports parallel edges (multiple edges between the same vertices).
        """
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
        """
        Compute the Euler characteristic χ = V - E + C.
        For a graph, this equals the number of connected components minus
        the cycle rank: χ = 1 - r.
        """
        with self._lock:
            return len(self._vertices) - self._edges + self._components

    def cycle_rank(self):
        """
        Compute the cycle rank (also called the circuit rank or nullity).
        For an undirected graph, r = E - V + C, where C is the number of
        connected components. This represents the number of independent cycles.
        """
        with self._lock:
            return self._edges - len(self._vertices) + self._components

    def is_connected(self):
        """Return True if the graph is connected (single component)."""
        with self._lock:
            return self._components <= 1

    def __repr__(self):
        with self._lock:
            return (f"<Graph: V={len(self._vertices)}, E={self._edges}, "
                    f"C={self._components}, χ={self.euler_characteristic()}, "
                    f"r={self.cycle_rank()}>")
