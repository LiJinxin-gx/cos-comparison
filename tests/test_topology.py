# -*- coding: utf-8 -*-
"""Directed graph (topology) tests for cos_comparison.interface.tools.math_tool.

Covers the DirectedGraph class: counts, degrees, weak/strong connectivity,
topological sort, DAG detection, reachability, Eulerian path/circuit tests,
parallel arcs, self-loops and recursion-free behaviour on deep graphs.
"""

import unittest


class TestDirectedGraphBasic(unittest.TestCase):
    def test_empty_graph_counts(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        self.assertEqual(g.vertices_count(), 0)
        self.assertEqual(g.edges_count(), 0)
        self.assertEqual(g.weak_components_count(), 0)
        self.assertTrue(g.is_weakly_connected())
        self.assertEqual(g.strong_components(), [])

    def test_basic_counts(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        self.assertEqual(g.vertices_count(), 3)
        self.assertEqual(g.edges_count(), 2)
        self.assertEqual(g.weak_components_count(), 1)
        self.assertTrue(g.is_weakly_connected())

    def test_parallel_arcs_and_self_loop(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        g.add_edge("a", "b")
        g.add_edge("a", "b")
        g.add_edge("b", "b")
        self.assertEqual(g.vertices_count(), 2)
        self.assertEqual(g.edges_count(), 3)
        self.assertEqual(g.weak_components_count(), 1)
        self.assertEqual(g.in_degree("a"), 0)
        self.assertEqual(g.out_degree("a"), 2)
        self.assertEqual(g.in_degree("b"), 3)
        self.assertEqual(g.out_degree("b"), 1)

    def test_degrees_unknown_vertex_zero(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        g.add_edge("a", "b")
        self.assertEqual(g.in_degree("zz"), 0)
        self.assertEqual(g.out_degree("zz"), 0)

    def test_neighbors(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        g.add_edge("a", "b")
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        g.add_edge("b", "a")
        self.assertEqual(set(g.neighbors("a")), {"b", "c"})
        self.assertEqual(g.neighbors("b"), ("a",))
        self.assertEqual(g.neighbors("c"), ())
        with self.assertRaises(KeyError):
            g.neighbors("unknown")

    def test_repr(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        g.add_edge("a", "b")
        self.assertIn("DirectedGraph", repr(g))


class TestDirectedGraphWeakConnectivity(unittest.TestCase):
    def test_multiple_components(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        for i in range(3):
            g.add_edge(i, i + 1)
        for i in range(10, 13):
            g.add_edge(i, i + 1)
        self.assertEqual(g.weak_components_count(), 2)
        self.assertFalse(g.is_weakly_connected())

    def test_isolated_vertex_component(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        g.add_edge("a", "b")
        g.add_edge("x", "x")
        self.assertEqual(g.weak_components_count(), 2)


class TestDirectedGraphStrongComponents(unittest.TestCase):
    def test_single_vertex_no_component_edge(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        comps = g.strong_components()
        self.assertEqual(len(comps), 3)
        self.assertEqual({frozenset(c) for c in comps}, {frozenset({v}) for v in "abc"})

    def test_two_vertex_cycle(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        comps = g.strong_components()
        self.assertEqual(len(comps), 1)
        self.assertEqual(set(comps[0]), {"a", "b"})

    def test_three_vertex_cycle(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        for e in [("a", "b"), ("b", "c"), ("c", "a")]:
            g.add_edge(*e)
        comps = g.strong_components()
        self.assertEqual(len(comps), 1)
        self.assertEqual(set(comps[0]), {"a", "b", "c"})

    def test_self_loop_single_component(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        g.add_edge("x", "x")
        comps = g.strong_components()
        self.assertEqual(len(comps), 1)
        self.assertEqual(comps[0], ["x"])

    def test_mixed_graph(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        edges = [("a", "b"), ("b", "c"), ("c", "a"),   # SCC: a,b,c
                 ("b", "d"), ("d", "e"),               # d -> e chain
                 ("e", "d")]                           # SCC: d,e
        for e in edges:
            g.add_edge(*e)
        comps = g.strong_components()
        comp_sets = {frozenset(c) for c in comps}
        self.assertEqual(comp_sets, {frozenset({"a", "b", "c"}), frozenset({"d", "e"})})

    def test_count(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        for e in [("a", "b"), ("b", "c"), ("c", "a"), ("d", "e")]:
            g.add_edge(*e)
        # SCCs: {a,b,c}, {d}, {e}
        self.assertEqual(g.strong_components_count(), 3)


class TestDirectedGraphTopologicalSort(unittest.TestCase):
    def test_dag_order_valid(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        for e in [("a", "b"), ("b", "c"), ("a", "c"), ("d", "b")]:
            g.add_edge(*e)
        order = g.topological_sort()
        self.assertIsNotNone(order)
        pos = {v: i for i, v in enumerate(order)}
        for u, v in [("a", "b"), ("b", "c"), ("a", "c"), ("d", "b")]:
            self.assertLess(pos[u], pos[v])

    def test_parallel_arc_dag(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        g.add_edge("a", "b")
        g.add_edge("a", "b")
        order = g.topological_sort()
        self.assertEqual(order, ["a", "b"])

    def test_cycle_returns_none(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        for e in [("a", "b"), ("b", "c"), ("c", "a")]:
            g.add_edge(*e)
        self.assertIsNone(g.topological_sort())

    def test_self_loop_returns_none(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        g.add_edge("a", "a")
        self.assertIsNone(g.topological_sort())

    def test_is_dag(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        dag = DirectedGraph()
        dag.add_edge("a", "b")
        self.assertTrue(dag.is_dag())
        cyc = DirectedGraph()
        cyc.add_edge("a", "b")
        cyc.add_edge("b", "a")
        self.assertFalse(cyc.is_dag())

    def test_large_dag(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        n = 10000
        for i in range(n - 1):
            g.add_edge(i, i + 1)
        order = g.topological_sort()
        self.assertEqual(len(order), n)
        self.assertEqual(order[0], 0)
        self.assertEqual(order[-1], n - 1)


class TestDirectedGraphReachability(unittest.TestCase):
    def test_reachable(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        for e in [("a", "b"), ("b", "c"), ("c", "d"), ("a", "x"), ("y", "z")]:
            g.add_edge(*e)
        self.assertTrue(g.reachable("a", "d"))
        self.assertTrue(g.reachable("a", "x"))
        self.assertTrue(g.reachable("y", "z"))
        self.assertFalse(g.reachable("d", "a"))
        self.assertFalse(g.reachable("x", "z"))
        self.assertFalse(g.reachable("a", "missing"))

    def test_reachable_same_vertex(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        g.add_edge("a", "b")
        self.assertTrue(g.reachable("a", "a"))
        self.assertFalse(g.reachable("a", "c"))

    def test_reachable_via_self_loop_only(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        g.add_edge("a", "a")
        self.assertTrue(g.reachable("a", "a"))
        self.assertFalse(g.reachable("a", "b"))


class TestDirectedGraphEulerian(unittest.TestCase):
    def test_eulerian_circuit_cycle(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        for e in [("a", "b"), ("b", "c"), ("c", "a")]:
            g.add_edge(*e)
        self.assertTrue(g.has_eulerian_circuit())
        self.assertTrue(g.has_eulerian_path())

    def test_eulerian_circuit_unbalanced(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        for e in [("a", "b"), ("b", "c")]:
            g.add_edge(*e)
        self.assertFalse(g.has_eulerian_circuit())
        self.assertTrue(g.has_eulerian_path())

    def test_eulerian_path_chain(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        for e in [("a", "b"), ("b", "c"), ("c", "d")]:
            g.add_edge(*e)
        self.assertFalse(g.has_eulerian_circuit())
        self.assertTrue(g.has_eulerian_path())

    def test_eulerian_self_loop(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        g.add_edge("x", "x")
        self.assertTrue(g.has_eulerian_circuit())
        self.assertTrue(g.has_eulerian_path())

    def test_eulerian_parallel_balanced(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        self.assertTrue(g.has_eulerian_circuit())

    def test_eulerian_parallel_unbalanced(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        g.add_edge("a", "b")
        g.add_edge("a", "b")
        self.assertFalse(g.has_eulerian_circuit())
        self.assertFalse(g.has_eulerian_path())

    def test_eulerian_path_one_start_one_end(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        for e in [("a", "b"), ("b", "c"), ("c", "a"), ("a", "d")]:
            g.add_edge(*e)
        # degrees: a out=2 in=1, b out=1 in=1, c out=1 in=1, d out=0 in=1
        self.assertFalse(g.has_eulerian_circuit())
        self.assertTrue(g.has_eulerian_path())

    def test_eulerian_path_bad_degrees(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        for e in [("a", "b"), ("a", "c")]:
            g.add_edge(*e)
        self.assertFalse(g.has_eulerian_path())

    def test_eulerian_disconnected(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        for e in [("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")]:
            g.add_edge(*e)
        self.assertFalse(g.has_eulerian_circuit())
        self.assertFalse(g.has_eulerian_path())

    def test_eulerian_isolated_vertex_ignored(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        for e in [("a", "b"), ("b", "c"), ("c", "a")]:
            g.add_edge(*e)
        with g._lock:
            g._add_vertex("iso")
        self.assertTrue(g.has_eulerian_circuit())

    def test_eulerian_detached_self_loop_rejected(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        for e in [("a", "b"), ("b", "c"), ("c", "a")]:
            g.add_edge(*e)
        g.add_edge("alone", "alone")
        self.assertFalse(g.has_eulerian_circuit())
        self.assertFalse(g.has_eulerian_path())

    def test_eulerian_empty(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        self.assertTrue(g.has_eulerian_circuit())
        self.assertTrue(g.has_eulerian_path())


class TestDirectedGraphNoRecursion(unittest.TestCase):
    def test_deep_chain_all_algorithms(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        n = 5000
        g = DirectedGraph()
        for i in range(n - 1):
            g.add_edge(i, i + 1)
        # A recursive Tarjan / Kahn would raise RecursionError here.
        self.assertEqual(g.strong_components_count(), n)
        order = g.topological_sort()
        self.assertEqual(len(order), n)
        self.assertTrue(g.reachable(0, n - 1))
        self.assertFalse(g.reachable(n - 1, 0))

    def test_deep_cycle(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        n = 5000
        g = DirectedGraph()
        for i in range(n):
            g.add_edge(i, (i + 1) % n)
        self.assertEqual(g.strong_components_count(), 1)
        self.assertIsNone(g.topological_sort())
        self.assertFalse(g.is_dag())
        self.assertTrue(g.has_eulerian_circuit())


class TestDirectedGraphLockAndIsolation(unittest.TestCase):
    def test_default_lock_usable(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph
        g = DirectedGraph()
        g.add_edge(1, 2)
        with g._lock:
            self.assertEqual(g.vertices_count(), 2)

    def test_independent_from_graph(self):
        from cos_comparison.interface.tools.math_tool.topology import DirectedGraph, Graph
        d = DirectedGraph()
        d.add_edge("a", "b")
        u = Graph()
        u.add_edge("b", "a")
        self.assertEqual(d.vertices_count(), 2)
        self.assertEqual(u.vertices_count(), 2)
        self.assertEqual(d.edges_count(), 1)
        self.assertEqual(u.edges_count(), 1)
        self.assertNotEqual(type(d), type(u))


if __name__ == "__main__":
    unittest.main()