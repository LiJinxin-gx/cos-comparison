"""brain_layer.logic robustness protocol tests: duck-typed containers and
graphs, event/value domains, serialization and injection points."""

import copy
import pickle
import unittest
from decimal import Decimal
from fractions import Fraction

from cos_comparison.brain_layer.logic.probability_logic import (
    _resolve,
    chain_probability,
    union_probability,
    default_probability_func,
    strict_probability_func,
    default_add_bind,
    IntersectionEvent,
    UnionEvent,
    GlobalEvent,
    global_event,
    EventBinds,
    EventContextProtocol,
    event_bind,
    event_context,
    DirectedGraph,
)
from cos_comparison.brain_layer.logic.symbol_logic import (
    Logic,
    Logic_bind,
    Logic_context,
    Atomic_proposition,
    Variable,
    UnsupportedError,
    LogicError,
    is_true,
    is_sure,
    is_uncertain,
    Logic_sure,
    default_judge_func,
    sure_true,
)


class TestStrictChainEventDomain(unittest.TestCase):
    """Strict chain conditions expand container events only: no string
    character expansion, no crashes on non-iterable events."""

    def test_multi_char_events(self):
        binds = {("wet", "rain"): 0.8,
                 ("slippery", IntersectionEvent("wet", "rain")): 0.5}
        self.assertAlmostEqual(
            _resolve(binds, "slippery", "rain", strict=True), 0.4)

    def test_variable_conditions(self):
        vr, vw = Variable("rain", 1), Variable("wet", 2)
        binds = {(vw, vr): 0.8, ("A", IntersectionEvent(vw, vr)): 0.5}
        self.assertAlmostEqual(_resolve(binds, "A", vr, strict=True), 0.4)

    def test_global_event_condition(self):
        binds = {("X", global_event): 0.5,
                 ("A", IntersectionEvent("X", global_event)): 0.5}
        self.assertAlmostEqual(
            _resolve(binds, "A", global_event, strict=True), 0.25)

    def test_tuple_condition_members(self):
        binds = {("B", ("C", "D")): 0.8,
                 ("A", IntersectionEvent("B", "C", "D")): 0.5}
        self.assertAlmostEqual(
            _resolve(binds, "A", ("C", "D"), strict=True), 0.4)


class TestEventSerialization(unittest.TestCase):
    def test_copy_pickle(self):
        for cls in (UnionEvent, IntersectionEvent):
            event = cls(1, 2)
            copied = copy.copy(event)
            self.assertEqual(copied, event)
            self.assertEqual(type(copied), cls)
            self.assertEqual(copied.name, event.name)
            restored = pickle.loads(pickle.dumps(event))
            self.assertEqual(restored, event)
            self.assertEqual(type(restored), cls)


class TestKindProtocol(unittest.TestCase):
    def test_subclass_members_compare(self):
        class SubUnion(UnionEvent):
            pass
        self.assertEqual(SubUnion(1, 2), UnionEvent(1, 2))
        self.assertEqual(hash(SubUnion(1, 2)), hash(UnionEvent(1, 2)))

    def test_kind_still_separates(self):
        self.assertFalse(UnionEvent(1, 2) == IntersectionEvent(1, 2))
        self.assertTrue(UnionEvent(1, 2) != IntersectionEvent(1, 2))

    def test_plain_frozenset_never_equal(self):
        self.assertFalse(UnionEvent(1, 2) == frozenset({1, 2}))
        self.assertFalse(frozenset({1, 2}) == UnionEvent(1, 2))


class TestGlobalEventBenchmark(unittest.TestCase):
    def test_fresh_instance_matches(self):
        self.assertEqual(
            _resolve({("A", global_event): 0.5}, "A", GlobalEvent()), 0.5)
        self.assertEqual(GlobalEvent(), global_event)
        self.assertEqual(hash(GlobalEvent()), hash(global_event))


class TestValueDomain(unittest.TestCase):
    def test_decimal_chain(self):
        binds = {("B", "C"): Decimal("0.5"), ("A", "B"): Decimal("0.5")}
        self.assertEqual(_resolve(binds, "A", "C"), Decimal("0.25"))

    def test_fraction_chain(self):
        binds = {("B", "C"): Fraction(1, 2), ("A", "B"): Fraction(1, 2)}
        self.assertEqual(_resolve(binds, "A", "C"), Fraction(1, 4))

    def test_decimal_strict_chain(self):
        binds = {("B", "C"): Decimal("0.8"),
                 ("A", IntersectionEvent("B", "C")): Decimal("0.5")}
        self.assertEqual(_resolve(binds, "A", "C", strict=True), Decimal("0.4"))


class TestDuckContainers(unittest.TestCase):
    def test_list_of_triples(self):
        binds = [("B", "C", 0.8), ("A", "B", 0.5)]
        self.assertAlmostEqual(chain_probability(binds, "A", "C"), 0.4)
        self.assertEqual(_resolve(binds, "A", "B"), 0.5)

    def test_mutable_mapping(self):
        from collections.abc import MutableMapping

        class MyBinds(MutableMapping):
            def __init__(self):
                self._d = {}
            def __getitem__(self, key):
                return self._d[key]
            def __setitem__(self, key, value):
                self._d[key] = value
            def __delitem__(self, key):
                del self._d[key]
            def __iter__(self):
                return iter(self._d)
            def __len__(self):
                return len(self._d)

        binds = MyBinds()
        default_add_bind(binds, [("A", "B", 0.7)])
        self.assertEqual(_resolve(binds, "A", "B"), 0.7)

    def test_event_bind_as_container(self):
        eb = event_bind(event="A")
        eb.bind("B", 0.7)
        self.assertEqual(_resolve(eb, "A", "B"), 0.7)
        self.assertEqual(len(eb), 1)
        self.assertIn("B", eb)
        self.assertEqual(eb["B"], 0.7)
        self.assertEqual(eb.get("B"), 0.7)

    def test_default_add_bind_append(self):
        binds = []
        default_add_bind(binds, [("A", "B", 0.5)])
        self.assertEqual(binds, [("A", "B", 0.5)])


class TestDuckGraphAndContext(unittest.TestCase):
    class DuckGraph:
        def __init__(self):
            self._adj = {}
        def add_edge(self, u, v):
            self._adj.setdefault(u, []).append(v)
        def neighbors(self, v):
            if v not in self._adj:
                raise KeyError(v)
            return tuple(self._adj[v])

    def test_graph_factory(self):
        self.assertAlmostEqual(
            chain_probability({("B", "C"): 0.8, ("A", "B"): 0.5}, "A", "C",
                              graph_factory=self.DuckGraph), 0.4)

    def test_judge_graph_factory(self):
        binds = [Logic_bind("B", "C"), Logic_bind("A", "B")]
        self.assertEqual(
            default_judge_func(binds, "A", "C", graph_factory=self.DuckGraph),
            sure_true)

    def test_path_func_injection(self):
        calls = []
        def fake_path(graph, src, dst):
            calls.append((src, dst))
            return None
        self.assertEqual(
            chain_probability({("A", "B"): 0.5}, "A", "B",
                              path_func=fake_path), 0.5)
        self.assertEqual(calls, [])
        chain_probability({("B", "C"): 0.5}, "C", "B", path_func=fake_path)
        self.assertEqual(calls, [("B", "C")])

    def test_duck_context(self):
        class Ctx:
            pass
        ctx = Ctx()
        ctx.binds = {("A", "B"): 0.7}
        self.assertEqual(default_probability_func(ctx, "A", "B"), 0.7)
        self.assertEqual(strict_probability_func(ctx, "A", "B"), 0.7)


class TestEngineInjection(unittest.TestCase):
    class ProbeFactory:
        def __init__(self):
            self.instances = []
        def __call__(self):
            graph = DirectedGraph()
            self.instances.append(graph)
            return graph

    def test_event_binds_factory(self):
        factory = self.ProbeFactory()
        binds = EventBinds(graph_factory=factory)
        binds.add_bind([("A", "B", 0.5)])
        self.assertEqual(binds.resolve("A", "B"), 0.5)
        self.assertEqual(len(factory.instances), 1)
        binds.add_bind([("B", "C", 0.5)])
        binds.resolve("A", "C")
        self.assertEqual(len(factory.instances), 2)

    def test_event_binds_kwargs_not_data(self):
        factory = self.ProbeFactory()
        binds = EventBinds(graph_factory=factory)
        self.assertNotIn("graph_factory", binds)

    def test_protocol_factory(self):
        factory = self.ProbeFactory()
        engine = EventContextProtocol(graph_factory=factory)
        engine.add_bind(None, [("A", "B", 0.4)])
        self.assertEqual(engine.resolve(None, "A", "B"), 0.4)
        self.assertEqual(len(factory.instances), 1)

    def test_direct_setitem_invalidates(self):
        binds = EventBinds()
        binds[("A", "B")] = 0.7
        self.assertEqual(binds.resolve("A", "B"), 0.7)

    def test_copy_keeps_engine_state(self):
        binds = EventBinds(strict=True)
        binds.add_bind([("A", "B", 0.5)])
        other = binds.copy()
        self.assertIsInstance(other, EventBinds)
        self.assertTrue(other.strict)
        self.assertEqual(other.stats["adds"], 1)

    def test_strict_chain_probability_entry(self):
        binds = {("B", "C"): 0.8, ("A", "B"): 0.5}
        self.assertEqual(chain_probability(binds, "A", "C", strict=True), 0.0)
        self.assertAlmostEqual(chain_probability(binds, "A", "C"), 0.4)

    def test_strict_functions_distinguish_paths(self):
        ctx = event_context()
        ctx.add_bind([("B", global_event, 0.8), ("A", "B", 0.5)])
        self.assertAlmostEqual(
            default_probability_func(ctx, "A", global_event), 0.4)
        self.assertEqual(
            strict_probability_func(ctx, "A", global_event), 0.0)

    def test_union_strict_distinguishes_paths(self):
        ctx = event_context()
        ctx.add_bind([("B", global_event, 0.8), ("A", "B", 0.5),
                      ("A", IntersectionEvent("B", global_event), 0.4)])
        self.assertAlmostEqual(
            union_probability(ctx, "A", "B", global_event), 0.8)
        self.assertAlmostEqual(
            union_probability(ctx, "A", "B", global_event, strict=True), 0.72)


class TestUnbindAndIteration(unittest.TestCase):
    def test_unbind_missing(self):
        eb = event_bind("n", "ev")
        self.assertEqual(eb.unbind("x"), 1)

    def test_iteration_non_pair_keys(self):
        ctx = event_context()
        ctx.binds["rain"] = 0.2
        ctx.binds[123] = 0.3
        triples = list(ctx)
        self.assertIn(("rain", 0.2), triples)
        self.assertIn((123, 0.3), triples)


class TestSymbolProtocol(unittest.TestCase):
    def test_keys_requires_arg_names(self):
        with self.assertRaises(UnsupportedError):
            Atomic_proposition(1, 2).keys()

    def test_logic_bind_mapping_contract(self):
        bind = Logic_bind("r", "w")
        self.assertIn("reason", bind)
        self.assertEqual(bind["reason"], "r")
        with self.assertRaises(KeyError):
            bind["nope"]

    def test_falsy_slot_functions_still_used(self):
        calls = []
        class FalsyInit:
            def __bool__(self):
                return False
            def __call__(self, binds, *args, **kwargs):
                calls.append(1)
                return 0
        ctx = Logic_context(init_func=FalsyInit())
        self.assertEqual(ctx.initialize(), 0)
        self.assertEqual(calls, [1])

    def test_mapping_style_rule_objects(self):
        rules = [{"reason": "rain", "result": "wet"},
                 {"reason": "wet", "result": "slippery"}]
        self.assertEqual(
            default_judge_func(rules, "rain", "slippery"), sure_true)
        path = default_judge_func(rules, "rain", "slippery", return_path=True)
        self.assertEqual([rule["reason"] for rule in path], ["rain", "wet"])

    def test_rule_missing_field(self):
        with self.assertRaises(TypeError):
            default_judge_func([object()], "a", "b")

    def test_judge_path_func_injection(self):
        calls = []
        def fake_path(graph, src, dst):
            calls.append((src, dst))
            return None
        binds = [Logic_bind("r", "w")]
        self.assertEqual(
            default_judge_func(binds, "r", "w", path_func=fake_path),
            Logic.SURE)
        self.assertEqual(calls, [("r", "w")])

    def test_variable_unhashable_value(self):
        variable = Variable("v", [1, 2])
        self.assertEqual(hash(variable), hash(Variable("w", [1, 2])))
        self.assertEqual(variable, Variable("w", [1, 2]))

    def test_subclass_equality(self):
        class SubVariable(Variable):
            pass
        self.assertEqual(SubVariable("a", 1), Variable("b", 1))


class TestSymbolMathHardening(unittest.TestCase):
    """Math-hardening round: rule availability via status, argument-name
    validation, NaN reflexivity, path-contract errors and the explicit
    logic predicates."""

    def test_refuted_status_rule_excluded(self):
        ctx = Logic_context()
        ctx.binds = [Logic_bind("r", "w", status=Logic.SURE),
                     Logic_bind("w", "z", status=Logic.TRUE)]
        self.assertEqual(ctx.logic_judge("r", "w"), Logic.SURE)
        self.assertEqual(ctx.logic_judge("w", "z"), sure_true)
        self.assertEqual(ctx.logic_judge("r", "z"), Logic.SURE)

    def test_asserted_default_rule_participates(self):
        ctx = Logic_context()
        ctx.binds = [Logic_bind("r", "w"), Logic_bind("w", "z")]
        self.assertEqual(ctx.logic_judge("r", "z"), sure_true)

    def test_mapping_rule_status_key(self):
        rules = [{"reason": "a", "result": "b", "status": Logic.SURE},
                 {"reason": "b", "result": "c"}]
        self.assertEqual(default_judge_func(rules, "a", "b"), Logic.SURE)
        self.assertEqual(default_judge_func(rules, "a", "c"), Logic.SURE)
        self.assertEqual(default_judge_func(rules, "b", "c"), sure_true)

    def test_path_without_supporting_rule_raises(self):
        binds = [Logic_bind("a", "b")]
        with self.assertRaises(LogicError):
            default_judge_func(
                binds, "a", "b",
                path_func=lambda graph, src, dst: [src, "x", dst],
                return_path=True)
        # a real path still returns its proving rules
        rules = default_judge_func(binds, "a", "b", return_path=True)
        self.assertEqual(rules, binds)

    def test_nan_variable_reflexive(self):
        variable = Variable("x", float("nan"))
        self.assertTrue(variable == variable)
        self.assertEqual(hash(variable), hash(variable))

    def test_arg_names_validation(self):
        with self.assertRaises(ValueError):
            Atomic_proposition(1, 2, 3, arg_names=("a", "b"))
        with self.assertRaises(ValueError):
            Atomic_proposition(1, 2, arg_names=("a", "a"))
        prop = Atomic_proposition(1, 2, arg_names=("a", "b"))
        self.assertEqual(prop.keys(), ("a", "b"))
        self.assertEqual(len(prop), 2)

    def test_logic_predicates(self):
        # is_true: carries Logic.TRUE
        self.assertTrue(is_true(Logic.TRUE))
        self.assertTrue(is_true(sure_true))
        self.assertFalse(is_true(Logic.SURE))
        # is_sure: determined (SURE = sure)
        self.assertTrue(is_sure(Logic.SURE))
        self.assertTrue(is_sure(sure_true))
        self.assertFalse(is_sure(Logic.TRUE))
        self.assertFalse(is_sure(Logic(0)))
        # is_uncertain: not determined
        self.assertTrue(is_uncertain(Logic(0)))
        self.assertTrue(is_uncertain(Logic.TRUE))
        self.assertFalse(is_uncertain(Logic.SURE))
        self.assertFalse(is_uncertain(sure_true))
        self.assertIs(Logic_sure, Logic.SURE)


class TestStatementJudge(unittest.TestCase):
    """Statement semantics: the judge decides the correctness of the full
    statement (a, b[, limit[, is_true]]) - the unpacked Logic_bind slots."""

    def _ctx(self):
        ctx = Logic_context()
        ctx.binds = [Logic_bind("r", "w"), Logic_bind("w", "z")]
        return ctx

    def test_full_statement_unpacking(self):
        ctx = self._ctx()
        self.assertEqual(ctx.logic_judge(*Logic_bind("r", "w")), sure_true)

    def test_truth_claim_not_derivable(self):
        ctx = self._ctx()
        self.assertEqual(ctx.logic_judge(*Logic_bind("x", "y")), Logic.SURE)

    def test_falsehood_claim_correct(self):
        ctx = self._ctx()
        self.assertEqual(
            ctx.logic_judge(*Logic_bind("x", "y", status=Logic.SURE)),
            sure_true)

    def test_falsehood_claim_incorrect(self):
        ctx = self._ctx()
        self.assertEqual(
            ctx.logic_judge(*Logic_bind("r", "w", status=Logic.SURE)),
            Logic.SURE)

    def test_statement_limit_bounds_endpoints(self):
        ctx = self._ctx()
        self.assertEqual(ctx.logic_judge("r", "w", {"r", "w"}), sure_true)
        self.assertEqual(ctx.logic_judge("r", "w", {"r"}), Logic.SURE)

    def test_kwargs_full_statement(self):
        ctx = self._ctx()
        self.assertEqual(
            ctx.logic_judge(a="x", b="y", is_true=Logic.SURE), sure_true)
        self.assertEqual(
            ctx.logic_judge(a="r", b="w", limit={"r"}), Logic.SURE)

    def test_return_path_statement_forms(self):
        ctx = self._ctx()
        self.assertEqual(
            ctx.logic_judge(*Logic_bind("r", "w"), return_path=True),
            [ctx.binds[0]])
        self.assertIsNone(
            ctx.logic_judge(*Logic_bind("x", "y", status=Logic.SURE),
                            return_path=True))
        self.assertIsNone(ctx.logic_judge("x", "y", return_path=True))


class TestPackageExports(unittest.TestCase):
    def test_core_symbols_exported(self):
        from cos_comparison.brain_layer.logic import (
            global_event, sure_true, Logic_true, Logic_sure, Logic)
        self.assertIs(Logic_true, Logic.TRUE)
        self.assertIs(Logic_sure, Logic.SURE)
        self.assertEqual(sure_true, Logic.TRUE | Logic.SURE)
        self.assertIsInstance(global_event, GlobalEvent)


if __name__ == "__main__":
    unittest.main()
