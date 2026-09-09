"""brain_layer rigorous derivation tests: probability chain rule (strict
vs non-strict modes) and protocol-style judge (unpacked binds, uncertain
flag)."""

import unittest

from cos_comparison.brain_layer.logic.probability_logic import (
    IntersectionEvent,
    _resolve,
    chain_probability,
)
from cos_comparison.brain_layer.logic.symbol_logic import (
    Logic,
    Logic_bind,
    Logic_context,
    default_judge_func,
    sure_true,
)


class TestProbabilityChainRule(unittest.TestCase):
    """P(A|B): exact hits first; strict requires exact/axiom derivations
    (0.0 when anything needs an unstated hypothesis); non-strict falls back
    to the Markov chain product."""

    def test_reflexive(self):
        self.assertEqual(_resolve({}, "A", "A"), 1.0)
        self.assertEqual(_resolve({}, "A", "A", strict=True), 1.0)

    def test_direct_hit(self):
        binds = {("A", "B"): 0.7}
        self.assertEqual(_resolve(binds, "A", "B"), 0.7)
        self.assertEqual(_resolve(binds, "A", "B", strict=True), 0.7)

    def test_non_strict_markov_chain(self):
        # C -> B -> A : P(A|C) = P(B|C) * P(A|B)  (Markov approximation)
        binds = {("B", "C"): 0.8, ("A", "B"): 0.5}
        self.assertAlmostEqual(chain_probability(binds, "A", "C"), 0.4)

    def test_strict_markov_chain_refused(self):
        # Only binary bindings: the exact chain needs P(A | B&C) explicitly;
        # Markov approximation is not allowed in strict mode -> 0.0
        binds = {("B", "C"): 0.8, ("A", "B"): 0.5}
        self.assertEqual(_resolve(binds, "A", "C", strict=True), 0.0)

    def test_strict_exact_chain_with_intersections(self):
        # Exact multiplication rule: P(A|C) = P(B|C) * P(A | B&C)
        cond = IntersectionEvent("B", "C")
        binds = {("B", "C"): 0.8, ("A", cond): 0.5}
        self.assertAlmostEqual(_resolve(binds, "A", "C", strict=True), 0.4)

    def test_strict_three_step_exact_chain(self):
        # C -> B -> D -> A : P(A|C) = P(B|C) * P(D|BC) * P(A|BCD)
        cond2 = IntersectionEvent("B", "C")
        cond3 = IntersectionEvent("D", "B", "C")
        binds = {("B", "C"): 0.8,
                 ("D", cond2): 0.6,
                 ("A", cond3): 0.5}
        self.assertAlmostEqual(_resolve(binds, "A", "C", strict=True),
                               0.8 * 0.6 * 0.5)

    def test_strict_missing_step_returns_zero(self):
        binds = {("B", "C"): 0.8}  # missing P(A | B&C)
        self.assertEqual(_resolve(binds, "A", "C", strict=True), 0.0)

    def test_bayes_axiom_strict(self):
        # A4: P(A|B) = P(B|A) * P(A|C) / P(B|C) (reference-invariant ratio)
        binds = {("B", "A"): 0.5, ("A", "C"): 0.4, ("B", "C"): 0.8}
        self.assertAlmostEqual(_resolve(binds, "A", "B", strict=True),
                               0.5 * 0.4 / 0.8)

    def test_unresolved_zero_both_modes(self):
        self.assertEqual(_resolve({}, "A", "B"), 0.0)
        self.assertEqual(_resolve({}, "A", "B", strict=True), 0.0)

    def test_event_binds_strict_flag(self):
        from cos_comparison.brain_layer.logic.probability_logic import event_context
        ctx = event_context()  # auto-wired EventBinds engine, strict=False
        ctx.add_bind([("B", "C", 0.8), ("A", "B", 0.5)])
        self.assertAlmostEqual(ctx.bind_probability("A", "C"), 0.4)
        # strict engine: Markov refused -> 0.0
        ctx.binds.strict = True
        self.assertEqual(ctx.bind_probability("A", "C"), 0.0)
        ctx.add_bind([("A", IntersectionEvent("B", "C"), 0.5)])
        self.assertAlmostEqual(ctx.bind_probability("A", "C"), 0.4)


class TestJudgeProtocol(unittest.TestCase):
    """judge receives unpacked binds (*args_bind, **kwargs_bind); rigorous
    default parsing returns the uncertain flag when it cannot judge."""

    def _ctx(self):
        ctx = Logic_context()
        ctx.binds = [Logic_bind("rain", "wet"),
                     Logic_bind("wet", "slippery")]
        return ctx

    def test_positional_pair(self):
        ctx = self._ctx()
        self.assertEqual(ctx.logic_judge("rain", "slippery"), sure_true)

    def test_keyword_pair(self):
        ctx = self._ctx()
        self.assertEqual(ctx.logic_judge(a="rain", b="slippery"), sure_true)

    def test_missing_events_uncertain(self):
        ctx = self._ctx()
        self.assertEqual(ctx.logic_judge("rain"), Logic.SURE)
        self.assertEqual(ctx.logic_judge(), Logic.SURE)

    def test_unknown_pair_uncertain(self):
        ctx = self._ctx()
        self.assertEqual(ctx.logic_judge("rain", "fire"), Logic.SURE)

    def test_reflexive(self):
        ctx = self._ctx()
        self.assertEqual(ctx.logic_judge("rain", "rain"), sure_true)

    def test_return_path(self):
        ctx = self._ctx()
        path = ctx.logic_judge("rain", "slippery", return_path=True)
        self.assertIsInstance(path, list)
        self.assertEqual([b.reason for b in path], ["rain", "wet"])
        self.assertIsNone(ctx.logic_judge("rain", "fire", return_path=True))
        self.assertEqual(ctx.logic_judge("rain", "rain", return_path=True), [])

    def test_judge_func_direct_unpack(self):
        ctx = self._ctx()
        # direct call: judge_func(binds, *args_bind, **kwargs_bind)
        self.assertEqual(
            default_judge_func(ctx.binds, "wet", "slippery"), sure_true)
        self.assertEqual(default_judge_func(ctx.binds, "wet", "x"),
                         Logic.SURE)


if __name__ == "__main__":
    unittest.main()
