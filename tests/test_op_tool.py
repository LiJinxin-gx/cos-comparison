"""op_tool tests: Python built-in operations in functional form.

Stdlib only, non-GUI.  Verifies the operator-set surface (names, grouping,
callability), semantics against plain expressions for every group, the
reflected / in-place / protocol variants, and error propagation.
"""

import unittest

from cos_comparison.interface import tools
from cos_comparison.interface.tools import op_tool
from cos_comparison.interface.tools.op_tool import (
    add, sub, mul, matmul, truediv, floordiv, mod, pow as op_pow,
    radd, rsub, rmul, rmatmul, rtruediv, rfloordiv, rmod, rpow,
    iadd, isub, imul, imatmul, itruediv, ifloordiv, imod, ipow,
    and_, or_, xor, lshift, rshift,
    rand, ror, rxor, rlshift, rrshift,
    iand, ior, ixor, ilshift, irshift,
    eq, ne, lt, le, gt, ge,
    neg, pos, abs as op_abs, invert, not_,
    getitem, setitem, delitem, contains, concat, countOf, indexOf,
    call, truth, len_, length_hint,
)


class _Dot:
    """Custom __matmul__ / __rmatmul__ carrier (stdlib types lack @)."""

    def __matmul__(self, other):
        return "fwd"

    def __rmatmul__(self, other):
        return "rev"


class TestSurface(unittest.TestCase):
    def test_all_names_callable(self):
        for name in op_tool.__all__:
            self.assertTrue(callable(getattr(op_tool, name)),
                            msg="%s is not callable" % name)

    def test_star_import_clean(self):
        ns = {}
        exec("from cos_comparison.interface.tools.op_tool import *", ns)
        self.assertEqual(set(ns) - {"__builtins__"}, set(op_tool.__all__))

    def test_accessible_via_tools_package(self):
        self.assertIs(tools.add, op_tool.add)
        self.assertIs(tools.contains, op_tool.contains)

    def test_count_matches_all(self):
        counts = [8, 8, 8, 5, 5, 5, 6, 5, 11]
        self.assertEqual(len(op_tool.__all__), sum(counts))
        self.assertEqual(len(op_tool.__all__), len(set(op_tool.__all__)))


class TestArithmetic(unittest.TestCase):
    def test_binary(self):
        self.assertEqual(add(2, 3), 2 + 3)
        self.assertEqual(sub(7, 2), 7 - 2)
        self.assertEqual(mul(4, 5), 4 * 5)
        self.assertEqual(truediv(7, 2), 7 / 2)
        self.assertEqual(floordiv(7, 2), 7 // 2)
        self.assertEqual(mod(7, 3), 7 % 3)
        self.assertEqual(op_pow(2, 10), 2 ** 10)
        self.assertEqual(truediv(1, 2), 0.5)

    def test_matmul(self):
        self.assertEqual(matmul(_Dot(), 1), "fwd")

    def test_pow_keyword_mod(self):
        self.assertEqual(op_pow(2, 10, 1000), pow(2, 10, 1000))

    def test_reflected(self):
        # r*(a, b) == b.__r*(a): the `a op b` fallback path.
        self.assertEqual(rsub(1, 10), (10).__rsub__(1))
        self.assertEqual(rsub(1, 10), 1 - 10)
        self.assertEqual(rmul(3, "ab"), "ababab")
        self.assertEqual(rmul(3, 5), 3 * 5)
        self.assertEqual(radd(1, 10), (10).__radd__(1))
        self.assertEqual(radd(1, 10), 1 + 10)
        self.assertEqual(rtruediv(2, 1), (1).__rtruediv__(2))
        self.assertEqual(rfloordiv(3, 10), (10).__rfloordiv__(3))
        self.assertEqual(rfloordiv(2, 1), 2 // 1)
        self.assertEqual(rmod(3, 10), (10).__rmod__(3))
        self.assertEqual(rmod(3, 10), 3 % 10)
        self.assertEqual(rpow(2, 10), (10).__rpow__(2))
        self.assertEqual(rpow(2, 10), 2 ** 10)
        self.assertEqual(rmatmul(1, _Dot()), "rev")

    def test_in_place(self):
        a = [1, 2]
        self.assertIs(iadd(a, [3]), a)
        self.assertEqual(a, [1, 2, 3])
        self.assertEqual(imul(5, 3), 15)          # int: no __imul__, fallback
        self.assertEqual(isub(5, 3), 2)
        self.assertEqual(itruediv(7, 2), 7 / 2)
        self.assertEqual(ifloordiv(7, 2), 7 // 2)
        self.assertEqual(imod(7, 3), 7 % 3)
        self.assertEqual(ipow(2, 5), 32)
        self.assertEqual(imatmul(_Dot(), 1), "fwd")

    def test_errors_propagate(self):
        with self.assertRaises(ZeroDivisionError):
            truediv(1, 0)
        with self.assertRaises(TypeError):
            add(1, "a")


class TestBitwise(unittest.TestCase):
    def test_binary(self):
        self.assertEqual(and_(12, 10), 12 & 10)
        self.assertEqual(or_(12, 10), 12 | 10)
        self.assertEqual(xor(12, 10), 12 ^ 10)
        self.assertEqual(lshift(1, 4), 1 << 4)
        self.assertEqual(rshift(16, 2), 16 >> 2)

    def test_reflected(self):
        self.assertEqual(rand(10, 12), (12).__rand__(10))
        self.assertEqual(rand(10, 12), 12 & 10)
        self.assertEqual(ror(10, 12), 12 | 10)
        self.assertEqual(rxor(10, 12), 12 ^ 10)
        self.assertEqual(rlshift(4, 1), (1).__rlshift__(4))
        self.assertEqual(rlshift(4, 1), 4 << 1)
        self.assertEqual(rrshift(16, 2), (2).__rrshift__(16))
        self.assertEqual(rrshift(16, 2), 16 >> 2)

    def test_in_place(self):
        self.assertEqual(iand(12, 10), 12 & 10)
        self.assertEqual(ior(12, 10), 12 | 10)
        self.assertEqual(ixor(12, 10), 12 ^ 10)
        self.assertEqual(ilshift(1, 4), 1 << 4)
        self.assertEqual(irshift(16, 2), 16 >> 2)


class TestComparisonAndUnary(unittest.TestCase):
    def test_comparisons(self):
        self.assertTrue(eq(1, 1))
        self.assertFalse(eq(1, 2))
        self.assertTrue(ne(1, 2))
        self.assertTrue(lt(1, 2))
        self.assertTrue(le(2, 2))
        self.assertTrue(gt(2, 1))
        self.assertTrue(ge(2, 2))
        self.assertEqual(lt(1, 2), 1 < 2)
        self.assertEqual(eq("a", "a"), "a" == "a")

    def test_unary(self):
        self.assertEqual(neg(5), -5)
        self.assertEqual(pos(5), +5)
        self.assertEqual(op_abs(-5), 5)
        self.assertEqual(op_abs(-2.5), 2.5)
        self.assertEqual(invert(5), ~5)
        self.assertIs(not_(True), False)
        self.assertIs(not_(0), True)


class TestProtocols(unittest.TestCase):
    def test_container(self):
        self.assertEqual(getitem([1, 2, 3], 1), [1, 2, 3][1])
        self.assertEqual(getitem({"k": 9}, "k"), 9)
        d = {"k": 1}
        self.assertIsNone(setitem(d, "k", 2))
        self.assertEqual(d, {"k": 2})
        seq = [1, 2, 3]
        self.assertIsNone(delitem(seq, 0))
        self.assertEqual(seq, [2, 3])
        self.assertTrue(contains([1, 2], 2))
        self.assertTrue(contains("abc", "b"))
        self.assertFalse(contains([1], 5))

    def test_sequence_functions(self):
        self.assertEqual(concat([1], [2, 3]), [1] + [2, 3])
        self.assertEqual(concat("ab", "cd"), "abcd")
        self.assertEqual(concat((1,), (2,)), (1, 2))
        self.assertEqual(countOf([1, 2, 2, 3], 2), 2)
        self.assertEqual(countOf([1, 2, 2, 3], 9), 0)
        self.assertEqual(indexOf([7, 8, 9], 9), 2)
        with self.assertRaises(ValueError):
            indexOf([7, 8], 1)

    def test_callable_protocols(self):
        self.assertEqual(call(len, [1, 2, 3]), 3)
        self.assertEqual(call(add, 2, 3), 5)
        self.assertIs(truth(0), False)
        self.assertIs(truth(""), False)
        self.assertIs(truth([1]), True)
        self.assertEqual(len_([1, 2, 3]), 3)
        self.assertEqual(len_("ab"), 2)
        self.assertEqual(length_hint([1, 2, 3]), 3)
        self.assertEqual(length_hint(iter([1, 2, 3])), 3)

        class NoHint:
            def __iter__(self):
                return iter(())

        self.assertEqual(length_hint(NoHint()), 0)
        self.assertEqual(length_hint(NoHint(), 7), 7)


class TestAsCallbacks(unittest.TestCase):
    def test_as_reduce_and_mapping(self):
        from functools import reduce
        self.assertEqual(reduce(add, [1, 2, 3, 4], 0), 10)
        self.assertEqual(list(map(mul, [1, 2, 3], [10, 20, 30])),
                         [10, 40, 90])

    def test_curried_positional(self):
        from functools import partial
        self.assertEqual(partial(mul, 2)(21), 42)
        self.assertEqual(partial(add, 10)(5), 15)


if __name__ == "__main__":
    unittest.main()
