"""Unified value grammar tests: Python-style literals, built-in operator
infix expressions, nested functional calls, value conditions and bare
value lines - across the shell / batch / app executors.  Every parser
stage is iterative (deep nesting never recurses).  Stdlib only, non-GUI.
"""

import unittest

from cos_comparison.shell_tool import shell as SH
from cos_comparison.shell_tool.value import (
    ValueEnv,
    compile_value,
)


def run_value(text, data=None, vars=None, resolve=None):
    env = ValueEnv(data=data, vars=vars if vars is not None else {},
                   resolve=resolve, var_index={})
    return compile_value(text).execute(env)


class TestPythonLiterals(unittest.TestCase):
    def test_tuples(self):
        self.assertEqual(run_value("(1, 2, 3)"), (1, 2, 3))
        self.assertEqual(run_value("()"), ())
        self.assertEqual(run_value("(1,)"), (1,))
        self.assertEqual(run_value("(1, (2, 3))"), (1, (2, 3)))
        self.assertEqual(run_value("('a', 'b')"), ("a", "b"))
        self.assertEqual(run_value('("a,b", 1)'), ("a,b", 1))

    def test_grouping_is_not_a_tuple(self):
        # Python semantics: a single element without a comma is a group
        self.assertEqual(run_value("(5)"), 5)
        self.assertEqual(run_value("(1+2)"), 3)

    def test_lists(self):
        self.assertEqual(run_value("[1, 2, 3]"), [1, 2, 3])
        self.assertEqual(run_value("[]"), [])
        self.assertEqual(run_value("[data[0], 2]", data=[10]), [10, 2])

    def test_dicts(self):
        self.assertEqual(run_value("{1: 'a', 2: 10}"), {1: "a", 2: 10})
        self.assertEqual(run_value("{}"), {})
        self.assertEqual(run_value('{"k": data[0], "s": [1, 2]}',
                                   data=[10]),
                         {"k": 10, "s": [1, 2]})

    def test_space_separated_forms_rejected(self):
        # Python-style only: multi-element groups without commas are
        # nested calls (heads must be names) or errors
        with self.assertRaises(ValueError):
            compile_value("[1 2]")
        with self.assertRaises(ValueError):
            compile_value("(1 2 3)")


class TestNestedCalls(unittest.TestCase):
    def test_call_resolution(self):
        import operator
        env = ValueEnv(data=[10], resolve=lambda n: {
            "add": operator.add, "sum2": lambda a, b: a + b}.get(n))
        self.assertEqual(
            compile_value("(add 1 2)").execute(env), 3)
        self.assertEqual(
            compile_value("(add data[0] 5)").execute(env), 15)
        self.assertEqual(
            compile_value("(sum2 (add 1 2) 3)").execute(env), 6)

    def test_symbol_operator_as_head(self):
        import operator
        env = ValueEnv(data=[4], resolve=lambda n: {
            "+": operator.add, "*": operator.mul,
            "==": operator.eq}.get(n))
        self.assertEqual(compile_value("(+ 2 3)").execute(env), 5)
        self.assertEqual(compile_value("(* (+ 1 2) 4)").execute(env), 12)
        self.assertEqual(compile_value("(== 3 3)").execute(env), True)


class TestInfixExpressions(unittest.TestCase):
    def test_operators(self):
        data = [10, 3]
        self.assertEqual(run_value("data[0]+1", data), 11)
        self.assertEqual(run_value("data[0]-data[1]", data), 7)
        self.assertEqual(run_value("data[0]*data[1]", data), 30)
        self.assertEqual(run_value("data[0]/2", data), 5.0)
        self.assertEqual(run_value("data[0]//3", data), 3)
        self.assertEqual(run_value("data[0]%3", data), 1)
        self.assertEqual(run_value("2**10", data), 1024)
        self.assertEqual(run_value("1<<4", data), 16)
        self.assertEqual(run_value("5&-3", data), 5 & -3)

    def test_precedence_and_parentheses(self):
        self.assertEqual(run_value("1+2*3"), 7)
        self.assertEqual(run_value("(1+2)*3"), 9)
        self.assertEqual(run_value("2**3**2"), 2 ** 3 ** 2)
        self.assertEqual(run_value("--5"), 5)
        self.assertEqual(run_value("-data[0]", [5]), -5)

    def test_comparisons_and_logic(self):
        self.assertEqual(run_value("1<2"), True)
        self.assertEqual(run_value("data[0] <= 30 and data[0] >= 20",
                                   [25]), True)
        self.assertEqual(run_value("x==1 or x==5", vars={"x": 5}), True)
        self.assertEqual(run_value("not x==1", vars={"x": 5}), True)
        self.assertEqual(run_value("1 <= x < 5", vars={"x": 3}), True)

    def test_words_are_variables_then_strings(self):
        self.assertEqual(run_value("x+1", vars={"x": 5}), 6)
        self.assertEqual(run_value("plain", vars={}), "plain")


class TestDeepNestingIterative(unittest.TestCase):
    def test_two_thousand_grouping_parens(self):
        # Python semantics: every level is a group (one element, no
        # comma), so 2000 nested groups fold down to the inner value -
        # and the parser never recurses
        deep = "(" * 2000 + "1" + ")" * 2000
        self.assertEqual(run_value(deep), 1)

    def test_two_thousand_nested_calls_and_groups(self):
        import operator
        env = ValueEnv(resolve=lambda n: operator.add if n == "add"
                       else None)
        head = "(add 1 1)"
        for _ in range(500):
            head = "(add %s 1)" % head
        self.assertEqual(compile_value(head).execute(env), 502)

    def test_two_thousand_deep_expression_parens(self):
        expr = "(" * 2000 + "1" + ")" * 2000
        self.assertEqual(run_value(expr), 1)


class TestShellUnified(unittest.TestCase):
    """Instruction-level integration through the shell namespace."""

    def _fresh(self):
        SH._ns["vars"].clear()
        SH._ns["_var_index"].clear()
        SH._ns["_var_seq"] = 0

    def test_expression_lines_and_nested_calls(self):
        self._fresh()
        txt = (
            "let a 2+3 -> 1\n"
            "let b (* a 4) -> 2\n"
            "+ a b -> 3\n"
            "data[3] * 2 + 1 -> 4\n"
            "let c [1, 2, a] -> 5\n"
            "let d (1, 2.5, 'x') -> 6\n"
            "let e {k: data[4], 's': c} -> 7\n"
        )
        items = SH.parse_instruction_file(txt, SH.resolve_callable)
        data, stats = SH.execute_instruction_items(items, data={})
        self.assertEqual(stats["run"], 7)
        self.assertEqual(data[4], 51)
        self.assertEqual(SH.variables()["b"], 20)
        self.assertEqual(SH.variables()["c"], [1, 2, 5])
        self.assertEqual(SH.variables()["d"], (1, 2.5, "x"))
        self.assertEqual(SH.variables()["e"], {"k": 51, "s": [1, 2, 5]})

    def test_symbol_instruction_head(self):
        self._fresh()
        items = SH.parse_instruction_file(
            "+ 1 2 -> 0\n* 3 4 -> 1", SH.resolve_callable)
        data, _ = SH.execute_instruction_items(items, data={})
        self.assertEqual(data, {0: 3, 1: 12})

    def test_while_with_value_condition(self):
        self._fresh()
        txt = (
            "let n 0 -> 1\n"
            "WHILE n < 3\n"
            "let n n+1 -> 2\n"
            "END\n"
        )
        items = SH.parse_instruction_file(txt, SH.resolve_callable)
        data, _ = SH.execute_instruction_items(items, data={})
        self.assertEqual(SH.variables()["n"], 3)
        self.assertEqual(data[2], "3")

    def test_if_with_value_condition(self):
        self._fresh()
        txt = (
            "IF (== 3 3)\n"
            "let r ok -> 3\n"
            "ELSE\n"
            "let r bad -> 3\n"
            "END\n"
            "IF data[9] > 5\n"
            "let big True -> 4\n"
            "ELSE\n"
            "let big False -> 4\n"
            "END\n"
        )
        items = SH.parse_instruction_file(txt, SH.resolve_callable)
        data, _ = SH.execute_instruction_items(items, data={9: 7})
        self.assertEqual(SH.variables()["r"], "ok")
        self.assertEqual(SH.variables()["big"], True)

    def test_bare_value_line(self):
        self._fresh()
        items = SH.parse_instruction_file(
            "1 + 2 * 3 -> 0\n(max 3 9) -> 1\n", SH.resolve_callable)
        data, _ = SH.execute_instruction_items(items, data={})
        self.assertEqual(data.get(0), 7)
        self.assertEqual(data.get(1), 9)

    def test_infix_inside_batch(self):
        from cos_comparison.shell_tool.batch import run_batch
        self._fresh()
        data, _ = run_batch([
            "(data[0]+1)*2 -> 0",
            "== data[0] 22 -> 2",
        ], data={0: 10})
        self.assertEqual(data.get(0), 22)
        self.assertEqual(data.get(2), True)

    def test_app_reader_expression_capabilities(self):
        from cos_comparison.app import Docker
        self._fresh()

        def store(data, key, value):
            data[key] = value
            return value

        d = Docker(namespace={"store": store})
        d.operate_pool = d.read_operate_flow(
            "let n 6 -> 10\n"
            "store 9 (n*7)+1 -> 11\n")
        drv = d.run()
        self.assertTrue(drv.wait(timeout=5))
        self.assertEqual(d.data_pool.get(9), 43)


class TestVariableUnpackForms(unittest.TestCase):
    def test_unpack_and_map_unpack(self):
        def pairs(*values):
            return sum(values)

        SH.register_callable("pairs", pairs)
        try:
            items = SH.parse_instruction_file(
                "list (1, 2, 3) -> 0\n"
                "pairs %data[0] -> 1\n", SH.resolve_callable)
            data, _ = SH.execute_instruction_items(items, data={})
            self.assertEqual(data, {0: [1, 2, 3], 1: 6})
        finally:
            SH.ns_delete(["pairs"])

    def test_pointer_forms(self):
        SH._ns["vars"].clear()
        SH._ns["_var_index"].clear()
        SH._ns["_var_seq"] = 0
        SH.ns_set("x", "5")
        env = ValueEnv(data=[], vars=SH.variables(),
                       var_index=SH.variable_indexes())
        self.assertEqual(compile_value("&x").execute(env),
                         SH.variable_indexes()["x"])
        self.assertEqual(compile_value("*&x").execute(env), 5)
        SH.ns_delete(["x"])


if __name__ == "__main__":
    unittest.main()
