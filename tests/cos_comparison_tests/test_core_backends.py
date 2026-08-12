# -*- coding: utf-8 -*-
"""Backend loading and cross-backend parity.

Each backend runs in a *fresh* interpreter (see common.py): inside one
process, switching backends leaves objects of the previous backend
behind and the new backend's dispatch rejects them
(TypeError: takes at most 19 arguments (24 given)).
"""

import unittest

from cos_comparison import core

from . import common


class TestBackendControl(unittest.TestCase):
    def test_mode_tuples(self):
        mode = core.get_mode()
        self.assertIsInstance(mode, tuple)
        self.assertIn(".cos_comparison", mode)  # pure Python is mandatory
        self.assertEqual(core.get_available_backends(),
                         (".cos_comparison_pydll", ".cos_comparison_c",
                          ".cos_comparison"))

    def test_set_mode_accepts_str_and_list(self):
        core.set_mode(".cos_comparison")
        core.set_mode([".cos_comparison"])
        core.set_mode(".cos_comparison_pydll")  # restore default

    def test_set_mode_bad_type(self):
        with self.assertRaises(TypeError):
            core.set_mode(123)
        with self.assertRaises(TypeError):
            core.set_mode((1, 2))

    def test_set_mode_unknown_backend_raises(self):
        with self.assertRaises(ImportError):
            core.set_mode(".does_not_exist")

    def test_set_mode_recovers_after_failure(self):
        try:
            core.set_mode(".does_not_exist")
        except ImportError:
            pass
        # loader must restore the previous working backend
        r = core.cos_comparison_passive(
            core.create_void_list((3, 3)), window_size=(1, 1))
        self.assertIsNotNone(r)


class TestBackendParity(unittest.TestCase):
    """Run the identical workload on every backend in a fresh process and
    compare the results bit for bit."""

    WORKLOAD = (
        "data = core.create_void_list((8, 8))\n"
        "for i in range(8):\n"
        "    for j in range(8):\n"
        "        data[i, j] = float((i * 3 + j) % 5) * 0.5\n"
        "r = core.cos_comparison_passive(data, window_size=(3, 3), d=(0, 1))\n"
        "vals = [float(r[i, j]) for i in range(r.shape[0]) for j in range(r.shape[1])]\n"
        "print(json.dumps({'vals': vals, 'shape': list(r.shape)}))\n"
    )

    def test_all_backends_report_values(self):
        results = {}
        for backend in common.BACKENDS:
            code, out, err = common.run_backend(backend, self.WORKLOAD)
            self.assertEqual(code, 0, "%s crashed: %s" % (backend, err[-500:]))
            data = json_result(out)
            self.assertTrue(data, "%s returned no data" % backend)
            results[backend] = data["vals"]
            self.assertGreater(len(data["vals"]), 0)
        # parity across backends
        ref = results[common.BACKENDS[0]]
        for backend, vals in results.items():
            self.assertEqual(vals, ref,
                             "%s differs from %s" % (backend, common.BACKENDS[0]))

    def test_shapes_match(self):
        shapes = {}
        for backend in common.BACKENDS:
            code, out, err = common.run_backend(backend, self.WORKLOAD)
            shapes[backend] = json_result(out)["shape"]
        self.assertEqual(len(set(tuple(s) for s in shapes.values())), 1,
                         "shapes differ: %r" % shapes)

    def test_pure_python_reference_matches_hand_formula(self):
        # plain cosine similarity: A=[1,0], B=[0.5,0.5]
        # -> 0.5 / (1 * sqrt(0.5)) = 1/sqrt(2)
        body = (
            "a = core.create_void_list((2, 1))\n"
            "b = core.create_void_list((2, 1))\n"
            "a[0,0] = 1.0; a[1,0] = 0.0\n"
            "b[0,0] = 0.5; b[1,0] = 0.5\n"
            "s = core.cos(a, b)\n"
            "print(json.dumps({'s': float(s)}))\n"
        )
        code, out, err = common.run_backend(".cos_comparison", body)
        s = json_result(out)["s"]
        self.assertAlmostEqual(s, 1.0 / 2.0 ** 0.5, places=9)

    def test_pydll_and_ctypes_available(self):
        # the wheel we built ships both C artifacts; verify they load
        for backend in (".cos_comparison_pydll", ".cos_comparison_c"):
            code, out, err = common.run_backend(
                backend, "print(json.dumps({'r': 1}))\n")
            self.assertEqual(code, 0, "%s failed to load: %s"
                             % (backend, err[-500:]))


def json_result(out):
    for line in out.splitlines():
        if line.startswith("{"):
            import json
            return json.loads(line)
    return None


class TestKnownDivergences(unittest.TestCase):
    """Documented backend divergences (observed on cos-comparison
    0.4.0).  These are *current behavior* checks, not desired-behavior
    checks: the three backends are NOT 100% interchangeable.

    1. scalar tensor (shape=()):
         pure Python   -> tensor works (v[()] == default)
         ctypes        -> returns a bare float, not a tensor
         pydll         -> heap corruption on interpreter exit
                         (STATUS_HEAP_CORRUPTION, exit code 0xC0000374)
    2. scalar in-place add (t += 1.0):
         pure Python   -> ok
         pydll         -> TypeError "operands must be vector_map_as_tensor"
    3. unexpected keyword arguments:
         pure Python   -> accepted (silently)
         pydll         -> TypeError "unexpected keyword argument"
    4. active without kernel:
         pure Python   -> ValueError
         pydll         -> TypeError "missing required argument"
    5. abs(tensor):    every backend returns a plain float (the L2
                       norm), not a tensor - consistent, but surprising
    6. infer_shape(scalar): pure Python -> None, pydll -> ()
    7. same-process backend switch:
         objects from the old backend are rejected by the new backend's
         dispatch (TypeError: takes at most 19 arguments (24 given)).
    """

    def test_scalar_tensor_pure_python(self):
        code, out, err = common.run_backend(
            ".cos_comparison",
            "v = core.create_void_list((), default=5.0)\n"
            "print(json.dumps({'shape': list(v.shape), 'v': float(v[()])}))\n")
        data = json_result(out)
        self.assertEqual(data, {"shape": [], "v": 5.0})

    def test_scalar_tensor_ctypes_returns_float(self):
        code, out, err = common.run_backend(
            ".cos_comparison_c",
            "v = core.create_void_list((), default=5.0)\n"
            "print(json.dumps({'type': type(v).__name__}))\n")
        data = json_result(out)
        self.assertEqual(data.get("type"), "float",
                         "ctypes scalar create: %r" % data)

    def test_scalar_tensor_pydll_crashes(self):
        code, out, err = common.run_backend(
            ".cos_comparison_pydll",
            "v = core.create_void_list((), default=5.0)\n"
            "print(json.dumps({'shape': list(v.shape)}))\n")
        self.assertNotEqual(code, 0,
                            "pydll scalar tensor did not crash (exit %d)"
                            % code)

    def test_scalar_iadd_divergence(self):
        for backend, expected_ok in ((".cos_comparison", True),
                                     (".cos_comparison_pydll", False)):
            body = ("t = core.create_void_list((2, 2), default=2.0)\n"
                    "try:\n"
                    "    t += 1.0\n"
                    "    print(json.dumps({'ok': True}))\n"
                    "except Exception as e:\n"
                    "    print(json.dumps({'ok': False, 'e': type(e).__name__}))\n")
            code, out, err = common.run_backend(backend, body)
            data = json_result(out)
            self.assertEqual(data.get("ok"), expected_ok,
                             "%s iadd: %r" % (backend, data))

    def test_abs_returns_float_consistently(self):
        # abs(tensor) does not return a tensor on any backend in 0.4.0:
        # it returns a plain float (the L2 norm).  Both C backends agree
        # with pure Python on the value.
        results = {}
        for backend in common.BACKENDS:
            body = ("t = core.create_void_list((2, 2), default=-2.0)\n"
                    "try:\n"
                    "    r = abs(t)\n"
                    "    print(json.dumps({'v': float(r), 't': type(r).__name__}))\n"
                    "except Exception as e:\n"
                    "    print(json.dumps({'e': type(e).__name__}))\n")
            code, out, err = common.run_backend(backend, body)
            data = json_result(out)
            self.assertEqual(data.get("t"), "float", "%s abs: %r"
                             % (backend, data))
            results[backend] = data.get("v")
        self.assertEqual(len(set(results.values())), 1, results)

    def test_infer_shape_scalar_divergence(self):
        for backend, expected in ((".cos_comparison", None),
                                  (".cos_comparison_pydll", ())):
            body = ("r = core.infer_shape(3.14)\n"
                    "print(json.dumps({'r': list(r) if r is not None else None}))\n")
            code, out, err = common.run_backend(backend, body)
            data = json_result(out)
            got = None if data.get("r") is None else tuple(data["r"])
            self.assertEqual(got, expected,
                             "%s infer_shape: %r" % (backend, data))

    def test_kwargs_strictness_divergence(self):
        body = ("v = core.create_void_list((4, 4))\n"
                "try:\n"
                "    r = core.cos_comparison_passive(v, window_size=(3, 3),\n"
                "                                     unknown_option=1)\n"
                "    print(json.dumps({'ok': True}))\n"
                "except Exception as e:\n"
                "    print(json.dumps({'ok': False, 'e': type(e).__name__}))\n")
        for backend, expected_ok in ((".cos_comparison", True),
                                     (".cos_comparison_pydll", False)):
            code, out, err = common.run_backend(backend, body)
            data = json_result(out)
            self.assertEqual(data.get("ok"), expected_ok,
                             "%s kwargs: %r" % (backend, data))

    def test_active_no_kernel_exception_type(self):
        for backend, expected in ((".cos_comparison", "ValueError"),
                                  (".cos_comparison_pydll", "TypeError")):
            body = ("v = core.create_void_list((4, 4))\n"
                    "try:\n"
                    "    core.cos_comparison_active(v)\n"
                    "    print(json.dumps({'e': 'None'}))\n"
                    "except Exception as ex:\n"
                    "    print(json.dumps({'e': type(ex).__name__}))\n")
            code, out, err = common.run_backend(backend, body)
            self.assertEqual(json_result(out).get("e"), expected,
                             "%s no-kernel: %r" % (backend, json_result(out)))


if __name__ == "__main__":
    unittest.main()
