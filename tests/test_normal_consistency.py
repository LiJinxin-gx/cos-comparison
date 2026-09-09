# -*- coding: utf-8 -*-
"""Cross-backend consistency on normal (non-empty) data: the identical
workload must produce identical results on every backend (bit for bit
where the values are integers/doubles produced by the same C formulas)."""

import unittest

import testutil

WORKLOAD = (
    "import json\n"
    "from cos_comparison import core\n"
    "results = {}\n"
    # arithmetic on 1D tensors
    "a = core.create_void_list((5,))\n"
    "b = core.create_void_list((5,))\n"
    "for i in range(5):\n"
    "    core.set_item(a, (i,), float(i+1))\n"
    "    core.set_item(b, (i,), float(i+1))\n"
    "results['add_1d'] = [core.get_item(a+b, (i,)) for i in range(5)]\n"
    "results['sub_1d'] = [core.get_item(a-b, (i,)) for i in range(5)]\n"
    "results['mul_1d'] = [core.get_item(a*b, (i,)) for i in range(5)]\n"
    "results['add_scalar'] = [core.get_item(a+2.0, (i,)) for i in range(5)]\n"
    "results['rmul_scalar'] = [core.get_item(3.0*a, (i,)) for i in range(5)]\n"
    "results['neg'] = [core.get_item(-a, (i,)) for i in range(5)]\n"
    "results['abs'] = float(abs(a))\n"
    "results['mean'] = float(a.mean())\n"
    "results['variance'] = float(a.variance())\n"
    # 2D arithmetic
    "c = core.create_void_list((3,3))\n"
    "d = core.create_void_list((3,3))\n"
    "for i in range(3):\n"
    "    for j in range(3):\n"
    "        core.set_item(c, (i,j), float(i*3+j+1))\n"
    "        core.set_item(d, (i,j), 1.0)\n"
    "results['add_2d'] = [core.get_item(c+d, (i,j)) for i in range(3) for j in range(3)]\n"
    # slicing
    "s = a[1:4]\n"
    "results['slice'] = [core.get_item(s, (i,)) for i in range(3)]\n"
    # infer_shape
    "results['infer_list'] = list(core.infer_shape([[1,2],[3,4]]))\n"
    "results['infer_tensor'] = list(core.infer_shape(a))\n"
    # load_as_default_data
    "data = core.load_as_default_data([[1.0,2.0,3.0],[4.0,5.0,6.0]])\n"
    "results['load_shape'] = list(data.shape)\n"
    # passive similarity map
    "p = core.cos_comparison_passive([1.0,2.0,3.0,4.0], window_size=(3,))\n"
    "results['passive_len'] = len(p) if hasattr(p, '__len__') else 'n/a'\n"
    "d2 = core.create_void_list((8, 8))\n"
    "for i in range(8):\n"
    "    for j in range(8):\n"
    "        d2[i, j] = float((i * 3 + j) % 5) * 0.5\n"
    "r = core.cos_comparison_passive(d2, window_size=(3, 3), d=(0, 1))\n"
    "results['passive_map'] = [float(r[i, j]) for i in range(r.shape[0]) for j in range(r.shape[1])]\n"
    "results['passive_shape'] = list(r.shape)\n"
    # whole-tensor cos
    "results['cos_self'] = float(core.cos(d2, d2))\n"
    "print(json.dumps(results))\n"
)


class TestNormalConsistency(unittest.TestCase):
    def test_all_backends_bit_identical(self):
        results = {}
        for backend in testutil.BACKENDS:
            rc, out, err = testutil.run_backend(backend, WORKLOAD)
            self.assertEqual(rc, 0, "%s crashed: %s" % (backend, err[-500:]))
            data = testutil.json_result(out)
            self.assertTrue(data, "%s returned no data" % backend)
            results[backend] = data
        ref = results[testutil.BACKENDS[0]]
        for backend, data in results.items():
            self.assertEqual(data, ref,
                             "%s differs from %s" % (backend, testutil.BACKENDS[0]))


if __name__ == "__main__":
    unittest.main()
