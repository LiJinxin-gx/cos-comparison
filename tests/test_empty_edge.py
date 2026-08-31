# -*- coding: utf-8 -*-
"""Degenerate / empty / edge inputs across all three backends.

Every case runs in a fresh subprocess (so a segfault in one never kills
the suite) with a sanitised environment and a neutral cwd (no source
tree shadowing, no PYTHONPATH pollution - see testutil.py).

Invariants asserted per case:
  * no backend crashes (exit code must not be negative / killed);
  * the outcome class (OK vs RAISED) is identical on all three backends.
"""

import unittest

import testutil

BACKEND_IMPORTS = [
    ("py", "from cos_comparison.core import cos_comparison as m"),
    ("pydll", "from cos_comparison.core import cos_comparison_pydll as m"),
    ("ctypes", "import cos_comparison.core.cos_comparison_c as m"),
]

CASES = [
    # --- Construction ---
    ("empty_list", "t=m.vector_map_as_tensor(vector=[]); print(t.shape)"),
    ("empty_tuple", "t=m.vector_map_as_tensor(vector=()); print(t.shape)"),
    ("empty_nested", "t=m.vector_map_as_tensor(vector=[[]]); print(t.shape)"),
    ("empty_with_shape0", "t=m.vector_map_as_tensor(vector=[],shape=(0,)); print(t.shape)"),
    ("shape_zero_3d", "t=m.vector_map_as_tensor(vector=[],shape=(0,3,2)); print(t.shape)"),
    ("scalar_int", "t=m.vector_map_as_tensor(vector=5); print(t.shape)"),
    ("scalar_float", "t=m.vector_map_as_tensor(vector=3.14); print(t.shape)"),
    ("none_input", "t=m.vector_map_as_tensor(vector=None); print(t.shape)"),
    ("bool_input", "t=m.vector_map_as_tensor(vector=True); print(t.shape)"),
    ("empty_with_default", "t=m.create_void_list((0,)); print(t.shape)"),
    ("empty_2d", "t=m.create_void_list((0,3)); print(t.shape)"),
    # --- Operations on empty tensors ---
    ("len_empty", "t=m.vector_map_as_tensor(vector=[],shape=(0,)); print(len(t))"),
    ("len_empty_default", "t=m.vector_map_as_tensor(vector=[]); print(len(t))"),
    ("iter_empty", "t=m.vector_map_as_tensor(vector=[],shape=(0,)); print(list(t))"),
    ("shape_empty_2d", "t=m.create_void_list((0,3)); print(t.shape)"),
    # --- Core algorithms with empty input ---
    ("cos_passive_empty", "m.cos_comparison_passive([])"),
    ("cos_active_empty", "m.cos_comparison_active([], [])"),
    ("mean_local_empty", "m.mean_local([])"),
    ("local_variance_empty", "m.local_variance([])"),
    ("cos_passive_zero_shape", "t=m.create_void_list((0,)); m.cos_comparison_passive(t)"),
    ("mean_local_zero_shape", "t=m.create_void_list((0,)); m.mean_local(t)"),
    # --- load_as_default_data ---
    ("load_as_default_empty", "m.load_as_default_data([])"),
    ("load_as_default_empty_nested", "m.load_as_default_data([[]])"),
    ("load_as_default_zero_tensor", "t=m.create_void_list((0,)); print(m.load_as_default_data(t).shape)"),
    # --- load_data ---
    ("load_data_both_empty", "print(m.load_data([], []))"),
    ("load_data_src_empty", "t=m.create_void_list((3,)); print(m.load_data([], t))"),
    ("load_data_tgt_empty", "t=m.create_void_list((3,)); print(m.load_data(t, []))"),
    # --- infer_shape ---
    ("infer_shape_empty", "print(m.infer_shape([]))"),
    ("infer_shape_empty_nested", "print(m.infer_shape([[]]))"),
    ("infer_shape_none", "print(m.infer_shape(None))"),
    ("infer_shape_scalar", "print(m.infer_shape(5))"),
    # --- multiple_chain / add_chain ---
    ("multiple_chain_empty", "print(m.multiple_chain([]))"),
    ("add_chain_empty", "print(m.add_chain([]))"),
    # --- Nested empty lists ---
    ("nested_empty_2d", "t=m.vector_map_as_tensor(vector=[[],[]]); print(t.shape)"),
    # --- Indexing empty tensor (should raise, not crash) ---
    ("index_empty_tensor", "t=m.create_void_list((0,)); t[0]"),
    # --- Buffer protocol on empty tensor ---
    ("buffer_empty", "t=m.create_void_list((0,)); mv=memoryview(t); print(mv.shape, len(mv))"),
    # --- String input (should not crash) ---
    ("string_input", "t=m.vector_map_as_tensor(vector='abc'); print(t.shape)"),
    # --- Arithmetic on empty tensors ---
    ("add_empty", "a=m.create_void_list((0,)); b=m.create_void_list((0,)); r=a+b; print(r.shape)"),
    ("sub_empty", "a=m.create_void_list((0,)); b=m.create_void_list((0,)); r=a-b; print(r.shape)"),
    ("mul_empty", "a=m.create_void_list((0,)); b=m.create_void_list((0,)); r=a*b; print(r.shape)"),
    ("add_scalar_empty", "a=m.create_void_list((0,)); r=a+1.0; print(r.shape)"),
    ("mul_scalar_empty", "a=m.create_void_list((0,)); r=a*2.0; print(r.shape)"),
    # --- Slicing empty tensors ---
    ("slice_empty_full", "a=m.create_void_list((0,)); print(a[:].shape)"),
    ("slice_empty_step", "a=m.create_void_list((0,)); print(a[::2].shape)"),
    ("slice_empty_2d", "a=m.create_void_list((0,3)); print(a[:,:].shape)"),
    # --- Indexing empty tensors ---
    ("index_empty_neg", "a=m.create_void_list((0,)); a[-1]"),
    ("index_empty_2d", "a=m.create_void_list((0,3)); a[0,0]"),
    # --- set_item on empty ---
    ("setitem_empty", "a=m.create_void_list((0,)); m.set_item(a,(0,),1.0)"),
    # --- cos_full / chain_compute with empty ---
    ("cos_full_empty", "a=m.create_void_list((0,)); b=m.create_void_list((0,)); m.cos_full(a,b)"),
    ("cos_full_empty_2d", "a=m.create_void_list((0,3)); b=m.create_void_list((0,3)); m.cos_full(a,b)"),
    ("chain_compute_empty", "a=m.create_void_list((0,)); m.chain_compute(a, a, a, [(0,1)])"),
    # --- load_as_default_data with parameters on empty ---
    ("load_as_default_empty_params", "m.load_as_default_data([], start=(0,), shape=(0,), step=(1,))"),
    ("load_as_default_nested_empty", "m.load_as_default_data([[],[],[]])"),
    # --- deeply nested empty / bool / zero-size with strides ---
    ("deep_empty", "t=m.vector_map_as_tensor(vector=[[[[]]]]); print(t.shape)"),
    ("bool_list", "t=m.vector_map_as_tensor(vector=[True,False,True]); print(t.shape)"),
    ("zero_strides", "t=m.create_void_list((0,)); print(t.strides)"),
    # --- repr / properties of empty tensors ---
    ("repr_empty", "a=m.create_void_list((0,)); print(repr(a)[:40])"),
    ("dim_empty", "a=m.create_void_list((0,)); print(a.dimension)"),
    ("dim_empty_2d", "a=m.create_void_list((0,3)); print(a.dimension)"),
    ("tensor_size_empty", "a=m.create_void_list((0,)); print(a.tensor_size)"),
    # --- multiple_chain / add_chain with zero ---
    ("multiple_chain_zero", "print(m.multiple_chain((0,3)))"),
    ("add_chain_zero", "print(m.add_chain((0,3)))"),
    # --- empty vector + non-empty shape ---
    ("empty_vec_nonempty_shape", "t=m.vector_map_as_tensor(vector=[], shape=(3,)); print(t.shape)"),
    # --- create_void_list 3d with zero ---
    ("create_3d_zero", "t=m.create_void_list((2,0,3)); print(t.shape)"),
    ("infer_shape_empty_tensor", "t=m.create_void_list((0,)); print(m.infer_shape(t))"),
    ("load_data_shape_mismatch", "a=m.create_void_list((2,2)); b=m.create_void_list((3,3)); print(m.load_data(a,b))"),
    ("inf_scalar", "t=m.vector_map_as_tensor(vector=float('inf')); print(t.shape)"),
    ("empty_tuple_shape", "t=m.vector_map_as_tensor(vector=[], shape=()); print(t.shape)"),
]


class TestEmptyAndEdgeInputs(unittest.TestCase):
    """Systematic degenerate-input sweep; see module docstring for the
    invariants."""

    def test_no_crash_and_consistent_class_across_backends(self):
        failures = []
        for name, code in CASES:
            outcomes = {}
            for backend, imp in BACKEND_IMPORTS:
                rc, out, err = testutil.run_probe(imp, code)
                if rc < 0:
                    failures.append("%s crashed on %s (rc=%d): %s"
                                    % (name, backend, rc, err[-200:]))
                    continue
                outcomes[backend] = "ok" if rc == 0 else "raised"
            if len(set(outcomes.values())) != 1:
                failures.append("%s outcome class diverges: %r"
                                % (name, outcomes))
        self.assertEqual(failures, [], "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
