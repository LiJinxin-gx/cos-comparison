# -*- coding: utf-8 -*-
"""
cos_comparison test package.

Runs against the *installed* cos-comparison (site-packages), not the
source tree.  Always execute with the venv_test interpreter and
``python -E`` (or set an empty PYTHONPATH) so that a user-level
PYTHONPATH (e.g. D:\\pysite with an older cos-comparison) cannot
shadow the package under test.

Modules:
  test_imports        - every subpackage imports and exports its API
  test_core_tensor    - vector_map_as_tensor data structure
  test_core_shape     - infer_shape / create_void_list / dimensions
  test_core_data      - load_as_default_data / load_data
  test_core_algorithms- passive / active / cos / mean / variance
  test_core_backends  - backend switching + cross-backend parity
  test_interface_api  - interface layer (process / parallel / async)
  test_layers         - data / sense / memory / brain / action / ...
"""
