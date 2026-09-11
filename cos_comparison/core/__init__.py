# core/__init__.py
"""
Backend loader for cos_comparison: loads the best available backend in
priority order and forwards all attributes to it (high-frequency core APIs
are hot-injected into the module namespace for runtime performance).

Backend model (v0.5.0):

  * core/config.json is an ORDERED LIST - the list order is the import
    (priority) order, the C extension first by default; every entry maps
    a CALL NAME (the "name" - anything set_mode accepts) to its import
    module (the package-relative "module").  Several entries may point at
    the same module - that is how legacy call names stay compatible:
    "c", ".cos_comparison_pydll" and the ctypes-era call name
    ".cos_comparison_c" all load ".cos_comparison_pydll"; "py" and
    ".cos_comparison" load ".cos_comparison".
  * importing tries the list in order (each module once - the first entry
    naming a module represents it); the pure Python module stays the
    mandatory final fallback.
  * the ctypes backend and its module are removed: the compiled
    C extension (core/cos_comparison_pydll.<tag>.pyd, built from
    core/include/cos_comparison_pydll.c) IS the C backend.

Configuration example::

    [{"name": "c", "module": ".cos_comparison_pydll"},
     {"name": ".cos_comparison_pydll", "module": ".cos_comparison_pydll"},
     {"name": ".cos_comparison_c", "module": ".cos_comparison_pydll"},
     {"name": "py", "module": ".cos_comparison"},
     {"name": ".cos_comparison", "module": ".cos_comparison"}]
"""

import importlib
import json
import os.path
from typing import Any, Dict, List, Optional, Set, Tuple

# -------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

# Default call-name list (ORDER = import priority): the compiled C
# extension first, then the pure Python core and the legacy call names.
_DEFAULT_BACKENDS = [
    {"name": "c", "module": ".cos_comparison_pydll"},
    {"name": ".cos_comparison_pydll", "module": ".cos_comparison_pydll"},
    {"name": ".cos_comparison_c", "module": ".cos_comparison_pydll"},
    {"name": "py", "module": ".cos_comparison"},
    {"name": ".cos_comparison", "module": ".cos_comparison"},
]

_BACKEND_ORDER: Tuple[str, ...] = ()      # import order (one entry per module)
_BACKEND_NAMES: Tuple[str, ...] = ()      # all configured call names, in order
_BACKENDS: Dict[str, Dict[str, Any]] = {}  # call name -> {"module"}
_PURE_MODULE: str = ".cos_comparison"


def _load_config() -> None:
    """Read config.json (an ordered list) and build the backend tables."""
    global _BACKEND_ORDER, _BACKEND_NAMES, _BACKENDS, _PURE_MODULE
    raw = _DEFAULT_BACKENDS

    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
            if isinstance(config, list):
                raw = config
        except Exception:
            # Fall back to defaults if the config is missing or invalid
            raw = _DEFAULT_BACKENDS

    backends = {}
    names = []
    order = []
    seen_modules = set()
    for item in raw:
        if isinstance(item, str):
            name = item
            module = item
        elif isinstance(item, dict):
            name = item.get("name")
            module = item.get("module") or name
        else:
            continue
        if not name:
            continue
        if not module.startswith("."):
            module = "." + module
        backends[name] = {"module": module}
        names.append(name)
        if module not in seen_modules:
            seen_modules.add(module)
            order.append(name)

    # Ensure the pure Python module is present as the final fallback
    if not any(entry["module"] == ".cos_comparison"
               for entry in backends.values()):
        backends["py"] = {"module": ".cos_comparison"}
        names.append("py")
        order.append("py")

    _PURE_MODULE = ".cos_comparison"
    _BACKENDS = backends
    _BACKEND_ORDER = tuple(order)
    _BACKEND_NAMES = tuple(names)


_load_config()

# -------------------------------------------------------------------
# 2. Backend loading
# -------------------------------------------------------------------
_backend: Dict[str, Any] = {}
_current_backend_name: Optional[str] = None
_module_globals = globals()

# Every backend public export is hot-injected into the module namespace
# for zero-overhead access; __getattr__ remains the fallback.
_HOT_API: Set[str] = set()   # names currently injected into module globals

# Module-level API (backend exports are appended on load; private names
# such as _cos are exported deliberately).
__all__ = ["get_mode", "get_available_backends", "set_mode"]

# Names owned by this module; never overwritten by backend injection.
_SELF_NAMES = frozenset(("get_mode", "get_available_backends", "set_mode", "__all__"))


def _lookup_call_name(call_name: str) -> Optional[Dict[str, Any]]:
    """Look up a call name in the configured mapping (accepts the name
    with or without the leading dot)."""
    if call_name in _BACKENDS:
        return _BACKENDS[call_name]
    dotted = "." + call_name if not call_name.startswith(".") \
        else call_name[1:]
    if dotted in _BACKENDS:
        return _BACKENDS[dotted]
    return None


def _load_backend(call_name: str) -> bool:
    """Load the module mapped by a call name and store its attributes
    (multiple call names may share one module - repeated loads of the
    same module are idempotent hot-injections)."""
    global _backend, _current_backend_name
    entry = _lookup_call_name(call_name)
    if entry is None:
        return False
    module_name = entry["module"]
    try:
        mod = importlib.import_module(module_name, package=__package__)
        # Export the public API (__all__ if defined, else non-underscore names)
        if hasattr(mod, '__all__'):
            public_attrs = mod.__all__
        else:
            public_attrs = [name for name in dir(mod) if not name.startswith('_')]

        # Build new backend dict
        new_backend = {}
        for attr_name in public_attrs:
            if attr_name.startswith("__") and attr_name.endswith("__"):
                continue
            # Skip __all__ entries missing from this backend (API differences)
            attr = getattr(mod, attr_name, None)
            if attr is None:
                continue
            new_backend[attr_name] = attr

        # Backfill missing public APIs from the pure Python backend so the
        # external surface is identical across backends.
        if module_name != _PURE_MODULE:
            try:
                _pure_mod = importlib.import_module(
                    _PURE_MODULE, package=__package__)
                _pure_all = getattr(_pure_mod, "__all__", ())
                for _fill in _pure_all:
                    if _fill in new_backend:
                        continue
                    _fill_attr = getattr(_pure_mod, _fill, None)
                    if _fill_attr is not None:
                        new_backend[_fill] = _fill_attr
            except Exception:
                pass

        # Normalize `no_done` across backends: the C extension rejects
        # keyword args, so substitute the pure-Python placeholder for parity.
        if "no_done" in new_backend:
            try:
                new_backend["no_done"](_parity_probe=0)
            except TypeError:
                try:
                    _norm_no_done = getattr(importlib.import_module(
                        _PURE_MODULE, package=__package__), "no_done")
                except Exception:
                    _norm_no_done = None
                if _norm_no_done is not None:
                    new_backend["no_done"] = _norm_no_done

        # Replace the previous backend state: drop old hot APIs first
        for old_name in _HOT_API:
            if old_name in _module_globals:
                del _module_globals[old_name]
        _HOT_API.clear()

        _backend.clear()
        _backend.update(new_backend)
        # Hot-inject the full public surface (skip this module's own names)
        for _name, _attr in new_backend.items():
            if _name in _SELF_NAMES:
                continue
            _module_globals[_name] = _attr
            _HOT_API.add(_name)
            if _name not in __all__:
                __all__.append(_name)

        _current_backend_name = call_name
        return True
    except Exception:
        return False


def _load_available_backend(forced_names: Optional[Tuple[str, ...]] = None) -> None:
    """Load the first available backend from priority order."""
    global _current_backend_name
    old_backend = _backend.copy()
    old_hot = {k: _module_globals[k] for k in _HOT_API if k in _module_globals}
    old_name = _current_backend_name

    candidates = forced_names if forced_names is not None else _BACKEND_ORDER
    for name in candidates:
        if _load_backend(name):
            return

    # If all backends failed, restore old state
    for old_name_hot in _HOT_API:
        if old_name_hot in _module_globals:
            del _module_globals[old_name_hot]
    _backend.clear()
    _backend.update(old_backend)
    _module_globals.update(old_hot)
    _current_backend_name = old_name

    raise ImportError(
        f"No available backend among {candidates}. "
        "Ensure at least the pure Python backend is importable."
    )


# Auto-load the first available backend on import
_load_available_backend()

# -------------------------------------------------------------------
# 3. Public API
# -------------------------------------------------------------------
def get_mode() -> Tuple[str, ...]:
    """Return the currently enabled call names in priority order
    (immutable)."""
    return _BACKEND_ORDER


def get_available_backends() -> Tuple[str, ...]:
    """Return all configured call names in priority order (including
    disabled ones, immutable)."""
    return _BACKEND_NAMES


def set_mode(backends):
    """
    Force usage of a specific call name or list of call names in order.

    backends : str or list/tuple of str - any configured call name
    ('c', 'py', '.cos_comparison', '.cos_comparison_pydll',
    '.cos_comparison_c', ...), with or without the dot prefix; names are
    attempted in priority order.  Names sharing one module load that
    module (legacy compatibility).
    """
    if isinstance(backends, str):
        backends = (backends,)
    elif not isinstance(backends, (list, tuple)):
        raise TypeError("backends must be a str or list/tuple of str")

    for b in backends:
        if not isinstance(b, str):
            raise TypeError(f"backend name must be str, got {type(b)}")

    _load_available_backend(forced_names=tuple(backends))


# -------------------------------------------------------------------
# 4. Attribute proxy (fallback for non-hot APIs)
# -------------------------------------------------------------------
def __getattr__(name: str) -> Any:
    """Forward missing attribute lookup to the loaded backend."""
    try:
        return _backend[name]
    except KeyError:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Current backend: {_current_backend_name}"
        ) from None


def __dir__() -> List[str]:
    """Include backend attributes in autocompletion."""
    return sorted(set(_backend.keys()) | set(_module_globals.keys()))
