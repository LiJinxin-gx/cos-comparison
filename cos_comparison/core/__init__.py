# core/__init__.py
"""
core backend loader for cos_comparison.

This module loads the best available backend in priority order.
All attributes are forwarded to the active backend.
High-frequency core APIs are directly injected into module namespace
for maximum runtime performance while maintaining full extensibility.
"""

import importlib
import json
import os.path
from typing import Any, Dict, List, Optional, Tuple

# -------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

# Default backends in priority order (relative to the core package)
# ctypes backend ships as an optional fallback; pure Python is always last
_DEFAULT_BACKENDS = (
    {"name": ".cos_comparison_pydll", "enabled": True},
    {"name": ".cos_comparison_c", "enabled": True},
    {"name": ".cos_comparison", "enabled": True},
)

_BACKEND_ORDER: Tuple[str, ...] = ()
_BACKEND_NAMES: Tuple[str, ...] = ()


def _load_config() -> None:
    """Read config.json and build backend lists."""
    global _BACKEND_ORDER, _BACKEND_NAMES
    raw = _DEFAULT_BACKENDS
    
    # Fast path: skip file IO if config doesn't exist
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                # Support new config format with "backends" key
                if isinstance(config, dict) and "backends" in config:
                    raw = config["backends"]
                elif isinstance(config, list):
                    raw = config
        except Exception:
            # Silently fall back to defaults if config is missing or invalid
            raw = _DEFAULT_BACKENDS

    # Normalize: if list of strings, convert to list of dicts
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        raw = [{"name": item, "enabled": True} for item in raw]
    elif not isinstance(raw, list):
        raw = _DEFAULT_BACKENDS

    # Normalize: ensure relative import (prepend dot if missing)
    normalized = []
    for item in raw:
        if not isinstance(item, dict) or "name" not in item:
            continue
        name = item["name"]
        if not name.startswith("."):
            name = "." + name
        normalized.append({**item, "name": name})

    order_list = [item["name"] for item in normalized if item.get("enabled", True)]
    names_list = [item["name"] for item in normalized]

    # Ensure pure Python backend is always present as final fallback
    if ".cos_comparison" not in order_list:
        order_list.append(".cos_comparison")
    if ".cos_comparison" not in names_list:
        names_list.append(".cos_comparison")
    
    _BACKEND_ORDER = tuple(order_list)
    _BACKEND_NAMES = tuple(names_list)


_load_config()

# -------------------------------------------------------------------
# 2. Backend loading
# -------------------------------------------------------------------
_backend: Dict[str, Any] = {}
_current_backend_name: Optional[str] = None
_module_globals = globals()

# High-frequency core APIs that are directly injected into module namespace
# for zero-overhead access, __getattr__ remains as fallback for other attributes
_HOT_API = {
    "create_void_list", "load_as_default_data", "infer_shape", "vector_map_as_tensor",
    "vector_chain_compute", "set_item", "get_item",
    "_cos", "_mod", "_cosmod", "_default_algorithm",
    "NaN",
}


def _load_backend(module_name: str) -> bool:
    """Import a backend module and store its attributes."""
    global _backend, _current_backend_name
    try:
        mod = importlib.import_module(module_name, package=__package__)
        # Only export public API defined in __all__ to hide internal implementation details,
        # fall back to non-underscore names if __all__ is not defined
        if hasattr(mod, '__all__'):
            public_attrs = mod.__all__
        else:
            public_attrs = [name for name in dir(mod) if not name.startswith('_')]
        
        # Build new backend dict
        new_backend = {}
        new_hot = {}
        for attr_name in public_attrs:
            if attr_name.startswith("__") and attr_name.endswith("__"):
                continue
            # Gracefully skip attributes that are in __all__ but missing (backend API differences)
            attr = getattr(mod, attr_name, None)
            if attr is None:
                continue
            new_backend[attr_name] = attr
            if attr_name in _HOT_API:
                new_hot[attr_name] = attr
        
        # Replace old backend state
        # Remove old hot APIs
        for old_name in _HOT_API:
            if old_name in _module_globals:
                del _module_globals[old_name]
        
        _backend.clear()
        _backend.update(new_backend)
        _module_globals.update(new_hot)
        
        _current_backend_name = module_name
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
    """Return the currently enabled backends in priority order (immutable)."""
    return _BACKEND_ORDER


def get_available_backends() -> Tuple[str, ...]:
    """Return all configured backends (including disabled ones, immutable)."""
    return _BACKEND_NAMES


def set_mode(backends):
    """
    Force usage of a specific backend or list of backends in order.

    Parameters
    ----------
    backends : str or list/tuple of str
        Name(s) of the backend(s) to attempt, in priority order.
        Valid names: 'cos_comparison_pydll', 'cos_comparison_c', 'cos_comparison'
    """
    if isinstance(backends, str):
        backends = (backends,)
    elif not isinstance(backends, (list, tuple)):
        raise TypeError("backends must be a str or list/tuple of str")

    # Normalize names to relative imports
    normalized = []
    for b in backends:
        if not isinstance(b, str):
            raise TypeError(f"backend name must be str, got {type(b)}")
        if not b.startswith("."):
            b = "." + b
        normalized.append(b)

    _load_available_backend(forced_names=tuple(normalized))


# -------------------------------------------------------------------
# 4. Attribute proxy (fallback for non-hot APIs, maintains full extensibility)
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
