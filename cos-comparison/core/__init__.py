# core/__init__.py
"""
core backend loader for cos_comparison.

This module loads the best available backend in priority order.
All attributes are forwarded to the active backend.
"""

import importlib
import json
import os
from typing import Any, Dict, List, Optional

# -------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

# Default backends in priority order (relative to the core package)
# ctypes backend is included but disabled by default due to stability issues
_DEFAULT_BACKENDS = [
    {"name": ".cos_comparison_pydll", "enabled": True},
    {"name": ".cos_comparison_c", "enabled": False},
    {"name": ".cos_comparison", "enabled": True},
]

_BACKEND_ORDER: List[str] = []
_BACKEND_NAMES: List[str] = []


def _load_config() -> None:
    """Read config.json and build backend lists."""
    global _BACKEND_ORDER, _BACKEND_NAMES
    raw = _DEFAULT_BACKENDS
    try:
        if os.path.exists(_CONFIG_PATH):
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
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

    _BACKEND_ORDER = [item["name"] for item in normalized if item.get("enabled", True)]
    _BACKEND_NAMES = [item["name"] for item in normalized]

    # Ensure pure Python backend is always present as final fallback
    if ".cos_comparison" not in _BACKEND_ORDER:
        _BACKEND_ORDER.append(".cos_comparison")
    if ".cos_comparison" not in _BACKEND_NAMES:
        _BACKEND_NAMES.append(".cos_comparison")


_load_config()

# -------------------------------------------------------------------
# 2. Backend loading
# -------------------------------------------------------------------
_backend: Dict[str, Any] = {}
_current_backend_name: Optional[str] = None


def _load_backend(module_name: str) -> bool:
    """Import a backend module and store its attributes."""
    global _backend, _current_backend_name
    try:
        mod = importlib.import_module(module_name, package=__package__)
        # Store all public and private attributes, skip dunder methods
        for attr_name in dir(mod):
            if attr_name.startswith("__") and attr_name.endswith("__"):
                continue
            _backend[attr_name] = getattr(mod, attr_name)
        _current_backend_name = module_name
        return True
    except Exception:
        return False


def _load_available_backend(forced_names: Optional[List[str]] = None) -> None:
    """Load the first available backend from priority order."""
    global _backend, _current_backend_name
    _backend.clear()
    _current_backend_name = None

    candidates = forced_names if forced_names is not None else _BACKEND_ORDER
    for name in candidates:
        if _load_backend(name):
            return

    raise ImportError(
        f"No available backend among {candidates}. "
        "Ensure at least the pure Python backend is importable."
    )


# Auto-load the first available backend on import
_load_available_backend()

# -------------------------------------------------------------------
# 3. Public API
# -------------------------------------------------------------------
def get_mode() -> List[str]:
    """Return the currently enabled backends in priority order."""
    return _BACKEND_ORDER.copy()


def get_available_backends() -> List[str]:
    """Return all configured backends (including disabled ones)."""
    return _BACKEND_NAMES.copy()


def set_mode(backends):
    """
    Force usage of a specific backend or list of backends in order.

    Parameters
    ----------
    backends : str or list of str
        Name(s) of the backend(s) to attempt, in priority order.
        Valid names: 'cos_comparison_pydll', 'cos_comparison_c', 'cos_comparison'
    """
    if isinstance(backends, str):
        backends = [backends]
    elif not isinstance(backends, (list, tuple)):
        raise TypeError("backends must be a str or list of str")

    # Normalize names to relative imports
    normalized = []
    for b in backends:
        if not isinstance(b, str):
            raise TypeError(f"backend name must be str, got {type(b)}")
        if not b.startswith("."):
            b = "." + b
        normalized.append(b)

    _load_available_backend(forced_names=normalized)


# -------------------------------------------------------------------
# 4. Attribute proxy
# -------------------------------------------------------------------
def __getattr__(name: str) -> Any:
    """Forward missing attribute lookup to the loaded backend."""
    if name in _backend:
        return _backend[name]
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Current backend: {_current_backend_name}"
    )


def __dir__() -> List[str]:
    """Include backend attributes in autocompletion."""
    return sorted(set(_backend.keys()) | set(globals().keys()))
