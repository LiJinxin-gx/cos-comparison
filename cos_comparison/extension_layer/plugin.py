"""extension_layer - plugin batch aggregation (proactive hosting).

No registration mechanism: plugins / resources / direct-call objects are
hosted as-is through keyed pools, so external plugins need no
modification to be used.  All pools default to lists (key = index) and
duck-support any keyed container (mapping keys, etc.).
"""

from collections.abc import Mapping

__all__ = ("PluginPool",)


def _set_keyed(container, key, value):
    """Keyed store: mappings take container[key] = value; sequences
    append (out-of-range index keys append as well)."""
    if isinstance(container, Mapping):
        container[key] = value
        return
    try:
        container[key] = value
    except (TypeError, IndexError, KeyError):
        container.append(value)


class PluginPool:
    """Proactive plugin batch aggregation with keyed access.

    Three pools (each defaults to a list, key = index):
      resources   hosted resources (shared data / contexts)
      plugins     hosted plugins (external objects, unchanged)
      func_pool   directly callable objects (functions / callables)

    Usage: add_* stores by key; call_func / call_plugin invoke; get_*
    retrieves.  Keys index the container directly (duck typing - a
    mapping accepts any key, a list accepts indices).
    """

    __slots__ = ("func_pool", "plugins", "resources")

    def __init__(self, resources=None, plugins=None, func_pool=None):
        self.resources = resources if resources is not None else []
        self.plugins = plugins if plugins is not None else []
        self.func_pool = func_pool if func_pool is not None else []

    def add_resource(self, key, value):
        """Store a resource under a key."""
        _set_keyed(self.resources, key, value)

    def add_plugin(self, key, value):
        """Host a plugin (external object, unchanged) under a key."""
        _set_keyed(self.plugins, key, value)

    def add_func(self, key, value):
        """Store a directly callable object under a key."""
        _set_keyed(self.func_pool, key, value)

    def call_func(self, name, *args, **kwargs):
        """Call func_pool[name](*args, **kwargs)."""
        return self.func_pool[name](*args, **kwargs)

    def call_plugin(self, name, method, *args, **kwargs):
        """Call plugins[name].method(*args, **kwargs)."""
        return getattr(self.plugins[name], method)(*args, **kwargs)

    def get_plugin_attr(self, name, attr):
        """Attribute of plugins[name]."""
        return getattr(self.plugins[name], attr)

    def get_resource(self, name):
        """The resource stored at key ``name``."""
        return self.resources[name]

    def get_plugin(self, name):
        """The plugin hosted at key ``name``."""
        return self.plugins[name]
