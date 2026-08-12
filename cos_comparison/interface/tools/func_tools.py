"""
func_tools.py - generic operation helpers shared across the project.
"""

def no_done(*args, **kwargs):
    """No-op fallback used as a default for optional operation hooks."""
    pass

__all__ = ("no_done",)