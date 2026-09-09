"""version command: print the package version (standalone script).

Run via: python -m <package> version
"""
import os
import sys

__all__ = ()


def read_version():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "VERSION.txt")
    try:
        with open(os.path.normpath(path), encoding="utf-8-sig") as fh:
            return fh.read().strip()
    except OSError:
        return "unknown"


if __name__ == "__main__":
    print(read_version())
    sys.exit(0)
