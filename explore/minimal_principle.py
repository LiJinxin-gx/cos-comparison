# -*- coding: utf-8 -*-
"""
Minimal successful principle (from v11.24 group hierarchy, MNIST 97.08%).

Continuous mapping group:
  down = f_{i,i+1} (coarsen, average pool)
  up   = f_{i+1,i} (structural inverse, nearest)

Primitive isolation (never average active+passive):
  active  = raw values
  passive = local differences (gap space, a DIFFERENT data type)

Level isolation: each level stores features independently, no value flow.
Fusion at decision layer by max, not weighted average.
"""
import math


def down(t, f):
    """continuous coarsening f_{i,i+1}."""
    h, w = len(t), len(t[0])
    return [[sum(t[i*f+di][j*f+dj]
                 for di in range(f) for dj in range(f))/(f*f)
             for j in range(w//f)] for i in range(h//f)]


def up(t, H, W):
    """structural inverse f_{i+1,i} (nearest, not exact)."""
    h, w = len(t), len(t[0])
    return [[t[min(i*h//H, h-1)][min(j*w//W, w-1)] for j in range(W)] for i in range(H)]


def passive(t):
    """local differences = gap space (DIFFERENT type from active)."""
    h, w = len(t), len(t[0])
    return [[max(abs(t[i][j]-t[i][j-1]) if j else 0,
                 abs(t[i][j]-t[i-1][j]) if i else 0)
             for j in range(w)] for i in range(h)]


def cos(a, b):
    """same-primitive cosine only."""
    d = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return d/(na*nb) if na > 1e-10 and nb > 1e-10 else 0.0


def mod(a, b):
    """magnitude similarity (good for sparse data)."""
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return 2*math.sqrt(na*nb)/(na+nb)


def flat(t):
    return [x for row in t for x in row]
