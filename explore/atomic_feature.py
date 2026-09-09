"""
Atomic Feature Extractor for contrast-point matching.

Receives a 2D tensor (list of lists), outputs a feature vector.
Uses passive boundary extraction + pooling (pure standard library).

Principle: structure-first, local analysis, no global features.
No numpy or third-party dependencies.
"""

# Multi-scale directions for passive boundary
_DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]
_BOUNDARY_SCALES = [1, 2, 3]
_POOL_WINDOWS = [2, 4]
_POOL_STATS = ['max', 'mean', 'var']


def _to_list(data):
    """Convert duck-typed 2D data to list of lists of floats."""
    if isinstance(data, list) and data and isinstance(data[0], list):
        return [[float(x) for x in row] for row in data]
    # Handle numpy-like or other iterables
    rows = []
    for row in data:
        rows.append([float(x) for x in row])
    return rows


def _shape(arr):
    """Return (height, width) of 2D list."""
    return len(arr), len(arr[0]) if arr else 0


def passive_boundary(data, d=1):
    """
    Passive mode boundary extraction.
    Output = element gaps (boundaries), not element values.
    Multi-directional max response.
    """
    arr = _to_list(data)
    h, w = _shape(arr)
    result = [[0.0] * w for _ in range(h)]
    for dx, dy in _DIRECTIONS:
        for i in range(h):
            for j in range(w):
                ni, nj = i + dx * d, j + dy * d
                if 0 <= ni < h and 0 <= nj < w:
                    diff = abs(arr[i][j] - arr[ni][nj])
                    if diff > result[i][j]:
                        result[i][j] = diff
    return result


def _block_stats(block, stat):
    """Compute statistic over a flat list of values."""
    if not block:
        return 0.0
    if stat == 'max':
        return max(block)
    if stat == 'mean':
        return sum(block) / len(block)
    if stat == 'var':
        m = sum(block) / len(block)
        return sum((x - m) ** 2 for x in block) / len(block)
    return 0.0


def pool(data, window=2, stat='max'):
    """Non-overlapping pooling of a 2D tensor."""
    arr = _to_list(data)
    h, w = _shape(arr)
    oh, ow = h // window, w // window
    if oh == 0 or ow == 0:
        return [[arr[0][0]]] if arr else [[0.0]]
    pooled = [[0.0] * ow for _ in range(oh)]
    for i in range(oh):
        for j in range(ow):
            block = []
            for di in range(window):
                for dj in range(window):
                    block.append(arr[i * window + di][j * window + dj])
            pooled[i][j] = _block_stats(block, stat)
    return pooled


def _flatten(arr):
    """Flatten 2D list to 1D."""
    return [x for row in arr for x in row]


def extract(data, boundary_scales=None, pool_windows=None, pool_stats=None):
    """
    Extract atomic feature vector from a 2D tensor.

    Args:
        data: 2D tensor (list of lists)
        boundary_scales: list of d values for passive boundary (default [1,2,3])
        pool_windows: list of pooling window sizes (default [2,4])
        pool_stats: list of pooling statistics (default ['max','mean','var'])

    Returns:
        list of features (concatenated boundary + raw pooling)
    """
    if boundary_scales is None:
        boundary_scales = _BOUNDARY_SCALES
    if pool_windows is None:
        pool_windows = _POOL_WINDOWS
    if pool_stats is None:
        pool_stats = _POOL_STATS

    arr = _to_list(data)
    features = []

    # Group A: passive boundary features (distinctive boundaries)
    for d in boundary_scales:
        b = passive_boundary(arr, d)
        for w in pool_windows:
            for s in pool_stats:
                features.extend(_flatten(pool(b, w, s)))

    # Group B: raw tensor pooling (broader structure)
    for w in pool_windows:
        for s in ['max', 'mean']:
            features.extend(_flatten(pool(arr, w, s)))

    return features


def extract_grouped(data, boundary_scales=None, pool_windows=None, pool_stats=None):
    """
    Extract features in TWO GROUPS for asymmetric storage.

    Returns:
        (boundary_features, raw_features) - separate lists
    """
    if boundary_scales is None:
        boundary_scales = _BOUNDARY_SCALES
    if pool_windows is None:
        pool_windows = _POOL_WINDOWS
    if pool_stats is None:
        pool_stats = _POOL_STATS

    arr = _to_list(data)
    boundary_feats = []
    raw_feats = []

    for d in boundary_scales:
        b = passive_boundary(arr, d)
        for w in pool_windows:
            for s in pool_stats:
                boundary_feats.extend(_flatten(pool(b, w, s)))

    for w in pool_windows:
        for s in ['max', 'mean']:
            raw_feats.extend(_flatten(pool(arr, w, s)))

    return boundary_feats, raw_feats


def binarize(features, top_percent=10):
    """
    Binarize feature vector: top X% response = 1, else = 0.
    Creates contrast point set for Jaccard matching.

    Args:
        features: list of floats
        top_percent: percentage of top values to keep (default 10)

    Returns:
        list of 0/1 integers
    """
    if not features:
        return []
    sorted_vals = sorted(features, reverse=True)
    n_keep = max(1, int(len(features) * top_percent / 100))
    threshold = sorted_vals[min(n_keep - 1, len(sorted_vals) - 1)]
    return [1 if v >= threshold else 0 for v in features]


if __name__ == '__main__':
    # Demo: extract features from a random 64x64 tensor
    import random
    random.seed(42)
    demo = [[random.random() for _ in range(64)] for _ in range(64)]
    feat = extract(demo)
    binary = binarize(feat, top_percent=10)
    print(f"Feature dim: {len(feat)}")
    print(f"Binary density: {sum(binary) / len(binary):.3f}")
    print(f"Active points: {sum(binary)}")
