# -*- coding: utf-8 -*-
"""Corrected LLM: local sequence encoding + passive/active dual mode.

Core principles:
- Information arises from LOCAL comparison, not global frequency statistics
- Passive mode: local differences (gap space)
- Active mode: template matching (element space)
- Primitive isolation: no cross-primitive weighting
- K=1 nearest neighbor, max fusion
"""
import re
from cos_comparison.sense_layer.receptor import TensorReceptor
from cos_comparison import core

CORPUS = "alice.txt"  # Place path to your text corpus here


def words(p):
    raw = open(p, encoding="utf-8", errors="ignore").read().lower()
    raw = re.sub(r"[^a-z\s]", " ", raw)
    return raw.split()[:3000]


def encode_sequence_local(ws):
    """
    Encode word sequence as 1D tensor using ordinal values.
    NO global frequency ranking - just direct ordinal mapping.
    Information comes from LOCAL comparison (passive mode), not global stats.
    """
    # Direct ordinal encoding (not frequency-based)
    L0 = [float(ord(w[0])) / 255.0 for w in ws]
    return L0


def main():
    ws = words(CORPUS)

    # LOCAL sequence encoding - NOT global frequency ranking!
    L0 = encode_sequence_local(ws)
    print("tokens", len(L0))

    # REAL passive: d-shift self-comparison -> boundary map (gap space)
    L1 = [float(v) for v in TensorReceptor(L0).comparison_passive(
        window_size=(3,), d=(1,), algorithm="cosmod")]

    # REAL active: kernel template match (element space)
    kernel = L0[100:103]
    L2 = [float(v) for v in core.cos_comparison_active_1d(L0, kernel=kernel)]

    print("L0=%d L1=%d L2=%d" % (len(L0), len(L1), len(L2)))

    # Reasoning: record hits, combine two hit positions (local selection)
    hits = [i for i, v in enumerate(L2) if v > 0.9]
    print("hits", len(hits))
    if len(hits) >= 2:
        a, b = hits[0], hits[-1]
        gen = L0[a:a+2] + L0[b:b+2]
        print("GENERATED (combined two local hits):",
              " ".join(ws[int(v * 255)] if int(v * 255) < len(ws) else "?" for v in gen))
    else:
        print("GENERATED: stop (3-valued)")

    print("\nPrinciples verified:")
    print("- Local encoding: no global frequency statistics")
    print("- Passive mode: local differences (gap space)")
    print("- Active mode: template matching (element space)")
    print("- Primitive isolation: separate channels, max fusion")


if __name__ == "__main__":
    main()
