"""
Text Classification Demo: frequency encoding + hierarchical matching.

Demonstrates:
1. Text -> 1D tensor via character frequency (unit_map principle)
2. 3-level hierarchical pooling on 1D sequence
3. Active cosine matching, K=1 nearest neighbor
4. Level isolation: take max across levels (no weighted fusion)

Data loading separated from processing:
- load_text(): file -> string
- encode_text(): string -> 1D tensor (frequency map)
- HierarchicalTextClassifier1D: tensor -> classification

Pure standard library only.
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_text(path):
    """Load text file as string."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def encode_text_frequency(text, vocab_size=128):
    """
    Encode text as 1D frequency tensor.
    No external knowledge, fully data-driven.
    """
    freq = [0.0] * vocab_size
    for ch in text:
        idx = ord(ch) % vocab_size
        freq[idx] += 1.0
    total = sum(freq)
    if total > 0:
        freq = [f / total for f in freq]
    return freq


def pool_1d(flat, factor):
    """1D average pooling (continuous down-sampling)."""
    n = len(flat)
    new_n = n // factor
    result = []
    for i in range(new_n):
        s = 0.0
        for j in range(factor):
            s += flat[i * factor + j]
        result.append(s / factor)
    return result


def cosine_sim(a, b):
    """Cosine similarity (active mode only)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


class HierarchicalTextClassifier1D:
    """
    1D hierarchical classifier for text frequency vectors.

    Group-theoretic principles:
    - 3 pooling levels [1,2,4] (continuous down-sampling)
    - Active cosine matching only
    - K=1 nearest neighbor
    - Level isolation: take max across levels (no weighted fusion)
    """

    def __init__(self, pool_factors=None):
        if pool_factors is None:
            pool_factors = [1, 2, 4]
        self.pool_factors = pool_factors
        self.prototypes = {}

    def _extract_levels(self, flat):
        """Extract features at each level independently (level isolation)."""
        levels = []
        current = flat
        for factor in self.pool_factors:
            if factor > 1:
                current = pool_1d(current, factor)
            levels.append(list(current))
        return levels

    def fit(self, texts, labels):
        """Store prototypes at each level independently."""
        self.prototypes = {}
        for text, label in zip(texts, labels):
            freq = encode_text_frequency(text)
            levels = self._extract_levels(freq)
            if label not in self.prototypes:
                self.prototypes[label] = []
            self.prototypes[label].append(levels)

    def predict(self, text):
        """Predict label using max across levels (no weighted sum)."""
        freq = encode_text_frequency(text)
        query_levels = self._extract_levels(freq)
        scores = {}

        for label, proto_list in self.prototypes.items():
            best_score = 0.0
            for proto_levels in proto_list:
                # Take max across levels, not weighted sum
                level_score = 0.0
                for lvl_idx in range(len(query_levels)):
                    sim = cosine_sim(query_levels[lvl_idx], proto_levels[lvl_idx])
                    if sim > level_score:
                        level_score = sim
                if level_score > best_score:
                    best_score = level_score
            scores[label] = best_score

        return max(scores, key=scores.get)

    def accuracy(self, texts, labels):
        correct = sum(1 for t, l in zip(texts, labels) if self.predict(t) == l)
        return correct / len(labels)


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo_text.py <data_dir>")
        print("Expected structure: data_dir/class_name/texts...")
        print("\nRunning synthetic demo...")

        def make_text(category, length=500):
            if category == "news":
                chars = "aeioubcdfghlmnprst " * 3 + "newsreport"
            elif category == "legal":
                chars = "abcdefilmnoprstu " * 3 + "whereasthereof"
            else:
                chars = "aeioursthnl " * 3 + "productbuyprice"
            import random
            random.seed(hash(category) % 2**31)
            return ''.join(random.choice(chars) for _ in range(length))

        train_texts = [make_text(i % 3) for i in range(150)]
        train_labels = [f"cat_{i % 3}" for i in range(150)]
        test_texts = [make_text(i % 3) for i in range(45)]
        test_labels = [f"cat_{i % 3}" for i in range(45)]
    else:
        data_dir = sys.argv[1]
        classes = sorted(d for d in os.listdir(data_dir)
                         if os.path.isdir(os.path.join(data_dir, d)))
        train_texts, train_labels = [], []
        test_texts, test_labels = [], []
        for cls in classes:
            cls_dir = os.path.join(data_dir, cls)
            files = sorted(os.listdir(cls_dir))[:30]
            for f in files[9:]:
                train_texts.append(load_text(os.path.join(cls_dir, f)))
                train_labels.append(cls)
            for f in files[:9]:
                test_texts.append(load_text(os.path.join(cls_dir, f)))
                test_labels.append(cls)

    print("Training text classifier...")
    clf = HierarchicalTextClassifier1D()
    clf.fit(train_texts, train_labels)

    print("Evaluating...")
    acc = clf.accuracy(test_texts, test_labels)
    print(f"\nAccuracy: {acc * 100:.1f}%")
    print(f"Principle: frequency encoding + 3-level pooling + active cosine + max across levels")


if __name__ == '__main__':
    main()
