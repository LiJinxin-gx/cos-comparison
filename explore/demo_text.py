"""
Text Classification Demo using Atomic Contrast Point Matching.

Demonstrates:
1. Text -> frequency tensor (using UnitMap, no linguistic prior)
2. Reshape to 2D tensor for feature extraction
3. Binarization + contrast matching

Principle: text is just mapped symbols; frequency encoding captures structure.
No POS tagging, no word embeddings, no deep learning.

Pure standard library + cos_comparison: no numpy or third-party dependencies.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from text_encoder import TextFrequencyEncoder
from atomic_feature import extract, binarize
from contrast_match import ContrastMatcher


def frequency_to_2d(freq_tensor, shape=None):
    """
    Reshape 1D frequency vector to 2D tensor for spatial feature extraction.
    If shape is None, use square-ish shape.
    """
    n = len(freq_tensor)
    dim = len(freq_tensor[0]) if n > 0 else 0
    if shape is None:
        h = int(dim ** 0.5)
        w = dim // h
        if h * w < dim:
            w += 1
        shape = (h, w)
    result = []
    for vec in freq_tensor:
        padded = vec + [0.0] * (shape[0] * shape[1] - len(vec))
        mat = []
        for i in range(shape[0]):
            mat.append(padded[i * shape[1]:(i + 1) * shape[1]])
        result.append(mat)
    return result


def classify_texts(train_texts, train_labels, test_texts, test_labels,
                   mode='character', top_percent=10):
    """
    Full text classification pipeline.

    Args:
        train_texts: list of training strings
        train_labels: list of training labels
        test_texts: list of test strings
        test_labels: list of test labels
        mode: 'character' or 'word' encoding
        top_percent: binarization threshold

    Returns:
        accuracy float
    """
    # Step 1: Text -> frequency tensor (UnitMap)
    print("Encoding text to frequency tensors...")
    encoder = TextFrequencyEncoder(mode=mode, max_vocab=5000)
    train_freq = encoder.fit_transform(train_texts)
    test_freq = encoder.transform(test_texts)
    train_freq = encoder.normalize(train_freq)
    test_freq = encoder.normalize(test_freq)
    print(f"Vocab size: {encoder.vocab_size}, Train samples: {len(train_freq)}")

    # Step 2: Reshape to 2D tensors
    train_2d = frequency_to_2d(train_freq)
    test_2d = frequency_to_2d(test_freq)
    print(f"2D shape: {len(train_2d[0])}x{len(train_2d[0][0])}")

    # Step 3: Extract atomic features
    print("Extracting atomic features...")
    train_feats = [extract(t) for t in train_2d]
    test_feats = [extract(t) for t in test_2d]

    # Step 4: Binarize (contrast point sets)
    train_bin = [binarize(f, top_percent=top_percent) for f in train_feats]
    test_bin = [binarize(f, top_percent=top_percent) for f in test_feats]

    # Step 5: Contrast matching
    print("Matching...")
    matcher = ContrastMatcher()
    matcher.fit(train_bin, train_labels)
    acc = matcher.accuracy(test_bin, test_labels, k=1)
    return acc


def main():
    # Demo: simple text classification with synthetic data
    print("=== Text Classification Demo ===")
    print("Synthetic data: 3 classes of character patterns\n")

    import random
    random.seed(42)

    # Generate synthetic text data with distinguishable patterns
    def generate_text(class_id, length=50):
        if class_id == 0:
            chars = 'abc'
        elif class_id == 1:
            chars = 'xyz'
        else:
            chars = '123'
        return ''.join(random.choice(chars) for _ in range(length))

    train_texts = [generate_text(i % 3) for i in range(150)]
    train_labels = [f"class_{i % 3}" for i in range(150)]
    test_texts = [generate_text(i % 3) for i in range(60)]
    test_labels = [f"class_{i % 3}" for i in range(60)]

    acc = classify_texts(train_texts, train_labels, test_texts, test_labels,
                         mode='character', top_percent=10)
    print(f"\nAccuracy: {acc*100:.1f}% (expected ~100% for separable patterns)")


if __name__ == '__main__':
    main()
