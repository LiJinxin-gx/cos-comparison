"""
Contrast Point Matcher using Jaccard similarity.

Receives binary feature sets (lists of 0/1), outputs predictions.
Principle: contrast point set intersection (Jaccard), selective combination
(no weighted voting), K-nearest-neighbor.

Pure standard library: no numpy or third-party dependencies.
"""
from collections import Counter


def _to_set(binary_vector):
    """Convert binary vector to set of active indices."""
    return {i for i, v in enumerate(binary_vector) if v}


def jaccard_similarity(set_a, set_b):
    """
    Jaccard similarity between two binary sets.
    J = |A ∩ B| / |A ∪ B|
    Accepts either sets of indices or binary vectors.
    """
    if not isinstance(set_a, set):
        set_a = _to_set(set_a)
    if not isinstance(set_b, set):
        set_b = _to_set(set_b)
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def jaccard_matrix(train_sets, test_sets):
    """
    Compute Jaccard similarity matrix between all test and train samples.

    Args:
        train_sets: list of binary vectors (or sets)
        test_sets: list of binary vectors (or sets)

    Returns:
        list of lists: (n_test, n_train) similarity matrix
    """
    train_sets = [s if isinstance(s, set) else _to_set(s) for s in train_sets]
    test_sets = [s if isinstance(s, set) else _to_set(s) for s in test_sets]
    matrix = []
    for ts in test_sets:
        row = []
        for tr in train_sets:
            if not ts and not tr:
                row.append(0.0)
            else:
                inter = len(ts & tr)
                union = len(ts | tr)
                row.append(inter / union if union > 0 else 0.0)
        matrix.append(row)
    return matrix


def knn_predict(similarity_matrix, train_labels, k=1):
    """
    K-nearest-neighbor prediction using similarity matrix.
    For k>1: majority vote (selective combination, NOT weighted).

    Args:
        similarity_matrix: list of lists (n_test, n_train) similarity scores
        train_labels: list of labels for training samples
        k: number of nearest neighbors

    Returns:
        list of predictions
    """
    predictions = []
    for row in similarity_matrix:
        # Get indices of top-k highest similarities
        indexed = list(enumerate(row))
        indexed.sort(key=lambda x: x[1], reverse=True)
        top_k_idx = [idx for idx, _ in indexed[:k]]
        top_k_labels = [train_labels[idx] for idx in top_k_idx]
        counter = Counter(top_k_labels)
        predictions.append(counter.most_common(1)[0][0])
    return predictions


class ContrastMatcher:
    """
    Atomic contrast point matcher.

    Usage:
        matcher = ContrastMatcher()
        matcher.fit(train_binary, train_labels)
        preds = matcher.predict(test_binary, k=1)
    """

    def __init__(self):
        self.train_sets = []  # list of sets
        self.train_labels = []

    def fit(self, train_binary, train_labels):
        """
        Store training binary feature sets and labels.

        Args:
            train_binary: list of binary vectors (list of 0/1)
            train_labels: list of labels
        """
        self.train_sets = [_to_set(v) for v in train_binary]
        self.train_labels = list(train_labels)

    def predict(self, test_binary, k=1):
        """
        Predict labels for test binary feature sets.

        Args:
            test_binary: list of binary vectors
            k: number of nearest neighbors (default 1, proven best)

        Returns:
            list of predicted labels
        """
        test_sets = [_to_set(v) for v in test_binary]
        sim = jaccard_matrix(self.train_sets, test_sets)
        return knn_predict(sim, self.train_labels, k=k)

    def accuracy(self, test_binary, test_labels, k=1):
        """Compute accuracy on test set."""
        preds = self.predict(test_binary, k=k)
        correct = sum(1 for p, t in zip(preds, test_labels) if p == t)
        return correct / len(test_labels)


if __name__ == '__main__':
    # Demo: random binary sets
    import random
    random.seed(42)
    train_bin = [[1 if random.random() > 0.9 else 0 for _ in range(1000)]
                 for _ in range(100)]
    train_labels = [f"class_{i % 5}" for i in range(100)]
    test_bin = [[1 if random.random() > 0.9 else 0 for _ in range(1000)]
                for _ in range(20)]
    test_labels = [f"class_{i % 5}" for i in range(20)]

    matcher = ContrastMatcher()
    matcher.fit(train_bin, train_labels)
    acc = matcher.accuracy(test_bin, test_labels, k=1)
    print(f"Random demo accuracy: {acc*100:.1f}% (expected ~20% for 5 classes)")
