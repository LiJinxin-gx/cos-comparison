"""
Hierarchical Matcher: coarse-to-fine matching with active masking.

Level 1 (coarse): boundary features select top-N candidate classes
Level 2 (fine): combined features match within candidates

Principle: continuous mapping level isolation, active matching masking,
selective combination (no weighted voting).

Pure standard library: no numpy or third-party dependencies.
"""
from contrast_match import jaccard_matrix, _to_set


class HierarchicalMatcher:
    """
    Two-level hierarchical matcher.

    Usage:
        matcher = HierarchicalMatcher()
        matcher.fit(train_boundary, train_raw, train_labels)
        preds = matcher.predict(test_boundary, test_raw, coarse_n=10)
    """

    def __init__(self):
        self.train_boundary = []  # list of sets
        self.train_raw = []
        self.train_labels = []
        self.train_combined = []

    def fit(self, train_boundary, train_raw, train_labels):
        """
        Store training features in two groups.

        Args:
            train_boundary: list of binary vectors (passive boundary features)
            train_raw: list of binary vectors (raw pooling features)
            train_labels: list of labels
        """
        self.train_boundary = [_to_set(v) for v in train_boundary]
        self.train_raw = [_to_set(v) for v in train_raw]
        self.train_labels = list(train_labels)
        self.train_combined = [
            b | r for b, r in zip(self.train_boundary, self.train_raw)
        ]

    def _coarse_match(self, test_boundary):
        """Level 1: coarse matching using boundary features."""
        test_sets = [_to_set(v) for v in test_boundary]
        return jaccard_matrix(self.train_boundary, test_sets)

    def predict(self, test_boundary, test_raw, coarse_n=10):
        """
        Predict using hierarchical matching.

        Args:
            test_boundary: list of binary vectors
            test_raw: list of binary vectors
            coarse_n: number of candidate classes for coarse level

        Returns:
            list of predicted labels
        """
        test_boundary_sets = [_to_set(v) for v in test_boundary]
        test_raw_sets = [_to_set(v) for v in test_raw]
        test_combined = [
            b | r for b, r in zip(test_boundary_sets, test_raw_sets)
        ]

        # Level 1: coarse matching
        coarse_sim = jaccard_matrix(self.train_boundary, test_boundary_sets)

        predictions = []
        for i in range(len(test_combined)):
            # Get candidate classes from coarse level
            indexed = list(enumerate(coarse_sim[i]))
            indexed.sort(key=lambda x: x[1], reverse=True)
            top_idx = [idx for idx, _ in indexed[:coarse_n * 3]]
            candidate_classes = set(self.train_labels[idx] for idx in top_idx)

            # Level 2: fine match within candidates only
            best_score = -1.0
            best_label = self.train_labels[0]
            for j in range(len(self.train_combined)):
                if self.train_labels[j] not in candidate_classes:
                    continue
                ts = test_combined[i]
                tr = self.train_combined[j]
                if not ts and not tr:
                    score = 0.0
                else:
                    inter = len(ts & tr)
                    union = len(ts | tr)
                    score = inter / union if union > 0 else 0.0
                if score > best_score:
                    best_score = score
                    best_label = self.train_labels[j]
            predictions.append(best_label)

        return predictions

    def accuracy(self, test_boundary, test_raw, test_labels, coarse_n=10):
        """Compute accuracy."""
        preds = self.predict(test_boundary, test_raw, coarse_n=coarse_n)
        correct = sum(1 for p, t in zip(preds, test_labels) if p == t)
        return correct / len(test_labels)


if __name__ == '__main__':
    # Demo
    import random
    random.seed(42)
    n_train, n_test, dim_b, dim_r = 100, 20, 500, 200
    train_b = [[1 if random.random() > 0.9 else 0 for _ in range(dim_b)]
               for _ in range(n_train)]
    train_r = [[1 if random.random() > 0.9 else 0 for _ in range(dim_r)]
               for _ in range(n_train)]
    train_labels = [f"class_{i % 5}" for i in range(n_train)]
    test_b = [[1 if random.random() > 0.9 else 0 for _ in range(dim_b)]
              for _ in range(n_test)]
    test_r = [[1 if random.random() > 0.9 else 0 for _ in range(dim_r)]
              for _ in range(n_test)]
    test_labels = [f"class_{i % 5}" for i in range(n_test)]

    matcher = HierarchicalMatcher()
    matcher.fit(train_b, train_r, train_labels)
    acc = matcher.accuracy(test_b, test_r, test_labels, coarse_n=10)
    print(f"Hierarchical demo accuracy: {acc*100:.1f}%")
