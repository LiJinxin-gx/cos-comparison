# -*- coding: utf-8 -*-
"""
Local k-NN Learner (PCML v53).

The simplest rigorous non-parametric learner:
- Knowledge stored as local prototypes
- Prediction = unweighted mean of k nearest local experiences
- Strictly local: no weights, no gradient, no global function
- Zero-forgetting: new prototypes never overwrite old ones

Core principles:
1. Information arises from local comparison
2. Absolute locality: each prototype is an independent local memory
3. Level isolation: prototypes stored independently, no global averaging
4. K=1 base: k is neighborhood size parameter, not voting weight

Zero dependencies: pure Python standard library.
"""
import math


# ============================================================
# Core Primitives
# ============================================================

def euclidean_distance(a, b):
    """Euclidean distance between two vectors. Local comparison."""
    if len(a) != len(b):
        raise ValueError("Vector dimensions must match")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ============================================================
# Local k-NN Learner
# ============================================================

class LocalKNNLearner:
    """
    Local k-NN world model / predictor.
    
    Principles:
    - Each experience is stored as an independent prototype (zero-forgetting)
    - Prediction: unweighted average of k nearest local prototypes
    - Strictly local: no global function, no weights, no gradient
    - As prototypes grow denser, k-nearest window shrinks effectively
    
    This is the non-parametric memory-and-retrieval system:
    strong as a transparent similarity core.
    """
    
    def __init__(self, k=5):
        self.k = k
        self.prototypes = []  # list of (state, action) pairs
    
    def remember(self, state, action):
        """
        Store a new experience as an independent prototype.
        Zero-forgetting: never overwrite or modify existing prototypes.
        """
        self.prototypes.append((state, action))
    
    def predict(self, query_state):
        """
        Predict by averaging the k nearest local experiences.
        Unweighted: all k neighbors contribute equally.
        """
        if not self.prototypes:
            raise ValueError("No prototypes stored yet")
        
        # Compute distances to all prototypes (local comparison)
        distances = []
        for state, action in self.prototypes:
            d = euclidean_distance(query_state, state)
            distances.append((d, action))
        
        # Sort by distance, take k nearest
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        # Unweighted average (simple local mean)
        n = len(k_nearest)
        if n == 0:
            return None
        
        # Handle scalar or vector actions
        first_action = k_nearest[0][1]
        if isinstance(first_action, (int, float)):
            return sum(a for _, a in k_nearest) / n
        else:
            # Vector action
            dim = len(first_action)
            result = [0.0] * dim
            for _, action in k_nearest:
                for i in range(dim):
                    result[i] += action[i]
            return [x / n for x in result]
    
    @property
    def size(self):
        return len(self.prototypes)


# ============================================================
# Demo / Test
# ============================================================

def test_simple_regression():
    """Test simple 1D regression: y = x^2 + noise"""
    print("=" * 60)
    print("Local k-NN Learner Demo")
    print("=" * 60)
    print("\nCore principles:")
    print("- Local prototypes (zero-forgetting memory)")
    print("- Unweighted k-nearest local mean")
    print("- No global function, no weights, no gradient")
    print("- Strictly local: global is special local")
    print()
    
    import random
    random.seed(42)
    
    # Generate training data: y = x^2
    learner = LocalKNNLearner(k=5)
    
    print("Training (storing prototypes)...")
    for _ in range(200):
        x = random.uniform(-5, 5)
        y = x ** 2 + random.gauss(0, 0.5)
        learner.remember([x], y)
    
    print(f"Stored {learner.size} prototypes")
    
    # Test
    print("\nTesting...")
    test_points = [-4, -2, 0, 2, 4]
    correct = 0
    for x in test_points:
        pred = learner.predict([x])
        true_val = x ** 2
        error = abs(pred - true_val)
        print(f"  x={x:+.1f}: predicted={pred:.2f}, true={true_val:.2f}, error={error:.2f}")
        if error < 1.0:
            correct += 1
    
    print(f"\nAccuracy (error < 1.0): {correct}/{len(test_points)}")
    
    # Zero-forgetting test
    print("\nZero-forgetting test:")
    print(f"  Before adding new region: {learner.size} prototypes")
    for _ in range(20):
        x = random.uniform(10, 12)
        y = x ** 2 + random.gauss(0, 0.5)
        learner.remember([x], y)
    print(f"  After adding new region: {learner.size} prototypes")
    
    # Check old region still works
    pred_old = learner.predict([0])
    print(f"  Prediction at old region (x=0): {pred_old:.2f} (should be ~0)")
    print(f"  Old region unaffected: {'YES' if abs(pred_old) < 2.0 else 'NO'}")
    
    print("\nPrinciple verified: local k-NN = memory + retrieval")


if __name__ == "__main__":
    test_simple_regression()
