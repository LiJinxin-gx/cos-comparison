"""
Random tools: an abstract random facility defaulting to the `random` module,
with a built-in cryptographically secure backend built on the standard
library `secrets` module.

The abstract base defines three primitives; every public operation is
derived from them, so a custom backend only needs to implement the
primitives to gain the full API.
"""

import random
import secrets
from abc import ABC, abstractmethod


class RandomToolBase(ABC):
    """
    Abstract random tool.

    Only the primitive `random()` is mandatory. The other two primitives
    have generic default implementations derived from `random()` and may
    be overridden by backends that need exact uniformity or custom entropy:
      * `random()`        - a float uniformly distributed in [0.0, 1.0)
      * `_randbelow(n)`   - an int uniformly distributed in [0, n)
      * `_weighted_index` - an index drawn proportionally to a weight sequence

    All public operations (generation, sorting, selection, weighted
    selection) are derived from these primitives, so a custom backend
    implementing only `random()` still gains the full API. Concrete
    backends (RandomTool, SecureRandomTool) implement all three primitives
    and the complete feature set.
    """

    __slots__ = ()

    @abstractmethod
    def random(self):
        """Return a random float uniformly distributed in [0.0, 1.0)."""

    def _randbelow(self, n):
        """
        Return a random int uniformly distributed in [0, n).
        Generic implementation derived from random(); backends requiring
        exact uniformity without floating-point bias should override it.
        """
        if n <= 0:
            raise ValueError("_randbelow() n must be positive")
        r = int(self.random() * n)
        return n - 1 if r >= n else r

    def _weighted_index(self, weights):
        """
        Return an index into `weights` drawn proportionally to the weights.
        Generic implementation derived from random(); backends requiring
        exact integer-weighted draws should override it.
        """
        total = 0.0
        for w in weights:
            if w < 0:
                raise ValueError("weights must be non-negative")
            total += w
        if total <= 0:
            raise ValueError("total of weights must be positive")
        r = self.random() * total
        upto = 0.0
        for i, w in enumerate(weights):
            upto += w
            if r < upto:
                return i
        return len(weights) - 1

    def seed(self, a=None):
        """Seed the underlying generator. Rejected by non-seedable backends."""
        raise NotImplementedError("seed() is not supported by this backend.")

    def uniform(self, a, b):
        """Return a random float uniformly distributed in [a, b]."""
        return a + (b - a) * self.random()

    def randint(self, a, b):
        """Return a random int uniformly distributed in [a, b] (inclusive)."""
        if a > b:
            raise ValueError("empty range for randint(): a must be <= b")
        return a + self._randbelow(b - a + 1)

    def randrange(self, start, stop=None, step=1):
        """Return a random int drawn from range(start, stop, step)."""
        if stop is None:
            start, stop = 0, start
        if step == 0:
            raise ValueError("randrange() step argument must not be zero")
        width = stop - start
        if step > 0:
            n = (width + step - 1) // step
        else:
            n = (width + step + 1) // step
        if n <= 0:
            raise ValueError("empty range for randrange()")
        return start + step * self._randbelow(n)

    def choice(self, seq):
        """Return a single random element from a non-empty sequence."""
        if not seq:
            raise IndexError("choice() cannot select from an empty sequence")
        return seq[self._randbelow(len(seq))]

    def shuffle(self, seq):
        """Shuffle a mutable sequence in place (iterative Fisher-Yates)."""
        for i in range(len(seq) - 1, 0, -1):
            j = self._randbelow(i + 1)
            seq[i], seq[j] = seq[j], seq[i]

    def sample(self, population, k):
        """Return k distinct elements drawn without replacement."""
        if not isinstance(k, int):
            raise TypeError("sample() k must be an integer")
        if k < 0:
            raise ValueError("sample() k must be non-negative")
        n = len(population)
        if k > n:
            raise ValueError("sample() k must be <= len(population)")
        seq = list(population)
        for i in range(n - 1, n - 1 - k, -1):
            j = self._randbelow(i + 1)
            seq[i], seq[j] = seq[j], seq[i]
        return seq[n - k:]

    def choices(self, population, weights=None, k=1):
        """Return k elements drawn with replacement, optionally weighted."""
        if not isinstance(k, int):
            raise TypeError("choices() k must be an integer")
        if k < 0:
            raise ValueError("choices() k must be non-negative")
        n = len(population)
        if weights is None:
            if not population:
                raise IndexError("choices() cannot select from an empty sequence")
            return [self.choice(population) for _ in range(k)]
        if len(weights) != n:
            raise ValueError("weights must match the length of population")
        if not population:
            raise ValueError("choices() cannot select from an empty weighted sequence")
        return [population[self._weighted_index(weights)] for _ in range(k)]

    def __repr__(self):
        return "<%s>" % (type(self).__name__,)


class DelegatedRandomTool(RandomToolBase):
    """
    Concrete delegation tool, sharing the level of RandomTool and
    SecureRandomTool. Implements the delegation abstraction: functions are
    injected at construction time and every primitive forwards to the
    injected functions at call time.

    Each slot accepts either a callable or an object. The tool judges the
    delegation difficulty itself: an object is only bound when it exposes a
    method whose signature matches the primitive directly (low difficulty),
    and is left unresolved when adapting it would require wrapping (high
    difficulty):
      * random_func      - () -> float, binds object.random
      * randbelow_func   - (n) -> int, binds object.randrange/randbelow
      * weighted_func    - (weights) -> index, binds object._weighted_index
                           (object.choices is NOT adapted - too hard)
      * seed_func        - (a) -> None, binds object.seed

    Unresolved slots raise NotImplementedError when called.
    """

    __slots__ = ("_random_func", "_randbelow_func", "_weighted_func", "_seed_func")

    def __init__(self, random_func=None, randbelow_func=None,
                 weighted_func=None, seed_func=None):
        if (random_func is not None and not callable(random_func)
                and randbelow_func is None and weighted_func is None
                and seed_func is None):
            source = random_func
            self._random_func = self._resolve(source, "random")
            self._randbelow_func = self._resolve(source, "randbelow")
            self._weighted_func = self._resolve(source, "weighted")
            self._seed_func = self._resolve(source, "seed")
        else:
            self._random_func = self._resolve(random_func, "random")
            self._randbelow_func = self._resolve(randbelow_func, "randbelow")
            self._weighted_func = self._resolve(weighted_func, "weighted")
            self._seed_func = self._resolve(seed_func, "seed")

    def _resolve(self, source, kind):
        """Resolve an injected callable or object into the matching function."""
        if source is None:
            return None
        if callable(source):
            return source
        if kind == "random":
            method = getattr(source, "random", None)
            return method if callable(method) else None
        if kind == "randbelow":
            for name in ("randrange", "randbelow"):
                method = getattr(source, name, None)
                if callable(method):
                    return method
            return None
        if kind == "weighted":
            method = getattr(source, "_weighted_index", None)
            return method if callable(method) else None
        if kind == "seed":
            method = getattr(source, "seed", None)
            return method if callable(method) else None
        return None

    def random(self):
        """Delegate the float primitive to the injected function."""
        if self._random_func is None:
            raise NotImplementedError("no random source injected")
        return self._random_func()

    def _randbelow(self, n):
        """Delegate the integer primitive to the injected function."""
        if self._randbelow_func is None:
            raise NotImplementedError("no randbelow source injected")
        return self._randbelow_func(n)

    def _weighted_index(self, weights):
        """Delegate the weighted primitive to the injected function."""
        if self._weighted_func is None:
            raise NotImplementedError("no weighted source injected")
        return self._weighted_func(weights)

    def seed(self, a=None):
        """Delegate seeding to the injected function."""
        if self._seed_func is None:
            raise NotImplementedError("no seed source injected")
        return self._seed_func(a)


class RandomTool(RandomToolBase):
    """
    Default random tool backed by the `random` module.
    Each instance owns an independent random.Random state, so seeding one
    instance never disturbs another. Fast, reproducible via seed().
    """

    __slots__ = ("_random",)

    def __init__(self, rand=None):
        """Wrap `rand` (default: a fresh random.Random instance)."""
        self._random = random.Random() if rand is None else rand

    def random(self):
        return self._random.random()

    def _randbelow(self, n):
        return self._random.randrange(n)

    def _weighted_index(self, weights):
        total = 0.0
        for w in weights:
            if w < 0:
                raise ValueError("weights must be non-negative")
            total += w
        if total <= 0:
            raise ValueError("total of weights must be positive")
        r = self.random() * total
        upto = 0.0
        for i, w in enumerate(weights):
            upto += w
            if r < upto:
                return i
        return len(weights) - 1

    def seed(self, a=None):
        """Seed the wrapped random generator for reproducible sequences."""
        self._random.seed(a)


class SecureRandomTool(RandomToolBase):
    """
    Cryptographically secure random tool built on the standard library
    `secrets` module (os.urandom). Intentionally not seedable - seeding
    would break its security guarantees.
    """

    __slots__ = ()

    def random(self):
        return secrets.randbits(53) / float(1 << 53)

    def _randbelow(self, n):
        return secrets.randbelow(n)

    def _weighted_index(self, weights):
        if all(type(w) is int for w in weights):
            if any(w < 0 for w in weights):
                raise ValueError("weights must be non-negative")
            total = sum(weights)
            if total <= 0:
                raise ValueError("total of weights must be positive")
            r = secrets.randbelow(total)
            upto = 0
            for i, w in enumerate(weights):
                upto += w
                if r < upto:
                    return i
            return len(weights) - 1
        total = 0.0
        for w in weights:
            if w < 0:
                raise ValueError("weights must be non-negative")
            total += w
        if total <= 0:
            raise ValueError("total of weights must be positive")
        r = self.random() * total
        upto = 0.0
        for i, w in enumerate(weights):
            upto += w
            if r < upto:
                return i
        return len(weights) - 1

    def seed(self, a=None):
        raise TypeError("SecureRandomTool cannot be seeded (cryptographic entropy)")


default = RandomTool()
secure = SecureRandomTool()

__all__ = ("RandomToolBase", "DelegatedRandomTool", "RandomTool",
           "SecureRandomTool", "default", "secure")
