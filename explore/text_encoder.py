"""
Text Frequency Encoder using UnitMap.

Converts text streams to frequency tensors for contrast-point matching.
Uses cos_comparison.interface.tools.math_tool.unit_map.

Principle: structure-first, frequency-based encoding, no linguistic prior
(no POS tagging, no word embeddings). Characters/words are atomic units.

Pure standard library + cos_comparison: no numpy or third-party dependencies.
"""
from cos_comparison.interface.tools.math_tool.unit_map import UnitMap


class TextFrequencyEncoder:
    """
    Encode text as frequency tensors using UnitMap.

    Two modes:
    - character: treat each character as a unit
    - word: treat each word as a unit (split by whitespace)

    Usage:
        encoder = TextFrequencyEncoder(mode='character')
        encoder.fit(texts)  # build vocabulary
        tensor = encoder.transform(texts)  # list of lists (frequency vectors)
    """

    def __init__(self, mode='character', max_vocab=10000):
        """
        Args:
            mode: 'character' or 'word'
            max_vocab: maximum vocabulary size (top by frequency)
        """
        self.mode = mode
        self.max_vocab = max_vocab
        self.unit_map = UnitMap()
        self.vocab = {}  # unit -> index
        self.vocab_size = 0

    def _tokenize(self, text):
        """Split text into units based on mode."""
        if self.mode == 'character':
            return list(text)
        elif self.mode == 'word':
            return text.split()
        else:
            return list(text)

    def fit(self, texts):
        """
        Build vocabulary from texts using UnitMap frequency counting.

        Args:
            texts: list of strings
        """
        self.unit_map = UnitMap()
        for text in texts:
            tokens = self._tokenize(text)
            self.unit_map.add(tokens)

        # Get most common units (limited by max_vocab)
        common = self.unit_map.most_common(k=self.max_vocab)
        self.vocab = {}
        for idx, item in enumerate(common):
            # most_common returns (unit, count, runs) or (unit, count)
            if isinstance(item, tuple):
                unit = item[0]
            else:
                unit = item
            self.vocab[unit] = idx
        self.vocab_size = len(self.vocab)
        return self

    def transform(self, texts):
        """
        Transform texts to frequency tensors.

        Args:
            texts: list of strings

        Returns:
            list of lists: (n_texts, vocab_size) frequency counts
        """
        result = []
        for text in texts:
            tokens = self._tokenize(text)
            local_map = UnitMap()
            local_map.add(tokens)
            vec = [0.0] * self.vocab_size
            for unit, idx in self.vocab.items():
                vec[idx] = float(local_map.count(unit))
            result.append(vec)
        return result

    def fit_transform(self, texts):
        """Fit vocabulary and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    def normalize(self, tensor):
        """
        Normalize frequency tensor to [0, 1] per sample (L1 norm).
        Makes frequencies comparable across texts of different lengths.
        """
        result = []
        for vec in tensor:
            total = sum(vec)
            if total == 0:
                result.append(vec[:])
            else:
                result.append([v / total for v in vec])
        return result


def encode_texts(texts, mode='character', max_vocab=10000, normalize=True):
    """
    Convenience function: encode list of texts to frequency tensor.

    Args:
        texts: list of strings
        mode: 'character' or 'word'
        max_vocab: max vocabulary size
        normalize: whether to L1-normalize

    Returns:
        (list of lists, encoder)
    """
    encoder = TextFrequencyEncoder(mode=mode, max_vocab=max_vocab)
    tensor = encoder.fit_transform(texts)
    if normalize:
        tensor = encoder.normalize(tensor)
    return tensor, encoder


if __name__ == '__main__':
    # Demo
    texts = [
        "hello world",
        "hello there",
        "world peace",
        "hello hello",
    ]
    tensor, encoder = encode_texts(texts, mode='character')
    print(f"Vocab size: {encoder.vocab_size}")
    print(f"Tensor shape: ({len(tensor)}, {len(tensor[0])})")
    print(f"Sample frequencies: {tensor[0][:10]}")
