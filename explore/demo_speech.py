"""
Speech Recognition Demo using Atomic Contrast Point Matching.

Demonstrates the full pipeline:
1. Audio -> spectrogram (2D tensor)
2. Atomic feature extraction (passive boundary + pooling)
3. Binarization (contrast point sets)
4. Hierarchical matching (coarse boundary -> fine combined)

This is a self-contained demo. Data loading is separate from training.
Training only receives tensors.

Pure standard library + cos_comparison for core logic.
scipy/Pillow are optional (only for audio loading).
"""
import os
import sys

# Add explore directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atomic_feature import extract_grouped, binarize
from hierarchical_match import HierarchicalMatcher
from contrast_match import ContrastMatcher


def audio_to_spectrogram(audio, sr=16000, nperseg=256, size=64):
    """
    Convert 1D audio array to 2D spectrogram tensor.
    Simple STFT using pure Python (no numpy/scipy required for core).
    This is the ONLY data-loading function; training receives tensors.
    """
    import math
    n = len(audio)
    hop = nperseg // 2
    n_frames = max(1, (n - nperseg) // hop + 1)
    n_freqs = nperseg // 2 + 1

    # Simple DFT per frame (pure Python)
    spec = []
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + nperseg]
        if len(frame) < nperseg:
            frame = frame + [0.0] * (nperseg - len(frame))
        # Apply Hann window
        windowed = [frame[j] * (0.5 - 0.5 * math.cos(2 * math.pi * j / nperseg))
                    for j in range(nperseg)]
        # DFT magnitudes (first n_freqs bins)
        magnitudes = []
        for k in range(n_freqs):
            real = sum(windowed[j] * math.cos(2 * math.pi * k * j / nperseg)
                       for j in range(nperseg))
            imag = sum(-windowed[j] * math.sin(2 * math.pi * k * j / nperseg)
                       for j in range(nperseg))
            magnitudes.append(math.log1p(math.sqrt(real * real + imag * imag)))
        spec.append(magnitudes)

    # Resize to size x size using nearest-neighbor
    if not spec:
        return [[0.0] * size for _ in range(size)]
    result = [[0.0] * size for _ in range(size)]
    max_val = max(max(row) for row in spec) + 1e-10
    for i in range(size):
        src_i = min(int(i * n_freqs / size), n_freqs - 1)
        for j in range(size):
            src_j = min(int(j * n_frames / size), n_frames - 1)
            result[i][j] = spec[src_j][src_i] / max_val * 255.0
    return result


def load_wav(path, target_len=16000):
    """Load WAV file to 1D list of floats using standard library wave module."""
    import wave
    import struct
    try:
        with wave.open(path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None

    # Decode samples based on sample width
    if sampwidth == 2:
        fmt = f'<{n_frames * n_channels}h'
        samples = struct.unpack(fmt, raw)
    elif sampwidth == 1:
        samples = [b - 128 for b in raw]
    else:
        print(f"Unsupported sample width: {sampwidth}")
        return None

    # Take first channel and normalize to [-1, 1]
    data = [float(samples[i]) / 32768.0 for i in range(0, len(samples), n_channels)]

    # Resample to 16000 Hz if needed
    if framerate != 16000:
        ratio = 16000 / framerate
        new_len = int(len(data) * ratio)
        data = [data[min(int(i / ratio), len(data) - 1)] for i in range(new_len)]

    if len(data) < target_len:
        data = data + [0.0] * (target_len - len(data))
    else:
        data = data[:target_len]
    return data


def prepare_dataset(data_dir, n_per_class=50, test_ratio=0.3):
    """
    Load audio files from directory structure:
        data_dir/
            class1/
                file1.wav
                file2.wav
            class2/
                ...

    Returns train/test tensors and labels.
    """
    classes = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Found {len(classes)} classes: {classes}")

    train_specs, train_labels = [], []
    test_specs, test_labels = [], []

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files = sorted([f for f in os.listdir(cls_dir) if f.endswith('.wav')])[:n_per_class]
        n_test = int(len(files) * test_ratio)
        train_files = files[n_test:]
        test_files = files[:n_test]

        for f in train_files:
            audio = load_wav(os.path.join(cls_dir, f))
            if audio is not None:
                spec = audio_to_spectrogram(audio)
                if spec is not None:
                    train_specs.append(spec)
                    train_labels.append(cls)

        for f in test_files:
            audio = load_wav(os.path.join(cls_dir, f))
            if audio is not None:
                spec = audio_to_spectrogram(audio)
                if spec is not None:
                    test_specs.append(spec)
                    test_labels.append(cls)

    print(f"Train: {len(train_specs)}, Test: {len(test_specs)}")
    return train_specs, train_labels, test_specs, test_labels


def extract_features(specs, boundary_thresh=10, raw_thresh=10):
    """
    Extract grouped features and binarize.
    Training only receives tensors (specs), outputs binary sets.
    """
    boundary_feats, raw_feats = [], []
    for spec in specs:
        b, r = extract_grouped(spec)
        boundary_feats.append(binarize(b, top_percent=boundary_thresh))
        raw_feats.append(binarize(r, top_percent=raw_thresh))
    return boundary_feats, raw_feats


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo_speech.py <data_dir> [n_per_class]")
        print("Example: python demo_speech.py /mnt/d/testdata/speech_commands 100")
        # Run with synthetic data demo
        print("\nRunning synthetic demo...")
        import random
        random.seed(42)
        n_train, n_test = 100, 30
        train_specs = [[[random.random() for _ in range(64)] for _ in range(64)]
                       for _ in range(n_train)]
        train_labels = [f"class_{i % 5}" for i in range(n_train)]
        test_specs = [[[random.random() for _ in range(64)] for _ in range(64)]
                      for _ in range(n_test)]
        test_labels = [f"class_{i % 5}" for i in range(n_test)]
    else:
        data_dir = sys.argv[1]
        n_per_class = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        train_specs, train_labels, test_specs, test_labels = prepare_dataset(
            data_dir, n_per_class=n_per_class)

    # Extract features (tensors in -> binary sets out)
    print("\nExtracting features...")
    train_b, train_r = extract_features(train_specs)
    test_b, test_r = extract_features(test_specs)
    print(f"Boundary dim: {len(train_b[0])}, Raw dim: {len(train_r[0])}")

    # Method 1: Flat contrast matching
    print("\n=== Flat Contrast Matching (K=1) ===")
    train_combined = [b + r for b, r in zip(train_b, train_r)]
    test_combined = [b + r for b, r in zip(test_b, test_r)]
    matcher = ContrastMatcher()
    matcher.fit(train_combined, train_labels)
    acc_flat = matcher.accuracy(test_combined, test_labels, k=1)
    print(f"Accuracy: {acc_flat*100:.1f}%")

    # Method 2: Hierarchical matching
    print("\n=== Hierarchical Matching (coarse boundary -> fine combined) ===")
    hmatcher = HierarchicalMatcher()
    hmatcher.fit(train_b, train_r, train_labels)
    acc_hier = hmatcher.accuracy(test_b, test_r, test_labels, coarse_n=10)
    print(f"Accuracy: {acc_hier*100:.1f}%")

    print(f"\nBest: {max(acc_flat, acc_hier)*100:.1f}%")


if __name__ == '__main__':
    main()
