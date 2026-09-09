"""
Image Classification Demo using Atomic Contrast Point Matching.

Demonstrates:
1. Image -> 2D tensor (grayscale, via tkinter)
2. Atomic feature extraction (passive boundary + pooling)
3. Binarization + contrast matching

Principle: local analysis, passive boundary extraction, no global features.

Pure standard library + cos_comparison: no Pillow, numpy, or other deps.
Image loading uses tkinter PhotoImage (GIF/PNG/PGM/PPM formats).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atomic_feature import extract_grouped, binarize
from hierarchical_match import HierarchicalMatcher
from contrast_match import ContrastMatcher


def load_image(path, size=64):
    """
    Load image to 2D grayscale tensor (list of lists) using tkinter.
    Supports GIF, PNG, PGM, PPM formats (tkinter PhotoImage limitations).
    This is the ONLY data-loading function; training receives tensors.
    Pure standard library: no Pillow or other third-party deps.
    """
    import tkinter
    try:
        root = tkinter.Tk()
        root.withdraw()
        photo = tkinter.PhotoImage(file=path)
        # Resize using subsample/zoom if needed
        orig_w, orig_h = photo.width(), photo.height()
        if orig_w != size or orig_h != size:
            # Simple resize via zoom/subsample (integer factors only)
            # For arbitrary resize, use pixel sampling
            pass
        result = []
        for i in range(size):
            row = []
            src_y = min(int(i * orig_h / size), orig_h - 1)
            for j in range(size):
                src_x = min(int(j * orig_w / size), orig_w - 1)
                # get() returns "R G B" string
                rgb = photo.get(src_x, src_y)
                if isinstance(rgb, str):
                    parts = rgb.split()
                    r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                else:
                    r, g, b = rgb[0], rgb[1], rgb[2]
                # Luminance grayscale
                gray = 0.299 * r + 0.587 * g + 0.114 * b
                row.append(gray)
            result.append(row)
        root.destroy()
        return result
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None


def prepare_image_dataset(data_dir, size=64, n_per_class=50, test_ratio=0.3):
    """
    Load images from directory structure:
        data_dir/
            class1/
                img1.png
                img2.jpg
            class2/
                ...
    """
    classes = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Found {len(classes)} classes")

    train_imgs, train_labels = [], []
    test_imgs, test_labels = [], []

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        files = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(exts)])[:n_per_class]
        n_test = int(len(files) * test_ratio)
        train_files = files[n_test:]
        test_files = files[:n_test]

        for f in train_files:
            img = load_image(os.path.join(cls_dir, f), size=size)
            if img is not None:
                train_imgs.append(img)
                train_labels.append(cls)

        for f in test_files:
            img = load_image(os.path.join(cls_dir, f), size=size)
            if img is not None:
                test_imgs.append(img)
                test_labels.append(cls)

    print(f"Train: {len(train_imgs)}, Test: {len(test_imgs)}")
    return train_imgs, train_labels, test_imgs, test_labels


def extract_image_features(images, boundary_thresh=10, raw_thresh=10):
    """Extract grouped features from list of 2D tensors."""
    boundary_feats, raw_feats = [], []
    for img in images:
        b, r = extract_grouped(img)
        boundary_feats.append(binarize(b, top_percent=boundary_thresh))
        raw_feats.append(binarize(r, top_percent=raw_thresh))
    return boundary_feats, raw_feats


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo_image.py <data_dir> [size] [n_per_class]")
        print("\nRunning synthetic demo...")
        import random
        random.seed(42)

        # Synthetic image data with patterns
        def make_pattern(class_id, size=64):
            img = [[0.0] * size for _ in range(size)]
            if class_id == 0:  # horizontal stripes
                for i in range(0, size, 8):
                    for di in range(4):
                        for j in range(size):
                            img[i + di][j] = 255.0
            elif class_id == 1:  # vertical stripes
                for j in range(0, size, 8):
                    for dj in range(4):
                        for i in range(size):
                            img[i][j + dj] = 255.0
            else:  # checkerboard
                for i in range(0, size, 8):
                    for j in range(0, size, 8):
                        if (i // 8 + j // 8) % 2 == 0:
                            for di in range(8):
                                for dj in range(8):
                                    img[i + di][j + dj] = 255.0
            # Add noise
            for i in range(size):
                for j in range(size):
                    img[i][j] += random.gauss(0, 10)
            return img

        train_imgs = [make_pattern(i % 3) for i in range(90)]
        train_labels = [f"class_{i % 3}" for i in range(90)]
        test_imgs = [make_pattern(i % 3) for i in range(30)]
        test_labels = [f"class_{i % 3}" for i in range(30)]
    else:
        data_dir = sys.argv[1]
        size = int(sys.argv[2]) if len(sys.argv) > 2 else 64
        n_per_class = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        train_imgs, train_labels, test_imgs, test_labels = prepare_image_dataset(
            data_dir, size=size, n_per_class=n_per_class)

    # Extract features
    print("\nExtracting features...")
    train_b, train_r = extract_image_features(train_imgs)
    test_b, test_r = extract_image_features(test_imgs)
    print(f"Boundary dim: {len(train_b[0])}, Raw dim: {len(train_r[0])}")

    # Flat matching
    print("\n=== Flat Contrast Matching ===")
    train_combined = [b + r for b, r in zip(train_b, train_r)]
    test_combined = [b + r for b, r in zip(test_b, test_r)]
    matcher = ContrastMatcher()
    matcher.fit(train_combined, train_labels)
    acc_flat = matcher.accuracy(test_combined, test_labels, k=1)
    print(f"Accuracy: {acc_flat*100:.1f}%")

    # Hierarchical matching
    print("\n=== Hierarchical Matching ===")
    hmatcher = HierarchicalMatcher()
    hmatcher.fit(train_b, train_r, train_labels)
    acc_hier = hmatcher.accuracy(test_b, test_r, test_labels, coarse_n=10)
    print(f"Accuracy: {acc_hier*100:.1f}%")

    print(f"\nBest: {max(acc_flat, acc_hier)*100:.1f}%")


if __name__ == '__main__':
    main()
