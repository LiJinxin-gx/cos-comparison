"""
Image Classification Demo: 3-level hierarchical pooling + active cosine matching.

Demonstrates the validated core principles:
1. Continuous mapping level isolation (3 pooling levels [1,2,4])
2. Active mode only (cosine similarity, no passive averaging)
3. K=1 nearest neighbor (no voting, no weighted fusion)
4. Level isolation: take max across levels (no weighted fusion)

Data loading is separated from processing:
- load_image(): file -> 2D tensor (tkinter, standard library only)
- HierarchicalClassifier: tensor -> classification

Pure standard library: no numpy, no third-party deps.
Image loading uses tkinter PhotoImage (GIF/PNG/PGM/PPM).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from group_hierarchical import GroupHierarchicalClassifier


def load_image(path, size=28):
    """
    Load image to 2D grayscale tensor using tkinter.
    Supports GIF, PNG, PGM, PPM formats.

    This is the ONLY data-loading function.
    Processing code receives pure 2D tensors.
    """
    import tkinter
    root = tkinter.Tk()
    root.withdraw()
    try:
        photo = tkinter.PhotoImage(file=path)
        orig_w, orig_h = photo.width(), photo.height()
        result = []
        for i in range(size):
            row = []
            src_y = min(int(i * orig_h / size), orig_h - 1)
            for j in range(size):
                src_x = min(int(j * orig_w / size), orig_w - 1)
                rgb = photo.get(src_x, src_y)
                if isinstance(rgb, str):
                    parts = rgb.split()
                    r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                else:
                    r, g, b = rgb[0], rgb[1], rgb[2]
                gray = 0.299 * r + 0.587 * g + 0.114 * b
                row.append(gray)
            result.append(row)
        return result
    finally:
        root.destroy()


def load_dataset(data_dir, size=28, n_per_class=50, test_ratio=0.3):
    """
    Load image dataset from directory structure:
        data_dir/
            class1/
                img1.png
                ...
            class2/
                ...

    Returns: (train_images, train_labels, test_images, test_labels)
    """
    classes = sorted(d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d)))
    print(f"Found {len(classes)} classes")

    train_imgs, train_labels = [], []
    test_imgs, test_labels = [], []
    exts = ('.png', '.gif', '.pgm', '.ppm')

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(exts))[:n_per_class]
        n_test = int(len(files) * test_ratio)
        for f in files[n_test:]:
            img = load_image(os.path.join(cls_dir, f), size=size)
            if img:
                train_imgs.append(img)
                train_labels.append(cls)
        for f in files[:n_test]:
            img = load_image(os.path.join(cls_dir, f), size=size)
            if img:
                test_imgs.append(img)
                test_labels.append(cls)

    print(f"Train: {len(train_imgs)}, Test: {len(test_imgs)}")
    return train_imgs, train_labels, test_imgs, test_labels


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo_image.py <data_dir> [size] [n_per_class]")
        print("\nRunning synthetic demo...")
        import random
        random.seed(42)

        def make_pattern(class_id, size=28):
            img = [[0.0] * size for _ in range(size)]
            if class_id == 0:  # horizontal
                for i in range(0, size, 6):
                    for di in range(3):
                        for j in range(size):
                            img[i + di][j] = 200.0
            elif class_id == 1:  # vertical
                for j in range(0, size, 6):
                    for dj in range(3):
                        for i in range(size):
                            img[i][j + dj] = 200.0
            else:  # diagonal
                for i in range(size):
                    j = (i + 8) % size
                    for dj in range(-2, 3):
                        img[i][(j + dj) % size] = 200.0
            for i in range(size):
                for j in range(size):
                    img[i][j] += random.gauss(0, 5)
            return img

        train_imgs = [make_pattern(i % 3) for i in range(300)]
        train_labels = [f"pattern_{i % 3}" for i in range(300)]
        test_imgs = [make_pattern(i % 3) for i in range(90)]
        test_labels = [f"pattern_{i % 3}" for i in range(90)]
    else:
        data_dir = sys.argv[1]
        size = int(sys.argv[2]) if len(sys.argv) > 2 else 28
        n_per = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        train_imgs, train_labels, test_imgs, test_labels = load_dataset(
            data_dir, size=size, n_per_class=n_per)

    # Train and evaluate
    print("\nTraining hierarchical classifier...")
    clf = GroupHierarchicalClassifier(
        pool_factors=[1, 2, 4],
        use_passive=False
    )
    clf.fit(train_imgs, train_labels)

    print("Evaluating...")
    acc = clf.accuracy(test_imgs, test_labels)
    print(f"\nAccuracy: {acc * 100:.1f}%")
    print(f"Principle: 3-level pooling [1,2,4], active cosine, K=1, max across levels")


if __name__ == '__main__':
    main()
