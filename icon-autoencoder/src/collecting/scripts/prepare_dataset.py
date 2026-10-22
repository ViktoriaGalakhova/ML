#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path
import random
import sys

from PIL import Image
import cairosvg
import imagehash
import numpy as np

REPO_URL = "https://github.com/tabler/tabler-icons"
ICONS_SUBDIR = "icons"


def download_repo(target_dir):
    import git
    import stat

    if os.path.exists(target_dir):
        def remove_readonly(func, path, _):
            os.chmod(path, stat.S_IWRITE)
            func(path)

        shutil.rmtree(target_dir, onerror=remove_readonly)

    try:
        git.Repo.clone_from(REPO_URL, target_dir)
        print(f"Clone repo")
    except Exception as e:
        print(f"error: {e}")
        sys.exit(1)


def find_svg_files(base_path):
    search_path = os.path.join(base_path, ICONS_SUBDIR)

    if not os.path.exists(search_path):
        print(f"folder {search_path} not found")
        sys.exit(1)

    svg_files = []
    for root, dirs, files in os.walk(search_path):
        for file in files:
            if file.lower().endswith('.svg'):
                svg_files.append(os.path.join(root, file))

    return svg_files


def svg_to_png(svg_path, size):
    try:
        png_data = cairosvg.svg2png(url=svg_path, output_width=size * 2, output_height=size * 2)
        return png_data
    except:
        return None


def process_icon(svg_path, target_size):
    from io import BytesIO

    png_data = svg_to_png(svg_path, target_size)
    if png_data is None:
        return None

    img = Image.open(BytesIO(png_data))

    if img.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    w, h = img.size
    ratio = min(target_size / w, target_size / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    result = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    result.paste(img_resized, (paste_x, paste_y))

    return result


def is_empty_image(img, threshold=0.95):
    arr = np.array(img)
    white_pixels = np.sum(np.all(arr > 240, axis=2))
    total_pixels = arr.shape[0] * arr.shape[1]
    return (white_pixels / total_pixels) > threshold


def process_all(svg_files, output_dir, size):
    print(f"\n Found {len(svg_files)} files")
    print("Processing")

    os.makedirs(output_dir, exist_ok=True)

    processed = []
    hashes = set()

    for svg_path in svg_files:
        img = process_icon(svg_path, size)

        if img is None:
            continue

        if is_empty_image(img):
            continue

        img_hash = imagehash.average_hash(img)
        if img_hash in hashes:
            continue
        hashes.add(img_hash)

        processed.append(img)

    print(f"\nSaving {len(processed)} images")
    for idx, img in enumerate(processed, 1):
        filename = f"{idx:06d}.png"
        img.save(os.path.join(output_dir, filename), 'PNG')

    return processed


def split_dataset(n_total, splits, seed):
    random.seed(seed)

    indices = list(range(1, n_total + 1))
    random.shuffle(indices)

    train_ratio, val_ratio, test_ratio = splits
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def save_splits(train, val, test, splits_dir):
    os.makedirs(splits_dir, exist_ok=True)

    with open(os.path.join(splits_dir, 'train.txt'), 'w') as f:
        for idx in train:
            f.write(f"{idx:06d}.png\n")

    with open(os.path.join(splits_dir, 'val.txt'), 'w') as f:
        for idx in val:
            f.write(f"{idx:06d}.png\n")

    with open(os.path.join(splits_dir, 'test.txt'), 'w') as f:
        for idx in test:
            f.write(f"{idx:06d}.png\n")

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    base_dir = Path('src/collecting')
    raw_dir = base_dir / 'data_raw' / 'repo'
    processed_dir = base_dir / 'data_processed' / 'images'
    splits_dir = base_dir / 'splits'

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    download_repo(str(raw_dir))

    svg_files = find_svg_files(str(raw_dir))
    n_original = len(svg_files)

    if n_original == 0:
        print("Error: no SVG files found")
        sys.exit(1)

    processed = process_all(svg_files, str(processed_dir), args.size)
    n_processed = len(processed)

    print(f"\nTotal processed: {n_processed} images")

    print("\nCreating splits")
    train, val, test = split_dataset(n_processed, [0.8, 0.1, 0.1], args.seed)
    save_splits(train, val, test, str(splits_dir))

    print(f"\nDone:images saved to {processed_dir}")


if __name__ == '__main__':
    main()