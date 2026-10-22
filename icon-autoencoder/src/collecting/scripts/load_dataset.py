from pathlib import Path
from PIL import Image
import numpy as np


def load_split(split_name='train'):
    split_file = Path('src/collecting/splits') / f'{split_name}.txt'
    with open(split_file, 'r') as f:
        return [line.strip() for line in f]


def load_image(filename):
    img_path = Path('src/collecting/data_processed/images') / filename
    img = Image.open(img_path)
    return np.array(img)


def load_batch(filenames, normalize=True):
    batch = []
    for fname in filenames:
        img = load_image(fname)
        if normalize:
            img = img.astype(np.float32) / 255.0
        batch.append(img)
    return np.array(batch)


if __name__ == '__main__':
    train_files = load_split('train')
    val_files = load_split('val')
    test_files = load_split('test')
    
    print(f"Train: {len(train_files)}")
    print(f"Val: {len(val_files)}")
    print(f"Test: {len(test_files)}")

    img = load_image(train_files[0])
    print(f"\nImages: {img.shape}, dtype: {img.dtype}")

    batch = load_batch(train_files[:16])
    print(f"Batch: {batch.shape}, range: [{batch.min():.2f}, {batch.max():.2f}]")
