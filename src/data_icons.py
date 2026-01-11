import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class IconDataset(Dataset):
    def __init__(self, images_dir, split_file):
        self.images_dir = Path(images_dir)
        
        with open(split_file, 'r') as f:
            self.filenames = [line.strip() for line in f if line.strip()]
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = self.images_dir / fname
        
        img = Image.open(img_path).convert('RGB')
        img_arr = np.array(img, dtype=np.float32) / 255.0
        
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)
        
        return img_tensor


def get_data_loaders(images_dir, splits_dir, batch_size, num_workers=0):
    train_ds = IconDataset(images_dir, os.path.join(splits_dir, 'train.txt'))
    val_ds = IconDataset(images_dir, os.path.join(splits_dir, 'val.txt'))
    test_ds = IconDataset(images_dir, os.path.join(splits_dir, 'test.txt'))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
