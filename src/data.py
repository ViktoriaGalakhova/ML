import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from src.config import Config


class DogDuckDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = row["label_int"]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def get_transforms(is_train=False):
    if is_train:
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=Config.FLIP_PROB),
            transforms.ColorJitter(
                brightness=Config.COLOR_JITTER,
                contrast=Config.COLOR_JITTER,
                saturation=Config.COLOR_JITTER
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_data_loaders():
    splits_dir = Path("data/splits")
    
    train_ds = DogDuckDataset(
        splits_dir / "train.csv",
        transform=get_transforms(is_train=True)
    )
    
    val_ds = DogDuckDataset(
        splits_dir / "val.csv",
        transform=get_transforms(is_train=False)
    )
    
    test_ds = DogDuckDataset(
        splits_dir / "test.csv",
        transform=get_transforms(is_train=False)
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False,
    )
    
    return train_loader, val_loader, test_loader
