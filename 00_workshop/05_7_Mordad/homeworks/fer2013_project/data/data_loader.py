import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class FER2013Dataset(Dataset):
    def __init__(self, csv_file, split="Training", transform=None):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data["Usage"] == split]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        emotion = row["emotion"]
        pixels = np.array(row["pixels"].split(), dtype="uint8").reshape(48, 48)
        image = Image.fromarray(pixels).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, emotion


def get_dataloaders(csv_path, batch_size, image_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = FER2013Dataset(
        csv_path, split="Training", transform=train_transform
    )
    val_dataset = FER2013Dataset(
        csv_path, split="PublicTest", transform=test_val_transform
    )
    test_dataset = FER2013Dataset(
        csv_path, split="PrivateTest", transform=test_val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, test_loader
