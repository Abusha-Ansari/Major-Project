from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS, IMAGE_SIZE


def train_transforms():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def eval_transforms():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def make_loader(data_dir, batch_size=DEFAULT_BATCH_SIZE, train=True, shuffle=None):
    data_dir = Path(data_dir)
    transform = train_transforms() if train else eval_transforms()
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    if shuffle is None:
        shuffle = train
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=DEFAULT_NUM_WORKERS,
    )
    return loader, dataset
