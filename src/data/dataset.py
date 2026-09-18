from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from src.config import BATCH_SIZE, DATA_DIR, IMAGE_SIZE, SEED, TEST_SPLIT, TRAIN_SPLIT

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ]
)


def _stratified_split(targets):
    train_idx, temp_idx = train_test_split(
        range(len(targets)),
        train_size=TRAIN_SPLIT,
        stratify=targets,
        random_state=SEED,
    )
    temp_targets = [targets[i] for i in temp_idx]
    val_size = TEST_SPLIT / (1 - TRAIN_SPLIT)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_size,
        stratify=temp_targets,
        random_state=SEED,
    )
    return train_idx, val_idx, test_idx


def get_dataloaders(batch_size: int = BATCH_SIZE, num_workers: int = 2):
    train_base = datasets.ImageFolder(DATA_DIR, transform=TRAIN_TRANSFORM)
    eval_base = datasets.ImageFolder(DATA_DIR, transform=EVAL_TRANSFORM)

    train_idx, val_idx, test_idx = _stratified_split(train_base.targets)

    train_set = Subset(train_base, train_idx)
    val_set = Subset(eval_base, val_idx)
    test_set = Subset(eval_base, test_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_base.classes
