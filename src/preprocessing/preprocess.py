from torchvision import transforms
from src.utils.dataset import ImageFolderDataset

def build_transforms(image_size, mean, std, aug=False):
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    if aug:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose(base)

def make_datasets(cfg):
    transform_train = build_transforms(
        cfg["preprocessing"]["image_size"],
        cfg["preprocessing"]["normalize_mean"],
        cfg["preprocessing"]["normalize_std"],
        aug=True
    )
    transform_eval = build_transforms(
        cfg["preprocessing"]["image_size"],
        cfg["preprocessing"]["normalize_mean"],
        cfg["preprocessing"]["normalize_std"],
        aug=False
    )
    dataset = ImageFolderDataset(cfg["paths"]["data_raw"], transform=transform_eval)
    return dataset, transform_train, transform_eval
