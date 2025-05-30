import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

class UltrasoundDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_image_paths_and_labels(root_dir):
    classes = sorted(os.listdir(root_dir))  # Sort to keep class indices consistent
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    image_paths = []
    labels = []

    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(cls_dir, img_name))
                labels.append(class_to_idx[cls])

    return image_paths, labels, class_to_idx


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform


def prepare_dataloaders(train_dir, test_dir, batch_size=8):
    # Get paths and labels
    train_paths, train_labels, class_to_idx = get_image_paths_and_labels(train_dir)
    test_paths, test_labels, _ = get_image_paths_and_labels(test_dir)

    # Split train into train/val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )

    # Transforms
    train_transform, test_transform = get_transforms()

    # Datasets
    train_dataset = UltrasoundDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = UltrasoundDataset(val_paths, val_labels, transform=test_transform)
    test_dataset = UltrasoundDataset(test_paths, test_labels, transform=test_transform)

    # DataLoaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_to_idx