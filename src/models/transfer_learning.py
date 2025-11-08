import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from torchvision import transforms
from torchvision.models import ResNet18_Weights, VGG16_Weights


def build_transfer_model(backbone_name, num_classes):
    if backbone_name == "resnet18":
        net = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    elif backbone_name == "vgg16":
        net = models.vgg16(weights=VGG16_Weights.DEFAULT)
        in_features = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(in_features, num_classes)
        return net
    else:
        raise ValueError(f"Unsupported backbone for transfer: {backbone_name}")


def train_transfer_model(dataset, num_classes, epochs=12, batch_size=32, lr=0.001, backbone="resnet18"):
    # Transformation pour ResNet/VGG
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet18 et VGG16 attendent 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # AdapterDataset pour appliquer la transform
    class AdapterDataset(torch.utils.data.Dataset):
        def __init__(self, base, transform):
            self.base = base
            self.transform = transform
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            img, y = self.base[idx]
            img = self.transform(img)  # ⚠️ convertit PIL → Tensor
            return img, y

    adapted = AdapterDataset(dataset, transform)

    # Split train/val
    val_size = int(0.2 * len(adapted))
    train_size = len(adapted) - val_size
    train_ds, val_ds = random_split(adapted, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Construire le modèle
    model = build_transfer_model(backbone, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Boucle d’entraînement
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, lbls in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                outputs = model(imgs)
                val_loss += criterion(outputs, lbls).item()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model
