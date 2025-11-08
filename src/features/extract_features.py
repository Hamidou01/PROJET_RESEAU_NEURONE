import torch
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


# === Fonction pour choisir le backbone ===
def get_backbone(name: str):
    if name == "vgg16":
        from torchvision.models import VGG16_Weights
        net = models.vgg16(weights=VGG16_Weights.DEFAULT)
        return net.features

    elif name == "alexnet":
        from torchvision.models import AlexNet_Weights
        net = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        return net.features

    elif name == "inceptionv3":
        from torchvision.models import Inception_V3_Weights
        net = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        net.eval()
        # On garde uniquement les blocs convolutionnels + pooling global
        backbone = torch.nn.Sequential(
            net.Conv2d_1a_3x3,
            net.Conv2d_2a_3x3,
            net.Conv2d_2b_3x3,
            net.Conv2d_3b_1x1,
            net.Conv2d_4a_3x3,
            net.Mixed_5b,
            net.Mixed_5c,
            net.Mixed_5d,
            net.Mixed_6a,
            net.Mixed_6b,
            net.Mixed_6c,
            net.Mixed_6d,
            net.Mixed_6e,
            net.Mixed_7a,
            net.Mixed_7b,
            net.Mixed_7c,
            torch.nn.AdaptiveAvgPool2d((1, 1))  # vecteur global
        )
        return backbone

    else:
        raise ValueError(f"Backbone inconnu: {name}")


# === Fonction d’extraction des features ===
def extract_features(backbone_name: str, dataset, batch_size: int = 16):
    backbone = get_backbone(backbone_name)
    backbone.eval()

    # Taille d’entrée spécifique à chaque backbone
    size = 299 if backbone_name == "inceptionv3" else 224
    adapter = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])

    # Dataset adapté pour appliquer la bonne transformation
    class AdapterDataset(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            img, y = self.base[idx]
            # img doit être PIL.Image → dataset doit être créé avec transform=None
            img = adapter(img)
            return img, y

    adapted = AdapterDataset(dataset)
    loader = DataLoader(adapted, batch_size=batch_size, shuffle=False)

    feats_list, labels_list = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"Features: {backbone_name}"):
            f = backbone(x)
            f = f.view(f.size(0), -1)  # aplatissement
            feats_list.append(f.cpu().numpy())
            labels_list.append(y.cpu().numpy())

    X = np.vstack(feats_list)
    y = np.hstack(labels_list)
    return X, y
