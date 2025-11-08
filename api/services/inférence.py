import torch
from torchvision import transforms
from PIL import Image

class TransferInference:
    def __init__(self, model, class_to_idx):
        self.model = model.eval()
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

    def predict(self, pil_image: Image.Image):
        x = self.transform(pil_image).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            pred_idx = torch.argmax(logits, dim=1).item()
        return self.idx_to_class[pred_idx]
