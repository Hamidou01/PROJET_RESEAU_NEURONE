from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from src.utils.dataset import ImageFolderDataset
from src.utils.helpers import load_config

app = FastAPI()

# Activer CORS pour permettre au frontend (http://127.0.0.1:8080) d'appeler l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tu peux mettre ["http://127.0.0.1:8080"] pour restreindre
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger config et dataset pour récupérer les classes
cfg = load_config()
dataset = ImageFolderDataset(cfg["paths"]["data_raw"], transform=None)
num_classes = len(dataset.class_to_idx)
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

# Charger le modèle ResNet18
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
state_path = f"{cfg['paths']['models_transfer']}/resnet18_{num_classes}classes.pth"
model.load_state_dict(torch.load(state_path, map_location="cpu"))
model.eval()

# Transformation pour prédiction
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.get("/")
def root():
    return {"message": "API is running. Use /predict to classify images."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        pred_idx = torch.argmax(logits, dim=1).item()
    return {"predicted_class": idx_to_class[pred_idx]}
