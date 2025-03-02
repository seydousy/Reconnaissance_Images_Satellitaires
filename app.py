from flask import Flask, render_template, request, redirect
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

app = Flask(__name__)

# Définition du modèle (SimpleCNN, à adapter selon ton entraînement)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)  # Pour des images de 64x64, après 2 fois pooling, taille = 16x16
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 32, 64, 64)
        x = self.pool(x)                     # (B, 32, 32, 32)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 64, 32, 32)
        x = self.pool(x)                     # (B, 64, 16, 16)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instanciation et chargement du modèle
num_classes = 10
model = SimpleCNN(num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("satellite_model.pth", map_location=device))
model.to(device)
model.eval()

# Transformation à appliquer sur l'image uploadée
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Classes du dataset EuroSAT
classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

classes_fr = {
    'AnnualCrop': 'Culture Annuelle',
    'Forest': 'Forêt',
    'HerbaceousVegetation': 'Végétation Herbacée',
    'Highway': 'Autoroute',
    'Industrial': 'Zone Industrielle',
    'Pasture': 'Pâturage',
    'PermanentCrop': 'Culture Permanente',
    'Residential': 'Zone Résidentielle',
    'River': 'Rivière',
    'SeaLake': 'Mer/Lac'
}

# Dossier pour stocker les images uploadées
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image)
            input_tensor = input_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                prediction = classes_fr[classes[pred_idx]] 
    return render_template("index.html", prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
