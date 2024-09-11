import os
from typing import List

from model import ConvNet
from PIL import Image
from torchvision import datasets, transforms

import torch
from torch.utils.data import DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_dir(folder: str, model, map_classes: List):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isdir(file_path):
            predict_dir(file_path, model, map_classes)
        elif not file.startswith("."):
            print(f"Predicting {file_path}")
            image = Image.open(file_path)
            image = transform(image)
            image = image.unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                outputs = torch.softmax(model(image.to(device)), dim=1)
                value, predicted = torch.max(outputs, 1)
                predicted_class = predicted.item()
                predicted_score = value.item()
                print(
                    f"Class: {map_classes[predicted_class]}\nScore: {predicted_score*100:.2f}%\n"
                )


DATA_FOLDER = "../data/birds/valid"
MODEL_PATH = "../models/birds-121.pth"
model_input_size = 224

# Load data
transform = transforms.Compose(
    [
        transforms.Resize((model_input_size, model_input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize with ImageNet standards
    ]
)
dataset = datasets.ImageFolder(root=DATA_FOLDER, transform=transform)
classes = [
    [key for key, value in dataset.class_to_idx.items() if value == i][0]
    for i in range(len(dataset.class_to_idx))
]
print(f"Dataset: Number of classes: {len(dataset.classes)}")
print(f"Dataset: Classes: \n{classes}")
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

# Loading model
model = ConvNet(len(dataset.classes)).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Prediction
predict_dir(DATA_FOLDER, model, classes)
