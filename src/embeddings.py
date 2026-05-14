import os
import pickle
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision import models

# =========================
# CONFIG
# =========================

DATASET_PATH = "dataset/dataset"
OUTPUT_FILE = "embeddings.pkl"

# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MODEL
# =========================

# Load pretrained MobileNetV2
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Remove classifier head
model = model.features

# Evaluation mode
model.eval()
model.to(device)

# =========================
# IMAGE TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# STORAGE
# =========================

data = []

# =========================
# LOOP OVER DATASET
# =========================
valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
for label in sorted(os.listdir(DATASET_PATH)):
    class_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(class_path):
        continue

    count = 0
    for filename in sorted(os.listdir(class_path)):
        image_path = os.path.join(class_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        if not os.path.isfile(image_path) or ext not in valid_extensions:
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model(img_tensor)
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                embedding = features.view(-1).cpu().numpy()
            data.append({"label": label, "image": image_path, "embedding": embedding})
            count += 1
        except Exception as e:
            print(f"Error with {image_path}: {e}")

    print(f"[{label}] {count} images processed")
# =========================
# SAVE
# =========================

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(data, f)

print(f"\nSaved embeddings to {OUTPUT_FILE}")