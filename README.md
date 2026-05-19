# Animals Detection — Few-Shot Image Classification

## Project Description

This project implements a few-shot image classification system capable of identifying animal species from images using a transfer learning approach. Rather than training a deep neural network from scratch, we leverage MobileNetV2 — a lightweight convolutional neural network pre-trained on ImageNet (1.2 million images, 1000 classes) — as a frozen feature extractor. The model's classifier head is removed, and the remaining backbone produces 1280-dimensional embedding vectors that capture rich visual representations of any input image.

Classification is performed using a Prototypical Network approach: for each class, a prototype (centroid) is computed by averaging the embeddings of all available support images. At inference time, a query image is embedded and its class is determined by finding the nearest prototype using cosine similarity. This approach requires no gradient-based training and achieves strong performance even with a limited number of examples per class, making it a practical solution for few-shot learning scenarios.

The project covers the full machine learning pipeline: data loading, feature extraction, prototype computation, model evaluation, and deployment as a REST API served inside a Docker container.

## Dataset Description

The dataset used in this project is sourced from Kaggle: [Animals141](https://www.kaggle.com/datasets/sharansmenon/animals141). It contains images of animal species organized in individual folders, each named after the species using its Latin (scientific) name (e.g. `panthera-tigris`, `aquila-chrysaetos`). A `translation.json` file maps each Latin name to its common English name. The dataset covers approximately 150 animal species spanning mammals, birds, reptiles, fish, insects, and prehistoric animals, with around 30 to 50 images per class. Images are in JPEG format and vary in resolution. All images are resized to 224×224 pixels during preprocessing to match MobileNetV2's expected input format. The dataset is not included in this repository — download it from Kaggle and place it at `dataset/dataset/` before running the pipeline.

## Commands to Run

### Prerequisites
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install torch torchvision pillow scikit-learn numpy onnx onnxruntime matplotlib
```

### 1. Generate embeddings
```bash
python src/embeddings.py
```

### 2. Compute class prototypes
```bash
python src/classifier.py
```

### 3. Export model to ONNX (for Docker)
```bash
python3 -c "
import torch
from torchvision import models
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.features.eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model.features, dummy, 'API_App/model.onnx',
    input_names=['input'], output_names=['output'], opset_version=11)
print('ONNX export done')
"
```

### 4. Build and run Docker
```bash
cp prototypes.pkl API_App/prototypes.pkl
docker build -t animals-detection ./API_App
docker run -p 8000:8000 animals-detection
```

### 5. Use the web interface
Open [http://localhost:8000](http://localhost:8000) in your browser and upload any animal image.

### 6. Use the API directly
```bash
curl -X POST http://localhost:8000/predict -F "file=@your_image.jpg"
```

### 7. Visualize predictions locally
```bash
python src/visualize_predictions.py dataset/dataset/aquila-chrysaetos/aquila-chrysaetos_5_291f193f.jpg
```

## Project Results

The prototypical network with MobileNetV2 embeddings achieves strong top-1 accuracy on the support set across all 150 classes. Because the backbone was pre-trained on ImageNet — which includes a large variety of animal species — the extracted features are highly discriminative even without any fine-tuning. Cosine similarity between embeddings proves effective at separating species that are visually distinct (e.g. fish vs. birds), while also correctly grouping visually similar species (e.g. different big cats) into nearby regions of the embedding space.

Three strategies were evaluated and compared in `summary.ipynb`: (1) a baseline k-nearest neighbours classifier on raw pixel values, which performs poorly due to sensitivity to background and lighting; (2) k-nearest neighbours on MobileNetV2 embeddings, which improves significantly by operating in a semantically meaningful feature space; and (3) the final prototypical network with cosine similarity, which outperforms both baselines by computing stable per-class centroids that are robust to intra-class variation. The prototypical approach also scales efficiently to 150 classes with no retraining required when new classes are added — only new embeddings and a prototype recomputation are needed.

The Docker image is kept lightweight (~400MB) by using ONNX Runtime instead of PyTorch at inference time, avoiding the ~2GB PyTorch dependency in the container. The REST API exposes a `/predict` endpoint returning the top-5 predicted classes with confidence scores, and a `/health` endpoint listing all available classes.

## Repository Structure