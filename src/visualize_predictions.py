import pickle
import numpy as np
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

MODEL_PATH = "API_App/model.onnx"
PROTOTYPES_PATH = "API_App/prototypes.pkl"

session = ort.InferenceSession(MODEL_PATH)
with open(PROTOTYPES_PATH, "rb") as f:
    prototypes = pickle.load(f)

def preprocess(image):
    image = image.resize((224, 224))
    img = np.array(image).astype(np.float32) / 255.0
    img = (img - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
    return img.transpose(2,0,1)[np.newaxis].astype(np.float32)

def predict_top5(image_path):
    image = Image.open(image_path).convert("RGB")
    inp = preprocess(image)
    output = session.run(None, {"input": inp})[0]
    embedding = output.mean(axis=(2,3)).flatten()
    labels = list(prototypes.keys())
    proto_matrix = np.array([prototypes[l] for l in labels])
    sims = cosine_similarity([embedding], proto_matrix)[0]
    top5_idx = np.argsort(sims)[::-1][:5]
    return image, [(labels[i], float(sims[i])) for i in top5_idx]

def visualize(image_path, true_label=None):
    image, top5 = predict_top5(image_path)
    true_label = true_label or os.path.basename(os.path.dirname(image_path))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Few-shot classification · MobileNetV2 backbone (3.4M params) · "
        f"{len(prototypes)} classes · cosine similarity on 1280-dim embeddings",
        fontsize=9, color='gray'
    )

    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(f"True class: {true_label}", fontsize=11)

    labels = [p[0] for p in top5]
    scores = [p[1]*100 for p in top5]
    colors = ['#1D9E75' if l == true_label else '#378ADD' for l in labels]

    bars = ax2.barh(range(4,-1,-1), scores, color=colors, height=0.6)
    ax2.set_yticks(range(4,-1,-1))
    ax2.set_yticklabels([f"{i+1}. {l}" for i,l in enumerate(labels)], fontsize=10)
    ax2.set_xlabel("Confidence (%)")
    ax2.set_xlim(0, 100)
    ax2.set_title("Top 5 predictions", fontsize=11)
    for bar, score in zip(bars, scores):
        ax2.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                 f"{score:.1f}%", va='center', fontsize=9)

    correct = mpatches.Patch(color='#1D9E75', label='Correct class')
    other   = mpatches.Patch(color='#378ADD', label='Other prediction')
    ax2.legend(handles=[correct, other], fontsize=9)

    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nTop 5 predictions for '{true_label}':")
    for i, (label, score) in enumerate(top5):
        marker = "✓" if label == true_label else " "
        print(f"  {marker} {i+1}. {label}: {score*100:.1f}%")

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else \
        "dataset/dataset/aquila-chrysaetos/aquila-chrysaetos_5_291f193f.jpg"
    visualize(image_path)
