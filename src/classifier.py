import pickle
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# LOAD EMBEDDINGS
# =========================

with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

# =========================
# GROUP EMBEDDINGS BY LABEL
# =========================

class_embeddings = defaultdict(list)

for item in data:
    label = item["label"]
    embedding = item["embedding"]

    class_embeddings[label].append(embedding)

# =========================
# COMPUTE PROTOTYPES
# =========================

prototypes = {}

for label, embeddings in class_embeddings.items():

    embeddings = np.array(embeddings)

    # Mean vector
    centroid = embeddings.mean(axis=0)

    prototypes[label] = centroid

    print(f"Prototype created for class: {label}")

# =========================
# SAVE PROTOTYPES
# =========================

with open("prototypes.pkl", "wb") as f:
    pickle.dump(prototypes, f)

print("\nSaved prototypes to prototypes.pkl")


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict(embedding, prototypes):

    best_label = None
    best_score = -1

    for label, centroid in prototypes.items():

        score = cosine_similarity(
            [embedding],
            [centroid]
        )[0][0]

        print(f"{label}: cosine similarity = {score:.4f}")

        if score > best_score:
            best_score = score
            best_label = label

    return best_label, best_score