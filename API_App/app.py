import pickle, json
import numpy as np
from PIL import Image
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import io

app = FastAPI()

session = ort.InferenceSession("model.onnx")
with open("prototypes.pkl", "rb") as f:
    prototypes = pickle.load(f)
with open("translation.json", "r") as f:
    translation = json.load(f)

def preprocess(image):
    image = image.resize((224, 224))
    img = np.array(image).astype(np.float32) / 255.0
    img = (img - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
    return img.transpose(2,0,1)[np.newaxis].astype(np.float32)

def get_top5(image):
    inp = preprocess(image)
    output = session.run(None, {"input": inp})[0]
    embedding = output.mean(axis=(2,3)).flatten()
    labels = list(prototypes.keys())
    proto_matrix = np.array([prototypes[l] for l in labels])
    sims = cosine_similarity([embedding], proto_matrix)[0]
    top5_idx = np.argsort(sims)[::-1][:5]
    return [(labels[i], float(sims[i])) for i in top5_idx]

@app.get("/", response_class=HTMLResponse)
def index():
    return open("index.html").read()

@app.get("/health")
def health():
    return {"status": "ok", "classes": len(prototypes)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    top5 = get_top5(image)
    return {"predictions": [
        {
            "class": l,
            "common_name": translation.get(l, l),
            "confidence": round(s, 4)
        } for l, s in top5
    ]}
