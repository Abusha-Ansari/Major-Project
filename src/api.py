import io
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import CLASS_NAMES, MODELS_DIR, PROJECT_ROOT
from data_utils import eval_transforms
from model import build_model, load_checkpoint


MODEL_PATH = MODELS_DIR / "xray_global.pt"
STATIC_DIR = PROJECT_ROOT / "app" / "static"

app = FastAPI(title="X-ray Pneumonia Detection API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None


def get_model():
    global model
    if model is not None:
        return model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    loaded_model = build_model(pretrained=False).to(device)
    load_checkpoint(loaded_model, str(MODEL_PATH), device)
    loaded_model.eval()
    model = loaded_model
    return model


@app.on_event("startup")
def load_model_on_startup():
    try:
        get_model()
        print(f"Loaded model from {MODEL_PATH}")
    except FileNotFoundError as exc:
        print(exc)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "device": str(device),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.") from exc

    try:
        active_model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    tensor = eval_transforms()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = active_model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()
        predicted_index = int(probabilities.argmax().item())

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index].item()),
        "probabilities": [
            {
                "class_name": class_name,
                "probability": float(probability),
            }
            for class_name, probability in zip(CLASS_NAMES, probabilities.tolist())
        ],
    }
