import io
import shutil
import sys
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import CLASS_NAMES, MODELS_DIR, PROJECT_ROOT
from data_utils import eval_transforms
from model import build_model, load_checkpoint, save_checkpoint
from train_utils import fedavg


MODEL_PATH = MODELS_DIR / "xray_global.pt"
STATIC_DIR = PROJECT_ROOT / "app" / "static"
FL_STORAGE_DIR = PROJECT_ROOT / "server_storage" / "fl_updates"

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


def reset_model_cache():
    global model
    model = None


def safe_name(value):
    cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)
    return cleaned.strip("_") or "hospital"


def load_update(path):
    update = torch.load(path, map_location="cpu")
    if "model_state" not in update or "sample_count" not in update:
        raise ValueError(f"Invalid update file: {path}")
    return update


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


@app.get("/admin")
def admin():
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "device": str(device),
    }


@app.get("/fl/status")
def fl_status(round_id: str = Query("current")):
    round_dir = FL_STORAGE_DIR / safe_name(round_id)
    update_files = sorted(round_dir.glob("*.pt")) if round_dir.exists() else []
    updates = []

    for update_file in update_files:
        try:
            update = load_update(update_file)
            updates.append(
                {
                    "file": update_file.name,
                    "hospital_name": update.get("hospital_name", "unknown"),
                    "sample_count": int(update["sample_count"]),
                }
            )
        except (ValueError, RuntimeError):
            updates.append({"file": update_file.name, "error": "Unreadable update"})

    return {
        "round_id": round_id,
        "pending_updates": len(update_files),
        "updates": updates,
    }


@app.get("/fl/model")
def download_global_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Model not found at {MODEL_PATH}")
    return FileResponse(MODEL_PATH, filename="xray_global.pt")


@app.post("/fl/submit-update")
async def submit_update(
    file: UploadFile = File(...),
    hospital_name: str = Form(...),
    sample_count: int = Form(...),
    round_id: str = Form("current"),
):
    if sample_count <= 0:
        raise HTTPException(status_code=400, detail="sample_count must be greater than zero.")

    round_dir = FL_STORAGE_DIR / safe_name(round_id)
    round_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    update_name = f"{safe_name(hospital_name)}_{timestamp}.pt"
    update_path = round_dir / update_name

    raw_bytes = await file.read()
    try:
        uploaded = torch.load(io.BytesIO(raw_bytes), map_location="cpu")
        model_state = uploaded.get("model_state", uploaded)
        if not isinstance(model_state, dict):
            raise ValueError("Uploaded file does not contain a model state dictionary.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded update is not a valid PyTorch weights file.") from exc

    torch.save(
        {
            "hospital_name": hospital_name,
            "sample_count": sample_count,
            "model_state": model_state,
            "received_at": timestamp,
        },
        update_path,
    )

    return {
        "status": "received",
        "round_id": round_id,
        "hospital_name": hospital_name,
        "sample_count": sample_count,
        "saved_as": update_name,
    }


@app.post("/fl/aggregate")
def aggregate_updates(round_id: str = Query("current"), clear_updates: bool = Query(True)):
    round_dir = FL_STORAGE_DIR / safe_name(round_id)
    update_files = sorted(round_dir.glob("*.pt")) if round_dir.exists() else []
    if not update_files:
        raise HTTPException(status_code=400, detail=f"No updates found for round '{round_id}'.")

    client_states = []
    sample_counts = []
    hospitals = []

    for update_file in update_files:
        try:
            update = load_update(update_file)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read update {update_file.name}.") from exc

        client_states.append(update["model_state"])
        sample_counts.append(int(update["sample_count"]))
        hospitals.append(update.get("hospital_name", update_file.stem))

    averaged_state = fedavg(client_states, sample_counts)
    global_model = build_model(pretrained=False)
    global_model.load_state_dict(averaged_state)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(global_model, str(MODEL_PATH))
    reset_model_cache()
    get_model()

    aggregated_dir = FL_STORAGE_DIR / "aggregated"
    aggregated_dir.mkdir(parents=True, exist_ok=True)
    archive_path = aggregated_dir / f"{safe_name(round_id)}_{int(time.time())}.pt"
    torch.save(
        {
            "round_id": round_id,
            "hospitals": hospitals,
            "sample_counts": sample_counts,
            "model_state": averaged_state,
            "aggregated_at": int(time.time()),
        },
        archive_path,
    )

    if clear_updates:
        shutil.rmtree(round_dir)

    return {
        "status": "aggregated",
        "round_id": round_id,
        "num_updates": len(update_files),
        "total_samples": sum(sample_counts),
        "hospitals": hospitals,
        "global_model": str(MODEL_PATH),
        "archive": str(archive_path),
        "cleared_updates": clear_updates,
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
