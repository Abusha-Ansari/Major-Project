import io
import shutil
import sys
import time
from pathlib import Path

import torch
from torch import nn
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import CLASS_NAMES, DEFAULT_BATCH_SIZE, MODELS_DIR, PROJECT_ROOT, TEST_DIR
from data_utils import eval_transforms, make_loader
from metrics_store import get_all_metrics, save_baseline_metrics, save_round_metrics
from model import build_model, load_checkpoint, save_checkpoint
from train_utils import evaluate, fedavg

from evaluate import run_evaluation, load_report
from xai import explain_image

from pydantic import BaseModel

from auth_store import (
    clear_round_state, get_round_state, has_submitted,
    list_hospitals, login_hospital, mark_submitted,
    register_hospital, validate_token,
)


class HospitalRegisterRequest(BaseModel):
    """Request body for hospital registration."""
    name: str
    password: str


class HospitalLoginRequest(BaseModel):
    """Request body for hospital login (name + password)."""
    name: str
    password: str


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


@app.get("/privacy")
def privacy():
    return FileResponse(STATIC_DIR / "privacy.html")


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
    request: Request,
    file: UploadFile = File(...),
    hospital_name: str = Form(...),
    sample_count: int = Form(...),
    round_id: str = Form("current"),
):
    if sample_count <= 0:
        raise HTTPException(status_code=400, detail="sample_count must be greater than zero.")

    # ── Token validation (Mandatory) ──
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Please register or login to submit updates.",
        )
    token = auth_header[7:].strip()
    if not validate_token(hospital_name, token):
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token for this hospital. Please re-login.",
        )

    # ── Duplicate submission prevention ────────────────────────────────────
    hospital_key = safe_name(hospital_name)
    if has_submitted(round_id, hospital_key):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Hospital '{hospital_name}' has already submitted weights for "
                f"round '{round_id}'. Wait for the admin to aggregate before re-submitting."
            ),
        )

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

    mark_submitted(round_id, hospital_key, hospital_name)

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

    # ── post-aggregation evaluation ────────────────────────────────────────
    eval_info: dict = {}
    if TEST_DIR and TEST_DIR.exists():
        try:
            active_model = get_model()
            test_loader, _ = make_loader(
                str(TEST_DIR), batch_size=DEFAULT_BATCH_SIZE, train=False
            )
            eval_metrics = evaluate(
                active_model, test_loader, nn.CrossEntropyLoss(), device
            )
            saved = save_round_metrics(
                round_id=round_id,
                accuracy=eval_metrics["accuracy"],
                loss=eval_metrics["loss"],
                num_hospitals=len(hospitals),
                total_samples=sum(sample_counts),
            )
            eval_info = {
                "metrics_saved": True,
                "accuracy": eval_metrics["accuracy"],
                "loss": eval_metrics["loss"],
                "round_num": saved["round_num"],
            }
        except Exception as exc:
            print(f"[metrics] Post-aggregation evaluation failed: {exc}")
            eval_info = {"metrics_saved": False, "error": str(exc)}
    else:
        eval_info = {
            "metrics_saved": False,
            "reason": "TEST_DIR not configured or does not exist",
        }

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
    clear_round_state(round_id)  # always reset submission tracking after aggregation

    return {
        "status": "aggregated",
        "round_id": round_id,
        "num_updates": len(update_files),
        "total_samples": sum(sample_counts),
        "hospitals": hospitals,
        "global_model": str(MODEL_PATH),
        "archive": str(archive_path),
        "cleared_updates": clear_updates,
        "evaluation": eval_info,
    }


@app.get("/fl/metrics")
def get_metrics():
    """Return all persisted training metrics (baseline + per-round history)."""
    return get_all_metrics()


@app.post("/fl/evaluate")
def trigger_evaluation(
    set_as_baseline: bool = Query(False),
    label: str = Query("current"),
):
    """
    Evaluate the current global model on the configured TEST_DIR.

    Returns a full report: accuracy, loss, precision, recall, F1,
    confusion matrix, and per-class breakdown.

    - ``set_as_baseline=true``  — also saves the result as the metrics baseline.
    - ``label``                 — human label stored in the report (e.g. "baseline").
    """
    if not TEST_DIR or not TEST_DIR.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Test directory not found: '{TEST_DIR}'. "
                "Set TEST_DIR in src/config.py to your test split folder."
            ),
        )

    try:
        active_model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    test_loader, _ = make_loader(
        str(TEST_DIR), batch_size=DEFAULT_BATCH_SIZE, train=False
    )

    report = run_evaluation(
        model=active_model,
        loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        class_names=CLASS_NAMES,
        label="baseline" if set_as_baseline else label,
        save=True,
    )

    if set_as_baseline:
        save_baseline_metrics(
            accuracy=report["accuracy"],
            loss=report["loss"],
            samples=report["samples"],
        )
        report["saved_as"] = "baseline"

    return report


@app.get("/fl/evaluation-report")
def get_evaluation_report():
    """Return the last saved evaluation report, or 404 if none exists."""
    report = load_report()
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="No evaluation report found. Run POST /fl/evaluate first.",
        )
    return report


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    include_xai: bool = Query(True, description="Whether to compute Grad-CAM heatmaps"),
    alpha: float = Query(0.5, description="Heatmap blend transparency (0.0 - 1.0)"),
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.") from exc

    try:
        active_model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if include_xai:
        result = explain_image(
            model=active_model,
            image_input=image,
            device=device,
            alpha=alpha,
        )
        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "gradcam_overlay": result["overlay_base64"],
            "gradcam_heatmap": result["heatmap_base64"],
        }

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
        "gradcam_overlay": None,
        "gradcam_heatmap": None,
    }


# ── Hospital portal ───────────────────────────────────────────────────────────

@app.get("/hospital")
def hospital_portal():
    """Serve the hospital web portal (register, login, check round status)."""
    html_path = STATIC_DIR / "hospital.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="hospital.html not found.")
    return FileResponse(html_path)


@app.post("/hospital/register")
def hospital_register(body: HospitalRegisterRequest):
    """Register a new hospital. Returns {status, name, key, token}."""
    name = body.name.strip()
    password = body.password.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Hospital name cannot be empty.")
    if not password:
        raise HTTPException(status_code=400, detail="Password cannot be empty.")
    try:
        entry = register_hospital(name, password)
        return {
            "status": "registered",
            "name": entry["name"],
            "key":  entry["key"],
            "token": entry["token"],
            "registered_at": entry["registered_at"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/hospital/login")
def hospital_login(body: HospitalLoginRequest):
    """Login with name + password. Returns {status, name, key, token} on success."""
    name = body.name.strip()
    password = body.password.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Hospital name cannot be empty.")
    if not password:
        raise HTTPException(status_code=400, detail="Password cannot be empty.")
    try:
        entry = login_hospital(name, password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return {
        "status": "ok",
        "name": entry["name"],
        "key":  entry["key"],
        "token": entry["token"],
        "registered_at": entry["registered_at"],
    }


@app.get("/hospital/list")
def hospital_list():
    """Return all registered hospitals (token omitted for privacy)."""
    return [
        {"name": h["name"], "key": h["key"], "registered_at": h["registered_at"]}
        for h in list_hospitals()
    ]


@app.get("/fl/round-state")
def round_state(round_id: str = Query("current")):
    """Return which hospitals have already submitted for the given round."""
    return {"round_id": round_id, **get_round_state(round_id)}
