# Federated Learning Pneumonia Detection — Full Analysis & Implementation Plan

---

## Phase 1: Complete Project Analysis

---

### 1. Project Overview

**Purpose:**  
This project implements a **Federated Learning (FL) system for detecting pneumonia from chest X-ray images**. It enables multiple hospitals to collaboratively train a shared deep learning model without ever sharing their private patient data. Only trained model weights are exchanged.

**Problem Solved:**  
In healthcare, patient data is sensitive and legally protected (HIPAA, GDPR). Traditional centralized ML requires all data to be pooled on one server, which is impractical and unethical for hospitals. This system solves that by keeping images local while still benefiting from collective learning.

**Federated Learning Implementation:**
- Each hospital trains a local copy of the global model on its own X-ray data.
- Only the resulting model weight tensors (`.pt` files) are sent to the central server.
- The server performs **FedAvg** (Federated Averaging) — a weighted average of all hospital weights proportional to sample counts.
- The aggregated global model is then distributed back to hospitals for the next round.

**Complete Training Lifecycle:**
1. A global model (ResNet18) is initialized or optionally pre-trained on a small seed dataset.
2. Hospitals download the current global model.
3. Each hospital trains locally for N epochs on its own X-ray images.
4. Each hospital submits weight updates (`.pt` file) to the central server.
5. Once enough hospitals submit, the admin triggers aggregation.
6. The server runs FedAvg → saves new `xray_global.pt`.
7. This cycle repeats for multiple federated rounds.

---

### 2. Project Structure

```
Major Project/
├── app/
│   └── static/
│       ├── index.html          ← End-user X-ray prediction page
│       └── admin.html          ← Admin federated learning dashboard
├── client_work/
│   └── Hospital_1/
│       ├── global_model.pt     ← Downloaded global model per hospital
│       └── local_update.pt     ← Trained local weights per hospital
├── data/
│   └── hospitals_xray/
│       ├── hospital_1/         ← NORMAL/ + PNEUMONIA/ images
│       ├── hospital_2/
│       └── hospital_3/
├── models/
│   └── xray_global.pt          ← Current global model (~42MB ResNet18)
├── scripts/
│   ├── prepare_xray_dataset.py ← Extracts ZIP, normalizes folder structure
│   └── split_hospitals.py      ← Round-robin splits training data into hospital folders
├── server_storage/
│   └── fl_updates/
│       ├── current/            ← Pending weight files from hospitals
│       └── aggregated/         ← Archived post-aggregation models
├── src/
│   ├── __init__.py
│   ├── api.py                  ← FastAPI server (REST API + static pages)
│   ├── config.py               ← Project-wide constants (paths, class names, hyperparams)
│   ├── data_utils.py           ← Image transforms + DataLoader factory
│   ├── federated_train.py      ← Local-folder simulation of full FL rounds
│   ├── hospital_client.py      ← CLI + library for hospital training+submission
│   ├── hospital_desktop.py     ← Tkinter GUI for hospital operators
│   ├── model.py                ← ResNet18 model builder + checkpoint I/O
│   ├── predict.py              ← CLI single-image inference tool
│   ├── train_initial.py        ← Optional central seed training script
│   └── train_utils.py          ← Training loop, evaluation, FedAvg algorithm
├── .venv/                      ← Virtual environment
├── requirements.txt
├── README.md
└── overview.txt
```

---

### 3. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.x | All components |
| ML Framework | PyTorch + TorchVision | Model training, inference, weight I/O |
| Model Architecture | ResNet18 (ImageNet pretrained) | Transfer learning for X-ray classification |
| Backend API | FastAPI + Uvicorn | REST API, static file serving |
| Frontend | Vanilla HTML/CSS/JS | Web prediction page + admin dashboard |
| Desktop GUI | Tkinter + ttk | Hospital operator client application |
| Data Transforms | TorchVision transforms | Image preprocessing, augmentation |
| Dataset Format | ImageFolder (NORMAL/PNEUMONIA) | Standard PyTorch dataset format |
| HTTP Client | requests | Hospital → Server weight submission |
| Evaluation Tools | scikit-learn, matplotlib | Present in requirements, unused in code |
| Progress Display | tqdm | Training loop progress bars |
| Build/Deploy | uvicorn CLI | ASGI server startup |
| Dependency Mgmt | pip + requirements.txt | No lock file |

---

### 4. Workflow Analysis

```
[SETUP PHASE]
    ↓
1. Admin extracts dataset ZIP via prepare_xray_dataset.py
    → Creates data/xray_raw/train/ and data/xray_raw/test/
    ↓
2. Admin splits training data into hospital partitions via split_hospitals.py
    → Creates data/hospitals_xray/hospital_{1,2,3}/NORMAL/ + PNEUMONIA/
    ↓
3. (Optional) Admin runs train_initial.py to create a seed model
    → Trains ResNet18 on seed data for N epochs
    → Saves models/xray_global.pt
    ↓
[SERVER PHASE]
4. Admin starts FastAPI server: uvicorn src.api:app
    → Loads xray_global.pt into memory at startup
    → Serves http://127.0.0.1:8000 (prediction) and /admin (dashboard)
    ↓
[HOSPITAL TRAINING PHASE — repeated per round]
5. Hospital operator opens hospital_desktop.py (GUI) or runs hospital_client.py (CLI)
    a. Downloads global model from GET /fl/model
    b. Trains local copy for N local epochs on NORMAL/PNEUMONIA images
    c. Saves local_update.pt with {model_state, sample_count, class_names}
    d. POSTs local_update.pt to POST /fl/submit-update
    ↓
[AGGREGATION PHASE]
6. Admin views pending updates at /admin dashboard
    → Clicks "Aggregate Updates" → POST /fl/aggregate
    → Server loads all .pt files from server_storage/fl_updates/current/
    → Runs FedAvg (weighted average by sample count)
    → Saves new xray_global.pt
    → Archives result to server_storage/fl_updates/aggregated/
    → Clears current round's pending files
    ↓
[INFERENCE PHASE]
7. End user visits http://127.0.0.1:8000
    → Uploads chest X-ray image
    → POST /predict → returns NORMAL/PNEUMONIA + confidence + probabilities
```

---

### 5. Frontend Analysis

#### `/` → `index.html` — End-User Prediction Page

| Aspect | Details |
|---|---|
| Purpose | Allows end users to upload an X-ray and get a NORMAL/PNEUMONIA prediction |
| Components | Dropzone image preview, Choose Image button, Predict button, result aside panel, probability bars |
| State | `selectedFile` JS var, DOM element refs |
| API Calls | `GET /health` (on load), `POST /predict` (on predict click) |
| UI Limitations | No dark mode, no drag-and-drop, no batch prediction, no model version display, no history |

#### `/admin` → `admin.html` — Admin Federated Dashboard

| Aspect | Details |
|---|---|
| Purpose | Admin view of federated rounds — see pending hospital updates, trigger aggregation |
| Components | 3-metric grid (model status, pending updates, device), round ID input, hospital updates table, aggregate button |
| State | None (pure DOM manipulation) |
| API Calls | `GET /health`, `GET /fl/status?round_id=`, `POST /fl/aggregate?round_id=` |
| UI Limitations | No training metrics charts, no per-round history, no hospital authentication, no confusion matrix, no model comparison, no security/privacy explainer, no real-time progress |

---

### 6. Backend Analysis

#### APIs (all in `src/api.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves `index.html` |
| `/admin` | GET | Serves `admin.html` |
| `/health` | GET | Returns model loaded status, path, device |
| `/fl/status` | GET | Lists pending updates for a given round_id |
| `/fl/model` | GET | Downloads current global model as `.pt` |
| `/fl/submit-update` | POST | Accepts hospital weight file + metadata |
| `/fl/aggregate` | POST | Runs FedAvg, saves global model, archives result |
| `/predict` | POST | Inference on uploaded image, returns class + confidence |

#### Services / Logic

- **`get_model()` / `reset_model_cache()`**: Lazy singleton model loader with cache invalidation after aggregation.
- **`safe_name()`**: Sanitizes hospital names to filesystem-safe strings.
- **`load_update()`**: Validates `.pt` files have required keys.
- **FedAvg in `aggregate_updates()`**: Delegates to `train_utils.fedavg()`.

#### Federated Aggregation Logic (`train_utils.py → fedavg()`)

- Iterates all tensor keys in state_dicts.
- For integer tensors (BatchNorm running stats): uses first client's value (not averaged — this is a known limitation).
- For float tensors: computes weighted sum `Σ(weight_i × count_i / total_count)`.

#### Authentication
- **None currently.** Any client can submit any name, any round ID.

---

### 7. Machine Learning Pipeline

#### Dataset Loading
- **Format**: ImageFolder (subfolders = class names: NORMAL, PNEUMONIA)
- **Factory**: `make_loader(data_dir, batch_size, train)` in `data_utils.py`

#### Data Preprocessing
- **Train**: Resize(224×224) → RandomHorizontalFlip → RandomRotation(8°) → ToTensor → Normalize(ImageNet stats)
- **Eval**: Resize(224×224) → ToTensor → Normalize(ImageNet stats)

#### Model Architecture
- **Base**: ResNet18 (11M parameters, ~42MB)
- **Modification**: Final FC layer replaced: `Linear(512, 2)` (binary: NORMAL=0, PNEUMONIA=1)
- **Initialization**: ImageNet pretrained weights → transfer learning

#### Loss Function
- `nn.CrossEntropyLoss` — standard multi-class (used for binary here)

#### Optimizer
- `torch.optim.Adam` with lr=1e-4 (default)

#### Training Loop (`train_one_epoch`)
- Mini-batch SGD, loss.backward(), optimizer.step()
- Returns per-epoch: loss, accuracy, samples

#### Validation (`evaluate`)
- `@torch.no_grad()` context
- Returns: loss, accuracy, samples

#### Weight Saving (`save_checkpoint`)
- Saves `{model_state, class_names}` dict as `.pt` file

#### Weight Aggregation (FedAvg)
- Weighted average of state_dicts proportional to sample_count
- Note: Integer tensors (BN running mean/var) take first client's value only

#### Evaluation Metrics
- Only accuracy and loss currently tracked
- scikit-learn is in requirements but **never imported or used in any source file**
- matplotlib is in requirements but **never imported or used in any source file**
- No confusion matrix, precision, recall, F1-score implemented anywhere

---

### 8. Current Feature List

| Feature | Status | Files | Limitations |
|---|---|---|---|
| ResNet18 model for X-ray classification | ✅ Done | `model.py`, `config.py` | Binary only, no multi-class |
| Initial seed training script | ✅ Done | `train_initial.py` | No metric persistence |
| Local-folder FL simulation (no server) | ✅ Done | `federated_train.py` | Prints metrics, does not save them |
| FastAPI central server | ✅ Done | `api.py` | No auth, no metrics storage |
| Hospital CLI client | ✅ Done | `hospital_client.py` | No duplicate prevention |
| Hospital Tkinter desktop GUI | ✅ Done | `hospital_desktop.py` | No progress bar, no epoch display |
| Weight submission via HTTP | ✅ Done | `hospital_client.py`, `api.py` | No identity verification |
| FedAvg aggregation | ✅ Done | `train_utils.py`, `api.py` | BN stats from only 1st client |
| Web prediction page | ✅ Done | `index.html` | No drag-drop, no history |
| Admin dashboard | ✅ Done | `admin.html` | No charts, no auth, no metrics |
| Dataset preparation scripts | ✅ Done | `scripts/` | Works for xray only |
| X-ray prediction API | ✅ Done | `api.py → /predict` | Single image only |
| Model download by hospitals | ✅ Done | `api.py → /fl/model` | No versioning |
| Training metrics visualization | ❌ Missing | — | Not implemented |
| Hospital authentication | ❌ Missing | — | Not implemented |
| Duplicate submission prevention | ❌ Missing | — | Not implemented |
| Local training progress display | ❌ Missing | — | Not implemented |
| Confusion matrix | ❌ Missing | — | Not implemented |
| Precision/Recall/F1 | ❌ Missing | — | Not implemented |
| Security/privacy explainer page | ❌ Missing | — | Not implemented |
| CT scan pipeline | ❌ Missing | — | Planned per README |

---

### 9. Existing Problems

#### Missing Functionality
- No training metrics persistence (accuracy/loss per round not saved anywhere)
- No confusion matrix or classification report
- No hospital authentication or registration
- No duplicate submission prevention per round
- No real-time training progress (epoch/batch/loss) in desktop app
- CT scan pipeline planned but absent
- No security/privacy explainer UI
- scikit-learn and matplotlib in requirements but never used

#### Bugs
- **FedAvg BN integer tensor bug**: Integer tensors (BatchNorm running_mean, running_var) only take the first client's values — other hospitals' statistics are silently dropped
- **No server-side round tracking**: `round_id` is just a string directory name; there's no round counter or state machine
- **Model cache not thread-safe**: `reset_model_cache()` + `get_model()` called sequentially without locking in `aggregate_updates()`
- **`on_event("startup")` deprecated**: FastAPI deprecated `@app.on_event("startup")` in favor of `lifespan` context manager

#### Code Duplication
- `safe_name()` logic duplicated in `api.py` and `hospital_client.py`
- CLASS_NAMES hardcoded in `scripts/split_hospitals.py` (not imported from config)

#### Performance Bottlenecks
- `DEFAULT_NUM_WORKERS=0` means no parallel data loading (CPU bottleneck)
- No DataLoader pinning (`pin_memory=True`) for GPU transfer

#### Poor UI/UX
- Admin dashboard has no charts or visual feedback for training progress
- Hospital desktop has no progress bar, no real-time epoch/batch display
- No model performance comparison (initial vs federated)
- Both HTML pages use Arial — no Google Fonts or modern typography
- No dark mode toggle
- No responsive design for admin table

#### Technical Debt
- No database; state is entirely file-based
- No authentication layer at all
- No input validation beyond basic checks
- No logging framework (print statements only)
- No unit tests or test suite
- No Dockerfile or deployment config

#### Scalability Concerns
- File-based round state won't scale beyond local demo
- Single global model file — concurrent writes during aggregation could corrupt it
- No async training trigger (aggregation is synchronous HTTP call)

---

### 10. Code Dependency Map

```
config.py
    ← model.py
    ← data_utils.py
    ← train_utils.py (indirect via model.py)
    ← train_initial.py
    ← federated_train.py
    ← hospital_client.py
    ← api.py

model.py
    ← train_initial.py
    ← federated_train.py
    ← hospital_client.py
    ← api.py

data_utils.py
    ← train_initial.py
    ← federated_train.py
    ← hospital_client.py
    ← api.py (eval_transforms only)

train_utils.py
    ← train_initial.py
    ← federated_train.py
    ← hospital_client.py
    ← api.py (fedavg only)

hospital_client.py
    ← hospital_desktop.py

api.py
    [Serves] index.html → POST /predict, GET /health
    [Serves] admin.html → GET /fl/status, GET /health, POST /fl/aggregate
```

---

---

## Phase 2: Task Analysis

---

### Task 1 — Training Metrics

**Current Status:** Metrics computed but immediately discarded (printed to console). No persistence, no visualization.

| Item | Details |
|---|---|
| Files to Modify | `src/api.py`, `src/train_utils.py`, `app/static/admin.html` |
| New Files | `src/metrics_store.py` (JSON persistence), potentially `app/static/metrics.html` |
| Dependencies | Chart.js (CDN), existing `train_utils.py` |
| Risks | Backward compatibility of stored JSON format; concurrent write during aggregation |
| Complexity | Medium |
| Approach | Store per-round metrics in `server_storage/metrics.json`. Expose via `GET /fl/metrics`. Render line charts in admin page using Chart.js CDN. |

---

### Task 2 — Better Hospital Workflow

**Current Status:** No auth, no identity verification, no progress display, no duplicate prevention.

| Item | Details |
|---|---|
| Files to Modify | `src/api.py`, `src/hospital_client.py`, `src/hospital_desktop.py` |
| New Files | `src/auth_store.py` (hospital registry JSON), `app/static/hospital.html` (web login) |
| Dependencies | None beyond existing stack |
| Risks | File-based auth is not cryptographically secure; progress callbacks require thread-safe queue |
| Complexity | Medium-High |
| Approach | JSON file as hospital registry with name→token mapping. Track submitted hospitals per round in `server_storage/round_state.json`. Desktop app progress via callback hooks in `train_one_epoch`. |

---

### Task 3 — Validation & Evaluation Report

**Current Status:** scikit-learn in requirements but never used. No confusion matrix, no F1/Precision/Recall.

| Item | Details |
|---|---|
| Files to Modify | `src/api.py`, `src/train_utils.py` or new `src/evaluate.py`, `app/static/admin.html` |
| New Files | `src/evaluate.py`, `server_storage/evaluation_report.json` |
| Dependencies | scikit-learn (already in requirements), matplotlib (already in requirements) |
| Risks | Evaluation requires a test dataset accessible from the server — currently only global model is on server |
| Complexity | High |
| Approach | Add `POST /fl/evaluate` endpoint that runs evaluation on stored test set, computes sklearn metrics, saves report JSON. Render in new admin tab with confusion matrix heatmap via Canvas API. |

---

### Task 4 — Security & Privacy Explanation

**Current Status:** No explainer page. Privacy is inherent in the architecture but not visualized.

| Item | Details |
|---|---|
| Files to Modify | `src/api.py` (add route), `app/static/admin.html` (add nav link) |
| New Files | `app/static/privacy.html` |
| Dependencies | None |
| Risks | Low — purely informational page |
| Complexity | Low |
| Approach | Dedicated HTML page with animated SVG/CSS workflow diagram, sections for each privacy property, and a roadmap for Differential Privacy. |

---

### Task 5 — CT Scan Pipeline

**Current Status:** Mentioned in README and overview as Phase 5. Completely absent.

| Item | Details |
|---|---|
| Files to Modify | `src/api.py` (mount CT routes or create separate app) |
| New Files | `src/ct/config_ct.py`, `src/ct/model_ct.py`, `src/ct/data_utils_ct.py`, `src/ct/train_utils_ct.py`, `src/ct/api_ct.py`, `src/ct/hospital_client_ct.py`, `app/static/ct_admin.html`, `scripts/prepare_ct_dataset.py`, `scripts/split_ct_hospitals.py` |
| Dependencies | Same PyTorch stack; different dataset format (CT volumes may require 3D preprocessing) |
| Risks | CT scan datasets vary widely in format (DICOM vs PNG/JPEG slices). Architecture must be decided based on dataset format. |
| Complexity | Very High |
| Approach | Mirror the entire MRI/X-ray pipeline structure under `src/ct/` prefix. Use a separate model endpoint prefix `/ct/`. Keep shared utilities (FedAvg, base model builder) reusable. |

---

---

## Phase 3: Implementation Roadmap

---

### Implementation Phase 1 — Foundation & Metrics (Task 1)

**Objective:** Persist and visualize training metrics per federated round.

#### Files to Modify
- [MODIFY] [api.py](file:///c:/Users/Acer/Desktop/Major%20Project/src/api.py) — Add `/fl/metrics` GET endpoint, update aggregate to save metrics
- [MODIFY] [train_utils.py](file:///c:/Users/Acer/Desktop/Major%20Project/src/train_utils.py) — Return richer history from `train_model`, add evaluate-with-collect
- [MODIFY] [admin.html](file:///c:/Users/Acer/Desktop/Major%20Project/app/static/admin.html) — Add Chart.js charts for loss/accuracy per round

#### Files to Create
- [NEW] `src/metrics_store.py` — JSON read/write for metrics persistence
- [NEW] `server_storage/metrics.json` — Created on first aggregation

#### Changes
1. **`metrics_store.py`**: `load_metrics()`, `save_round_metrics(round_id, data)`, `get_all_metrics()` using atomic JSON write.
2. **`api.py → aggregate_updates()`**: After FedAvg, compute test-set evaluation (requires admin to pass test dir or use a stored test set path in config). Save `{round_id, accuracy, loss, hospitals, samples, timestamp}` to `metrics.json`.
3. **`GET /fl/metrics`**: Returns all rounds' metrics as JSON.
4. **`admin.html`**: Add Chart.js (CDN) panel with dual-axis line chart (accuracy + loss per round), legend distinguishing initial vs federated rounds.

#### Validation
- Run 2 FL rounds → verify `metrics.json` updates
- Open admin page → verify charts render with correct data

---

### Implementation Phase 2 — Hospital Workflow (Task 2)

**Objective:** Hospital auth, real-time progress, duplicate prevention.

#### Files to Modify
- [MODIFY] [api.py](file:///c:/Users/Acer/Desktop/Major%20Project/src/api.py) — Add `/hospital/register`, `/hospital/login`, `/fl/round-state` endpoints; modify `submit_update` to check duplicates
- [MODIFY] [hospital_client.py](file:///c:/Users/Acer/Desktop/Major%20Project/src/hospital_client.py) — Add auth token passing; add progress callbacks to training
- [MODIFY] [hospital_desktop.py](file:///c:/Users/Acer/Desktop/Major%20Project/src/hospital_desktop.py) — Add progress bar widget, epoch/batch labels, auth fields

#### Files to Create
- [NEW] `src/auth_store.py` — Hospital registry (name → token, registered_at)
- [NEW] `server_storage/hospitals.json` — Hospital registry
- [NEW] `server_storage/round_state.json` — Tracks which hospitals submitted per round
- [NEW] `app/static/hospital.html` — Browser-based hospital login/register page

#### Changes
1. **`auth_store.py`**: Register, lookup, validate hospital identity. Use UUID tokens.
2. **`api.py`**:
   - `POST /hospital/register` — register name, return token
   - `POST /hospital/login` — returns existing token
   - Modified `submit_update` — check token header, check if already submitted this round
3. **`hospital_client.py`**: Accept `--token` argument; pass as `Authorization` header.
4. **`hospital_desktop.py`**: Add ttk.Progressbar, `StringVar` for epoch/batch labels; hook into `train_one_epoch` via callback.

#### Validation
- Register same hospital twice → expect 409 Conflict
- Submit twice in same round → expect 409 Conflict with clear UI message
- Training progress visible in desktop app

---

### Implementation Phase 3 — Evaluation Reports (Task 3)

**Objective:** Confusion matrix, precision, recall, F1 — initial vs federated comparison.

#### Files to Modify
- [MODIFY] [api.py](file:///c:/Users/Acer/Desktop/Major%20Project/src/api.py) — Add `POST /fl/evaluate` endpoint
- [MODIFY] [admin.html](file:///c:/Users/Acer/Desktop/Major%20Project/app/static/admin.html) — Add evaluation report tab

#### Files to Create
- [NEW] `src/evaluate.py` — Full evaluation pipeline using sklearn
- [NEW] `server_storage/evaluation_report.json` — Stores latest evaluation results

#### Changes
1. **`evaluate.py`**:
   - `run_evaluation(model, test_loader, device)` → returns `{accuracy, precision, recall, f1, confusion_matrix, class_report}`
   - Uses `sklearn.metrics.classification_report`, `confusion_matrix`
2. **`api.py → POST /fl/evaluate`**: Loads test data from configured path, runs evaluation, saves report.
3. **`GET /fl/evaluation-report`**: Returns saved report JSON.
4. **`admin.html`**: New "Evaluation" tab section:
   - Confusion matrix heatmap (Canvas API or Chart.js matrix plugin)
   - Summary table (accuracy, precision, recall, F1)
   - Comparison table: Initial model vs Current federated model
   - Percentage improvement badges

#### Validation
- Trigger evaluation via API → verify JSON saved
- Admin page renders confusion matrix correctly
- Comparison row shows improvement percentages

---

### Implementation Phase 4 — Security & Privacy Page (Task 4)

**Objective:** Visual, informational page explaining the FL privacy model.

#### Files to Create
- [NEW] `app/static/privacy.html` — Full explainer page

#### Files to Modify
- [MODIFY] [api.py](file:///c:/Users/Acer/Desktop/Major%20Project/src/api.py) — Add `GET /privacy` route
- [MODIFY] [index.html](file:///c:/Users/Acer/Desktop/Major%20Project/app/static/index.html) — Add nav link to privacy page
- [MODIFY] [admin.html](file:///c:/Users/Acer/Desktop/Major%20Project/app/static/admin.html) — Add nav link

#### Changes
1. Animated CSS/SVG workflow diagram showing:
   - Hospital icon → local training → weight icon → server icon → aggregated model icon
   - Explicit "No Patient Data Leaves" callout
2. Sections: What is FL? | What stays local? | What is transmitted? | Server's role | Differential Privacy Roadmap
3. Differential Privacy design box (future): noise addition → ε budget → per-layer clipping → planned integration points

#### Validation
- Page renders correctly
- All nav links work
- Diagram animates properly

---

### Implementation Phase 5 — CT Scan Pipeline (Task 5)

**Objective:** Mirror full pipeline for CT scan data, completely separate from X-ray.

#### New Folder Structure
```
src/ct/
├── __init__.py
├── config_ct.py          ← CT-specific constants (CT_CLASS_NAMES, paths, etc.)
├── model_ct.py           ← CT model (separate ResNet or different arch)
├── data_utils_ct.py      ← CT-specific transforms (grayscale, CLAHE, etc.)
├── train_utils_ct.py     ← CT training loop (reuses or extends base train_utils)
├── evaluate_ct.py        ← CT evaluation pipeline
├── hospital_client_ct.py ← CT hospital client
└── federated_train_ct.py ← CT local FL simulation

app/static/
├── ct_index.html          ← CT prediction page
└── ct_admin.html          ← CT admin dashboard

scripts/
├── prepare_ct_dataset.py  ← CT dataset preparation
└── split_ct_hospitals.py  ← CT hospital splitting

models/
└── ct_global.pt           ← CT global model

server_storage/
└── ct_fl_updates/
    ├── current/
    └── aggregated/

data/
└── hospitals_ct/
    ├── hospital_1/
    └── hospital_2/
```

#### Files to Modify
- [MODIFY] [api.py](file:///c:/Users/Acer/Desktop/Major%20Project/src/api.py) — Mount CT routes under `/ct/*` prefix or add separate router

#### Shared vs Separate Components

| Component | Decision | Rationale |
|---|---|---|
| `fedavg()` | **Shared** from `train_utils.py` | Algorithm is model-agnostic |
| `train_one_epoch()` | **Shared** (parametric) | Same PyTorch training loop |
| Model architecture | **Separate** `model_ct.py` | CT may need different input channels or architecture |
| Data transforms | **Separate** `data_utils_ct.py` | CT preprocessing differs from X-ray |
| Config constants | **Separate** `config_ct.py` | Different paths, class names |
| API routes | **Separate** router mounted on `/ct/` | Clean URL namespace |
| Frontend pages | **Separate** | Different UI layout |

---

### Implementation Phase 6 — Global UI Overhaul

**Objective:** Modern, premium design across all pages.

#### Changes
- Google Fonts (Inter) across all pages
- Dark mode support
- Glassmorphism panels
- Animated metric counters
- Responsive mobile layout
- Favicon and proper meta tags
- Navigation bar consistent across all pages

---

### Implementation Phase 7 — Bug Fixes

| Bug | Fix |
|---|---|
| FedAvg BN integer tensor | Average BN running stats from all clients, not just first |
| `@app.on_event("startup")` deprecated | Migrate to `lifespan` context manager |
| `safe_name()` duplication | Centralize in `config.py` or shared utility |
| `CLASS_NAMES` in `split_hospitals.py` | Import from config |
| `DEFAULT_NUM_WORKERS=0` | Expose as config parameter; set to CPU count on non-Windows |
| No thread lock on model cache | Add `threading.Lock` around model singleton |

---

### Implementation Phase 8 — Testing

#### Unit Tests (pytest)
- `test_fedavg.py` — FedAvg with known state dicts; BN tensor handling
- `test_model.py` — build_model, save/load checkpoint roundtrip
- `test_data_utils.py` — make_loader with temp dataset
- `test_evaluate.py` — evaluate() with mock model and loader
- `test_metrics_store.py` — save/load metrics JSON
- `test_auth_store.py` — register, login, duplicate prevention

#### Integration Tests
- `test_api.py` — FastAPI TestClient: all endpoints
- `test_fl_workflow.py` — Full round: submit → aggregate → metrics

#### Manual Checklist
- [ ] Upload X-ray → verify NORMAL/PNEUMONIA result
- [ ] Submit weights from 2 hospitals → aggregate → model updates
- [ ] Register hospital → login → submit → duplicate rejected
- [ ] Training progress visible in desktop app
- [ ] Metrics chart shows per-round accuracy/loss
- [ ] Confusion matrix renders correctly
- [ ] Privacy page diagram animates
- [ ] CT pipeline fully independent

---

## Phase 4: Testing Strategy

### Per-Feature Testing

| Feature | Unit | Integration | Manual | Edge Cases |
|---|---|---|---|---|
| Training Metrics | Metrics JSON R/W | API → JSON → Chart render | Verify chart updates after round | Empty rounds, missing test set |
| Hospital Auth | Register/login/token | Full submission with token | Duplicate registration | Empty name, SQL injection chars |
| Duplicate Prevention | Round state logic | Submit twice via API | UI shows "already submitted" | Round cleared after aggregation |
| Progress Display | Callback fires per batch | Training thread → UI update | Progress bar moves smoothly | Single-batch epoch |
| Evaluation Report | sklearn metrics computation | API → JSON → Admin render | Confusion matrix correct | Empty test set, perfect accuracy |
| Privacy Page | — | Nav links | All animations work | Mobile viewport |
| CT Pipeline | Model build, data load | Full CT round simulation | Separate from X-ray completely | Missing CT model file |

---

## Phase 5: Final Deliverables Checklist

### Files to Modify
- [ ] `src/api.py` — Metrics, auth, evaluation endpoints; lifespan migration
- [ ] `src/train_utils.py` — Richer history, progress callbacks, BN fix
- [ ] `src/hospital_client.py` — Auth token support, progress callbacks
- [ ] `src/hospital_desktop.py` — Progress bar, epoch display, auth fields
- [ ] `src/config.py` — Centralize safe_name, num_workers config
- [ ] `app/static/index.html` — Nav links, modern design
- [ ] `app/static/admin.html` — Charts, evaluation tab, modern design

### Files to Add
- [ ] `src/metrics_store.py`
- [ ] `src/auth_store.py`
- [ ] `src/evaluate.py`
- [ ] `src/ct/__init__.py`
- [ ] `src/ct/config_ct.py`
- [ ] `src/ct/model_ct.py`
- [ ] `src/ct/data_utils_ct.py`
- [ ] `src/ct/hospital_client_ct.py`
- [ ] `src/ct/federated_train_ct.py`
- [ ] `src/ct/evaluate_ct.py`
- [ ] `app/static/privacy.html`
- [ ] `app/static/hospital.html`
- [ ] `app/static/ct_admin.html`
- [ ] `app/static/ct_index.html`
- [ ] `scripts/prepare_ct_dataset.py`
- [ ] `scripts/split_ct_hospitals.py`
- [ ] `tests/test_fedavg.py`
- [ ] `tests/test_api.py`
- [ ] `tests/test_metrics_store.py`
- [ ] `tests/test_auth_store.py`

### Features Completed (after implementation)
- [ ] Training metrics persistence + visualization
- [ ] Hospital authentication + registration
- [ ] Duplicate submission prevention
- [ ] Real-time training progress in desktop app
- [ ] Confusion matrix + classification report
- [ ] Initial vs federated model comparison
- [ ] Security & privacy explainer page
- [ ] CT scan separate pipeline
- [ ] Bug fixes (FedAvg BN, startup event, thread lock)
- [ ] UI overhaul (modern design, Google Fonts)

### APIs Added
- `GET /fl/metrics`
- `POST /fl/evaluate`
- `GET /fl/evaluation-report`
- `POST /hospital/register`
- `POST /hospital/login`
- `GET /privacy`
- `GET /ct/*` (all CT pipeline routes)

### Database Changes
- `server_storage/metrics.json` — per-round metrics
- `server_storage/hospitals.json` — hospital registry
- `server_storage/round_state.json` — per-round submission tracking
- `server_storage/evaluation_report.json` — latest evaluation

---

## Open Questions

> [!IMPORTANT]
> **Q1 — Test Dataset on Server**: The evaluation report (Task 3) requires a test dataset accessible from the central server. Currently the server only stores model weights. Where should the test set live? Options:
> - **Option A**: Admin configures `TEST_DIR` in `config.py` pointing to `data/xray_raw/test/` (suitable for demo, not for production)
> - **Option B**: Admin can upload a test set ZIP via a new API endpoint
> 
> Recommend Option A for the academic demo scope.

> [!IMPORTANT]
> **Q2 — CT Dataset Format**: The CT scan pipeline depends on the dataset format. Is the CT dataset available in:
> - JPEG/PNG slices in ImageFolder format (same as X-ray)?
> - DICOM format (requires `pydicom` + preprocessing)?
> 
> This determines if the CT pipeline reuses `data_utils.py` or needs a full custom loader.

> [!IMPORTANT]
> **Q3 — Hospital Auth Security Level**: For this academic project, is a simple UUID token (stored in a JSON file) sufficient for hospital authentication, or is a full JWT/OAuth2 flow required?

> [!WARNING]
> **Breaking Change Risk**: Adding `Authorization` header requirement to `/fl/submit-update` will break the existing `hospital_client.py` CLI and all existing invocations. A grace period (token optional) can be added initially.

---

## Recommended Execution Order

1. ✅ **Phase 1** (Metrics) — Highest impact, foundational data layer
2. ✅ **Phase 2** (Hospital Workflow) — Most user-facing improvement
3. ✅ **Phase 3** (Evaluation Reports) — Depends on metrics infrastructure
4. ✅ **Phase 4** (Privacy Page) — Independent, low risk
5. ✅ **Phase 5** (CT Pipeline) — Most complex, completely independent
6. ✅ **Phase 6** (UI Overhaul) — Can be done alongside any phase
7. ✅ **Phase 7** (Bug Fixes) — Should be done early, but flagged separately
8. ✅ **Phase 8** (Testing) — Throughout, not just at the end
