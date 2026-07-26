# Comprehensive Setup & Operations Guide
## Federated Learning Pneumonia Detection System

This guide provides step-by-step instructions to set up, prepare datasets, partition data across hospital clients, initialize global models, run the central server, and perform federated learning rounds via CLI or Desktop GUI applications.

---

## 1. System Overview & Workflow Architecture

```
                               ┌─────────────────────────┐
                               │   Raw Kaggle Dataset    │
                               │   (chest_xray.zip)      │
                               └────────────┬────────────┘
                                            │
                                 prepare_xray_dataset.py
                                            │
                               ┌────────────▼────────────┐
                               │   data/xray_raw/        │
                               │   ├── train/            │
                               │   └── test/             │
                               └──────┬───────────┬──────┘
                                      │           │
                     split_hospitals.py           train_initial.py
                                      │           │
           ┌──────────────────────────┴┐         ┌▼─────────────────────────┐
           │ data/hospitals_xray/      │         │ models/xray_global.pt    │
           │ ├── hospital_1/           │         │ (Initial Seed Model)     │
           │ ├── hospital_2/           │         └────────────┬─────────────┘
           │ └── hospital_3/           │                      │
           └─────────────┬─────────────┘                      │
                         │                                    │
                         │        ┌───────────────────────────┴──────────┐
                         │        │ Central FastAPI Server               │
                         │        │ (uvicorn src.api:app)                │
                         │        └─────────────▲────────────────────────┘
                         │                      │
           ┌─────────────┴─────────────┐        │ Weight Updates (.pt)
           │ Hospital Clients          │        │ + Bearer Token
           │ (CLI / Desktop Tkinter)   ├────────┘
           └───────────────────────────┘
```

---

## 2. Prerequisites & Environment Setup

### Prerequisites
- **Python 3.9+** installed
- **pip** and **virtualenv**
- Optional: CUDA-capable GPU for faster PyTorch training

### Step 2.1: Clone/Open Project Directory
Navigating to your project directory:
```bash
cd "c:/Users/Acer/Desktop/Major Project"
```

### Step 2.2: Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate on Windows (CMD)
.venv\Scripts\activate.bat

# Activate on Linux/macOS
source .venv/bin/activate
```

### Step 2.3: Install Dependencies
```bash
pip install -r requirements.txt
```

*Key dependencies include:* `torch`, `torchvision`, `fastapi`, `uvicorn`, `requests`, `scikit-learn`, `matplotlib`, `tqdm`.

---

## 3. Data Preparation & Partitioning

### Step 3.1: Prepare & Normalize Raw Dataset
If you have a raw Kaggle dataset (e.g., `chest_xray.zip` or `chest_xray` directory containing `train/` and `test/` folders with `NORMAL/` and `PNEUMONIA/` subdirectories):

Run `prepare_xray_dataset.py` to extract and normalize the folder layout into `data/xray_raw/`:

```bash
python scripts/prepare_xray_dataset.py --zip-path "C:/Users/Acer/Desktop/chest_xray.zip" --output-dir data/xray_raw
```

*Target output structure:*
```
data/xray_raw/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

### Step 3.2: Partition Data Into Hospital Clients
Split the normalized training set across multiple hospital client partitions (e.g., 3 hospitals):

```bash
python scripts/split_hospitals.py --source-dir data/xray_raw/train --output-dir data/hospitals_xray --num-hospitals 3 --seed 42
```

*Output layout created:*
```
data/hospitals_xray/
├── hospital_1/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── hospital_2/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── hospital_3/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## 4. Initializing the Global Model

You can either seed a trained initial global model or start from a pre-trained ResNet18 backbone.

### Step 4.1: Seed Train Global Model
Train an initial model on a small seed dataset or full dataset for a few epochs:

```bash
python src/train_initial.py --train-dir data/xray_raw/train --test-dir data/xray_raw/test --epochs 3 --batch-size 32 --output models/xray_global.pt
```

*(Add `--cpu` flag if running without GPU acceleration)*

---

## 5. Running the Central Federated Server

Start the FastAPI central server using `uvicorn`:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

### Server Web Interfaces & Endpoints
- **End-User Prediction Web UI:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Admin Federated Dashboard:** [http://127.0.0.1:8000/admin](http://127.0.0.1:8000/admin)
- **Privacy & Security Explainer:** [http://127.0.0.1:8000/privacy](http://127.0.0.1:8000/privacy)
- **Hospital Web Portal:** [http://127.0.0.1:8000/hospital](http://127.0.0.1:8000/hospital)
- **API Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 6. Hospital Registration & Token Setup

Each participating hospital requires registration and a unique **Bearer Token** to authenticate weight submissions to the central server.

### Option A: Via Hospital Web Portal
1. Open [http://127.0.0.1:8000/hospital](http://127.0.0.1:8000/hospital).
2. Register a new hospital (e.g. `hospital_1`) with a password.
3. Save the generated **Bearer Token** for client submissions.

### Option B: Via Desktop GUI Client
1. Launch the Desktop GUI:
   ```bash
   python src/hospital_desktop.py
   ```
2. Navigate to **Register/Login**, enter `hospital_1` and your password, click **Register**.
3. Copy the persistent token displayed in the UI.

---

## 7. Executing Federated Learning Rounds

Hospitals can perform local training and submit weight updates using three distinct workflows:

### Workflow A: Using the CLI Hospital Client (`hospital_client.py`)

Run training for individual hospitals from separate command terminals:

**Hospital 1:**
```bash
python src/hospital_client.py --server-url http://127.0.0.1:8000 --hospital-name hospital_1 --dataset-dir data/hospitals_xray/hospital_1 --epochs 1 --token "YOUR_HOSPITAL_1_TOKEN"
```

**Hospital 2:**
```bash
python src/hospital_client.py --server-url http://127.0.0.1:8000 --hospital-name hospital_2 --dataset-dir data/hospitals_xray/hospital_2 --epochs 1 --token "YOUR_HOSPITAL_2_TOKEN"
```

**Hospital 3:**
```bash
python src/hospital_client.py --server-url http://127.0.0.1:8000 --hospital-name hospital_3 --dataset-dir data/hospitals_xray/hospital_3 --epochs 1 --token "YOUR_HOSPITAL_3_TOKEN"
```

*(Add `--cpu` if GPU is unavailable)*

---

### Workflow B: Using the Desktop Tkinter GUI (`hospital_desktop.py`)

1. Start the desktop app:
   ```bash
   python src/hospital_desktop.py
   ```
2. Enter Hospital Name (`hospital_1`), Server URL (`http://127.0.0.1:8000`), and Dataset Path (`data/hospitals_xray/hospital_1`).
3. Click **Authenticate / Login**.
4. Set Epochs (e.g., 1) and click **Download Model & Train Local Update**.
5. Once local training finishes, click **Submit Update to Server**.

---

### Workflow C: Local Offline Simulation (`federated_train.py`)

If you want to run a multi-round federated training simulation locally without starting the web server:

```bash
python src/federated_train.py --hospitals-dir data/hospitals_xray --test-dir data/xray_raw/test --rounds 5 --local-epochs 1 --batch-size 32
```

---

## 8. Server Aggregation & Global Model Evaluation

### Step 8.1: Trigger Aggregation
Once clients have submitted their updates for the round:

1. Open the **Admin Dashboard** at [http://127.0.0.1:8000/admin](http://127.0.0.1:8000/admin).
2. Click **Trigger Aggregation (FedAvg)**.
3. The server computes the sample-weighted average of client updates and saves the updated global checkpoint (`models/xray_global.pt`).

### Step 8.2: View Evaluation Metrics
- The server automatically evaluates the aggregated global model against `data/xray_raw/test`.
- Review loss/accuracy trends, precision, recall, F1 scores, and the confusion matrix in the **Evaluation Report** tab on the Admin Dashboard.

---

## 9. Summary of Key CLI Commands

| Action | Command |
|---|---|
| **Install Dependencies** | `pip install -r requirements.txt` |
| **Prepare Dataset** | `python scripts/prepare_xray_dataset.py --zip-path <path_to_zip> --output-dir data/xray_raw` |
| **Partition Dataset** | `python scripts/split_hospitals.py --source-dir data/xray_raw/train --output-dir data/hospitals_xray --num-hospitals 3` |
| **Seed Train Model** | `python src/train_initial.py --train-dir data/xray_raw/train --test-dir data/xray_raw/test` |
| **Run Server** | `uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload` |
| **Run Desktop GUI** | `python src/hospital_desktop.py` |
| **CLI Hospital Client** | `python src/hospital_client.py --server-url http://127.0.0.1:8000 --hospital-name hospital_1 --dataset-dir data/hospitals_xray/hospital_1 --token <token>` |
| **Offline FL Simulation** | `python src/federated_train.py --hospitals-dir data/hospitals_xray --test-dir data/xray_raw/test --rounds 5` |

---
