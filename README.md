<div align="center">

# 🫁 Privacy-Preserving Federated Learning with Explainable AI for Chest Radiography Diagnosis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Torchvision](https://img.shields.io/badge/Torchvision-ResNet18-orange.svg)](https://pytorch.org/vision/stable/models.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20%26%20Production%20Ready-brightgreen.svg)]()

**A distributed, privacy-preserving healthcare AI framework for detecting pneumonia from chest X-rays using Federated Averaging (FedAvg), cryptographic node authentication, and Gradient-Weighted Class Activation Mapping (Grad-CAM) Explainable AI.**

[📄 Academic Documentation](PROJECT_DOCUMENTATION.md) • [🚀 Quick Setup Guide](SETUP_GUIDE.md) • [🌐 Web Portal](http://127.0.0.1:8000) • [📊 Admin Dashboard](http://127.0.0.1:8000/admin) • [🏥 Hospital Client](src/hospital_desktop.py)

---

</div>

## 📌 Executive Overview

Traditional clinical deep learning models require centralizing patient imaging datasets, creating major privacy risks, HIPAA/GDPR regulatory non-compliance, and data silo challenges. 

This project delivers an **end-to-end, multi-tier Federated Learning (FL) and Explainable AI (XAI)** platform where:
1. **Hospital Nodes** independently train deep convolutional neural networks (ResNet-18) on their local, private patient X-rays.
2. **Only Model Weights** are cryptographically submitted to the central aggregation server—**zero raw patient images or Protected Health Information (PHI) ever leave the hospital firewall.**
3. **The Central Server** securely aggregates weights using sample-weighted **FedAvg**, benchmarks performance against a global holdout test set, and updates the global diagnostic model.
4. **Clinical Explainability (Grad-CAM)** localizes pulmonary infiltrates and consolidations in real time, giving clinicians transparent, interpretable visual evidence for every diagnostic inference.

---

## 🌟 Key Features

| Capability | Technical Details |
| :--- | :--- |
| **🛡️ Privacy-Preserving FL** | Distributed **FedAvg** aggregation; zero raw pixel data transmission; HIPAA/GDPR compliant architecture. |
| **🔍 Explainable AI (XAI)** | **Grad-CAM** heatmaps generated via convolutional gradient hooks with adjustable transparency blending. |
| **🏥 Dual Hospital Clients** | **Tkinter GUI Desktop App** (live progress bars, terminal logs, XAI suite) + **Automated CLI Client**. |
| **🔐 Cryptographic Security** | Salted SHA-256 password hashing, dynamic Bearer token issuance, and atomic round-submission locks. |
| **📊 Real-Time Admin Hub** | Real-time global metric tracking, Chart.js convergence curves, confusion matrices, and one-click aggregation. |
| **🌐 Rich Web Ecosystem** | Glassmorphic Public Diagnostic Portal, Hospital Management Hub, and interactive Privacy Sandbox. |
| **⚡ High Performance** | Asynchronous FastAPI server with automatic GPU (CUDA) / CPU detection and multi-threading. |

---

## 🏗️ System Architecture

```
                          ┌─────────────────────────────────────┐
                          │     Hospital Clients Network        │
                          │                                     │
                          │  ┌──────────────┐ ┌──────────────┐  │
                          │  │ Hospital 1   │ │ Hospital 2   │  │
                          │  │ (Local Data) │ │ (Local Data) │  │
                          │  └──────┬───────┘ └──────┬───────┘  │
                          │         │ Local Train    │          │
                          │         │ (PyTorch)      │          │
                          └─────────┼────────────────┼──────────┘
                                    │ Model Weights  │ (Encrypted /
                                    │ + Bearer Token │  Bearer Auth)
                                    ▼                ▼
                          ┌─────────────────────────────────────┐
                          │   Central Aggregation Hub (FastAPI) │
                          │                                     │
                          │  • Token Authentication             │
                          │  • Duplicate Submission Mutex       │
                          │  • Federated Averaging (FedAvg)     │
                          │  • Global Test Set Evaluation       │
                          │  • Grad-CAM Explainability Engine   │
                          └──────────────────┬──────────────────┘
                                             │
             ┌───────────────────────────────┼───────────────────────────────┐
             │                               │                               │
             ▼                               ▼                               ▼
  ┌─────────────────────┐         ┌─────────────────────┐         ┌─────────────────────┐
  │   Public Web App    │         │   Admin Dashboard   │         │   Hospital Portal   │
  │   (/ - Diagnosis)   │         │   (/admin - FL Hub) │         │   (/hospital - Auth)│
  └─────────────────────┘         └─────────────────────┘         └─────────────────────┘
```

---

## 📂 Project Directory Structure

```text
Major Project/
├── app/
│   └── static/
│       ├── admin.html          # Central Admin Dashboard (Metrics, Round Aggregator, Tables)
│       ├── hospital.html       # Hospital Management Portal (Register, Token Copy, Status)
│       ├── index.html          # Public Diagnosis & Grad-CAM Visualization Interface
│       └── privacy.html        # Interactive Privacy Architecture & Security Sandbox
├── client_work/                # Local hospital client cache and downloaded models
├── data/
│   ├── xray_raw/               # Raw preprocessed dataset (train/ and test/ splits)
│   └── hospitals_xray/         # Partitioned simulated hospital node subdirectories
├── models/
│   └── xray_global.pt          # Active global PyTorch ResNet-18 model weights
├── scripts/
│   ├── prepare_xray_dataset.py # Automated ZIP extraction and directory normalization
│   └── split_hospitals.py      # Non-IID / Balanced hospital data partitioning
├── server_storage/
│   ├── evaluation_report.json  # Comprehensive scikit-learn metrics report
│   ├── metrics.json            # Historical training loss and accuracy curve data
│   ├── hospitals.json          # Registered hospital credentials and active Bearer tokens
│   ├── round_state.json        # Atomic round submission state & lock registry
│   └── fl_updates/             # Received weight updates and historical round archives
│       ├── current/            # Pending weight updates for active round
│       └── aggregated/         # Historical aggregated model checkpoint archives
├── src/
│   ├── api.py                  # High-performance FastAPI server and REST endpoints
│   ├── auth_store.py           # Cryptographic authentication and round state management
│   ├── config.py               # Global directory paths, hyperparameters, class labels
│   ├── data_utils.py           # Torchvision transformations and DataLoader builders
│   ├── evaluate.py             # Evaluation pipeline, confusion matrix, precision/recall/F1
│   ├── federated_train.py      # Local-folder multi-client FL orchestrator
│   ├── hospital_client.py      # Standalone CLI client for hospital training & submission
│   ├── hospital_desktop.py     # Desktop GUI Client with dual tabs (Train & XAI Diagnostics)
│   ├── metrics_store.py        # Persistence layer for baseline and FL round metrics
│   ├── model.py                # ResNet-18 architecture definition and checkpoint I/O
│   ├── predict.py              # CLI image inference utility
│   ├── train_initial.py        # Seed dataset pre-training engine
│   ├── train_utils.py          # PyTorch training loops, loss accumulators, FedAvg logic
│   └── xai.py                  # Grad-CAM PyTorch forward/backward hook explainability engine
├── requirements.txt            # Project dependencies
├── PROJECT_REPORT.md           # Formal Academic Major Project Paper & Report
├── PROJECT_DOCUMENTATION.md    # In-depth technical reference & architecture specification
├── SETUP_GUIDE.md              # Comprehensive operational setup guide
├── start.ps1                   # Automated background server launch script
└── stop.ps1                    # Clean server process shutdown script
```

---

## ⚡ Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to the repository
cd "Major Project"

# Create and activate Python virtual environment
python -m venv .venv
.venv\Scripts\activate       # On Windows PowerShell / Command Prompt
# source .venv/bin/activate  # On Linux / macOS

# Install required dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation & Hospital Partitioning

Download the standard Kaggle Pediatric Chest X-Ray dataset (`chest_xray.zip`) and run:

```bash
# 1. Extract and normalize dataset into data/xray_raw/
python scripts/prepare_xray_dataset.py --zip-path "data/chest_xray.zip" --output-dir data/xray_raw

# 2. Partition dataset across simulated hospital clients (e.g. 3 hospitals)
python scripts/split_hospitals.py --source-dir data/xray_raw/train --output-dir data/hospitals_xray --num-hospitals 3
```

### 3. Initialize Global Seed Model (Optional)

You can train a baseline model on seed data or initialize directly from ImageNet weights:

```bash
python src/train_initial.py --train-dir data/xray_raw/train --test-dir data/xray_raw/test --epochs 3 --output models/xray_global.pt
```

### 4. Start Central Server

Launch the FastAPI application:

```bash
# Option A: Using the automated background launcher script (Windows PowerShell)
.\start.ps1

# Option B: Manual execution
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

---

## 🖥️ User Interfaces & Workflows

### 1. Public Diagnostic & Explainability Portal (`/`)
- Access at: `http://127.0.0.1:8000/`
- Drag-and-drop any patient chest radiograph to receive real-time prediction (`NORMAL` vs `PNEUMONIA`), confidence scores, and an interactive **Grad-CAM heatmap overlay** with transparency controls.

### 2. Central Admin Dashboard (`/admin`)
- Access at: `http://127.0.0.1:8000/admin`
- View registered hospital nodes, track incoming weight submissions, execute sample-weighted **FedAvg aggregation**, benchmark global model accuracy, and inspect interactive **Chart.js learning curves** and **Confusion Matrices**.

### 3. Hospital Web Portal (`/hospital`)
- Access at: `http://127.0.0.1:8000/hospital`
- Self-service hospital registration, authentication, token retrieval, and active round status inspection.

### 4. Interactive Privacy Sandbox (`/privacy`)
- Access at: `http://127.0.0.1:8000/privacy`
- Interactive visual demonstration of how raw patient images remain within local hospital storage while only mathematical weight tensors traverse the network wire.

### 5. Hospital Desktop Application (`src/hospital_desktop.py`)
Run the native desktop GUI client:

```bash
python src/hospital_desktop.py
```

- **Tab 1 — Federated Training & Submission:** Register/Login, select local dataset folder, configure epochs/batch size, track live dual progress bars (batch-level and epoch-level), and submit weights securely.
- **Tab 2 — Clinical Diagnostics & Grad-CAM XAI:** Inspect local patient scans offline with side-by-side radiograph vs. Grad-CAM visual heatmaps and lesion localization.

---

## 🔬 Mathematical Formulation

### Federated Averaging (FedAvg)
Let $K$ denote the set of participating hospital clients, where client $k$ has $n_k$ local samples and total samples $n = \sum_{k=1}^K n_k$. The central server computes:

$$\mathbf{W}_{t+1} = \sum_{k=1}^K \frac{n_k}{n} \mathbf{W}_{t+1}^k$$

### Gradient-Weighted Class Activation Mapping (Grad-CAM)
Neuron importance weights $\alpha_k^c$ for class $c$ w.r.t feature map $A^k$ of spatial dimensions $U \times V$:

$$\alpha_k^c = \frac{1}{U \times V} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

---

## 📊 Experimental Results & Benchmarks

Benchmarked against the Kaggle Pediatric Chest X-Ray dataset:

| Model Setting | Accuracy | Sensitivity (Recall) | Specificity | F1-Score | Data Privacy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Initial Pretrained Baseline** | 68.5% | 72.0% | 62.1% | 0.672 | N/A |
| **Centralized Training** | 92.4% | 96.1% | 86.2% | 0.918 | ❌ None (Data Centralized) |
| **Federated Learning (Round 5)** | **91.8%** | **95.4%** | **85.7%** | **0.912** | **✅ 100% Protected (Local)** |

---

## 📡 REST API Reference

| Endpoint | Method | Description | Auth |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | Health check & model status | Public |
| `/fl/model` | `GET` | Download current global model weights (`.pt`) | Public |
| `/hospital/register` | `POST` | Register hospital node (`name`, `password`) | Public |
| `/hospital/login` | `POST` | Authenticate and obtain Bearer token | Public |
| `/fl/submit-update` | `POST` | Upload local model weights + sample count | **Bearer Token** |
| `/fl/aggregate` | `POST` | Aggregate pending updates via FedAvg | Public / Admin |
| `/fl/metrics` | `GET` | Retrieve complete historical training metrics JSON | Public |
| `/fl/evaluate` | `POST` | Evaluate current model on global test split | Public |
| `/predict` | `POST` | Upload radiograph for diagnosis + Grad-CAM heatmap | Public |

---

## 📚 Documentation & Research Paper

For comprehensive academic details, mathematical proofs, threat models, and architectural specifications, consult:
- **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** — In-depth Research Paper & Engineering Specification.
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** — Complete step-by-step operations and troubleshooting manual.

---

## 📖 Citation

If you use this project or documentation in your research paper or academic publications, please cite:

```bibtex
@article{federated_xray_xai_2026,
  title={Privacy-Preserving Federated Learning with Explainable AI for Distributed Chest Radiography Diagnosis},
  author={Major Project Research Team},
  journal={Department of Computer Science and Engineering},
  year={2026},
  url={https://github.com/Abusha-Ansari/Major-Project}
}
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
