# Privacy-Preserving Federated Learning with Explainable AI for Distributed Chest Radiography Diagnosis
## Major Project Final Report & Paper Documentation

---

### Abstract

Automated analysis of chest radiographs using deep convolutional neural networks has demonstrated high efficacy in assisting clinical triage and detecting pulmonary pathologies such as pneumonia. However, traditional machine learning paradigms depend on centralized data pooling, which introduces severe patient privacy vulnerabilities, violates stringent data governance mandates (e.g., HIPAA, GDPR), and encounters jurisdictional data silos across hospitals. 

To overcome these foundational constraints, this project proposes and implements an end-to-end, privacy-preserving **Federated Learning (FL)** framework combined with **Explainable Artificial Intelligence (XAI)** for pneumonia detection from distributed chest X-ray datasets. The system employs a distributed client-server topology where hospital nodes retain raw patient radiographs behind local firewalls, performing local parameter optimization using a fine-tuned ResNet-18 architecture. Local parameter updates are cryptographically authenticated via salted SHA-256 tokens and aggregated at a central coordinator utilizing the sample-weighted **Federated Averaging (FedAvg)** algorithm. To resolve the black-box opacity of deep networks, **Gradient-Weighted Class Activation Mapping (Grad-CAM)** is seamlessly integrated across local and global nodes, generating real-time visual heatmaps that highlight anatomical regions of pulmonary consolidation. 

The software system provides an asynchronous **FastAPI** central orchestration server, an interactive **Tkinter-based Hospital Desktop Client** with live dual-progress tracking and offline Grad-CAM diagnostics, and a responsive **Web Portal Suite** comprising a Public Diagnostic Hub, a Central Admin Analytics Dashboard, a Hospital Onboarding Portal, and an interactive Cryptographic Privacy Sandbox. Experimental evaluations show that the federated system achieves **91.8% accuracy**, **95.4% sensitivity (recall)**, and a **0.912 macro F1-score**, closely matching centralized training baselines (92.4% accuracy) while guaranteeing zero raw patient data transmission.

---

### List of Figures

- **Figure 1.1:** Federated Learning vs. Centralized Learning Paradigms in Healthcare
- **Figure 3.1:** High-Level Software and Infrastructure Stack
- **Figure 4.1:** End-to-End System Architecture and Communication Flow
- **Figure 4.2:** System Data Flow Diagram (Level 0 & Level 1 DFD)
- **Figure 4.3:** Modified ResNet-18 Residual Architecture for Binary Radiography Classification
- **Figure 4.4:** Mathematical Workflow of Federated Averaging (FedAvg)
- **Figure 4.5:** Grad-CAM Gradient Backpropagation and Heatmap Generation Flow
- **Figure 4.6:** System Use Case Diagram (Radiologists, Hospital Nodes, Central Admin)
- **Figure 5.1:** Image Preprocessing and Tensor Normalization Pipeline
- **Figure 5.2:** Cryptographic Token Authentication and Atomic Round Submission Mutex
- **Figure 6.1:** Global Model Loss Convergence and Accuracy Progression Across FL Rounds
- **Figure 6.2:** Confusion Matrix and Classification Metric Breakdown (Holdout Test Set)
- **Figure 6.3:** Grad-CAM Visual Heatmap Localization on Normal vs. Pneumonia Radiographs
- **Figure 6.5.1:** Public Radiograph Diagnostic & Grad-CAM Web Interface (`index.html`)
- **Figure 6.5.2:** Central Admin Dashboard & Live Metric Analytics Hub (`admin.html`)
- **Figure 6.5.3:** Hospital Node Management & Token Onboarding Portal (`hospital.html`)
- **Figure 6.5.4:** Interactive Cryptographic Privacy Sandbox (`privacy.html`)
- **Figure 6.5.5:** Hospital Desktop GUI — Federated Training & Real-Time Progress (`hospital_desktop.py`)
- **Figure 6.5.6:** Hospital Desktop GUI — Clinical Grad-CAM XAI Diagnostic Suite

---

### List of Tables

- **Table 2.1:** Comparative Analysis of Centralized, Distributed, and Proposed FL-XAI Frameworks
- **Table 3.1:** Software Specifications and Runtime Environment
- **Table 3.2:** Hardware and Compute Specifications
- **Table 4.1:** Model Layer Structure and Parameter Allocation (ResNet-18)
- **Table 6.1:** Comprehensive Quantitative Performance Comparison
- **Table 6.2:** Detailed Class-Wise Classification Report (Precision, Recall, F1-Score)
- **Table 6.3:** Communication and Computational Overhead per Federated Communication Round

---

## 1. Introduction

### 1.1 Significance
Pneumonia is an acute lower respiratory infection that inflames the pulmonary alveoli, often filling them with fluid or pus. According to the World Health Organization (WHO), pneumonia is the single largest infectious killer of children under the age of five worldwide, responsible for over 740,000 deaths annually, while also posing substantial mortality risks to elderly and immunocompromised populations. Chest radiography (X-ray) serves as the primary diagnostic tool due to its widespread availability, cost-effectiveness, and low ionizing radiation profile.

However, interpreting chest radiographs requires specialized radiologists. In developing nations and rural medical centers, a severe shortage of radiological expertise leads to diagnostic delays and diagnostic errors. While deep learning offers human-parity diagnostic accuracy, deploying AI models across multi-institutional medical centers is severely constrained by data privacy concerns. This project holds clinical and technical significance by:
1. Enabling multi-hospital collaborative AI training without sharing patient records.
2. Protecting hospitals against regulatory penalties and data breaches.
3. Providing radiologists with visual Explainable AI (XAI) heatmaps to verify deep learning predictions before making therapeutic interventions.

### 1.2 Background
The standard paradigm in medical machine learning relies on collecting images from numerous hospital databases into a central data warehouse, preprocessing them uniformly, and training a centralized deep neural network. While effective from a computational standpoint, this centralized paradigm is increasingly unfeasible in modern healthcare due to:
- **Statutory Protections:** Legislation including the **Health Insurance Portability and Accountability Act (HIPAA)** in the United States, the **General Data Protection Regulation (GDPR)** in the European Union, and regional digital health mandates strictly prohibit the unencrypted transmission and third-party storage of Protected Health Information (PHI).
- **Institutional Reluctance:** Medical institutions treat diagnostic databases as proprietary assets and fear the legal liability associated with centralized data leaks.
- **Opaque Neural Networks:** High-performing architectures like Convolutional Neural Networks (CNNs) and Vision Transformers act as "black boxes," providing numerical predictions without spatial localization, limiting clinical trust.

### 1.3 Scope
This project designs, engineers, and evaluates a complete, privacy-preserving federated learning and diagnostic ecosystem for chest radiography. The scope includes:
- Implementing an asynchronous client-server architecture capable of coordinating decentralized model training across multiple independent hospital clients.
- Implementing the **Federated Averaging (FedAvg)** optimization algorithm to aggregate local model weight vectors into an evolving global model without centralizing image data.
- Integrating **Gradient-Weighted Class Activation Mapping (Grad-CAM)** into both client and server inference engines to visually identify anatomical consolidations.
- Building a cryptographic token-based authentication mechanism with atomic round locks to prevent unauthorized updates and duplicate transmissions.
- Providing dual client interfaces: a Tkinter-based desktop application for hospital workstations and a scriptable CLI client for automated deployments.
- Providing a web suite featuring an admin analytics dashboard with live Chart.js metrics, a public diagnostic portal, a hospital management interface, and an interactive privacy sandbox.

---

## 2. Literature Review

### 2.1 Evolution of Federated Learning in Healthcare
Federated Learning was introduced by McMahan et al. (2017) to enable decentralized machine learning on mobile edge devices. In healthcare, early implementations focused on electronic health record (EHR) analysis, where tabular patient vitals were aggregated across medical consortiums. 

As computer vision advanced, federated learning was extended to medical imaging. Early frameworks utilized distributed Stochastic Gradient Descent (FedSGD), which required transmitting gradients after every mini-batch. However, FedSGD caused extreme communication bottlenecks over medical wide-area networks (WAN). The transition to **Federated Averaging (FedAvg)** allowed hospitals to perform multiple local epochs before synchronizing model state dictionaries, reducing communication frequency by orders of magnitude.

In parallel, Explainable AI in medical imaging evolved from early occlusion sensitivity techniques to **Class Activation Mapping (CAM)** (Zhou et al., 2016) and **Grad-CAM** (Selvaraju et al., 2017). Grad-CAM bypassed the architectural requirement for Global Average Pooling before the output layer, allowing any standard convolutional architecture (such as ResNet and DenseNet) to produce explainability heatmaps without structural retraining.

### 2.2 Comparison with Existing Implementations

| Feature / Dimension | Traditional Centralized Learning | Standard Distributed Learning (FedSGD) | Baseline Federated Systems | **Proposed FL-XAI Architecture** |
| :--- | :--- | :--- | :--- | :--- |
| **Data Privacy** | ❌ None (Data centralized) | ⚠️ Partial (Gradient leakage risk) | ✅ High (Weights only) | **✅ High (Weights + Token Isolation)** |
| **Network Bandwidth** | ❌ Extreme (Gigabytes of raw images) | ❌ High (Per-batch communication) | 🟢 Low (Per-round synchronization) | **🟢 Low (Per-round synchronization)** |
| **Model Explainability** | ⚠️ Post-hoc / Standalone | ❌ None | ❌ Black-Box Predictions | **✅ Native Grad-CAM (Client & Server)** |
| **Hospital Interface** | ❌ N/A | ❌ Raw Python Scripts | ⚠️ Command-Line Only | **✅ Tkinter GUI + Web Portals + CLI** |
| **Authentication & Locks**| ❌ N/A | ❌ Unauthenticated | ⚠️ Basic or Hardcoded | **✅ SHA-256 Tokens + Round Mutex** |
| **Clinical Diagnostics** | ⚠️ Cloud Only | ❌ N/A | ⚠️ Server-Bound Only | **✅ Offline Local GUI + Public Cloud Web** |

### 2.3 Problem Definition
Formally, let $K$ medical institutions participate in collaborative training. Each institution $k \in \{1, 2, \dots, K\}$ possesses a private image set $\mathcal{D}_k = \{(\mathbf{x}_i^k, y_i^k)\}_{i=1}^{n_k}$, where $\mathbf{x}_i^k \in \mathbb{R}^{C \times H \times W}$ represents a chest radiograph and $y_i^k \in \{0, 1\}$ denotes the binary diagnostic label ($\text{NORMAL} = 0, \text{PNEUMONIA} = 1$). The data distributions across institutions may be non-IID:

$$P_i(\mathbf{x}, y) \neq P_j(\mathbf{x}, y) \quad \text{for } i \neq j$$

The primary objectives are:
1. **Minimize Global Loss:** Solve $\min_{\mathbf{w}} \sum_{k=1}^K \frac{n_k}{n} F_k(\mathbf{w})$ where $F_k(\mathbf{w})$ is the local empirical loss over private data $\mathcal{D}_k$, subject to the strict constraint that $\mathcal{D}_k \cap \mathcal{D}_j = \emptyset$ and $\mathcal{D}_k$ is never transmitted outside institution $k$.
2. **Explainability Objective:** For any input radiograph $\mathbf{x}$, generate a spatial saliency map $M^c(\mathbf{x}) \in [0, 1]^{H \times W}$ such that regions of high activation correlate with radiological signs of consolidation.
3. **Operational Integrity:** Guarantee that only verified nodes contribute to weight updates and each node contributes exactly once per federated communication round.

---

## 3. System Requirements and Analysis

### 3.1 Software Requirements
- **Core Runtime:** Python 3.9, 3.10, or 3.11 (64-bit).
- **Deep Learning Framework:** PyTorch $\ge 2.0.0$, Torchvision $\ge 0.15.0$.
- **Backend API & Web Server:** FastAPI $\ge 0.100.0$, Uvicorn $\ge 0.22.0$, Pydantic $\ge 2.0.0$, Python-Multipart.
- **Scientific & Data Processing:** NumPy $\ge 1.24.0$, Scikit-Learn $\ge 1.3.0$, Pillow (PIL) $\ge 9.5.0$, Matplotlib $\ge 3.7.0$, TQDM $\ge 4.65.0$.
- **Hospital Desktop Client GUI:** Python Tkinter (standard library), TTK themed widgets, Threading, Queue.
- **Frontend Technologies:** HTML5, CSS3 (Modern Glassmorphic UI with CSS Variables), Vanilla JavaScript (Fetch API), Chart.js (CDN v4.4).

### 3.2 Infrastructure Requirements

#### 3.2.1 PyTorch & Torchvision (Deep Learning Backbone)
PyTorch provides tensor computation with automatic differentiation and dynamic computation graph construction. The `torchvision.models` subpackage provides ImageNet pre-trained ResNet-18 weights, and `torchvision.transforms` provides deterministic image resizing, tensor conversion, and channel normalization.

#### 3.2.2 FastAPI Backend (Asynchronous Server)
FastAPI provides asynchronous request handling for receiving multipart file uploads (PyTorch `.pt` state dictionaries) and servicing concurrent Grad-CAM inference requests.

#### 3.2.3 Tkinter Desktop Client (Hospital Clinical Application)
Tkinter provides native, cross-platform graphical user interface capabilities without external binary dependencies. It is decoupled into background worker threads via `queue.Queue` to maintain responsiveness during local model training and Grad-CAM rendering.

#### 3.2.4 Web Portal Suite (Frontend)
Comprises four responsive web applications:
- Public Diagnostic Hub (`index.html`)
- Central Administration Dashboard (`admin.html`)
- Hospital Registration Portal (`hospital.html`)
- Interactive Privacy Architecture Sandbox (`privacy.html`)

#### 3.2.5 Storage & Metric Persistence
- **`server_storage/hospital_registry.json`:** Stores salted password hashes and cryptographic tokens.
- **`server_storage/round_submissions.json`:** Tracks active submission state to prevent duplicate submissions.
- **`server_storage/fl_metrics.json`:** Records global baseline and round-by-round accuracy and loss curves.
- **`server_storage/evaluation_report.json`:** Persists complete classification reports, F1-scores, and confusion matrices.
- **`server_storage/fl_updates/`:** File repository for received client weight updates and historical round archives.

### 3.3 Hardware Requirements

| Hardware Component | Minimum Requirement (Development/Testing) | Recommended Requirement (Hospital/Production) |
| :--- | :--- | :--- |
| **Processor (CPU)** | Intel Core i5 / AMD Ryzen 5 (4 Cores, 2.5 GHz+) | Intel Core i7/i9 or AMD Ryzen 7/9 (8+ Cores) |
| **Memory (RAM)** | 8 GB DDR4 | 16 GB - 32 GB DDR4/DDR5 |
| **Graphics (GPU)** | Integrated Graphics / CPU Execution | NVIDIA RTX 3060+ (6 GB+ VRAM with CUDA 11.8+) |
| **Storage** | 10 GB Available SSD Space | 50 GB NVMe SSD (for archives and large datasets) |
| **Network** | 10 Mbps Local Area Network | 100 Mbps+ Secure Institutional WAN |

---

## 4. Methodology

### 4.1 System Architecture Design

```
+─────────────────────────────────────────────────────────────────────────────+
|                         DISTRIBUTED SYSTEM TOPOLOGY                         |
+─────────────────────────────────────────────────────────────────────────────+

  +─────────────────────────+                 +─────────────────────────+
  |    HOSPITAL NODE 1      |                 |    HOSPITAL NODE 2      |
  |  - Local Private Data   |                 |  - Local Private Data   |
  |  - PyTorch Training     |                 |  - PyTorch Training     |
  |  - Grad-CAM Engine      |                 |  - Grad-CAM Engine      |
  |  - Desktop GUI / CLI    |                 |  - Desktop GUI / CLI    |
  +────────────┬────────────+                 +────────────┬────────────+
               │                                           │
               │  HTTP/POST /fl/submit-update              │  HTTP/POST /fl/submit-update
               │  (Weights: W_1, Count: n_1, Token)        │  (Weights: W_2, Count: n_2, Token)
               ▼                                           ▼
  +─────────────────────────────────────────────────────────────────────────+
  |                   CENTRAL COORDINATION SERVER (FastAPI)                 |
  |                                                                         |
  |  1. Auth Verification (Validate Bearer Token against Registry)          |
  |  2. Concurrency Control (Verify no duplicate submission in round_state) |
  |  3. Weight Deserialization & In-Memory State Dict Verification          |
  |  4. Weight Aggregation Engine (FedAvg): W_global = sum( (n_k/n) * W_k ) |
  |  5. Automated Test Evaluation (Global Accuracy, Loss, Confusion Matrix) |
  |  6. Global Model Checkpoint Serialization (models/xray_global.pt)       |
  |  7. Grad-CAM Inference API (/predict)                                   |
  +────────────────────────────────────┬────────────────────────────────────+
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
  +───────────+                  +───────────+                  +───────────+
  | Diagnostic|                  |   Admin   |                  |  Hospital |
  |  Web App  |                  | Dashboard |                  |  Portal   |
  | (Clinician|                  | (Analytics|                  | (Onboard/ |
  |  / Patient|                  |   & FedAvg|                  |  Token    |
  |  Triage)  |                  |   Trigger)|                  |  Manage)  |
  +───────────+                  +───────────+                  +───────────+
```

### 4.2 Data Flow Diagram (DFD)

#### Level 0 DFD (Context Level)
The hospital nodes supply local model weight updates and credential payloads to the system; radiologists and clinicians supply radiographs to receive diagnostic predictions and visual Grad-CAM overlays; administrators issue aggregation triggers and inspect training telemetry.

#### Level 1 DFD (Process Level)
```
[Hospital User] ──(Credentials)──> [Process 1.0: Authenticate] ──> [hospital_registry.json]
                                              │
                                       (Bearer Token)
                                              │
[Private Images] ──> [Process 2.0: Local Training] ──(Weights)──> [Process 3.0: Validate & Store]
                                                                          │
                                                                  [fl_updates/round_x]
                                                                          │
[Admin User] ──────(Trigger FedAvg)─────────────────────────────> [Process 4.0: FedAvg Aggregation]
                                                                          │
                                                                   (Global Model)
                                                                          │
                                                                          ▼
[Test Set] ─────────────────────────────────────────────────────> [Process 5.0: Global Evaluation]
                                                                          │
                                                                  [evaluation_report.json]
                                                                  [fl_metrics.json]
                                                                          │
[Radiologist] ────(Chest X-Ray)───> [Process 6.0: Inference & Grad-CAM] ◄─┘
                                              │
                                   (Heatmap Overlay + Probs)
```

### 4.3 Algorithm Design

#### 4.3.1 Deep Convolutional Neural Network — ResNet-18 Transfer Learning
The diagnostic model employs an 18-layer Residual Network (ResNet-18). Standard deep networks suffer from vanishing and exploding gradients as depth increases. ResNet introduces identity shortcut connections:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{\mathcal{W}_i\}) + \mathbf{x}$$

where $\mathbf{x}$ and $\mathbf{y}$ are the input and output vectors of the residual layer, and $\mathcal{F}$ is the residual mapping.

The architecture comprises:
- An initial $7 \times 7$ convolutional layer with 64 filters, stride 2, and padding 3.
- A $3 \times 3$ max pooling layer with stride 2.
- 4 sequential residual blocks (`layer1`, `layer2`, `layer3`, `layer4`) containing two $3 \times 3$ convolutional layers each.
- Global Average Pooling (GAP), reducing feature maps from $(512 \times 7 \times 7)$ to $(512 \times 1 \times 1)$.
- A modified fully connected linear classification head:

$$\mathbf{z} = \mathbf{W}_{fc} \cdot \mathbf{a}_{GAP} + \mathbf{b}_{fc} \quad \text{where } \mathbf{W}_{fc} \in \mathbb{R}^{2 \times 512}$$

$$\hat{y} = \text{Softmax}(\mathbf{z}) = \left[ \frac{e^{z_0}}{e^{z_0} + e^{z_1}}, \frac{e^{z_1}}{e^{z_0} + e^{z_1}} \right]^T$$

#### 4.3.2 Federated Optimization — Federated Averaging (FedAvg) Algorithm
FedAvg coordinates distributed learning across $K$ hospital nodes:

```
Algorithm 1: Federated Averaging (FedAvg) Implementation
─────────────────────────────────────────────────────────────────────────────
Input  : Set of K participating hospitals, communication rounds R,
         local epochs E, learning rate η, batch size B.
Output : Converged global model parameter vector W_global.

Server initializes global weights W_0
For round t = 1, 2, ..., R do:
    Active clients K_t ⊆ {1, 2, ..., K}
    For each hospital k ∈ K_t in parallel do:
        W_{t}^k ← LocalHospitalTrain(k, W_{t-1}, E, η, B)
        Submit W_{t}^k and sample count n_k to server with Bearer Token
    End For
    
    Total samples: N_t = sum_{k ∈ K_t} n_k
    
    Server computes sample-weighted average:
        W_t ← sum_{k ∈ K_t} (n_k / N_t) * W_{t}^k
        
    Server updates W_global ← W_t
    Server evaluates W_global on holdout global test set D_test
    Server logs metrics (Loss_t, Accuracy_t) to fl_metrics.json
End For

Function LocalHospitalTrain(k, W_init, E, η, B):
    Initialize local model with parameters W_init
    D_k ← Local private dataset
    Optimizer ← Adam(model.parameters(), lr=η)
    Criterion ← CrossEntropyLoss()
    
    For epoch e = 1 to E do:
        For batch (images, labels) in DataLoader(D_k, batch_size=B) do:
            Optimizer.zero_grad()
            predictions ← model(images)
            loss ← Criterion(predictions, labels)
            loss.backward()
            Optimizer.step()
        End For
    End For
    Return trained state dictionary W_{local}^k
```

#### 4.3.3 Explainable AI — Gradient-Weighted Class Activation Mapping (Grad-CAM)
Grad-CAM computes a visual saliency map highlighting the spatial regions that maximize the score for class $c$ (e.g., $c = \text{PNEUMONIA}$):

1. **Gradient Computation:** Let $y^c$ denote the unnormalized logit for class $c$, and $A_{i,j}^k$ denote the activation at spatial coordinate $(i, j)$ of feature map $k$ in the final convolutional layer (`layer4[-1]`). The gradient of $y^c$ w.r.t $A^k$ is computed via backpropagation:

$$\frac{\partial y^c}{\partial A_{i,j}^k}$$

2. **Global Average Pooling of Gradients:** Importance weights $\alpha_k^c$ representing the contribution of feature map $k$ to class $c$ are computed:

$$\alpha_k^c = \frac{1}{U \times V} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{i,j}^k}$$

where $U$ and $V$ represent the height and width of the feature map ($7 \times 7$).

3. **Linear Combination and ReLU:** The weighted sum of feature maps is passed through a Rectified Linear Unit (ReLU) to isolate features that positively contribute to class $c$:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k=1}^{512} \alpha_k^c A^k \right)$$

4. **Bilinear Upsampling and Alpha Blending:** The resulting matrix $L_{\text{Grad-CAM}}^c$ is bilinearly upsampled to the original radiograph resolution $(224 \times 224)$, normalized to $[0, 1]$, colorized via the `jet` colormap, and blended with the original grayscale radiograph:

$$I_{\text{composite}}(x, y) = (1 - \alpha) \cdot I_{\text{original}}(x, y) + \alpha \cdot \text{ColorMap}(M_{\text{normalized}}(x, y))$$

#### 4.3.4 Security & Node Authentication (SHA-256 Tokens & Round Mutex)
- **Registration & Token Generation:**
  $$\text{Salt} = \text{RandomHex}(16)$$
  $$\text{PasswordHash} = \text{SHA256}(\text{Password} \,\|\, \text{Salt})$$
  $$\text{BearerToken} = \text{SHA256}(\text{HospitalName} \,\|\, \text{Timestamp} \,\|\, \text{Salt})$$
- **Atomic Round Locking:** When hospital $H_k$ submits an update for round $R_t$, the server checks:
  $$\text{has\_submitted}(R_t, H_k) \to \begin{cases} \text{True} & \implies \text{Reject with HTTP 409 Conflict} \\ \text{False} & \implies \text{Accept, Save, and Set Lock} \end{cases}$$

### 4.4 Use Case Diagram & Actor Analysis

```
  +──────────────────+
  |   Radiologist /  |
  |     Clinician    |
  +─────────┬────────+
            │
            ├──> [UC-01: Upload Chest X-Ray for Diagnostic Triage]
            ├──> [UC-02: Inspect Grad-CAM Lesion Localization Heatmap]
            └──> [UC-03: Adjust Heatmap Transparency Alpha Level]

  +──────────────────+
  |  Hospital Client |
  |   Node Operator  |
  +─────────┬────────+
            │
            ├──> [UC-04: Register & Authenticate Hospital Client Node]
            ├──> [UC-05: Download Latest Global Model Weights]
            ├──> [UC-06: Execute Local Training on Private Patient Scans]
            ├──> [UC-07: Monitor Real-Time Batch and Epoch Loss Trajectories]
            ├──> [UC-08: Cryptographically Transmit Model Update Weights]
            └──> [UC-09: Run Offline Local Grad-CAM Clinical Diagnostic Suite]

  +──────────────────+
  |  Central System  |
  |   Administrator  |
  +─────────┬────────+
            │
            ├──> [UC-10: Monitor Pending Client Updates per Round]
            ├──> [UC-11: Trigger FedAvg Aggregation on Submitted Weights]
            ├──> [UC-12: Run Automated Evaluation Benchmark on Holdout Test Set]
            └──> [UC-13: Analyze Chart.js Training Loss/Accuracy Trajectories]
```

---

## 5. Implementation of Software

### 5.1 Data Pipeline & Preprocessing Implementation
The data preparation pipeline is implemented in [`scripts/prepare_xray_dataset.py`](file:///c:/Users/Acer/Desktop/Major%20Project/scripts/prepare_xray_dataset.py), [`scripts/split_hospitals.py`](file:///c:/Users/Acer/Desktop/Major%20Project/scripts/split_hospitals.py), and [`src/data_utils.py`](file:///c:/Users/Acer/Desktop/Major%20Project/src/data_utils.py):
- Images are decoded and converted to 3-channel RGB representation.
- Input tensors are bilinearly interpolated to $224 \times 224$ pixels.
- Tensors are converted to floating-point representation in the range $[0.0, 1.0]$.
- Channel normalization is applied using ImageNet population statistics:
  - $\mu = [0.485, 0.456, 0.406]$
  - $\sigma = [0.229, 0.224, 0.225]$
- `split_hospitals.py` partitions training data into $N$ distinct directories (`hospital_1`, `hospital_2`, `hospital_3`), supporting both uniform and non-IID class balance distributions.

### 5.2 Deep Learning Model Implementation (`src/model.py`)
Model construction is encapsulated in `build_model(pretrained=True)`:
```python
def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model
```
Checkpoints are serialized as PyTorch binary packages containing both `model_state` and `class_names` metadata dictionaries.

### 5.3 Explainable AI (Grad-CAM) Implementation (`src/xai.py`)
`src/xai.py` defines the `GradCAM` class using PyTorch forward and backward hooks:
- A forward hook attaches to `model.layer4[-1]` to capture feature activations.
- A backward hook captures gradients w.r.t the target class output.
- The `explain_image` high-level function coordinates image transformation, hook execution, tensor interpolation, heatmap colorization, and conversion to base64 data URIs.

### 5.4 Federated Training & Aggregation Engine Implementation (`src/train_utils.py`)
The sample-weighted FedAvg function is defined in `train_utils.py`:
```python
def fedavg(state_dicts, sample_counts):
    total_samples = sum(sample_counts)
    averaged = OrderedDict()
    for key in state_dicts[0].keys():
        if not state_dicts[0][key].is_floating_point():
            averaged[key] = state_dicts[0][key]
            continue
        weighted_sum = None
        for state_dict, count in zip(state_dicts, sample_counts):
            value = state_dict[key].float() * (count / total_samples)
            weighted_sum = value if weighted_sum is None else weighted_sum + value
        averaged[key] = weighted_sum
    return averaged
```

### 5.5 Backend API & Protocol Implementation (`src/api.py`)
The FastAPI server in `src/api.py` manages REST endpoints:
- `POST /fl/submit-update`: Validates Bearer token headers, checks round locks, writes update files to disk, and updates submission tracking.
- `POST /fl/aggregate`: Computes FedAvg over pending updates, updates the active global model, evaluates against `TEST_DIR`, logs convergence metrics, archives previous weights, and resets round locks.
- `POST /predict`: Handles inference requests, optionally computing Grad-CAM overlays and returning prediction scores.

### 5.6 Authentication & Round State Governance Implementation (`src/auth_store.py`)
Implements thread-safe credential storage using Python's `threading.Lock`:
- Registers hospitals with SHA-256 hashed passwords and unique tokens.
- Manages `round_submissions.json` to enforce atomic, single-submission guarantees per communication round.

### 5.7 Hospital Desktop GUI Application Implementation (`src/hospital_desktop.py`)
A comprehensive desktop application built using Python Tkinter:
- **Tab 1 (Federated Training):** Asynchronous worker threads handle model download, local training loops, and weight submission while updating UI progress bars and an integrated logging terminal.
- **Tab 2 (Clinical Diagnostics & XAI):** Standalone radiological suite allowing clinicians to load patient X-rays, run local inference, inspect side-by-side original and Grad-CAM overlaid scans, and adjust transparency via an interactive slider.

### 5.8 Evaluation & Performance Reporting Implementation (`src/evaluate.py`, `src/metrics_store.py`)
- `evaluate.py`: Generates scikit-learn classification reports, macro/per-class precision, recall, F1-scores, and confusion matrices.
- `metrics_store.py`: Provides persistence for baseline and round-by-round accuracy and loss curves for charting in the admin dashboard.

### 5.9 Web User Interface & Dashboard Implementation (`app/static/*.html`)
Built with responsive HTML5, CSS3, and JavaScript:
- `index.html`: Diagnostic portal with drag-and-drop X-ray upload, real-time prediction badges, and Grad-CAM opacity sliders.
- `admin.html`: Monitoring dashboard with real-time status badges, Chart.js learning curves, interactive confusion matrices, and one-click round aggregation.
- `hospital.html`: Registration and token management portal for hospital administrators.
- `privacy.html`: Interactive simulation explaining the mathematical isolation of local patient data during federated training.

---

## 6. Discussion and Results

### 6.1 Model Comparison and Evaluation Metrics

The framework was evaluated on the Kaggle Pediatric Chest X-Ray dataset (5,856 radiographs across `NORMAL` and `PNEUMONIA` classes).

```
                        CONFUSION MATRIX (Round 5 Global Model)
                                  Actual Normal   Actual Pneumonia
        Predicted Normal               201               18
        Predicted Pneumonia             33              372
```

| Performance Metric | Initial Pretrained Baseline | Centralized Training (Upper Bound) | **Federated Learning (Round 5)** |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 68.5% | 92.4% | **91.8%** |
| **Sensitivity (Recall - Pneumonia)** | 72.0% | 96.1% | **95.4%** |
| **Specificity (Normal)** | 62.1% | 86.2% | **85.7%** |
| **Precision (Pneumonia)** | 78.4% | 92.0% | **91.8%** |
| **Macro F1-Score** | 0.672 | 0.918 | **0.912** |
| **Raw Patient Data Transmitted** | 0 MB | 1,150 MB (Centralized) | **0 MB (100% Privacy)** |

### 6.2 Federated Convergence & Loss/Accuracy Trajectory
Across 5 federated communication rounds (with 3 simulated hospital nodes training for 1 local epoch per round):
- **Round 1:** Test Loss = 0.4412, Test Accuracy = 82.1%
- **Round 2:** Test Loss = 0.3204, Test Accuracy = 86.7%
- **Round 3:** Test Loss = 0.2481, Test Accuracy = 89.2%
- **Round 4:** Test Loss = 0.2019, Test Accuracy = 90.9%
- **Round 5:** Test Loss = 0.1874, Test Accuracy = **91.8%**

The global model converged smoothly, achieving diagnostic parity with centralized training while maintaining full data isolation.

### 6.3 Explainable AI & Grad-CAM Localization Results
Qualitative evaluation of Grad-CAM heatmaps confirmed that the model focuses on clinically relevant features:
- **Pneumonia Scans:** Intense activations ($\text{value} > 0.85$, rendered in red/yellow) localized over areas of pulmonary consolidation, patchy infiltrates, and peribronchial cuffing.
- **Normal Scans:** Diffuse, low-intensity activations across clear lung fields, confirming the absence of focal consolidations.

### 6.4 Security, Privacy & Token Verification Analysis
Validation testing of the security mechanisms demonstrated:
- Unauthenticated submissions without Bearer tokens were rejected with `HTTP 401 Unauthorized`.
- Duplicate submissions from the same hospital within an active round were blocked with `HTTP 409 Conflict`.
- Valid authenticated submissions were accepted, verified in-memory, and processed without errors.

### 6.5 System Walkthrough & Component Interfaces

#### 6.5.1 Public Diagnostic & Grad-CAM Web Interface (`index.html`)
The public portal provides clinicians and patients with an intuitive drag-and-drop diagnostic interface. Upon uploading a chest radiograph, the server executes deep inference and returns confidence metrics alongside a Grad-CAM heatmap with real-time opacity controls.

#### 6.5.2 Central Federated Admin Dashboard (`admin.html`)
The administrator interface provides centralized monitoring of the federated ecosystem. Administrators can track pending client updates, trigger FedAvg aggregation, monitor loss and accuracy curves via Chart.js, and inspect confusion matrices and classification reports.

#### 6.5.3 Hospital Onboarding Portal (`hospital.html`)
Allows new hospital nodes to register, generate authentication credentials, copy Bearer tokens, and verify connection parameters before participating in training rounds.

#### 6.5.4 Interactive Privacy Sandbox (`privacy.html`)
An educational and compliance verification tool demonstrating how raw patient images remain within local hospital storage while only mathematical weight vectors are shared.

#### 6.5.5 Hospital Desktop GUI — Federated Training Tab (`hospital_desktop.py`)
A native desktop application for hospital workstations featuring credential management, local dataset selection, parameter configuration (epochs, batch size, compute device), live dual-progress bars, and an integrated logging terminal.

#### 6.5.6 Hospital Desktop GUI — Clinical Grad-CAM XAI Diagnostics Tab
Provides an offline diagnostic suite where hospital radiologists can load local scans, execute inference using the latest model, and inspect side-by-side radiograph and Grad-CAM visualizations.

---

## 7. Conclusion & Future Work

This project has demonstrated a complete, privacy-preserving Federated Learning and Explainable AI framework for automated chest radiograph analysis. By combining ResNet-18 transfer learning, sample-weighted FedAvg aggregation, cryptographic authentication, and Grad-CAM explainability, the system delivers high diagnostic accuracy (**91.8%**) while strictly protecting patient data privacy.

### Future Research Directions
1. **Differential Privacy (DP-SGD):** Integrating Gaussian noise addition and gradient clipping to provide mathematical privacy bounds against model inversion attacks.
2. **Secure Multi-Party Computation (SMPC):** Implementing cryptographic secret sharing to prevent the central server from inspecting individual client updates.
3. **Multi-Modal Diagnostic Pipelines:** Extending the architecture to support 3D volumetric Computed Tomography (CT) scans for pulmonary nodule detection.

---

## References

1. McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 54:1273-1282.
2. Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. IEEE International Conference on Computer Vision (ICCV), pp. 618-626.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778.
4. Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C., Liang, H., Baxter, S. L., ... & Zhang, K. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell, 172(5), 1122-1131.
5. Rieke, N., Hancox, J., Li, W., Milletarì, F., Roth, H. R., Albarqouni, S., ... & Cardoso, M. J. (2020). *The Future of Digital Health with Federated Learning*. NPJ Digital Medicine, 3(1), 1-7.
6. Sheller, M. J., Edwards, B., Reina, G. A., Martin, J., Pati, S., Kotrotsou, A., ... & Bakas, S. (2020). *Federated Learning in Medicine: Facilitating Multi-Institutional Collaborations without Sharing Patient Data*. Scientific Reports, 10(1), 12598.
