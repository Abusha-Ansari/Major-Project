# Federated Learning for Pneumonia Detection

This project starts with the X-ray model for pneumonia detection. Hospitals are simulated as local folders, and each hospital trains on its own private images. Only model weights are aggregated by the central server.

## Why an Initial Model Is Needed

Yes, the system needs an initial global model. In this starter version the initial model is a ResNet18 network with ImageNet-pretrained features and a new binary classification head:

- `NORMAL`
- `PNEUMONIA`

You can optionally train this initial model on a small central seed dataset before federated learning starts. If no seed training is used, the ImageNet-pretrained model is still a valid initial global model.

## Expected Dataset ZIP

Your X-ray ZIP should contain this structure:

```text
data/
  train/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Prepare Dataset

Place your X-ray ZIP file somewhere on your machine, then run:

```bash
python scripts/prepare_xray_dataset.py --zip-path "path\to\xray.zip" --output-dir data\xray_raw
```

This creates:

```text
data/xray_raw/
  train/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

## Create Simulated Hospitals

```bash
python scripts/split_hospitals.py --source-dir data\xray_raw\train --output-dir data\hospitals_xray --num-hospitals 3
```

This creates:

```text
data/hospitals_xray/
  hospital_1/
    NORMAL/
    PNEUMONIA/
  hospital_2/
  hospital_3/
```

## Optional Initial Central Training

```bash
python src/train_initial.py --train-dir data\xray_raw\train --test-dir data\xray_raw\test --epochs 3 --output models\xray_global.pt
```

## Federated Training

```bash
python src/federated_train.py --hospitals-dir data\hospitals_xray --test-dir data\xray_raw\test --rounds 5 --local-epochs 1 --global-model models\xray_global.pt
```

The final global model is saved to:

```text
models/xray_global.pt
```

## Predict One Image

```bash
python src/predict.py --model-path models\xray_global.pt --image-path "path\to\image.jpeg"
```

## Run Web Prediction Interface

After `models/xray_global.pt` exists, install the updated dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn src.api:app --reload
```

Open this URL in your browser:

```text
http://127.0.0.1:8000
```

The web page lets you upload one chest X-ray and returns:

- prediction: `NORMAL` or `PNEUMONIA`
- confidence
- class probabilities

## Run Hospital Desktop Client

Keep the FastAPI server running:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Then open the hospital desktop app in another terminal:

```bash
python src/hospital_desktop.py
```

In the app:

1. Keep server URL as `http://127.0.0.1:8000`.
2. Enter a hospital name, for example `Hospital 1`.
3. Select a hospital dataset folder, for example:

```text
data/hospitals_xray/hospital_1
```

4. Click `Train Locally and Submit Weights`.

The hospital app downloads the global model, trains locally, and sends only the updated weights to the central server.

You can do the same from the command line:

```bash
python src/hospital_client.py --hospital-name "Hospital 1" --dataset-dir data\hospitals_xray\hospital_1 --epochs 1
```

After one or more hospitals submit updates, aggregate them on the server:

```bash
curl -X POST "http://127.0.0.1:8000/fl/aggregate?round_id=current"
```

Check pending submitted updates:

```bash
curl "http://127.0.0.1:8000/fl/status?round_id=current"
```

## Suggested Project Phases

1. X-ray model with local folder hospitals.
2. FastAPI central server.
3. Hospital desktop app.
4. End-user web interface for prediction.
5. CT scan model as a separate pipeline.
