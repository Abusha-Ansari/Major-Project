import argparse
import sys
from pathlib import Path

import requests
import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_BATCH_SIZE, DEFAULT_LR
from data_utils import make_loader
from model import build_model, load_checkpoint
from train_utils import train_model


def log_message(message, logger=None):
    if logger:
        logger(message)
    else:
        print(message)


def download_global_model(server_url, output_path, logger=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{server_url.rstrip('/')}/fl/model"
    log_message(f"Downloading global model from {url}", logger)

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def train_local_update(
    dataset_dir,
    global_model_path,
    output_path,
    epochs=1,
    batch_size=DEFAULT_BATCH_SIZE,
    lr=DEFAULT_LR,
    force_cpu=False,
    logger=None,
):
    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    log_message(f"Using device: {device}", logger)

    train_loader, train_dataset = make_loader(dataset_dir, batch_size=batch_size, train=True)
    log_message(f"Loaded {len(train_dataset)} local samples from {dataset_dir}", logger)

    local_model = build_model(pretrained=False).to(device)
    load_checkpoint(local_model, str(global_model_path), device)
    train_model(local_model, train_loader, device=device, epochs=epochs, lr=lr)

    torch.save(
        {
            "model_state": {key: value.detach().cpu() for key, value in local_model.state_dict().items()},
            "sample_count": len(train_dataset),
            "class_names": train_dataset.classes,
        },
        output_path,
    )
    log_message(f"Saved local update to {output_path}", logger)
    return output_path, len(train_dataset)


def submit_update(server_url, update_path, hospital_name, sample_count, round_id="current", logger=None):
    url = f"{server_url.rstrip('/')}/fl/submit-update"
    log_message(f"Submitting weights to {url}", logger)

    with Path(update_path).open("rb") as update_file:
        response = requests.post(
            url,
            files={"file": ("local_update.pt", update_file, "application/octet-stream")},
            data={
                "hospital_name": hospital_name,
                "sample_count": str(sample_count),
                "round_id": round_id,
            },
            timeout=300,
        )
    response.raise_for_status()
    return response.json()


def train_and_submit(
    server_url,
    hospital_name,
    dataset_dir,
    round_id="current",
    epochs=1,
    batch_size=DEFAULT_BATCH_SIZE,
    lr=DEFAULT_LR,
    work_dir="client_work",
    force_cpu=False,
    logger=None,
):
    hospital_key = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in hospital_name)
    client_dir = Path(work_dir) / (hospital_key or "hospital")
    global_model_path = client_dir / "global_model.pt"
    update_path = client_dir / "local_update.pt"

    download_global_model(server_url, global_model_path, logger=logger)
    update_path, sample_count = train_local_update(
        dataset_dir=dataset_dir,
        global_model_path=global_model_path,
        output_path=update_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        force_cpu=force_cpu,
        logger=logger,
    )
    result = submit_update(
        server_url=server_url,
        update_path=update_path,
        hospital_name=hospital_name,
        sample_count=sample_count,
        round_id=round_id,
        logger=logger,
    )
    log_message(f"Server response: {result}", logger)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train one hospital client and submit weights.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--hospital-name", required=True)
    parser.add_argument("--dataset-dir", required=True, help="Hospital folder with NORMAL/PNEUMONIA folders.")
    parser.add_argument("--round-id", default="current")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    train_and_submit(
        server_url=args.server_url,
        hospital_name=args.hospital_name,
        dataset_dir=args.dataset_dir,
        round_id=args.round_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        force_cpu=args.cpu,
    )


if __name__ == "__main__":
    main()
